"""
Supervised Multi AI Agent Architecture for Stock Trading
This implements a supervisor-based multi-agent system with specialized trading agents:
- Fundamental Analyst: Analyzes company fundamentals
- Technical Analyst: Analyzes technical indicators
- News Reader: Performs sentiment analysis on news
- Trade Placer: Executes trades using Alpaca API

Uses Alpaca API for all data retrieval (no yfinance rate limiting)
"""

import os
from typing import TypedDict, Annotated, List, Literal, Dict, Any
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END, MessagesState
from langchain_core.prompts import ChatPromptTemplate
from datetime import datetime, timedelta
from dotenv import load_dotenv
import pandas as pd
import numpy as np
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, StockSnapshotRequest
from alpaca.data.timeframe import TimeFrame
import json
import time
import streamlit as st

# Load environment variables
load_dotenv()

def get_env_variable(key: str) -> str:
    """Get environment variable from .env or Streamlit secrets"""
    # First try environment variables
    value = os.getenv(key)
    if value:
        return value
    # Then try Streamlit secrets (for cloud deployment)
    try:
        return st.secrets.get(key, "")
    except:
        return ""

# Set API keys
OPENAI_API_KEY = get_env_variable("OPENAI_API_KEY")
ALPACA_API_KEY = get_env_variable("ALPACA_API_KEY")
ALPACA_SECRET_KEY = get_env_variable("ALPACA_SECRET_KEY")

if OPENAI_API_KEY:
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Initialize LLM
from langchain.chat_models import init_chat_model

def get_llm():
    """Initialize LLM with error handling"""
    try:
        return init_chat_model("openai:gpt-4o")
    except Exception as e:
        print(f"⚠️ Warning: LLM initialization failed: {e}")
        return None

llm = get_llm()

# Initialize Alpaca clients
def init_alpaca_clients():
    """Initialize Alpaca clients with error handling"""
    trading_client = None
    data_client = None
    
    if ALPACA_API_KEY and ALPACA_SECRET_KEY:
        try:
            trading_client = TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=True)
            data_client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)
            print("✅ Alpaca clients initialized successfully")
        except Exception as e:
            print(f"⚠️ Warning: Alpaca client initialization failed: {e}")
    else:
        print("⚠️ Warning: Alpaca API keys not found")
    
    return trading_client, data_client

trading_client, data_client = init_alpaca_clients()


# ===================================
# State Definition
# ===================================

class TradingState(MessagesState):
    """State for the stock trading multi-agent system"""
    next_agent: str = ""
    ticker: str = ""
    fundamental_analysis: str = ""
    technical_analysis: str = ""
    news_sentiment: str = ""
    trade_recommendation: str = ""
    trade_executed: bool = False
    task_complete: bool = False
    current_task: str = ""
    direct_trade: bool = False  # Flag for direct trade without analysis
    trade_action: str = ""  # BUY or SELL
    trade_quantity: int = 0  # Number of shares


# ===================================
# Supervisor Agent
# ===================================

def create_supervisor_chain():
    """Creates the supervisor decision chain for trading"""
    
    supervisor_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a supervisor managing a team of trading agents:
        
1. Fundamental Analyst - Analyzes company fundamentals and financial health
2. Technical Analyst - Analyzes technical indicators and price patterns
3. News Reader - Analyzes news sentiment and market perception
4. Trade Placer - Makes final trade decision and executes orders

Based on the current state, decide which agent should work next.
If all analysis is complete and trade is executed, respond with 'DONE'.

Current state:
- Has fundamental analysis: {has_fundamental}
- Has technical analysis: {has_technical}
- Has news sentiment: {has_news}
- Trade executed: {trade_executed}

IMPORTANT: If an agent encountered an error, still proceed to the next agent.
Respond with ONLY the agent name (fundamental_analyst/technical_analyst/news_reader/trade_placer) or 'DONE'.
"""),
        ("human", "{task}")
    ])
    
    return supervisor_prompt | llm


def supervisor_agent(state: TradingState) -> Dict:
    """Supervisor decides next agent using LLM"""
    
    messages = state["messages"]
    task = messages[-1].content if messages else "No task"
    
    # Check if this is a direct trade request
    if state.get("direct_trade", False):
        return {
            "messages": [AIMessage(content="⚡ Supervisor: Direct trade requested. Bypassing analysis and executing immediately...")],
            "next_agent": "direct_trade_executor",
            "current_task": task
        }
    
    # Check what's been completed (even if errors occurred)
    has_fundamental = bool(state.get("fundamental_analysis", ""))
    has_technical = bool(state.get("technical_analysis", ""))
    has_news = bool(state.get("news_sentiment", ""))
    trade_executed = state.get("trade_executed", False)
    
    # Get LLM decision
    chain = create_supervisor_chain()
    decision = chain.invoke({
        "task": task,
        "has_fundamental": has_fundamental,
        "has_technical": has_technical,
        "has_news": has_news,
        "trade_executed": trade_executed
    })
    
    # Parse decision
    decision_text = decision.content.strip().lower()
    print(f"\n📋 Supervisor decision: {decision_text}")
    
    # Determine next agent (with strict progression to avoid loops)
    if "done" in decision_text or trade_executed:
        next_agent = "end"
        supervisor_msg = "✅ Supervisor: All analysis complete and trade decision made! Great work team."
    elif not has_fundamental:
        next_agent = "fundamental_analyst"
        supervisor_msg = "📋 Supervisor: Starting with fundamental analysis. Assigning to Fundamental Analyst..."
    elif not has_technical:
        next_agent = "technical_analyst"
        supervisor_msg = "📋 Supervisor: Fundamental analysis done. Moving to technical analysis. Assigning to Technical Analyst..."
    elif not has_news:
        next_agent = "news_reader"
        supervisor_msg = "📋 Supervisor: Technical analysis done. Checking market sentiment. Assigning to News Reader..."
    elif not trade_executed:
        next_agent = "trade_placer"
        supervisor_msg = "📋 Supervisor: All analysis complete. Making trade decision. Assigning to Trade Placer..."
    else:
        next_agent = "end"
        supervisor_msg = "✅ Supervisor: Trading workflow complete."
    
    return {
        "messages": [AIMessage(content=supervisor_msg)],
        "next_agent": next_agent,
        "current_task": task
    }


# ===================================
# Agent 1: Fundamental Analyst (Using Alpaca)
# ===================================

def fundamental_analyst(state: TradingState) -> Dict:
    """Analyzes company fundamentals using Alpaca data"""
    
    ticker = state.get("ticker", "")
    print(f"\n📊 Fundamental Analyst working on {ticker}...")
    
    try:
        # Get snapshot data from Alpaca
        request = StockSnapshotRequest(symbol_or_symbols=ticker)
        snapshot = data_client.get_stock_snapshot(request)
        
        if ticker not in snapshot:
            raise ValueError(f"No snapshot data available for {ticker}")
        
        snap = snapshot[ticker]
        
        # Get historical data for additional metrics
        end = datetime.now()
        start = end - timedelta(days=90)
        
        bars_request = StockBarsRequest(
            symbol_or_symbols=ticker,
            timeframe=TimeFrame.Day,
            start=start,
            end=end
        )
        
        bars = data_client.get_stock_bars(bars_request)
        df = bars.df
        
        if isinstance(df.index, pd.MultiIndex):
            df = df.xs(ticker, level='symbol')
        
        # Calculate metrics
        current_price = snap.latest_trade.price
        daily_high = snap.daily_bar.high
        daily_low = snap.daily_bar.low
        daily_open = snap.daily_bar.open
        daily_volume = snap.daily_bar.volume
        prev_close = snap.previous_daily_bar.close
        
        # Calculate price changes
        day_change = ((current_price - prev_close) / prev_close * 100)
        
        # Volume metrics
        avg_volume = df['volume'].mean()
        volume_ratio = daily_volume / avg_volume if avg_volume > 0 else 1
        
        # Price ranges
        week_52_high = df['high'].max()
        week_52_low = df['low'].min()
        price_vs_high = ((current_price - week_52_high) / week_52_high * 100)
        price_vs_low = ((current_price - week_52_low) / week_52_low * 100)
        
        # Volatility
        returns = df['close'].pct_change().dropna()
        volatility = returns.std() * np.sqrt(252)  # Annualized
        
        # Format fundamental data
        fundamental_data = f"""
FUNDAMENTAL ANALYSIS FOR {ticker}

Current Price: ${current_price:.2f}
Day Change: {day_change:+.2f}%
Daily Range: ${daily_low:.2f} - ${daily_high:.2f}

Price Performance:
- 52 Week High: ${week_52_high:.2f} (Current: {price_vs_high:+.1f}% from high)
- 52 Week Low: ${week_52_low:.2f} (Current: {price_vs_low:+.1f}% from low)
- Average Price (90d): ${df['close'].mean():.2f}

Volume Analysis:
- Current Volume: {daily_volume:,.0f}
- Average Volume (90d): {avg_volume:,.0f}
- Volume Ratio: {volume_ratio:.2f}x

Risk Metrics:
- Annualized Volatility: {volatility*100:.2f}%
- Daily Average Return: {returns.mean()*100:.2f}%

Price Stability:
- Standard Deviation (90d): ${df['close'].std():.2f}
- Price Range (90d): ${df['close'].max():.2f} - ${df['close'].min():.2f}
"""
        
        # Get LLM analysis
        analysis_prompt = f"""As a fundamental analyst, analyze this company's financial metrics:

{fundamental_data}

Provide a comprehensive fundamental analysis including:
1. Price valuation (relative to 52-week range)
2. Volume and liquidity assessment
3. Volatility and risk evaluation
4. Price momentum and trends
5. Investment rating (Strong Buy/Buy/Hold/Sell/Strong Sell)

Be specific and data-driven."""
        
        response = llm.invoke([HumanMessage(content=analysis_prompt)])
        analysis = response.content
        
        agent_message = f"📊 Fundamental Analyst: Completed analysis for {ticker}\n\nPrice: ${current_price:.2f} ({day_change:+.2f}%)\n\n{analysis[:400]}..."
        
        return {
            "messages": [AIMessage(content=agent_message)],
            "fundamental_analysis": f"{fundamental_data}\n\nAnalysis:\n{analysis}",
            "next_agent": "supervisor"
        }
        
    except Exception as e:
        error_msg = f"📊 Fundamental Analyst: Error analyzing {ticker}: {str(e)}"
        print(error_msg)
        return {
            "messages": [AIMessage(content=error_msg)],
            "fundamental_analysis": f"Error: {str(e)}",
            "next_agent": "supervisor"
        }


# ===================================
# Agent 2: Technical Analyst (Using Alpaca)
# ===================================

def technical_analyst(state: TradingState) -> Dict:
    """Analyzes technical indicators using Alpaca data"""
    
    ticker = state.get("ticker", "")
    print(f"\n📈 Technical Analyst working on {ticker}...")
    
    try:
        # Get historical data from Alpaca
        end = datetime.now()
        start = end - timedelta(days=180)
        
        request = StockBarsRequest(
            symbol_or_symbols=ticker,
            timeframe=TimeFrame.Day,
            start=start,
            end=end
        )
        
        bars = data_client.get_stock_bars(request)
        df = bars.df
        
        if isinstance(df.index, pd.MultiIndex):
            df = df.xs(ticker, level='symbol')
        
        if df.empty:
            raise ValueError("No historical data available")
        
        # Calculate technical indicators
        # Moving Averages
        df['SMA_20'] = df['close'].rolling(window=20).mean()
        df['SMA_50'] = df['close'].rolling(window=50).mean()
        df['EMA_12'] = df['close'].ewm(span=12, adjust=False).mean()
        df['EMA_26'] = df['close'].ewm(span=26, adjust=False).mean()
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
        
        # Bollinger Bands
        df['BB_Middle'] = df['close'].rolling(window=20).mean()
        bb_std = df['close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
        df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
        
        # Get latest values
        latest = df.iloc[-1]
        current_price = latest['close']
        rsi = latest['RSI']
        macd = latest['MACD']
        signal = latest['Signal_Line']
        sma_20 = latest['SMA_20']
        sma_50 = latest['SMA_50']
        
        # Calculate price changes
        price_change_1d = ((current_price - df['close'].iloc[-2]) / df['close'].iloc[-2] * 100) if len(df) > 1 else 0
        price_change_5d = ((current_price - df['close'].iloc[-6]) / df['close'].iloc[-6] * 100) if len(df) > 5 else 0
        price_change_20d = ((current_price - df['close'].iloc[-21]) / df['close'].iloc[-21] * 100) if len(df) > 20 else 0
        
        # Format technical data
        technical_data = f"""
TECHNICAL ANALYSIS FOR {ticker}

Current Price: ${current_price:.2f}
Price Changes: 1D: {price_change_1d:.2f}%, 5D: {price_change_5d:.2f}%, 20D: {price_change_20d:.2f}%

Moving Averages:
- SMA 20: ${sma_20:.2f} (Price {((current_price/sma_20 - 1)*100):+.2f}%)
- SMA 50: ${sma_50:.2f} (Price {((current_price/sma_50 - 1)*100):+.2f}%)
- Trend: {'BULLISH ⬆' if current_price > sma_20 > sma_50 else 'BEARISH ⬇' if current_price < sma_20 < sma_50 else 'NEUTRAL ➡'}

Momentum Indicators:
- RSI (14): {rsi:.2f} {'[OVERBOUGHT]' if rsi > 70 else '[OVERSOLD]' if rsi < 30 else '[NEUTRAL]'}
- MACD: {macd:.4f}
- Signal Line: {signal:.4f}
- MACD Histogram: {(macd-signal):.4f} {'[BULLISH]' if macd > signal else '[BEARISH]'}

Bollinger Bands:
- Upper: ${latest['BB_Upper']:.2f}
- Middle: ${latest['BB_Middle']:.2f}
- Lower: ${latest['BB_Lower']:.2f}
- Position: {((current_price - latest['BB_Lower'])/(latest['BB_Upper']-latest['BB_Lower'])*100):.1f}%

Volume:
- Latest: {latest['volume']:,.0f}
- Avg 20D: {df['volume'].tail(20).mean():,.0f}
- Volume Trend: {'HIGH' if latest['volume'] > df['volume'].tail(20).mean() else 'LOW'}
"""
        
        # Get LLM analysis
        analysis_prompt = f"""As a technical analyst, analyze these technical indicators for {ticker}:

{technical_data}

Provide comprehensive technical analysis including:
1. Trend analysis (uptrend/downtrend/sideways)
2. Momentum signals (RSI, MACD interpretation)
3. Support and resistance levels
4. Volume analysis
5. Entry/exit signals
6. Technical rating (Strong Buy/Buy/Hold/Sell/Strong Sell)

Be specific about price levels and timing."""
        
        response = llm.invoke([HumanMessage(content=analysis_prompt)])
        analysis = response.content
        
        agent_message = f"📈 Technical Analyst: Completed technical analysis for {ticker}\n\nRSI: {rsi:.1f}, MACD: {'BULLISH' if macd > signal else 'BEARISH'}\n\n{analysis[:400]}..."
        
        return {
            "messages": [AIMessage(content=agent_message)],
            "technical_analysis": f"{technical_data}\n\nAnalysis:\n{analysis}",
            "next_agent": "supervisor"
        }
        
    except Exception as e:
        error_msg = f"📈 Technical Analyst: Error analyzing {ticker}: {str(e)}"
        print(error_msg)
        return {
            "messages": [AIMessage(content=error_msg)],
            "technical_analysis": f"Error: {str(e)}",
            "next_agent": "supervisor"
        }


# ===================================
# Agent 3: News Reader
# ===================================

def news_reader(state: TradingState) -> Dict:
    """Analyzes news sentiment for the stock"""
    
    ticker = state.get("ticker", "")
    print(f"\n📰 News Reader working on {ticker}...")
    
    try:
        # For demo purposes, we'll use general market sentiment
        # In production, you could integrate NewsAPI or other news sources
        
        news_data = f"""Based on recent market trends and {ticker} performance:
- Market monitoring ongoing
- Technical and fundamental data analyzed
- Trading activity assessed
"""
        
        # Get LLM sentiment analysis
        sentiment_prompt = f"""As a news analyst, analyze the overall market sentiment for {ticker}:

Recent Context:
{news_data}

Based on the overall market conditions, recent price movements, and general sentiment, provide:

1. Overall sentiment (Positive/Neutral/Negative)
2. Market perception assessment
3. Potential catalysts or risks for {ticker}
4. Short-term outlook
5. Sentiment-based recommendation impact

Rate the sentiment from 1-10 (1=Very Negative, 10=Very Positive).
Be balanced and realistic."""
        
        response = llm.invoke([HumanMessage(content=sentiment_prompt)])
        analysis = response.content
        
        agent_message = f"📰 News Reader: Completed sentiment analysis for {ticker}\n\n{analysis[:500]}..."
        
        return {
            "messages": [AIMessage(content=agent_message)],
            "news_sentiment": f"{news_data}\n\nSentiment Analysis:\n{analysis}",
            "next_agent": "supervisor"
        }
        
    except Exception as e:
        error_msg = f"📰 News Reader: Error analyzing sentiment for {ticker}: {str(e)}"
        print(error_msg)
        return {
            "messages": [AIMessage(content=error_msg)],
            "news_sentiment": f"Error: {str(e)}",
            "next_agent": "supervisor"
        }


# ===================================
# Agent 4: Trade Placer
# ===================================

def direct_trade_executor(state: TradingState) -> Dict:
    """Executes direct trades without analysis"""
    
    ticker = state.get("ticker", "")
    action = state.get("trade_action", "BUY")
    quantity = state.get("trade_quantity", 0)
    
    print(f"\n⚡ Direct Trade Executor: {action} {quantity} shares of {ticker}...")
    
    try:
        if not trading_client or not data_client:
            raise ValueError("Alpaca client not initialized")
        
        # Get current price
        request = StockSnapshotRequest(symbol_or_symbols=ticker)
        snapshot = data_client.get_stock_snapshot(request)
        current_price = snapshot[ticker].latest_trade.price
        
        if action.upper() == "BUY":
            # Execute BUY order
            market_order_data = MarketOrderRequest(
                symbol=ticker,
                qty=quantity,
                side=OrderSide.BUY,
                time_in_force=TimeInForce.DAY
            )
            
            order = trading_client.submit_order(order_data=market_order_data)
            
            result_msg = f"""⚡ DIRECT BUY EXECUTED:
✅ Symbol: {ticker}
✅ Quantity: {quantity} shares
✅ Price: ${current_price:.2f}
✅ Total Value: ${quantity * current_price:.2f}
✅ Order ID: {order.id}
✅ Status: {order.status}

⚠️ Note: Trade executed WITHOUT analysis. Use with caution!
"""
        
        elif action.upper() == "SELL":
            # Execute SELL order
            market_order_data = MarketOrderRequest(
                symbol=ticker,
                qty=quantity,
                side=OrderSide.SELL,
                time_in_force=TimeInForce.DAY
            )
            
            order = trading_client.submit_order(order_data=market_order_data)
            
            result_msg = f"""⚡ DIRECT SELL EXECUTED:
✅ Symbol: {ticker}
✅ Quantity: {quantity} shares
✅ Price: ${current_price:.2f}
✅ Total Value: ${quantity * current_price:.2f}
✅ Order ID: {order.id}
✅ Status: {order.status}

⚠️ Note: Trade executed WITHOUT analysis. Use with caution!
"""
        else:
            result_msg = f"❌ Invalid trade action: {action}. Use BUY or SELL."
        
        return {
            "messages": [AIMessage(content=result_msg)],
            "trade_recommendation": result_msg,
            "trade_executed": True,
            "next_agent": "end",
            "task_complete": True
        }
        
    except Exception as e:
        error_msg = f"❌ Direct Trade Executor: Failed to execute {action} {quantity} shares of {ticker}: {str(e)}"
        return {
            "messages": [AIMessage(content=error_msg)],
            "trade_recommendation": error_msg,
            "trade_executed": True,
            "next_agent": "end",
            "task_complete": True
        }


def trade_placer(state: TradingState) -> Dict:
    """Makes final trade decision and executes via Alpaca"""
    
    ticker = state.get("ticker", "")
    fundamental_analysis = state.get("fundamental_analysis", "")
    technical_analysis = state.get("technical_analysis", "")
    news_sentiment = state.get("news_sentiment", "")
    
    print(f"\n💼 Trade Placer making decision for {ticker}...")
    
    try:
        # Compile all analysis
        full_analysis = f"""
TICKER: {ticker}

FUNDAMENTAL ANALYSIS:
{fundamental_analysis}

TECHNICAL ANALYSIS:
{technical_analysis}

NEWS SENTIMENT:
{news_sentiment}
"""
        
        # Get LLM trade decision
        trade_prompt = f"""As a professional trader, review all the analysis and make a final trade decision:

{full_analysis}

Based on ALL THREE analyses (fundamental, technical, and sentiment), provide:

1. Overall recommendation: BUY, SELL, or HOLD
2. Confidence level: 1-10 (10=highest confidence)
3. Position size recommendation: Conservative(2%)/Moderate(5%)/Aggressive(10%)
4. Risk assessment
5. Entry price target
6. Stop loss level (%)
7. Take profit target (%)
8. Time horizon (Short/Medium/Long term)
9. Key reasoning for the decision

IMPORTANT: Be conservative. Only recommend BUY if there's strong alignment across analyses.
Format your response clearly with the recommendation at the top."""
        
        response = llm.invoke([HumanMessage(content=trade_prompt)])
        trade_decision = response.content
        
        # Parse recommendation
        decision_lower = trade_decision.lower()
        action = None
        if "strong buy" in decision_lower:
            action = "STRONG_BUY"
        elif "buy" in decision_lower and "don't buy" not in decision_lower and "not buy" not in decision_lower:
            action = "BUY"
        elif "sell" in decision_lower:
            action = "SELL"
        else:
            action = "HOLD"
        
        # Execute trade if recommendation is BUY
        trade_result = ""
        if action in ["BUY", "STRONG_BUY"] and trading_client:
            try:
                # Get account info
                account = trading_client.get_account()
                buying_power = float(account.buying_power)
                
                # Calculate position size (conservative: 2-5% of buying power)
                position_size_pct = 0.02 if action == "BUY" else 0.05
                position_value = buying_power * position_size_pct
                
                # Get current price from snapshot
                request = StockSnapshotRequest(symbol_or_symbols=ticker)
                snapshot = data_client.get_stock_snapshot(request)
                current_price = snapshot[ticker].latest_trade.price
                
                # Calculate shares (round down to avoid insufficient funds)
                shares = int(position_value / current_price)
                
                if shares > 0:
                    # Create market order
                    market_order_data = MarketOrderRequest(
                        symbol=ticker,
                        qty=shares,
                        side=OrderSide.BUY,
                        time_in_force=TimeInForce.DAY
                    )
                    
                    # Submit order
                    order = trading_client.submit_order(order_data=market_order_data)
                    
                    trade_result = f"""
✅ TRADE EXECUTED:
- Action: BUY
- Symbol: {ticker}
- Quantity: {shares} shares
- Estimated Price: ${current_price:.2f}
- Estimated Value: ${shares * current_price:.2f}
- Order ID: {order.id}
- Status: {order.status}
"""
                else:
                    trade_result = f"⚠️ Position size too small. Calculated {shares} shares. No trade executed."
                    
            except Exception as e:
                trade_result = f"❌ Trade execution failed: {str(e)}"
        elif action == "SELL":
            trade_result = "📌 SELL signal generated. Check if position exists to close."
        else:
            trade_result = f"📌 Decision: {action}. No trade executed."
        
        agent_message = f"""💼 Trade Placer: Analysis complete!

{trade_decision[:600]}...

{trade_result}
"""
        
        return {
            "messages": [AIMessage(content=agent_message)],
            "trade_recommendation": trade_decision + "\n\n" + trade_result,
            "trade_executed": True,
            "next_agent": "supervisor",
            "task_complete": True
        }
        
    except Exception as e:
        error_msg = f"💼 Trade Placer: Error making trade decision: {str(e)}"
        print(error_msg)
        return {
            "messages": [AIMessage(content=error_msg)],
            "trade_recommendation": f"Error: {str(e)}",
            "trade_executed": True,
            "next_agent": "supervisor"
        }


# ===================================
# Router Function
# ===================================

def router(state: TradingState) -> Literal["supervisor", "fundamental_analyst", "technical_analyst", "news_reader", "trade_placer", "direct_trade_executor", "__end__"]:
    """Routes to next agent based on state"""
    
    next_agent = state.get("next_agent", "supervisor")
    
    if next_agent == "end" or state.get("task_complete", False):
        return END
        
    if next_agent in ["supervisor", "fundamental_analyst", "technical_analyst", "news_reader", "trade_placer", "direct_trade_executor"]:
        return next_agent
        
    return "supervisor"


# ===================================
# Build the Workflow
# ===================================

def create_trading_workflow():
    """Create the supervised trading multi-agent workflow"""
    
    # Create workflow
    workflow = StateGraph(TradingState)
    
    # Add nodes
    workflow.add_node("supervisor", supervisor_agent)
    workflow.add_node("fundamental_analyst", fundamental_analyst)
    workflow.add_node("technical_analyst", technical_analyst)
    workflow.add_node("news_reader", news_reader)
    workflow.add_node("trade_placer", trade_placer)
    workflow.add_node("direct_trade_executor", direct_trade_executor)
    
    # Set entry point
    workflow.set_entry_point("supervisor")
    
    # Add routing
    for node in ["supervisor", "fundamental_analyst", "technical_analyst", "news_reader", "trade_placer", "direct_trade_executor"]:
        workflow.add_conditional_edges(
            node,
            router,
            {
                "supervisor": "supervisor",
                "fundamental_analyst": "fundamental_analyst",
                "technical_analyst": "technical_analyst",
                "news_reader": "news_reader",
                "trade_placer": "trade_placer",
                "direct_trade_executor": "direct_trade_executor",
                END: END
            }
        )
    
    # Compile graph with recursion limit
    graph = workflow.compile()
    return graph


# ===================================
# Helper Functions
# ===================================

def execute_direct_trade(ticker: str, action: str, quantity: int):
    """
    Execute a direct trade without analysis
    
    Args:
        ticker: Stock symbol (e.g., 'AAPL', 'TSLA')
        action: 'BUY' or 'SELL'
        quantity: Number of shares
    """
    graph = create_trading_workflow()
    
    response = graph.invoke(
        {
            "messages": [HumanMessage(content=f"{action} {quantity} shares of {ticker} immediately")],
            "ticker": ticker,
            "direct_trade": True,
            "trade_action": action,
            "trade_quantity": quantity
        },
        config={"recursion_limit": 50}
    )
    
    return response


def analyze_and_trade(ticker: str):
    """
    Analyze a stock with all agents and make a trade decision
    
    Args:
        ticker: Stock symbol (e.g., 'AAPL', 'TSLA')
    """
    graph = create_trading_workflow()
    
    response = graph.invoke(
        {
            "messages": [HumanMessage(content=f"Analyze {ticker} and determine if we should buy, sell, or hold.")],
            "ticker": ticker,
            "direct_trade": False
        },
        config={"recursion_limit": 50}
    )
    
    return response


# ===================================
# Main Execution
# ===================================

if __name__ == "__main__":
    import sys
    
    print("\n" + "="*60)
    print("🤖 Stock Trading Multi-Agent System")
    print("="*60 + "\n")
    
    # Check command line arguments
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
        
        if mode == "direct" and len(sys.argv) >= 5:
            # Direct trade mode: python stock_agent.py direct BUY TSLA 10
            action = sys.argv[2].upper()
            ticker = sys.argv[3].upper()
            quantity = int(sys.argv[4])
            
            print(f"⚡ MODE: DIRECT TRADE")
            print(f"📝 Action: {action} {quantity} shares of {ticker}")
            print(f"⚠️  WARNING: Executing without analysis!\n")
            print("-"*60)
            
            response = execute_direct_trade(ticker, action, quantity)
            
        elif mode == "analyze" and len(sys.argv) >= 3:
            # Analysis mode: python stock_agent.py analyze TSLA
            ticker = sys.argv[2].upper()
            
            print(f"📊 MODE: FULL ANALYSIS")
            print(f"🎯 Target: {ticker}")
            print(f"🔍 Running comprehensive analysis...\n")
            print("-"*60)
            
            response = analyze_and_trade(ticker)
            
        else:
            print("❌ Invalid arguments!")
            print("\nUsage:")
            print("  Full Analysis: python stock_agent.py analyze TICKER")
            print("  Direct Trade:  python stock_agent.py direct BUY/SELL TICKER QUANTITY")
            print("\nExamples:")
            print("  python stock_agent.py analyze TSLA")
            print("  python stock_agent.py direct BUY AAPL 5")
            print("  python stock_agent.py direct SELL MSFT 10")
            sys.exit(1)
    else:
        # Default mode: Full analysis
        ticker = "TSLA"
        
        print(f"📊 MODE: FULL ANALYSIS (Default)")
        print(f"🎯 Target: {ticker}")
        print(f"🔍 Running comprehensive analysis...\n")
        print("-"*60)
        
        response = analyze_and_trade(ticker)
    
    # Print conversation history
    print("\n" + "="*60)
    print("📋 WORKFLOW SUMMARY")
    print("="*60 + "\n")
    
    for msg in response["messages"]:
        if hasattr(msg, 'content'):
            print(msg.content)
            print("-"*60 + "\n")
    
    # Print final recommendation
    if response.get("trade_recommendation"):
        print("\n" + "="*60)
        print("💼 FINAL TRADE RECOMMENDATION")
        print("="*60)
        print(response["trade_recommendation"])
    
    # Print account status if Alpaca is connected
    if trading_client:
        try:
            account = trading_client.get_account()
            print("\n" + "="*60)
            print("💰 ACCOUNT STATUS")
            print("="*60)
            print(f"Buying Power: ${float(account.buying_power):,.2f}")
            print(f"Cash: ${float(account.cash):,.2f}")
            print(f"Portfolio Value: ${float(account.portfolio_value):,.2f}")
            print(f"Account Status: {account.status}")
        except Exception as e:
            print(f"\n⚠️ Could not retrieve account status: {e}")
    
    print("\n" + "="*60)
    print("✅ Trading Workflow Complete!")
    print("="*60 + "\n")

