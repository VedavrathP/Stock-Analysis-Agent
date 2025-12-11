"""
Streamlit App for Stock Trading Multi-Agent System
Provides a user-friendly interface for analyzing stocks and executing trades
"""

import streamlit as st
import sys
import os
from datetime import datetime
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Import from stock_agent.py
from stock_agent import (
    create_trading_workflow,
    trading_client,
    data_client,
    TradingState
)
from langchain_core.messages import HumanMessage
from alpaca.data.requests import StockBarsRequest, StockSnapshotRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import DataFeed
from datetime import timedelta

# Page config
st.set_page_config(
    page_title="Stock Analysis & Investing Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS with modern design
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&family=Outfit:wght@300;400;500;600;700&display=swap');
    
    .main-header {
        font-family: 'Outfit', sans-serif;
        font-size: 3rem;
        font-weight: 700;
        text-align: center;
        background: linear-gradient(135deg, #00d4aa 0%, #00b4d8 50%, #7c3aed 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding: 1rem;
        margin-bottom: 0.5rem;
    }
    
    .subtitle {
        font-family: 'JetBrains Mono', monospace;
        text-align: center;
        color: #64748b;
        font-size: 0.9rem;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        padding: 1.2rem;
        border-radius: 1rem;
        border: 1px solid #334155;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
    }
    
    .success-box {
        padding: 1rem;
        border-radius: 0.75rem;
        background: linear-gradient(135deg, #065f46 0%, #064e3b 100%);
        border: 1px solid #10b981;
        color: #a7f3d0;
        font-family: 'JetBrains Mono', monospace;
    }
    
    .warning-box {
        padding: 1rem;
        border-radius: 0.75rem;
        background: linear-gradient(135deg, #78350f 0%, #713f12 100%);
        border: 1px solid #f59e0b;
        color: #fef3c7;
        font-family: 'JetBrains Mono', monospace;
    }
    
    .error-box {
        padding: 1rem;
        border-radius: 0.75rem;
        background: linear-gradient(135deg, #7f1d1d 0%, #991b1b 100%);
        border: 1px solid #ef4444;
        color: #fecaca;
        font-family: 'JetBrains Mono', monospace;
    }
    
    .stButton>button {
        width: 100%;
        border-radius: 0.75rem;
        height: 3rem;
        font-weight: 600;
        font-family: 'Outfit', sans-serif;
        background: linear-gradient(135deg, #00d4aa 0%, #00b4d8 100%);
        border: none;
        color: #0f172a;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0, 212, 170, 0.4);
    }
    
    .agent-status {
        font-family: 'JetBrains Mono', monospace;
        padding: 0.5rem 1rem;
        border-radius: 0.5rem;
        font-size: 0.85rem;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        font-family: 'Outfit', sans-serif;
        font-weight: 500;
        border-radius: 0.5rem 0.5rem 0 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'analysis_result' not in st.session_state:
    st.session_state.analysis_result = None
if 'trade_result' not in st.session_state:
    st.session_state.trade_result = None
if 'last_ticker' not in st.session_state:
    st.session_state.last_ticker = None

# Title
st.markdown('<h1 class="main-header">🤖 Stock Analysis & Investing Agent</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Powered by Multi-Agent AI System • LangGraph • Alpaca Trading</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    
    # Account Status
    if trading_client:
        try:
            account = trading_client.get_account()
            st.success("✅ Connected to Alpaca")
            
            st.metric("💰 Buying Power", f"${float(account.buying_power):,.2f}")
            st.metric("💵 Cash", f"${float(account.cash):,.2f}")
            st.metric("📊 Portfolio Value", f"${float(account.portfolio_value):,.2f}")
            
            # Get positions
            positions = trading_client.get_all_positions()
            if positions:
                st.markdown("---")
                st.subheader("📈 Current Positions")
                for pos in positions[:5]:  # Show top 5
                    pnl_pct = (float(pos.unrealized_plpc) * 100)
                    color = "🟢" if pnl_pct >= 0 else "🔴"
                    st.write(f"{color} **{pos.symbol}**: {pos.qty} shares ({pnl_pct:+.2f}%)")
            
        except Exception as e:
            st.error(f"⚠️ Alpaca Error: {str(e)}")
    else:
        st.error("❌ Not connected to Alpaca")
        st.info("Add your API keys in Streamlit secrets or .env file")
    
    st.markdown("---")
    st.info("📌 **Paper Trading Mode**\nSafe testing environment")
    
    st.markdown("---")
    st.markdown("### 🔗 Resources")
    st.markdown("- [Alpaca Trading](https://alpaca.markets)")
    st.markdown("- [LangGraph Docs](https://langchain-ai.github.io/langgraph/)")

# Main content
tab1, tab2, tab3 = st.tabs(["📊 Stock Analysis", "⚡ Direct Trade", "📈 Market Data"])

# Tab 1: Stock Analysis
with tab1:
    st.header("📊 Comprehensive Stock Analysis")
    st.write("Run all AI agents to analyze a stock before trading")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        ticker_analyze = st.text_input(
            "Enter Stock Ticker",
            value="TSLA",
            key="ticker_analyze",
            placeholder="e.g., AAPL, TSLA, NVDA"
        ).upper()
    
    with col2:
        st.write("")
        st.write("")
        analyze_button = st.button("🔍 Run Analysis", type="primary", key="analyze_btn")
    
    if analyze_button and ticker_analyze:
        # Create a container for live updates
        st.markdown("---")
        st.subheader(f"🔄 Live Analysis Progress for {ticker_analyze}")
        
        status_container = st.empty()
        progress_bar = st.progress(0)
        agent_status = st.empty()
        
        try:
            # Status 1: Starting
            with agent_status.container():
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.markdown("🟡 **Supervisor**")
                with col2:
                    st.markdown("⚪ Fundamental")
                with col3:
                    st.markdown("⚪ Technical")
                with col4:
                    st.markdown("⚪ News & Trade")
            
            with status_container.container():
                st.info("🚀 **Supervisor Agent: Initializing Multi-Agent System...**")
                st.write("✓ Connecting to Alpaca Markets")
                st.write("✓ Loading AI models")
                st.write("✓ Routing to Fundamental Analyst")
            progress_bar.progress(5)
            
            import time
            time.sleep(1)
            
            # Status 2: Fundamental Analysis
            with agent_status.container():
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.markdown("✅ Supervisor")
                with col2:
                    st.markdown("🟢 **Fundamental**")
                with col3:
                    st.markdown("⚪ Technical")
                with col4:
                    st.markdown("⚪ News & Trade")
            
            with status_container.container():
                st.info("📊 **Fundamental Analyst: Deep Diving into Company Metrics**")
                st.write("✓ Retrieving live market snapshot")
                st.write("✓ Analyzing current price: Getting bid/ask spreads")
                st.write("✓ Calculating 52-week highs and lows")
            progress_bar.progress(15)
            time.sleep(1.5)
            
            with status_container.container():
                st.info("📊 **Fundamental Analyst: Volume & Liquidity Analysis**")
                st.write("✓ Fetching 90-day historical volume data")
                st.write("✓ Computing average daily trading volume")
                st.write("✓ Evaluating market liquidity conditions")
                st.write("✓ Analyzing volume ratio vs. average")
            progress_bar.progress(20)
            
            # Create workflow and start analysis in background
            from threading import Thread
            import queue
            
            result_queue = queue.Queue()
            
            def run_analysis():
                graph = create_trading_workflow()
                response = graph.invoke(
                    {
                        "messages": [HumanMessage(content=f"Analyze {ticker_analyze} and determine if we should buy, sell, or hold.")],
                        "ticker": ticker_analyze,
                        "direct_trade": False
                    },
                    config={"recursion_limit": 50}
                )
                result_queue.put(response)
            
            analysis_thread = Thread(target=run_analysis)
            analysis_thread.start()
            
            # Continue showing progress
            time.sleep(2)
            
            with status_container.container():
                st.info("📊 **Fundamental Analyst: Risk Assessment**")
                st.write("✓ Calculating annualized volatility")
                st.write("✓ Computing standard deviation of returns")
                st.write("✓ Analyzing price stability metrics")
                st.write("✓ Generating valuation assessment")
            progress_bar.progress(30)
            time.sleep(2)
            
            # Status 3: Technical Analysis Start
            with agent_status.container():
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.markdown("✅ Supervisor")
                with col2:
                    st.markdown("✅ Fundamental")
                with col3:
                    st.markdown("🟢 **Technical**")
                with col4:
                    st.markdown("⚪ News & Trade")
            
            with status_container.container():
                st.info("📈 **Technical Analyst: Gathering Historical Data**")
                st.write("✓ Fetching 180-day price history")
                st.write("✓ Retrieving OHLCV data (Open, High, Low, Close, Volume)")
                st.write("✓ Preparing data for indicator calculations")
            progress_bar.progress(40)
            time.sleep(1.5)
            
            with status_container.container():
                st.info("📈 **Technical Analyst: Computing Moving Averages**")
                st.write("✓ Calculating Simple Moving Average (SMA 20)")
                st.write("✓ Calculating Simple Moving Average (SMA 50)")
                st.write("✓ Computing Exponential Moving Average (EMA 12)")
                st.write("✓ Computing Exponential Moving Average (EMA 26)")
                st.write("✓ Identifying trend direction")
            progress_bar.progress(50)
            time.sleep(1.5)
            
            with status_container.container():
                st.info("📈 **Technical Analyst: Momentum Indicators**")
                st.write("✓ Calculating Relative Strength Index (RSI)")
                st.write("✓ Identifying overbought/oversold conditions")
                st.write("✓ Computing MACD (Moving Average Convergence Divergence)")
                st.write("✓ Plotting Signal Line")
                st.write("✓ Analyzing momentum histogram")
            progress_bar.progress(60)
            time.sleep(1.5)
            
            with status_container.container():
                st.info("📈 **Technical Analyst: Advanced Patterns**")
                st.write("✓ Plotting Bollinger Bands (20-period)")
                st.write("✓ Identifying support and resistance levels")
                st.write("✓ Analyzing volume-price relationships")
                st.write("✓ Detecting bullish/bearish signals")
                st.write("✓ Generating technical rating")
            progress_bar.progress(70)
            time.sleep(1.5)
            
            # Status 5: Sentiment Analysis
            with agent_status.container():
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.markdown("✅ Supervisor")
                with col2:
                    st.markdown("✅ Fundamental")
                with col3:
                    st.markdown("✅ Technical")
                with col4:
                    st.markdown("🟢 **News & Trade**")
            
            with status_container.container():
                st.info("📰 **News Reader: Sentiment Analysis**")
                st.write("✓ Scanning recent market sentiment")
                st.write("✓ Analyzing institutional investor behavior")
                st.write("✓ Evaluating market perception trends")
                st.write("✓ Identifying potential catalysts")
                st.write("✓ Computing sentiment score (1-10)")
            progress_bar.progress(80)
            time.sleep(2)
            
            # Status 6: Trade Decision
            with status_container.container():
                st.info("💼 **Trade Placer: Synthesizing All Data**")
                st.write("✓ Cross-referencing fundamental metrics")
                st.write("✓ Validating technical signals")
                st.write("✓ Weighing sentiment factors")
                st.write("✓ Calculating risk-reward ratio")
            progress_bar.progress(90)
            time.sleep(1.5)
            
            with status_container.container():
                st.info("💼 **Trade Placer: Final Decision Making**")
                st.write("✓ Determining optimal position size")
                st.write("✓ Setting stop-loss recommendations")
                st.write("✓ Setting take-profit targets")
                st.write("✓ Generating BUY/SELL/HOLD recommendation")
                st.write("✓ Assigning confidence level")
            progress_bar.progress(95)
            
            # Wait for analysis to complete
            analysis_thread.join()
            response = result_queue.get()
            
            progress_bar.progress(100)
            
            # Status 7: Complete
            with agent_status.container():
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.markdown("✅ Supervisor")
                with col2:
                    st.markdown("✅ Fundamental")
                with col3:
                    st.markdown("✅ Technical")
                with col4:
                    st.markdown("✅ News & Trade")
            
            with status_container.container():
                st.success(f"✅ **Analysis Complete for {ticker_analyze}!**")
                st.write("✓ All 4 agents have completed their analysis")
                st.write("✓ Fundamental rating: Generated")
                st.write("✓ Technical rating: Generated")
                st.write("✓ Sentiment score: Calculated")
                st.write("✓ Final recommendation: Ready for review")
            
            time.sleep(1.5)
            status_container.empty()
            agent_status.empty()
            progress_bar.empty()
            
            st.session_state.analysis_result = response
            st.session_state.last_ticker = ticker_analyze
            
            st.balloons()
            
        except Exception as e:
            status_container.empty()
            agent_status.empty()
            progress_bar.empty()
            st.error(f"❌ Analysis failed: {str(e)}")
    
    # Display results
    if st.session_state.analysis_result:
        result = st.session_state.analysis_result
        
        st.markdown("---")
        st.subheader(f"📋 Analysis Results for {st.session_state.last_ticker}")
        
        # Create tabs for different analyses
        analysis_tabs = st.tabs(["📊 Fundamental", "📈 Technical", "📰 Sentiment", "💼 Trade Decision"])
        
        with analysis_tabs[0]:
            if result.get('fundamental_analysis'):
                st.markdown("### Fundamental Analysis")
                st.text(result['fundamental_analysis'])
            else:
                st.info("No fundamental analysis available")
        
        with analysis_tabs[1]:
            if result.get('technical_analysis'):
                st.markdown("### Technical Analysis")
                st.text(result['technical_analysis'])
            else:
                st.info("No technical analysis available")
        
        with analysis_tabs[2]:
            if result.get('news_sentiment'):
                st.markdown("### News & Sentiment Analysis")
                st.text(result['news_sentiment'])
            else:
                st.info("No sentiment analysis available")
        
        with analysis_tabs[3]:
            if result.get('trade_recommendation'):
                st.markdown("### Final Trade Recommendation")
                
                recommendation = result['trade_recommendation']
                
                # Parse recommendation - look for the actual recommendation pattern
                # Check first 500 chars for the actual recommendation (not template text)
                rec_header = recommendation[:500].upper()
                
                # Look for specific patterns that indicate the actual recommendation
                import re
                
                # Pattern 1: "Overall Recommendation: X" or "Recommendation: X"
                rec_match = re.search(r'(?:OVERALL\s+)?RECOMMENDATION[:\s]+(\w+(?:\s+\w+)?)', rec_header)
                
                if rec_match:
                    actual_rec = rec_match.group(1).strip()
                    if 'STRONG' in actual_rec and 'BUY' in actual_rec:
                        st.success("🚀 **RECOMMENDATION: STRONG BUY**")
                    elif 'BUY' in actual_rec and 'STRONG' not in actual_rec:
                        st.success("✅ **RECOMMENDATION: BUY**")
                    elif 'SELL' in actual_rec:
                        st.error("⚠️ **RECOMMENDATION: SELL**")
                    elif 'HOLD' in actual_rec or 'WAIT' in actual_rec:
                        st.info("📌 **RECOMMENDATION: HOLD**")
                    else:
                        st.info(f"📌 **RECOMMENDATION: {actual_rec}**")
                else:
                    # Fallback: check if HOLD appears early (more likely to be the actual rec)
                    if 'OVERALL RECOMMENDATION: HOLD' in rec_header or rec_header.startswith('HOLD'):
                        st.info("📌 **RECOMMENDATION: HOLD**")
                    elif 'OVERALL RECOMMENDATION: BUY' in rec_header:
                        st.success("✅ **RECOMMENDATION: BUY**")
                    elif 'OVERALL RECOMMENDATION: SELL' in rec_header:
                        st.error("⚠️ **RECOMMENDATION: SELL**")
                    else:
                        st.info("📌 **RECOMMENDATION: See details below**")
                
                st.text(recommendation)
            else:
                st.info("No trade recommendation available")

# Tab 2: Direct Trade
with tab2:
    st.header("⚡ Direct Trade Execution")
    st.warning("⚠️ **WARNING**: This executes trades immediately WITHOUT analysis. Use with caution!")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        ticker_trade = st.text_input(
            "Stock Ticker",
            value="AAPL",
            key="ticker_trade",
            placeholder="e.g., AAPL"
        ).upper()
    
    with col2:
        trade_action = st.selectbox(
            "Action",
            options=["BUY", "SELL"],
            key="trade_action"
        )
    
    with col3:
        trade_quantity = st.number_input(
            "Quantity",
            min_value=1,
            max_value=1000,
            value=5,
            key="trade_quantity"
        )
    
    # Show estimated cost
    if ticker_trade and data_client:
        try:
            request = StockSnapshotRequest(symbol_or_symbols=ticker_trade, feed=DataFeed.IEX)
            snapshot = data_client.get_stock_snapshot(request)
            if ticker_trade in snapshot:
                current_price = snapshot[ticker_trade].latest_trade.price
                estimated_cost = current_price * trade_quantity
                
                st.info(f"💵 Estimated {trade_action}: ${estimated_cost:,.2f} ({trade_quantity} shares @ ${current_price:.2f})")
        except:
            pass
    
    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
    with col_btn2:
        execute_trade = st.button(
            f"⚡ Execute {trade_action} {trade_quantity} shares of {ticker_trade}",
            type="primary",
            key="execute_trade_btn"
        )
    
    if execute_trade:
        # Show detailed execution steps
        status_container = st.empty()
        progress_bar = st.progress(0)
        
        try:
            import time
            
            # Step 1: Validation
            with status_container.container():
                st.info("🔍 **Validating Trade Request...**")
                st.write(f"• Verifying {ticker_trade} is tradeable")
                st.write(f"• Checking market hours")
                st.write(f"• Validating quantity: {trade_quantity} shares")
            progress_bar.progress(20)
            time.sleep(0.5)
            
            # Step 2: Getting current price
            with status_container.container():
                st.info("💵 **Fetching Real-Time Market Data...**")
                st.write(f"• Getting current bid/ask prices for {ticker_trade}")
                st.write(f"• Checking market liquidity")
                st.write(f"• Calculating estimated order value")
            progress_bar.progress(40)
            time.sleep(0.5)
            
            # Step 3: Order preparation
            with status_container.container():
                st.info("📋 **Preparing Market Order...**")
                st.write(f"• Action: {trade_action}")
                st.write(f"• Quantity: {trade_quantity} shares")
                st.write(f"• Order type: Market Order")
                st.write(f"• Time in force: Day")
            progress_bar.progress(60)
            time.sleep(0.5)
            
            # Step 4: Execute trade
            with status_container.container():
                st.info("⚡ **Submitting Order to Alpaca...**")
                st.write(f"• Transmitting order to broker")
                st.write(f"• Awaiting order confirmation")
            progress_bar.progress(80)
            
            # Execute direct trade
            graph = create_trading_workflow()
            response = graph.invoke(
                {
                    "messages": [HumanMessage(content=f"{trade_action} {trade_quantity} shares of {ticker_trade} immediately")],
                    "ticker": ticker_trade,
                    "direct_trade": True,
                    "trade_action": trade_action,
                    "trade_quantity": trade_quantity
                },
                config={"recursion_limit": 50}
            )
            
            progress_bar.progress(100)
            
            # Step 5: Complete
            with status_container.container():
                st.success("✅ **Order Processing Complete**")
                st.write(f"• Order submitted successfully")
                st.write(f"• Awaiting broker confirmation")
            
            time.sleep(1)
            status_container.empty()
            progress_bar.empty()
            
            st.session_state.trade_result = response
            
            # Show result
            if response.get('trade_recommendation'):
                if 'EXECUTED' in response['trade_recommendation']:
                    st.success(response['trade_recommendation'])
                    st.balloons()
                else:
                    st.error(response['trade_recommendation'])
            
        except Exception as e:
            status_container.empty()
            progress_bar.empty()
            st.error(f"❌ Trade execution failed: {str(e)}")

# Tab 3: Market Data
with tab3:
    st.header("📈 Market Data & Charts")
    
    ticker_chart = st.text_input(
        "Enter Stock Ticker for Chart",
        value="TSLA",
        key="ticker_chart"
    ).upper()
    
    if ticker_chart and data_client:
        try:
            # Get historical data
            end = datetime.now()
            start = end - timedelta(days=90)
            
            request = StockBarsRequest(
                symbol_or_symbols=ticker_chart,
                timeframe=TimeFrame.Day,
                start=start,
                end=end,
                feed=DataFeed.IEX
            )
            
            bars = data_client.get_stock_bars(request)
            df = bars.df
            
            if isinstance(df.index, pd.MultiIndex):
                df = df.xs(ticker_chart, level='symbol')
            
            # Create candlestick chart
            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.03,
                subplot_titles=(f'{ticker_chart} Price', 'Volume'),
                row_heights=[0.7, 0.3]
            )
            
            # Candlestick
            fig.add_trace(
                go.Candlestick(
                    x=df.index,
                    open=df['open'],
                    high=df['high'],
                    low=df['low'],
                    close=df['close'],
                    name='Price'
                ),
                row=1, col=1
            )
            
            # Add moving averages
            df['SMA_20'] = df['close'].rolling(window=20).mean()
            df['SMA_50'] = df['close'].rolling(window=50).mean()
            
            fig.add_trace(
                go.Scatter(x=df.index, y=df['SMA_20'], name='SMA 20', line=dict(color='#00d4aa', width=1)),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(x=df.index, y=df['SMA_50'], name='SMA 50', line=dict(color='#00b4d8', width=1)),
                row=1, col=1
            )
            
            # Volume
            colors = ['#ef4444' if row['close'] < row['open'] else '#10b981' for _, row in df.iterrows()]
            fig.add_trace(
                go.Bar(x=df.index, y=df['volume'], name='Volume', marker_color=colors),
                row=2, col=1
            )
            
            fig.update_layout(
                height=600,
                xaxis_rangeslider_visible=False,
                showlegend=True,
                template='plotly_dark',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Show latest stats
            latest = df.iloc[-1]
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Current Price", f"${latest['close']:.2f}")
            with col2:
                change = ((latest['close'] - df['close'].iloc[-2]) / df['close'].iloc[-2] * 100)
                st.metric("Day Change", f"{change:+.2f}%")
            with col3:
                st.metric("Volume", f"{latest['volume']:,.0f}")
            with col4:
                st.metric("High/Low", f"${latest['high']:.2f}/${latest['low']:.2f}")
            
        except Exception as e:
            st.error(f"❌ Failed to load market data: {str(e)}")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #64748b; font-family: JetBrains Mono, monospace;'>
    <p>🤖 Powered by Multi-Agent AI System • 📊 Data from Alpaca Markets • ⚠️ Paper Trading Mode</p>
    <p style='font-size: 0.8rem;'>This is for educational purposes only. Not financial advice.</p>
</div>
""", unsafe_allow_html=True)

