# 🤖 Stock Trading AI Agent

<div align="center">

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![LangGraph](https://img.shields.io/badge/LangGraph-Multi--Agent-green.svg)
![Alpaca](https://img.shields.io/badge/Alpaca-Paper%20Trading-yellow.svg)
[![License: MIT](https://img.shields.io/badge/License-MIT-purple.svg)](https://opensource.org/licenses/MIT)

**An intelligent multi-agent stock trading system powered by LangGraph and GPT-4**

[Live Demo](https://your-app-name.streamlit.app) • [Report Bug](https://github.com/yourusername/stock-trading-agent/issues) • [Request Feature](https://github.com/yourusername/stock-trading-agent/issues)

</div>

---

## 🎯 Overview

This project implements a **supervised multi-agent architecture** for stock trading analysis and execution. It uses specialized AI agents working together to provide comprehensive stock analysis before making trading decisions.

### 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    SUPERVISOR AGENT                          │
│              (Orchestrates workflow & routing)               │
└─────────────────────────┬───────────────────────────────────┘
                          │
        ┌─────────────────┼─────────────────┐
        ▼                 ▼                 ▼
┌───────────────┐ ┌───────────────┐ ┌───────────────┐
│  FUNDAMENTAL  │ │   TECHNICAL   │ │     NEWS      │
│   ANALYST     │ │    ANALYST    │ │    READER     │
│               │ │               │ │               │
│ • Price Data  │ │ • RSI, MACD   │ │ • Sentiment   │
│ • Volume      │ │ • SMA, EMA    │ │ • Market News │
│ • 52-wk Range │ │ • Bollinger   │ │ • Catalysts   │
│ • Volatility  │ │ • Support/Res │ │ • Risk Flags  │
└───────┬───────┘ └───────┬───────┘ └───────┬───────┘
        │                 │                 │
        └─────────────────┼─────────────────┘
                          ▼
              ┌───────────────────────┐
              │     TRADE PLACER      │
              │                       │
              │ • Final Decision      │
              │ • Position Sizing     │
              │ • Order Execution     │
              │ • Risk Management     │
              └───────────────────────┘
```

## ✨ Features

- **🤖 Multi-Agent System**: Four specialized AI agents working together
- **📊 Comprehensive Analysis**: Fundamental, Technical, and Sentiment analysis
- **📈 Real-Time Data**: Live market data from Alpaca Markets
- **⚡ Direct Trading**: Execute trades immediately or after analysis
- **🎨 Beautiful UI**: Modern Streamlit interface with dark theme
- **📱 Responsive Design**: Works on desktop and mobile
- **🔒 Paper Trading**: Safe testing environment (no real money risk)

## 🚀 Quick Start

### Prerequisites

- Python 3.9 or higher
- OpenAI API key
- Alpaca Markets account (free paper trading)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/stock-trading-agent.git
   cd stock-trading-agent
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   
   Create a `.env` file in the project root:
   ```env
   OPENAI_API_KEY=sk-your-openai-api-key
   ALPACA_API_KEY=your-alpaca-api-key
   ALPACA_SECRET_KEY=your-alpaca-secret-key
   ```

4. **Run the application**
   ```bash
   cd code
   streamlit run app.py
   ```

5. **Open in browser**
   
   Navigate to `http://localhost:8501`

## 🌐 Deploy to Streamlit Cloud

1. **Fork this repository** to your GitHub account

2. **Go to [Streamlit Cloud](https://share.streamlit.io/)**

3. **Deploy new app**:
   - Repository: `yourusername/stock-trading-agent`
   - Branch: `main`
   - Main file path: `code/app.py`

4. **Add Secrets** in Streamlit Cloud Settings:
   ```toml
   OPENAI_API_KEY = "sk-your-openai-api-key"
   ALPACA_API_KEY = "your-alpaca-api-key"
   ALPACA_SECRET_KEY = "your-alpaca-secret-key"
   ```

5. **Click Deploy!** 🎉

## 📁 Project Structure

```
stock-trading-agent/
├── code/
│   ├── app.py              # Streamlit web application
│   └── stock_agent.py      # Multi-agent trading system
├── .streamlit/
│   ├── config.toml         # Streamlit configuration
│   └── secrets.toml.example # Template for secrets
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── .gitignore             # Git ignore patterns
└── .env.example           # Environment variables template
```

## 🔧 Configuration

### API Keys

| Key | Description | Where to get it |
|-----|-------------|-----------------|
| `OPENAI_API_KEY` | OpenAI GPT-4 access | [OpenAI Platform](https://platform.openai.com/) |
| `ALPACA_API_KEY` | Alpaca trading API | [Alpaca Markets](https://alpaca.markets/) |
| `ALPACA_SECRET_KEY` | Alpaca secret key | [Alpaca Markets](https://alpaca.markets/) |

### Streamlit Secrets

For cloud deployment, add secrets in Streamlit Cloud dashboard under **Settings > Secrets**.

## 📊 Usage

### Stock Analysis Mode

1. Enter a stock ticker (e.g., `TSLA`, `AAPL`, `NVDA`)
2. Click **"Run Analysis"**
3. Watch as each agent performs their analysis:
   - 📊 Fundamental Analyst evaluates company metrics
   - 📈 Technical Analyst computes indicators
   - 📰 News Reader assesses sentiment
   - 💼 Trade Placer makes final decision
4. Review the comprehensive analysis and trade recommendation

### Direct Trade Mode

⚠️ **Use with caution** - This executes trades immediately!

1. Enter stock ticker
2. Select action (BUY/SELL)
3. Enter quantity
4. Click **"Execute Trade"**

### Market Data View

- View candlestick charts with moving averages
- Monitor volume trends
- Track key price metrics

## 🛡️ Safety Features

- **Paper Trading Only**: Uses Alpaca paper trading API
- **Conservative Position Sizing**: 2-5% of buying power per trade
- **Analysis-First Approach**: Full analysis before trade recommendations
- **Error Handling**: Graceful handling of API failures

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Disclaimer

**This is for educational purposes only. Not financial advice.**

- This application uses paper trading (simulated money)
- Past performance does not guarantee future results
- Always do your own research before making investment decisions
- The creators are not responsible for any financial losses

## 🙏 Acknowledgments

- [LangChain](https://langchain.com/) - AI application framework
- [LangGraph](https://langchain-ai.github.io/langgraph/) - Multi-agent orchestration
- [OpenAI](https://openai.com/) - GPT-4 language model
- [Alpaca Markets](https://alpaca.markets/) - Trading API
- [Streamlit](https://streamlit.io/) - Web application framework

---

<div align="center">

**Built with ❤️ by Stock Trading AI Team**

⭐ Star this repo if you find it helpful!

</div>
