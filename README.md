# Advanced Indian Market Intraday Trading Bot

## Overview
A comprehensive, modular, and adaptive intraday trading bot for Indian stock markets (NSE/BSE) using Zerodha Kite Connect, AI/ML, and Telegram notifications. Supports multiple strategies, adaptive weights, RL agent, and robust error handling.

## Features
- Multi-strategy pool: Trend Following, Mean Reversion, Momentum, Arbitrage, Statistical Arbitrage, Market Making
- Adaptive weights and RL agent for self-improvement
- Real-time Telegram notifications and controls
- Trade logging, analytics, and batch reporting
- Secure config via `.env` (see `.env.example`)
- Comprehensive logging and error handling
- Unit/integration tests with pytest
- CSV-based storage (no DB required)
- **Dynamic stock universe selection** with robust intraday filters
- **Sentiment analysis overlay** using Hugging Face models and news/tweets

## Dynamic Stock Universe Filtering
The bot uses Kite Connect’s `instruments()` API to fetch all NSE equity instruments and applies industry-backed intraday filters:
- **Price**: ₹10–₹5,000 (avoids penny/illiquid stocks)
- **Volume**: Daily average > 200,000 (configurable)
- **52-week range**: Only stocks in the 30–70% range (mid-range, avoids overbought/oversold)
- **Volatility**: ATR not too low (configurable)
- **Ban/illiquid/expired**: Excludes stocks in ban list or expired

**References:**
- [Elearnmarkets: How to Filter Stocks for Intraday Trading](https://blog.elearnmarkets.com/filter-stock-for-intraday-trading/)
- [Investopedia: How to Pick Stocks for Intraday Trading](https://www.investopedia.com/day-trading/pick-stocks-intraday-trading/)
- [Groww: How to Select Stocks for Intraday](https://groww.in/blog/how-to-select-stocks-for-intraday)
- [StockPathshala: How to Select Stocks for Intraday](https://stockpathshala.com/how-to-select-stocks-for-intraday/)

## Sentiment Analysis Integration
- Uses a Hugging Face transformer model (e.g., `finiteautomata/twitter-roberta-base-sentiment-analysis`)
- Fetches news headlines (NewsAPI) or tweets (Twitter API) for each stock
- Classifies sentiment as +1 (Bullish), 0 (Neutral), -1 (Bearish)
- Sentiment score is integrated into the composite signal with a configurable weight
- Strongly negative sentiment deprioritizes or filters out stocks, even if technicals are strong

## Configuration Options
- **min_price**: Minimum stock price (default: 10)
- **max_price**: Maximum stock price (default: 5000)
- **min_volume**: Minimum daily average volume (default: 200,000)
- **min_atr**: Minimum ATR for volatility (default: 1)
- **max_atr**: Maximum ATR (default: 100)
- **min_range**: Minimum 52-week range % (default: 0.3)
- **max_range**: Maximum 52-week range % (default: 0.7)
- **ban_list**: List of banned/illiquid stocks
- **sentiment_weight**: Weight of sentiment in composite signal (default: 0.2)
- **sentiment_threshold**: Threshold for filtering out negative sentiment (default: -0.5)
- All options are configurable in the relevant module constructors or config files

## Testing
- Unit tests for stock universe filtering: `tests/test_stock_universe.py`
- Integration tests for sentiment parser: `tests/test_sentiment_parser.py`
- Run all tests with:
  ```bash
  pytest
  ```

## How Sentiment & Universe Filtering Improve Intraday Trading
- **Stock universe filtering** ensures only liquid, mid-range, and sufficiently volatile stocks are traded, reducing slippage and improving fill quality.
- **Sentiment overlay** helps avoid stocks with negative news/tweets, reducing the risk of technical “traps” and improving win rate.
- Both modules are fully integrated into the signal engine and trading loop for adaptive, data-driven decision making.

## Architecture
- `analyzers/stock_universe.py`: Dynamic universe selection and filtering
- `analyzers/sentiment_parser.py`: Sentiment scoring from news/tweets
- `core/signal_engine.py`: Composite signal calculation (with sentiment)
- `core/weights_manager.py`: Adaptive weights
- `core/capital_allocator.py`: Capital allocation
- `core/volatility_filter.py`: Volatility/time filter
- `core/risk_reward.py`: Dynamic SL/TP
- `core/strategy_switcher.py`: Strategy switching
- `core/order_executor.py`: Smart order execution
- `core/paper_trader.py`: Paper trading
- `notify/telegram.py`: Telegram bot and notifications
- `main.py`: Main trading loop

## Extending
- Add new filters or sentiment sources by editing the relevant analyzer modules
- All parameters are configurable via `.env` and config files

## Disclaimer
This bot is for educational purposes only. Use at your own risk. Always paper trade before going live. 
