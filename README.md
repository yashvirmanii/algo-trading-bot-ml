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

## Setup
1. **Clone the repo**
2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
3. **Create a `.env` file** (see `.env.example`)
4. **Run the bot**
   ```bash
   python main.py
   ```

## Security
- Store all API keys and tokens in `.env` (never hardcode)
- Only allow authorized Telegram chat IDs
- Sensitive data is never logged

## Logging
- All modules use Python's `logging` for info, warning, error, and debug logs
- Logs are stored in `logs/` directory

## Testing
- Run all tests with:
  ```bash
  pytest
  ```
- Tests cover all core modules and strategies

## Architecture
- `core/strategies/`: All strategy modules
- `core/strategies/strategy_pool.py`: Pool manager for random strategy selection/weighting
- `core/risk.py`: Risk management
- `data/storage.py`: Trade logging and analytics (CSV)
- `notify/telegram.py`: Telegram bot and notifications
- `main.py`: Main trading loop

## Extending
- Add new strategies by creating a new module in `core/strategies/` and registering in the pool
- All parameters are configurable via `.env` and config files

## Disclaimer
This bot is for educational purposes only. Use at your own risk. Always paper trade before going live. 
