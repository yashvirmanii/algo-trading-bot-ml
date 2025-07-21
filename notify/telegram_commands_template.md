# Telegram Bot Commands & Messages — Template

This document lists all the planned Telegram commands and automated messages for the intraday trading bot, along with their descriptions and expected actions.

---

## **User Commands**

| Command                | Description                                      | Action/Effect                                  |
|------------------------|--------------------------------------------------|------------------------------------------------|
| /start                 | Start the bot and receive a welcome message      | Bot sends intro, status, and help info         |
| /stop                  | Stop all trading activity (kill switch)          | Bot halts trading, confirms shutdown           |
| /status                | Get current bot status and portfolio snapshot    | Bot replies with PnL, open trades, health      |
| /mode [live/paper]     | Switch between live and paper trading            | Bot toggles mode, confirms change              |
| /pnl                   | Get current day's PnL and trade summary          | Bot sends PnL, win rate, trade count           |
| /universe              | Show current stock universe being traded         | Bot lists all stocks in today’s universe       |
| /weights               | Show current strategy/indicator weights          | Bot sends table of weights                     |
| /sentiment             | Show current sentiment scores for universe       | Bot sends sentiment summary                    |
| /help                  | List all available commands                      | Bot sends this help message                    |

---

## **Automated Messages**

| Message Type           | Description                                      | Example Content                                |
|------------------------|--------------------------------------------------|------------------------------------------------|
| Trade Alert            | New trade executed (entry/exit)                  | "Bought RELIANCE at 2450.0, SL: 2420, TP: 2500"|
| Capital Update         | Hourly update on capital usage and PnL           | "Capital used: 65%, PnL: +₹2,300"             |
| Strategy Update        | Change in strategy weights or active strategy     | "Switched to mean reversion, weights updated"  |
| Sentiment Update       | Change in sentiment for a stock/universe         | "Sentiment for TCS: Bullish"                   |
| Risk Alert             | Breach of risk rule (drawdown, max trades, etc.) | "Max drawdown hit, trading paused for today"   |
| Kill Switch Activated  | Bot stopped by user or emergency                 | "Trading halted by user command"               |
| Portfolio Snapshot     | End-of-day or on-demand portfolio summary        | "EOD: 5 trades, PnL: +₹1,800, Win rate: 60%"   |

---

## **Notes**
- All commands should be sent as messages to the Telegram bot.
- Only authorized users (by chat ID) can control the bot.
- Automated messages are sent proactively by the bot based on events. 