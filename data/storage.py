"""
Trade Logging and Analytics Module

This module provides:
- Persistent logging of every trade (entry, exit, PnL, signals, strategy, etc.) to CSV
- Methods to load and analyze trade history (win rate, average PnL, total trades)
- Foundation for performance analytics and adaptive learning

All trade data is stored in CSV for easy access and future migration to a database.
"""

import pandas as pd
import os
from datetime import datetime

class TradeLogger:
    def __init__(self, log_file='trade_log.csv'):
        self.log_file = log_file
        self.columns = [
            'timestamp', 'symbol', 'side', 'qty', 'entry_price', 'exit_price', 'pnl',
            'signals', 'strategy', 'outcome', 'notes'
        ]
        if not os.path.exists(self.log_file):
            pd.DataFrame(columns=self.columns).to_csv(self.log_file, index=False)

    def log_trade(self, symbol, side, qty, entry_price, exit_price, pnl, signals, strategy, outcome, notes=''):
        row = {
            'timestamp': datetime.now().isoformat(),
            'symbol': symbol,
            'side': side,
            'qty': qty,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'pnl': pnl,
            'signals': signals,
            'strategy': strategy,
            'outcome': outcome,
            'notes': notes
        }
        df = pd.DataFrame([row])
        df.to_csv(self.log_file, mode='a', header=False, index=False)

    def load_trades(self):
        return pd.read_csv(self.log_file)

    def analyze_performance(self):
        df = self.load_trades()
        if df.empty:
            return {}
        win_rate = (df['pnl'] > 0).mean()
        avg_pnl = df['pnl'].mean()
        total_trades = len(df)
        return {
            'win_rate': win_rate,
            'avg_pnl': avg_pnl,
            'total_trades': total_trades
        } 