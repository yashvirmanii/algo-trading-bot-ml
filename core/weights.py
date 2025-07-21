"""
Adaptive Weights Engine

This module dynamically adjusts the weights of trading indicators/strategies based on recent trade performance:
- Analyzes trade logs to compute win rates per signal
- Increases weights for high-performing signals, decreases for underperformers
- Normalizes and returns updated weights for use in strategy selection

This enables the bot to learn and adapt to changing market conditions over time.
"""

import pandas as pd
from data.storage import TradeLogger

class AdaptiveWeights:
    def __init__(self, indicators=None, trade_logger=None):
        if indicators is None:
            indicators = {'RSI': 0.2, 'MACD': 0.3, 'ATR': 0.5}
        self.weights = indicators
        self.trade_logger = trade_logger or TradeLogger()

    def update_weights(self):
        df = self.trade_logger.load_trades()
        if df.empty:
            return self.weights
        # Calculate win rate per signal (assume signals column is a dict or str)
        for ind in self.weights:
            # Filter trades where this indicator was used (simple contains check)
            relevant = df[df['signals'].astype(str).str.contains(ind)]
            if not relevant.empty:
                win_rate = (relevant['pnl'] > 0).mean()
                if win_rate > 0.6:
                    self.weights[ind] += 0.05
                elif win_rate < 0.4:
                    self.weights[ind] -= 0.05
        # Normalize weights
        total = sum(self.weights.values())
        for k in self.weights:
            self.weights[k] = max(0, self.weights[k] / total)
        return self.weights

    def get_weights(self):
        return self.weights 