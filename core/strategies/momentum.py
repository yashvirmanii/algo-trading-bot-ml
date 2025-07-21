"""
Momentum Strategy
"""
import pandas as pd

class MomentumStrategy:
    def __init__(self, window=14):
        self.window = window

    def generate_signals(self, df):
        df['roc'] = df['close'].pct_change(periods=self.window)
        df['signal'] = 0
        df.loc[df['roc'] > 0, 'signal'] = 1
        df.loc[df['roc'] < 0, 'signal'] = -1
        return df 