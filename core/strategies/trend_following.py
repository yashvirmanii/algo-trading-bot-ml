"""
Trend Following Strategy
"""
import pandas as pd

class TrendFollowingStrategy:
    def __init__(self, short_window=10, long_window=50):
        self.short_window = short_window
        self.long_window = long_window

    def generate_signals(self, df):
        df['short_ma'] = df['close'].rolling(window=self.short_window).mean()
        df['long_ma'] = df['close'].rolling(window=self.long_window).mean()
        df['breakout'] = (df['close'] > df['close'].rolling(window=self.long_window).max()).astype(int)
        df['signal'] = 0
        df.loc[df['short_ma'] > df['long_ma'], 'signal'] = 1
        df.loc[df['short_ma'] < df['long_ma'], 'signal'] = -1
        # If breakout, boost signal
        df.loc[df['breakout'] == 1, 'signal'] = 1
        return df 