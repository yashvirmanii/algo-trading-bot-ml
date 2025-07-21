"""
Mean Reversion Strategy
"""
import pandas as pd

class MeanReversionStrategy:
    def __init__(self, window=20, threshold=1.5):
        self.window = window
        self.threshold = threshold

    def generate_signals(self, df):
        df['mean'] = df['close'].rolling(window=self.window).mean()
        df['std'] = df['close'].rolling(window=self.window).std()
        df['zscore'] = (df['close'] - df['mean']) / (df['std'] + 1e-9)
        df['signal'] = 0
        df.loc[df['zscore'] < -self.threshold, 'signal'] = 1  # Buy
        df.loc[df['zscore'] > self.threshold, 'signal'] = -1  # Sell
        return df 