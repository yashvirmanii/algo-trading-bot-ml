"""
Statistical Arbitrage (Pairs Trading) Strategy
"""
import pandas as pd

class StatisticalArbitrageStrategy:
    def __init__(self, window=20, threshold=1.5):
        self.window = window
        self.threshold = threshold

    def generate_signals(self, df1, df2):
        spread = df1['close'] - df2['close']
        mean = spread.rolling(window=self.window).mean()
        std = spread.rolling(window=self.window).std()
        zscore = (spread - mean) / (std + 1e-9)
        df = pd.DataFrame({'spread': spread, 'zscore': zscore})
        df['signal'] = 0
        df.loc[df['zscore'] < -self.threshold, 'signal'] = 1  # Buy spread
        df.loc[df['zscore'] > self.threshold, 'signal'] = -1  # Sell spread
        return df 