"""
Moving Average Crossover Strategy with Adaptive Weights
"""

import pandas as pd

class MovingAverageStrategy:
    def __init__(self, short_window=5, long_window=20, weights=None):
        self.short_window = short_window
        self.long_window = long_window
        # Default weights if not provided
        self.weights = weights or {'MA': 0.5, 'RSI': 0.2, 'MACD': 0.2, 'ATR': 0.1}

    def set_weights(self, weights):
        self.weights = weights

    def compute_rsi(self, df, window=14):
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / (loss + 1e-9)
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def compute_macd(self, df):
        ema12 = df['close'].ewm(span=12, adjust=False).mean()
        ema26 = df['close'].ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        return macd

    def compute_atr(self, df, window=14):
        high_low = df['high'] - df['low']
        high_close = (df['high'] - df['close'].shift()).abs()
        low_close = (df['low'] - df['close'].shift()).abs()
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        atr = true_range.rolling(window=window).mean()
        return atr

    def generate_signals(self, df):
        # Moving Average
        df['short_ma'] = df['close'].rolling(window=self.short_window).mean()
        df['long_ma'] = df['close'].rolling(window=self.long_window).mean()
        df['ma_signal'] = 0
        df.loc[df['short_ma'] > df['long_ma'], 'ma_signal'] = 1
        df.loc[df['short_ma'] < df['long_ma'], 'ma_signal'] = -1

        # RSI
        df['rsi'] = self.compute_rsi(df)
        df['rsi_signal'] = 0
        df.loc[df['rsi'] > 70, 'rsi_signal'] = -1
        df.loc[df['rsi'] < 30, 'rsi_signal'] = 1

        # MACD
        df['macd'] = self.compute_macd(df)
        df['macd_signal'] = 0
        df.loc[df['macd'] > 0, 'macd_signal'] = 1
        df.loc[df['macd'] < 0, 'macd_signal'] = -1

        # ATR (volatility filter, not directional)
        df['atr'] = self.compute_atr(df)
        df['atr_signal'] = 0  # Placeholder, could be used for risk sizing

        # Weighted sum of signals
        df['signal_score'] = (
            self.weights.get('MA', 0) * df['ma_signal'] +
            self.weights.get('RSI', 0) * df['rsi_signal'] +
            self.weights.get('MACD', 0) * df['macd_signal']
            # ATR not used for direction here
        )
        # Threshold: >0.5 = buy, <-0.5 = sell, else hold
        df['signal'] = 0
        df.loc[df['signal_score'] > 0.5, 'signal'] = 1
        df.loc[df['signal_score'] < -0.5, 'signal'] = -1
        return df 