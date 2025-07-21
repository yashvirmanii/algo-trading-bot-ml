"""
Stock Screener Module

This module is responsible for:
- Fetching the stock universe (e.g., NIFTY 100) using Zerodha Kite Connect API
- Downloading historical OHLCV data for each stock
- Calculating technical features (e.g., ATR, volume)
- Filtering stocks based on configurable criteria (price, volume, volatility)
- Returning a DataFrame of tradable stocks for the day

All data is sourced live from Zerodha, ensuring up-to-date screening for intraday trading.
"""

import pandas as pd
from broker.zerodha import ZerodhaBroker

class StockScreener:
    def __init__(self):
        self.broker = ZerodhaBroker()

    def fetch_kite_data(self):
        # Get stock universe (e.g., NIFTY 100)
        symbols = self.broker.get_stock_universe()
        stocks_data = []
        for symbol in symbols:
            try:
                hist = self.broker.get_historical_data(symbol)
                if not hist.empty:
                    last_close = hist['close'].iloc[-1]
                    volume = hist['volume'].iloc[-1]
                    atr = (hist['high'] - hist['low']).rolling(window=5).mean().iloc[-1]
                    stocks_data.append({
                        'symbol': symbol,
                        'close': last_close,
                        'volume': volume,
                        'atr': atr,
                        'data': hist
                    })
            except Exception as e:
                continue
        return pd.DataFrame(stocks_data)

    def apply_filters(self, df):
        # Filter: price < 500, high volume, ATR (volatility) > threshold
        filtered = df[(df['close'] < 500) & (df['volume'] > 100000) & (df['atr'] > 1)]
        return filtered

    def get_tradable_stocks(self):
        df = self.fetch_kite_data()
        filtered = self.apply_filters(df)
        return filtered 