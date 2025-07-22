"""
Stock Universe Analyzer

Fetches all NSE equity instruments from Kite Connect and applies intraday trading filters:
- Price between ₹10 and ₹1,000
- Daily average volume > 200k (configurable)
- 52-week high/low range within 30–70% (mid-range)
- ATR-based volatility not too low
- Excludes penny/illiquid/ban/expired stocks

References:
- blog.elearnmarkets.com/filter-stock-for-intraday-trading/
- investopedia.com/day-trading/pick-stocks-intraday-trading/
- groww.in/blog/how-to-select-stocks-for-intraday
- stockpathshala.com/how-to-select-stocks-for-intraday/
"""
import pandas as pd
from datetime import datetime, timedelta

class StockUniverse:
    def __init__(self, broker, min_price=10, max_price=1000, min_volume=200_000, min_atr=1, max_atr=100, min_range=0.3, max_range=0.7, ban_list=None):
        self.broker = broker
        self.min_price = min_price
        self.max_price = max_price
        self.min_volume = min_volume
        self.min_atr = min_atr
        self.max_atr = max_atr
        self.min_range = min_range
        self.max_range = max_range
        self.ban_list = set(ban_list) if ban_list else set()

    def fetch_instruments(self):
        # Get all NSE equity instruments
        df = pd.DataFrame(self.broker.kite.instruments('NSE'))
        df = df[df['segment'] == 'NSE']
        df = df[df['instrument_type'] == 'EQ']
        df = df[df['expiry'].isnull()]
        return df

    def filter(self, df):
        # Price filter
        df = df[(df['last_price'] >= self.min_price) & (df['last_price'] <= self.max_price)]
        # Volume filter
        df = df[df['volume'] >= self.min_volume]
        # 52-week range filter
        range_pct = (df['last_price'] - df['low_52_week']) / (df['high_52_week'] - df['low_52_week'] + 1e-9)
        df = df[(range_pct >= self.min_range) & (range_pct <= self.max_range)]
        # Ban/illiquid filter
        df = df[~df['tradingsymbol'].isin(self.ban_list)]
        # ATR filter (fetch recent OHLCV for ATR calculation)
        keep = []
        for _, row in df.iterrows():
            try:
                hist = self.broker.get_historical_data(row['tradingsymbol'], interval='15minute', days=5)
                atr = (hist['high'] - hist['low']).rolling(window=14).mean().iloc[-1]
                if self.min_atr <= atr <= self.max_atr:
                    keep.append(True)
                else:
                    keep.append(False)
            except Exception:
                keep.append(False)
        df = df[keep]
        return df

    def get_universe(self):
        df = self.fetch_instruments()
        filtered = self.filter(df)
        return filtered['tradingsymbol'].tolist() 