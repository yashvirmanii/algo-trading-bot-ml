import pytest
import pandas as pd
from analyzers.stock_universe import StockUniverse

class MockBroker:
    def __init__(self, hist_atr=2):
        self.kite = self
        self.hist_atr = hist_atr
    def instruments(self, seg):
        return [
            {'tradingsymbol': 'ABC', 'segment': 'NSE', 'instrument_type': 'EQ', 'expiry': None, 'last_price': 100, 'volume': 300000, 'low_52_week': 50, 'high_52_week': 200},
            {'tradingsymbol': 'XYZ', 'segment': 'NSE', 'instrument_type': 'EQ', 'expiry': None, 'last_price': 5, 'volume': 100000, 'low_52_week': 2, 'high_52_week': 10},
            {'tradingsymbol': 'DEF', 'segment': 'NSE', 'instrument_type': 'EQ', 'expiry': None, 'last_price': 1000, 'volume': 500000, 'low_52_week': 800, 'high_52_week': 1200},
        ]
    def get_historical_data(self, symbol, interval, days):
        df = pd.DataFrame({'high': [10]*15, 'low': [8]*15})
        return df

def test_stock_universe_filters():
    broker = MockBroker()
    su = StockUniverse(broker, min_price=10, max_price=5000, min_volume=200_000, min_atr=1, max_atr=3, min_range=0.2, max_range=0.8)
    universe = su.get_universe()
    assert 'ABC' in universe
    assert 'XYZ' not in universe  # price/volume too low
    assert 'DEF' in universe
    su = StockUniverse(broker, ban_list=['DEF'])
    universe = su.get_universe()
    assert 'DEF' not in universe 