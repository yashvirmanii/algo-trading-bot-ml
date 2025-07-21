"""
Volatility and Time Filter Module

Allows trades only when volatility (ATR or std dev) is within a preferred range and during allowed trading hours (9:30 AM to 2:30 PM IST).
"""
from datetime import datetime, time

class VolatilityFilter:
    def __init__(self, min_atr=1, max_atr=10, start_time=time(9, 30), end_time=time(14, 30)):
        self.min_atr = min_atr
        self.max_atr = max_atr
        self.start_time = start_time
        self.end_time = end_time

    def is_trade_allowed(self, atr, now=None):
        # atr: current ATR value
        # now: current datetime (for testing), defaults to now
        if now is None:
            now = datetime.now()
        current_time = now.time()
        in_time = self.start_time <= current_time <= self.end_time
        in_vol = self.min_atr <= atr <= self.max_atr
        return in_time and in_vol 