"""
Capital Allocator Module

Dynamically allocates trade capital based on signal score strength and daily capital limits.
Example: if signal confidence is 80%, invest 80% of available capital.
"""

class CapitalAllocator:
    def __init__(self, total_capital, max_daily_usage=1.0):
        self.total_capital = total_capital
        self.max_daily_usage = max_daily_usage  # e.g. 0.8 for 80%
        self.used_capital = 0

    def allocate(self, signal_score):
        # signal_score: 0-1 (confidence)
        available = self.total_capital * self.max_daily_usage - self.used_capital
        allocation = min(signal_score * self.total_capital, available)
        self.used_capital += allocation
        return allocation

    def reset(self):
        self.used_capital = 0 