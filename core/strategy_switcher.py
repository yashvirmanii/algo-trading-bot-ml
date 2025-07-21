"""
Strategy Switcher Module

Dynamically switches between trading strategies (e.g., trend-following, scalping) based on:
- Time of day
- Volatility
- Recent success rate (win rate, PnL)

Exposes select_strategy() method.
"""
from datetime import datetime, time

class StrategySwitcher:
    def __init__(self, strategies, performance_log=None):
        self.strategies = strategies  # dict: {name: strategy_class}
        self.performance_log = performance_log or {}

    def select_strategy(self, now=None, volatility=None):
        # now: current datetime (for testing), defaults to now
        if now is None:
            now = datetime.now()
        current_time = now.time()
        # Example logic:
        if time(9, 30) <= current_time < time(11, 0):
            return self.strategies.get('trend_following')
        elif time(11, 0) <= current_time < time(13, 0):
            if volatility and volatility > 5:
                return self.strategies.get('scalping')
            else:
                return self.strategies.get('mean_reversion')
        else:
            # Use recent performance to pick best
            best = max(self.performance_log.items(), key=lambda x: x[1]['win_rate'], default=(None, None))[0]
            return self.strategies.get(best, self.strategies.get('trend_following')) 