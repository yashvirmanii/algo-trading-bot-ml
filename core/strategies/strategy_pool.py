"""
Strategy Pool Manager

This module manages a pool of trading strategies:
- Randomly selects a subset of strategies for each trading batch
- Assigns random weights to each selected strategy
- Combines their signals into a single composite signal
- Tracks and reports which strategies and weights were used for each batch

This approach enables dynamic, diversified, and adaptive trading, improving robustness and reducing overfitting to any single strategy.
"""
import random
import numpy as np
from core.strategies.trend_following import TrendFollowingStrategy
from core.strategies.mean_reversion import MeanReversionStrategy
from core.strategies.momentum import MomentumStrategy
from core.strategies.arbitrage import ArbitrageStrategy
from core.strategies.stat_arb import StatisticalArbitrageStrategy
from core.strategies.market_making import MarketMakingStrategy

class StrategyPoolManager:
    def __init__(self):
        self.strategy_classes = [
            TrendFollowingStrategy,
            MeanReversionStrategy,
            MomentumStrategy,
            ArbitrageStrategy,
            StatisticalArbitrageStrategy,
            MarketMakingStrategy
        ]
        self.last_batch_report = None

    def pick_strategies(self, n=None):
        n = n or random.randint(2, len(self.strategy_classes))
        selected = random.sample(self.strategy_classes, n)
        weights = np.random.dirichlet(np.ones(n), size=1)[0]
        return selected, weights

    def run_batch(self, dfs):
        # dfs: dict of {symbol: df} or list of dfs for pairs
        selected, weights = self.pick_strategies()
        signals = []
        strategy_names = []
        for i, strat_cls in enumerate(selected):
            strat = strat_cls()
            # For arbitrage/stat arb, need two dfs
            if strat_cls in [ArbitrageStrategy, StatisticalArbitrageStrategy]:
                if isinstance(dfs, list) and len(dfs) >= 2:
                    sig_df = strat.generate_signals(dfs[0], dfs[1])
                else:
                    continue
            else:
                if isinstance(dfs, dict):
                    # Use first df for demo
                    sig_df = strat.generate_signals(list(dfs.values())[0])
                else:
                    sig_df = strat.generate_signals(dfs)
            signals.append(sig_df['signal'].values)
            strategy_names.append(strat_cls.__name__)
        # Weighted sum of signals
        if signals:
            combined_signal = np.average(signals, axis=0, weights=weights)
        else:
            combined_signal = np.zeros(len(list(dfs.values())[0]))
        # Save batch report
        self.last_batch_report = {
            'strategies': strategy_names,
            'weights': weights.tolist(),
            'combined_signal': combined_signal.tolist()
        }
        return combined_signal

    def get_last_batch_report(self):
        return self.last_batch_report 