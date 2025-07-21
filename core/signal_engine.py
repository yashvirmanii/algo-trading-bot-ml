"""
Signal Engine Module

Computes a composite buy/sell signal using a weighted sum of all analysis techniques (RSI, MACD, EMA, Volume, Fundamentals, Sentiment, etc.).
Each technique's signal is multiplied by its weight and summed for a final score.
"""
import numpy as np

class SignalEngine:
    def __init__(self, weights_manager):
        self.weights_manager = weights_manager

    def compute_composite_signal(self, signals_dict):
        # signals_dict: {technique: signal_value}
        weights = self.weights_manager.get_weights()
        score = 0
        for tech, signal in signals_dict.items():
            score += weights.get(tech, 0) * signal
        return score 