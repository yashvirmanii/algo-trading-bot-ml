"""
Signal Engine Module

Computes a composite buy/sell signal using a weighted sum of all analysis techniques (RSI, MACD, EMA, Volume, Fundamentals, Sentiment, etc.).
Each technique's signal is multiplied by its weight and summed for a final score.
Sentiment is integrated with a configurable weight and can filter out stocks with strong negative sentiment.
"""
import numpy as np

class SignalEngine:
    def __init__(self, weights_manager, sentiment_weight=0.2, sentiment_threshold=-0.5):
        self.weights_manager = weights_manager
        self.sentiment_weight = sentiment_weight
        self.sentiment_threshold = sentiment_threshold

    def compute_composite_signal(self, signals_dict, sentiment_score=None):
        # signals_dict: {technique: signal_value}
        weights = self.weights_manager.get_weights()
        score = 0
        for tech, signal in signals_dict.items():
            score += weights.get(tech, 0) * signal
        # Integrate sentiment
        if sentiment_score is not None:
            score += self.sentiment_weight * sentiment_score
            # If sentiment is strongly negative, filter out
            if sentiment_score <= self.sentiment_threshold:
                score = min(score, 0)  # Do not allow buy
        return score 