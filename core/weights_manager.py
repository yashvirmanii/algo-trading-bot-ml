"""
Weights Manager Module

Manages and adaptively updates weights for each analysis technique (RSI, MACD, EMA, Volume, Fundamentals, Sentiment, etc.).
Weights are updated every 3–4 hours based on win rate, PnL, and accuracy.
Logs and sends weight update summaries to Telegram.
"""
import numpy as np
import pandas as pd
import logging

class WeightsManager:
    def __init__(self, techniques=None, notifier=None):
        if techniques is None:
            techniques = ['RSI', 'MACD', 'EMA', 'Volume', 'Fundamentals', 'Sentiment']
        self.weights = {tech: 1/len(techniques) for tech in techniques}
        self.notifier = notifier
        self.history = []

    def update_weights(self, trade_log_df):
        # Example: update weights based on win rate per technique
        for tech in self.weights:
            relevant = trade_log_df[trade_log_df['signals'].astype(str).str.contains(tech)]
            win_rate = (relevant['pnl'] > 0).mean() if not relevant.empty else 0.5
            # Simple rule: increase weight if win_rate > 0.6, decrease if < 0.4
            if win_rate > 0.6:
                self.weights[tech] += 0.05
            elif win_rate < 0.4:
                self.weights[tech] -= 0.05
        # Normalize
        total = sum(self.weights.values())
        for k in self.weights:
            self.weights[k] = max(0, self.weights[k] / total)
        self.history.append(self.weights.copy())
        self.log_weights()
        return self.weights

    def log_weights(self):
        msg = f"[Weights Update] {self.weights}"
        logging.info(msg)
        if self.notifier:
            self.notifier.send_message(msg)

    def get_weights(self):
        return self.weights 