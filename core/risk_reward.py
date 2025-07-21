"""
Risk-to-Reward Logic Module

Dynamically adjusts Stop Loss (SL) and Target (TP) for each trade based on:
- Signal quality/confidence
- ATR-based volatility
- User-defined base risk-to-reward (R:R) ratio
"""

class RiskRewardManager:
    def __init__(self, base_rr=2.0, min_rr=1.0, max_rr=4.0):
        self.base_rr = base_rr
        self.min_rr = min_rr
        self.max_rr = max_rr

    def calculate_sl_tp(self, entry, direction, atr, signal_confidence):
        # direction: 1 for long, -1 for short
        # signal_confidence: 0-1
        # ATR used for volatility-based SL/TP
        rr = self.base_rr * (0.5 + signal_confidence)  # scale R:R with confidence
        rr = min(max(rr, self.min_rr), self.max_rr)
        sl_dist = atr  # SL = 1x ATR
        tp_dist = rr * sl_dist
        if direction == 1:
            sl = entry - sl_dist
            tp = entry + tp_dist
        else:
            sl = entry + sl_dist
            tp = entry - tp_dist
        return sl, tp, rr 