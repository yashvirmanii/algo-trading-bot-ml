"""
Reinforcement Learning Agent Module

This module provides:
- A custom trading environment for RL training (state, action, reward, step logic)
- An RL agent (DQN) using stable-baselines3 for learning optimal trading policies
- Methods to train the agent and use it for live action selection

This enables the bot to improve its trading logic through experience and feedback from real trades.
"""

import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.envs import DummyVecEnv

class TradingEnv:
    def __init__(self, data):
        self.data = data
        self.current_step = 0
        self.done = False
        self.position = 0  # 0: flat, 1: long, -1: short
        self.balance = 100000  # Example starting capital
        self.entry_price = 0

    def reset(self):
        self.current_step = 0
        self.done = False
        self.position = 0
        self.balance = 100000
        self.entry_price = 0
        return self._get_state()

    def _get_state(self):
        # Example: return last N prices and position
        state = np.array([
            self.data['close'].iloc[self.current_step],
            self.position
        ])
        return state

    def step(self, action):
        # Actions: 0 = hold, 1 = buy, 2 = sell
        reward = 0
        price = self.data['close'].iloc[self.current_step]
        if action == 1 and self.position == 0:  # Buy
            self.position = 1
            self.entry_price = price
        elif action == 2 and self.position == 1:  # Sell
            reward = price - self.entry_price
            self.balance += reward
            self.position = 0
        self.current_step += 1
        if self.current_step >= len(self.data) - 1:
            self.done = True
        return self._get_state(), reward, self.done, {}

# RL Agent wrapper
class RLTrader:
    def __init__(self, data):
        self.env = DummyVecEnv([lambda: TradingEnv(data)])
        self.model = DQN('MlpPolicy', self.env, verbose=0)

    def train(self, timesteps=10000):
        self.model.learn(total_timesteps=timesteps)

    def act(self, state):
        action, _ = self.model.predict(state, deterministic=True)
        return action 