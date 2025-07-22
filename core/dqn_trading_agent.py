"""
Deep Q-Network Trading Agent for Reinforcement Learning-based Trading.

This module implements a sophisticated DQN trading agent that:
- State space: market features, portfolio state, strategy performance
- Action space: buy, sell, hold, adjust_position_size
- Reward function: risk-adjusted returns with drawdown penalties
- Uses experience replay buffer and target network updates
- Implements epsilon-greedy exploration with decay
- Includes prioritized experience replay and tensorboard logging
"""

import logging
import os
import json
import pickle
import random
import math
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Any, Union
from collections import deque, namedtuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logger = logging.getLogger(__name__)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")

# Experience tuple for replay buffer
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])


@dataclass
class TradingState:
    """Complete trading state representation."""
    # Market features (30 dimensions)
    price_momentum_1m: float
    price_momentum_5m: float
    price_momentum_15m: float
    price_momentum_30m: float
    price_volatility: float
    volume_ratio: float
    volume_trend: float
    volume_momentum: float
    rsi: float
    macd: float
    macd_signal: float
    bb_position: float
    bb_width: float
    sma_20: float
    ema_12: float
    ema_26: float
    atr: float
    stoch_k: float
    stoch_d: float
    williams_r: float
    cci: float
    adx: float
    trend_strength: float
    support_resistance_distance: float
    breakout_probability: float
    mean_reversion_signal: float
    market_regime_trending: float
    market_regime_sideways: float
    market_regime_volatile: float
    news_sentiment: float
    
    # Portfolio state (10 dimensions)
    cash_ratio: float
    position_size: float
    unrealized_pnl: float
    realized_pnl: float
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    avg_trade_duration: float
    trades_count: float
    
    # Strategy performance (10 dimensions)
    momentum_performance: float
    mean_reversion_performance: float
    breakout_performance: float
    trend_following_performance: float
    scalping_performance: float
    momentum_confidence: float
    mean_reversion_confidence: float
    breakout_confidence: float
    trend_following_confidence: float
    scalping_confidence: float
    
    def to_tensor(self) -> torch.Tensor:
        """Convert state to tensor for neural network input."""
        return torch.tensor([
            # Market features
            self.price_momentum_1m, self.price_momentum_5m, self.price_momentum_15m, self.price_momentum_30m,
            self.price_volatility, self.volume_ratio, self.volume_trend, self.volume_momentum,
            self.rsi, self.macd, self.macd_signal, self.bb_position, self.bb_width,
            self.sma_20, self.ema_12, self.ema_26, self.atr, self.stoch_k, self.stoch_d,
            self.williams_r, self.cci, self.adx, self.trend_strength, self.support_resistance_distance,
            self.breakout_probability, self.mean_reversion_signal, self.market_regime_trending,
            self.market_regime_sideways, self.market_regime_volatile, self.news_sentiment,
            
            # Portfolio state
            self.cash_ratio, self.position_size, self.unrealized_pnl, self.realized_pnl,
            self.total_return, self.sharpe_ratio, self.max_drawdown, self.win_rate,
            self.avg_trade_duration, self.trades_count,
            
            # Strategy performance
            self.momentum_performance, self.mean_reversion_performance, self.breakout_performance,
            self.trend_following_performance, self.scalping_performance, self.momentum_confidence,
            self.mean_reversion_confidence, self.breakout_confidence, self.trend_following_confidence,
            self.scalping_confidence
        ], dtype=torch.float32)
    
    @classmethod
    def get_feature_names(cls) -> List[str]:
        """Get feature names for interpretability."""
        return [
            # Market features
            'price_momentum_1m', 'price_momentum_5m', 'price_momentum_15m', 'price_momentum_30m',
            'price_volatility', 'volume_ratio', 'volume_trend', 'volume_momentum',
            'rsi', 'macd', 'macd_signal', 'bb_position', 'bb_width',
            'sma_20', 'ema_12', 'ema_26', 'atr', 'stoch_k', 'stoch_d',
            'williams_r', 'cci', 'adx', 'trend_strength', 'support_resistance_distance',
            'breakout_probability', 'mean_reversion_signal', 'market_regime_trending',
            'market_regime_sideways', 'market_regime_volatile', 'news_sentiment',
            
            # Portfolio state
            'cash_ratio', 'position_size', 'unrealized_pnl', 'realized_pnl',
            'total_return', 'sharpe_ratio', 'max_drawdown', 'win_rate',
            'avg_trade_duration', 'trades_count',
            
            # Strategy performance
            'momentum_performance', 'mean_reversion_performance', 'breakout_performance',
            'trend_following_performance', 'scalping_performance', 'momentum_confidence',
            'mean_reversion_confidence', 'breakout_confidence', 'trend_following_confidence',
            'scalping_confidence'
        ]


class TradingAction:
    """Trading action definitions."""
    BUY = 0
    SELL = 1
    HOLD = 2
    ADJUST_POSITION_SIZE = 3
    
    @classmethod
    def get_action_names(cls) -> List[str]:
        return ['BUY', 'SELL', 'HOLD', 'ADJUST_POSITION_SIZE']


@dataclass
class TradingReward:
    """Trading reward calculation result."""
    total_reward: float
    return_component: float
    risk_component: float
    drawdown_penalty: float
    transaction_cost: float
    
    def to_dict(self) -> Dict[str, float]:
        return asdict(self)c
lass DQNNetwork(nn.Module):
    """Deep Q-Network for trading decisions."""
    
    def __init__(self, state_dim: int = 50, action_dim: int = 4, hidden_dims: List[int] = [256, 128, 64]):
        """
        Initialize DQN network.
        
        Args:
            state_dim: State space dimension
            action_dim: Action space dimension
            hidden_dims: Hidden layer dimensions
        """
        super(DQNNetwork, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Input layer
        layers = []
        prev_dim = state_dim
        
        # Hidden layers
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.BatchNorm1d(hidden_dim)
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, action_dim))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            state: State tensor (batch_size, state_dim)
            
        Returns:
            Q-values for each action (batch_size, action_dim)
        """
        return self.network(state)


class PrioritizedReplayBuffer:
    """Prioritized experience replay buffer."""
    
    def __init__(self, capacity: int = 100000, alpha: float = 0.6, beta: float = 0.4):
        """
        Initialize prioritized replay buffer.
        
        Args:
            capacity: Maximum buffer size
            alpha: Prioritization exponent
            beta: Importance sampling exponent
        """
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = 0.001
        
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.position = 0
        self.size = 0
    
    def add(self, experience: Experience, priority: Optional[float] = None):
        """Add experience to buffer."""
        if priority is None:
            priority = max(self.priorities) if self.size > 0 else 1.0
        
        if self.size < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience
        
        self.priorities[self.position] = priority
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size: int) -> Tuple[List[Experience], np.ndarray, np.ndarray]:
        """Sample batch with prioritized sampling."""
        if self.size == 0:
            return [], np.array([]), np.array([])
        
        # Calculate sampling probabilities
        priorities = self.priorities[:self.size]
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()
        
        # Sample indices
        indices = np.random.choice(self.size, batch_size, p=probabilities, replace=False)
        
        # Calculate importance sampling weights
        weights = (self.size * probabilities[indices]) ** (-self.beta)
        weights /= weights.max()
        
        # Get experiences
        experiences = [self.buffer[idx] for idx in indices]
        
        # Update beta
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        return experiences, indices, weights
    
    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        """Update priorities for given indices."""
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
    
    def __len__(self) -> int:
        return self.size


class RewardCalculator:
    """Calculate trading rewards with risk adjustment."""
    
    def __init__(self, 
                 return_weight: float = 0.6,
                 risk_weight: float = 0.2,
                 drawdown_weight: float = 0.15,
                 transaction_cost_weight: float = 0.05):
        """
        Initialize reward calculator.
        
        Args:
            return_weight: Weight for return component
            risk_weight: Weight for risk component
            drawdown_weight: Weight for drawdown penalty
            transaction_cost_weight: Weight for transaction costs
        """
        self.return_weight = return_weight
        self.risk_weight = risk_weight
        self.drawdown_weight = drawdown_weight
        self.transaction_cost_weight = transaction_cost_weight
        
        # Risk-free rate (annualized)
        self.risk_free_rate = 0.02
        
        # Transaction cost (basis points)
        self.transaction_cost_bps = 5  # 0.05%
    
    def calculate_reward(self, 
                        action: int,
                        current_return: float,
                        portfolio_value: float,
                        previous_portfolio_value: float,
                        volatility: float,
                        max_drawdown: float,
                        position_size: float) -> TradingReward:
        """
        Calculate comprehensive trading reward.
        
        Args:
            action: Trading action taken
            current_return: Current period return
            portfolio_value: Current portfolio value
            previous_portfolio_value: Previous portfolio value
            volatility: Portfolio volatility
            max_drawdown: Maximum drawdown
            position_size: Current position size
            
        Returns:
            TradingReward object with detailed breakdown
        """
        try:
            # Return component
            period_return = (portfolio_value - previous_portfolio_value) / previous_portfolio_value
            return_component = period_return * self.return_weight
            
            # Risk component (Sharpe ratio-like)
            if volatility > 0:
                risk_adjusted_return = (period_return - self.risk_free_rate / 252) / volatility
                risk_component = risk_adjusted_return * self.risk_weight
            else:
                risk_component = 0.0
            
            # Drawdown penalty
            drawdown_penalty = -abs(max_drawdown) * self.drawdown_weight
            
            # Transaction cost
            transaction_cost = 0.0
            if action in [TradingAction.BUY, TradingAction.SELL]:
                transaction_cost = -abs(position_size) * self.transaction_cost_bps / 10000 * self.transaction_cost_weight
            
            # Total reward
            total_reward = return_component + risk_component + drawdown_penalty + transaction_cost
            
            return TradingReward(
                total_reward=total_reward,
                return_component=return_component,
                risk_component=risk_component,
                drawdown_penalty=drawdown_penalty,
                transaction_cost=transaction_cost
            )
            
        except Exception as e:
            logger.error(f"Error calculating reward: {e}")
            return TradingReward(
                total_reward=0.0,
                return_component=0.0,
                risk_component=0.0,
                drawdown_penalty=0.0,
                transaction_cost=0.0
            )


class DQNTradingAgent:
    """
    Deep Q-Network Trading Agent for reinforcement learning-based trading.
    
    Implements DQN with experience replay, target networks, and prioritized sampling
    for intelligent trading decisions.
    """
    
    def __init__(self,
                 state_dim: int = 50,
                 action_dim: int = 4,
                 hidden_dims: List[int] = [256, 128, 64],
                 learning_rate: float = 0.001,
                 gamma: float = 0.99,
                 epsilon_start: float = 1.0,
                 epsilon_end: float = 0.01,
                 epsilon_decay: int = 10000,
                 batch_size: int = 32,
                 buffer_size: int = 100000,
                 target_update_freq: int = 1000,
                 model_dir: str = "models",
                 log_dir: str = "logs"):
        """
        Initialize DQN Trading Agent.
        
        Args:
            state_dim: State space dimension
            action_dim: Action space dimension
            hidden_dims: Hidden layer dimensions
            learning_rate: Learning rate for optimizer
            gamma: Discount factor
            epsilon_start: Initial exploration rate
            epsilon_end: Final exploration rate
            epsilon_decay: Epsilon decay steps
            batch_size: Training batch size
            buffer_size: Experience replay buffer size
            target_update_freq: Target network update frequency
            model_dir: Directory for model storage
            log_dir: Directory for tensorboard logs
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.model_dir = model_dir
        self.log_dir = log_dir
        
        # Create directories
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        
        # Initialize networks
        self.q_network = DQNNetwork(state_dim, action_dim, hidden_dims).to(device)
        self.target_network = DQNNetwork(state_dim, action_dim, hidden_dims).to(device)
        
        # Copy weights to target network
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Experience replay buffer
        self.replay_buffer = PrioritizedReplayBuffer(buffer_size)
        
        # Reward calculator
        self.reward_calculator = RewardCalculator()
        
        # Training state
        self.steps_done = 0
        self.episode_count = 0
        self.training_losses = []
        self.episode_rewards = []
        self.episode_returns = []
        
        # Tensorboard writer
        self.writer = SummaryWriter(log_dir)
        
        # Performance tracking
        self.portfolio_history = []
        self.action_history = []
        self.reward_history = []
        
        logger.info(f"DQNTradingAgent initialized with {state_dim}D state, {action_dim} actions")
    
    def get_epsilon(self) -> float:
        """Get current epsilon value for exploration."""
        return self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
               math.exp(-1. * self.steps_done / self.epsilon_decay)
    
    def select_action(self, state: TradingState, training: bool = True) -> int:
        """
        Select action using epsilon-greedy policy.
        
        Args:
            state: Current trading state
            training: Whether in training mode
            
        Returns:
            Selected action index
        """
        try:
            if training and random.random() < self.get_epsilon():
                # Random action (exploration)
                action = random.randrange(self.action_dim)
                logger.debug(f"Random action selected: {TradingAction.get_action_names()[action]}")
            else:
                # Greedy action (exploitation)
                with torch.no_grad():
                    state_tensor = state.to_tensor().unsqueeze(0).to(device)
                    q_values = self.q_network(state_tensor)
                    action = q_values.max(1)[1].item()
                    logger.debug(f"Greedy action selected: {TradingAction.get_action_names()[action]} (Q-value: {q_values.max().item():.4f})")
            
            self.steps_done += 1
            return action
            
        except Exception as e:
            logger.error(f"Error selecting action: {e}")
            return TradingAction.HOLD  # Default to hold on error
    
    def store_experience(self, 
                        state: TradingState,
                        action: int,
                        reward: float,
                        next_state: TradingState,
                        done: bool):
        """Store experience in replay buffer."""
        try:
            experience = Experience(
                state=state.to_tensor(),
                action=action,
                reward=reward,
                next_state=next_state.to_tensor(),
                done=done
            )
            
            self.replay_buffer.add(experience)
            
            # Store for tracking
            self.action_history.append(action)
            self.reward_history.append(reward)
            
            logger.debug(f"Stored experience: action={TradingAction.get_action_names()[action]}, reward={reward:.4f}")
            
        except Exception as e:
            logger.error(f"Error storing experience: {e}")
    
    def train_step(self) -> Optional[float]:
        """Perform one training step."""
        try:
            if len(self.replay_buffer) < self.batch_size:
                return None
            
            # Sample batch from replay buffer
            experiences, indices, weights = self.replay_buffer.sample(self.batch_size)
            
            if not experiences:
                return None
            
            # Prepare batch tensors
            states = torch.stack([exp.state for exp in experiences]).to(device)
            actions = torch.tensor([exp.action for exp in experiences], dtype=torch.long).to(device)
            rewards = torch.tensor([exp.reward for exp in experiences], dtype=torch.float32).to(device)
            next_states = torch.stack([exp.next_state for exp in experiences]).to(device)
            dones = torch.tensor([exp.done for exp in experiences], dtype=torch.bool).to(device)
            weights_tensor = torch.tensor(weights, dtype=torch.float32).to(device)
            
            # Current Q-values
            current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
            
            # Next Q-values from target network
            with torch.no_grad():
                next_q_values = self.target_network(next_states).max(1)[0]
                target_q_values = rewards + (self.gamma * next_q_values * ~dones)
            
            # Calculate loss with importance sampling weights
            td_errors = target_q_values.unsqueeze(1) - current_q_values
            loss = (weights_tensor.unsqueeze(1) * td_errors.pow(2)).mean()
            
            # Optimize
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Update priorities in replay buffer
            priorities = td_errors.abs().detach().cpu().numpy().flatten() + 1e-6
            self.replay_buffer.update_priorities(indices, priorities)
            
            # Update target network
            if self.steps_done % self.target_update_freq == 0:
                self.target_network.load_state_dict(self.q_network.state_dict())
                logger.info("Target network updated")
            
            # Log training metrics
            self.training_losses.append(loss.item())
            self.writer.add_scalar('Training/Loss', loss.item(), self.steps_done)
            self.writer.add_scalar('Training/Epsilon', self.get_epsilon(), self.steps_done)
            self.writer.add_scalar('Training/Buffer_Size', len(self.replay_buffer), self.steps_done)
            
            return loss.item()
            
        except Exception as e:
            logger.error(f"Error in training step: {e}")
            return None    d
ef train_episode(self, 
                     states: List[TradingState],
                     actions: List[int],
                     rewards: List[float],
                     portfolio_values: List[float]) -> Dict[str, float]:
        """
        Train on a complete episode.
        
        Args:
            states: List of states in episode
            actions: List of actions taken
            rewards: List of rewards received
            portfolio_values: List of portfolio values
            
        Returns:
            Episode training metrics
        """
        try:
            episode_loss = 0.0
            training_steps = 0
            
            # Store experiences and train
            for i in range(len(states) - 1):
                # Store experience
                done = (i == len(states) - 2)  # Last transition
                self.store_experience(states[i], actions[i], rewards[i], states[i + 1], done)
                
                # Train step
                loss = self.train_step()
                if loss is not None:
                    episode_loss += loss
                    training_steps += 1
            
            # Calculate episode metrics
            total_reward = sum(rewards)
            total_return = (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0]
            
            # Store episode metrics
            self.episode_rewards.append(total_reward)
            self.episode_returns.append(total_return)
            self.episode_count += 1
            
            # Log episode metrics
            avg_loss = episode_loss / training_steps if training_steps > 0 else 0.0
            
            self.writer.add_scalar('Episode/Total_Reward', total_reward, self.episode_count)
            self.writer.add_scalar('Episode/Total_Return', total_return, self.episode_count)
            self.writer.add_scalar('Episode/Average_Loss', avg_loss, self.episode_count)
            self.writer.add_scalar('Episode/Portfolio_Value', portfolio_values[-1], self.episode_count)
            
            # Action distribution
            action_counts = np.bincount(actions, minlength=self.action_dim)
            for i, count in enumerate(action_counts):
                self.writer.add_scalar(f'Actions/{TradingAction.get_action_names()[i]}', 
                                     count / len(actions), self.episode_count)
            
            logger.info(f"Episode {self.episode_count}: Reward={total_reward:.4f}, "
                       f"Return={total_return:.4f}, Loss={avg_loss:.4f}")
            
            return {
                'episode': self.episode_count,
                'total_reward': total_reward,
                'total_return': total_return,
                'average_loss': avg_loss,
                'training_steps': training_steps,
                'epsilon': self.get_epsilon()
            }
            
        except Exception as e:
            logger.error(f"Error training episode: {e}")
            return {}
    
    def evaluate_policy(self, 
                       states: List[TradingState],
                       true_actions: List[int],
                       true_rewards: List[float]) -> Dict[str, float]:
        """
        Evaluate current policy on given states.
        
        Args:
            states: List of states to evaluate
            true_actions: True actions taken
            true_rewards: True rewards received
            
        Returns:
            Evaluation metrics
        """
        try:
            predicted_actions = []
            q_values_list = []
            
            # Get policy predictions
            with torch.no_grad():
                for state in states:
                    state_tensor = state.to_tensor().unsqueeze(0).to(device)
                    q_values = self.q_network(state_tensor)
                    action = q_values.max(1)[1].item()
                    
                    predicted_actions.append(action)
                    q_values_list.append(q_values.squeeze().cpu().numpy())
            
            # Calculate metrics
            action_accuracy = np.mean(np.array(predicted_actions) == np.array(true_actions))
            
            # Q-value statistics
            q_values_array = np.array(q_values_list)
            avg_q_value = np.mean(q_values_array)
            q_value_std = np.std(q_values_array)
            
            # Action distribution comparison
            true_action_dist = np.bincount(true_actions, minlength=self.action_dim) / len(true_actions)
            pred_action_dist = np.bincount(predicted_actions, minlength=self.action_dim) / len(predicted_actions)
            
            # KL divergence between action distributions
            kl_divergence = np.sum(true_action_dist * np.log(true_action_dist / (pred_action_dist + 1e-8) + 1e-8))
            
            return {
                'action_accuracy': action_accuracy,
                'avg_q_value': avg_q_value,
                'q_value_std': q_value_std,
                'kl_divergence': kl_divergence,
                'true_action_dist': true_action_dist.tolist(),
                'pred_action_dist': pred_action_dist.tolist()
            }
            
        except Exception as e:
            logger.error(f"Error evaluating policy: {e}")
            return {}
    
    def get_q_values(self, state: TradingState) -> Dict[str, float]:
        """Get Q-values for all actions given a state."""
        try:
            with torch.no_grad():
                state_tensor = state.to_tensor().unsqueeze(0).to(device)
                q_values = self.q_network(state_tensor).squeeze().cpu().numpy()
                
                return {
                    action_name: float(q_value)
                    for action_name, q_value in zip(TradingAction.get_action_names(), q_values)
                }
                
        except Exception as e:
            logger.error(f"Error getting Q-values: {e}")
            return {}
    
    def save_model(self, filename: Optional[str] = None) -> bool:
        """Save the trained model."""
        try:
            if filename is None:
                filename = f"dqn_trading_agent_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth"
            
            filepath = os.path.join(self.model_dir, filename)
            
            # Save model state
            torch.save({
                'q_network_state_dict': self.q_network.state_dict(),
                'target_network_state_dict': self.target_network.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'steps_done': self.steps_done,
                'episode_count': self.episode_count,
                'training_losses': self.training_losses,
                'episode_rewards': self.episode_rewards,
                'episode_returns': self.episode_returns,
                'model_config': {
                    'state_dim': self.state_dim,
                    'action_dim': self.action_dim,
                    'learning_rate': self.learning_rate,
                    'gamma': self.gamma,
                    'epsilon_start': self.epsilon_start,
                    'epsilon_end': self.epsilon_end,
                    'epsilon_decay': self.epsilon_decay
                }
            }, filepath)
            
            logger.info(f"Model saved to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            return False
    
    def load_model(self, filepath: str) -> bool:
        """Load a trained model."""
        try:
            if not os.path.exists(filepath):
                logger.error(f"Model file not found: {filepath}")
                return False
            
            checkpoint = torch.load(filepath, map_location=device)
            
            # Load model states
            self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
            self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # Load training state
            self.steps_done = checkpoint.get('steps_done', 0)
            self.episode_count = checkpoint.get('episode_count', 0)
            self.training_losses = checkpoint.get('training_losses', [])
            self.episode_rewards = checkpoint.get('episode_rewards', [])
            self.episode_returns = checkpoint.get('episode_returns', [])
            
            logger.info(f"Model loaded from {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
    
    def plot_training_progress(self, save_path: Optional[str] = None) -> str:
        """Plot training progress metrics."""
        try:
            if not self.episode_rewards:
                logger.warning("No training data to plot")
                return ""
            
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
            
            episodes = range(1, len(self.episode_rewards) + 1)
            
            # Episode rewards
            ax1.plot(episodes, self.episode_rewards, 'b-', alpha=0.7)
            ax1.set_xlabel('Episode')
            ax1.set_ylabel('Total Reward')
            ax1.set_title('Episode Rewards')
            ax1.grid(True, alpha=0.3)
            
            # Episode returns
            ax2.plot(episodes, self.episode_returns, 'g-', alpha=0.7)
            ax2.set_xlabel('Episode')
            ax2.set_ylabel('Total Return')
            ax2.set_title('Episode Returns')
            ax2.grid(True, alpha=0.3)
            
            # Training losses (if available)
            if self.training_losses:
                loss_steps = range(1, len(self.training_losses) + 1)
                ax3.plot(loss_steps, self.training_losses, 'r-', alpha=0.7)
                ax3.set_xlabel('Training Step')
                ax3.set_ylabel('Loss')
                ax3.set_title('Training Loss')
                ax3.grid(True, alpha=0.3)
            
            # Epsilon decay
            epsilon_values = [self.epsilon_end + (self.epsilon_start - self.epsilon_end) * 
                            math.exp(-1. * step / self.epsilon_decay) 
                            for step in range(0, self.steps_done, max(1, self.steps_done // 100))]
            epsilon_steps = range(0, self.steps_done, max(1, self.steps_done // 100))
            
            ax4.plot(epsilon_steps, epsilon_values, 'purple', alpha=0.7)
            ax4.set_xlabel('Steps')
            ax4.set_ylabel('Epsilon')
            ax4.set_title('Exploration Rate Decay')
            ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save plot
            if save_path is None:
                save_path = os.path.join(self.model_dir, "training_progress.png")
            
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Training progress plot saved to {save_path}")
            return save_path
            
        except Exception as e:
            logger.error(f"Error plotting training progress: {e}")
            return ""
    
    def get_model_summary(self) -> Dict[str, Any]:
        """Get comprehensive model summary."""
        try:
            # Model architecture info
            total_params = sum(p.numel() for p in self.q_network.parameters())
            trainable_params = sum(p.numel() for p in self.q_network.parameters() if p.requires_grad)
            
            # Training statistics
            avg_reward = np.mean(self.episode_rewards) if self.episode_rewards else 0.0
            avg_return = np.mean(self.episode_returns) if self.episode_returns else 0.0
            avg_loss = np.mean(self.training_losses[-100:]) if len(self.training_losses) >= 100 else 0.0
            
            # Action statistics
            if self.action_history:
                action_counts = np.bincount(self.action_history, minlength=self.action_dim)
                action_distribution = action_counts / len(self.action_history)
            else:
                action_distribution = np.zeros(self.action_dim)
            
            return {
                'model_architecture': {
                    'state_dim': self.state_dim,
                    'action_dim': self.action_dim,
                    'total_parameters': total_params,
                    'trainable_parameters': trainable_params
                },
                'training_state': {
                    'steps_done': self.steps_done,
                    'episode_count': self.episode_count,
                    'current_epsilon': self.get_epsilon(),
                    'buffer_size': len(self.replay_buffer)
                },
                'performance_metrics': {
                    'average_episode_reward': avg_reward,
                    'average_episode_return': avg_return,
                    'average_training_loss': avg_loss,
                    'total_experiences': len(self.action_history)
                },
                'action_distribution': {
                    action_name: float(prob)
                    for action_name, prob in zip(TradingAction.get_action_names(), action_distribution)
                },
                'hyperparameters': {
                    'learning_rate': self.learning_rate,
                    'gamma': self.gamma,
                    'epsilon_start': self.epsilon_start,
                    'epsilon_end': self.epsilon_end,
                    'epsilon_decay': self.epsilon_decay,
                    'batch_size': self.batch_size,
                    'target_update_freq': self.target_update_freq
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting model summary: {e}")
            return {"error": str(e)}
    
    def close(self):
        """Close tensorboard writer and cleanup."""
        try:
            self.writer.close()
            logger.info("DQN Trading Agent closed successfully")
        except Exception as e:
            logger.error(f"Error closing DQN Trading Agent: {e}")


# Utility functions
def create_sample_trading_state() -> TradingState:
    """Create a sample trading state for testing."""
    return TradingState(
        # Market features
        price_momentum_1m=np.random.normal(0, 0.01),
        price_momentum_5m=np.random.normal(0, 0.02),
        price_momentum_15m=np.random.normal(0, 0.03),
        price_momentum_30m=np.random.normal(0, 0.04),
        price_volatility=np.random.uniform(0.01, 0.05),
        volume_ratio=np.random.uniform(0.5, 2.0),
        volume_trend=np.random.normal(0, 0.1),
        volume_momentum=np.random.normal(0, 0.05),
        rsi=np.random.uniform(0, 1),
        macd=np.random.normal(0, 0.01),
        macd_signal=np.random.normal(0, 0.01),
        bb_position=np.random.uniform(0, 1),
        bb_width=np.random.uniform(0.01, 0.1),
        sma_20=np.random.normal(0, 0.02),
        ema_12=np.random.normal(0, 0.02),
        ema_26=np.random.normal(0, 0.02),
        atr=np.random.uniform(0.01, 0.05),
        stoch_k=np.random.uniform(0, 1),
        stoch_d=np.random.uniform(0, 1),
        williams_r=np.random.uniform(-1, 0),
        cci=np.random.normal(0, 50),
        adx=np.random.uniform(0, 1),
        trend_strength=np.random.uniform(0, 1),
        support_resistance_distance=np.random.uniform(0, 0.1),
        breakout_probability=np.random.uniform(0, 1),
        mean_reversion_signal=np.random.uniform(-1, 1),
        market_regime_trending=np.random.uniform(0, 1),
        market_regime_sideways=np.random.uniform(0, 1),
        market_regime_volatile=np.random.uniform(0, 1),
        news_sentiment=np.random.uniform(-1, 1),
        
        # Portfolio state
        cash_ratio=np.random.uniform(0, 1),
        position_size=np.random.uniform(-1, 1),
        unrealized_pnl=np.random.normal(0, 0.05),
        realized_pnl=np.random.normal(0, 0.1),
        total_return=np.random.normal(0.05, 0.2),
        sharpe_ratio=np.random.normal(1.0, 0.5),
        max_drawdown=np.random.uniform(-0.2, 0),
        win_rate=np.random.uniform(0.3, 0.8),
        avg_trade_duration=np.random.uniform(30, 300),
        trades_count=np.random.uniform(0, 100),
        
        # Strategy performance
        momentum_performance=np.random.normal(0.02, 0.1),
        mean_reversion_performance=np.random.normal(0.01, 0.08),
        breakout_performance=np.random.normal(0.03, 0.12),
        trend_following_performance=np.random.normal(0.025, 0.1),
        scalping_performance=np.random.normal(0.005, 0.05),
        momentum_confidence=np.random.uniform(0, 1),
        mean_reversion_confidence=np.random.uniform(0, 1),
        breakout_confidence=np.random.uniform(0, 1),
        trend_following_confidence=np.random.uniform(0, 1),
        scalping_confidence=np.random.uniform(0, 1)
    )


def simulate_trading_episode(agent: DQNTradingAgent, 
                           num_steps: int = 100) -> Dict[str, Any]:
    """Simulate a trading episode for testing."""
    states = []
    actions = []
    rewards = []
    portfolio_values = []
    
    # Initial state and portfolio value
    state = create_sample_trading_state()
    portfolio_value = 100000.0  # Starting with $100k
    
    for step in range(num_steps):
        states.append(state)
        portfolio_values.append(portfolio_value)
        
        # Select action
        action = agent.select_action(state, training=True)
        actions.append(action)
        
        # Simulate market movement and calculate reward
        market_return = np.random.normal(0.0001, 0.01)  # Daily return
        
        if action == TradingAction.BUY:
            portfolio_return = market_return * 1.0  # Full exposure
        elif action == TradingAction.SELL:
            portfolio_return = -market_return * 1.0  # Short position
        elif action == TradingAction.ADJUST_POSITION_SIZE:
            portfolio_return = market_return * 0.5  # Half exposure
        else:  # HOLD
            portfolio_return = 0.0
        
        # Update portfolio value
        new_portfolio_value = portfolio_value * (1 + portfolio_return)
        
        # Calculate reward
        reward_obj = agent.reward_calculator.calculate_reward(
            action=action,
            current_return=portfolio_return,
            portfolio_value=new_portfolio_value,
            previous_portfolio_value=portfolio_value,
            volatility=0.01,  # Simplified
            max_drawdown=-0.05,  # Simplified
            position_size=0.5  # Simplified
        )
        
        rewards.append(reward_obj.total_reward)
        portfolio_value = new_portfolio_value
        
        # Create next state (simplified)
        state = create_sample_trading_state()
    
    # Train episode
    episode_metrics = agent.train_episode(states, actions, rewards, portfolio_values)
    
    return {
        'episode_metrics': episode_metrics,
        'final_portfolio_value': portfolio_value,
        'total_return': (portfolio_value - 100000.0) / 100000.0,
        'num_steps': num_steps
    }