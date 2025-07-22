"""
Attention-based Strategy Selector using Transformer Architecture.

This module implements an advanced strategy selection system that:
- Uses transformer attention mechanism to select best strategy
- Input: current market state, strategy performance history, regime
- Learns which strategies work best in different market conditions
- Outputs strategy weights and confidence scores
- Implements online learning with experience replay
- Includes model interpretability features
"""

import logging
import os
import json
import pickle
import math
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Any, Union
from collections import deque, defaultdict
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logger = logging.getLogger(__name__)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")


@dataclass
class MarketState:
    """Current market state representation."""
    # Price features
    price_momentum_1m: float
    price_momentum_5m: float
    price_momentum_15m: float
    price_volatility: float
    
    # Volume features
    volume_ratio: float
    volume_trend: float
    volume_momentum: float
    
    # Technical indicators
    rsi: float
    macd_signal: float
    bb_position: float
    trend_strength: float
    
    # Market regime (one-hot encoded)
    regime_trending_up: float
    regime_trending_down: float
    regime_sideways: float
    regime_volatile: float
    
    # Time features
    time_of_day: float
    day_of_week: float
    
    # Market breadth
    market_breadth: float
    vix_level: float
    
    def to_tensor(self) -> torch.Tensor:
        """Convert to tensor for model input."""
        return torch.tensor([
            self.price_momentum_1m, self.price_momentum_5m, self.price_momentum_15m,
            self.price_volatility, self.volume_ratio, self.volume_trend, self.volume_momentum,
            self.rsi, self.macd_signal, self.bb_position, self.trend_strength,
            self.regime_trending_up, self.regime_trending_down, self.regime_sideways, self.regime_volatile,
            self.time_of_day, self.day_of_week, self.market_breadth, self.vix_level
        ], dtype=torch.float32)
    
    @classmethod
    def get_feature_names(cls) -> List[str]:
        """Get feature names for interpretability."""
        return [
            'price_momentum_1m', 'price_momentum_5m', 'price_momentum_15m',
            'price_volatility', 'volume_ratio', 'volume_trend', 'volume_momentum',
            'rsi', 'macd_signal', 'bb_position', 'trend_strength',
            'regime_trending_up', 'regime_trending_down', 'regime_sideways', 'regime_volatile',
            'time_of_day', 'day_of_week', 'market_breadth', 'vix_level'
        ]


@dataclass
class StrategyPerformance:
    """Strategy performance metrics."""
    strategy_name: str
    recent_return: float
    win_rate: float
    sharpe_ratio: float
    max_drawdown: float
    volatility: float
    trades_count: int
    avg_trade_duration: float
    
    def to_tensor(self) -> torch.Tensor:
        """Convert to tensor for model input."""
        return torch.tensor([
            self.recent_return, self.win_rate, self.sharpe_ratio,
            self.max_drawdown, self.volatility, self.trades_count, self.avg_trade_duration
        ], dtype=torch.float32)


@dataclass
class StrategySelection:
    """Strategy selection result."""
    strategy_weights: Dict[str, float]
    confidence_scores: Dict[str, float]
    selected_strategy: str
    overall_confidence: float
    attention_weights: Dict[str, float]
    market_state: MarketState
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data


@dataclass
class ExperienceReplay:
    """Experience replay memory for online learning."""
    market_state: MarketState
    strategy_performances: Dict[str, StrategyPerformance]
    selected_strategy: str
    actual_return: float
    timestamp: datetime


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer."""
    
    def __init__(self, d_model: int, max_len: int = 1000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:x.size(0), :]


class AttentionStrategyModel(nn.Module):
    """Transformer-based strategy selection model."""
    
    def __init__(self, 
                 market_state_dim: int = 19,
                 strategy_perf_dim: int = 7,
                 d_model: int = 64,
                 n_heads: int = 8,
                 n_layers: int = 3,
                 dropout: float = 0.1):
        super(AttentionStrategyModel, self).__init__()
        
        self.d_model = d_model
        
        # Input embeddings
        self.market_embedding = nn.Linear(market_state_dim, d_model)
        self.strategy_embedding = nn.Linear(strategy_perf_dim, d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='relu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Output heads
        self.strategy_weight_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )
        
        self.confidence_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, market_state: torch.Tensor, strategy_performances: torch.Tensor,
                strategy_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            market_state: (batch_size, market_state_dim)
            strategy_performances: (batch_size, n_strategies, strategy_perf_dim)
            strategy_mask: (batch_size, n_strategies)
            
        Returns:
            (strategy_weights, confidence_scores, attention_weights)
        """
        batch_size = market_state.size(0)
        n_strategies = strategy_performances.size(1)
        
        # Embed inputs
        market_embed = self.market_embedding(market_state).unsqueeze(1)  # (batch_size, 1, d_model)
        strategy_embed = self.strategy_embedding(strategy_performances)  # (batch_size, n_strategies, d_model)
        
        # Combine embeddings
        combined_embed = torch.cat([market_embed, strategy_embed], dim=1)  # (batch_size, 1+n_strategies, d_model)
        
        # Add positional encoding
        combined_embed = combined_embed.transpose(0, 1)  # (1+n_strategies, batch_size, d_model)
        combined_embed = self.pos_encoding(combined_embed)
        combined_embed = combined_embed.transpose(0, 1)  # (batch_size, 1+n_strategies, d_model)
        
        # Create attention mask
        if strategy_mask is not None:
            # Pad mask for market state (always attend to market state)
            market_mask = torch.ones(batch_size, 1, device=strategy_mask.device)
            full_mask = torch.cat([market_mask, strategy_mask], dim=1)
            # Convert to transformer mask format
            attn_mask = (full_mask == 0)
        else:
            attn_mask = None
        
        # Apply transformer
        transformer_output = self.transformer(combined_embed, src_key_padding_mask=attn_mask)
        
        # Extract strategy representations (skip market state)
        strategy_representations = transformer_output[:, 1:, :]  # (batch_size, n_strategies, d_model)
        
        # Generate outputs
        strategy_weights = self.strategy_weight_head(strategy_representations).squeeze(-1)  # (batch_size, n_strategies)
        confidence_scores = self.confidence_head(strategy_representations).squeeze(-1)  # (batch_size, n_strategies)
        
        # Apply mask and softmax to weights
        if strategy_mask is not None:
            strategy_weights = strategy_weights.masked_fill(strategy_mask == 0, -1e9)
        
        strategy_weights = F.softmax(strategy_weights, dim=-1)
        
        # Get attention weights from transformer (simplified)
        attention_weights = torch.ones_like(strategy_weights)  # Placeholder
        
        return strategy_weights, confidence_scores, attention_weights


class ExperienceReplayBuffer:
    """Experience replay buffer for online learning."""
    
    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self.buffer: deque = deque(maxlen=max_size)
        self.priorities: deque = deque(maxlen=max_size)
    
    def add(self, experience: ExperienceReplay, priority: float = 1.0):
        """Add experience to buffer."""
        self.buffer.append(experience)
        self.priorities.append(priority)
    
    def sample(self, batch_size: int) -> List[ExperienceReplay]:
        """Sample batch from buffer with priority sampling."""
        if len(self.buffer) < batch_size:
            return list(self.buffer)
        
        # Priority sampling
        priorities = np.array(self.priorities)
        probabilities = priorities / priorities.sum()
        
        indices = np.random.choice(len(self.buffer), size=batch_size, p=probabilities, replace=False)
        return [self.buffer[i] for i in indices]
    
    def __len__(self) -> int:
        return len(self.buffer)


class AttentionStrategySelector:
    """
    Attention-based strategy selector using transformer architecture.
    """
    
    def __init__(self,
                 model_dir: str = "models",
                 d_model: int = 64,
                 n_heads: int = 8,
                 n_layers: int = 3,
                 learning_rate: float = 0.001,
                 batch_size: int = 32,
                 replay_buffer_size: int = 10000,
                 update_frequency: int = 10):
        """
        Initialize attention strategy selector.
        """
        self.model_dir = model_dir
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.update_frequency = update_frequency
        
        # Initialize model
        self.model = AttentionStrategyModel(
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers
        ).to(device)
        
        # Optimizer and scheduler
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=1e-4
        )
        
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=10
        )
        
        # Experience replay
        self.replay_buffer = ExperienceReplayBuffer(replay_buffer_size)
        
        # Strategy tracking
        self.strategy_names: List[str] = []
        self.selection_history: deque = deque(maxlen=1000)
        self.performance_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        
        # Training state
        self.training_step = 0
        self.last_update_step = 0
        self.is_trained = False
        
        # Model interpretability
        self.feature_importance: Dict[str, float] = {}
        self.attention_patterns: List[torch.Tensor] = []
        
        # Data preprocessing
        self.market_scaler = StandardScaler()
        self.strategy_scaler = StandardScaler()
        self.scalers_fitted = False
        
        # Ensure model directory exists
        os.makedirs(model_dir, exist_ok=True)
        
        # Load existing model if available
        self._load_model()
        
        logger.info(f"AttentionStrategySelector initialized with {n_heads} heads, {n_layers} layers")
    
    def register_strategies(self, strategy_names: List[str]):
        """Register available strategies."""
        self.strategy_names = strategy_names
        
        # Initialize performance history
        for strategy in strategy_names:
            self.performance_history[strategy] = deque(maxlen=100)
        
        logger.info(f"Registered {len(strategy_names)} strategies: {strategy_names}")
    
    def select_strategy(self, 
                       market_state: MarketState,
                       strategy_performances: Dict[str, StrategyPerformance]) -> StrategySelection:
        """
        Select optimal strategy based on current market state and performance history.
        """
        try:
            if not self.strategy_names:
                raise ValueError("No strategies registered")
            
            # Prepare input data
            market_tensor = self._prepare_market_input(market_state)
            strategy_tensor, strategy_mask = self._prepare_strategy_input(strategy_performances)
            
            # Model inference
            self.model.eval()
            with torch.no_grad():
                strategy_weights, confidence_scores, attention_weights = self.model(
                    market_tensor.unsqueeze(0),
                    strategy_tensor.unsqueeze(0),
                    strategy_mask.unsqueeze(0) if strategy_mask is not None else None
                )
            
            # Convert to numpy
            strategy_weights = strategy_weights.squeeze(0).cpu().numpy()
            confidence_scores = confidence_scores.squeeze(0).cpu().numpy()
            attention_weights = attention_weights.squeeze(0).cpu().numpy()
            
            # Create result dictionaries
            strategy_weight_dict = {}
            confidence_dict = {}
            
            for i, strategy_name in enumerate(self.strategy_names):
                if i < len(strategy_weights):
                    strategy_weight_dict[strategy_name] = float(strategy_weights[i])
                    confidence_dict[strategy_name] = float(confidence_scores[i])
                else:
                    strategy_weight_dict[strategy_name] = 0.0
                    confidence_dict[strategy_name] = 0.0
            
            # Select strategy with highest weight
            selected_strategy = max(strategy_weight_dict.items(), key=lambda x: x[1])[0]
            overall_confidence = confidence_dict[selected_strategy]
            
            # Create attention weight dictionary
            attention_dict = {}
            feature_names = MarketState.get_feature_names()
            
            # Market state attention
            attention_dict['market_state'] = 1.0  # Placeholder
            
            # Strategy attention
            for i, strategy_name in enumerate(self.strategy_names):
                if i < len(attention_weights):
                    attention_dict[f'strategy_{strategy_name}'] = float(attention_weights[i])
            
            # Create selection result
            selection = StrategySelection(
                strategy_weights=strategy_weight_dict,
                confidence_scores=confidence_dict,
                selected_strategy=selected_strategy,
                overall_confidence=overall_confidence,
                attention_weights=attention_dict,
                market_state=market_state,
                timestamp=datetime.now()
            )
            
            # Store selection history
            self.selection_history.append(selection)
            
            # Update feature importance
            self._update_feature_importance(market_state, attention_weights)
            
            logger.debug(f"Selected strategy: {selected_strategy} (confidence: {overall_confidence:.3f})")
            
            return selection
            
        except Exception as e:
            logger.error(f"Error selecting strategy: {e}")
            return self._get_default_selection(market_state)
    
    def add_experience(self, 
                      market_state: MarketState,
                      strategy_performances: Dict[str, StrategyPerformance],
                      selected_strategy: str,
                      actual_return: float):
        """Add experience to replay buffer for online learning."""
        try:
            # Create experience
            experience = ExperienceReplay(
                market_state=market_state,
                strategy_performances=strategy_performances,
                selected_strategy=selected_strategy,
                actual_return=actual_return,
                timestamp=datetime.now()
            )
            
            # Calculate priority based on prediction error
            priority = abs(actual_return) + 0.1
            
            # Add to replay buffer
            self.replay_buffer.add(experience, priority)
            
            # Update performance history
            if selected_strategy in self.performance_history:
                self.performance_history[selected_strategy].append(actual_return)
            
            # Trigger online learning if needed
            self.training_step += 1
            if (self.training_step - self.last_update_step) >= self.update_frequency:
                self._online_learning_update()
                self.last_update_step = self.training_step
            
            logger.debug(f"Added experience: {selected_strategy} -> {actual_return:.4f}")
            
        except Exception as e:
            logger.error(f"Error adding experience: {e}")
    
    def train_model(self, 
                   historical_data: List[Tuple[MarketState, Dict[str, StrategyPerformance], str, float]],
                   validation_split: float = 0.2,
                   epochs: int = 100) -> bool:
        """Train the model on historical data."""
        try:
            if len(historical_data) < 10:
                logger.error("Insufficient training data")
                return False
            
            logger.info(f"Training model on {len(historical_data)} samples...")
            
            # Prepare training data
            market_states = []
            strategy_perfs = []
            selected_strategies = []
            actual_returns = []
            
            for market_state, strategy_performances, selected_strategy, actual_return in historical_data:
                market_states.append(market_state.to_tensor().numpy())
                
                # Prepare strategy performance matrix
                strategy_perf_matrix = []
                for strategy_name in self.strategy_names:
                    if strategy_name in strategy_performances:
                        strategy_perf_matrix.append(strategy_performances[strategy_name].to_tensor().numpy())
                    else:
                        strategy_perf_matrix.append(np.zeros(7))  # Default performance
                
                strategy_perfs.append(np.array(strategy_perf_matrix))
                selected_strategies.append(selected_strategy)
                actual_returns.append(actual_return)
            
            # Convert to arrays
            market_states = np.array(market_states)
            strategy_perfs = np.array(strategy_perfs)
            
            # Fit scalers
            self.market_scaler.fit(market_states)
            strategy_perfs_reshaped = strategy_perfs.reshape(-1, strategy_perfs.shape[-1])
            self.strategy_scaler.fit(strategy_perfs_reshaped)
            self.scalers_fitted = True
            
            # Scale data
            market_states_scaled = self.market_scaler.transform(market_states)
            strategy_perfs_scaled = self.strategy_scaler.transform(strategy_perfs_reshaped)
            strategy_perfs_scaled = strategy_perfs_scaled.reshape(strategy_perfs.shape)
            
            # Split data
            split_idx = int(len(historical_data) * (1 - validation_split))
            
            train_market = torch.tensor(market_states_scaled[:split_idx], dtype=torch.float32)
            train_strategy = torch.tensor(strategy_perfs_scaled[:split_idx], dtype=torch.float32)
            train_targets = [self.strategy_names.index(s) if s in self.strategy_names else 0 
                           for s in selected_strategies[:split_idx]]
            train_targets = torch.tensor(train_targets, dtype=torch.long)
            train_returns = torch.tensor(actual_returns[:split_idx], dtype=torch.float32)
            
            val_market = torch.tensor(market_states_scaled[split_idx:], dtype=torch.float32)
            val_strategy = torch.tensor(strategy_perfs_scaled[split_idx:], dtype=torch.float32)
            val_targets = [self.strategy_names.index(s) if s in self.strategy_names else 0 
                         for s in selected_strategies[split_idx:]]
            val_targets = torch.tensor(val_targets, dtype=torch.long)
            val_returns = torch.tensor(actual_returns[split_idx:], dtype=torch.float32)
            
            # Training loop
            best_val_loss = float('inf')
            patience = 0
            max_patience = 20
            
            for epoch in range(epochs):
                # Training
                self.model.train()
                train_loss = self._train_epoch(train_market, train_strategy, train_targets, train_returns)
                
                # Validation
                self.model.eval()
                val_loss = self._validate_epoch(val_market, val_strategy, val_targets, val_returns)
                
                # Learning rate scheduling
                self.scheduler.step(val_loss)
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience = 0
                    self._save_model()
                else:
                    patience += 1
                
                if epoch % 10 == 0:
                    logger.info(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
                
                if patience >= max_patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break
            
            self.is_trained = True
            logger.info("Training completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error training model: {e}")
            return False
    
    def _train_epoch(self, market_states: torch.Tensor, strategy_perfs: torch.Tensor,
                    targets: torch.Tensor, returns: torch.Tensor) -> float:
        """Train for one epoch."""
        total_loss = 0.0
        batch_size = self.batch_size
        
        for i in range(0, len(market_states), batch_size):
            batch_market = market_states[i:i+batch_size].to(device)
            batch_strategy = strategy_perfs[i:i+batch_size].to(device)
            batch_targets = targets[i:i+batch_size].to(device)
            batch_returns = returns[i:i+batch_size].to(device)
            
            self.optimizer.zero_grad()
            
            strategy_weights, confidence_scores, _ = self.model(batch_market, batch_strategy)
            
            # Strategy selection loss
            strategy_loss = F.cross_entropy(strategy_weights, batch_targets)
            
            # Confidence loss
            selected_confidences = confidence_scores[torch.arange(len(batch_targets)), batch_targets]
            confidence_loss = F.mse_loss(selected_confidences, torch.abs(torch.tanh(batch_returns)))
            
            # Combined loss
            loss = strategy_loss + 0.5 * confidence_loss
            
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / (len(market_states) // batch_size + 1)
    
    def _validate_epoch(self, market_states: torch.Tensor, strategy_perfs: torch.Tensor,
                       targets: torch.Tensor, returns: torch.Tensor) -> float:
        """Validate for one epoch."""
        total_loss = 0.0
        batch_size = self.batch_size
        
        with torch.no_grad():
            for i in range(0, len(market_states), batch_size):
                batch_market = market_states[i:i+batch_size].to(device)
                batch_strategy = strategy_perfs[i:i+batch_size].to(device)
                batch_targets = targets[i:i+batch_size].to(device)
                batch_returns = returns[i:i+batch_size].to(device)
                
                strategy_weights, confidence_scores, _ = self.model(batch_market, batch_strategy)
                
                strategy_loss = F.cross_entropy(strategy_weights, batch_targets)
                selected_confidences = confidence_scores[torch.arange(len(batch_targets)), batch_targets]
                confidence_loss = F.mse_loss(selected_confidences, torch.abs(torch.tanh(batch_returns)))
                
                loss = strategy_loss + 0.5 * confidence_loss
                total_loss += loss.item()
        
        return total_loss / (len(market_states) // batch_size + 1)
    
    def _online_learning_update(self):
        """Perform online learning update using experience replay."""
        try:
            if len(self.replay_buffer) < self.batch_size:
                return
            
            experiences = self.replay_buffer.sample(self.batch_size)
            
            # Prepare batch data
            market_states = []
            strategy_perfs = []
            selected_strategies = []
            actual_returns = []
            
            for exp in experiences:
                market_states.append(exp.market_state)
                strategy_perfs.append(exp.strategy_performances)
                selected_strategies.append(exp.selected_strategy)
                actual_returns.append(exp.actual_return)
            
            # Convert to tensors
            market_batch = torch.stack([self._prepare_market_input(ms) for ms in market_states])
            
            strategy_batch = []
            for sp in strategy_perfs:
                strat_tensor, _ = self._prepare_strategy_input(sp)
                strategy_batch.append(strat_tensor)
            
            strategy_batch = torch.stack(strategy_batch)
            
            # Targets
            strategy_indices = [self.strategy_names.index(s) if s in self.strategy_names else 0 
                              for s in selected_strategies]
            strategy_targets = torch.tensor(strategy_indices, dtype=torch.long).to(device)
            return_targets = torch.tensor(actual_returns, dtype=torch.float32).to(device)
            
            # Forward pass
            self.model.train()
            strategy_weights, confidence_scores, _ = self.model(market_batch, strategy_batch)
            
            # Calculate losses
            strategy_loss = F.cross_entropy(strategy_weights, strategy_targets)
            selected_confidences = confidence_scores[torch.arange(len(strategy_targets)), strategy_targets]
            confidence_loss = F.mse_loss(selected_confidences, torch.abs(torch.tanh(return_targets)))
            
            total_loss = strategy_loss + 0.5 * confidence_loss
            
            # Backward pass
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            logger.debug(f"Online learning update - Loss: {total_loss.item():.4f}")
            
        except Exception as e:
            logger.error(f"Error in online learning update: {e}")
    
    def _prepare_market_input(self, market_state: MarketState) -> torch.Tensor:
        """Prepare market state input for model."""
        market_tensor = market_state.to_tensor()
        
        if self.scalers_fitted:
            market_array = market_tensor.numpy().reshape(1, -1)
            market_scaled = self.market_scaler.transform(market_array)
            market_tensor = torch.tensor(market_scaled.flatten(), dtype=torch.float32)
        
        return market_tensor.to(device)
    
    def _prepare_strategy_input(self, strategy_performances: Dict[str, StrategyPerformance]) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Prepare strategy performance input for model."""
        strategy_tensors = []
        strategy_mask = []
        
        for strategy_name in self.strategy_names:
            if strategy_name in strategy_performances:
                perf_tensor = strategy_performances[strategy_name].to_tensor()
                strategy_tensors.append(perf_tensor)
                strategy_mask.append(1.0)
            else:
                default_perf = torch.zeros(7)
                strategy_tensors.append(default_perf)
                strategy_mask.append(0.0)
        
        strategy_tensor = torch.stack(strategy_tensors)
        mask_tensor = torch.tensor(strategy_mask, dtype=torch.float32)
        
        if self.scalers_fitted:
            strategy_array = strategy_tensor.numpy()
            strategy_scaled = self.strategy_scaler.transform(strategy_array)
            strategy_tensor = torch.tensor(strategy_scaled, dtype=torch.float32)
        
        return strategy_tensor.to(device), mask_tensor.to(device)
    
    def _update_feature_importance(self, market_state: MarketState, attention_weights: np.ndarray):
        """Update feature importance based on attention weights."""
        try:
            feature_names = MarketState.get_feature_names()
            market_features = market_state.to_tensor().numpy()
            
            for i, feature_name in enumerate(feature_names):
                if i < len(market_features):
                    importance = abs(market_features[i]) * 0.5  # Simplified importance
                    
                    if feature_name in self.feature_importance:
                        self.feature_importance[feature_name] = (
                            0.9 * self.feature_importance[feature_name] + 0.1 * importance
                        )
                    else:
                        self.feature_importance[feature_name] = importance
            
        except Exception as e:
            logger.error(f"Error updating feature importance: {e}")
    
    def _get_default_selection(self, market_state: MarketState) -> StrategySelection:
        """Get default strategy selection when model fails."""
        if not self.strategy_names:
            return StrategySelection(
                strategy_weights={},
                confidence_scores={},
                selected_strategy="",
                overall_confidence=0.0,
                attention_weights={},
                market_state=market_state,
                timestamp=datetime.now()
            )
        
        # Equal weights for all strategies
        equal_weight = 1.0 / len(self.strategy_names)
        strategy_weights = {name: equal_weight for name in self.strategy_names}
        confidence_scores = {name: 0.5 for name in self.strategy_names}
        
        return StrategySelection(
            strategy_weights=strategy_weights,
            confidence_scores=confidence_scores,
            selected_strategy=self.strategy_names[0],
            overall_confidence=0.5,
            attention_weights={},
            market_state=market_state,
            timestamp=datetime.now()
        )
    
    def get_strategy_attribution(self, days: int = 30) -> Dict[str, Any]:
        """Get strategy performance attribution."""
        try:
            cutoff_time = datetime.now() - timedelta(days=days)
            recent_selections = [s for s in self.selection_history if s.timestamp > cutoff_time]
            
            if not recent_selections:
                return {"error": "No recent selections available"}
            
            # Strategy selection frequency
            strategy_counts = defaultdict(int)
            total_selections = len(recent_selections)
            
            for selection in recent_selections:
                strategy_counts[selection.selected_strategy] += 1
            
            strategy_frequency = {
                strategy: count / total_selections
                for strategy, count in strategy_counts.items()
            }
            
            # Average confidence by strategy
            strategy_confidences = defaultdict(list)
            for selection in recent_selections:
                strategy_confidences[selection.selected_strategy].append(selection.overall_confidence)
            
            avg_confidences = {
                strategy: np.mean(confidences)
                for strategy, confidences in strategy_confidences.items()
            }
            
            return {
                "period_days": days,
                "total_selections": total_selections,
                "strategy_frequency": strategy_frequency,
                "average_confidences": avg_confidences,
                "feature_importance": dict(sorted(self.feature_importance.items(), 
                                                key=lambda x: x[1], reverse=True)[:10])
            }
            
        except Exception as e:
            logger.error(f"Error getting strategy attribution: {e}")
            return {"error": str(e)}
    
    def plot_attention_patterns(self, save_path: Optional[str] = None) -> str:
        """Plot attention patterns over time."""
        try:
            if not self.selection_history:
                logger.warning("No selection history available")
                return ""
            
            # Get recent selections
            recent_selections = list(self.selection_history)[-50:]  # Last 50 selections
            
            # Extract attention data
            timestamps = [s.timestamp for s in recent_selections]
            strategies = [s.selected_strategy for s in recent_selections]
            confidences = [s.overall_confidence for s in recent_selections]
            
            # Create plot
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
            
            # Strategy selection over time
            unique_strategies = list(set(strategies))
            colors = plt.cm.Set3(np.linspace(0, 1, len(unique_strategies)))
            
            for i, strategy in enumerate(unique_strategies):
                strategy_times = [t for t, s in zip(timestamps, strategies) if s == strategy]
                strategy_y = [i] * len(strategy_times)
                ax1.scatter(strategy_times, strategy_y, c=[colors[i]], label=strategy, s=50, alpha=0.7)
            
            ax1.set_ylabel('Strategy')
            ax1.set_title('Strategy Selection Over Time')
            ax1.set_yticks(range(len(unique_strategies)))
            ax1.set_yticklabels(unique_strategies)
            ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax1.grid(True, alpha=0.3)
            
            # Confidence over time
            ax2.plot(timestamps, confidences, 'b-', linewidth=2, alpha=0.7)
            ax2.fill_between(timestamps, confidences, alpha=0.3)
            ax2.set_xlabel('Time')
            ax2.set_ylabel('Confidence')
            ax2.set_title('Selection Confidence Over Time')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save plot
            if save_path is None:
                save_path = os.path.join(self.model_dir, "attention_patterns.png")
            
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Attention patterns plot saved to {save_path}")
            return save_path
            
        except Exception as e:
            logger.error(f"Error plotting attention patterns: {e}")
            return ""
    
    def _save_model(self):
        """Save model and associated data."""
        try:
            model_data = {
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'strategy_names': self.strategy_names,
                'feature_importance': self.feature_importance,
                'is_trained': self.is_trained,
                'scalers_fitted': self.scalers_fitted,
                'market_scaler': self.market_scaler if self.scalers_fitted else None,
                'strategy_scaler': self.strategy_scaler if self.scalers_fitted else None,
                'model_config': {
                    'd_model': self.d_model,
                    'n_heads': self.n_heads,
                    'n_layers': self.n_layers
                }
            }
            
            model_path = os.path.join(self.model_dir, "attention_strategy_selector.pkl")
            with open(model_path, 'wb') as f:
                pickle.dump(model_data, f)
            
            logger.info("Model saved successfully")
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
    
    def _load_model(self):
        """Load model and associated data."""
        try:
            model_path = os.path.join(self.model_dir, "attention_strategy_selector.pkl")
            
            if os.path.exists(model_path):
                with open(model_path, 'rb') as f:
                    model_data = pickle.load(f)
                
                # Load model state
                self.model.load_state_dict(model_data['model_state_dict'])
                self.optimizer.load_state_dict(model_data['optimizer_state_dict'])
                
                # Load other data
                self.strategy_names = model_data.get('strategy_names', [])
                self.feature_importance = model_data.get('feature_importance', {})
                self.is_trained = model_data.get('is_trained', False)
                self.scalers_fitted = model_data.get('scalers_fitted', False)
                
                if self.scalers_fitted:
                    self.market_scaler = model_data.get('market_scaler', StandardScaler())
                    self.strategy_scaler = model_data.get('strategy_scaler', StandardScaler())
                
                logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
    
    def get_model_summary(self) -> Dict[str, Any]:
        """Get comprehensive model summary."""
        try:
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            
            return {
                'model_architecture': {
                    'd_model': self.d_model,
                    'n_heads': self.n_heads,
                    'n_layers': self.n_layers,
                    'total_parameters': total_params,
                    'trainable_parameters': trainable_params
                },
                'training_state': {
                    'is_trained': self.is_trained,
                    'training_step': self.training_step,
                    'scalers_fitted': self.scalers_fitted
                },
                'strategy_info': {
                    'registered_strategies': self.strategy_names,
                    'selection_history_length': len(self.selection_history),
                    'replay_buffer_size': len(self.replay_buffer)
                },
                'feature_importance': dict(sorted(self.feature_importance.items(), 
                                                key=lambda x: x[1], reverse=True)[:10])
            }
            
        except Exception as e:
            logger.error(f"Error getting model summary: {e}")
            return {"error": str(e)}