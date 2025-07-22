"""
Multi-Agent Coordinator for Trading Bot

This module implements a sophisticated coordination system that manages multiple
reinforcement learning agents, each specialized for different trading strategies.
It handles agent communication, resource allocation, conflict resolution, and
meta-learning for optimal agent selection.

Key Features:
- Manages 6+ specialized trading agents simultaneously
- Implements agent communication and coordination protocols
- Uses meta-learning to adapt agent selection strategies
- Handles resource allocation and budgeting between agents
- Prevents conflicting trades through consensus mechanisms
- Provides agent lifecycle management and performance tracking
"""

import os
import json
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, NamedTuple
from dataclasses import dataclass, asdict
from enum import Enum
from collections import defaultdict, deque
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import pickle

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AgentStatus(Enum):
    """Agent lifecycle status"""
    INACTIVE = "inactive"
    STARTING = "starting"
    ACTIVE = "active"
    PAUSED = "paused"
    STOPPING = "stopping"
    ERROR = "error"


class TradeSignalType(Enum):
    """Types of trade signals"""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    CLOSE = "close"


class ConflictResolution(Enum):
    """Conflict resolution strategies"""
    CONSENSUS = "consensus"
    WEIGHTED_VOTE = "weighted_vote"
    PERFORMANCE_BASED = "performance_based"
    RESOURCE_BASED = "resource_based"


@dataclass
class TradeSignal:
    """Trade signal from an agent"""
    agent_id: str
    symbol: str
    signal_type: TradeSignalType
    confidence: float
    quantity: int
    price: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    reasoning: str = ""
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class AgentPerformance:
    """Agent performance metrics"""
    agent_id: str
    total_trades: int = 0
    winning_trades: int = 0
    total_pnl: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    avg_trade_duration: float = 0.0
    win_rate: float = 0.0
    avg_return_per_trade: float = 0.0
    volatility: float = 0.0
    last_updated: datetime = None
    
    def __post_init__(self):
        if self.last_updated is None:
            self.last_updated = datetime.now()
    
    def update_metrics(self, trade_pnl: float, trade_duration: float):
        """Update performance metrics with new trade"""
        self.total_trades += 1
        self.total_pnl += trade_pnl
        
        if trade_pnl > 0:
            self.winning_trades += 1
        
        self.win_rate = self.winning_trades / self.total_trades
        self.avg_return_per_trade = self.total_pnl / self.total_trades
        
        # Update average trade duration
        self.avg_trade_duration = (
            (self.avg_trade_duration * (self.total_trades - 1) + trade_duration) / 
            self.total_trades
        )
        
        self.last_updated = datetime.now()


@dataclass
class AgentResource:
    """Resource allocation for an agent"""
    agent_id: str
    allocated_capital: float
    used_capital: float = 0.0
    max_positions: int = 5
    current_positions: int = 0
    cpu_priority: float = 1.0
    memory_limit: float = 1.0  # GB
    
    @property
    def available_capital(self) -> float:
        return self.allocated_capital - self.used_capital
    
    @property
    def capital_utilization(self) -> float:
        return self.used_capital / self.allocated_capital if self.allocated_capital > 0 else 0.0
    
    @property
    def position_utilization(self) -> float:
        return self.current_positions / self.max_positions if self.max_positions > 0 else 0.0


class MetaLearningNetwork(nn.Module):
    """Neural network for meta-learning agent selection"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 128, num_agents: int = 6):
        super().__init__()
        self.input_dim = input_dim
        self.num_agents = num_agents
        
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Agent selection head
        self.agent_selector = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, num_agents),
            nn.Softmax(dim=-1)
        )
        
        # Resource allocation head
        self.resource_allocator = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, num_agents),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, x):
        features = self.feature_extractor(x)
        agent_weights = self.agent_selector(features)
        resource_weights = self.resource_allocator(features)
        return agent_weights, resource_weights


class TradingAgent:
    """Base class for trading agents"""
    
    def __init__(self, agent_id: str, strategy_type: str):
        self.agent_id = agent_id
        self.strategy_type = strategy_type
        self.status = AgentStatus.INACTIVE
        self.performance = AgentPerformance(agent_id=agent_id)
        self.last_signal_time = None
        self.active_positions = {}
        
    def generate_signal(self, market_data: Dict[str, Any]) -> Optional[TradeSignal]:
        """Generate trading signal based on market data"""
        raise NotImplementedError("Subclasses must implement generate_signal")
    
    def update_performance(self, trade_pnl: float, trade_duration: float):
        """Update agent performance metrics"""
        self.performance.update_metrics(trade_pnl, trade_duration)
    
    def start(self):
        """Start the agent"""
        self.status = AgentStatus.STARTING
        logger.info(f"Starting agent {self.agent_id}")
        self.status = AgentStatus.ACTIVE
    
    def stop(self):
        """Stop the agent"""
        self.status = AgentStatus.STOPPING
        logger.info(f"Stopping agent {self.agent_id}")
        self.status = AgentStatus.INACTIVE
    
    def pause(self):
        """Pause the agent"""
        self.status = AgentStatus.PAUSED
        logger.info(f"Pausing agent {self.agent_id}")
    
    def resume(self):
        """Resume the agent"""
        if self.status == AgentStatus.PAUSED:
            self.status = AgentStatus.ACTIVE
            logger.info(f"Resuming agent {self.agent_id}")


class MultiAgentCoordinator:
    """
    Coordinates multiple trading agents with sophisticated resource allocation,
    conflict resolution, and meta-learning capabilities.
    """
    
    def __init__(
        self,
        total_capital: float = 1000000.0,
        max_agents: int = 6,
        conflict_resolution: ConflictResolution = ConflictResolution.WEIGHTED_VOTE,
        meta_learning_enabled: bool = True,
        model_dir: str = "models/multi_agent",
        log_dir: str = "logs/multi_agent"
    ):
        self.total_capital = total_capital
        self.max_agents = max_agents
        self.conflict_resolution = conflict_resolution
        self.meta_learning_enabled = meta_learning_enabled
        self.model_dir = model_dir
        self.log_dir = log_dir
        
        # Create directories
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        
        # Agent management
        self.agents: Dict[str, TradingAgent] = {}
        self.agent_resources: Dict[str, AgentResource] = {}
        self.agent_performances: Dict[str, AgentPerformance] = {}
        
        # Communication and coordination
        self.signal_queue = deque(maxlen=1000)
        self.consensus_threshold = 0.6
        self.coordination_lock = threading.Lock()
        
        # Meta-learning
        if meta_learning_enabled:
            self.meta_network = MetaLearningNetwork(
                input_dim=50,  # Market state dimension
                hidden_dim=128,
                num_agents=max_agents
            )
            self.meta_optimizer = optim.Adam(self.meta_network.parameters(), lr=0.001)
            self.meta_experiences = deque(maxlen=10000)
        
        # Performance tracking
        self.performance_history = defaultdict(list)
        self.trade_history = []
        self.resource_utilization_history = []
        
        # Execution
        self.executor = ThreadPoolExecutor(max_workers=max_agents)
        self.is_running = False
        
        logger.info(f"MultiAgentCoordinator initialized with {max_agents} max agents")
    
    def register_agent(
        self, 
        agent: TradingAgent, 
        initial_capital: float = None,
        max_positions: int = 5
    ) -> bool:
        """Register a new trading agent"""
        if len(self.agents) >= self.max_agents:
            logger.error(f"Cannot register agent {agent.agent_id}: max agents reached")
            return False
        
        if agent.agent_id in self.agents:
            logger.error(f"Agent {agent.agent_id} already registered")
            return False
        
        # Calculate initial capital allocation
        if initial_capital is None:
            initial_capital = self.total_capital / self.max_agents
        
        # Register agent
        self.agents[agent.agent_id] = agent
        self.agent_resources[agent.agent_id] = AgentResource(
            agent_id=agent.agent_id,
            allocated_capital=initial_capital,
            max_positions=max_positions
        )
        self.agent_performances[agent.agent_id] = agent.performance
        
        logger.info(f"Registered agent {agent.agent_id} with ${initial_capital:,.2f} capital")
        return True
    
    def unregister_agent(self, agent_id: str) -> bool:
        """Unregister an agent"""
        if agent_id not in self.agents:
            logger.error(f"Agent {agent_id} not found")
            return False
        
        # Stop agent if running
        agent = self.agents[agent_id]
        if agent.status == AgentStatus.ACTIVE:
            agent.stop()
        
        # Remove from all tracking
        del self.agents[agent_id]
        del self.agent_resources[agent_id]
        del self.agent_performances[agent_id]
        
        logger.info(f"Unregistered agent {agent_id}")
        return True
    
    def start_agent(self, agent_id: str) -> bool:
        """Start a specific agent"""
        if agent_id not in self.agents:
            logger.error(f"Agent {agent_id} not found")
            return False
        
        agent = self.agents[agent_id]
        try:
            agent.start()
            logger.info(f"Started agent {agent_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to start agent {agent_id}: {e}")
            agent.status = AgentStatus.ERROR
            return False
    
    def stop_agent(self, agent_id: str) -> bool:
        """Stop a specific agent"""
        if agent_id not in self.agents:
            logger.error(f"Agent {agent_id} not found")
            return False
        
        agent = self.agents[agent_id]
        try:
            agent.stop()
            logger.info(f"Stopped agent {agent_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to stop agent {agent_id}: {e}")
            return False
    
    def start_all_agents(self):
        """Start all registered agents"""
        logger.info("Starting all agents...")
        for agent_id in self.agents:
            self.start_agent(agent_id)
        self.is_running = True
    
    def stop_all_agents(self):
        """Stop all agents"""
        logger.info("Stopping all agents...")
        for agent_id in self.agents:
            self.stop_agent(agent_id)
        self.is_running = False
    
    def collect_signals(self, market_data: Dict[str, Any]) -> List[TradeSignal]:
        """Collect trading signals from all active agents"""
        signals = []
        
        # Use thread pool to collect signals concurrently
        future_to_agent = {}
        for agent_id, agent in self.agents.items():
            if agent.status == AgentStatus.ACTIVE:
                future = self.executor.submit(agent.generate_signal, market_data)
                future_to_agent[future] = agent_id
        
        # Collect results
        for future in as_completed(future_to_agent, timeout=5.0):
            agent_id = future_to_agent[future]
            try:
                signal = future.result()
                if signal is not None:
                    signals.append(signal)
                    self.signal_queue.append(signal)
            except Exception as e:
                logger.error(f"Error collecting signal from agent {agent_id}: {e}")
        
        return signals
    
    def resolve_conflicts(self, signals: List[TradeSignal]) -> List[TradeSignal]:
        """Resolve conflicts between competing trade signals"""
        if not signals:
            return []
        
        # Group signals by symbol
        symbol_signals = defaultdict(list)
        for signal in signals:
            symbol_signals[signal.symbol].append(signal)
        
        resolved_signals = []
        
        for symbol, symbol_signal_list in symbol_signals.items():
            if len(symbol_signal_list) == 1:
                # No conflict
                resolved_signals.extend(symbol_signal_list)
                continue
            
            # Resolve conflict based on strategy
            if self.conflict_resolution == ConflictResolution.CONSENSUS:
                resolved = self._resolve_by_consensus(symbol_signal_list)
            elif self.conflict_resolution == ConflictResolution.WEIGHTED_VOTE:
                resolved = self._resolve_by_weighted_vote(symbol_signal_list)
            elif self.conflict_resolution == ConflictResolution.PERFORMANCE_BASED:
                resolved = self._resolve_by_performance(symbol_signal_list)
            elif self.conflict_resolution == ConflictResolution.RESOURCE_BASED:
                resolved = self._resolve_by_resources(symbol_signal_list)
            else:
                resolved = symbol_signal_list[:1]  # Take first signal
            
            if resolved:
                resolved_signals.extend(resolved)
        
        return resolved_signals
    
    def _resolve_by_consensus(self, signals: List[TradeSignal]) -> List[TradeSignal]:
        """Resolve conflicts using consensus mechanism"""
        # Count signal types
        signal_counts = defaultdict(int)
        signal_map = defaultdict(list)
        
        for signal in signals:
            signal_counts[signal.signal_type] += 1
            signal_map[signal.signal_type].append(signal)
        
        # Find consensus (majority)
        total_signals = len(signals)
        for signal_type, count in signal_counts.items():
            if count / total_signals >= self.consensus_threshold:
                # Consensus reached, return highest confidence signal of this type
                consensus_signals = signal_map[signal_type]
                best_signal = max(consensus_signals, key=lambda s: s.confidence)
                return [best_signal]
        
        # No consensus, return empty list (no trade)
        return []
    
    def _resolve_by_weighted_vote(self, signals: List[TradeSignal]) -> List[TradeSignal]:
        """Resolve conflicts using weighted voting based on confidence"""
        # Weight signals by confidence and agent performance
        weighted_signals = defaultdict(float)
        signal_map = defaultdict(list)
        
        for signal in signals:
            agent_perf = self.agent_performances.get(signal.agent_id)
            performance_weight = agent_perf.win_rate if agent_perf else 0.5
            
            total_weight = signal.confidence * performance_weight
            weighted_signals[signal.signal_type] += total_weight
            signal_map[signal.signal_type].append((signal, total_weight))
        
        # Find highest weighted signal type
        if not weighted_signals:
            return []
        
        best_signal_type = max(weighted_signals.keys(), key=lambda k: weighted_signals[k])
        best_signals = signal_map[best_signal_type]
        
        # Return highest weighted signal of the best type
        best_signal = max(best_signals, key=lambda x: x[1])[0]
        return [best_signal]
    
    def _resolve_by_performance(self, signals: List[TradeSignal]) -> List[TradeSignal]:
        """Resolve conflicts based on agent performance"""
        best_signal = None
        best_performance = -float('inf')
        
        for signal in signals:
            agent_perf = self.agent_performances.get(signal.agent_id)
            if agent_perf:
                # Use Sharpe ratio as performance metric
                performance_score = agent_perf.sharpe_ratio
                if performance_score > best_performance:
                    best_performance = performance_score
                    best_signal = signal
        
        return [best_signal] if best_signal else []
    
    def _resolve_by_resources(self, signals: List[TradeSignal]) -> List[TradeSignal]:
        """Resolve conflicts based on available resources"""
        valid_signals = []
        
        for signal in signals:
            resource = self.agent_resources.get(signal.agent_id)
            if resource and resource.available_capital >= signal.quantity * signal.price:
                valid_signals.append(signal)
        
        if not valid_signals:
            return []
        
        # Among valid signals, choose the one with highest confidence
        best_signal = max(valid_signals, key=lambda s: s.confidence)
        return [best_signal]
    
    def allocate_resources(self, signals: List[TradeSignal]) -> List[TradeSignal]:
        """Allocate resources for approved signals"""
        approved_signals = []
        
        with self.coordination_lock:
            for signal in signals:
                resource = self.agent_resources.get(signal.agent_id)
                if not resource:
                    continue
                
                required_capital = signal.quantity * signal.price
                
                # Check resource availability
                if (resource.available_capital >= required_capital and 
                    resource.current_positions < resource.max_positions):
                    
                    # Allocate resources
                    resource.used_capital += required_capital
                    resource.current_positions += 1
                    
                    approved_signals.append(signal)
                    logger.info(f"Allocated ${required_capital:,.2f} to agent {signal.agent_id}")
                else:
                    logger.warning(f"Insufficient resources for agent {signal.agent_id}")
        
        return approved_signals
    
    def update_meta_learning(self, market_state: np.ndarray, selected_agents: List[str], 
                           performance_outcome: float):
        """Update meta-learning network with experience"""
        if not self.meta_learning_enabled:
            return
        
        # Create target vector for selected agents
        target_selection = np.zeros(self.max_agents)
        agent_ids = list(self.agents.keys())
        
        for agent_id in selected_agents:
            if agent_id in agent_ids:
                idx = agent_ids.index(agent_id)
                target_selection[idx] = 1.0
        
        # Normalize target
        if target_selection.sum() > 0:
            target_selection = target_selection / target_selection.sum()
        
        # Store experience
        experience = {
            'market_state': market_state,
            'target_selection': target_selection,
            'performance_outcome': performance_outcome,
            'timestamp': datetime.now()
        }
        self.meta_experiences.append(experience)
        
        # Train if enough experiences
        if len(self.meta_experiences) >= 32:
            self._train_meta_network()
    
    def _train_meta_network(self):
        """Train the meta-learning network"""
        if len(self.meta_experiences) < 32:
            return
        
        # Sample batch
        batch_size = min(32, len(self.meta_experiences))
        batch_indices = np.random.choice(len(self.meta_experiences), batch_size, replace=False)
        
        batch_states = []
        batch_targets = []
        
        for idx in batch_indices:
            exp = self.meta_experiences[idx]
            batch_states.append(exp['market_state'])
            batch_targets.append(exp['target_selection'])
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(batch_states))
        targets = torch.FloatTensor(np.array(batch_targets))
        
        # Forward pass
        agent_weights, resource_weights = self.meta_network(states)
        
        # Calculate loss
        loss = nn.MSELoss()(agent_weights, targets)
        
        # Backward pass
        self.meta_optimizer.zero_grad()
        loss.backward()
        self.meta_optimizer.step()
        
        logger.debug(f"Meta-learning loss: {loss.item():.4f}")
    
    def get_agent_recommendations(self, market_state: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Get agent selection and resource allocation recommendations"""
        if not self.meta_learning_enabled:
            # Equal weights for all agents
            num_agents = len(self.agents)
            equal_weights = np.ones(num_agents) / num_agents if num_agents > 0 else np.array([])
            return equal_weights, equal_weights
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(market_state).unsqueeze(0)
            agent_weights, resource_weights = self.meta_network(state_tensor)
            
            return agent_weights.squeeze().numpy(), resource_weights.squeeze().numpy()
    
    def coordinate_trading(self, market_data: Dict[str, Any]) -> List[TradeSignal]:
        """Main coordination method - orchestrates the entire trading process"""
        try:
            # 1. Collect signals from all active agents
            raw_signals = self.collect_signals(market_data)
            
            if not raw_signals:
                return []
            
            # 2. Resolve conflicts between competing signals
            resolved_signals = self.resolve_conflicts(raw_signals)
            
            # 3. Allocate resources for approved signals
            final_signals = self.allocate_resources(resolved_signals)
            
            # 4. Log coordination results
            logger.info(f"Coordination: {len(raw_signals)} raw → {len(resolved_signals)} resolved → {len(final_signals)} final")
            
            return final_signals
            
        except Exception as e:
            logger.error(f"Error in trading coordination: {e}")
            return []
    
    def update_agent_performance(self, agent_id: str, trade_pnl: float, trade_duration: float):
        """Update performance metrics for a specific agent"""
        if agent_id in self.agents:
            self.agents[agent_id].update_performance(trade_pnl, trade_duration)
            
            # Update resource allocation based on performance
            self._rebalance_resources()
    
    def _rebalance_resources(self):
        """Rebalance resource allocation based on agent performance"""
        if not self.agents:
            return
        
        # Calculate performance scores
        performance_scores = {}
        total_score = 0
        
        for agent_id, agent in self.agents.items():
            # Use a combination of win rate and Sharpe ratio
            perf = agent.performance
            score = (perf.win_rate * 0.6 + max(0, perf.sharpe_ratio / 3.0) * 0.4)
            performance_scores[agent_id] = max(0.1, score)  # Minimum allocation
            total_score += performance_scores[agent_id]
        
        # Rebalance capital allocation
        if total_score > 0:
            for agent_id, resource in self.agent_resources.items():
                if agent_id in performance_scores:
                    new_allocation = (performance_scores[agent_id] / total_score) * self.total_capital
                    
                    # Gradual rebalancing (don't change too quickly)
                    current_allocation = resource.allocated_capital
                    adjusted_allocation = current_allocation * 0.8 + new_allocation * 0.2
                    
                    resource.allocated_capital = adjusted_allocation
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        status = {
            'coordinator': {
                'is_running': self.is_running,
                'total_agents': len(self.agents),
                'active_agents': sum(1 for a in self.agents.values() if a.status == AgentStatus.ACTIVE),
                'total_capital': self.total_capital,
                'conflict_resolution': self.conflict_resolution.value,
                'meta_learning_enabled': self.meta_learning_enabled
            },
            'agents': {},
            'resources': {},
            'performance': {}
        }
        
        for agent_id, agent in self.agents.items():
            status['agents'][agent_id] = {
                'status': agent.status.value,
                'strategy_type': agent.strategy_type,
                'active_positions': len(agent.active_positions)
            }
            
            if agent_id in self.agent_resources:
                resource = self.agent_resources[agent_id]
                status['resources'][agent_id] = {
                    'allocated_capital': resource.allocated_capital,
                    'used_capital': resource.used_capital,
                    'available_capital': resource.available_capital,
                    'capital_utilization': resource.capital_utilization,
                    'position_utilization': resource.position_utilization
                }
            
            status['performance'][agent_id] = asdict(agent.performance)
        
        return status
    
    def save_state(self, filepath: str = None):
        """Save coordinator state to file"""
        if filepath is None:
            filepath = os.path.join(self.model_dir, f"coordinator_state_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl")
        
        state = {
            'agent_performances': {aid: asdict(perf) for aid, perf in self.agent_performances.items()},
            'agent_resources': {aid: asdict(res) for aid, res in self.agent_resources.items()},
            'performance_history': dict(self.performance_history),
            'trade_history': self.trade_history,
            'meta_experiences': list(self.meta_experiences)
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
        
        # Save meta-learning model
        if self.meta_learning_enabled:
            model_path = os.path.join(self.model_dir, "meta_network.pth")
            torch.save(self.meta_network.state_dict(), model_path)
        
        logger.info(f"Coordinator state saved to {filepath}")
    
    def load_state(self, filepath: str):
        """Load coordinator state from file"""
        try:
            with open(filepath, 'rb') as f:
                state = pickle.load(f)
            
            # Restore state
            self.performance_history = defaultdict(list, state.get('performance_history', {}))
            self.trade_history = state.get('trade_history', [])
            self.meta_experiences = deque(state.get('meta_experiences', []), maxlen=10000)
            
            # Load meta-learning model
            if self.meta_learning_enabled:
                model_path = os.path.join(self.model_dir, "meta_network.pth")
                if os.path.exists(model_path):
                    self.meta_network.load_state_dict(torch.load(model_path))
            
            logger.info(f"Coordinator state loaded from {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to load coordinator state: {e}")
    
    def shutdown(self):
        """Gracefully shutdown the coordinator"""
        logger.info("Shutting down MultiAgentCoordinator...")
        
        # Stop all agents
        self.stop_all_agents()
        
        # Save state
        self.save_state()
        
        # Shutdown executor
        self.executor.shutdown(wait=True)
        
        logger.info("MultiAgentCoordinator shutdown complete")


# Example specialized agent implementations
class MomentumAgent(TradingAgent):
    """Momentum-based trading agent"""
    
    def __init__(self, agent_id: str = "momentum_agent"):
        super().__init__(agent_id, "momentum")
    
    def generate_signal(self, market_data: Dict[str, Any]) -> Optional[TradeSignal]:
        """Generate momentum-based trading signal"""
        try:
            df = market_data.get('price_data')
            symbol = market_data.get('symbol', 'UNKNOWN')
            
            if df is None or len(df) < 20:
                return None
            
            # Simple momentum calculation
            returns = df['close'].pct_change(10).iloc[-1]
            volume_ratio = df['volume'].iloc[-1] / df['volume'].rolling(20).mean().iloc[-1]
            
            confidence = min(0.9, abs(returns) * 10 + (volume_ratio - 1) * 0.2)
            
            if returns > 0.02 and volume_ratio > 1.2:
                return TradeSignal(
                    agent_id=self.agent_id,
                    symbol=symbol,
                    signal_type=TradeSignalType.BUY,
                    confidence=confidence,
                    quantity=100,
                    price=df['close'].iloc[-1],
                    reasoning="Strong momentum with volume confirmation"
                )
            elif returns < -0.02 and volume_ratio > 1.2:
                return TradeSignal(
                    agent_id=self.agent_id,
                    symbol=symbol,
                    signal_type=TradeSignalType.SELL,
                    confidence=confidence,
                    quantity=100,
                    price=df['close'].iloc[-1],
                    reasoning="Negative momentum with volume confirmation"
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error in momentum agent signal generation: {e}")
            return None


class MeanReversionAgent(TradingAgent):
    """Mean reversion trading agent"""
    
    def __init__(self, agent_id: str = "mean_reversion_agent"):
        super().__init__(agent_id, "mean_reversion")
    
    def generate_signal(self, market_data: Dict[str, Any]) -> Optional[TradeSignal]:
        """Generate mean reversion trading signal"""
        try:
            df = market_data.get('price_data')
            symbol = market_data.get('symbol', 'UNKNOWN')
            
            if df is None or len(df) < 20:
                return None
            
            # Calculate RSI and Bollinger Bands
            rsi = df.get('rsi', pd.Series([50])).iloc[-1]
            bb_position = 0.5  # Simplified
            
            current_price = df['close'].iloc[-1]
            sma_20 = df['close'].rolling(20).mean().iloc[-1]
            deviation = (current_price - sma_20) / sma_20
            
            confidence = min(0.9, abs(deviation) * 5)
            
            if rsi < 30 and deviation < -0.03:
                return TradeSignal(
                    agent_id=self.agent_id,
                    symbol=symbol,
                    signal_type=TradeSignalType.BUY,
                    confidence=confidence,
                    quantity=100,
                    price=current_price,
                    reasoning="Oversold condition for mean reversion"
                )
            elif rsi > 70 and deviation > 0.03:
                return TradeSignal(
                    agent_id=self.agent_id,
                    symbol=symbol,
                    signal_type=TradeSignalType.SELL,
                    confidence=confidence,
                    quantity=100,
                    price=current_price,
                    reasoning="Overbought condition for mean reversion"
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error in mean reversion agent signal generation: {e}")
            return None


def create_default_agents() -> List[TradingAgent]:
    """Create a set of default trading agents"""
    agents = [
        MomentumAgent("momentum_agent"),
        MeanReversionAgent("mean_reversion_agent"),
        TradingAgent("breakout_agent", "breakout"),
        TradingAgent("trend_following_agent", "trend_following"),
        TradingAgent("scalping_agent", "scalping"),
        TradingAgent("arbitrage_agent", "arbitrage")
    ]
    
    return agents


if __name__ == "__main__":
    # Example usage
    coordinator = MultiAgentCoordinator(
        total_capital=1000000.0,
        max_agents=6,
        conflict_resolution=ConflictResolution.WEIGHTED_VOTE,
        meta_learning_enabled=True
    )
    
    # Register agents
    agents = create_default_agents()
    for agent in agents:
        coordinator.register_agent(agent, initial_capital=150000.0)
    
    # Start coordination
    coordinator.start_all_agents()
    
    # Example market data
    market_data = {
        'symbol': 'RELIANCE',
        'price_data': pd.DataFrame({
            'close': np.random.randn(100).cumsum() + 2000,
            'volume': np.random.randint(1000, 10000, 100),
            'rsi': np.random.uniform(20, 80, 100)
        })
    }
    
    # Coordinate trading
    signals = coordinator.coordinate_trading(market_data)
    print(f"Generated {len(signals)} coordinated signals")
    
    # Print system status
    status = coordinator.get_system_status()
    print(f"System status: {status['coordinator']}")
    
    # Shutdown
    coordinator.shutdown()