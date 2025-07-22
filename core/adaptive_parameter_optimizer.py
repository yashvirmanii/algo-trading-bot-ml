"""
Adaptive Parameter Optimizer for Trading Strategies.

This module implements an advanced parameter optimization system that:
- Automatically adjusts strategy parameters based on recent performance
- Uses Bayesian optimization to find optimal parameter ranges
- Implements parameter bounds and constraints for each strategy
- Tracks parameter performance over rolling windows
- Reverts parameters if performance degrades
- Includes gradual adjustment and stability checks
"""

import logging
import os
import json
import pickle
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Any, Callable, Union
from collections import deque, defaultdict
import numpy as np
import pandas as pd
from scipy import stats
from skopt import gp_minimize, forest_minimize, gbrt_minimize
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args
from skopt.acquisition import gaussian_ei
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class ParameterBounds:
    """Parameter bounds and constraints definition."""
    name: str
    param_type: str  # 'real', 'integer', 'categorical'
    bounds: Union[Tuple[float, float], List[Any]]  # (min, max) for real/int, [options] for categorical
    default_value: Any
    description: str
    constraints: Optional[Dict[str, Any]] = None  # Additional constraints
    stability_threshold: float = 0.1  # Maximum change per adjustment (for real/int)
    
    def create_skopt_dimension(self):
        """Create scikit-optimize dimension object."""
        if self.param_type == 'real':
            return Real(self.bounds[0], self.bounds[1], name=self.name)
        elif self.param_type == 'integer':
            return Integer(self.bounds[0], self.bounds[1], name=self.name)
        elif self.param_type == 'categorical':
            return Categorical(self.bounds, name=self.name)
        else:
            raise ValueError(f"Unknown parameter type: {self.param_type}")


@dataclass
class ParameterSet:
    """A set of parameters for a strategy."""
    strategy_name: str
    parameters: Dict[str, Any]
    timestamp: datetime
    performance_score: Optional[float] = None
    trades_count: int = 0
    win_rate: float = 0.0
    avg_return: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    stability_score: float = 1.0  # How stable these parameters are
    
    def calculate_composite_score(self, weights: Dict[str, float] = None) -> float:
        """Calculate composite performance score."""
        if weights is None:
            weights = {
                'win_rate': 0.3,
                'avg_return': 0.3,
                'sharpe_ratio': 0.2,
                'max_drawdown': -0.1,  # Negative because lower is better
                'stability_score': 0.1
            }
        
        score = (
            self.win_rate * weights['win_rate'] +
            self.avg_return * weights['avg_return'] +
            self.sharpe_ratio * weights['sharpe_ratio'] +
            self.max_drawdown * weights['max_drawdown'] +
            self.stability_score * weights['stability_score']
        )
        
        return score


@dataclass
class OptimizationResult:
    """Result of parameter optimization."""
    strategy_name: str
    old_parameters: Dict[str, Any]
    new_parameters: Dict[str, Any]
    expected_improvement: float
    confidence_score: float
    optimization_method: str
    timestamp: datetime
    rollback_available: bool = True
    
    def get_parameter_changes(self) -> Dict[str, Dict[str, Any]]:
        """Get detailed parameter changes."""
        changes = {}
        
        for param_name in set(list(self.old_parameters.keys()) + list(self.new_parameters.keys())):
            old_val = self.old_parameters.get(param_name)
            new_val = self.new_parameters.get(param_name)
            
            if old_val != new_val:
                changes[param_name] = {
                    'old_value': old_val,
                    'new_value': new_val,
                    'change_pct': self._calculate_change_percentage(old_val, new_val)
                }
        
        return changes
    
    def _calculate_change_percentage(self, old_val: Any, new_val: Any) -> Optional[float]:
        """Calculate percentage change for numeric values."""
        try:
            if isinstance(old_val, (int, float)) and isinstance(new_val, (int, float)):
                if old_val != 0:
                    return ((new_val - old_val) / old_val) * 100
                else:
                    return float('inf') if new_val != 0 else 0.0
        except:
            pass
        return None


class PerformanceTracker:
    """Tracks strategy performance over rolling windows."""
    
    def __init__(self, window_size: int = 50):
        self.window_size = window_size
        self.performance_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=window_size))
        self.trade_history: Dict[str, List[Dict]] = defaultdict(list)
    
    def add_trade_result(self, strategy_name: str, trade_result: Dict[str, Any]):
        """Add a trade result for performance tracking."""
        self.trade_history[strategy_name].append({
            **trade_result,
            'timestamp': datetime.now()
        })
        
        # Keep only recent trades
        cutoff_date = datetime.now() - timedelta(days=30)
        self.trade_history[strategy_name] = [
            trade for trade in self.trade_history[strategy_name]
            if trade['timestamp'] > cutoff_date
        ]
    
    def calculate_performance_metrics(self, strategy_name: str, 
                                    lookback_days: int = 7) -> Dict[str, float]:
        """Calculate performance metrics for a strategy."""
        cutoff_date = datetime.now() - timedelta(days=lookback_days)
        recent_trades = [
            trade for trade in self.trade_history[strategy_name]
            if trade['timestamp'] > cutoff_date
        ]
        
        if not recent_trades:
            return {
                'trades_count': 0,
                'win_rate': 0.0,
                'avg_return': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'total_return': 0.0
            }
        
        # Calculate metrics
        returns = [trade.get('return_pct', 0.0) for trade in recent_trades]
        wins = [r for r in returns if r > 0]
        
        win_rate = len(wins) / len(returns) if returns else 0.0
        avg_return = np.mean(returns) if returns else 0.0
        
        # Sharpe ratio (assuming daily returns)
        if len(returns) > 1 and np.std(returns) > 0:
            sharpe_ratio = (avg_return / np.std(returns)) * np.sqrt(252)  # Annualized
        else:
            sharpe_ratio = 0.0
        
        # Maximum drawdown
        cumulative_returns = np.cumsum(returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = cumulative_returns - running_max
        max_drawdown = np.min(drawdowns) if len(drawdowns) > 0 else 0.0
        
        total_return = np.sum(returns)
        
        return {
            'trades_count': len(recent_trades),
            'win_rate': win_rate,
            'avg_return': avg_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'total_return': total_return
        }


class AdaptiveParameterOptimizer:
    """
    Advanced parameter optimizer using Bayesian optimization.
    
    Automatically adjusts strategy parameters based on performance feedback
    with stability checks and rollback functionality.
    """
    
    def __init__(self, 
                 data_dir: str = "data",
                 optimization_frequency: int = 7,  # days
                 min_trades_for_optimization: int = 10,
                 performance_window: int = 14,  # days
                 stability_threshold: float = 0.15,
                 confidence_threshold: float = 0.6):
        """
        Initialize the adaptive parameter optimizer.
        
        Args:
            data_dir: Directory for data storage
            optimization_frequency: How often to run optimization (days)
            min_trades_for_optimization: Minimum trades needed before optimization
            performance_window: Window for performance evaluation (days)
            stability_threshold: Maximum parameter change per optimization
            confidence_threshold: Minimum confidence for parameter changes
        """
        self.data_dir = data_dir
        self.optimization_frequency = optimization_frequency
        self.min_trades_for_optimization = min_trades_for_optimization
        self.performance_window = performance_window
        self.stability_threshold = stability_threshold
        self.confidence_threshold = confidence_threshold
        
        # Strategy configurations
        self.strategy_configs: Dict[str, Dict[str, ParameterBounds]] = {}
        self.current_parameters: Dict[str, Dict[str, Any]] = {}
        self.parameter_history: Dict[str, List[ParameterSet]] = defaultdict(list)
        self.optimization_history: Dict[str, List[OptimizationResult]] = defaultdict(list)
        
        # Performance tracking
        self.performance_tracker = PerformanceTracker()
        
        # Optimization state
        self.last_optimization: Dict[str, datetime] = {}
        self.optimization_in_progress: Dict[str, bool] = defaultdict(bool)
        
        # File paths
        self.configs_file = os.path.join(data_dir, "strategy_configs.json")
        self.parameters_file = os.path.join(data_dir, "current_parameters.json")
        self.history_file = os.path.join(data_dir, "parameter_history.pkl")
        self.optimization_file = os.path.join(data_dir, "optimization_history.pkl")
        
        # Ensure data directory exists
        os.makedirs(data_dir, exist_ok=True)
        
        # Load existing data
        self._load_configurations()
        
        logger.info(f"AdaptiveParameterOptimizer initialized for {len(self.strategy_configs)} strategies")
    
    def register_strategy(self, 
                         strategy_name: str, 
                         parameter_bounds: List[ParameterBounds],
                         initial_parameters: Optional[Dict[str, Any]] = None):
        """
        Register a strategy with its parameter bounds.
        
        Args:
            strategy_name: Name of the strategy
            parameter_bounds: List of parameter bounds definitions
            initial_parameters: Initial parameter values (optional)
        """
        try:
            # Store parameter bounds
            self.strategy_configs[strategy_name] = {
                bound.name: bound for bound in parameter_bounds
            }
            
            # Set initial parameters
            if initial_parameters:
                self.current_parameters[strategy_name] = initial_parameters.copy()
            else:
                # Use default values from bounds
                self.current_parameters[strategy_name] = {
                    bound.name: bound.default_value 
                    for bound in parameter_bounds
                }
            
            # Initialize optimization tracking
            self.last_optimization[strategy_name] = datetime.now()
            
            # Save configurations
            self._save_configurations()
            
            logger.info(f"Registered strategy '{strategy_name}' with {len(parameter_bounds)} parameters")
            
        except Exception as e:
            logger.error(f"Error registering strategy {strategy_name}: {e}")
    
    def add_trade_result(self, strategy_name: str, trade_result: Dict[str, Any]):
        """
        Add a trade result for performance tracking.
        
        Args:
            strategy_name: Name of the strategy
            trade_result: Dictionary containing trade results
                         Must include: 'return_pct', 'win' (bool), 'pnl'
        """
        try:
            if strategy_name not in self.strategy_configs:
                logger.warning(f"Strategy {strategy_name} not registered")
                return
            
            # Add to performance tracker
            self.performance_tracker.add_trade_result(strategy_name, trade_result)
            
            # Check if optimization is needed
            self._check_optimization_trigger(strategy_name)
            
        except Exception as e:
            logger.error(f"Error adding trade result for {strategy_name}: {e}")
    
    def optimize_strategy_parameters(self, 
                                   strategy_name: str,
                                   force_optimization: bool = False) -> Optional[OptimizationResult]:
        """
        Optimize parameters for a specific strategy.
        
        Args:
            strategy_name: Name of the strategy to optimize
            force_optimization: Force optimization even if conditions not met
            
        Returns:
            OptimizationResult if optimization was performed, None otherwise
        """
        try:
            if strategy_name not in self.strategy_configs:
                logger.error(f"Strategy {strategy_name} not registered")
                return None
            
            if self.optimization_in_progress[strategy_name]:
                logger.info(f"Optimization already in progress for {strategy_name}")
                return None
            
            # Check if optimization is needed
            if not force_optimization and not self._should_optimize(strategy_name):
                return None
            
            logger.info(f"Starting parameter optimization for {strategy_name}")
            self.optimization_in_progress[strategy_name] = True
            
            try:
                # Get current performance metrics
                current_metrics = self.performance_tracker.calculate_performance_metrics(
                    strategy_name, self.performance_window
                )
                
                if current_metrics['trades_count'] < self.min_trades_for_optimization:
                    logger.info(f"Insufficient trades for {strategy_name} optimization")
                    return None
                
                # Perform Bayesian optimization
                optimization_result = self._run_bayesian_optimization(strategy_name, current_metrics)
                
                if optimization_result:
                    # Apply gradual parameter adjustment
                    adjusted_result = self._apply_gradual_adjustment(optimization_result)
                    
                    # Update parameters if confidence is sufficient
                    if adjusted_result.confidence_score >= self.confidence_threshold:
                        self._update_strategy_parameters(adjusted_result)
                        logger.info(f"Updated parameters for {strategy_name}")
                    else:
                        logger.info(f"Optimization confidence too low for {strategy_name}: {adjusted_result.confidence_score:.3f}")
                
                return optimization_result
                
            finally:
                self.optimization_in_progress[strategy_name] = False
                self.last_optimization[strategy_name] = datetime.now()
            
        except Exception as e:
            logger.error(f"Error optimizing parameters for {strategy_name}: {e}")
            self.optimization_in_progress[strategy_name] = False
            return None   
 def _run_bayesian_optimization(self, 
                                  strategy_name: str, 
                                  current_metrics: Dict[str, float]) -> Optional[OptimizationResult]:
        """Run Bayesian optimization to find better parameters."""
        try:
            bounds_config = self.strategy_configs[strategy_name]
            current_params = self.current_parameters[strategy_name]
            
            # Create optimization dimensions
            dimensions = []
            param_names = []
            
            for param_name, bounds in bounds_config.items():
                dimensions.append(bounds.create_skopt_dimension())
                param_names.append(param_name)
            
            # Define objective function
            @use_named_args(dimensions)
            def objective(**params):
                return -self._evaluate_parameter_set(strategy_name, params, current_metrics)
            
            # Get historical parameter sets for initial points
            initial_points = self._get_historical_parameter_points(strategy_name, param_names)
            
            # Choose optimization method based on parameter count
            n_params = len(dimensions)
            n_calls = min(50, max(20, n_params * 3))  # Adaptive number of calls
            
            if n_params <= 5:
                # Use Gaussian Process for small parameter spaces
                result = gp_minimize(
                    func=objective,
                    dimensions=dimensions,
                    n_calls=n_calls,
                    n_initial_points=min(10, len(initial_points) + 5),
                    initial_point_generator='random',
                    acq_func='EI',  # Expected Improvement
                    random_state=42
                )
                method = "Gaussian Process"
            elif n_params <= 20:
                # Use Random Forest for medium parameter spaces
                result = forest_minimize(
                    func=objective,
                    dimensions=dimensions,
                    n_calls=n_calls,
                    n_initial_points=min(15, len(initial_points) + 8),
                    random_state=42
                )
                method = "Random Forest"
            else:
                # Use Gradient Boosting for large parameter spaces
                result = gbrt_minimize(
                    func=objective,
                    dimensions=dimensions,
                    n_calls=n_calls,
                    n_initial_points=min(20, len(initial_points) + 10),
                    random_state=42
                )
                method = "Gradient Boosting"
            
            # Extract optimal parameters
            optimal_params = {}
            for i, param_name in enumerate(param_names):
                optimal_params[param_name] = result.x[i]
            
            # Calculate expected improvement
            current_score = self._evaluate_parameter_set(strategy_name, current_params, current_metrics)
            optimal_score = self._evaluate_parameter_set(strategy_name, optimal_params, current_metrics)
            expected_improvement = optimal_score - current_score
            
            # Calculate confidence score based on optimization convergence
            confidence_score = self._calculate_optimization_confidence(result, expected_improvement)
            
            optimization_result = OptimizationResult(
                strategy_name=strategy_name,
                old_parameters=current_params.copy(),
                new_parameters=optimal_params,
                expected_improvement=expected_improvement,
                confidence_score=confidence_score,
                optimization_method=method,
                timestamp=datetime.now()
            )
            
            # Store optimization result
            self.optimization_history[strategy_name].append(optimization_result)
            self._save_optimization_history()
            
            logger.info(f"Bayesian optimization completed for {strategy_name}: "
                       f"Expected improvement: {expected_improvement:.4f}, "
                       f"Confidence: {confidence_score:.3f}")
            
            return optimization_result
            
        except Exception as e:
            logger.error(f"Error in Bayesian optimization for {strategy_name}: {e}")
            return None
    
    def _evaluate_parameter_set(self, 
                               strategy_name: str, 
                               parameters: Dict[str, Any],
                               baseline_metrics: Dict[str, float]) -> float:
        """
        Evaluate a parameter set's expected performance.
        
        This is a simplified evaluation - in practice, you might want to
        run backtests or use more sophisticated performance prediction.
        """
        try:
            # Get historical performance for similar parameter sets
            similar_performance = self._find_similar_parameter_performance(
                strategy_name, parameters
            )
            
            if similar_performance:
                # Use historical performance as basis
                base_score = np.mean([perf['composite_score'] for perf in similar_performance])
            else:
                # Use current baseline performance
                base_score = self._calculate_composite_score(baseline_metrics)
            
            # Apply parameter-specific adjustments
            adjustment_factor = self._calculate_parameter_adjustment_factor(
                strategy_name, parameters
            )
            
            # Add some noise to prevent overfitting
            noise = np.random.normal(0, 0.01)
            
            final_score = base_score * adjustment_factor + noise
            
            return final_score
            
        except Exception as e:
            logger.error(f"Error evaluating parameter set: {e}")
            return 0.0
    
    def _calculate_composite_score(self, metrics: Dict[str, float]) -> float:
        """Calculate composite performance score from metrics."""
        weights = {
            'win_rate': 0.3,
            'avg_return': 0.3,
            'sharpe_ratio': 0.2,
            'max_drawdown': -0.1,  # Negative because lower is better
        }
        
        score = 0.0
        for metric, weight in weights.items():
            if metric in metrics:
                if metric == 'max_drawdown':
                    # For drawdown, lower (more negative) is worse
                    score += metrics[metric] * weight
                else:
                    score += metrics[metric] * weight
        
        return score
    
    def _calculate_parameter_adjustment_factor(self, 
                                             strategy_name: str, 
                                             parameters: Dict[str, Any]) -> float:
        """
        Calculate adjustment factor based on parameter values.
        
        This provides domain knowledge about how parameters might affect performance.
        """
        try:
            factor = 1.0
            bounds_config = self.strategy_configs[strategy_name]
            
            # Apply parameter-specific logic
            for param_name, value in parameters.items():
                if param_name not in bounds_config:
                    continue
                
                bounds = bounds_config[param_name]
                
                # Example adjustments (customize based on your strategies)
                if 'stop_loss' in param_name.lower():
                    # Tighter stops might reduce drawdown but also reduce profits
                    if bounds.param_type in ['real', 'integer']:
                        normalized_value = (value - bounds.bounds[0]) / (bounds.bounds[1] - bounds.bounds[0])
                        # Prefer moderate stop losses
                        factor *= 1.0 - 0.1 * abs(normalized_value - 0.5)
                
                elif 'take_profit' in param_name.lower():
                    # Higher take profits might increase average returns
                    if bounds.param_type in ['real', 'integer']:
                        normalized_value = (value - bounds.bounds[0]) / (bounds.bounds[1] - bounds.bounds[0])
                        factor *= 1.0 + 0.05 * normalized_value
                
                elif 'period' in param_name.lower() or 'window' in param_name.lower():
                    # Moderate periods often work better than extremes
                    if bounds.param_type in ['real', 'integer']:
                        normalized_value = (value - bounds.bounds[0]) / (bounds.bounds[1] - bounds.bounds[0])
                        factor *= 1.0 - 0.05 * abs(normalized_value - 0.5)
            
            return max(0.5, min(1.5, factor))  # Clamp to reasonable range
            
        except Exception as e:
            logger.error(f"Error calculating parameter adjustment factor: {e}")
            return 1.0
    
    def _find_similar_parameter_performance(self, 
                                          strategy_name: str, 
                                          parameters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Find historical performance for similar parameter sets."""
        try:
            if strategy_name not in self.parameter_history:
                return []
            
            similar_performance = []
            
            for param_set in self.parameter_history[strategy_name]:
                similarity = self._calculate_parameter_similarity(
                    parameters, param_set.parameters
                )
                
                if similarity > 0.8 and param_set.performance_score is not None:
                    similar_performance.append({
                        'composite_score': param_set.performance_score,
                        'similarity': similarity,
                        'trades_count': param_set.trades_count
                    })
            
            # Sort by similarity and return top matches
            similar_performance.sort(key=lambda x: x['similarity'], reverse=True)
            return similar_performance[:5]
            
        except Exception as e:
            logger.error(f"Error finding similar parameter performance: {e}")
            return []
    
    def _calculate_parameter_similarity(self, 
                                      params1: Dict[str, Any], 
                                      params2: Dict[str, Any]) -> float:
        """Calculate similarity between two parameter sets."""
        try:
            if not params1 or not params2:
                return 0.0
            
            common_params = set(params1.keys()) & set(params2.keys())
            if not common_params:
                return 0.0
            
            similarities = []
            
            for param_name in common_params:
                val1, val2 = params1[param_name], params2[param_name]
                
                if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                    # Numeric similarity
                    if val1 == val2:
                        sim = 1.0
                    else:
                        max_val = max(abs(val1), abs(val2))
                        if max_val > 0:
                            sim = 1.0 - abs(val1 - val2) / max_val
                        else:
                            sim = 1.0
                else:
                    # Categorical similarity
                    sim = 1.0 if val1 == val2 else 0.0
                
                similarities.append(sim)
            
            return np.mean(similarities)
            
        except Exception as e:
            logger.error(f"Error calculating parameter similarity: {e}")
            return 0.0
    
    def _calculate_optimization_confidence(self, 
                                         optimization_result: Any, 
                                         expected_improvement: float) -> float:
        """Calculate confidence in optimization result."""
        try:
            confidence = 0.5  # Base confidence
            
            # Factor 1: Convergence quality
            if hasattr(optimization_result, 'func_vals'):
                func_vals = optimization_result.func_vals
                if len(func_vals) > 5:
                    # Check if optimization converged (values stabilized)
                    recent_vals = func_vals[-5:]
                    if np.std(recent_vals) < 0.01:  # Low variance = good convergence
                        confidence += 0.2
            
            # Factor 2: Expected improvement magnitude
            if expected_improvement > 0.05:  # 5% improvement
                confidence += 0.2
            elif expected_improvement > 0.02:  # 2% improvement
                confidence += 0.1
            
            # Factor 3: Number of optimization calls
            if hasattr(optimization_result, 'x_iters'):
                n_calls = len(optimization_result.x_iters)
                if n_calls >= 30:
                    confidence += 0.1
            
            return min(1.0, confidence)
            
        except Exception as e:
            logger.error(f"Error calculating optimization confidence: {e}")
            return 0.5
    
    def _apply_gradual_adjustment(self, optimization_result: OptimizationResult) -> OptimizationResult:
        """Apply gradual parameter adjustment to avoid sudden jumps."""
        try:
            strategy_name = optimization_result.strategy_name
            bounds_config = self.strategy_configs[strategy_name]
            
            adjusted_params = {}
            
            for param_name, new_value in optimization_result.new_parameters.items():
                old_value = optimization_result.old_parameters.get(param_name, new_value)
                
                if param_name in bounds_config:
                    bounds = bounds_config[param_name]
                    
                    if bounds.param_type in ['real', 'integer']:
                        # Calculate maximum allowed change
                        param_range = bounds.bounds[1] - bounds.bounds[0]
                        max_change = param_range * bounds.stability_threshold
                        
                        # Limit the change
                        change = new_value - old_value
                        if abs(change) > max_change:
                            change = np.sign(change) * max_change
                        
                        adjusted_value = old_value + change
                        
                        # Ensure within bounds
                        adjusted_value = max(bounds.bounds[0], 
                                           min(bounds.bounds[1], adjusted_value))
                        
                        adjusted_params[param_name] = adjusted_value
                    else:
                        # For categorical parameters, use original optimization result
                        adjusted_params[param_name] = new_value
                else:
                    adjusted_params[param_name] = new_value
            
            # Create adjusted optimization result
            adjusted_result = OptimizationResult(
                strategy_name=optimization_result.strategy_name,
                old_parameters=optimization_result.old_parameters,
                new_parameters=adjusted_params,
                expected_improvement=optimization_result.expected_improvement * 0.8,  # Reduce due to gradual adjustment
                confidence_score=optimization_result.confidence_score,
                optimization_method=optimization_result.optimization_method + " (Gradual)",
                timestamp=optimization_result.timestamp
            )
            
            return adjusted_result
            
        except Exception as e:
            logger.error(f"Error applying gradual adjustment: {e}")
            return optimization_result
    
    def _update_strategy_parameters(self, optimization_result: OptimizationResult):
        """Update strategy parameters with optimization result."""
        try:
            strategy_name = optimization_result.strategy_name
            
            # Update current parameters
            self.current_parameters[strategy_name] = optimization_result.new_parameters.copy()
            
            # Create parameter set record
            current_metrics = self.performance_tracker.calculate_performance_metrics(
                strategy_name, self.performance_window
            )
            
            param_set = ParameterSet(
                strategy_name=strategy_name,
                parameters=optimization_result.new_parameters.copy(),
                timestamp=datetime.now(),
                trades_count=current_metrics['trades_count'],
                win_rate=current_metrics['win_rate'],
                avg_return=current_metrics['avg_return'],
                sharpe_ratio=current_metrics['sharpe_ratio'],
                max_drawdown=current_metrics['max_drawdown']
            )
            
            param_set.performance_score = param_set.calculate_composite_score()
            
            # Add to parameter history
            self.parameter_history[strategy_name].append(param_set)
            
            # Keep only recent parameter sets (last 100)
            if len(self.parameter_history[strategy_name]) > 100:
                self.parameter_history[strategy_name] = self.parameter_history[strategy_name][-100:]
            
            # Save updated parameters
            self._save_configurations()
            self._save_parameter_history()
            
            logger.info(f"Updated parameters for {strategy_name}")
            
        except Exception as e:
            logger.error(f"Error updating strategy parameters: {e}")
    
    def _get_historical_parameter_points(self, 
                                       strategy_name: str, 
                                       param_names: List[str]) -> List[List[Any]]:
        """Get historical parameter points for optimization initialization."""
        try:
            if strategy_name not in self.parameter_history:
                return []
            
            points = []
            for param_set in self.parameter_history[strategy_name][-20:]:  # Last 20 sets
                point = []
                for param_name in param_names:
                    if param_name in param_set.parameters:
                        point.append(param_set.parameters[param_name])
                    else:
                        # Use default value if parameter not found
                        bounds = self.strategy_configs[strategy_name][param_name]
                        point.append(bounds.default_value)
                
                if len(point) == len(param_names):
                    points.append(point)
            
            return points
            
        except Exception as e:
            logger.error(f"Error getting historical parameter points: {e}")
            return []    de
f _should_optimize(self, strategy_name: str) -> bool:
        """Check if strategy should be optimized."""
        try:
            # Check if enough time has passed
            last_opt = self.last_optimization.get(strategy_name)
            if last_opt:
                days_since_last = (datetime.now() - last_opt).days
                if days_since_last < self.optimization_frequency:
                    return False
            
            # Check if enough trades have occurred
            metrics = self.performance_tracker.calculate_performance_metrics(
                strategy_name, self.performance_window
            )
            
            if metrics['trades_count'] < self.min_trades_for_optimization:
                return False
            
            # Check if performance is declining
            recent_performance = self._get_recent_performance_trend(strategy_name)
            if recent_performance and recent_performance < -0.02:  # 2% decline
                logger.info(f"Performance declining for {strategy_name}, triggering optimization")
                return True
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking optimization trigger: {e}")
            return False
    
    def _check_optimization_trigger(self, strategy_name: str):
        """Check if optimization should be triggered after new trade."""
        try:
            if self._should_optimize(strategy_name):
                # Run optimization in background (you might want to use threading)
                self.optimize_strategy_parameters(strategy_name)
        except Exception as e:
            logger.error(f"Error checking optimization trigger: {e}")
    
    def _get_recent_performance_trend(self, strategy_name: str) -> Optional[float]:
        """Get recent performance trend for a strategy."""
        try:
            if strategy_name not in self.parameter_history:
                return None
            
            recent_sets = self.parameter_history[strategy_name][-5:]  # Last 5 parameter sets
            if len(recent_sets) < 3:
                return None
            
            scores = [ps.performance_score for ps in recent_sets if ps.performance_score is not None]
            if len(scores) < 3:
                return None
            
            # Calculate trend using linear regression
            x = np.arange(len(scores))
            slope, _, _, _, _ = stats.linregress(x, scores)
            
            return slope
            
        except Exception as e:
            logger.error(f"Error getting performance trend: {e}")
            return None
    
    def rollback_parameters(self, strategy_name: str, steps_back: int = 1) -> bool:
        """
        Rollback parameters to a previous state.
        
        Args:
            strategy_name: Name of the strategy
            steps_back: Number of optimization steps to roll back
            
        Returns:
            True if rollback successful, False otherwise
        """
        try:
            if strategy_name not in self.optimization_history:
                logger.error(f"No optimization history for {strategy_name}")
                return False
            
            if len(self.optimization_history[strategy_name]) < steps_back:
                logger.error(f"Not enough optimization history for rollback")
                return False
            
            # Get the target optimization result
            target_index = -(steps_back + 1)  # Go back to before the last N optimizations
            if abs(target_index) > len(self.optimization_history[strategy_name]):
                target_index = 0  # Go to the beginning
            
            target_optimization = self.optimization_history[strategy_name][target_index]
            rollback_params = target_optimization.old_parameters
            
            # Update current parameters
            self.current_parameters[strategy_name] = rollback_params.copy()
            
            # Create rollback parameter set
            param_set = ParameterSet(
                strategy_name=strategy_name,
                parameters=rollback_params.copy(),
                timestamp=datetime.now(),
                stability_score=1.0  # High stability for rollback
            )
            
            # Add to parameter history
            self.parameter_history[strategy_name].append(param_set)
            
            # Save updated parameters
            self._save_configurations()
            self._save_parameter_history()
            
            logger.info(f"Rolled back parameters for {strategy_name} by {steps_back} steps")
            return True
            
        except Exception as e:
            logger.error(f"Error rolling back parameters for {strategy_name}: {e}")
            return False
    
    def get_current_parameters(self, strategy_name: str) -> Optional[Dict[str, Any]]:
        """Get current parameters for a strategy."""
        return self.current_parameters.get(strategy_name)
    
    def get_parameter_history(self, strategy_name: str, limit: int = 10) -> List[ParameterSet]:
        """Get parameter history for a strategy."""
        if strategy_name not in self.parameter_history:
            return []
        
        return self.parameter_history[strategy_name][-limit:]
    
    def get_optimization_history(self, strategy_name: str, limit: int = 10) -> List[OptimizationResult]:
        """Get optimization history for a strategy."""
        if strategy_name not in self.optimization_history:
            return []
        
        return self.optimization_history[strategy_name][-limit:]
    
    def get_strategy_performance_summary(self, strategy_name: str) -> Dict[str, Any]:
        """Get comprehensive performance summary for a strategy."""
        try:
            if strategy_name not in self.strategy_configs:
                return {"error": "Strategy not registered"}
            
            # Current parameters
            current_params = self.current_parameters.get(strategy_name, {})
            
            # Recent performance metrics
            metrics = self.performance_tracker.calculate_performance_metrics(
                strategy_name, self.performance_window
            )
            
            # Parameter history summary
            param_history = self.parameter_history.get(strategy_name, [])
            optimization_history = self.optimization_history.get(strategy_name, [])
            
            # Performance trend
            trend = self._get_recent_performance_trend(strategy_name)
            
            # Last optimization info
            last_optimization = None
            if optimization_history:
                last_opt = optimization_history[-1]
                last_optimization = {
                    'timestamp': last_opt.timestamp.isoformat(),
                    'expected_improvement': last_opt.expected_improvement,
                    'confidence_score': last_opt.confidence_score,
                    'method': last_opt.optimization_method
                }
            
            return {
                'strategy_name': strategy_name,
                'current_parameters': current_params,
                'performance_metrics': metrics,
                'performance_trend': trend,
                'parameter_sets_count': len(param_history),
                'optimizations_count': len(optimization_history),
                'last_optimization': last_optimization,
                'next_optimization_due': self._get_next_optimization_date(strategy_name).isoformat() if self._get_next_optimization_date(strategy_name) else None
            }
            
        except Exception as e:
            logger.error(f"Error getting performance summary for {strategy_name}: {e}")
            return {"error": str(e)}
    
    def _get_next_optimization_date(self, strategy_name: str) -> Optional[datetime]:
        """Get the next scheduled optimization date."""
        last_opt = self.last_optimization.get(strategy_name)
        if last_opt:
            return last_opt + timedelta(days=self.optimization_frequency)
        return None
    
    def force_parameter_update(self, 
                             strategy_name: str, 
                             new_parameters: Dict[str, Any],
                             reason: str = "Manual update") -> bool:
        """
        Force update parameters without optimization.
        
        Args:
            strategy_name: Name of the strategy
            new_parameters: New parameter values
            reason: Reason for the update
            
        Returns:
            True if update successful, False otherwise
        """
        try:
            if strategy_name not in self.strategy_configs:
                logger.error(f"Strategy {strategy_name} not registered")
                return False
            
            # Validate parameters against bounds
            bounds_config = self.strategy_configs[strategy_name]
            validated_params = {}
            
            for param_name, value in new_parameters.items():
                if param_name in bounds_config:
                    bounds = bounds_config[param_name]
                    
                    # Validate bounds
                    if bounds.param_type in ['real', 'integer']:
                        if not (bounds.bounds[0] <= value <= bounds.bounds[1]):
                            logger.error(f"Parameter {param_name} value {value} out of bounds {bounds.bounds}")
                            return False
                    elif bounds.param_type == 'categorical':
                        if value not in bounds.bounds:
                            logger.error(f"Parameter {param_name} value {value} not in allowed values {bounds.bounds}")
                            return False
                    
                    validated_params[param_name] = value
                else:
                    logger.warning(f"Unknown parameter {param_name} for strategy {strategy_name}")
            
            # Update parameters
            old_params = self.current_parameters.get(strategy_name, {}).copy()
            self.current_parameters[strategy_name].update(validated_params)
            
            # Create parameter set record
            param_set = ParameterSet(
                strategy_name=strategy_name,
                parameters=self.current_parameters[strategy_name].copy(),
                timestamp=datetime.now(),
                stability_score=0.5  # Lower stability for manual updates
            )
            
            # Add to parameter history
            self.parameter_history[strategy_name].append(param_set)
            
            # Create optimization result record for tracking
            optimization_result = OptimizationResult(
                strategy_name=strategy_name,
                old_parameters=old_params,
                new_parameters=validated_params,
                expected_improvement=0.0,
                confidence_score=1.0,  # High confidence for manual updates
                optimization_method=f"Manual ({reason})",
                timestamp=datetime.now(),
                rollback_available=True
            )
            
            self.optimization_history[strategy_name].append(optimization_result)
            
            # Save configurations
            self._save_configurations()
            self._save_parameter_history()
            self._save_optimization_history()
            
            logger.info(f"Manually updated parameters for {strategy_name}: {reason}")
            return True
            
        except Exception as e:
            logger.error(f"Error forcing parameter update for {strategy_name}: {e}")
            return False
    
    def cleanup_old_data(self, days_to_keep: int = 90) -> Dict[str, int]:
        """
        Clean up old parameter and optimization history.
        
        Args:
            days_to_keep: Number of days of data to keep
            
        Returns:
            Dictionary with cleanup statistics
        """
        try:
            cutoff_date = datetime.now() - timedelta(days=days_to_keep)
            cleanup_stats = {'parameter_sets_removed': 0, 'optimizations_removed': 0}
            
            # Clean parameter history
            for strategy_name in self.parameter_history:
                old_count = len(self.parameter_history[strategy_name])
                self.parameter_history[strategy_name] = [
                    ps for ps in self.parameter_history[strategy_name]
                    if ps.timestamp > cutoff_date
                ]
                new_count = len(self.parameter_history[strategy_name])
                cleanup_stats['parameter_sets_removed'] += old_count - new_count
            
            # Clean optimization history
            for strategy_name in self.optimization_history:
                old_count = len(self.optimization_history[strategy_name])
                self.optimization_history[strategy_name] = [
                    opt for opt in self.optimization_history[strategy_name]
                    if opt.timestamp > cutoff_date
                ]
                new_count = len(self.optimization_history[strategy_name])
                cleanup_stats['optimizations_removed'] += old_count - new_count
            
            # Save cleaned data
            if cleanup_stats['parameter_sets_removed'] > 0 or cleanup_stats['optimizations_removed'] > 0:
                self._save_parameter_history()
                self._save_optimization_history()
            
            logger.info(f"Cleaned up old data: {cleanup_stats}")
            return cleanup_stats
            
        except Exception as e:
            logger.error(f"Error cleaning up old data: {e}")
            return {'parameter_sets_removed': 0, 'optimizations_removed': 0}
    
    def export_strategy_data(self, strategy_name: str, export_path: str) -> bool:
        """Export all data for a specific strategy."""
        try:
            if strategy_name not in self.strategy_configs:
                logger.error(f"Strategy {strategy_name} not registered")
                return False
            
            export_data = {
                'strategy_name': strategy_name,
                'strategy_config': {
                    name: asdict(bounds) for name, bounds in self.strategy_configs[strategy_name].items()
                },
                'current_parameters': self.current_parameters.get(strategy_name, {}),
                'parameter_history': [
                    asdict(ps) for ps in self.parameter_history.get(strategy_name, [])
                ],
                'optimization_history': [
                    asdict(opt) for opt in self.optimization_history.get(strategy_name, [])
                ],
                'export_timestamp': datetime.now().isoformat()
            }
            
            # Convert datetime objects to strings for JSON serialization
            def convert_datetime(obj):
                if isinstance(obj, datetime):
                    return obj.isoformat()
                return obj
            
            with open(export_path, 'w') as f:
                json.dump(export_data, f, indent=2, default=convert_datetime)
            
            logger.info(f"Exported strategy data for {strategy_name} to {export_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting strategy data: {e}")
            return False
    
    def _save_configurations(self):
        """Save current configurations to disk."""
        try:
            # Save strategy configs (convert ParameterBounds to dict)
            configs_data = {}
            for strategy_name, bounds_dict in self.strategy_configs.items():
                configs_data[strategy_name] = {
                    name: asdict(bounds) for name, bounds in bounds_dict.items()
                }
            
            with open(self.configs_file, 'w') as f:
                json.dump(configs_data, f, indent=2)
            
            # Save current parameters
            with open(self.parameters_file, 'w') as f:
                json.dump(self.current_parameters, f, indent=2)
            
        except Exception as e:
            logger.error(f"Error saving configurations: {e}")
    
    def _save_parameter_history(self):
        """Save parameter history to disk."""
        try:
            with open(self.history_file, 'wb') as f:
                pickle.dump(dict(self.parameter_history), f)
        except Exception as e:
            logger.error(f"Error saving parameter history: {e}")
    
    def _save_optimization_history(self):
        """Save optimization history to disk."""
        try:
            with open(self.optimization_file, 'wb') as f:
                pickle.dump(dict(self.optimization_history), f)
        except Exception as e:
            logger.error(f"Error saving optimization history: {e}")
    
    def _load_configurations(self):
        """Load configurations from disk."""
        try:
            # Load strategy configs
            if os.path.exists(self.configs_file):
                with open(self.configs_file, 'r') as f:
                    configs_data = json.load(f)
                
                for strategy_name, bounds_dict in configs_data.items():
                    self.strategy_configs[strategy_name] = {}
                    for name, bounds_data in bounds_dict.items():
                        bounds = ParameterBounds(**bounds_data)
                        self.strategy_configs[strategy_name][name] = bounds
            
            # Load current parameters
            if os.path.exists(self.parameters_file):
                with open(self.parameters_file, 'r') as f:
                    self.current_parameters = json.load(f)
            
            # Load parameter history
            if os.path.exists(self.history_file):
                with open(self.history_file, 'rb') as f:
                    history_data = pickle.load(f)
                    for strategy_name, param_sets in history_data.items():
                        self.parameter_history[strategy_name] = param_sets
            
            # Load optimization history
            if os.path.exists(self.optimization_file):
                with open(self.optimization_file, 'rb') as f:
                    opt_data = pickle.load(f)
                    for strategy_name, opt_results in opt_data.items():
                        self.optimization_history[strategy_name] = opt_results
            
            # Initialize last optimization times
            for strategy_name in self.strategy_configs:
                if strategy_name not in self.last_optimization:
                    self.last_optimization[strategy_name] = datetime.now() - timedelta(days=self.optimization_frequency)
            
        except Exception as e:
            logger.error(f"Error loading configurations: {e}")
    
    def get_all_strategies_summary(self) -> Dict[str, Any]:
        """Get summary of all registered strategies."""
        try:
            summary = {
                'total_strategies': len(self.strategy_configs),
                'strategies': {}
            }
            
            for strategy_name in self.strategy_configs:
                summary['strategies'][strategy_name] = self.get_strategy_performance_summary(strategy_name)
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting all strategies summary: {e}")
            return {'error': str(e)}