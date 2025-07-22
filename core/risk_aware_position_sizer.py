"""
Risk-Aware Position Sizer using Machine Learning

This module implements an advanced position sizing system that uses machine learning
to predict optimal position sizes based on trade confidence, market volatility,
portfolio state, and risk constraints.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque, defaultdict
import warnings
warnings.filterwarnings('ignore')

# ML libraries
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

from scipy import stats
from scipy.optimize import minimize_scalar
import pickle
import os
import json

logger = logging.getLogger(__name__)


@dataclass
class PositionSizingConfig:
    """Configuration for position sizing system"""
    # Risk parameters
    max_position_size: float = 0.1  # Maximum 10% of portfolio per trade
    min_position_size: float = 0.001  # Minimum 0.1% of portfolio
    max_portfolio_risk: float = 0.2  # Maximum 20% portfolio at risk
    max_drawdown_limit: float = 0.15  # Stop trading if drawdown > 15%
    
    # Kelly Criterion parameters
    kelly_multiplier: float = 0.25  # Conservative Kelly (25% of full Kelly)
    kelly_lookback: int = 100  # Trades to look back for Kelly calculation
    min_kelly_trades: int = 20  # Minimum trades needed for Kelly
    
    # Model training parameters
    retrain_frequency: int = 50  # Retrain every 50 trades
    min_training_samples: int = 100
    
    # Ensemble weights
    ensemble_weights: Dict[str, float] = field(default_factory=lambda: {
        'gradient_boosting': 0.4,
        'random_forest': 0.3,
        'xgboost': 0.2,
        'lightgbm': 0.1
    })


@dataclass
class TradeContext:
    """Context information for a trade"""
    symbol: str
    signal_confidence: float
    strategy_type: str
    entry_price: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    market_volatility: float = 0.02
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class PortfolioState:
    """Current portfolio state"""
    total_capital: float
    available_capital: float
    current_positions: int
    total_exposure: float
    unrealized_pnl: float
    realized_pnl: float
    max_drawdown: float
    recent_returns: List[float] = field(default_factory=list)
    win_rate: float = 0.5
    avg_win: float = 0.0
    avg_loss: float = 0.0
    sharpe_ratio: float = 0.0
    volatility: float = 0.02


@dataclass
class PositionSizeResult:
    """Result from position sizing calculation"""
    recommended_size: float  # As percentage of portfolio
    recommended_quantity: int  # Number of shares/units
    confidence_interval: Tuple[float, float]  # Lower and upper bounds
    kelly_size: float  # Kelly Criterion recommendation
    risk_adjusted_size: float  # Risk-adjusted recommendation
    ensemble_predictions: Dict[str, float]  # Individual model predictions
    risk_metrics: Dict[str, float]  # Risk analysis
    reasoning: List[str]  # Explanation of sizing decision
    uncertainty_score: float  # Uncertainty in the prediction
    max_loss_estimate: float  # Estimated maximum loss
    expected_return: float  # Expected return estimate
    timestamp: datetime = field(default_factory=datetime.now)
clas
s KellyCriterionCalculator:
    """Calculates Kelly Criterion with ML-based probability estimates"""
    
    def __init__(self, config: PositionSizingConfig):
        self.config = config
        self.trade_history = deque(maxlen=config.kelly_lookback)
        self.probability_model = GradientBoostingRegressor(
            n_estimators=50, max_depth=4, learning_rate=0.1, random_state=42
        )
        self.is_trained = False
    
    def add_trade_result(self, trade_context: TradeContext, actual_return: float):
        """Add trade result to history"""
        trade_record = {
            'confidence': trade_context.signal_confidence,
            'volatility': trade_context.market_volatility,
            'strategy': trade_context.strategy_type,
            'return': actual_return,
            'win': 1 if actual_return > 0 else 0,
            'timestamp': trade_context.timestamp
        }
        self.trade_history.append(trade_record)
        
        # Train model if we have enough data
        if len(self.trade_history) >= self.config.min_kelly_trades and len(self.trade_history) % 10 == 0:
            self._train_probability_model()
    
    def calculate_kelly_fraction(self, trade_context: TradeContext, 
                                portfolio_state: PortfolioState) -> Tuple[float, Dict[str, float]]:
        """Calculate Kelly fraction with ML-based probability estimates"""
        if len(self.trade_history) < self.config.min_kelly_trades:
            return self._simple_kelly(portfolio_state)
        
        try:
            # Predict win probability
            win_probability = self._predict_win_probability(trade_context)
            
            # Estimate win/loss amounts
            avg_win, avg_loss = self._estimate_win_loss_amounts()
            
            # Calculate Kelly fraction
            if avg_loss != 0:
                kelly_fraction = (win_probability * avg_win - (1 - win_probability) * abs(avg_loss)) / abs(avg_loss)
            else:
                kelly_fraction = 0.0
            
            # Apply Kelly multiplier for conservative sizing
            kelly_fraction *= self.config.kelly_multiplier
            
            # Ensure reasonable bounds
            kelly_fraction = max(0, min(kelly_fraction, self.config.max_position_size))
            
            kelly_metrics = {
                'win_probability': win_probability,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'raw_kelly': kelly_fraction / self.config.kelly_multiplier,
                'adjusted_kelly': kelly_fraction
            }
            
            return kelly_fraction, kelly_metrics
            
        except Exception as e:
            logger.error(f"Error calculating Kelly fraction: {e}")
            return self._simple_kelly(portfolio_state)
    
    def _simple_kelly(self, portfolio_state: PortfolioState) -> Tuple[float, Dict[str, float]]:
        """Simple Kelly calculation based on portfolio statistics"""
        win_rate = portfolio_state.win_rate
        avg_win = portfolio_state.avg_win if portfolio_state.avg_win > 0 else 0.02
        avg_loss = abs(portfolio_state.avg_loss) if portfolio_state.avg_loss < 0 else 0.02
        
        if avg_loss > 0:
            kelly_fraction = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_loss
        else:
            kelly_fraction = 0.0
        
        kelly_fraction *= self.config.kelly_multiplier
        kelly_fraction = max(0, min(kelly_fraction, self.config.max_position_size))
        
        kelly_metrics = {
            'win_probability': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'raw_kelly': kelly_fraction / self.config.kelly_multiplier,
            'adjusted_kelly': kelly_fraction
        }
        
        return kelly_fraction, kelly_metrics
    
    def _train_probability_model(self):
        """Train ML model to predict win probability"""
        if len(self.trade_history) < self.config.min_kelly_trades:
            return
        
        try:
            X, y = [], []
            
            for trade in list(self.trade_history):
                features = [
                    trade['confidence'],
                    trade['volatility'],
                    {'momentum': 1, 'mean_reversion': 2, 'breakout': 3, 'trend_following': 4, 'scalping': 5}.get(trade['strategy'], 0)
                ]
                X.append(features)
                y.append(trade['win'])
            
            X = np.array(X)
            y = np.array(y)
            
            self.probability_model.fit(X, y)
            self.is_trained = True
            
        except Exception as e:
            logger.error(f"Error training probability model: {e}")
    
    def _predict_win_probability(self, trade_context: TradeContext) -> float:
        """Predict win probability using trained model"""
        if not self.is_trained:
            return 0.5
        
        try:
            features = np.array([[
                trade_context.signal_confidence,
                trade_context.market_volatility,
                {'momentum': 1, 'mean_reversion': 2, 'breakout': 3, 'trend_following': 4, 'scalping': 5}.get(trade_context.strategy_type, 0)
            ]])
            
            prediction = self.probability_model.predict(features)[0]
            # Convert to probability using sigmoid-like function
            probability = max(0.1, min(0.9, prediction))
            return probability
            
        except Exception as e:
            logger.error(f"Error predicting win probability: {e}")
            return 0.5
    
    def _estimate_win_loss_amounts(self) -> Tuple[float, float]:
        """Estimate average win and loss amounts"""
        if len(self.trade_history) < 10:
            return 0.02, -0.015  # Default estimates
        
        wins = [trade['return'] for trade in self.trade_history if trade['return'] > 0]
        losses = [trade['return'] for trade in self.trade_history if trade['return'] < 0]
        
        avg_win = np.mean(wins) if wins else 0.02
        avg_loss = np.mean(losses) if losses else -0.015
        
        return avg_win, avg_loss


class EnsemblePositionSizer:
    """Ensemble of ML models for position sizing"""
    
    def __init__(self, config: PositionSizingConfig):
        self.config = config
        self.models = {}
        self.scalers = {}
        self.training_data = deque(maxlen=1000)
        self.is_trained = False
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize ensemble models"""
        # Gradient Boosting
        self.models['gradient_boosting'] = GradientBoostingRegressor(
            n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42
        )
        
        # Random Forest
        self.models['random_forest'] = RandomForestRegressor(
            n_estimators=100, max_depth=8, random_state=42
        )
        
        # XGBoost (if available)
        if XGBOOST_AVAILABLE:
            self.models['xgboost'] = xgb.XGBRegressor(
                n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, verbosity=0
            )
        
        # LightGBM (if available)
        if LIGHTGBM_AVAILABLE:
            self.models['lightgbm'] = lgb.LGBMRegressor(
                n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, verbosity=-1
            )
        
        # Initialize scalers
        for model_name in self.models.keys():
            self.scalers[model_name] = RobustScaler()
    
    def add_training_sample(self, features: Dict[str, float], optimal_size: float, actual_return: float):
        """Add training sample for model improvement"""
        sample = {
            'features': features,
            'optimal_size': optimal_size,
            'actual_return': actual_return,
            'timestamp': datetime.now()
        }
        self.training_data.append(sample)
    
    def train_models(self) -> bool:
        """Train all ensemble models"""
        if len(self.training_data) < self.config.min_training_samples:
            return False
        
        try:
            X, y = self._prepare_training_data()
            if len(X) == 0:
                return False
            
            # Train each model
            for model_name, model in self.models.items():
                try:
                    X_scaled = self.scalers[model_name].fit_transform(X)
                    model.fit(X_scaled, y)
                    logger.info(f"Trained {model_name} model with {len(X)} samples")
                except Exception as e:
                    logger.error(f"Error training {model_name}: {e}")
                    continue
            
            self.is_trained = True
            return True
            
        except Exception as e:
            logger.error(f"Error training ensemble models: {e}")
            return False
    
    def predict_position_size(self, features: Dict[str, float]) -> Dict[str, float]:
        """Predict position size using ensemble models"""
        if not self.is_trained:
            return {name: self.config.max_position_size * 0.5 for name in self.models.keys()}
        
        predictions = {}
        feature_array = self._features_to_array(features)
        
        for model_name, model in self.models.items():
            try:
                X_scaled = self.scalers[model_name].transform(feature_array.reshape(1, -1))
                prediction = model.predict(X_scaled)[0]
                prediction = np.clip(prediction, self.config.min_position_size, self.config.max_position_size)
                predictions[model_name] = prediction
            except Exception as e:
                logger.error(f"Error predicting with {model_name}: {e}")
                predictions[model_name] = self.config.max_position_size * 0.5
        
        return predictions
    
    def _prepare_training_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data from collected samples"""
        X, y = [], []
        
        for sample in list(self.training_data):
            try:
                features_array = self._features_to_array(sample['features'])
                X.append(features_array)
                y.append(sample['optimal_size'])
            except Exception as e:
                logger.warning(f"Error processing training sample: {e}")
                continue
        
        return np.array(X), np.array(y)
    
    def _features_to_array(self, features: Dict[str, float]) -> np.ndarray:
        """Convert feature dictionary to array"""
        feature_names = [
            'signal_confidence', 'market_volatility', 'portfolio_exposure',
            'current_drawdown', 'recent_performance', 'win_rate',
            'sharpe_ratio', 'volatility_regime', 'position_count', 'available_capital_ratio'
        ]
        
        feature_array = []
        for name in feature_names:
            feature_array.append(features.get(name, 0.0))
        
        return np.array(feature_array)
class 
RiskAwarePositionSizer:
    """
    Advanced ML-based position sizing system that combines multiple risk models
    """
    
    def __init__(self, config: PositionSizingConfig = None, model_dir: str = "models/position_sizer"):
        self.config = config or PositionSizingConfig()
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        
        # Initialize components
        self.kelly_calculator = KellyCriterionCalculator(self.config)
        self.ensemble_sizer = EnsemblePositionSizer(self.config)
        
        # Performance tracking
        self.trade_count = 0
        self.performance_history = deque(maxlen=1000)
        self.sizing_history = deque(maxlen=1000)
        
        logger.info("RiskAwarePositionSizer initialized")
    
    def calculate_position_size(self, trade_context: TradeContext, 
                               portfolio_state: PortfolioState) -> PositionSizeResult:
        """
        Calculate optimal position size using ML ensemble and risk models
        """
        try:
            # Prepare features for ML models
            features = self._prepare_features(trade_context, portfolio_state)
            
            # Get Kelly Criterion recommendation
            kelly_size, kelly_metrics = self.kelly_calculator.calculate_kelly_fraction(
                trade_context, portfolio_state
            )
            
            # Get ensemble model predictions
            ensemble_predictions = self.ensemble_sizer.predict_position_size(features)
            
            # Calculate ensemble weighted average
            ensemble_size = self._calculate_ensemble_average(ensemble_predictions)
            
            # Apply risk adjustments
            risk_adjusted_size = self._apply_risk_adjustments(
                ensemble_size, trade_context, portfolio_state
            )
            
            # Apply portfolio constraints
            final_size = self._apply_portfolio_constraints(
                risk_adjusted_size, portfolio_state
            )
            
            # Calculate uncertainty and confidence intervals
            uncertainty_score, confidence_interval = self._calculate_uncertainty(
                ensemble_predictions, final_size
            )
            
            # Calculate position quantity
            recommended_quantity = self._calculate_quantity(
                final_size, trade_context, portfolio_state
            )
            
            # Generate reasoning
            reasoning = self._generate_reasoning(
                final_size, kelly_size, ensemble_size, risk_adjusted_size,
                trade_context, portfolio_state
            )
            
            # Calculate risk metrics
            risk_metrics = self._calculate_risk_metrics(
                final_size, trade_context, portfolio_state
            )
            
            # Create result
            result = PositionSizeResult(
                recommended_size=final_size,
                recommended_quantity=recommended_quantity,
                confidence_interval=confidence_interval,
                kelly_size=kelly_size,
                risk_adjusted_size=risk_adjusted_size,
                ensemble_predictions=ensemble_predictions,
                risk_metrics=risk_metrics,
                reasoning=reasoning,
                uncertainty_score=uncertainty_score,
                max_loss_estimate=risk_metrics.get('max_loss_estimate', 0.0),
                expected_return=risk_metrics.get('expected_return', 0.0)
            )
            
            # Store sizing decision for learning
            self._store_sizing_decision(trade_context, result, features)
            
            return result
            
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return self._get_default_result(trade_context, portfolio_state)
    
    def update_trade_result(self, trade_context: TradeContext, actual_return: float,
                           position_size_used: float):
        """Update models with actual trade results"""
        try:
            # Update Kelly calculator
            self.kelly_calculator.add_trade_result(trade_context, actual_return)
            
            # Calculate optimal size in hindsight for training
            optimal_size = self._calculate_optimal_size_hindsight(actual_return, position_size_used)
            
            # Add training sample to ensemble
            features = self._prepare_features(trade_context, None)
            self.ensemble_sizer.add_training_sample(features, optimal_size, actual_return)
            
            # Update performance tracking
            self.performance_history.append({
                'return': actual_return,
                'position_size': position_size_used,
                'timestamp': datetime.now()
            })
            
            self.trade_count += 1
            
            # Retrain models periodically
            if self.trade_count % self.config.retrain_frequency == 0:
                self._retrain_models()
            
        except Exception as e:
            logger.error(f"Error updating trade result: {e}")
    
    def _prepare_features(self, trade_context: TradeContext, 
                         portfolio_state: Optional[PortfolioState]) -> Dict[str, float]:
        """Prepare features for ML models"""
        features = {
            'signal_confidence': trade_context.signal_confidence,
            'market_volatility': trade_context.market_volatility,
        }
        
        if portfolio_state:
            features.update({
                'portfolio_exposure': portfolio_state.total_exposure / portfolio_state.total_capital,
                'current_drawdown': abs(portfolio_state.max_drawdown),
                'recent_performance': np.mean(portfolio_state.recent_returns) if portfolio_state.recent_returns else 0,
                'win_rate': portfolio_state.win_rate,
                'sharpe_ratio': portfolio_state.sharpe_ratio,
                'position_count': portfolio_state.current_positions,
                'available_capital_ratio': portfolio_state.available_capital / portfolio_state.total_capital
            })
        else:
            # Default values when portfolio state not available
            features.update({
                'portfolio_exposure': 0.5,
                'current_drawdown': 0.0,
                'recent_performance': 0.0,
                'win_rate': 0.5,
                'sharpe_ratio': 0.0,
                'position_count': 0,
                'available_capital_ratio': 1.0
            })
        
        # Add volatility regime
        features['volatility_regime'] = trade_context.market_volatility
        
        return features
    
    def _calculate_ensemble_average(self, predictions: Dict[str, float]) -> float:
        """Calculate weighted average of ensemble predictions"""
        if not predictions:
            return self.config.max_position_size * 0.5
        
        weighted_sum = 0.0
        total_weight = 0.0
        
        for model_name, prediction in predictions.items():
            weight = self.config.ensemble_weights.get(model_name, 0.25)
            weighted_sum += prediction * weight
            total_weight += weight
        
        if total_weight > 0:
            return weighted_sum / total_weight
        else:
            return np.mean(list(predictions.values()))
    
    def _apply_risk_adjustments(self, base_size: float, trade_context: TradeContext,
                               portfolio_state: PortfolioState) -> float:
        """Apply various risk adjustments to position size"""
        adjusted_size = base_size
        
        # Volatility adjustment
        vol_adjustment = min(1.0, 0.02 / trade_context.market_volatility)
        adjusted_size *= vol_adjustment
        
        # Performance adjustment
        if portfolio_state.recent_returns:
            recent_perf = np.mean(portfolio_state.recent_returns[-10:]) if len(portfolio_state.recent_returns) >= 10 else 0
            if recent_perf < -0.05:  # Recent losses > 5%
                adjusted_size *= 0.5  # Reduce size by half
            elif recent_perf > 0.05:  # Recent gains > 5%
                adjusted_size *= 1.2  # Increase size by 20%
        
        # Drawdown adjustment
        if portfolio_state.max_drawdown < -0.1:  # Drawdown > 10%
            drawdown_factor = 1 + portfolio_state.max_drawdown
            adjusted_size *= max(0.1, drawdown_factor)
        
        # Confidence adjustment
        confidence_factor = 0.5 + (trade_context.signal_confidence * 0.5)
        adjusted_size *= confidence_factor
        
        return adjusted_size
    
    def _apply_portfolio_constraints(self, size: float, portfolio_state: PortfolioState) -> float:
        """Apply portfolio-level constraints"""
        # Maximum position size constraint
        size = min(size, self.config.max_position_size)
        
        # Minimum position size constraint
        size = max(size, self.config.min_position_size)
        
        # Maximum portfolio risk constraint
        current_risk = portfolio_state.total_exposure / portfolio_state.total_capital
        if current_risk + size > self.config.max_portfolio_risk:
            size = max(0, self.config.max_portfolio_risk - current_risk)
        
        # Available capital constraint
        max_size_by_capital = portfolio_state.available_capital / portfolio_state.total_capital
        size = min(size, max_size_by_capital)
        
        # Drawdown limit constraint
        if abs(portfolio_state.max_drawdown) > self.config.max_drawdown_limit:
            size *= 0.1  # Severely reduce position sizes during large drawdowns
        
        return size
    
    def _calculate_uncertainty(self, predictions: Dict[str, float], 
                              final_size: float) -> Tuple[float, Tuple[float, float]]:
        """Calculate uncertainty and confidence intervals"""
        if not predictions:
            return 0.5, (final_size * 0.8, final_size * 1.2)
        
        pred_values = list(predictions.values())
        
        # Calculate uncertainty as coefficient of variation
        if len(pred_values) > 1:
            uncertainty = np.std(pred_values) / (np.mean(pred_values) + 1e-8)
        else:
            uncertainty = 0.1
        
        # Calculate confidence intervals
        std_dev = np.std(pred_values) if len(pred_values) > 1 else final_size * 0.1
        lower_bound = max(self.config.min_position_size, final_size - 1.96 * std_dev)
        upper_bound = min(self.config.max_position_size, final_size + 1.96 * std_dev)
        
        return uncertainty, (lower_bound, upper_bound)
    
    def _calculate_quantity(self, position_size: float, trade_context: TradeContext,
                           portfolio_state: PortfolioState) -> int:
        """Calculate number of shares/units to trade"""
        if position_size <= 0 or trade_context.entry_price <= 0:
            return 0
        
        position_value = portfolio_state.total_capital * position_size
        quantity = int(position_value / trade_context.entry_price)
        
        return max(0, quantity)
    
    def _generate_reasoning(self, final_size: float, kelly_size: float, 
                           ensemble_size: float, risk_adjusted_size: float,
                           trade_context: TradeContext, portfolio_state: PortfolioState) -> List[str]:
        """Generate human-readable reasoning for position size decision"""
        reasoning = []
        
        reasoning.append(f"Final position size: {final_size:.1%} of portfolio")
        reasoning.append(f"Kelly Criterion suggests: {kelly_size:.1%}")
        reasoning.append(f"ML ensemble suggests: {ensemble_size:.1%}")
        reasoning.append(f"Risk-adjusted size: {risk_adjusted_size:.1%}")
        
        if trade_context.signal_confidence > 0.8:
            reasoning.append("High signal confidence increases position size")
        elif trade_context.signal_confidence < 0.4:
            reasoning.append("Low signal confidence reduces position size")
        
        if trade_context.market_volatility > 0.03:
            reasoning.append("High market volatility reduces position size")
        elif trade_context.market_volatility < 0.015:
            reasoning.append("Low market volatility allows larger position size")
        
        if portfolio_state.max_drawdown < -0.1:
            reasoning.append("Current drawdown reduces position size for risk management")
        
        return reasoning
    
    def _calculate_risk_metrics(self, position_size: float, trade_context: TradeContext,
                               portfolio_state: PortfolioState) -> Dict[str, float]:
        """Calculate risk metrics for the position"""
        risk_metrics = {}
        
        # Maximum loss estimate
        if trade_context.stop_loss:
            max_loss_pct = abs(trade_context.entry_price - trade_context.stop_loss) / trade_context.entry_price
            risk_metrics['max_loss_estimate'] = position_size * max_loss_pct
        else:
            risk_metrics['max_loss_estimate'] = position_size * trade_context.market_volatility * 2
        
        # Expected return estimate
        if trade_context.take_profit:
            max_gain_pct = abs(trade_context.take_profit - trade_context.entry_price) / trade_context.entry_price
            expected_return = position_size * max_gain_pct * trade_context.signal_confidence
            risk_metrics['expected_return'] = expected_return
        else:
            risk_metrics['expected_return'] = position_size * 0.02 * trade_context.signal_confidence
        
        # Risk-reward ratio
        if risk_metrics['max_loss_estimate'] > 0:
            risk_metrics['risk_reward_ratio'] = risk_metrics['expected_return'] / risk_metrics['max_loss_estimate']
        else:
            risk_metrics['risk_reward_ratio'] = 1.0
        
        return risk_metrics
    
    def _store_sizing_decision(self, trade_context: TradeContext, result: PositionSizeResult,
                              features: Dict[str, float]):
        """Store sizing decision for analysis and learning"""
        decision_record = {
            'timestamp': datetime.now(),
            'symbol': trade_context.symbol,
            'signal_confidence': trade_context.signal_confidence,
            'recommended_size': result.recommended_size,
            'kelly_size': result.kelly_size,
            'ensemble_predictions': result.ensemble_predictions,
            'features': features
        }
        
        self.sizing_history.append(decision_record)
    
    def _calculate_optimal_size_hindsight(self, actual_return: float, 
                                         position_size_used: float) -> float:
        """Calculate what would have been optimal position size in hindsight"""
        if actual_return > 0:
            # For winning trades, could have used larger size (up to max)
            optimal_size = min(self.config.max_position_size, position_size_used * 1.5)
        else:
            # For losing trades, should have used smaller size
            optimal_size = max(self.config.min_position_size, position_size_used * 0.5)
        
        return optimal_size
    
    def _retrain_models(self):
        """Retrain ML models with accumulated data"""
        try:
            logger.info("Retraining position sizing models...")
            success = self.ensemble_sizer.train_models()
            if success:
                logger.info("Models retrained successfully")
            else:
                logger.warning("Model retraining failed")
        except Exception as e:
            logger.error(f"Error retraining models: {e}")
    
    def _get_default_result(self, trade_context: TradeContext, 
                           portfolio_state: PortfolioState) -> PositionSizeResult:
        """Get default result when calculation fails"""
        default_size = self.config.max_position_size * 0.5
        
        return PositionSizeResult(
            recommended_size=default_size,
            recommended_quantity=self._calculate_quantity(default_size, trade_context, portfolio_state),
            confidence_interval=(default_size * 0.8, default_size * 1.2),
            kelly_size=default_size,
            risk_adjusted_size=default_size,
            ensemble_predictions={'default': default_size},
            risk_metrics={'max_loss_estimate': default_size * 0.02, 'expected_return': default_size * 0.01},
            reasoning=['Default sizing due to calculation error'],
            uncertainty_score=0.5,
            max_loss_estimate=default_size * 0.02,
            expected_return=default_size * 0.01
        )
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary of position sizing decisions"""
        if not self.performance_history:
            return {'message': 'No performance data available'}
        
        returns = [p['return'] for p in self.performance_history]
        sizes = [p['position_size'] for p in self.performance_history]
        
        summary = {
            'total_trades': len(returns),
            'avg_return': np.mean(returns),
            'avg_position_size': np.mean(sizes),
            'win_rate': sum(1 for r in returns if r > 0) / len(returns),
            'sharpe_ratio': np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0,
            'total_return': sum(returns)
        }
        
        return summary
    
    def save_models(self, filepath: str = None):
        """Save trained models to disk"""
        if filepath is None:
            filepath = os.path.join(self.model_dir, f"position_sizer_models_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl")
        
        try:
            model_data = {
                'config': self.config,
                'ensemble_models': self.ensemble_sizer.models,
                'ensemble_scalers': self.ensemble_sizer.scalers,
                'kelly_model': self.kelly_calculator.probability_model,
                'trade_count': self.trade_count,
                'performance_history': list(self.performance_history)
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
            
            logger.info(f"Position sizing models saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving models: {e}")


# Example usage
if __name__ == "__main__":
    # Create configuration
    config = PositionSizingConfig(
        max_position_size=0.1,
        kelly_multiplier=0.25,
        retrain_frequency=50
    )
    
    # Initialize position sizer
    position_sizer = RiskAwarePositionSizer(config)
    
    # Create sample trade context
    trade_context = TradeContext(
        symbol="RELIANCE",
        signal_confidence=0.75,
        strategy_type="momentum",
        entry_price=2500.0,
        stop_loss=2400.0,
        take_profit=2700.0,
        market_volatility=0.025
    )
    
    # Create sample portfolio state
    portfolio_state = PortfolioState(
        total_capital=1000000.0,
        available_capital=800000.0,
        current_positions=3,
        total_exposure=200000.0,
        unrealized_pnl=5000.0,
        realized_pnl=15000.0,
        max_drawdown=-0.05,
        recent_returns=[0.01, -0.005, 0.02, 0.008, -0.01],
        win_rate=0.6,
        sharpe_ratio=1.2
    )
    
    # Calculate position size
    result = position_sizer.calculate_position_size(trade_context, portfolio_state)
    
    print(f"Recommended position size: {result.recommended_size:.2%}")
    print(f"Recommended quantity: {result.recommended_quantity}")
    print(f"Kelly size: {result.kelly_size:.2%}")
    print(f"Expected return: {result.expected_return:.2%}")
    print(f"Max loss estimate: {result.max_loss_estimate:.2%}")
    print("\nReasoning:")
    for reason in result.reasoning:
        print(f"- {reason}")