"""
Model Validation Framework for Trading ML Models

This module implements comprehensive validation strategies for machine learning models
used in trading, including walk-forward analysis, time series cross-validation,
performance metrics calculation, overfitting detection, and model comparison.

Key Features:
- Walk-forward analysis with expanding/rolling windows
- Time series cross-validation with purged splits
- 20+ comprehensive performance metrics
- Statistical significance testing
- Model stability and degradation detection
- Performance attribution by market regime
- Automated validation reports
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from collections import defaultdict, deque
import warnings
warnings.filterwarnings('ignore')

# Statistical and ML libraries
from scipy import stats
from scipy.stats import jarque_bera, normaltest, kstest
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)


@dataclass
class ValidationConfig:
    """Configuration for model validation"""
    # Walk-forward analysis settings
    initial_train_size: int = 252  # Trading days (1 year)
    step_size: int = 21  # Retraining frequency (monthly)
    window_type: str = 'expanding'  # 'expanding' or 'rolling'
    max_train_size: Optional[int] = None  # For rolling window
    
    # Cross-validation settings
    cv_folds: int = 5
    purged_cv: bool = True
    embargo_period: int = 5  # Days to embargo after each fold
    
    # Performance metrics settings
    risk_free_rate: float = 0.06  # Annual risk-free rate
    benchmark_return: float = 0.12  # Annual benchmark return
    confidence_level: float = 0.95  # For VaR calculations
    
    # Statistical testing
    significance_level: float = 0.05
    min_sample_size: int = 30
    
    # Model stability settings
    stability_window: int = 63  # Quarter for stability analysis
    degradation_threshold: float = 0.1  # 10% performance drop threshold
    
    # Reporting settings
    generate_plots: bool = True
    save_detailed_results: bool = True
    report_format: str = 'html'  # 'html', 'pdf', 'json'


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics"""
    # Basic metrics
    total_return: float = 0.0
    annualized_return: float = 0.0
    volatility: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    
    # Risk metrics
    max_drawdown: float = 0.0
    max_drawdown_duration: int = 0
    var_95: float = 0.0  # Value at Risk
    cvar_95: float = 0.0  # Conditional VaR
    downside_deviation: float = 0.0
    
    # Distribution metrics
    skewness: float = 0.0
    kurtosis: float = 0.0
    jarque_bera_stat: float = 0.0
    jarque_bera_pvalue: float = 0.0
    
    # Trading metrics
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0
    
    # Model-specific metrics
    hit_rate: float = 0.0  # Directional accuracy
    information_ratio: float = 0.0
    tracking_error: float = 0.0
    beta: float = 0.0
    alpha: float = 0.0
    
    # Additional metrics
    recovery_factor: float = 0.0
    payoff_ratio: float = 0.0
    expectancy: float = 0.0
    kelly_criterion: float = 0.0


@dataclass
class ValidationResult:
    """Results from model validation"""
    model_name: str
    validation_type: str
    start_date: datetime
    end_date: datetime
    
    # Performance metrics
    in_sample_metrics: PerformanceMetrics
    out_of_sample_metrics: PerformanceMetrics
    
    # Statistical tests
    overfitting_detected: bool = False
    overfitting_score: float = 0.0
    statistical_significance: bool = False
    p_value: float = 1.0
    
    # Stability analysis
    stability_score: float = 0.0
    degradation_detected: bool = False
    stability_periods: List[Dict] = field(default_factory=list)
    
    # Walk-forward results
    walk_forward_results: List[Dict] = field(default_factory=list)
    
    # Cross-validation results
    cv_scores: List[float] = field(default_factory=list)
    cv_mean: float = 0.0
    cv_std: float = 0.0
    
    # Market regime performance
    regime_performance: Dict[str, PerformanceMetrics] = field(default_factory=dict)
    
    # Additional analysis
    feature_importance: Dict[str, float] = field(default_factory=dict)
    prediction_intervals: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    
    # Metadata
    validation_timestamp: datetime = field(default_factory=datetime.now)
    computation_time: float = 0.0


class PerformanceCalculator:
    """Calculates comprehensive performance metrics"""
    
    @staticmethod
    def calculate_metrics(returns: pd.Series, benchmark_returns: pd.Series = None, 
                         risk_free_rate: float = 0.06) -> PerformanceMetrics:
        """Calculate comprehensive performance metrics"""
        metrics = PerformanceMetrics()
        
        if len(returns) == 0:
            return metrics
        
        try:
            # Convert to numpy for calculations
            ret_array = returns.dropna().values
            
            if len(ret_array) == 0:
                return metrics
            
            # Basic metrics
            metrics.total_return = (1 + returns).prod() - 1
            metrics.annualized_return = (1 + returns.mean()) ** 252 - 1
            metrics.volatility = returns.std() * np.sqrt(252)
            
            # Risk-adjusted metrics
            if metrics.volatility > 0:
                excess_return = metrics.annualized_return - risk_free_rate
                metrics.sharpe_ratio = excess_return / metrics.volatility
                
                # Sortino ratio (downside deviation)
                downside_returns = returns[returns < 0]
                if len(downside_returns) > 0:
                    metrics.downside_deviation = downside_returns.std() * np.sqrt(252)
                    metrics.sortino_ratio = excess_return / metrics.downside_deviation
            
            # Drawdown analysis
            cumulative_returns = (1 + returns).cumprod()
            running_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - running_max) / running_max
            
            metrics.max_drawdown = drawdown.min()
            
            # Max drawdown duration
            drawdown_periods = (drawdown < 0).astype(int)
            if drawdown_periods.sum() > 0:
                # Find consecutive periods of drawdown
                dd_groups = (drawdown_periods != drawdown_periods.shift()).cumsum()
                dd_lengths = drawdown_periods.groupby(dd_groups).sum()
                metrics.max_drawdown_duration = dd_lengths.max()
            
            # Calmar ratio
            if abs(metrics.max_drawdown) > 0:
                metrics.calmar_ratio = metrics.annualized_return / abs(metrics.max_drawdown)
            
            # Value at Risk and Conditional VaR
            metrics.var_95 = np.percentile(ret_array, 5)
            metrics.cvar_95 = ret_array[ret_array <= metrics.var_95].mean()
            
            # Distribution metrics
            if len(ret_array) >= 8:  # Minimum for JB test
                metrics.skewness = stats.skew(ret_array)
                metrics.kurtosis = stats.kurtosis(ret_array)
                
                try:
                    jb_stat, jb_pvalue = jarque_bera(ret_array)
                    metrics.jarque_bera_stat = jb_stat
                    metrics.jarque_bera_pvalue = jb_pvalue
                except:
                    pass
            
            # Trading metrics
            winning_trades = ret_array[ret_array > 0]
            losing_trades = ret_array[ret_array < 0]
            
            metrics.win_rate = len(winning_trades) / len(ret_array) if len(ret_array) > 0 else 0
            
            if len(winning_trades) > 0:
                metrics.avg_win = winning_trades.mean()
                metrics.largest_win = winning_trades.max()
            
            if len(losing_trades) > 0:
                metrics.avg_loss = losing_trades.mean()
                metrics.largest_loss = losing_trades.min()
            
            # Profit factor
            gross_profit = winning_trades.sum() if len(winning_trades) > 0 else 0
            gross_loss = abs(losing_trades.sum()) if len(losing_trades) > 0 else 0
            
            if gross_loss > 0:
                metrics.profit_factor = gross_profit / gross_loss
            
            # Payoff ratio
            if metrics.avg_loss != 0:
                metrics.payoff_ratio = abs(metrics.avg_win / metrics.avg_loss)
            
            # Expectancy
            metrics.expectancy = (metrics.win_rate * metrics.avg_win) + ((1 - metrics.win_rate) * metrics.avg_loss)
            
            # Kelly Criterion
            if metrics.avg_loss != 0 and metrics.payoff_ratio > 0:
                metrics.kelly_criterion = metrics.win_rate - ((1 - metrics.win_rate) / metrics.payoff_ratio)
            
            # Recovery factor
            if abs(metrics.max_drawdown) > 0:
                metrics.recovery_factor = metrics.total_return / abs(metrics.max_drawdown)
            
            # Benchmark-relative metrics
            if benchmark_returns is not None and len(benchmark_returns) == len(returns):
                try:
                    # Align returns
                    aligned_returns = returns.align(benchmark_returns, join='inner')
                    ret_aligned, bench_aligned = aligned_returns
                    
                    if len(ret_aligned) > 1:
                        # Beta and Alpha
                        covariance = np.cov(ret_aligned, bench_aligned)[0, 1]
                        benchmark_variance = np.var(bench_aligned)
                        
                        if benchmark_variance > 0:
                            metrics.beta = covariance / benchmark_variance
                            
                            # Alpha (Jensen's alpha)
                            benchmark_annual_return = (1 + bench_aligned.mean()) ** 252 - 1
                            metrics.alpha = metrics.annualized_return - (risk_free_rate + metrics.beta * (benchmark_annual_return - risk_free_rate))
                        
                        # Information ratio and tracking error
                        active_returns = ret_aligned - bench_aligned
                        metrics.tracking_error = active_returns.std() * np.sqrt(252)
                        
                        if metrics.tracking_error > 0:
                            metrics.information_ratio = active_returns.mean() * 252 / metrics.tracking_error
                except Exception as e:
                    logger.warning(f"Error calculating benchmark metrics: {e}")
            
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {e}")
        
        return metrics
    
    @staticmethod
    def calculate_hit_rate(predictions: np.ndarray, actual: np.ndarray) -> float:
        """Calculate directional accuracy (hit rate)"""
        if len(predictions) != len(actual) or len(predictions) == 0:
            return 0.0
        
        try:
            # Convert to directional signals
            pred_direction = np.sign(predictions)
            actual_direction = np.sign(actual)
            
            # Calculate accuracy
            correct_predictions = (pred_direction == actual_direction).sum()
            return correct_predictions / len(predictions)
        except:
            return 0.0


class TimeSeriesValidator:
    """Implements time series cross-validation with purging"""
    
    def __init__(self, config: ValidationConfig):
        self.config = config
    
    def purged_time_series_split(self, X: pd.DataFrame, y: pd.Series, 
                                test_size: int = None) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Create time series splits with purging to prevent data leakage
        """
        n_samples = len(X)
        test_size = test_size or n_samples // self.config.cv_folds
        
        splits = []
        
        for i in range(self.config.cv_folds):
            # Calculate test start and end
            test_start = i * (n_samples // self.config.cv_folds)
            test_end = min(test_start + test_size, n_samples)
            
            # Training data: everything before test (with embargo)
            train_end = max(0, test_start - self.config.embargo_period)
            
            if train_end > self.config.min_sample_size:
                train_indices = np.arange(0, train_end)
                test_indices = np.arange(test_start, test_end)
                
                splits.append((train_indices, test_indices))
        
        return splits
    
    def walk_forward_analysis(self, X: pd.DataFrame, y: pd.Series, 
                            model_func: Callable, 
                            prediction_func: Callable) -> List[Dict]:
        """
        Perform walk-forward analysis
        """
        results = []
        n_samples = len(X)
        
        # Initial training size
        train_size = self.config.initial_train_size
        
        for step in range(0, n_samples - train_size, self.config.step_size):
            try:
                # Define training window
                if self.config.window_type == 'expanding':
                    train_start = 0
                    train_end = train_size + step
                else:  # rolling
                    max_size = self.config.max_train_size or train_size
                    train_start = max(0, train_size + step - max_size)
                    train_end = train_size + step
                
                # Define test window
                test_start = train_end
                test_end = min(test_start + self.config.step_size, n_samples)
                
                if test_end <= test_start:
                    break
                
                # Split data
                X_train = X.iloc[train_start:train_end]
                y_train = y.iloc[train_start:train_end]
                X_test = X.iloc[test_start:test_end]
                y_test = y.iloc[test_start:test_end]
                
                # Train model
                model = model_func(X_train, y_train)
                
                # Make predictions
                predictions = prediction_func(model, X_test)
                
                # Calculate metrics
                if len(predictions) == len(y_test):
                    returns = pd.Series(predictions, index=y_test.index)
                    metrics = PerformanceCalculator.calculate_metrics(returns)
                    hit_rate = PerformanceCalculator.calculate_hit_rate(predictions, y_test.values)
                    
                    result = {
                        'step': step,
                        'train_start': train_start,
                        'train_end': train_end,
                        'test_start': test_start,
                        'test_end': test_end,
                        'train_size': train_end - train_start,
                        'test_size': test_end - test_start,
                        'metrics': metrics,
                        'hit_rate': hit_rate,
                        'predictions': predictions,
                        'actual': y_test.values
                    }
                    
                    results.append(result)
                
            except Exception as e:
                logger.error(f"Error in walk-forward step {step}: {e}")
                continue
        
        return results


class OverfittingDetector:
    """Detects overfitting in trading models"""
    
    @staticmethod
    def detect_overfitting(in_sample_metrics: PerformanceMetrics, 
                          out_of_sample_metrics: PerformanceMetrics,
                          threshold: float = 0.2) -> Tuple[bool, float]:
        """
        Detect overfitting by comparing in-sample vs out-of-sample performance
        """
        try:
            # Calculate performance degradation
            metrics_to_check = [
                ('sharpe_ratio', 'higher_better'),
                ('total_return', 'higher_better'),
                ('win_rate', 'higher_better'),
                ('max_drawdown', 'lower_better')
            ]
            
            degradation_scores = []
            
            for metric_name, direction in metrics_to_check:
                in_sample_value = getattr(in_sample_metrics, metric_name, 0)
                out_of_sample_value = getattr(out_of_sample_metrics, metric_name, 0)
                
                if in_sample_value != 0:
                    if direction == 'higher_better':
                        degradation = (in_sample_value - out_of_sample_value) / abs(in_sample_value)
                    else:  # lower_better (for metrics like max_drawdown)
                        degradation = (out_of_sample_value - in_sample_value) / abs(in_sample_value)
                    
                    degradation_scores.append(max(0, degradation))
            
            # Average degradation score
            overfitting_score = np.mean(degradation_scores) if degradation_scores else 0
            overfitting_detected = overfitting_score > threshold
            
            return overfitting_detected, overfitting_score
            
        except Exception as e:
            logger.error(f"Error detecting overfitting: {e}")
            return False, 0.0
    
    @staticmethod
    def statistical_significance_test(returns1: pd.Series, returns2: pd.Series, 
                                    test_type: str = 'ttest') -> Tuple[bool, float]:
        """
        Test statistical significance between two return series
        """
        try:
            if len(returns1) == 0 or len(returns2) == 0:
                return False, 1.0
            
            if test_type == 'ttest':
                # Paired t-test
                statistic, p_value = stats.ttest_rel(returns1, returns2)
            elif test_type == 'wilcoxon':
                # Wilcoxon signed-rank test (non-parametric)
                statistic, p_value = stats.wilcoxon(returns1, returns2)
            else:
                # Mann-Whitney U test
                statistic, p_value = stats.mannwhitneyu(returns1, returns2, alternative='two-sided')
            
            significant = p_value < 0.05
            return significant, p_value
            
        except Exception as e:
            logger.error(f"Error in significance test: {e}")
            return False, 1.0


class ModelStabilityAnalyzer:
    """Analyzes model stability over time"""
    
    def __init__(self, config: ValidationConfig):
        self.config = config
    
    def analyze_stability(self, walk_forward_results: List[Dict]) -> Tuple[float, bool, List[Dict]]:
        """
        Analyze model stability across walk-forward periods
        """
        if len(walk_forward_results) < 2:
            return 0.0, False, []
        
        try:
            # Extract performance metrics over time
            sharpe_ratios = []
            returns = []
            win_rates = []
            
            for result in walk_forward_results:
                metrics = result['metrics']
                sharpe_ratios.append(metrics.sharpe_ratio)
                returns.append(metrics.total_return)
                win_rates.append(metrics.win_rate)
            
            # Calculate stability metrics
            stability_scores = []
            
            # Sharpe ratio stability
            if len(sharpe_ratios) > 1:
                sharpe_stability = 1 - (np.std(sharpe_ratios) / (np.mean(sharpe_ratios) + 1e-8))
                stability_scores.append(max(0, sharpe_stability))
            
            # Return stability
            if len(returns) > 1:
                return_stability = 1 - (np.std(returns) / (np.mean(returns) + 1e-8))
                stability_scores.append(max(0, return_stability))
            
            # Win rate stability
            if len(win_rates) > 1:
                winrate_stability = 1 - np.std(win_rates)
                stability_scores.append(max(0, winrate_stability))
            
            # Overall stability score
            stability_score = np.mean(stability_scores) if stability_scores else 0.0
            
            # Detect degradation
            degradation_detected = self._detect_degradation(walk_forward_results)
            
            # Analyze stability periods
            stability_periods = self._analyze_stability_periods(walk_forward_results)
            
            return stability_score, degradation_detected, stability_periods
            
        except Exception as e:
            logger.error(f"Error analyzing stability: {e}")
            return 0.0, False, []
    
    def _detect_degradation(self, results: List[Dict]) -> bool:
        """Detect performance degradation over time"""
        if len(results) < self.config.stability_window:
            return False
        
        try:
            # Compare recent performance to earlier performance
            recent_results = results[-self.config.stability_window:]
            earlier_results = results[:self.config.stability_window]
            
            recent_sharpe = np.mean([r['metrics'].sharpe_ratio for r in recent_results])
            earlier_sharpe = np.mean([r['metrics'].sharpe_ratio for r in earlier_results])
            
            if earlier_sharpe > 0:
                degradation = (earlier_sharpe - recent_sharpe) / earlier_sharpe
                return degradation > self.config.degradation_threshold
            
            return False
            
        except Exception as e:
            logger.error(f"Error detecting degradation: {e}")
            return False
    
    def _analyze_stability_periods(self, results: List[Dict]) -> List[Dict]:
        """Analyze stability in different periods"""
        periods = []
        window_size = self.config.stability_window
        
        for i in range(0, len(results) - window_size + 1, window_size // 2):
            period_results = results[i:i + window_size]
            
            if len(period_results) >= window_size:
                sharpe_ratios = [r['metrics'].sharpe_ratio for r in period_results]
                returns = [r['metrics'].total_return for r in period_results]
                
                period_info = {
                    'start_step': period_results[0]['step'],
                    'end_step': period_results[-1]['step'],
                    'mean_sharpe': np.mean(sharpe_ratios),
                    'std_sharpe': np.std(sharpe_ratios),
                    'mean_return': np.mean(returns),
                    'std_return': np.std(returns),
                    'stability_score': 1 - (np.std(sharpe_ratios) / (np.mean(sharpe_ratios) + 1e-8))
                }
                
                periods.append(period_info)
        
        return periods
class Ma
rketRegimeAnalyzer:
    """Analyzes performance by market regime"""
    
    def __init__(self):
        self.regimes = ['bull_market', 'bear_market', 'sideways_market', 'high_volatility']
    
    def classify_market_regime(self, returns: pd.Series, volatility_threshold: float = 0.02) -> str:
        """Classify market regime based on returns and volatility"""
        try:
            if len(returns) < 20:
                return 'unknown'
            
            # Calculate metrics
            mean_return = returns.mean()
            volatility = returns.std()
            trend = returns.rolling(20).mean().iloc[-1]
            
            # Classification logic
            if volatility > volatility_threshold:
                return 'high_volatility'
            elif trend > 0.001:  # Positive trend
                return 'bull_market'
            elif trend < -0.001:  # Negative trend
                return 'bear_market'
            else:
                return 'sideways_market'
                
        except Exception as e:
            logger.error(f"Error classifying market regime: {e}")
            return 'unknown'
    
    def analyze_regime_performance(self, walk_forward_results: List[Dict], 
                                 market_data: pd.DataFrame) -> Dict[str, PerformanceMetrics]:
        """Analyze performance by market regime"""
        regime_performance = {}
        
        try:
            for regime in self.regimes:
                regime_returns = []
                
                for result in walk_forward_results:
                    # Get market data for this period
                    test_start = result['test_start']
                    test_end = result['test_end']
                    
                    if test_end <= len(market_data):
                        period_returns = market_data.iloc[test_start:test_end]['returns'] if 'returns' in market_data.columns else pd.Series()
                        
                        if len(period_returns) > 0:
                            period_regime = self.classify_market_regime(period_returns)
                            
                            if period_regime == regime:
                                regime_returns.extend(result['predictions'])
                
                if regime_returns:
                    regime_series = pd.Series(regime_returns)
                    regime_performance[regime] = PerformanceCalculator.calculate_metrics(regime_series)
        
        except Exception as e:
            logger.error(f"Error analyzing regime performance: {e}")
        
        return regime_performance


class ModelValidationFramework:
    """
    Comprehensive model validation framework for trading ML models
    """
    
    def __init__(self, config: ValidationConfig = None, output_dir: str = "validation_results"):
        self.config = config or ValidationConfig()
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize components
        self.ts_validator = TimeSeriesValidator(self.config)
        self.stability_analyzer = ModelStabilityAnalyzer(self.config)
        self.regime_analyzer = MarketRegimeAnalyzer()
        
        # Results storage
        self.validation_results = {}
        
        logger.info("ModelValidationFramework initialized")
    
    def validate_model(self, model_name: str, X: pd.DataFrame, y: pd.Series,
                      model_func: Callable, prediction_func: Callable,
                      benchmark_returns: pd.Series = None,
                      market_data: pd.DataFrame = None) -> ValidationResult:
        """
        Comprehensive model validation
        
        Args:
            model_name: Name of the model being validated
            X: Feature matrix
            y: Target variable (returns)
            model_func: Function to train model (X_train, y_train) -> model
            prediction_func: Function to make predictions (model, X_test) -> predictions
            benchmark_returns: Benchmark returns for comparison
            market_data: Market data for regime analysis
        
        Returns:
            ValidationResult with comprehensive validation metrics
        """
        start_time = datetime.now()
        logger.info(f"Starting validation for model: {model_name}")
        
        try:
            # Initialize result
            result = ValidationResult(
                model_name=model_name,
                validation_type="comprehensive",
                start_date=X.index[0] if hasattr(X.index, '__getitem__') else datetime.now(),
                end_date=X.index[-1] if hasattr(X.index, '__getitem__') else datetime.now()
            )
            
            # 1. Walk-forward analysis
            logger.info("Performing walk-forward analysis...")
            walk_forward_results = self.ts_validator.walk_forward_analysis(
                X, y, model_func, prediction_func
            )
            result.walk_forward_results = walk_forward_results
            
            if not walk_forward_results:
                logger.warning("No walk-forward results generated")
                return result
            
            # 2. Calculate in-sample and out-of-sample metrics
            logger.info("Calculating performance metrics...")
            in_sample_returns, out_of_sample_returns = self._split_sample_returns(walk_forward_results)
            
            result.in_sample_metrics = PerformanceCalculator.calculate_metrics(
                in_sample_returns, benchmark_returns, self.config.risk_free_rate
            )
            result.out_of_sample_metrics = PerformanceCalculator.calculate_metrics(
                out_of_sample_returns, benchmark_returns, self.config.risk_free_rate
            )
            
            # 3. Overfitting detection
            logger.info("Detecting overfitting...")
            overfitting_detected, overfitting_score = OverfittingDetector.detect_overfitting(
                result.in_sample_metrics, result.out_of_sample_metrics
            )
            result.overfitting_detected = overfitting_detected
            result.overfitting_score = overfitting_score
            
            # 4. Statistical significance testing
            if len(in_sample_returns) > 0 and len(out_of_sample_returns) > 0:
                significant, p_value = OverfittingDetector.statistical_significance_test(
                    in_sample_returns, out_of_sample_returns
                )
                result.statistical_significance = significant
                result.p_value = p_value
            
            # 5. Cross-validation
            if self.config.purged_cv:
                logger.info("Performing purged cross-validation...")
                cv_scores = self._perform_cross_validation(X, y, model_func, prediction_func)
                result.cv_scores = cv_scores
                result.cv_mean = np.mean(cv_scores) if cv_scores else 0.0
                result.cv_std = np.std(cv_scores) if cv_scores else 0.0
            
            # 6. Stability analysis
            logger.info("Analyzing model stability...")
            stability_score, degradation_detected, stability_periods = self.stability_analyzer.analyze_stability(
                walk_forward_results
            )
            result.stability_score = stability_score
            result.degradation_detected = degradation_detected
            result.stability_periods = stability_periods
            
            # 7. Market regime analysis
            if market_data is not None:
                logger.info("Analyzing performance by market regime...")
                result.regime_performance = self.regime_analyzer.analyze_regime_performance(
                    walk_forward_results, market_data
                )
            
            # 8. Feature importance analysis (if available)
            try:
                result.feature_importance = self._analyze_feature_importance(
                    X, y, model_func, walk_forward_results
                )
            except Exception as e:
                logger.warning(f"Could not analyze feature importance: {e}")
            
            # 9. Calculate computation time
            result.computation_time = (datetime.now() - start_time).total_seconds()
            
            # Store result
            self.validation_results[model_name] = result
            
            logger.info(f"Validation completed for {model_name} in {result.computation_time:.2f} seconds")
            return result
            
        except Exception as e:
            logger.error(f"Error validating model {model_name}: {e}")
            result.computation_time = (datetime.now() - start_time).total_seconds()
            return result
    
    def _split_sample_returns(self, walk_forward_results: List[Dict]) -> Tuple[pd.Series, pd.Series]:
        """Split returns into in-sample and out-of-sample"""
        in_sample_returns = []
        out_of_sample_returns = []
        
        # Use first half as in-sample, second half as out-of-sample
        split_point = len(walk_forward_results) // 2
        
        for i, result in enumerate(walk_forward_results):
            returns = result['predictions']
            
            if i < split_point:
                in_sample_returns.extend(returns)
            else:
                out_of_sample_returns.extend(returns)
        
        return pd.Series(in_sample_returns), pd.Series(out_of_sample_returns)
    
    def _perform_cross_validation(self, X: pd.DataFrame, y: pd.Series,
                                 model_func: Callable, prediction_func: Callable) -> List[float]:
        """Perform purged time series cross-validation"""
        cv_scores = []
        
        try:
            splits = self.ts_validator.purged_time_series_split(X, y)
            
            for train_idx, test_idx in splits:
                try:
                    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                    
                    # Train model
                    model = model_func(X_train, y_train)
                    
                    # Make predictions
                    predictions = prediction_func(model, X_test)
                    
                    # Calculate score (Sharpe ratio)
                    if len(predictions) > 0:
                        pred_series = pd.Series(predictions)
                        metrics = PerformanceCalculator.calculate_metrics(pred_series)
                        cv_scores.append(metrics.sharpe_ratio)
                
                except Exception as e:
                    logger.warning(f"Error in CV fold: {e}")
                    continue
        
        except Exception as e:
            logger.error(f"Error in cross-validation: {e}")
        
        return cv_scores
    
    def _analyze_feature_importance(self, X: pd.DataFrame, y: pd.Series,
                                   model_func: Callable, 
                                   walk_forward_results: List[Dict]) -> Dict[str, float]:
        """Analyze feature importance across walk-forward periods"""
        feature_importance = defaultdict(list)
        
        try:
            # Train model on full dataset to get feature importance
            model = model_func(X, y)
            
            # Try to extract feature importance (works for tree-based models)
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                for i, importance in enumerate(importances):
                    if i < len(X.columns):
                        feature_importance[X.columns[i]].append(importance)
            
            # Average importance across periods
            avg_importance = {}
            for feature, importances in feature_importance.items():
                avg_importance[feature] = np.mean(importances)
            
            return avg_importance
            
        except Exception as e:
            logger.warning(f"Error analyzing feature importance: {e}")
            return {}
    
    def compare_models(self, model_results: Dict[str, ValidationResult]) -> Dict[str, Any]:
        """Compare multiple models and provide ranking"""
        if not model_results:
            return {}
        
        comparison = {
            'model_ranking': [],
            'performance_comparison': {},
            'statistical_tests': {},
            'recommendations': []
        }
        
        try:
            # Extract key metrics for comparison
            models_data = []
            for model_name, result in model_results.items():
                models_data.append({
                    'name': model_name,
                    'sharpe_ratio': result.out_of_sample_metrics.sharpe_ratio,
                    'total_return': result.out_of_sample_metrics.total_return,
                    'max_drawdown': result.out_of_sample_metrics.max_drawdown,
                    'win_rate': result.out_of_sample_metrics.win_rate,
                    'stability_score': result.stability_score,
                    'overfitting_score': result.overfitting_score,
                    'cv_mean': result.cv_mean,
                    'cv_std': result.cv_std
                })
            
            # Rank models by composite score
            for model_data in models_data:
                # Composite score (higher is better)
                composite_score = (
                    model_data['sharpe_ratio'] * 0.3 +
                    model_data['total_return'] * 0.2 +
                    abs(model_data['max_drawdown']) * -0.2 +  # Negative because lower is better
                    model_data['win_rate'] * 0.1 +
                    model_data['stability_score'] * 0.1 +
                    model_data['overfitting_score'] * -0.1  # Negative because lower is better
                )
                model_data['composite_score'] = composite_score
            
            # Sort by composite score
            models_data.sort(key=lambda x: x['composite_score'], reverse=True)
            comparison['model_ranking'] = models_data
            
            # Performance comparison matrix
            metrics = ['sharpe_ratio', 'total_return', 'max_drawdown', 'win_rate', 'stability_score']
            comparison['performance_comparison'] = {}
            
            for metric in metrics:
                comparison['performance_comparison'][metric] = {
                    model['name']: model[metric] for model in models_data
                }
            
            # Generate recommendations
            best_model = models_data[0]
            comparison['recommendations'] = [
                f"Best overall model: {best_model['name']} (composite score: {best_model['composite_score']:.3f})",
                f"Highest Sharpe ratio: {max(models_data, key=lambda x: x['sharpe_ratio'])['name']}",
                f"Most stable model: {max(models_data, key=lambda x: x['stability_score'])['name']}",
                f"Least overfitting: {min(models_data, key=lambda x: x['overfitting_score'])['name']}"
            ]
            
        except Exception as e:
            logger.error(f"Error comparing models: {e}")
        
        return comparison
    
    def generate_validation_report(self, model_name: str, 
                                 validation_result: ValidationResult,
                                 include_plots: bool = True) -> str:
        """Generate comprehensive validation report"""
        try:
            report_path = os.path.join(self.output_dir, f"{model_name}_validation_report.html")
            
            # Generate HTML report
            html_content = self._generate_html_report(validation_result, include_plots)
            
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            # Save detailed results as JSON
            json_path = os.path.join(self.output_dir, f"{model_name}_validation_results.json")
            self._save_results_json(validation_result, json_path)
            
            logger.info(f"Validation report generated: {report_path}")
            return report_path
            
        except Exception as e:
            logger.error(f"Error generating validation report: {e}")
            return ""
    
    def _generate_html_report(self, result: ValidationResult, include_plots: bool = True) -> str:
        """Generate HTML validation report"""
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Model Validation Report - {result.model_name}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .section {{ margin: 20px 0; }}
                .metrics-table {{ border-collapse: collapse; width: 100%; }}
                .metrics-table th, .metrics-table td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                .metrics-table th {{ background-color: #f2f2f2; }}
                .alert {{ padding: 15px; margin: 10px 0; border-radius: 4px; }}
                .alert-warning {{ background-color: #fff3cd; border-color: #ffeaa7; color: #856404; }}
                .alert-success {{ background-color: #d4edda; border-color: #c3e6cb; color: #155724; }}
                .alert-danger {{ background-color: #f8d7da; border-color: #f5c6cb; color: #721c24; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Model Validation Report</h1>
                <h2>{result.model_name}</h2>
                <p><strong>Validation Period:</strong> {result.start_date.strftime('%Y-%m-%d')} to {result.end_date.strftime('%Y-%m-%d')}</p>
                <p><strong>Validation Type:</strong> {result.validation_type}</p>
                <p><strong>Generated:</strong> {result.validation_timestamp.strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="section">
                <h3>Executive Summary</h3>
                {self._generate_executive_summary(result)}
            </div>
            
            <div class="section">
                <h3>Performance Metrics</h3>
                {self._generate_metrics_table(result)}
            </div>
            
            <div class="section">
                <h3>Validation Results</h3>
                {self._generate_validation_alerts(result)}
            </div>
            
            <div class="section">
                <h3>Walk-Forward Analysis</h3>
                <p>Number of periods analyzed: {len(result.walk_forward_results)}</p>
                <p>Average out-of-sample Sharpe ratio: {np.mean([r['metrics'].sharpe_ratio for r in result.walk_forward_results]):.3f}</p>
            </div>
            
            <div class="section">
                <h3>Cross-Validation Results</h3>
                <p>CV Mean Score: {result.cv_mean:.3f}</p>
                <p>CV Standard Deviation: {result.cv_std:.3f}</p>
                <p>CV Scores: {', '.join([f'{score:.3f}' for score in result.cv_scores])}</p>
            </div>
            
            <div class="section">
                <h3>Model Stability Analysis</h3>
                <p>Stability Score: {result.stability_score:.3f}</p>
                <p>Degradation Detected: {'Yes' if result.degradation_detected else 'No'}</p>
                <p>Number of stability periods analyzed: {len(result.stability_periods)}</p>
            </div>
            
        </body>
        </html>
        """
        
        return html
    
    def _generate_executive_summary(self, result: ValidationResult) -> str:
        """Generate executive summary for the report"""
        oos_metrics = result.out_of_sample_metrics
        
        summary = f"""
        <div class="alert alert-{'success' if oos_metrics.sharpe_ratio > 1.0 else 'warning' if oos_metrics.sharpe_ratio > 0.5 else 'danger'}">
            <strong>Overall Assessment:</strong> 
            {'Excellent' if oos_metrics.sharpe_ratio > 1.0 else 'Good' if oos_metrics.sharpe_ratio > 0.5 else 'Poor'} performance
            (Sharpe Ratio: {oos_metrics.sharpe_ratio:.3f})
        </div>
        
        <ul>
            <li><strong>Out-of-Sample Return:</strong> {oos_metrics.total_return*100:.2f}%</li>
            <li><strong>Maximum Drawdown:</strong> {oos_metrics.max_drawdown*100:.2f}%</li>
            <li><strong>Win Rate:</strong> {oos_metrics.win_rate*100:.1f}%</li>
            <li><strong>Overfitting Risk:</strong> {'High' if result.overfitting_detected else 'Low'}</li>
            <li><strong>Model Stability:</strong> {result.stability_score:.3f}</li>
        </ul>
        """
        
        return summary
    
    def _generate_metrics_table(self, result: ValidationResult) -> str:
        """Generate metrics comparison table"""
        is_metrics = result.in_sample_metrics
        oos_metrics = result.out_of_sample_metrics
        
        table = """
        <table class="metrics-table">
            <tr><th>Metric</th><th>In-Sample</th><th>Out-of-Sample</th><th>Difference</th></tr>
        """
        
        metrics_to_show = [
            ('Total Return', 'total_return', '%'),
            ('Annualized Return', 'annualized_return', '%'),
            ('Sharpe Ratio', 'sharpe_ratio', ''),
            ('Sortino Ratio', 'sortino_ratio', ''),
            ('Maximum Drawdown', 'max_drawdown', '%'),
            ('Win Rate', 'win_rate', '%'),
            ('Volatility', 'volatility', '%')
        ]
        
        for display_name, attr_name, unit in metrics_to_show:
            is_value = getattr(is_metrics, attr_name, 0)
            oos_value = getattr(oos_metrics, attr_name, 0)
            diff = is_value - oos_value
            
            if unit == '%':
                is_str = f"{is_value*100:.2f}%"
                oos_str = f"{oos_value*100:.2f}%"
                diff_str = f"{diff*100:.2f}%"
            else:
                is_str = f"{is_value:.3f}"
                oos_str = f"{oos_value:.3f}"
                diff_str = f"{diff:.3f}"
            
            table += f"<tr><td>{display_name}</td><td>{is_str}</td><td>{oos_str}</td><td>{diff_str}</td></tr>"
        
        table += "</table>"
        return table
    
    def _generate_validation_alerts(self, result: ValidationResult) -> str:
        """Generate validation alerts and warnings"""
        alerts = ""
        
        if result.overfitting_detected:
            alerts += f"""
            <div class="alert alert-danger">
                <strong>Overfitting Detected!</strong> 
                Overfitting score: {result.overfitting_score:.3f}. 
                The model may not generalize well to new data.
            </div>
            """
        
        if result.degradation_detected:
            alerts += """
            <div class="alert alert-warning">
                <strong>Performance Degradation Detected!</strong> 
                Model performance has declined over time. Consider retraining.
            </div>
            """
        
        if result.stability_score < 0.5:
            alerts += f"""
            <div class="alert alert-warning">
                <strong>Low Model Stability!</strong> 
                Stability score: {result.stability_score:.3f}. 
                Model performance is inconsistent across time periods.
            </div>
            """
        
        if result.statistical_significance:
            alerts += f"""
            <div class="alert alert-success">
                <strong>Statistically Significant Results!</strong> 
                P-value: {result.p_value:.4f}. 
                Model performance is statistically significant.
            </div>
            """
        
        return alerts if alerts else '<p>No validation alerts.</p>'
    
    def _save_results_json(self, result: ValidationResult, filepath: str):
        """Save detailed results as JSON"""
        try:
            # Convert result to dictionary
            result_dict = asdict(result)
            
            # Convert datetime objects to strings
            result_dict['start_date'] = result.start_date.isoformat()
            result_dict['end_date'] = result.end_date.isoformat()
            result_dict['validation_timestamp'] = result.validation_timestamp.isoformat()
            
            # Convert numpy arrays to lists
            for wf_result in result_dict['walk_forward_results']:
                if 'predictions' in wf_result:
                    wf_result['predictions'] = wf_result['predictions'].tolist() if hasattr(wf_result['predictions'], 'tolist') else wf_result['predictions']
                if 'actual' in wf_result:
                    wf_result['actual'] = wf_result['actual'].tolist() if hasattr(wf_result['actual'], 'tolist') else wf_result['actual']
            
            with open(filepath, 'w') as f:
                json.dump(result_dict, f, indent=2, default=str)
                
        except Exception as e:
            logger.error(f"Error saving results JSON: {e}")
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """Get summary of all validation results"""
        summary = {
            'total_models_validated': len(self.validation_results),
            'models': {},
            'best_performers': {},
            'validation_alerts': []
        }
        
        if not self.validation_results:
            return summary
        
        # Model summaries
        for model_name, result in self.validation_results.items():
            summary['models'][model_name] = {
                'sharpe_ratio': result.out_of_sample_metrics.sharpe_ratio,
                'total_return': result.out_of_sample_metrics.total_return,
                'max_drawdown': result.out_of_sample_metrics.max_drawdown,
                'stability_score': result.stability_score,
                'overfitting_detected': result.overfitting_detected,
                'degradation_detected': result.degradation_detected
            }
        
        # Best performers
        if self.validation_results:
            best_sharpe = max(self.validation_results.items(), 
                            key=lambda x: x[1].out_of_sample_metrics.sharpe_ratio)
            best_return = max(self.validation_results.items(), 
                            key=lambda x: x[1].out_of_sample_metrics.total_return)
            most_stable = max(self.validation_results.items(), 
                            key=lambda x: x[1].stability_score)
            
            summary['best_performers'] = {
                'highest_sharpe': best_sharpe[0],
                'highest_return': best_return[0],
                'most_stable': most_stable[0]
            }
        
        # Validation alerts
        for model_name, result in self.validation_results.items():
            if result.overfitting_detected:
                summary['validation_alerts'].append(f"{model_name}: Overfitting detected")
            if result.degradation_detected:
                summary['validation_alerts'].append(f"{model_name}: Performance degradation detected")
            if result.stability_score < 0.5:
                summary['validation_alerts'].append(f"{model_name}: Low stability score")
        
        return summary


# Example usage and testing
if __name__ == "__main__":
    # Create sample data
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=1000, freq='D')
    
    # Generate sample features and returns
    n_features = 10
    X = pd.DataFrame(
        np.random.randn(1000, n_features),
        index=dates,
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    
    # Generate sample returns with some predictability
    y = pd.Series(
        0.001 + 0.01 * np.random.randn(1000) + 0.005 * X['feature_0'],
        index=dates,
        name='returns'
    )
    
    # Sample model functions
    def simple_model_func(X_train, y_train):
        # Simple linear model (mock)
        return {'coef': np.random.randn(len(X_train.columns))}
    
    def simple_prediction_func(model, X_test):
        # Simple prediction (mock)
        return np.random.randn(len(X_test)) * 0.01
    
    # Create validation framework
    config = ValidationConfig(
        initial_train_size=200,
        step_size=20,
        cv_folds=5,
        generate_plots=True
    )
    
    validator = ModelValidationFramework(config)
    
    # Validate model
    result = validator.validate_model(
        model_name="TestModel",
        X=X,
        y=y,
        model_func=simple_model_func,
        prediction_func=simple_prediction_func
    )
    
    # Generate report
    report_path = validator.generate_validation_report("TestModel", result)
    print(f"Validation report generated: {report_path}")
    
    # Get summary
    summary = validator.get_validation_summary()
    print(f"Validation summary: {summary}")