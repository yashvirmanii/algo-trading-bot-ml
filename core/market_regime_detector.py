"""
Market Regime Detector using Unsupervised Learning.

This module implements an advanced market regime detection system that:
- Classifies market into: trending_up, trending_down, sideways, volatile
- Uses clustering algorithms (GMM or K-means) on market features
- Features: volatility, volume, price momentum, correlation patterns
- Updates regime classification every 15 minutes
- Provides confidence scores for regime predictions
- Includes regime transition smoothing and history tracking
- Implements regime-based strategy switching logic
"""

import logging
import os
import json
import pickle
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Any, Union
from collections import deque, defaultdict
from enum import Enum
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from scipy import stats
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    """Market regime classifications."""
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    SIDEWAYS = "sideways"
    VOLATILE = "volatile"
    UNKNOWN = "unknown"


@dataclass
class RegimeFeatures:
    """Market regime features."""
    # Volatility features
    price_volatility: float  # Price volatility (ATR-based)
    volume_volatility: float  # Volume volatility
    intraday_range: float  # High-low range
    
    # Momentum features
    price_momentum_short: float  # Short-term momentum
    price_momentum_medium: float  # Medium-term momentum
    price_momentum_long: float  # Long-term momentum
    
    # Volume features
    volume_trend: float  # Volume trend
    volume_ratio: float  # Volume vs average
    volume_price_correlation: float  # Volume-price correlation
    
    # Trend features
    trend_strength: float  # Trend strength indicator
    trend_consistency: float  # Trend consistency
    support_resistance_strength: float  # S/R level strength
    
    # Market structure features
    higher_highs_lows: float  # Higher highs and lows pattern
    breakout_frequency: float  # Frequency of breakouts
    mean_reversion_tendency: float  # Mean reversion strength
    
    def to_array(self) -> np.ndarray:
        """Convert features to numpy array."""
        return np.array([
            self.price_volatility, self.volume_volatility, self.intraday_range,
            self.price_momentum_short, self.price_momentum_medium, self.price_momentum_long,
            self.volume_trend, self.volume_ratio, self.volume_price_correlation,
            self.trend_strength, self.trend_consistency, self.support_resistance_strength,
            self.higher_highs_lows, self.breakout_frequency, self.mean_reversion_tendency
        ])
    
    @classmethod
    def get_feature_names(cls) -> List[str]:
        """Get feature names."""
        return [
            'price_volatility', 'volume_volatility', 'intraday_range',
            'price_momentum_short', 'price_momentum_medium', 'price_momentum_long',
            'volume_trend', 'volume_ratio', 'volume_price_correlation',
            'trend_strength', 'trend_consistency', 'support_resistance_strength',
            'higher_highs_lows', 'breakout_frequency', 'mean_reversion_tendency'
        ]


@dataclass
class RegimeClassification:
    """Market regime classification result."""
    regime: MarketRegime
    confidence: float
    features: RegimeFeatures
    cluster_probabilities: Dict[str, float]
    timestamp: datetime
    smoothed_regime: Optional[MarketRegime] = None
    transition_probability: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        data['regime'] = self.regime.value
        data['smoothed_regime'] = self.smoothed_regime.value if self.smoothed_regime else None
        data['timestamp'] = self.timestamp.isoformat()
        return data


@dataclass
class RegimeTransition:
    """Market regime transition record."""
    from_regime: MarketRegime
    to_regime: MarketRegime
    timestamp: datetime
    confidence: float
    duration_minutes: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'from_regime': self.from_regime.value,
            'to_regime': self.to_regime.value,
            'timestamp': self.timestamp.isoformat(),
            'confidence': self.confidence,
            'duration_minutes': self.duration_minutes
        }


class FeatureEngineer:
    """Feature engineering pipeline for market regime detection."""
    
    def __init__(self, lookback_periods: Dict[str, int] = None):
        """
        Initialize feature engineer.
        
        Args:
            lookback_periods: Dictionary of lookback periods for different calculations
        """
        self.lookback_periods = lookback_periods or {
            'short': 5,
            'medium': 20,
            'long': 50,
            'volatility': 20,
            'volume': 20,
            'correlation': 30
        }
    
    def extract_features(self, df: pd.DataFrame) -> Optional[RegimeFeatures]:
        """
        Extract regime features from price data.
        
        Args:
            df: Price dataframe with OHLCV data
            
        Returns:
            RegimeFeatures object or None if insufficient data
        """
        try:
            if len(df) < max(self.lookback_periods.values()):
                logger.warning(f"Insufficient data for feature extraction: {len(df)} rows")
                return None
            
            # Volatility features
            price_volatility = self._calculate_price_volatility(df)
            volume_volatility = self._calculate_volume_volatility(df)
            intraday_range = self._calculate_intraday_range(df)
            
            # Momentum features
            price_momentum_short = self._calculate_momentum(df['close'], self.lookback_periods['short'])
            price_momentum_medium = self._calculate_momentum(df['close'], self.lookback_periods['medium'])
            price_momentum_long = self._calculate_momentum(df['close'], self.lookback_periods['long'])
            
            # Volume features
            volume_trend = self._calculate_volume_trend(df)
            volume_ratio = self._calculate_volume_ratio(df)
            volume_price_correlation = self._calculate_volume_price_correlation(df)
            
            # Trend features
            trend_strength = self._calculate_trend_strength(df)
            trend_consistency = self._calculate_trend_consistency(df)
            support_resistance_strength = self._calculate_support_resistance_strength(df)
            
            # Market structure features
            higher_highs_lows = self._calculate_higher_highs_lows(df)
            breakout_frequency = self._calculate_breakout_frequency(df)
            mean_reversion_tendency = self._calculate_mean_reversion_tendency(df)
            
            return RegimeFeatures(
                price_volatility=price_volatility,
                volume_volatility=volume_volatility,
                intraday_range=intraday_range,
                price_momentum_short=price_momentum_short,
                price_momentum_medium=price_momentum_medium,
                price_momentum_long=price_momentum_long,
                volume_trend=volume_trend,
                volume_ratio=volume_ratio,
                volume_price_correlation=volume_price_correlation,
                trend_strength=trend_strength,
                trend_consistency=trend_consistency,
                support_resistance_strength=support_resistance_strength,
                higher_highs_lows=higher_highs_lows,
                breakout_frequency=breakout_frequency,
                mean_reversion_tendency=mean_reversion_tendency
            )
            
        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            return None
    
    def _calculate_price_volatility(self, df: pd.DataFrame) -> float:
        """Calculate price volatility using ATR-based measure."""
        try:
            high_low = df['high'] - df['low']
            high_close = np.abs(df['high'] - df['close'].shift(1))
            low_close = np.abs(df['low'] - df['close'].shift(1))
            
            true_range = np.maximum(high_low, np.maximum(high_close, low_close))
            atr = true_range.rolling(self.lookback_periods['volatility']).mean().iloc[-1]
            
            # Normalize by price
            normalized_atr = atr / df['close'].iloc[-1]
            
            return float(normalized_atr) if not np.isnan(normalized_atr) else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating price volatility: {e}")
            return 0.0
    
    def _calculate_volume_volatility(self, df: pd.DataFrame) -> float:
        """Calculate volume volatility."""
        try:
            volume_returns = df['volume'].pct_change().dropna()
            vol_volatility = volume_returns.rolling(self.lookback_periods['volatility']).std().iloc[-1]
            
            return float(vol_volatility) if not np.isnan(vol_volatility) else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating volume volatility: {e}")
            return 0.0
    
    def _calculate_intraday_range(self, df: pd.DataFrame) -> float:
        """Calculate normalized intraday range."""
        try:
            intraday_range = (df['high'] - df['low']) / df['close']
            avg_range = intraday_range.rolling(self.lookback_periods['volatility']).mean().iloc[-1]
            
            return float(avg_range) if not np.isnan(avg_range) else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating intraday range: {e}")
            return 0.0
    
    def _calculate_momentum(self, prices: pd.Series, period: int) -> float:
        """Calculate price momentum over specified period."""
        try:
            if len(prices) < period + 1:
                return 0.0
            
            momentum = (prices.iloc[-1] - prices.iloc[-period-1]) / prices.iloc[-period-1]
            
            return float(momentum) if not np.isnan(momentum) else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating momentum: {e}")
            return 0.0
    
    def _calculate_volume_trend(self, df: pd.DataFrame) -> float:
        """Calculate volume trend using linear regression slope."""
        try:
            recent_volume = df['volume'].tail(self.lookback_periods['volume'])
            x = np.arange(len(recent_volume))
            
            if len(recent_volume) < 3:
                return 0.0
            
            slope, _, r_value, _, _ = stats.linregress(x, recent_volume)
            
            # Normalize slope by average volume
            avg_volume = recent_volume.mean()
            normalized_slope = slope / avg_volume if avg_volume > 0 else 0.0
            
            return float(normalized_slope) if not np.isnan(normalized_slope) else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating volume trend: {e}")
            return 0.0
    
    def _calculate_volume_ratio(self, df: pd.DataFrame) -> float:
        """Calculate current volume ratio to average."""
        try:
            current_volume = df['volume'].iloc[-1]
            avg_volume = df['volume'].rolling(self.lookback_periods['volume']).mean().iloc[-1]
            
            ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
            
            return float(ratio) if not np.isnan(ratio) else 1.0
            
        except Exception as e:
            logger.error(f"Error calculating volume ratio: {e}")
            return 1.0
    
    def _calculate_volume_price_correlation(self, df: pd.DataFrame) -> float:
        """Calculate volume-price correlation."""
        try:
            price_changes = df['close'].pct_change().dropna()
            volume_changes = df['volume'].pct_change().dropna()
            
            # Align series
            min_len = min(len(price_changes), len(volume_changes))
            if min_len < self.lookback_periods['correlation']:
                return 0.0
            
            price_changes = price_changes.tail(min_len)
            volume_changes = volume_changes.tail(min_len)
            
            correlation = price_changes.corr(volume_changes)
            
            return float(correlation) if not np.isnan(correlation) else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating volume-price correlation: {e}")
            return 0.0
    
    def _calculate_trend_strength(self, df: pd.DataFrame) -> float:
        """Calculate trend strength using ADX-like measure."""
        try:
            high = df['high']
            low = df['low']
            close = df['close']
            
            # Calculate directional movement
            plus_dm = high.diff()
            minus_dm = low.diff() * -1
            
            plus_dm[plus_dm < 0] = 0
            minus_dm[minus_dm < 0] = 0
            
            # True range
            tr1 = high - low
            tr2 = np.abs(high - close.shift(1))
            tr3 = np.abs(low - close.shift(1))
            true_range = np.maximum(tr1, np.maximum(tr2, tr3))
            
            # Smooth the values
            period = self.lookback_periods['medium']
            plus_di = 100 * (plus_dm.rolling(period).mean() / true_range.rolling(period).mean())
            minus_di = 100 * (minus_dm.rolling(period).mean() / true_range.rolling(period).mean())
            
            # Calculate ADX
            dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
            adx = dx.rolling(period).mean().iloc[-1]
            
            return float(adx / 100) if not np.isnan(adx) else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating trend strength: {e}")
            return 0.0
    
    def _calculate_trend_consistency(self, df: pd.DataFrame) -> float:
        """Calculate trend consistency."""
        try:
            prices = df['close']
            returns = prices.pct_change().dropna()
            
            if len(returns) < self.lookback_periods['medium']:
                return 0.0
            
            recent_returns = returns.tail(self.lookback_periods['medium'])
            
            # Calculate consistency as the ratio of same-direction moves
            positive_moves = (recent_returns > 0).sum()
            negative_moves = (recent_returns < 0).sum()
            total_moves = len(recent_returns)
            
            consistency = max(positive_moves, negative_moves) / total_moves
            
            return float(consistency) if not np.isnan(consistency) else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating trend consistency: {e}")
            return 0.0
    
    def _calculate_support_resistance_strength(self, df: pd.DataFrame) -> float:
        """Calculate support/resistance level strength."""
        try:
            prices = df['close']
            highs = df['high']
            lows = df['low']
            
            if len(prices) < self.lookback_periods['medium']:
                return 0.0
            
            # Find local maxima and minima
            recent_data = df.tail(self.lookback_periods['medium'])
            
            # Simple approach: count price touches near current levels
            current_price = prices.iloc[-1]
            price_range = current_price * 0.02  # 2% range
            
            touches = 0
            for _, row in recent_data.iterrows():
                if abs(row['high'] - current_price) <= price_range or abs(row['low'] - current_price) <= price_range:
                    touches += 1
            
            strength = touches / len(recent_data)
            
            return float(strength) if not np.isnan(strength) else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating support/resistance strength: {e}")
            return 0.0
    
    def _calculate_higher_highs_lows(self, df: pd.DataFrame) -> float:
        """Calculate higher highs and higher lows pattern strength."""
        try:
            highs = df['high']
            lows = df['low']
            
            if len(highs) < self.lookback_periods['medium']:
                return 0.0
            
            recent_highs = highs.tail(self.lookback_periods['medium'])
            recent_lows = lows.tail(self.lookback_periods['medium'])
            
            # Count higher highs and higher lows
            higher_highs = 0
            higher_lows = 0
            
            for i in range(1, len(recent_highs)):
                if recent_highs.iloc[i] > recent_highs.iloc[i-1]:
                    higher_highs += 1
                if recent_lows.iloc[i] > recent_lows.iloc[i-1]:
                    higher_lows += 1
            
            # Calculate pattern strength
            total_comparisons = len(recent_highs) - 1
            pattern_strength = (higher_highs + higher_lows) / (2 * total_comparisons) if total_comparisons > 0 else 0.0
            
            return float(pattern_strength) if not np.isnan(pattern_strength) else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating higher highs/lows: {e}")
            return 0.0
    
    def _calculate_breakout_frequency(self, df: pd.DataFrame) -> float:
        """Calculate frequency of breakouts from trading ranges."""
        try:
            prices = df['close']
            
            if len(prices) < self.lookback_periods['medium']:
                return 0.0
            
            recent_prices = prices.tail(self.lookback_periods['medium'])
            
            # Calculate rolling max and min
            window = 5
            rolling_max = recent_prices.rolling(window).max()
            rolling_min = recent_prices.rolling(window).min()
            
            # Count breakouts
            breakouts = 0
            for i in range(window, len(recent_prices)):
                current_price = recent_prices.iloc[i]
                prev_max = rolling_max.iloc[i-1]
                prev_min = rolling_min.iloc[i-1]
                
                if current_price > prev_max or current_price < prev_min:
                    breakouts += 1
            
            frequency = breakouts / (len(recent_prices) - window) if len(recent_prices) > window else 0.0
            
            return float(frequency) if not np.isnan(frequency) else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating breakout frequency: {e}")
            return 0.0
    
    def _calculate_mean_reversion_tendency(self, df: pd.DataFrame) -> float:
        """Calculate mean reversion tendency."""
        try:
            prices = df['close']
            
            if len(prices) < self.lookback_periods['medium']:
                return 0.0
            
            recent_prices = prices.tail(self.lookback_periods['medium'])
            
            # Calculate distance from moving average
            sma = recent_prices.rolling(self.lookback_periods['short']).mean()
            deviations = (recent_prices - sma) / sma
            
            # Calculate how often price reverts to mean
            reversions = 0
            for i in range(1, len(deviations)):
                if not np.isnan(deviations.iloc[i]) and not np.isnan(deviations.iloc[i-1]):
                    if deviations.iloc[i] * deviations.iloc[i-1] < 0:  # Sign change
                        reversions += 1
            
            reversion_tendency = reversions / (len(deviations) - 1) if len(deviations) > 1 else 0.0
            
            return float(reversion_tendency) if not np.isnan(reversion_tendency) else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating mean reversion tendency: {e}")
            return 0.0cla
ss MarketRegimeDetector:
    """
    Market regime detector using unsupervised learning.
    
    Classifies market conditions into different regimes and provides
    confidence scores and transition smoothing.
    """
    
    def __init__(self, 
                 data_dir: str = "data",
                 update_frequency_minutes: int = 15,
                 clustering_method: str = "gmm",
                 n_clusters: int = 4,
                 smoothing_window: int = 3,
                 min_confidence_threshold: float = 0.6):
        """
        Initialize market regime detector.
        
        Args:
            data_dir: Directory for data storage
            update_frequency_minutes: How often to update regime classification
            clustering_method: 'gmm' or 'kmeans'
            n_clusters: Number of clusters (should be 4 for the regimes)
            smoothing_window: Window size for regime transition smoothing
            min_confidence_threshold: Minimum confidence for regime classification
        """
        self.data_dir = data_dir
        self.update_frequency_minutes = update_frequency_minutes
        self.clustering_method = clustering_method.lower()
        self.n_clusters = n_clusters
        self.smoothing_window = smoothing_window
        self.min_confidence_threshold = min_confidence_threshold
        
        # Initialize components
        self.feature_engineer = FeatureEngineer()
        self.scaler = RobustScaler()
        self.pca = PCA(n_components=0.95)  # Keep 95% of variance
        self.clustering_model = None
        
        # Regime tracking
        self.regime_history: deque = deque(maxlen=1000)  # Keep last 1000 classifications
        self.transition_history: List[RegimeTransition] = []
        self.current_regime: Optional[RegimeClassification] = None
        self.last_update: Optional[datetime] = None
        
        # Regime mapping (cluster to regime)
        self.cluster_to_regime: Dict[int, MarketRegime] = {}
        
        # Model state
        self.is_trained = False
        self.training_data: List[RegimeFeatures] = []
        
        # File paths
        self.model_file = os.path.join(data_dir, "regime_detector_model.pkl")
        self.history_file = os.path.join(data_dir, "regime_history.json")
        self.transitions_file = os.path.join(data_dir, "regime_transitions.json")
        
        # Ensure data directory exists
        os.makedirs(data_dir, exist_ok=True)
        
        # Load existing data
        self._load_model_and_history()
        
        logger.info(f"MarketRegimeDetector initialized with {clustering_method} clustering")
    
    def train_model(self, historical_data: List[pd.DataFrame], regime_labels: Optional[List[str]] = None) -> bool:
        """
        Train the clustering model on historical data.
        
        Args:
            historical_data: List of price dataframes for training
            regime_labels: Optional manual regime labels for supervised mapping
            
        Returns:
            True if training successful, False otherwise
        """
        try:
            logger.info("Training market regime detection model...")
            
            # Extract features from all historical data
            all_features = []
            valid_labels = []
            
            for i, df in enumerate(historical_data):
                features = self.feature_engineer.extract_features(df)
                if features is not None:
                    all_features.append(features.to_array())
                    self.training_data.append(features)
                    
                    if regime_labels and i < len(regime_labels):
                        valid_labels.append(regime_labels[i])
            
            if len(all_features) < self.n_clusters * 5:  # Need at least 5 samples per cluster
                logger.error(f"Insufficient training data: {len(all_features)} samples")
                return False
            
            # Convert to numpy array
            feature_matrix = np.array(all_features)
            
            # Scale features
            feature_matrix_scaled = self.scaler.fit_transform(feature_matrix)
            
            # Apply PCA for dimensionality reduction
            feature_matrix_pca = self.pca.fit_transform(feature_matrix_scaled)
            
            # Train clustering model
            if self.clustering_method == "gmm":
                self.clustering_model = GaussianMixture(
                    n_components=self.n_clusters,
                    covariance_type='full',
                    random_state=42,
                    max_iter=200
                )
            else:  # kmeans
                self.clustering_model = KMeans(
                    n_clusters=self.n_clusters,
                    random_state=42,
                    n_init=10,
                    max_iter=300
                )
            
            # Fit the model
            cluster_labels = self.clustering_model.fit_predict(feature_matrix_pca)
            
            # Map clusters to regimes
            self._map_clusters_to_regimes(feature_matrix, cluster_labels, valid_labels)
            
            # Evaluate clustering quality
            silhouette_avg = silhouette_score(feature_matrix_pca, cluster_labels)
            calinski_score = calinski_harabasz_score(feature_matrix_pca, cluster_labels)
            
            logger.info(f"Clustering quality - Silhouette: {silhouette_avg:.3f}, Calinski-Harabasz: {calinski_score:.3f}")
            
            self.is_trained = True
            
            # Save the trained model
            self._save_model()
            
            logger.info("Model training completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error training model: {e}")
            return False
    
    def _map_clusters_to_regimes(self, 
                                feature_matrix: np.ndarray, 
                                cluster_labels: np.ndarray,
                                manual_labels: Optional[List[str]] = None):
        """Map cluster labels to market regimes based on feature characteristics."""
        try:
            cluster_characteristics = {}
            
            # Calculate characteristics for each cluster
            for cluster_id in range(self.n_clusters):
                cluster_mask = cluster_labels == cluster_id
                cluster_features = feature_matrix[cluster_mask]
                
                if len(cluster_features) == 0:
                    continue
                
                # Calculate mean features for this cluster
                mean_features = np.mean(cluster_features, axis=0)
                
                characteristics = {
                    'volatility': mean_features[0] + mean_features[1] + mean_features[2],  # Combined volatility
                    'momentum': mean_features[3] + mean_features[4] + mean_features[5],  # Combined momentum
                    'trend_strength': mean_features[9],  # Trend strength
                    'mean_reversion': mean_features[14],  # Mean reversion tendency
                    'sample_count': len(cluster_features)
                }
                
                cluster_characteristics[cluster_id] = characteristics
            
            # Map clusters to regimes based on characteristics
            sorted_clusters = sorted(cluster_characteristics.items(), 
                                   key=lambda x: x[1]['volatility'])
            
            for i, (cluster_id, chars) in enumerate(sorted_clusters):
                if chars['momentum'] > 0.1 and chars['trend_strength'] > 0.3:
                    # High momentum and trend strength -> Trending up
                    self.cluster_to_regime[cluster_id] = MarketRegime.TRENDING_UP
                elif chars['momentum'] < -0.1 and chars['trend_strength'] > 0.3:
                    # Negative momentum and trend strength -> Trending down
                    self.cluster_to_regime[cluster_id] = MarketRegime.TRENDING_DOWN
                elif chars['volatility'] > np.mean([c['volatility'] for c in cluster_characteristics.values()]):
                    # High volatility -> Volatile
                    self.cluster_to_regime[cluster_id] = MarketRegime.VOLATILE
                else:
                    # Low volatility and momentum -> Sideways
                    self.cluster_to_regime[cluster_id] = MarketRegime.SIDEWAYS
            
            # Ensure all regimes are represented
            used_regimes = set(self.cluster_to_regime.values())
            all_regimes = {MarketRegime.TRENDING_UP, MarketRegime.TRENDING_DOWN, 
                          MarketRegime.SIDEWAYS, MarketRegime.VOLATILE}
            
            missing_regimes = all_regimes - used_regimes
            unassigned_clusters = [i for i in range(self.n_clusters) if i not in self.cluster_to_regime]
            
            # Assign missing regimes to unassigned clusters
            for regime, cluster_id in zip(missing_regimes, unassigned_clusters):
                self.cluster_to_regime[cluster_id] = regime
            
            logger.info(f"Cluster to regime mapping: {self.cluster_to_regime}")
            
        except Exception as e:
            logger.error(f"Error mapping clusters to regimes: {e}")
            # Fallback mapping
            regime_list = [MarketRegime.TRENDING_UP, MarketRegime.TRENDING_DOWN, 
                          MarketRegime.SIDEWAYS, MarketRegime.VOLATILE]
            for i in range(min(self.n_clusters, len(regime_list))):
                self.cluster_to_regime[i] = regime_list[i]
    
    def detect_regime(self, df: pd.DataFrame, force_update: bool = False) -> Optional[RegimeClassification]:
        """
        Detect current market regime.
        
        Args:
            df: Price dataframe
            force_update: Force regime detection even if not due for update
            
        Returns:
            RegimeClassification object or None if detection failed
        """
        try:
            # Check if update is needed
            if not force_update and not self._should_update():
                return self.current_regime
            
            if not self.is_trained:
                logger.warning("Model not trained, cannot detect regime")
                return None
            
            # Extract features
            features = self.feature_engineer.extract_features(df)
            if features is None:
                logger.warning("Could not extract features for regime detection")
                return None
            
            # Prepare features for prediction
            feature_array = features.to_array().reshape(1, -1)
            feature_scaled = self.scaler.transform(feature_array)
            feature_pca = self.pca.transform(feature_scaled)
            
            # Predict cluster
            if self.clustering_method == "gmm":
                cluster_probs = self.clustering_model.predict_proba(feature_pca)[0]
                predicted_cluster = np.argmax(cluster_probs)
                confidence = cluster_probs[predicted_cluster]
                
                # Create probability dictionary
                cluster_probabilities = {
                    self.cluster_to_regime.get(i, MarketRegime.UNKNOWN).value: float(prob)
                    for i, prob in enumerate(cluster_probs)
                }
            else:  # kmeans
                predicted_cluster = self.clustering_model.predict(feature_pca)[0]
                
                # Calculate confidence based on distance to cluster centers
                distances = np.linalg.norm(feature_pca - self.clustering_model.cluster_centers_, axis=1)
                min_distance = distances[predicted_cluster]
                confidence = 1.0 / (1.0 + min_distance)  # Convert distance to confidence
                
                # Create probability dictionary (simplified for k-means)
                cluster_probabilities = {regime.value: 0.0 for regime in MarketRegime}
                predicted_regime = self.cluster_to_regime.get(predicted_cluster, MarketRegime.UNKNOWN)
                cluster_probabilities[predicted_regime.value] = confidence
            
            # Map cluster to regime
            predicted_regime = self.cluster_to_regime.get(predicted_cluster, MarketRegime.UNKNOWN)
            
            # Create classification result
            classification = RegimeClassification(
                regime=predicted_regime,
                confidence=confidence,
                features=features,
                cluster_probabilities=cluster_probabilities,
                timestamp=datetime.now()
            )
            
            # Apply smoothing
            smoothed_classification = self._apply_smoothing(classification)
            
            # Update current regime
            self.current_regime = smoothed_classification
            self.last_update = datetime.now()
            
            # Add to history
            self.regime_history.append(smoothed_classification)
            
            # Check for regime transitions
            self._check_regime_transition(smoothed_classification)
            
            # Save history
            self._save_history()
            
            logger.debug(f"Detected regime: {predicted_regime.value} (confidence: {confidence:.3f})")
            
            return smoothed_classification
            
        except Exception as e:
            logger.error(f"Error detecting regime: {e}")
            return None
    
    def _should_update(self) -> bool:
        """Check if regime detection should be updated."""
        if self.last_update is None:
            return True
        
        time_since_update = datetime.now() - self.last_update
        return time_since_update.total_seconds() >= self.update_frequency_minutes * 60
    
    def _apply_smoothing(self, classification: RegimeClassification) -> RegimeClassification:
        """Apply smoothing to regime transitions."""
        try:
            if len(self.regime_history) < self.smoothing_window:
                classification.smoothed_regime = classification.regime
                return classification
            
            # Get recent regime history
            recent_regimes = [r.regime for r in list(self.regime_history)[-self.smoothing_window:]]
            recent_regimes.append(classification.regime)
            
            # Count regime occurrences
            regime_counts = defaultdict(int)
            for regime in recent_regimes:
                regime_counts[regime] += 1
            
            # Find most common regime
            most_common_regime = max(regime_counts.items(), key=lambda x: x[1])[0]
            
            # Calculate transition probability
            if len(self.regime_history) > 0:
                last_regime = self.regime_history[-1].regime
                if last_regime != classification.regime:
                    # Calculate how often this transition occurs
                    transition_count = 0
                    total_transitions = 0
                    
                    for i in range(1, len(self.regime_history)):
                        if self.regime_history[i-1].regime == last_regime:
                            total_transitions += 1
                            if self.regime_history[i].regime == classification.regime:
                                transition_count += 1
                    
                    transition_probability = transition_count / total_transitions if total_transitions > 0 else 0.5
                else:
                    transition_probability = 0.0
            else:
                transition_probability = 0.0
            
            # Apply smoothing based on confidence and transition probability
            if (classification.confidence >= self.min_confidence_threshold and 
                transition_probability >= 0.3):
                smoothed_regime = classification.regime
            else:
                smoothed_regime = most_common_regime
            
            classification.smoothed_regime = smoothed_regime
            classification.transition_probability = transition_probability
            
            return classification
            
        except Exception as e:
            logger.error(f"Error applying smoothing: {e}")
            classification.smoothed_regime = classification.regime
            return classification
    
    def _check_regime_transition(self, classification: RegimeClassification):
        """Check for regime transitions and record them."""
        try:
            if len(self.regime_history) < 2:
                return
            
            previous_regime = self.regime_history[-2].smoothed_regime or self.regime_history[-2].regime
            current_regime = classification.smoothed_regime or classification.regime
            
            if previous_regime != current_regime:
                # Calculate duration of previous regime
                duration_minutes = 0
                for i in range(len(self.regime_history) - 2, -1, -1):
                    regime_record = self.regime_history[i]
                    record_regime = regime_record.smoothed_regime or regime_record.regime
                    
                    if record_regime == previous_regime:
                        duration_minutes += self.update_frequency_minutes
                    else:
                        break
                
                # Create transition record
                transition = RegimeTransition(
                    from_regime=previous_regime,
                    to_regime=current_regime,
                    timestamp=classification.timestamp,
                    confidence=classification.confidence,
                    duration_minutes=duration_minutes
                )
                
                self.transition_history.append(transition)
                
                # Keep only recent transitions (last 100)
                if len(self.transition_history) > 100:
                    self.transition_history = self.transition_history[-100:]
                
                logger.info(f"Regime transition: {previous_regime.value} -> {current_regime.value} "
                           f"(duration: {duration_minutes} min, confidence: {classification.confidence:.3f})")
            
        except Exception as e:
            logger.error(f"Error checking regime transition: {e}")
    
    def get_regime_statistics(self, days: int = 30) -> Dict[str, Any]:
        """Get regime statistics for the specified period."""
        try:
            cutoff_time = datetime.now() - timedelta(days=days)
            
            # Filter recent regime history
            recent_regimes = [
                r for r in self.regime_history 
                if r.timestamp > cutoff_time
            ]
            
            if not recent_regimes:
                return {"error": "No regime data available for specified period"}
            
            # Calculate regime distribution
            regime_counts = defaultdict(int)
            total_time = 0
            
            for regime_record in recent_regimes:
                regime = regime_record.smoothed_regime or regime_record.regime
                regime_counts[regime.value] += 1
                total_time += self.update_frequency_minutes
            
            # Calculate percentages
            regime_percentages = {
                regime: (count / len(recent_regimes)) * 100
                for regime, count in regime_counts.items()
            }
            
            # Calculate average confidence by regime
            regime_confidences = defaultdict(list)
            for regime_record in recent_regimes:
                regime = regime_record.smoothed_regime or regime_record.regime
                regime_confidences[regime.value].append(regime_record.confidence)
            
            avg_confidences = {
                regime: np.mean(confidences)
                for regime, confidences in regime_confidences.items()
            }
            
            # Recent transitions
            recent_transitions = [
                t for t in self.transition_history
                if t.timestamp > cutoff_time
            ]
            
            return {
                "period_days": days,
                "total_classifications": len(recent_regimes),
                "regime_distribution": regime_percentages,
                "average_confidences": avg_confidences,
                "total_transitions": len(recent_transitions),
                "current_regime": self.current_regime.regime.value if self.current_regime else None,
                "current_confidence": self.current_regime.confidence if self.current_regime else None,
                "last_update": self.last_update.isoformat() if self.last_update else None
            }
            
        except Exception as e:
            logger.error(f"Error getting regime statistics: {e}")
            return {"error": str(e)}
    
    def get_strategy_recommendations(self) -> Dict[str, Any]:
        """Get strategy recommendations based on current regime."""
        try:
            if not self.current_regime:
                return {"error": "No current regime available"}
            
            current_regime = self.current_regime.smoothed_regime or self.current_regime.regime
            confidence = self.current_regime.confidence
            
            recommendations = {
                "current_regime": current_regime.value,
                "confidence": confidence,
                "recommended_strategies": [],
                "risk_adjustments": {},
                "position_sizing": "normal"
            }
            
            # Strategy recommendations based on regime
            if current_regime == MarketRegime.TRENDING_UP:
                recommendations["recommended_strategies"] = [
                    "momentum", "trend_following", "breakout"
                ]
                recommendations["risk_adjustments"] = {
                    "stop_loss_multiplier": 1.2,
                    "take_profit_multiplier": 1.5,
                    "position_hold_time": "extended"
                }
                recommendations["position_sizing"] = "aggressive" if confidence > 0.8 else "normal"
                
            elif current_regime == MarketRegime.TRENDING_DOWN:
                recommendations["recommended_strategies"] = [
                    "short_momentum", "trend_following_short", "put_options"
                ]
                recommendations["risk_adjustments"] = {
                    "stop_loss_multiplier": 1.1,
                    "take_profit_multiplier": 1.3,
                    "position_hold_time": "normal"
                }
                recommendations["position_sizing"] = "conservative"
                
            elif current_regime == MarketRegime.SIDEWAYS:
                recommendations["recommended_strategies"] = [
                    "mean_reversion", "range_trading", "pairs_trading"
                ]
                recommendations["risk_adjustments"] = {
                    "stop_loss_multiplier": 0.8,
                    "take_profit_multiplier": 0.9,
                    "position_hold_time": "short"
                }
                recommendations["position_sizing"] = "normal"
                
            elif current_regime == MarketRegime.VOLATILE:
                recommendations["recommended_strategies"] = [
                    "volatility_trading", "straddle", "iron_condor"
                ]
                recommendations["risk_adjustments"] = {
                    "stop_loss_multiplier": 1.5,
                    "take_profit_multiplier": 2.0,
                    "position_hold_time": "very_short"
                }
                recommendations["position_sizing"] = "very_conservative"
            
            # Add confidence-based adjustments
            if confidence < 0.6:
                recommendations["position_sizing"] = "conservative"
                recommendations["additional_notes"] = "Low confidence - reduce position sizes"
            elif confidence > 0.9:
                recommendations["additional_notes"] = "High confidence - consider increasing allocation"
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error getting strategy recommendations: {e}")
            return {"error": str(e)}    
def plot_regime_history(self, days: int = 30, save_path: Optional[str] = None) -> str:
        """
        Plot regime history over time.
        
        Args:
            days: Number of days to plot
            save_path: Path to save the plot
            
        Returns:
            Path to saved plot
        """
        try:
            cutoff_time = datetime.now() - timedelta(days=days)
            
            # Filter recent regime history
            recent_regimes = [
                r for r in self.regime_history 
                if r.timestamp > cutoff_time
            ]
            
            if not recent_regimes:
                logger.warning("No regime data available for plotting")
                return ""
            
            # Prepare data for plotting
            timestamps = [r.timestamp for r in recent_regimes]
            regimes = [r.smoothed_regime or r.regime for r in recent_regimes]
            confidences = [r.confidence for r in recent_regimes]
            
            # Create regime mapping for plotting
            regime_mapping = {
                MarketRegime.TRENDING_UP: 3,
                MarketRegime.SIDEWAYS: 2,
                MarketRegime.VOLATILE: 1,
                MarketRegime.TRENDING_DOWN: 0
            }
            
            regime_values = [regime_mapping.get(r, 1) for r in regimes]
            
            # Create plot
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
            
            # Regime timeline
            colors = ['red', 'orange', 'gray', 'green']
            regime_names = ['Trending Down', 'Volatile', 'Sideways', 'Trending Up']
            
            for i, (regime_val, color, name) in enumerate(zip([0, 1, 2, 3], colors, regime_names)):
                mask = np.array(regime_values) == regime_val
                if np.any(mask):
                    ax1.scatter(np.array(timestamps)[mask], np.array(regime_values)[mask], 
                              c=color, label=name, s=50, alpha=0.7)
            
            ax1.set_ylabel('Market Regime')
            ax1.set_title(f'Market Regime History ({days} days)')
            ax1.set_yticks([0, 1, 2, 3])
            ax1.set_yticklabels(regime_names)
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Confidence over time
            ax2.plot(timestamps, confidences, color='blue', linewidth=2, alpha=0.7)
            ax2.axhline(y=self.min_confidence_threshold, color='red', linestyle='--', 
                       label=f'Min Confidence ({self.min_confidence_threshold})')
            ax2.set_xlabel('Time')
            ax2.set_ylabel('Confidence')
            ax2.set_title('Regime Detection Confidence')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save plot
            if save_path is None:
                save_path = os.path.join(self.data_dir, f"regime_history_{days}d.png")
            
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Regime history plot saved to {save_path}")
            return save_path
            
        except Exception as e:
            logger.error(f"Error plotting regime history: {e}")
            return ""
    
    def plot_regime_transitions(self, save_path: Optional[str] = None) -> str:
        """
        Plot regime transition matrix.
        
        Args:
            save_path: Path to save the plot
            
        Returns:
            Path to saved plot
        """
        try:
            if not self.transition_history:
                logger.warning("No transition data available for plotting")
                return ""
            
            # Create transition matrix
            regimes = [r.value for r in MarketRegime if r != MarketRegime.UNKNOWN]
            transition_matrix = np.zeros((len(regimes), len(regimes)))
            
            regime_to_idx = {regime: i for i, regime in enumerate(regimes)}
            
            for transition in self.transition_history:
                from_idx = regime_to_idx.get(transition.from_regime.value)
                to_idx = regime_to_idx.get(transition.to_regime.value)
                
                if from_idx is not None and to_idx is not None:
                    transition_matrix[from_idx, to_idx] += 1
            
            # Normalize to probabilities
            row_sums = transition_matrix.sum(axis=1, keepdims=True)
            transition_matrix = np.divide(transition_matrix, row_sums, 
                                        out=np.zeros_like(transition_matrix), 
                                        where=row_sums!=0)
            
            # Create heatmap
            plt.figure(figsize=(10, 8))
            sns.heatmap(transition_matrix, 
                       xticklabels=[r.replace('_', ' ').title() for r in regimes],
                       yticklabels=[r.replace('_', ' ').title() for r in regimes],
                       annot=True, 
                       fmt='.2f',
                       cmap='Blues',
                       cbar_kws={'label': 'Transition Probability'})
            
            plt.title('Market Regime Transition Matrix')
            plt.xlabel('To Regime')
            plt.ylabel('From Regime')
            plt.tight_layout()
            
            # Save plot
            if save_path is None:
                save_path = os.path.join(self.data_dir, "regime_transitions.png")
            
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Regime transition plot saved to {save_path}")
            return save_path
            
        except Exception as e:
            logger.error(f"Error plotting regime transitions: {e}")
            return ""
    
    def plot_feature_importance(self, save_path: Optional[str] = None) -> str:
        """
        Plot feature importance for regime classification.
        
        Args:
            save_path: Path to save the plot
            
        Returns:
            Path to saved plot
        """
        try:
            if not self.is_trained or not self.training_data:
                logger.warning("No training data available for feature importance")
                return ""
            
            # Calculate feature statistics by regime
            regime_features = defaultdict(list)
            
            # Get recent classifications with known regimes
            for regime_record in self.regime_history[-100:]:  # Last 100 records
                regime = regime_record.smoothed_regime or regime_record.regime
                if regime != MarketRegime.UNKNOWN:
                    regime_features[regime.value].append(regime_record.features.to_array())
            
            if not regime_features:
                logger.warning("No regime data available for feature importance")
                return ""
            
            # Calculate mean features for each regime
            feature_names = RegimeFeatures.get_feature_names()
            regime_means = {}
            
            for regime, features_list in regime_features.items():
                if features_list:
                    regime_means[regime] = np.mean(features_list, axis=0)
            
            # Create feature importance plot
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            axes = axes.flatten()
            
            colors = ['green', 'red', 'gray', 'orange']
            regime_order = ['trending_up', 'trending_down', 'sideways', 'volatile']
            
            for i, (regime, color) in enumerate(zip(regime_order, colors)):
                if regime in regime_means:
                    ax = axes[i]
                    values = regime_means[regime]
                    
                    bars = ax.bar(range(len(feature_names)), values, color=color, alpha=0.7)
                    ax.set_title(f'{regime.replace("_", " ").title()} Regime Features')
                    ax.set_xlabel('Features')
                    ax.set_ylabel('Feature Value')
                    ax.set_xticks(range(len(feature_names)))
                    ax.set_xticklabels(feature_names, rotation=45, ha='right')
                    ax.grid(True, alpha=0.3)
                    
                    # Highlight top features
                    top_indices = np.argsort(np.abs(values))[-3:]
                    for idx in top_indices:
                        bars[idx].set_alpha(1.0)
                        bars[idx].set_edgecolor('black')
                        bars[idx].set_linewidth(2)
            
            plt.tight_layout()
            
            # Save plot
            if save_path is None:
                save_path = os.path.join(self.data_dir, "feature_importance.png")
            
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Feature importance plot saved to {save_path}")
            return save_path
            
        except Exception as e:
            logger.error(f"Error plotting feature importance: {e}")
            return ""
    
    def _save_model(self):
        """Save the trained model and associated data."""
        try:
            model_data = {
                'clustering_model': self.clustering_model,
                'scaler': self.scaler,
                'pca': self.pca,
                'cluster_to_regime': {k: v.value for k, v in self.cluster_to_regime.items()},
                'clustering_method': self.clustering_method,
                'n_clusters': self.n_clusters,
                'is_trained': self.is_trained,
                'feature_names': RegimeFeatures.get_feature_names(),
                'training_timestamp': datetime.now().isoformat()
            }
            
            with open(self.model_file, 'wb') as f:
                pickle.dump(model_data, f)
            
            logger.info("Model saved successfully")
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
    
    def _load_model_and_history(self):
        """Load the trained model and history data."""
        try:
            # Load model
            if os.path.exists(self.model_file):
                with open(self.model_file, 'rb') as f:
                    model_data = pickle.load(f)
                
                self.clustering_model = model_data.get('clustering_model')
                self.scaler = model_data.get('scaler', RobustScaler())
                self.pca = model_data.get('pca', PCA(n_components=0.95))
                self.clustering_method = model_data.get('clustering_method', 'gmm')
                self.n_clusters = model_data.get('n_clusters', 4)
                self.is_trained = model_data.get('is_trained', False)
                
                # Load cluster to regime mapping
                cluster_mapping = model_data.get('cluster_to_regime', {})
                self.cluster_to_regime = {
                    int(k): MarketRegime(v) for k, v in cluster_mapping.items()
                }
                
                logger.info("Model loaded successfully")
            
            # Load regime history
            if os.path.exists(self.history_file):
                with open(self.history_file, 'r') as f:
                    history_data = json.load(f)
                
                for record_data in history_data:
                    try:
                        # Reconstruct RegimeFeatures
                        features_data = record_data['features']
                        features = RegimeFeatures(**features_data)
                        
                        # Reconstruct RegimeClassification
                        classification = RegimeClassification(
                            regime=MarketRegime(record_data['regime']),
                            confidence=record_data['confidence'],
                            features=features,
                            cluster_probabilities=record_data['cluster_probabilities'],
                            timestamp=datetime.fromisoformat(record_data['timestamp']),
                            smoothed_regime=MarketRegime(record_data['smoothed_regime']) if record_data.get('smoothed_regime') else None,
                            transition_probability=record_data.get('transition_probability', 0.0)
                        )
                        
                        self.regime_history.append(classification)
                        
                    except Exception as e:
                        logger.warning(f"Error loading regime record: {e}")
                        continue
                
                logger.info(f"Loaded {len(self.regime_history)} regime history records")
            
            # Load transition history
            if os.path.exists(self.transitions_file):
                with open(self.transitions_file, 'r') as f:
                    transitions_data = json.load(f)
                
                for trans_data in transitions_data:
                    try:
                        transition = RegimeTransition(
                            from_regime=MarketRegime(trans_data['from_regime']),
                            to_regime=MarketRegime(trans_data['to_regime']),
                            timestamp=datetime.fromisoformat(trans_data['timestamp']),
                            confidence=trans_data['confidence'],
                            duration_minutes=trans_data['duration_minutes']
                        )
                        
                        self.transition_history.append(transition)
                        
                    except Exception as e:
                        logger.warning(f"Error loading transition record: {e}")
                        continue
                
                logger.info(f"Loaded {len(self.transition_history)} transition records")
            
        except Exception as e:
            logger.error(f"Error loading model and history: {e}")
    
    def _save_history(self):
        """Save regime and transition history."""
        try:
            # Save regime history (last 500 records)
            recent_history = list(self.regime_history)[-500:]
            history_data = [record.to_dict() for record in recent_history]
            
            with open(self.history_file, 'w') as f:
                json.dump(history_data, f, indent=2)
            
            # Save transition history
            transitions_data = [trans.to_dict() for trans in self.transition_history]
            
            with open(self.transitions_file, 'w') as f:
                json.dump(transitions_data, f, indent=2)
            
        except Exception as e:
            logger.error(f"Error saving history: {e}")
    
    def export_regime_data(self, export_path: str) -> bool:
        """Export all regime data to a file."""
        try:
            export_data = {
                'model_info': {
                    'clustering_method': self.clustering_method,
                    'n_clusters': self.n_clusters,
                    'is_trained': self.is_trained,
                    'cluster_to_regime': {k: v.value for k, v in self.cluster_to_regime.items()},
                    'update_frequency_minutes': self.update_frequency_minutes,
                    'smoothing_window': self.smoothing_window,
                    'min_confidence_threshold': self.min_confidence_threshold
                },
                'regime_history': [record.to_dict() for record in self.regime_history],
                'transition_history': [trans.to_dict() for trans in self.transition_history],
                'current_regime': self.current_regime.to_dict() if self.current_regime else None,
                'export_timestamp': datetime.now().isoformat()
            }
            
            with open(export_path, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            logger.info(f"Regime data exported to {export_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting regime data: {e}")
            return False
    
    def cleanup_old_data(self, days_to_keep: int = 90) -> Dict[str, int]:
        """
        Clean up old regime and transition data.
        
        Args:
            days_to_keep: Number of days of data to keep
            
        Returns:
            Dictionary with cleanup statistics
        """
        try:
            cutoff_time = datetime.now() - timedelta(days=days_to_keep)
            
            # Clean regime history
            old_regime_count = len(self.regime_history)
            self.regime_history = deque(
                [r for r in self.regime_history if r.timestamp > cutoff_time],
                maxlen=1000
            )
            regime_removed = old_regime_count - len(self.regime_history)
            
            # Clean transition history
            old_transition_count = len(self.transition_history)
            self.transition_history = [
                t for t in self.transition_history if t.timestamp > cutoff_time
            ]
            transitions_removed = old_transition_count - len(self.transition_history)
            
            # Save cleaned data
            if regime_removed > 0 or transitions_removed > 0:
                self._save_history()
            
            cleanup_stats = {
                'regime_records_removed': regime_removed,
                'transitions_removed': transitions_removed
            }
            
            logger.info(f"Cleaned up old data: {cleanup_stats}")
            return cleanup_stats
            
        except Exception as e:
            logger.error(f"Error cleaning up old data: {e}")
            return {'regime_records_removed': 0, 'transitions_removed': 0}
    
    def get_current_regime(self) -> Optional[MarketRegime]:
        """Get the current market regime."""
        if self.current_regime:
            return self.current_regime.smoothed_regime or self.current_regime.regime
        return None
    
    def get_regime_confidence(self) -> float:
        """Get confidence in current regime classification."""
        if self.current_regime:
            return self.current_regime.confidence
        return 0.0
    
    def is_regime_stable(self, lookback_periods: int = 5) -> bool:
        """Check if the current regime has been stable."""
        if len(self.regime_history) < lookback_periods:
            return False
        
        recent_regimes = [
            r.smoothed_regime or r.regime 
            for r in list(self.regime_history)[-lookback_periods:]
        ]
        
        # Check if all recent regimes are the same
        return len(set(recent_regimes)) == 1
    
    def get_regime_duration(self) -> int:
        """Get duration of current regime in minutes."""
        if not self.current_regime:
            return 0
        
        current_regime = self.current_regime.smoothed_regime or self.current_regime.regime
        duration = 0
        
        for regime_record in reversed(list(self.regime_history)):
            record_regime = regime_record.smoothed_regime or regime_record.regime
            if record_regime == current_regime:
                duration += self.update_frequency_minutes
            else:
                break
        
        return duration


# Utility functions
def create_sample_regime_data(n_samples: int = 500) -> List[pd.DataFrame]:
    """Create sample data for regime detection testing."""
    np.random.seed(42)
    
    sample_data = []
    
    for i in range(n_samples):
        # Generate different regime patterns
        regime_type = i % 4
        
        if regime_type == 0:  # Trending up
            trend = 0.002
            volatility = 0.01
        elif regime_type == 1:  # Trending down
            trend = -0.002
            volatility = 0.01
        elif regime_type == 2:  # Sideways
            trend = 0.0
            volatility = 0.005
        else:  # Volatile
            trend = 0.0
            volatility = 0.03
        
        # Generate price data
        n_periods = 100
        returns = np.random.normal(trend, volatility, n_periods)
        prices = 100 * np.exp(np.cumsum(returns))
        
        # Create OHLCV data
        opens = prices + np.random.normal(0, 0.1, n_periods)
        highs = np.maximum(opens, prices) + np.abs(np.random.normal(0, 0.2, n_periods))
        lows = np.minimum(opens, prices) - np.abs(np.random.normal(0, 0.2, n_periods))
        volumes = np.random.lognormal(10, 0.5, n_periods)
        
        df = pd.DataFrame({
            'open': opens,
            'high': highs,
            'low': lows,
            'close': prices,
            'volume': volumes
        })
        
        sample_data.append(df)
    
    return sample_data