"""
Pattern Memory System for Trading Bot.

This module implements an advanced pattern recognition and memory system that:
- Stores historical trading patterns that led to losses
- Uses sliding window approach to identify recurring failure patterns
- Implements similarity matching using cosine similarity
- Provides pattern avoidance recommendations
- Includes LRU eviction and pattern persistence
"""

import logging
import os
import pickle
import json
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Any, Union
from collections import OrderedDict
import numpy as np
from scipy.spatial.distance import cosine
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class PatternFeatures:
    """Feature vector representing a trading pattern."""
    # Price action features (normalized)
    price_change_1m: float  # 1-minute price change
    price_change_5m: float  # 5-minute price change
    price_change_15m: float  # 15-minute price change
    price_volatility: float  # Price volatility measure
    
    # Volume features (normalized)
    volume_ratio: float  # Volume vs average
    volume_trend: float  # Volume trend direction
    volume_spike: float  # Volume spike indicator
    
    # Technical indicators (normalized 0-1)
    rsi_normalized: float  # RSI / 100
    macd_signal: float  # MACD signal strength
    bb_position: float  # Bollinger band position
    ma_distance: float  # Distance from moving average
    
    # Market context features
    market_trend: float  # Market trend strength (-1 to 1)
    volatility_regime: float  # Volatility level (0-1)
    time_of_day: float  # Time factor (0-1)
    day_of_week: float  # Day factor (0-1)
    
    # Trade setup features
    entry_confidence: float  # Original signal confidence
    risk_reward_ratio: float  # Expected R:R ratio
    stop_distance: float  # Stop loss distance (normalized)
    
    def to_vector(self) -> np.ndarray:
        """Convert features to numpy vector for similarity calculations."""
        return np.array([
            self.price_change_1m, self.price_change_5m, self.price_change_15m,
            self.price_volatility, self.volume_ratio, self.volume_trend,
            self.volume_spike, self.rsi_normalized, self.macd_signal,
            self.bb_position, self.ma_distance, self.market_trend,
            self.volatility_regime, self.time_of_day, self.day_of_week,
            self.entry_confidence, self.risk_reward_ratio, self.stop_distance
        ])
    
    @classmethod
    def get_feature_names(cls) -> List[str]:
        """Get list of feature names for visualization."""
        return [
            'price_change_1m', 'price_change_5m', 'price_change_15m',
            'price_volatility', 'volume_ratio', 'volume_trend',
            'volume_spike', 'rsi_normalized', 'macd_signal',
            'bb_position', 'ma_distance', 'market_trend',
            'volatility_regime', 'time_of_day', 'day_of_week',
            'entry_confidence', 'risk_reward_ratio', 'stop_distance'
        ]


@dataclass
class FailurePattern:
    """Represents a historical failure pattern."""
    pattern_id: str
    symbol: str
    timestamp: datetime
    features: PatternFeatures
    failure_type: str
    loss_amount: float
    loss_percentage: float
    strategy_used: str
    market_conditions: Dict[str, Any]
    failure_reasons: List[str]
    access_count: int = 0  # For LRU tracking
    last_accessed: datetime = None
    
    def __post_init__(self):
        if self.last_accessed is None:
            self.last_accessed = datetime.now()


@dataclass
class PatternMatch:
    """Represents a pattern match result."""
    matched_pattern: FailurePattern
    similarity_score: float
    confidence_score: float
    risk_level: str  # 'low', 'medium', 'high'
    avoidance_recommendation: str
    feature_contributions: Dict[str, float]  # Which features contributed most to match


class PatternMemorySystem:
    """
    Advanced pattern memory system for trading failure recognition.
    
    Uses sliding window approach with cosine similarity for pattern matching
    and LRU eviction for memory management.
    """
    
    def __init__(self, 
                 data_dir: str = "data",
                 max_patterns: int = 10000,
                 similarity_threshold: float = 0.7,
                 window_size: int = 50):
        """
        Initialize the pattern memory system.
        
        Args:
            data_dir: Directory for data storage
            max_patterns: Maximum number of patterns to store
            similarity_threshold: Minimum similarity for pattern matching
            window_size: Sliding window size for pattern analysis
        """
        self.data_dir = data_dir
        self.max_patterns = max_patterns
        self.similarity_threshold = similarity_threshold
        self.window_size = window_size
        
        # Pattern storage with LRU ordering
        self.patterns: OrderedDict[str, FailurePattern] = OrderedDict()
        
        # Pattern analysis cache
        self.similarity_cache: Dict[str, Dict[str, float]] = {}
        
        # Feature normalization parameters
        self.feature_stats: Dict[str, Dict[str, float]] = {}
        
        # File paths
        self.patterns_file = os.path.join(data_dir, "failure_patterns.pkl")
        self.stats_file = os.path.join(data_dir, "pattern_stats.json")
        
        # Ensure data directory exists
        os.makedirs(data_dir, exist_ok=True)
        
        # Load existing patterns
        self._load_patterns()
        
        logger.info(f"PatternMemorySystem initialized with {len(self.patterns)} patterns")
    
    def store_failure_pattern(self, 
                            symbol: str,
                            price_data: pd.DataFrame,
                            trade_context: Dict[str, Any],
                            failure_type: str,
                            loss_amount: float,
                            loss_percentage: float,
                            failure_reasons: List[str]) -> str:
        """
        Store a new failure pattern in memory.
        
        Args:
            symbol: Stock symbol
            price_data: Price/volume dataframe with technical indicators
            trade_context: Trade context information
            failure_type: Type of failure
            loss_amount: Absolute loss amount
            loss_percentage: Percentage loss
            failure_reasons: List of failure reasons
            
        Returns:
            Pattern ID of stored pattern
        """
        try:
            # Extract features from price data and context
            features = self._extract_features(price_data, trade_context)
            
            # Create pattern ID
            pattern_id = f"{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hash(str(features.to_vector()))}"
            
            # Create failure pattern
            pattern = FailurePattern(
                pattern_id=pattern_id,
                symbol=symbol,
                timestamp=datetime.now(),
                features=features,
                failure_type=failure_type,
                loss_amount=loss_amount,
                loss_percentage=loss_percentage,
                strategy_used=trade_context.get('strategy', 'unknown'),
                market_conditions=trade_context.get('market_conditions', {}),
                failure_reasons=failure_reasons
            )
            
            # Store pattern with LRU management
            self._store_pattern_lru(pattern)
            
            # Update feature statistics for normalization
            self._update_feature_stats(features)
            
            # Clear similarity cache
            self.similarity_cache.clear()
            
            # Persist to disk
            self._save_patterns()
            
            logger.info(f"Stored failure pattern {pattern_id} for {symbol}")
            return pattern_id
            
        except Exception as e:
            logger.error(f"Error storing failure pattern: {e}")
            return None
    
    def find_similar_patterns(self, 
                            symbol: str,
                            price_data: pd.DataFrame,
                            trade_context: Dict[str, Any],
                            top_k: int = 5) -> List[PatternMatch]:
        """
        Find similar historical failure patterns.
        
        Args:
            symbol: Stock symbol
            price_data: Current price/volume data
            trade_context: Current trade context
            top_k: Number of top matches to return
            
        Returns:
            List of PatternMatch objects sorted by similarity
        """
        try:
            if not self.patterns:
                return []
            
            # Extract features for current situation
            current_features = self._extract_features(price_data, trade_context)
            current_vector = current_features.to_vector()
            
            # Calculate similarities
            matches = []
            
            for pattern_id, pattern in self.patterns.items():
                # Update access tracking for LRU
                pattern.access_count += 1
                pattern.last_accessed = datetime.now()
                
                # Calculate similarity
                similarity = self._calculate_similarity(current_vector, pattern.features.to_vector())
                
                if similarity >= self.similarity_threshold:
                    # Calculate confidence and risk level
                    confidence = self._calculate_confidence(similarity, pattern)
                    risk_level = self._determine_risk_level(similarity, pattern)
                    
                    # Generate avoidance recommendation
                    recommendation = self._generate_avoidance_recommendation(pattern, similarity)
                    
                    # Calculate feature contributions
                    contributions = self._calculate_feature_contributions(
                        current_vector, pattern.features.to_vector()
                    )
                    
                    match = PatternMatch(
                        matched_pattern=pattern,
                        similarity_score=similarity,
                        confidence_score=confidence,
                        risk_level=risk_level,
                        avoidance_recommendation=recommendation,
                        feature_contributions=contributions
                    )
                    
                    matches.append(match)
            
            # Sort by similarity and return top k
            matches.sort(key=lambda x: x.similarity_score, reverse=True)
            
            # Move accessed patterns to end (LRU)
            for match in matches[:top_k]:
                pattern_id = match.matched_pattern.pattern_id
                self.patterns.move_to_end(pattern_id)
            
            logger.debug(f"Found {len(matches)} similar patterns for {symbol}")
            return matches[:top_k]
            
        except Exception as e:
            logger.error(f"Error finding similar patterns: {e}")
            return []
    
    def get_pattern_avoidance_score(self, 
                                  symbol: str,
                                  price_data: pd.DataFrame,
                                  trade_context: Dict[str, Any]) -> Tuple[float, List[str]]:
        """
        Get overall pattern avoidance score and recommendations.
        
        Args:
            symbol: Stock symbol
            price_data: Current price data
            trade_context: Trade context
            
        Returns:
            Tuple of (avoidance_score, recommendations)
            avoidance_score: 0-1, higher means avoid trade
        """
        try:
            similar_patterns = self.find_similar_patterns(symbol, price_data, trade_context)
            
            if not similar_patterns:
                return 0.0, []
            
            # Calculate weighted avoidance score
            total_weight = 0
            weighted_score = 0
            recommendations = []
            
            for match in similar_patterns:
                # Weight by similarity and recency
                days_old = (datetime.now() - match.matched_pattern.timestamp).days
                recency_weight = np.exp(-days_old / 30.0)  # 30-day half-life
                
                weight = match.similarity_score * recency_weight
                total_weight += weight
                
                # Score based on historical loss severity
                loss_severity = min(abs(match.matched_pattern.loss_percentage) / 10.0, 1.0)
                weighted_score += weight * loss_severity
                
                # Collect unique recommendations
                if match.avoidance_recommendation not in recommendations:
                    recommendations.append(match.avoidance_recommendation)
            
            final_score = weighted_score / total_weight if total_weight > 0 else 0.0
            
            logger.debug(f"Pattern avoidance score for {symbol}: {final_score:.3f}")
            return final_score, recommendations[:3]  # Top 3 recommendations
            
        except Exception as e:
            logger.error(f"Error calculating avoidance score: {e}")
            return 0.0, [] 
   def _extract_features(self, price_data: pd.DataFrame, trade_context: Dict[str, Any]) -> PatternFeatures:
        """Extract normalized features from price data and trade context."""
        try:
            # Ensure we have enough data
            if len(price_data) < self.window_size:
                # Pad with the first available values if insufficient data
                price_data = price_data.reindex(range(self.window_size), method='ffill')
            
            # Take last window_size rows
            data = price_data.tail(self.window_size).copy()
            
            # Price action features
            price_changes = data['close'].pct_change().fillna(0)
            price_change_1m = price_changes.iloc[-1] if len(price_changes) > 0 else 0.0
            price_change_5m = price_changes.tail(5).mean() if len(price_changes) >= 5 else 0.0
            price_change_15m = price_changes.tail(15).mean() if len(price_changes) >= 15 else 0.0
            price_volatility = price_changes.std() if len(price_changes) > 1 else 0.0
            
            # Volume features
            volume_mean = data['volume'].mean() if 'volume' in data.columns else 1000
            volume_ratio = (data['volume'].iloc[-1] / volume_mean) if volume_mean > 0 else 1.0
            volume_trend = np.corrcoef(range(len(data)), data['volume'])[0, 1] if len(data) > 1 else 0.0
            volume_spike = 1.0 if volume_ratio > 2.0 else 0.0
            
            # Technical indicators
            rsi = data.get('rsi', pd.Series([50] * len(data))).iloc[-1]
            rsi_normalized = rsi / 100.0 if rsi is not None else 0.5
            
            macd = data.get('macd', pd.Series([0] * len(data))).iloc[-1]
            macd_signal = np.tanh(macd) if macd is not None else 0.0  # Normalize to [-1, 1]
            
            # Bollinger band position (0 = lower, 0.5 = middle, 1 = upper)
            bb_upper = data.get('bb_upper', data['close']).iloc[-1]
            bb_lower = data.get('bb_lower', data['close']).iloc[-1]
            current_price = data['close'].iloc[-1]
            bb_position = ((current_price - bb_lower) / (bb_upper - bb_lower)) if bb_upper != bb_lower else 0.5
            bb_position = np.clip(bb_position, 0, 1)
            
            # Moving average distance
            ma_20 = data.get('sma_20', data['close']).iloc[-1]
            ma_distance = (current_price - ma_20) / ma_20 if ma_20 > 0 else 0.0
            ma_distance = np.tanh(ma_distance * 10)  # Normalize to [-1, 1]
            
            # Market context features
            market_conditions = trade_context.get('market_conditions', {})
            
            # Market trend: bullish=1, bearish=-1, sideways=0
            trend_map = {'bullish': 1.0, 'bearish': -1.0, 'sideways': 0.0}
            market_trend = trend_map.get(market_conditions.get('market_trend', 'sideways'), 0.0)
            
            # Volatility regime: low=0, medium=0.5, high=1
            vol_map = {'low': 0.0, 'medium': 0.5, 'high': 1.0}
            volatility_regime = vol_map.get(market_conditions.get('volatility_regime', 'medium'), 0.5)
            
            # Time of day: opening=0, mid_session=0.5, closing=1
            time_map = {'opening': 0.0, 'mid_session': 0.5, 'closing': 1.0}
            time_of_day = time_map.get(market_conditions.get('time_of_day', 'mid_session'), 0.5)
            
            # Day of week: Monday=0, Friday=1
            day_map = {'Monday': 0.0, 'Tuesday': 0.25, 'Wednesday': 0.5, 'Thursday': 0.75, 'Friday': 1.0}
            day_of_week = day_map.get(market_conditions.get('day_of_week', 'Wednesday'), 0.5)
            
            # Trade setup features
            entry_confidence = trade_context.get('signal_confidence', 0.5)
            risk_reward_ratio = trade_context.get('risk_reward_ratio', 1.0)
            risk_reward_ratio = np.clip(risk_reward_ratio / 5.0, 0, 1)  # Normalize assuming max R:R of 5
            
            stop_distance = trade_context.get('stop_distance_pct', 2.0)
            stop_distance = np.clip(stop_distance / 10.0, 0, 1)  # Normalize assuming max 10% stop
            
            return PatternFeatures(
                price_change_1m=np.clip(price_change_1m * 100, -10, 10) / 10,  # Normalize to [-1, 1]
                price_change_5m=np.clip(price_change_5m * 100, -10, 10) / 10,
                price_change_15m=np.clip(price_change_15m * 100, -10, 10) / 10,
                price_volatility=np.clip(price_volatility * 100, 0, 10) / 10,  # Normalize to [0, 1]
                volume_ratio=np.clip(volume_ratio, 0, 5) / 5,  # Normalize to [0, 1]
                volume_trend=np.clip(volume_trend, -1, 1),
                volume_spike=volume_spike,
                rsi_normalized=rsi_normalized,
                macd_signal=macd_signal,
                bb_position=bb_position,
                ma_distance=ma_distance,
                market_trend=market_trend,
                volatility_regime=volatility_regime,
                time_of_day=time_of_day,
                day_of_week=day_of_week,
                entry_confidence=entry_confidence,
                risk_reward_ratio=risk_reward_ratio,
                stop_distance=stop_distance
            )
            
        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            # Return default features
            return PatternFeatures(
                price_change_1m=0.0, price_change_5m=0.0, price_change_15m=0.0,
                price_volatility=0.0, volume_ratio=0.5, volume_trend=0.0,
                volume_spike=0.0, rsi_normalized=0.5, macd_signal=0.0,
                bb_position=0.5, ma_distance=0.0, market_trend=0.0,
                volatility_regime=0.5, time_of_day=0.5, day_of_week=0.5,
                entry_confidence=0.5, risk_reward_ratio=0.2, stop_distance=0.2
            )
    
    def _calculate_similarity(self, vector1: np.ndarray, vector2: np.ndarray) -> float:
        """Calculate cosine similarity between two feature vectors."""
        try:
            # Handle zero vectors
            if np.allclose(vector1, 0) or np.allclose(vector2, 0):
                return 0.0
            
            # Calculate cosine similarity (1 - cosine distance)
            similarity = 1 - cosine(vector1, vector2)
            
            # Ensure similarity is in [0, 1] range
            return np.clip(similarity, 0.0, 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating similarity: {e}")
            return 0.0
    
    def _calculate_confidence(self, similarity: float, pattern: FailurePattern) -> float:
        """Calculate confidence score for a pattern match."""
        try:
            # Base confidence from similarity
            base_confidence = similarity
            
            # Adjust for pattern age (newer patterns are more reliable)
            days_old = (datetime.now() - pattern.timestamp).days
            age_factor = np.exp(-days_old / 60.0)  # 60-day half-life
            
            # Adjust for pattern access frequency (more accessed = more validated)
            access_factor = min(pattern.access_count / 10.0, 1.0)
            
            # Adjust for loss severity (higher losses = more important to avoid)
            severity_factor = min(abs(pattern.loss_percentage) / 5.0, 1.0)
            
            # Combined confidence
            confidence = base_confidence * (0.5 + 0.2 * age_factor + 0.1 * access_factor + 0.2 * severity_factor)
            
            return np.clip(confidence, 0.0, 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating confidence: {e}")
            return similarity
    
    def _determine_risk_level(self, similarity: float, pattern: FailurePattern) -> str:
        """Determine risk level based on similarity and pattern characteristics."""
        try:
            # Base risk from similarity
            if similarity >= 0.9:
                base_risk = 'high'
            elif similarity >= 0.8:
                base_risk = 'medium'
            else:
                base_risk = 'low'
            
            # Adjust for loss severity
            loss_pct = abs(pattern.loss_percentage)
            if loss_pct >= 5.0:  # >= 5% loss
                if base_risk == 'low':
                    base_risk = 'medium'
                elif base_risk == 'medium':
                    base_risk = 'high'
            
            # Adjust for pattern recency
            days_old = (datetime.now() - pattern.timestamp).days
            if days_old <= 7:  # Recent pattern
                if base_risk == 'low':
                    base_risk = 'medium'
            
            return base_risk
            
        except Exception as e:
            logger.error(f"Error determining risk level: {e}")
            return 'medium'
    
    def _generate_avoidance_recommendation(self, pattern: FailurePattern, similarity: float) -> str:
        """Generate specific avoidance recommendation based on pattern."""
        try:
            recommendations = []
            
            # Base recommendation on failure type
            failure_type = pattern.failure_type.lower()
            
            if 'false_breakout' in failure_type:
                recommendations.append("Wait for volume confirmation before entering breakout trades")
            elif 'stop_loss' in failure_type:
                recommendations.append("Use wider stop losses or better position sizing")
            elif 'trend_reversal' in failure_type:
                recommendations.append("Check multiple timeframes before entry")
            elif 'low_volume' in failure_type:
                recommendations.append("Avoid trading during low volume periods")
            elif 'news_impact' in failure_type:
                recommendations.append("Check news sentiment before trade entry")
            
            # Add specific recommendations based on features
            features = pattern.features
            
            if features.rsi_normalized > 0.7:
                recommendations.append("Avoid entries when RSI is overbought")
            elif features.rsi_normalized < 0.3:
                recommendations.append("Avoid entries when RSI is oversold")
            
            if features.volume_ratio < 0.5:
                recommendations.append("Require higher volume for trade confirmation")
            
            if features.volatility_regime > 0.7:
                recommendations.append("Use wider stops during high volatility")
            
            if features.time_of_day > 0.8:  # Late in session
                recommendations.append("Avoid late-session entries")
            
            # Return most relevant recommendation
            if recommendations:
                return recommendations[0]
            else:
                return f"Similar pattern failed with {pattern.loss_percentage:.1f}% loss - exercise caution"
                
        except Exception as e:
            logger.error(f"Error generating recommendation: {e}")
            return "Exercise caution - similar pattern found in failure history"
    
    def _calculate_feature_contributions(self, current_vector: np.ndarray, pattern_vector: np.ndarray) -> Dict[str, float]:
        """Calculate which features contributed most to the similarity match."""
        try:
            feature_names = PatternFeatures.get_feature_names()
            contributions = {}
            
            # Calculate absolute differences for each feature
            differences = np.abs(current_vector - pattern_vector)
            
            # Convert to similarity contributions (lower difference = higher contribution)
            max_diff = np.max(differences) if np.max(differences) > 0 else 1.0
            similarities = 1.0 - (differences / max_diff)
            
            # Create contribution dictionary
            for i, name in enumerate(feature_names):
                contributions[name] = float(similarities[i])
            
            return contributions
            
        except Exception as e:
            logger.error(f"Error calculating feature contributions: {e}")
            return {}
    
    def _store_pattern_lru(self, pattern: FailurePattern):
        """Store pattern with LRU eviction policy."""
        try:
            # Add new pattern
            self.patterns[pattern.pattern_id] = pattern
            
            # Check if we need to evict old patterns
            if len(self.patterns) > self.max_patterns:
                # Remove oldest patterns (those at the beginning of OrderedDict)
                patterns_to_remove = len(self.patterns) - self.max_patterns
                
                for _ in range(patterns_to_remove):
                    oldest_id, oldest_pattern = self.patterns.popitem(last=False)
                    logger.debug(f"Evicted old pattern {oldest_id}")
            
        except Exception as e:
            logger.error(f"Error storing pattern with LRU: {e}")
    
    def _update_feature_stats(self, features: PatternFeatures):
        """Update feature statistics for normalization."""
        try:
            vector = features.to_vector()
            feature_names = PatternFeatures.get_feature_names()
            
            for i, name in enumerate(feature_names):
                value = vector[i]
                
                if name not in self.feature_stats:
                    self.feature_stats[name] = {'mean': value, 'std': 0.0, 'count': 1}
                else:
                    stats = self.feature_stats[name]
                    count = stats['count']
                    old_mean = stats['mean']
                    
                    # Update running statistics
                    new_count = count + 1
                    new_mean = old_mean + (value - old_mean) / new_count
                    new_std = np.sqrt(((count * stats['std']**2) + (value - old_mean) * (value - new_mean)) / new_count)
                    
                    self.feature_stats[name] = {
                        'mean': new_mean,
                        'std': new_std,
                        'count': new_count
                    }
            
        except Exception as e:
            logger.error(f"Error updating feature stats: {e}")
    
    def _save_patterns(self):
        """Save patterns and statistics to disk."""
        try:
            # Save patterns
            with open(self.patterns_file, 'wb') as f:
                pickle.dump(dict(self.patterns), f)
            
            # Save feature statistics
            with open(self.stats_file, 'w') as f:
                json.dump(self.feature_stats, f, indent=2)
            
            logger.debug(f"Saved {len(self.patterns)} patterns to disk")
            
        except Exception as e:
            logger.error(f"Error saving patterns: {e}")
    
    def _load_patterns(self):
        """Load patterns and statistics from disk."""
        try:
            # Load patterns
            if os.path.exists(self.patterns_file):
                with open(self.patterns_file, 'rb') as f:
                    patterns_dict = pickle.load(f)
                    self.patterns = OrderedDict(patterns_dict)
                
                logger.info(f"Loaded {len(self.patterns)} patterns from disk")
            
            # Load feature statistics
            if os.path.exists(self.stats_file):
                with open(self.stats_file, 'r') as f:
                    self.feature_stats = json.load(f)
                
                logger.debug("Loaded feature statistics from disk")
            
        except Exception as e:
            logger.error(f"Error loading patterns: {e}")
            self.patterns = OrderedDict()
            self.feature_stats = {}    def g
et_pattern_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about stored patterns."""
        try:
            if not self.patterns:
                return {"error": "No patterns stored"}
            
            patterns_list = list(self.patterns.values())
            
            # Basic statistics
            total_patterns = len(patterns_list)
            
            # Failure type distribution
            failure_types = {}
            for pattern in patterns_list:
                failure_type = pattern.failure_type
                failure_types[failure_type] = failure_types.get(failure_type, 0) + 1
            
            # Symbol distribution
            symbols = {}
            for pattern in patterns_list:
                symbol = pattern.symbol
                symbols[symbol] = symbols.get(symbol, 0) + 1
            
            # Loss statistics
            losses = [abs(p.loss_percentage) for p in patterns_list]
            avg_loss = np.mean(losses)
            max_loss = np.max(losses)
            min_loss = np.min(losses)
            
            # Age distribution
            now = datetime.now()
            ages = [(now - p.timestamp).days for p in patterns_list]
            avg_age = np.mean(ages)
            
            # Access statistics
            access_counts = [p.access_count for p in patterns_list]
            avg_access = np.mean(access_counts)
            
            return {
                "total_patterns": total_patterns,
                "failure_type_distribution": failure_types,
                "top_symbols": dict(sorted(symbols.items(), key=lambda x: x[1], reverse=True)[:10]),
                "loss_statistics": {
                    "average_loss_pct": round(avg_loss, 2),
                    "max_loss_pct": round(max_loss, 2),
                    "min_loss_pct": round(min_loss, 2)
                },
                "age_statistics": {
                    "average_age_days": round(avg_age, 1),
                    "oldest_pattern_days": max(ages),
                    "newest_pattern_days": min(ages)
                },
                "access_statistics": {
                    "average_access_count": round(avg_access, 1),
                    "most_accessed": max(access_counts),
                    "least_accessed": min(access_counts)
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting pattern statistics: {e}")
            return {"error": str(e)}
    
    def visualize_pattern_features(self, pattern_id: str, save_path: Optional[str] = None) -> str:
        """
        Create visualization of pattern features.
        
        Args:
            pattern_id: ID of pattern to visualize
            save_path: Optional path to save the plot
            
        Returns:
            Path to saved plot or error message
        """
        try:
            if pattern_id not in self.patterns:
                return f"Pattern {pattern_id} not found"
            
            pattern = self.patterns[pattern_id]
            features = pattern.features
            feature_names = PatternFeatures.get_feature_names()
            feature_values = features.to_vector()
            
            # Create figure
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
            
            # Feature values bar chart
            bars = ax1.bar(range(len(feature_names)), feature_values)
            ax1.set_xlabel('Features')
            ax1.set_ylabel('Normalized Values')
            ax1.set_title(f'Pattern Features: {pattern_id}\n{pattern.symbol} - {pattern.failure_type}')
            ax1.set_xticks(range(len(feature_names)))
            ax1.set_xticklabels(feature_names, rotation=45, ha='right')
            ax1.grid(True, alpha=0.3)
            
            # Color bars based on values
            for i, bar in enumerate(bars):
                if feature_values[i] > 0.7:
                    bar.set_color('red')
                elif feature_values[i] < 0.3:
                    bar.set_color('blue')
                else:
                    bar.set_color('gray')
            
            # Pattern metadata
            metadata_text = f"""
            Symbol: {pattern.symbol}
            Failure Type: {pattern.failure_type}
            Loss: {pattern.loss_percentage:.2f}%
            Strategy: {pattern.strategy_used}
            Date: {pattern.timestamp.strftime('%Y-%m-%d %H:%M')}
            Access Count: {pattern.access_count}
            """
            
            ax2.text(0.1, 0.5, metadata_text, transform=ax2.transAxes, 
                    fontsize=10, verticalalignment='center',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
            ax2.axis('off')
            
            plt.tight_layout()
            
            # Save plot
            if save_path is None:
                save_path = os.path.join(self.data_dir, f"pattern_{pattern_id}_features.png")
            
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Pattern visualization saved to {save_path}")
            return save_path
            
        except Exception as e:
            logger.error(f"Error visualizing pattern features: {e}")
            return f"Error: {str(e)}"
    
    def visualize_similarity_heatmap(self, top_n: int = 20, save_path: Optional[str] = None) -> str:
        """
        Create heatmap of pattern similarities.
        
        Args:
            top_n: Number of most recent patterns to include
            save_path: Optional path to save the plot
            
        Returns:
            Path to saved plot or error message
        """
        try:
            if len(self.patterns) < 2:
                return "Need at least 2 patterns for similarity heatmap"
            
            # Get most recent patterns
            recent_patterns = list(self.patterns.values())[-top_n:]
            n_patterns = len(recent_patterns)
            
            # Calculate similarity matrix
            similarity_matrix = np.zeros((n_patterns, n_patterns))
            pattern_labels = []
            
            for i, pattern1 in enumerate(recent_patterns):
                pattern_labels.append(f"{pattern1.symbol}_{pattern1.timestamp.strftime('%m%d')}")
                for j, pattern2 in enumerate(recent_patterns):
                    if i == j:
                        similarity_matrix[i, j] = 1.0
                    else:
                        similarity = self._calculate_similarity(
                            pattern1.features.to_vector(),
                            pattern2.features.to_vector()
                        )
                        similarity_matrix[i, j] = similarity
            
            # Create heatmap
            plt.figure(figsize=(12, 10))
            sns.heatmap(similarity_matrix, 
                       xticklabels=pattern_labels,
                       yticklabels=pattern_labels,
                       annot=True, 
                       fmt='.2f',
                       cmap='RdYlBu_r',
                       center=0.5)
            
            plt.title(f'Pattern Similarity Heatmap (Top {n_patterns} Recent Patterns)')
            plt.xlabel('Patterns')
            plt.ylabel('Patterns')
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            
            # Save plot
            if save_path is None:
                save_path = os.path.join(self.data_dir, "pattern_similarity_heatmap.png")
            
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Similarity heatmap saved to {save_path}")
            return save_path
            
        except Exception as e:
            logger.error(f"Error creating similarity heatmap: {e}")
            return f"Error: {str(e)}"
    
    def export_patterns(self, export_path: str, format: str = 'json') -> bool:
        """
        Export patterns to file.
        
        Args:
            export_path: Path to export file
            format: Export format ('json', 'csv', 'pickle')
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if format.lower() == 'json':
                # Convert patterns to JSON-serializable format
                export_data = []
                for pattern in self.patterns.values():
                    pattern_dict = asdict(pattern)
                    # Convert datetime to string
                    pattern_dict['timestamp'] = pattern.timestamp.isoformat()
                    pattern_dict['last_accessed'] = pattern.last_accessed.isoformat()
                    export_data.append(pattern_dict)
                
                with open(export_path, 'w') as f:
                    json.dump(export_data, f, indent=2)
            
            elif format.lower() == 'csv':
                # Flatten patterns for CSV export
                rows = []
                for pattern in self.patterns.values():
                    row = {
                        'pattern_id': pattern.pattern_id,
                        'symbol': pattern.symbol,
                        'timestamp': pattern.timestamp.isoformat(),
                        'failure_type': pattern.failure_type,
                        'loss_amount': pattern.loss_amount,
                        'loss_percentage': pattern.loss_percentage,
                        'strategy_used': pattern.strategy_used,
                        'access_count': pattern.access_count,
                        'last_accessed': pattern.last_accessed.isoformat()
                    }
                    
                    # Add feature values
                    feature_names = PatternFeatures.get_feature_names()
                    feature_values = pattern.features.to_vector()
                    for i, name in enumerate(feature_names):
                        row[f'feature_{name}'] = feature_values[i]
                    
                    rows.append(row)
                
                df = pd.DataFrame(rows)
                df.to_csv(export_path, index=False)
            
            elif format.lower() == 'pickle':
                with open(export_path, 'wb') as f:
                    pickle.dump(dict(self.patterns), f)
            
            else:
                logger.error(f"Unsupported export format: {format}")
                return False
            
            logger.info(f"Exported {len(self.patterns)} patterns to {export_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting patterns: {e}")
            return False
    
    def import_patterns(self, import_path: str, format: str = 'json') -> bool:
        """
        Import patterns from file.
        
        Args:
            import_path: Path to import file
            format: Import format ('json', 'pickle')
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not os.path.exists(import_path):
                logger.error(f"Import file not found: {import_path}")
                return False
            
            if format.lower() == 'json':
                with open(import_path, 'r') as f:
                    import_data = json.load(f)
                
                imported_count = 0
                for pattern_dict in import_data:
                    try:
                        # Convert string timestamps back to datetime
                        pattern_dict['timestamp'] = datetime.fromisoformat(pattern_dict['timestamp'])
                        pattern_dict['last_accessed'] = datetime.fromisoformat(pattern_dict['last_accessed'])
                        
                        # Recreate PatternFeatures object
                        features_dict = pattern_dict['features']
                        features = PatternFeatures(**features_dict)
                        pattern_dict['features'] = features
                        
                        # Create FailurePattern object
                        pattern = FailurePattern(**pattern_dict)
                        
                        # Store with LRU management
                        self._store_pattern_lru(pattern)
                        imported_count += 1
                        
                    except Exception as e:
                        logger.warning(f"Error importing pattern: {e}")
                        continue
                
                logger.info(f"Imported {imported_count} patterns from {import_path}")
                
            elif format.lower() == 'pickle':
                with open(import_path, 'rb') as f:
                    patterns_dict = pickle.load(f)
                
                imported_count = 0
                for pattern in patterns_dict.values():
                    self._store_pattern_lru(pattern)
                    imported_count += 1
                
                logger.info(f"Imported {imported_count} patterns from {import_path}")
            
            else:
                logger.error(f"Unsupported import format: {format}")
                return False
            
            # Save updated patterns
            self._save_patterns()
            return True
            
        except Exception as e:
            logger.error(f"Error importing patterns: {e}")
            return False
    
    def cleanup_old_patterns(self, days_to_keep: int = 180) -> int:
        """
        Remove patterns older than specified days.
        
        Args:
            days_to_keep: Number of days to keep patterns
            
        Returns:
            Number of patterns removed
        """
        try:
            cutoff_date = datetime.now() - timedelta(days=days_to_keep)
            
            patterns_to_remove = []
            for pattern_id, pattern in self.patterns.items():
                if pattern.timestamp < cutoff_date:
                    patterns_to_remove.append(pattern_id)
            
            # Remove old patterns
            for pattern_id in patterns_to_remove:
                del self.patterns[pattern_id]
            
            # Clear similarity cache
            self.similarity_cache.clear()
            
            # Save updated patterns
            if patterns_to_remove:
                self._save_patterns()
            
            logger.info(f"Cleaned up {len(patterns_to_remove)} old patterns")
            return len(patterns_to_remove)
            
        except Exception as e:
            logger.error(f"Error cleaning up old patterns: {e}")
            return 0
    
    def get_pattern_by_id(self, pattern_id: str) -> Optional[FailurePattern]:
        """Get a specific pattern by ID."""
        return self.patterns.get(pattern_id)
    
    def search_patterns(self, 
                       symbol: Optional[str] = None,
                       failure_type: Optional[str] = None,
                       min_loss_pct: Optional[float] = None,
                       max_age_days: Optional[int] = None) -> List[FailurePattern]:
        """
        Search patterns based on criteria.
        
        Args:
            symbol: Filter by symbol
            failure_type: Filter by failure type
            min_loss_pct: Minimum loss percentage
            max_age_days: Maximum age in days
            
        Returns:
            List of matching patterns
        """
        try:
            matching_patterns = []
            cutoff_date = datetime.now() - timedelta(days=max_age_days) if max_age_days else None
            
            for pattern in self.patterns.values():
                # Apply filters
                if symbol and pattern.symbol != symbol:
                    continue
                
                if failure_type and pattern.failure_type != failure_type:
                    continue
                
                if min_loss_pct and abs(pattern.loss_percentage) < min_loss_pct:
                    continue
                
                if cutoff_date and pattern.timestamp < cutoff_date:
                    continue
                
                matching_patterns.append(pattern)
            
            return matching_patterns
            
        except Exception as e:
            logger.error(f"Error searching patterns: {e}")
            return []