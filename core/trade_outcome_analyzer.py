"""
Trade Outcome Analyzer for categorizing and analyzing trade failures.

This module provides comprehensive analysis of completed trades, categorizing
failures and calculating probability patterns for similar setups.
"""

import logging
import csv
import os
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
import statistics
import math

# Configure logging
logger = logging.getLogger(__name__)


class FailureType(Enum):
    """Trade failure classifications."""
    FALSE_BREAKOUT = "false_breakout"
    STOP_LOSS_HIT = "stop_loss_hit"
    TREND_REVERSAL = "trend_reversal"
    LOW_VOLUME = "low_volume"
    NEWS_IMPACT = "news_impact"
    UNKNOWN = "unknown"


class TradeOutcome(Enum):
    """Trade outcome types."""
    WIN = "win"
    LOSS = "loss"
    BREAKEVEN = "breakeven"


@dataclass
class MarketConditions:
    """Market conditions at trade time."""
    market_trend: str  # bullish, bearish, sideways
    volatility_regime: str  # low, medium, high
    volume_profile: str  # above_average, average, below_average
    time_of_day: str  # opening, mid_session, closing
    day_of_week: str
    market_breadth: float  # advance/decline ratio
    vix_level: Optional[float] = None


@dataclass
class TechnicalIndicators:
    """Technical indicators used in trade decision."""
    rsi: Optional[float] = None
    macd_signal: Optional[str] = None  # bullish, bearish, neutral
    moving_avg_position: Optional[str] = None  # above, below, at
    volume_sma_ratio: Optional[float] = None
    atr: Optional[float] = None
    bollinger_position: Optional[str] = None  # upper, middle, lower
    support_resistance_distance: Optional[float] = None


@dataclass
class TradeContext:
    """Complete trade context information."""
    symbol: str
    entry_price: float
    exit_price: float
    entry_time: datetime
    exit_time: datetime
    quantity: int
    trade_type: str  # long, short
    strategy_used: str
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    market_conditions: Optional[MarketConditions] = None
    technical_indicators: Optional[TechnicalIndicators] = None
    news_sentiment: Optional[float] = None  # -1 to 1
    volume_at_entry: Optional[int] = None
    volume_at_exit: Optional[int] = None


@dataclass
class TradeAnalysisResult:
    """Result of trade outcome analysis."""
    trade_id: str
    symbol: str
    outcome: TradeOutcome
    failure_type: Optional[FailureType]
    pnl: float
    pnl_percentage: float
    confidence_score: float  # 0-1, confidence in failure classification
    failure_probability: float  # 0-1, probability of similar setup failing
    similar_trades_count: int
    analysis_timestamp: datetime
    failure_reasons: List[str]
    market_context_score: float  # how similar market conditions affect outcome
    technical_context_score: float  # how similar technical setup affects outcome
    recommendations: List[str]


class TradeOutcomeAnalyzer:
    """
    Analyzes completed trades and categorizes failures with pattern recognition.
    """
    
    def __init__(self, data_dir: str = "data", lookback_days: int = 90):
        """
        Initialize the trade outcome analyzer.
        
        Args:
            data_dir: Directory to store analysis results
            lookback_days: Days to look back for pattern analysis
        """
        self.data_dir = data_dir
        self.lookback_days = lookback_days
        self.analysis_file = os.path.join(data_dir, "trade_analysis.csv")
        self.trades_history: List[TradeAnalysisResult] = []
        
        # Ensure data directory exists
        os.makedirs(data_dir, exist_ok=True)
        
        # Load existing analysis
        self._load_existing_analysis()
        
        logger.info(f"TradeOutcomeAnalyzer initialized with {len(self.trades_history)} historical trades")
    
    def analyze_trade(self, trade_context: TradeContext) -> TradeAnalysisResult:
        """
        Analyze a completed trade and categorize its outcome.
        
        Args:
            trade_context: Complete trade information
            
        Returns:
            TradeAnalysisResult with detailed analysis
        """
        logger.info(f"Analyzing trade for {trade_context.symbol}")
        
        # Calculate basic metrics
        pnl = self._calculate_pnl(trade_context)
        pnl_percentage = self._calculate_pnl_percentage(trade_context)
        outcome = self._determine_outcome(pnl_percentage)
        
        # Classify failure type if it's a loss
        failure_type = None
        failure_reasons = []
        confidence_score = 0.0
        
        if outcome == TradeOutcome.LOSS:
            failure_type, failure_reasons, confidence_score = self._classify_failure(trade_context)
        
        # Calculate failure probability for similar setups
        failure_probability = self._calculate_failure_probability(trade_context)
        similar_trades_count = self._count_similar_trades(trade_context)
        
        # Calculate context scores
        market_context_score = self._calculate_market_context_score(trade_context)
        technical_context_score = self._calculate_technical_context_score(trade_context)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(trade_context, failure_type, failure_reasons)
        
        # Create analysis result
        analysis_result = TradeAnalysisResult(
            trade_id=f"{trade_context.symbol}_{trade_context.entry_time.strftime('%Y%m%d_%H%M%S')}",
            symbol=trade_context.symbol,
            outcome=outcome,
            failure_type=failure_type,
            pnl=pnl,
            pnl_percentage=pnl_percentage,
            confidence_score=confidence_score,
            failure_probability=failure_probability,
            similar_trades_count=similar_trades_count,
            analysis_timestamp=datetime.now(),
            failure_reasons=failure_reasons,
            market_context_score=market_context_score,
            technical_context_score=technical_context_score,
            recommendations=recommendations
        )
        
        # Store the analysis
        self.trades_history.append(analysis_result)
        self._save_analysis(analysis_result)
        
        logger.info(f"Trade analysis completed for {trade_context.symbol}: {outcome.value}")
        return analysis_result 
   def _calculate_pnl(self, trade_context: TradeContext) -> float:
        """Calculate profit/loss for the trade."""
        if trade_context.trade_type.lower() == "long":
            return (trade_context.exit_price - trade_context.entry_price) * trade_context.quantity
        else:  # short
            return (trade_context.entry_price - trade_context.exit_price) * trade_context.quantity
    
    def _calculate_pnl_percentage(self, trade_context: TradeContext) -> float:
        """Calculate percentage profit/loss."""
        if trade_context.trade_type.lower() == "long":
            return ((trade_context.exit_price - trade_context.entry_price) / trade_context.entry_price) * 100
        else:  # short
            return ((trade_context.entry_price - trade_context.exit_price) / trade_context.entry_price) * 100
    
    def _determine_outcome(self, pnl_percentage: float) -> TradeOutcome:
        """Determine trade outcome based on PnL."""
        if pnl_percentage > 0.1:  # > 0.1% profit
            return TradeOutcome.WIN
        elif pnl_percentage < -0.1:  # < -0.1% loss
            return TradeOutcome.LOSS
        else:
            return TradeOutcome.BREAKEVEN
    
    def _classify_failure(self, trade_context: TradeContext) -> Tuple[FailureType, List[str], float]:
        """
        Classify the type of trade failure with confidence score.
        
        Returns:
            Tuple of (failure_type, reasons, confidence_score)
        """
        reasons = []
        scores = {}  # failure_type -> confidence_score
        
        # Check for false breakout
        false_breakout_score = self._check_false_breakout(trade_context, reasons)
        if false_breakout_score > 0:
            scores[FailureType.FALSE_BREAKOUT] = false_breakout_score
        
        # Check for stop loss hit
        stop_loss_score = self._check_stop_loss_hit(trade_context, reasons)
        if stop_loss_score > 0:
            scores[FailureType.STOP_LOSS_HIT] = stop_loss_score
        
        # Check for trend reversal
        trend_reversal_score = self._check_trend_reversal(trade_context, reasons)
        if trend_reversal_score > 0:
            scores[FailureType.TREND_REVERSAL] = trend_reversal_score
        
        # Check for low volume
        low_volume_score = self._check_low_volume(trade_context, reasons)
        if low_volume_score > 0:
            scores[FailureType.LOW_VOLUME] = low_volume_score
        
        # Check for news impact
        news_impact_score = self._check_news_impact(trade_context, reasons)
        if news_impact_score > 0:
            scores[FailureType.NEWS_IMPACT] = news_impact_score
        
        # Determine primary failure type
        if scores:
            primary_failure = max(scores.items(), key=lambda x: x[1])
            return primary_failure[0], reasons, primary_failure[1]
        else:
            reasons.append("Unable to classify failure type")
            return FailureType.UNKNOWN, reasons, 0.5
    
    def _check_false_breakout(self, trade_context: TradeContext, reasons: List[str]) -> float:
        """Check if trade failed due to false breakout."""
        confidence = 0.0
        
        # Check if price quickly reversed after entry
        trade_duration = (trade_context.exit_time - trade_context.entry_time).total_seconds() / 60  # minutes
        
        if trade_duration < 30:  # Quick reversal within 30 minutes
            confidence += 0.3
            reasons.append("Quick price reversal within 30 minutes of entry")
        
        # Check volume confirmation
        if (trade_context.volume_at_entry and 
            trade_context.technical_indicators and 
            trade_context.technical_indicators.volume_sma_ratio):
            
            if trade_context.technical_indicators.volume_sma_ratio < 1.2:  # Low volume breakout
                confidence += 0.4
                reasons.append("Breakout occurred on below-average volume")
        
        # Check support/resistance distance
        if (trade_context.technical_indicators and 
            trade_context.technical_indicators.support_resistance_distance):
            
            if trade_context.technical_indicators.support_resistance_distance < 0.5:  # Very close to S/R
                confidence += 0.3
                reasons.append("Entry too close to support/resistance level")
        
        return min(confidence, 1.0)
    
    def _check_stop_loss_hit(self, trade_context: TradeContext, reasons: List[str]) -> float:
        """Check if trade failed due to stop loss being hit."""
        confidence = 0.0
        
        if not trade_context.stop_loss:
            return 0.0
        
        # Check if exit price is at or near stop loss
        stop_loss_tolerance = 0.002  # 0.2% tolerance
        
        if trade_context.trade_type.lower() == "long":
            price_diff = abs(trade_context.exit_price - trade_context.stop_loss) / trade_context.entry_price
        else:  # short
            price_diff = abs(trade_context.exit_price - trade_context.stop_loss) / trade_context.entry_price
        
        if price_diff <= stop_loss_tolerance:
            confidence = 0.9
            reasons.append("Exit price matches stop loss level")
        
        # Check if stop loss was too tight
        if trade_context.technical_indicators and trade_context.technical_indicators.atr:
            atr_multiple = abs(trade_context.entry_price - trade_context.stop_loss) / trade_context.technical_indicators.atr
            
            if atr_multiple < 1.5:  # Stop loss less than 1.5 ATR
                confidence += 0.2
                reasons.append("Stop loss was too tight relative to volatility (< 1.5 ATR)")
        
        return min(confidence, 1.0)
    
    def _check_trend_reversal(self, trade_context: TradeContext, reasons: List[str]) -> float:
        """Check if trade failed due to trend reversal."""
        confidence = 0.0
        
        if not trade_context.market_conditions:
            return 0.0
        
        # Check if market trend was against trade direction
        market_trend = trade_context.market_conditions.market_trend.lower()
        trade_type = trade_context.trade_type.lower()
        
        if ((trade_type == "long" and market_trend == "bearish") or 
            (trade_type == "short" and market_trend == "bullish")):
            confidence += 0.4
            reasons.append(f"Trade direction ({trade_type}) against market trend ({market_trend})")
        
        # Check technical indicators for trend reversal signals
        if trade_context.technical_indicators:
            tech = trade_context.technical_indicators
            
            # RSI divergence
            if tech.rsi:
                if ((trade_type == "long" and tech.rsi > 70) or 
                    (trade_type == "short" and tech.rsi < 30)):
                    confidence += 0.3
                    reasons.append(f"RSI showed overbought/oversold conditions ({tech.rsi})")
            
            # MACD signal
            if tech.macd_signal:
                if ((trade_type == "long" and tech.macd_signal == "bearish") or 
                    (trade_type == "short" and tech.macd_signal == "bullish")):
                    confidence += 0.3
                    reasons.append(f"MACD signal contradicted trade direction")
        
        return min(confidence, 1.0)
    
    def _check_low_volume(self, trade_context: TradeContext, reasons: List[str]) -> float:
        """Check if trade failed due to low volume."""
        confidence = 0.0
        
        # Check volume at entry
        if (trade_context.technical_indicators and 
            trade_context.technical_indicators.volume_sma_ratio):
            
            volume_ratio = trade_context.technical_indicators.volume_sma_ratio
            
            if volume_ratio < 0.7:  # Volume 30% below average
                confidence += 0.5
                reasons.append(f"Entry volume significantly below average ({volume_ratio:.2f}x)")
        
        # Check market conditions volume profile
        if (trade_context.market_conditions and 
            trade_context.market_conditions.volume_profile == "below_average"):
            confidence += 0.3
            reasons.append("Overall market volume was below average")
        
        # Check time of day (low volume periods)
        if trade_context.market_conditions:
            time_of_day = trade_context.market_conditions.time_of_day.lower()
            if time_of_day in ["mid_session"]:  # Lunch time low volume
                confidence += 0.2
                reasons.append("Trade executed during low-volume period")
        
        return min(confidence, 1.0)
    
    def _check_news_impact(self, trade_context: TradeContext, reasons: List[str]) -> float:
        """Check if trade failed due to news impact."""
        confidence = 0.0
        
        if trade_context.news_sentiment is None:
            return 0.0
        
        trade_type = trade_context.trade_type.lower()
        sentiment = trade_context.news_sentiment
        
        # Check if sentiment was strongly against trade direction
        if ((trade_type == "long" and sentiment < -0.5) or 
            (trade_type == "short" and sentiment > 0.5)):
            confidence += 0.6
            reasons.append(f"Strong negative sentiment ({sentiment:.2f}) against trade direction")
        
        # Check for moderate sentiment conflict
        elif ((trade_type == "long" and sentiment < -0.2) or 
              (trade_type == "short" and sentiment > 0.2)):
            confidence += 0.3
            reasons.append(f"Moderate sentiment conflict ({sentiment:.2f})")
        
        return min(confidence, 1.0)    
def _calculate_failure_probability(self, trade_context: TradeContext) -> float:
        """Calculate probability of failure for similar trade setups."""
        similar_trades = self._find_similar_trades(trade_context)
        
        if len(similar_trades) < 3:  # Need minimum sample size
            return 0.5  # Default probability when insufficient data
        
        # Count failures in similar trades
        failures = sum(1 for trade in similar_trades if trade.outcome == TradeOutcome.LOSS)
        failure_rate = failures / len(similar_trades)
        
        # Apply confidence interval adjustment based on sample size
        confidence_adjustment = self._calculate_confidence_adjustment(len(similar_trades))
        
        # Adjust for recency (more recent trades have higher weight)
        recency_weighted_rate = self._apply_recency_weighting(similar_trades)
        
        # Combine base rate with recency weighting
        final_probability = (failure_rate * 0.7) + (recency_weighted_rate * 0.3)
        
        # Apply confidence adjustment
        final_probability = max(0.1, min(0.9, final_probability + confidence_adjustment))
        
        logger.debug(f"Calculated failure probability: {final_probability:.3f} based on {len(similar_trades)} similar trades")
        return final_probability
    
    def _find_similar_trades(self, trade_context: TradeContext) -> List[TradeAnalysisResult]:
        """Find trades with similar characteristics."""
        similar_trades = []
        cutoff_date = datetime.now() - timedelta(days=self.lookback_days)
        
        for trade in self.trades_history:
            if trade.analysis_timestamp < cutoff_date:
                continue
            
            similarity_score = self._calculate_similarity_score(trade_context, trade)
            
            if similarity_score > 0.6:  # Threshold for similarity
                similar_trades.append(trade)
        
        return similar_trades
    
    def _calculate_similarity_score(self, current_trade: TradeContext, historical_trade: TradeAnalysisResult) -> float:
        """Calculate similarity score between trades."""
        score = 0.0
        factors = 0
        
        # Symbol similarity (same stock gets higher weight)
        if current_trade.symbol == historical_trade.symbol:
            score += 0.3
        factors += 1
        
        # Strategy similarity
        # Note: We'd need to store strategy info in TradeAnalysisResult for this
        # For now, we'll skip this factor
        
        # Market conditions similarity (if available)
        if current_trade.market_conditions:
            market_score = self._compare_market_conditions(current_trade, historical_trade)
            score += market_score * 0.3
        factors += 1
        
        # Technical indicators similarity
        if current_trade.technical_indicators:
            tech_score = self._compare_technical_indicators(current_trade, historical_trade)
            score += tech_score * 0.2
        factors += 1
        
        # Time-based similarity (same time of day, day of week)
        time_score = self._compare_time_factors(current_trade, historical_trade)
        score += time_score * 0.2
        factors += 1
        
        return score / factors if factors > 0 else 0.0
    
    def _compare_market_conditions(self, current_trade: TradeContext, historical_trade: TradeAnalysisResult) -> float:
        """Compare market conditions between trades."""
        # This is a simplified comparison - in practice, you'd want more sophisticated logic
        # For now, return a default similarity score
        return 0.5
    
    def _compare_technical_indicators(self, current_trade: TradeContext, historical_trade: TradeAnalysisResult) -> float:
        """Compare technical indicators between trades."""
        # This is a simplified comparison - in practice, you'd want more sophisticated logic
        # For now, return a default similarity score
        return 0.5
    
    def _compare_time_factors(self, current_trade: TradeContext, historical_trade: TradeAnalysisResult) -> float:
        """Compare time-based factors between trades."""
        score = 0.0
        
        # Compare time of day (simplified)
        if (current_trade.market_conditions and 
            hasattr(historical_trade, 'time_of_day')):  # We'd need to store this
            # For now, return default
            score += 0.5
        
        return score
    
    def _calculate_confidence_adjustment(self, sample_size: int) -> float:
        """Calculate confidence interval adjustment based on sample size."""
        if sample_size < 5:
            return 0.2  # High uncertainty
        elif sample_size < 10:
            return 0.1  # Medium uncertainty
        elif sample_size < 20:
            return 0.05  # Low uncertainty
        else:
            return 0.0  # Sufficient sample size
    
    def _apply_recency_weighting(self, similar_trades: List[TradeAnalysisResult]) -> float:
        """Apply recency weighting to failure rate calculation."""
        if not similar_trades:
            return 0.5
        
        weighted_sum = 0.0
        weight_sum = 0.0
        now = datetime.now()
        
        for trade in similar_trades:
            # Calculate weight based on recency (more recent = higher weight)
            days_ago = (now - trade.analysis_timestamp).days
            weight = math.exp(-days_ago / 30.0)  # Exponential decay with 30-day half-life
            
            # Add to weighted sum
            failure_value = 1.0 if trade.outcome == TradeOutcome.LOSS else 0.0
            weighted_sum += failure_value * weight
            weight_sum += weight
        
        return weighted_sum / weight_sum if weight_sum > 0 else 0.5
    
    def _count_similar_trades(self, trade_context: TradeContext) -> int:
        """Count number of similar trades in history."""
        return len(self._find_similar_trades(trade_context))
    
    def _calculate_market_context_score(self, trade_context: TradeContext) -> float:
        """Calculate how market conditions affect trade outcome."""
        # This would analyze how similar market conditions historically performed
        # For now, return a placeholder score
        return 0.5
    
    def _calculate_technical_context_score(self, trade_context: TradeContext) -> float:
        """Calculate how technical setup affects trade outcome."""
        # This would analyze how similar technical setups historically performed
        # For now, return a placeholder score
        return 0.5
    
    def _generate_recommendations(self, trade_context: TradeContext, 
                                failure_type: Optional[FailureType], 
                                failure_reasons: List[str]) -> List[str]:
        """Generate actionable recommendations based on analysis."""
        recommendations = []
        
        if failure_type == FailureType.FALSE_BREAKOUT:
            recommendations.extend([
                "Wait for volume confirmation before entering breakout trades",
                "Use wider stop losses for breakout strategies",
                "Consider waiting for pullback after initial breakout"
            ])
        
        elif failure_type == FailureType.STOP_LOSS_HIT:
            recommendations.extend([
                "Use wider stop losses based on ATR (minimum 2x ATR)",
                "Consider position sizing to accommodate wider stops",
                "Review stop loss placement relative to support/resistance"
            ])
        
        elif failure_type == FailureType.TREND_REVERSAL:
            recommendations.extend([
                "Align trade direction with overall market trend",
                "Use multiple timeframe analysis before entry",
                "Monitor key technical levels for trend continuation"
            ])
        
        elif failure_type == FailureType.LOW_VOLUME:
            recommendations.extend([
                "Avoid trading during low volume periods",
                "Require minimum volume confirmation for entries",
                "Focus on high-volume stocks and time periods"
            ])
        
        elif failure_type == FailureType.NEWS_IMPACT:
            recommendations.extend([
                "Check sentiment analysis before trade entry",
                "Avoid trading against strong sentiment",
                "Monitor news flow during trade execution"
            ])
        
        # Add general recommendations based on failure reasons
        if "Quick price reversal" in str(failure_reasons):
            recommendations.append("Consider using limit orders instead of market orders")
        
        if "below average" in str(failure_reasons).lower():
            recommendations.append("Increase minimum volume requirements for trade selection")
        
        return recommendations    def _l
oad_existing_analysis(self):
        """Load existing trade analysis from CSV file."""
        if not os.path.exists(self.analysis_file):
            logger.info("No existing analysis file found, starting fresh")
            return
        
        try:
            with open(self.analysis_file, 'r', newline='', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                
                for row in reader:
                    # Parse the row data back into TradeAnalysisResult
                    analysis_result = self._parse_csv_row(row)
                    if analysis_result:
                        self.trades_history.append(analysis_result)
            
            logger.info(f"Loaded {len(self.trades_history)} trade analyses from file")
            
        except Exception as e:
            logger.error(f"Error loading existing analysis: {e}")
    
    def _parse_csv_row(self, row: Dict[str, str]) -> Optional[TradeAnalysisResult]:
        """Parse CSV row back into TradeAnalysisResult object."""
        try:
            return TradeAnalysisResult(
                trade_id=row['trade_id'],
                symbol=row['symbol'],
                outcome=TradeOutcome(row['outcome']),
                failure_type=FailureType(row['failure_type']) if row['failure_type'] != 'None' else None,
                pnl=float(row['pnl']),
                pnl_percentage=float(row['pnl_percentage']),
                confidence_score=float(row['confidence_score']),
                failure_probability=float(row['failure_probability']),
                similar_trades_count=int(row['similar_trades_count']),
                analysis_timestamp=datetime.fromisoformat(row['analysis_timestamp']),
                failure_reasons=eval(row['failure_reasons']) if row['failure_reasons'] else [],
                market_context_score=float(row['market_context_score']),
                technical_context_score=float(row['technical_context_score']),
                recommendations=eval(row['recommendations']) if row['recommendations'] else []
            )
        except Exception as e:
            logger.error(f"Error parsing CSV row: {e}")
            return None
    
    def _save_analysis(self, analysis_result: TradeAnalysisResult):
        """Save trade analysis result to CSV file."""
        file_exists = os.path.exists(self.analysis_file)
        
        try:
            with open(self.analysis_file, 'a', newline='', encoding='utf-8') as file:
                fieldnames = [
                    'trade_id', 'symbol', 'outcome', 'failure_type', 'pnl', 'pnl_percentage',
                    'confidence_score', 'failure_probability', 'similar_trades_count',
                    'analysis_timestamp', 'failure_reasons', 'market_context_score',
                    'technical_context_score', 'recommendations'
                ]
                
                writer = csv.DictWriter(file, fieldnames=fieldnames)
                
                # Write header if file is new
                if not file_exists:
                    writer.writeheader()
                
                # Convert analysis result to dict and write
                row_data = asdict(analysis_result)
                
                # Convert enum values to strings
                row_data['outcome'] = row_data['outcome'].value
                row_data['failure_type'] = row_data['failure_type'].value if row_data['failure_type'] else None
                
                writer.writerow(row_data)
                
            logger.debug(f"Saved analysis for trade {analysis_result.trade_id}")
            
        except Exception as e:
            logger.error(f"Error saving analysis to CSV: {e}")
    
    def get_failure_statistics(self, days: int = 30) -> Dict[str, Any]:
        """
        Get failure statistics for the specified period.
        
        Args:
            days: Number of days to look back
            
        Returns:
            Dictionary with failure statistics
        """
        cutoff_date = datetime.now() - timedelta(days=days)
        recent_trades = [t for t in self.trades_history if t.analysis_timestamp >= cutoff_date]
        
        if not recent_trades:
            return {"error": "No trades found in specified period"}
        
        total_trades = len(recent_trades)
        losses = [t for t in recent_trades if t.outcome == TradeOutcome.LOSS]
        wins = [t for t in recent_trades if t.outcome == TradeOutcome.WIN]
        
        # Failure type distribution
        failure_types = {}
        for trade in losses:
            if trade.failure_type:
                failure_types[trade.failure_type.value] = failure_types.get(trade.failure_type.value, 0) + 1
        
        # Calculate statistics
        win_rate = len(wins) / total_trades * 100
        loss_rate = len(losses) / total_trades * 100
        
        avg_win = statistics.mean([t.pnl_percentage for t in wins]) if wins else 0
        avg_loss = statistics.mean([t.pnl_percentage for t in losses]) if losses else 0
        
        return {
            "period_days": days,
            "total_trades": total_trades,
            "win_rate": round(win_rate, 2),
            "loss_rate": round(loss_rate, 2),
            "average_win_pct": round(avg_win, 2),
            "average_loss_pct": round(avg_loss, 2),
            "failure_type_distribution": failure_types,
            "most_common_failure": max(failure_types.items(), key=lambda x: x[1])[0] if failure_types else None
        }
    
    def get_symbol_analysis(self, symbol: str, days: int = 60) -> Dict[str, Any]:
        """
        Get analysis for a specific symbol.
        
        Args:
            symbol: Stock symbol to analyze
            days: Number of days to look back
            
        Returns:
            Dictionary with symbol-specific analysis
        """
        cutoff_date = datetime.now() - timedelta(days=days)
        symbol_trades = [
            t for t in self.trades_history 
            if t.symbol == symbol and t.analysis_timestamp >= cutoff_date
        ]
        
        if not symbol_trades:
            return {"error": f"No trades found for {symbol} in specified period"}
        
        losses = [t for t in symbol_trades if t.outcome == TradeOutcome.LOSS]
        wins = [t for t in symbol_trades if t.outcome == TradeOutcome.WIN]
        
        # Calculate symbol-specific metrics
        total_trades = len(symbol_trades)
        win_rate = len(wins) / total_trades * 100
        
        avg_failure_probability = statistics.mean([t.failure_probability for t in symbol_trades])
        avg_confidence = statistics.mean([t.confidence_score for t in losses]) if losses else 0
        
        # Most common failure reasons
        all_reasons = []
        for trade in losses:
            all_reasons.extend(trade.failure_reasons)
        
        reason_counts = {}
        for reason in all_reasons:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
        
        return {
            "symbol": symbol,
            "period_days": days,
            "total_trades": total_trades,
            "win_rate": round(win_rate, 2),
            "average_failure_probability": round(avg_failure_probability, 3),
            "average_confidence_score": round(avg_confidence, 3),
            "common_failure_reasons": dict(sorted(reason_counts.items(), key=lambda x: x[1], reverse=True)[:5])
        }
    
    def export_detailed_report(self, filename: Optional[str] = None) -> str:
        """
        Export detailed analysis report to CSV.
        
        Args:
            filename: Optional custom filename
            
        Returns:
            Path to exported file
        """
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"trade_analysis_report_{timestamp}.csv"
        
        filepath = os.path.join(self.data_dir, filename)
        
        try:
            with open(filepath, 'w', newline='', encoding='utf-8') as file:
                if not self.trades_history:
                    file.write("No trade data available\n")
                    return filepath
                
                fieldnames = list(asdict(self.trades_history[0]).keys())
                writer = csv.DictWriter(file, fieldnames=fieldnames)
                writer.writeheader()
                
                for trade in self.trades_history:
                    row_data = asdict(trade)
                    # Convert enum values to strings
                    row_data['outcome'] = row_data['outcome'].value
                    row_data['failure_type'] = row_data['failure_type'].value if row_data['failure_type'] else None
                    writer.writerow(row_data)
            
            logger.info(f"Exported detailed report to {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Error exporting report: {e}")
            raise
    
    def cleanup_old_data(self, days_to_keep: int = 180):
        """
        Clean up old trade analysis data.
        
        Args:
            days_to_keep: Number of days of data to retain
        """
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        
        # Filter out old trades
        old_count = len(self.trades_history)
        self.trades_history = [
            t for t in self.trades_history 
            if t.analysis_timestamp >= cutoff_date
        ]
        
        removed_count = old_count - len(self.trades_history)
        
        if removed_count > 0:
            # Rewrite the CSV file with remaining data
            if os.path.exists(self.analysis_file):
                os.remove(self.analysis_file)
            
            for trade in self.trades_history:
                self._save_analysis(trade)
            
            logger.info(f"Cleaned up {removed_count} old trade analyses")
        
        return removed_count