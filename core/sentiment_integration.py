"""
Sentiment Integration Module

This module integrates sentiment analysis into the trading system as a supportive
enhancement layer. It combines sentiment scores with technical indicators using
weighted scoring and enables bidirectional trading based on sentiment-technical alignment.

Key Features:
- Weighted integration: 15-20% sentiment, 80-85% technical
- Bidirectional trading: Long and short opportunities
- Multi-agent enhancement: Each agent considers sentiment
- Risk-adjusted position sizing based on sentiment alignment
- Market regime awareness through sentiment analysis
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from analyzers.sentiment_analyzer import SentimentAnalyzer, SentimentResult, SentimentConfig

logger = logging.getLogger(__name__)


@dataclass
class SentimentTechnicalSignal:
    """Combined sentiment and technical signal"""
    symbol: str
    
    # Technical components
    technical_score: float  # -1 to +1
    technical_confidence: float  # 0 to 1
    technical_signals: Dict[str, float]  # Individual technical indicators
    
    # Sentiment components
    sentiment_score: float  # -1 to +1
    sentiment_confidence: float  # 0 to 1
    sentiment_category: str
    
    # Combined signal
    combined_score: float  # -1 to +1 (weighted combination)
    final_confidence: float  # 0 to 1
    trade_direction: str  # 'long', 'short', 'neutral'
    
    # Position sizing adjustments
    position_size_multiplier: float  # Multiplier for position size
    risk_adjustment: float  # Risk adjustment factor
    
    # Trading logic
    enable_short_selling: bool
    skip_trade: bool
    reasoning: List[str] = field(default_factory=list)
    
    timestamp: datetime = field(default_factory=datetime.now)


class SentimentTechnicalIntegrator:
    """
    Integrates sentiment analysis with technical indicators for enhanced trading decisions
    """
    
    def __init__(self, sentiment_weight: float = 0.18, technical_weight: float = 0.82):
        self.sentiment_weight = sentiment_weight
        self.technical_weight = technical_weight
        
        # Initialize sentiment analyzer
        sentiment_config = SentimentConfig(
            sentiment_weight_in_final_score=sentiment_weight,
            strong_positive_threshold=0.6,
            strong_negative_threshold=-0.6,
            cache_duration_minutes=30
        )
        self.sentiment_analyzer = SentimentAnalyzer(sentiment_config)
        
        # Performance tracking
        self.integration_history = []
        
        logger.info(f"SentimentTechnicalIntegrator initialized (sentiment: {sentiment_weight:.0%}, technical: {technical_weight:.0%})")
    
    def create_enhanced_signal(self, symbol: str, technical_data: Dict[str, float], 
                              market_data: pd.DataFrame) -> SentimentTechnicalSignal:
        """
        Create enhanced trading signal combining sentiment and technical analysis
        
        Args:
            symbol: Stock symbol
            technical_data: Technical indicators and scores
            market_data: Historical market data
            
        Returns:
            SentimentTechnicalSignal with combined analysis
        """
        try:
            # Get sentiment analysis
            sentiment_result = self.sentiment_analyzer.analyze_symbol_sentiment(symbol)
            
            # Extract technical components
            technical_score = technical_data.get('overall_score', 0.0)
            technical_confidence = technical_data.get('confidence', 0.5)
            
            # Calculate combined score
            combined_score = (
                sentiment_result.overall_sentiment * self.sentiment_weight +
                technical_score * self.technical_weight
            )
            
            # Calculate final confidence
            final_confidence = (
                sentiment_result.confidence * self.sentiment_weight +
                technical_confidence * self.technical_weight
            )
            
            # Determine trade direction and adjustments
            trade_direction, position_multiplier, risk_adjustment, enable_short, skip_trade, reasoning = \
                self._determine_trading_logic(sentiment_result, technical_data, combined_score, final_confidence)
            
            # Create enhanced signal
            enhanced_signal = SentimentTechnicalSignal(
                symbol=symbol,
                technical_score=technical_score,
                technical_confidence=technical_confidence,
                technical_signals=technical_data,
                sentiment_score=sentiment_result.overall_sentiment,
                sentiment_confidence=sentiment_result.confidence,
                sentiment_category=sentiment_result.sentiment_category,
                combined_score=combined_score,
                final_confidence=final_confidence,
                trade_direction=trade_direction,
                position_size_multiplier=position_multiplier,
                risk_adjustment=risk_adjustment,
                enable_short_selling=enable_short,
                skip_trade=skip_trade,
                reasoning=reasoning
            )
            
            # Track integration performance
            self._track_integration_performance(enhanced_signal)
            
            return enhanced_signal
            
        except Exception as e:
            logger.error(f"Error creating enhanced signal for {symbol}: {e}")
            return self._get_default_enhanced_signal(symbol, technical_data)
    
    def _determine_trading_logic(self, sentiment_result: SentimentResult, 
                                technical_data: Dict[str, float], 
                                combined_score: float, 
                                final_confidence: float) -> Tuple[str, float, float, bool, bool, List[str]]:
        """
        Determine trading logic based on sentiment-technical alignment
        
        Returns:
            Tuple of (trade_direction, position_multiplier, risk_adjustment, enable_short, skip_trade, reasoning)
        """
        reasoning = []
        
        # Extract key technical indicators
        rsi = technical_data.get('rsi', 50)
        macd_signal = technical_data.get('macd_signal', 'neutral')
        price_vs_vwap = technical_data.get('price_vs_vwap', 0)
        volume_ratio = technical_data.get('volume_ratio', 1.0)
        
        # 🟢 Strong Positive Sentiment Logic
        if sentiment_result.sentiment_category == 'strong_positive' and sentiment_result.confidence > 0.6:
            if combined_score > 0.4 and final_confidence > 0.6:
                # Bullish sentiment + bullish technicals
                trade_direction = 'long'
                position_multiplier = 1.0 + (sentiment_result.overall_sentiment * 0.3)  # Up to 30% increase
                risk_adjustment = 0.9  # Slightly reduce risk due to high confidence
                enable_short = False
                skip_trade = False
                reasoning.append("Strong positive sentiment aligns with bullish technicals")
                reasoning.append(f"Position size increased by {(position_multiplier-1)*100:.0f}% due to sentiment")
            else:
                # Bullish sentiment but weak technicals
                trade_direction = 'long'
                position_multiplier = 1.0 + (sentiment_result.overall_sentiment * 0.1)  # Small increase
                risk_adjustment = 1.1  # Increase risk due to technical weakness
                enable_short = False
                skip_trade = False
                reasoning.append("Positive sentiment but weak technicals - reduced position size")
        
        # 🔴 Strong Negative Sentiment Logic
        elif sentiment_result.sentiment_category == 'strong_negative' and sentiment_result.confidence > 0.6:
            # Check for technical weakness confirmation
            technical_weakness_confirmed = (
                rsi > 70 or  # Overbought
                macd_signal == 'bearish' or
                price_vs_vwap < -0.5 or  # Price well below VWAP
                volume_ratio > 1.5  # High volume (could indicate selling)
            )
            
            if technical_weakness_confirmed and combined_score < -0.3:
                # Strong negative sentiment + technical weakness = Short opportunity
                trade_direction = 'short'
                position_multiplier = 1.0 + (abs(sentiment_result.overall_sentiment) * 0.2)  # Up to 20% increase for shorts
                risk_adjustment = 0.95  # Slightly reduce risk for high-confidence shorts
                enable_short = True
                skip_trade = False
                reasoning.append("Strong negative sentiment with technical weakness - short opportunity")
                reasoning.append("Technical confirmation: " + 
                               ("RSI overbought, " if rsi > 70 else "") +
                               ("MACD bearish, " if macd_signal == 'bearish' else "") +
                               ("Price below VWAP, " if price_vs_vwap < -0.5 else "") +
                               ("High volume" if volume_ratio > 1.5 else ""))
            else:
                # Negative sentiment but no technical confirmation
                trade_direction = 'neutral'
                position_multiplier = 0.7  # Reduce long position sizes
                risk_adjustment = 1.2  # Increase risk due to negative sentiment
                enable_short = False
                skip_trade = True
                reasoning.append("Negative sentiment without technical confirmation - skip trade")
        
        # ⚪️ Neutral/Slightly Negative Sentiment Logic
        elif sentiment_result.sentiment_category in ['neutral', 'negative']:
            if abs(combined_score) < 0.2 or final_confidence < 0.5:
                # Low confidence or neutral signal
                trade_direction = 'neutral'
                position_multiplier = 0.8  # Reduce position size
                risk_adjustment = 1.1  # Increase risk due to uncertainty
                enable_short = False
                skip_trade = True
                reasoning.append("Neutral/weak sentiment with low confidence - skip trade")
            else:
                # Moderate signal strength
                trade_direction = 'long' if combined_score > 0 else 'neutral'
                position_multiplier = 0.9  # Slightly reduce position size
                risk_adjustment = 1.05  # Slightly increase risk
                enable_short = False
                skip_trade = combined_score <= 0
                reasoning.append("Moderate sentiment - conservative approach")
        
        # 🟢 Positive (but not strong) Sentiment Logic
        else:  # positive sentiment
            if combined_score > 0.3 and final_confidence > 0.5:
                # Positive sentiment + decent technicals
                trade_direction = 'long'
                position_multiplier = 1.0 + (sentiment_result.overall_sentiment * 0.15)  # Up to 15% increase
                risk_adjustment = 0.95  # Slightly reduce risk
                enable_short = False
                skip_trade = False
                reasoning.append("Positive sentiment supports technical signals")
            else:
                # Positive sentiment but weak overall signal
                trade_direction = 'long'
                position_multiplier = 1.0
                risk_adjustment = 1.0
                enable_short = False
                skip_trade = combined_score <= 0.1
                reasoning.append("Positive sentiment but weak overall signal")
        
        # Final adjustments based on confidence
        if final_confidence < 0.3:
            skip_trade = True
            reasoning.append("Overall confidence too low - skip trade")
        elif final_confidence < 0.5:
            position_multiplier *= 0.8  # Reduce position size for low confidence
            risk_adjustment *= 1.1  # Increase risk adjustment
            reasoning.append("Low confidence - reduced position size")
        
        return trade_direction, position_multiplier, risk_adjustment, enable_short, skip_trade, reasoning
    
    def enhance_agent_signals(self, agent_signals: List[Dict], market_data: Dict[str, pd.DataFrame]) -> List[Dict]:
        """
        Enhance multi-agent signals with sentiment analysis
        
        Args:
            agent_signals: List of signals from specialized agents
            market_data: Market data for each symbol
            
        Returns:
            Enhanced signals with sentiment integration
        """
        enhanced_signals = []
        
        for signal in agent_signals:
            try:
                symbol = signal.get('symbol')
                if not symbol:
                    enhanced_signals.append(signal)
                    continue
                
                # Get market data for symbol
                symbol_data = market_data.get(symbol, pd.DataFrame())
                
                # Extract technical data from signal
                technical_data = {
                    'overall_score': signal.get('confidence', 0.5) * 2 - 1,  # Convert 0-1 to -1 to +1
                    'confidence': signal.get('confidence', 0.5),
                    'rsi': signal.get('rsi', 50),
                    'macd_signal': signal.get('macd_signal', 'neutral'),
                    'price_vs_vwap': signal.get('price_vs_vwap', 0),
                    'volume_ratio': signal.get('volume_ratio', 1.0)
                }
                
                # Create enhanced signal
                enhanced_signal_obj = self.create_enhanced_signal(symbol, technical_data, symbol_data)
                
                # Update original signal with sentiment enhancements
                enhanced_signal = signal.copy()
                enhanced_signal.update({
                    'sentiment_score': enhanced_signal_obj.sentiment_score,
                    'sentiment_category': enhanced_signal_obj.sentiment_category,
                    'combined_score': enhanced_signal_obj.combined_score,
                    'final_confidence': enhanced_signal_obj.final_confidence,
                    'trade_direction': enhanced_signal_obj.trade_direction,
                    'position_size_multiplier': enhanced_signal_obj.position_size_multiplier,
                    'risk_adjustment': enhanced_signal_obj.risk_adjustment,
                    'enable_short_selling': enhanced_signal_obj.enable_short_selling,
                    'skip_trade': enhanced_signal_obj.skip_trade,
                    'sentiment_reasoning': enhanced_signal_obj.reasoning
                })
                
                enhanced_signals.append(enhanced_signal)
                
            except Exception as e:
                logger.error(f"Error enhancing signal: {e}")
                enhanced_signals.append(signal)  # Use original signal if enhancement fails
        
        return enhanced_signals
    
    def _track_integration_performance(self, enhanced_signal: SentimentTechnicalSignal):
        """Track sentiment-technical integration performance"""
        performance_record = {
            'symbol': enhanced_signal.symbol,
            'timestamp': enhanced_signal.timestamp,
            'sentiment_score': enhanced_signal.sentiment_score,
            'technical_score': enhanced_signal.technical_score,
            'combined_score': enhanced_signal.combined_score,
            'final_confidence': enhanced_signal.final_confidence,
            'trade_direction': enhanced_signal.trade_direction,
            'position_multiplier': enhanced_signal.position_size_multiplier,
            'skip_trade': enhanced_signal.skip_trade
        }
        
        self.integration_history.append(performance_record)
        
        # Keep only last 1000 records
        if len(self.integration_history) > 1000:
            self.integration_history = self.integration_history[-1000:]
    
    def _get_default_enhanced_signal(self, symbol: str, technical_data: Dict[str, float]) -> SentimentTechnicalSignal:
        """Get default enhanced signal when integration fails"""
        return SentimentTechnicalSignal(
            symbol=symbol,
            technical_score=technical_data.get('overall_score', 0.0),
            technical_confidence=technical_data.get('confidence', 0.5),
            technical_signals=technical_data,
            sentiment_score=0.0,
            sentiment_confidence=0.0,
            sentiment_category='neutral',
            combined_score=technical_data.get('overall_score', 0.0),
            final_confidence=technical_data.get('confidence', 0.5),
            trade_direction='long' if technical_data.get('overall_score', 0.0) > 0 else 'neutral',
            position_size_multiplier=1.0,
            risk_adjustment=1.0,
            enable_short_selling=False,
            skip_trade=False,
            reasoning=['Sentiment analysis failed - using technical signals only']
        )
    
    def get_integration_statistics(self) -> Dict[str, Any]:
        """Get sentiment-technical integration statistics"""
        if not self.integration_history:
            return {'message': 'No integration history available'}
        
        recent_records = self.integration_history[-100:]  # Last 100 integrations
        
        # Calculate statistics
        sentiment_scores = [r['sentiment_score'] for r in recent_records]
        technical_scores = [r['technical_score'] for r in recent_records]
        combined_scores = [r['combined_score'] for r in recent_records]
        
        # Count trade directions
        trade_directions = [r['trade_direction'] for r in recent_records]
        direction_counts = {
            'long': trade_directions.count('long'),
            'short': trade_directions.count('short'),
            'neutral': trade_directions.count('neutral')
        }
        
        # Count skipped trades
        skipped_trades = sum(1 for r in recent_records if r['skip_trade'])
        
        return {
            'total_integrations': len(self.integration_history),
            'sentiment_weight': self.sentiment_weight,
            'technical_weight': self.technical_weight,
            'average_sentiment_score': np.mean(sentiment_scores),
            'average_technical_score': np.mean(technical_scores),
            'average_combined_score': np.mean(combined_scores),
            'trade_direction_distribution': direction_counts,
            'skip_rate': skipped_trades / len(recent_records) if recent_records else 0,
            'short_opportunity_rate': direction_counts['short'] / len(recent_records) if recent_records else 0,
            'sentiment_technical_correlation': np.corrcoef(sentiment_scores, technical_scores)[0, 1] if len(sentiment_scores) > 1 else 0
        }


# Example usage and testing
if __name__ == "__main__":
    # Initialize integrator
    integrator = SentimentTechnicalIntegrator(sentiment_weight=0.18, technical_weight=0.82)
    
    # Test with sample technical data
    sample_technical_data = {
        'overall_score': 0.6,
        'confidence': 0.75,
        'rsi': 65,
        'macd_signal': 'bullish',
        'price_vs_vwap': 0.8,
        'volume_ratio': 1.3
    }
    
    # Create sample market data
    sample_market_data = pd.DataFrame({
        'close': [100, 101, 102, 103, 104],
        'volume': [1000, 1100, 1200, 1300, 1400]
    })
    
    # Test integration
    enhanced_signal = integrator.create_enhanced_signal('RELIANCE', sample_technical_data, sample_market_data)
    
    print(f"Enhanced Signal for RELIANCE:")
    print(f"Combined Score: {enhanced_signal.combined_score:.3f}")
    print(f"Final Confidence: {enhanced_signal.final_confidence:.3f}")
    print(f"Trade Direction: {enhanced_signal.trade_direction}")
    print(f"Position Multiplier: {enhanced_signal.position_size_multiplier:.2f}")
    print(f"Enable Short Selling: {enhanced_signal.enable_short_selling}")
    print(f"Skip Trade: {enhanced_signal.skip_trade}")
    print(f"Reasoning: {enhanced_signal.reasoning}")
    
    # Get statistics
    stats = integrator.get_integration_statistics()
    print(f"\nIntegration Statistics: {stats}")