"""
Specialized Trading Agents for Multi-Agent Coordinator

This module contains implementations of various specialized trading agents,
each focusing on different trading strategies and market conditions.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from core.multi_agent_coordinator import TradingAgent, TradeSignal, TradeSignalType

logger = logging.getLogger(__name__)


class BreakoutAgent(TradingAgent):
    """Breakout trading agent that identifies price breakouts from consolidation"""
    
    def __init__(self, agent_id: str = "breakout_agent", lookback_period: int = 20):
        super().__init__(agent_id, "breakout")
        self.lookback_period = lookback_period
        self.min_consolidation_bars = 10
    
    def generate_signal(self, market_data: Dict[str, Any]) -> Optional[TradeSignal]:
        """Generate breakout trading signal"""
        try:
            df = market_data.get('price_data')
            symbol = market_data.get('symbol', 'UNKNOWN')
            
            if df is None or len(df) < self.lookback_period + 5:
                return None
            
            # Calculate support and resistance levels
            recent_data = df.tail(self.lookback_period)
            resistance = recent_data['high'].max()
            support = recent_data['low'].min()
            
            current_price = df['close'].iloc[-1]
            current_volume = df['volume'].iloc[-1]
            avg_volume = df['volume'].rolling(20).mean().iloc[-1]
            
            # Check for consolidation period
            price_range = resistance - support
            avg_price = (resistance + support) / 2
            consolidation_ratio = price_range / avg_price
            
            # Volume confirmation
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
            
            # Calculate confidence based on breakout strength and volume
            if current_price > resistance:
                # Upward breakout
                breakout_strength = (current_price - resistance) / resistance
                confidence = min(0.95, breakout_strength * 20 + (volume_ratio - 1) * 0.3)
                
                if confidence > 0.6 and volume_ratio > 1.3:
                    return TradeSignal(
                        agent_id=self.agent_id,
                        symbol=symbol,
                        signal_type=TradeSignalType.BUY,
                        confidence=confidence,
                        quantity=int(100 * confidence),
                        price=current_price,
                        stop_loss=resistance * 0.995,  # Just below resistance (now support)
                        take_profit=current_price + (current_price - support) * 0.5,
                        reasoning=f"Upward breakout above {resistance:.2f} with {volume_ratio:.1f}x volume"
                    )
            
            elif current_price < support:
                # Downward breakout
                breakout_strength = (support - current_price) / support
                confidence = min(0.95, breakout_strength * 20 + (volume_ratio - 1) * 0.3)
                
                if confidence > 0.6 and volume_ratio > 1.3:
                    return TradeSignal(
                        agent_id=self.agent_id,
                        symbol=symbol,
                        signal_type=TradeSignalType.SELL,
                        confidence=confidence,
                        quantity=int(100 * confidence),
                        price=current_price,
                        stop_loss=support * 1.005,  # Just above support (now resistance)
                        take_profit=current_price - (resistance - current_price) * 0.5,
                        reasoning=f"Downward breakout below {support:.2f} with {volume_ratio:.1f}x volume"
                    )
            
            return None
            
        except Exception as e:
            logger.error(f"Error in breakout agent signal generation: {e}")
            return None


class TrendFollowingAgent(TradingAgent):
    """Trend following agent using multiple timeframe analysis"""
    
    def __init__(self, agent_id: str = "trend_following_agent"):
        super().__init__(agent_id, "trend_following")
        self.short_ma = 10
        self.long_ma = 30
        self.trend_strength_period = 14
    
    def generate_signal(self, market_data: Dict[str, Any]) -> Optional[TradeSignal]:
        """Generate trend following signal"""
        try:
            df = market_data.get('price_data')
            symbol = market_data.get('symbol', 'UNKNOWN')
            
            if df is None or len(df) < self.long_ma + 5:
                return None
            
            # Calculate moving averages
            short_ma = df['close'].rolling(self.short_ma).mean()
            long_ma = df['close'].rolling(self.long_ma).mean()
            
            current_price = df['close'].iloc[-1]
            current_short_ma = short_ma.iloc[-1]
            current_long_ma = long_ma.iloc[-1]
            
            # Trend direction
            trend_up = current_short_ma > current_long_ma
            ma_distance = abs(current_short_ma - current_long_ma) / current_long_ma
            
            # Trend strength using ADX-like calculation
            price_changes = df['close'].pct_change().abs()
            trend_strength = price_changes.rolling(self.trend_strength_period).mean().iloc[-1]
            
            # Volume confirmation
            volume_ratio = df['volume'].iloc[-1] / df['volume'].rolling(20).mean().iloc[-1]
            
            # Price position relative to MAs
            price_above_short = current_price > current_short_ma
            price_above_long = current_price > current_long_ma
            
            # Calculate confidence
            base_confidence = min(0.9, ma_distance * 50 + trend_strength * 10)
            volume_boost = min(0.2, (volume_ratio - 1) * 0.1)
            confidence = base_confidence + volume_boost
            
            if trend_up and price_above_short and price_above_long and confidence > 0.6:
                return TradeSignal(
                    agent_id=self.agent_id,
                    symbol=symbol,
                    signal_type=TradeSignalType.BUY,
                    confidence=confidence,
                    quantity=int(100 * confidence),
                    price=current_price,
                    stop_loss=current_long_ma * 0.98,
                    take_profit=current_price * (1 + ma_distance * 2),
                    reasoning=f"Strong uptrend: MA distance {ma_distance:.3f}, strength {trend_strength:.3f}"
                )
            
            elif not trend_up and not price_above_short and not price_above_long and confidence > 0.6:
                return TradeSignal(
                    agent_id=self.agent_id,
                    symbol=symbol,
                    signal_type=TradeSignalType.SELL,
                    confidence=confidence,
                    quantity=int(100 * confidence),
                    price=current_price,
                    stop_loss=current_long_ma * 1.02,
                    take_profit=current_price * (1 - ma_distance * 2),
                    reasoning=f"Strong downtrend: MA distance {ma_distance:.3f}, strength {trend_strength:.3f}"
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error in trend following agent signal generation: {e}")
            return None


class ScalpingAgent(TradingAgent):
    """High-frequency scalping agent for quick profits"""
    
    def __init__(self, agent_id: str = "scalping_agent"):
        super().__init__(agent_id, "scalping")
        self.min_spread_ratio = 0.001  # Minimum spread for scalping
        self.max_hold_time = 300  # 5 minutes max hold time
    
    def generate_signal(self, market_data: Dict[str, Any]) -> Optional[TradeSignal]:
        """Generate scalping signal based on short-term price movements"""
        try:
            df = market_data.get('price_data')
            symbol = market_data.get('symbol', 'UNKNOWN')
            
            if df is None or len(df) < 10:
                return None
            
            # Short-term momentum
            short_returns = df['close'].pct_change(1).tail(5)
            momentum = short_returns.mean()
            momentum_consistency = 1 - short_returns.std() if short_returns.std() > 0 else 1
            
            # Volume spike detection
            current_volume = df['volume'].iloc[-1]
            avg_volume_5 = df['volume'].tail(5).mean()
            volume_spike = current_volume / avg_volume_5 if avg_volume_5 > 0 else 1
            
            # Bid-ask spread proxy (using high-low range)
            current_spread = (df['high'].iloc[-1] - df['low'].iloc[-1]) / df['close'].iloc[-1]
            
            # Only trade if spread is reasonable
            if current_spread > self.min_spread_ratio * 3:  # Too wide spread
                return None
            
            current_price = df['close'].iloc[-1]
            
            # Quick momentum with volume confirmation
            if momentum > 0.002 and volume_spike > 1.5 and momentum_consistency > 0.7:
                confidence = min(0.8, momentum * 100 + (volume_spike - 1) * 0.2)
                
                return TradeSignal(
                    agent_id=self.agent_id,
                    symbol=symbol,
                    signal_type=TradeSignalType.BUY,
                    confidence=confidence,
                    quantity=200,  # Larger quantity for scalping
                    price=current_price,
                    stop_loss=current_price * 0.998,  # Tight stop loss
                    take_profit=current_price * 1.004,  # Small profit target
                    reasoning=f"Scalp long: momentum {momentum:.4f}, volume spike {volume_spike:.1f}x"
                )
            
            elif momentum < -0.002 and volume_spike > 1.5 and momentum_consistency > 0.7:
                confidence = min(0.8, abs(momentum) * 100 + (volume_spike - 1) * 0.2)
                
                return TradeSignal(
                    agent_id=self.agent_id,
                    symbol=symbol,
                    signal_type=TradeSignalType.SELL,
                    confidence=confidence,
                    quantity=200,
                    price=current_price,
                    stop_loss=current_price * 1.002,
                    take_profit=current_price * 0.996,
                    reasoning=f"Scalp short: momentum {momentum:.4f}, volume spike {volume_spike:.1f}x"
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error in scalping agent signal generation: {e}")
            return None


class ArbitrageAgent(TradingAgent):
    """Statistical arbitrage agent looking for price discrepancies"""
    
    def __init__(self, agent_id: str = "arbitrage_agent"):
        super().__init__(agent_id, "arbitrage")
        self.correlation_period = 30
        self.zscore_threshold = 2.0
    
    def generate_signal(self, market_data: Dict[str, Any]) -> Optional[TradeSignal]:
        """Generate arbitrage signal based on statistical relationships"""
        try:
            df = market_data.get('price_data')
            symbol = market_data.get('symbol', 'UNKNOWN')
            benchmark_data = market_data.get('benchmark_data')  # Market index data
            
            if df is None or benchmark_data is None or len(df) < self.correlation_period:
                return None
            
            # Calculate price ratio to benchmark
            stock_prices = df['close'].tail(self.correlation_period)
            benchmark_prices = benchmark_data['close'].tail(self.correlation_period)
            
            if len(benchmark_prices) != len(stock_prices):
                return None
            
            # Calculate price ratio
            price_ratio = stock_prices / benchmark_prices
            ratio_mean = price_ratio.mean()
            ratio_std = price_ratio.std()
            
            if ratio_std == 0:
                return None
            
            current_ratio = stock_prices.iloc[-1] / benchmark_prices.iloc[-1]
            zscore = (current_ratio - ratio_mean) / ratio_std
            
            current_price = df['close'].iloc[-1]
            
            # Mean reversion arbitrage
            if zscore > self.zscore_threshold:
                # Stock is overpriced relative to benchmark
                confidence = min(0.9, abs(zscore) / 4.0)
                
                return TradeSignal(
                    agent_id=self.agent_id,
                    symbol=symbol,
                    signal_type=TradeSignalType.SELL,
                    confidence=confidence,
                    quantity=int(50 * confidence),
                    price=current_price,
                    stop_loss=current_price * 1.01,
                    take_profit=current_price * (1 - 0.005 * abs(zscore)),
                    reasoning=f"Statistical arbitrage: Z-score {zscore:.2f} (overpriced)"
                )
            
            elif zscore < -self.zscore_threshold:
                # Stock is underpriced relative to benchmark
                confidence = min(0.9, abs(zscore) / 4.0)
                
                return TradeSignal(
                    agent_id=self.agent_id,
                    symbol=symbol,
                    signal_type=TradeSignalType.BUY,
                    confidence=confidence,
                    quantity=int(50 * confidence),
                    price=current_price,
                    stop_loss=current_price * 0.99,
                    take_profit=current_price * (1 + 0.005 * abs(zscore)),
                    reasoning=f"Statistical arbitrage: Z-score {zscore:.2f} (underpriced)"
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error in arbitrage agent signal generation: {e}")
            return None


class VolatilityAgent(TradingAgent):
    """Volatility-based trading agent"""
    
    def __init__(self, agent_id: str = "volatility_agent"):
        super().__init__(agent_id, "volatility")
        self.volatility_period = 20
        self.volatility_threshold = 0.02
    
    def generate_signal(self, market_data: Dict[str, Any]) -> Optional[TradeSignal]:
        """Generate volatility-based trading signal"""
        try:
            df = market_data.get('price_data')
            symbol = market_data.get('symbol', 'UNKNOWN')
            
            if df is None or len(df) < self.volatility_period + 5:
                return None
            
            # Calculate realized volatility
            returns = df['close'].pct_change()
            current_vol = returns.rolling(self.volatility_period).std().iloc[-1]
            vol_mean = returns.rolling(self.volatility_period * 2).std().mean()
            
            if vol_mean == 0:
                return None
            
            vol_ratio = current_vol / vol_mean
            current_price = df['close'].iloc[-1]
            
            # ATR for position sizing
            atr = df.get('atr', pd.Series([current_price * 0.02])).iloc[-1]
            
            # Volatility breakout strategy
            if vol_ratio > 1.5 and current_vol > self.volatility_threshold:
                # High volatility - expect continuation
                recent_return = returns.iloc[-1]
                confidence = min(0.85, (vol_ratio - 1) * 0.3 + abs(recent_return) * 10)
                
                if recent_return > 0:
                    return TradeSignal(
                        agent_id=self.agent_id,
                        symbol=symbol,
                        signal_type=TradeSignalType.BUY,
                        confidence=confidence,
                        quantity=max(50, int(100 / vol_ratio)),  # Smaller size in high vol
                        price=current_price,
                        stop_loss=current_price - atr * 2,
                        take_profit=current_price + atr * 3,
                        reasoning=f"Volatility breakout: vol ratio {vol_ratio:.2f}, recent return {recent_return:.3f}"
                    )
                else:
                    return TradeSignal(
                        agent_id=self.agent_id,
                        symbol=symbol,
                        signal_type=TradeSignalType.SELL,
                        confidence=confidence,
                        quantity=max(50, int(100 / vol_ratio)),
                        price=current_price,
                        stop_loss=current_price + atr * 2,
                        take_profit=current_price - atr * 3,
                        reasoning=f"Volatility breakout: vol ratio {vol_ratio:.2f}, recent return {recent_return:.3f}"
                    )
            
            elif vol_ratio < 0.7 and current_vol < self.volatility_threshold:
                # Low volatility - expect breakout
                # Look for compression patterns
                price_range = (df['high'].tail(10).max() - df['low'].tail(10).min()) / current_price
                
                if price_range < 0.02:  # Tight range
                    # Wait for direction signal
                    volume_ratio = df['volume'].iloc[-1] / df['volume'].rolling(20).mean().iloc[-1]
                    
                    if volume_ratio > 1.2:  # Volume expansion
                        recent_return = returns.iloc[-1]
                        confidence = min(0.75, (1 - vol_ratio) * 0.5 + volume_ratio * 0.2)
                        
                        if recent_return > 0.001:
                            return TradeSignal(
                                agent_id=self.agent_id,
                                symbol=symbol,
                                signal_type=TradeSignalType.BUY,
                                confidence=confidence,
                                quantity=150,  # Larger size in low vol
                                price=current_price,
                                stop_loss=current_price - atr,
                                take_profit=current_price + atr * 2,
                                reasoning=f"Low vol breakout: vol ratio {vol_ratio:.2f}, volume {volume_ratio:.1f}x"
                            )
                        elif recent_return < -0.001:
                            return TradeSignal(
                                agent_id=self.agent_id,
                                symbol=symbol,
                                signal_type=TradeSignalType.SELL,
                                confidence=confidence,
                                quantity=150,
                                price=current_price,
                                stop_loss=current_price + atr,
                                take_profit=current_price - atr * 2,
                                reasoning=f"Low vol breakout: vol ratio {vol_ratio:.2f}, volume {volume_ratio:.1f}x"
                            )
            
            return None
            
        except Exception as e:
            logger.error(f"Error in volatility agent signal generation: {e}")
            return None


def create_specialized_agents() -> List[TradingAgent]:
    """Create all specialized trading agents"""
    agents = [
        BreakoutAgent("breakout_agent"),
        TrendFollowingAgent("trend_following_agent"),
        ScalpingAgent("scalping_agent"),
        ArbitrageAgent("arbitrage_agent"),
        VolatilityAgent("volatility_agent")
    ]
    
    return agents


if __name__ == "__main__":
    # Test the specialized agents
    import matplotlib.pyplot as plt
    
    # Create sample market data
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=100, freq='5min')
    
    # Generate realistic price data
    returns = np.random.normal(0, 0.01, 100)
    prices = 2000 * np.exp(np.cumsum(returns))
    volumes = np.random.randint(1000, 10000, 100)
    
    df = pd.DataFrame({
        'datetime': dates,
        'close': prices,
        'high': prices * (1 + np.random.uniform(0, 0.01, 100)),
        'low': prices * (1 - np.random.uniform(0, 0.01, 100)),
        'volume': volumes
    })
    
    # Add technical indicators
    df['rsi'] = np.random.uniform(20, 80, 100)
    df['atr'] = df['close'] * 0.02
    
    market_data = {
        'symbol': 'TEST',
        'price_data': df,
        'benchmark_data': df.copy()  # Use same data as benchmark for testing
    }
    
    # Test each agent
    agents = create_specialized_agents()
    
    for agent in agents:
        signal = agent.generate_signal(market_data)
        if signal:
            print(f"{agent.agent_id}: {signal.signal_type.value} {signal.symbol} "
                  f"@ {signal.price:.2f} (confidence: {signal.confidence:.2f})")
            print(f"  Reasoning: {signal.reasoning}")
        else:
            print(f"{agent.agent_id}: No signal generated")
        print()