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


class MomentumScalperAgent(TradingAgent):
    """Intraday momentum scalper for quick 1-5 minute trades"""
    
    def __init__(self, agent_id: str = "momentum_scalper_agent"):
        super().__init__(agent_id, "momentum_scalper")
        self.min_volume_ratio = 1.5  # Minimum 1.5x average volume
        self.momentum_periods = [1, 3, 5]  # 1, 3, 5 minute momentum
        self.max_hold_minutes = 15  # Maximum 15 minutes hold
    
    def generate_signal(self, market_data: Dict[str, Any]) -> Optional[TradeSignal]:
        """Generate momentum scalping signal for intraday trading"""
        try:
            df = market_data.get('price_data')
            symbol = market_data.get('symbol', 'UNKNOWN')
            current_time = market_data.get('current_time', datetime.now())
            
            if df is None or len(df) < 10:
                return None
            
            # Check if we're in trading hours (avoid lunch time)
            hour = current_time.hour
            minute = current_time.minute
            
            # Avoid lunch time (11:30 AM - 1:00 PM) and late afternoon (after 3:15 PM)
            if (hour == 11 and minute >= 30) or (hour == 12) or (hour == 0) or (hour >= 15 and minute >= 15):
                return None
            
            current_price = df['close'].iloc[-1]
            current_volume = df['volume'].iloc[-1]
            
            # Volume confirmation - critical for intraday
            avg_volume_10 = df['volume'].tail(10).mean()
            volume_ratio = current_volume / avg_volume_10 if avg_volume_10 > 0 else 1.0
            
            if volume_ratio < self.min_volume_ratio:
                return None  # Insufficient volume
            
            # Calculate short-term momentum
            momentum_1m = df['close'].pct_change(1).iloc[-1] if len(df) > 1 else 0.0
            momentum_3m = df['close'].pct_change(3).iloc[-1] if len(df) > 3 else 0.0
            momentum_5m = df['close'].pct_change(5).iloc[-1] if len(df) > 5 else 0.0
            
            # VWAP calculation for intraday reference
            vwap = (df['close'] * df['volume']).sum() / df['volume'].sum()
            price_vs_vwap = (current_price - vwap) / vwap
            
            # Momentum consistency check
            momentum_signals = [1 if m > 0.002 else -1 if m < -0.002 else 0 for m in [momentum_1m, momentum_3m, momentum_5m]]
            momentum_consistency = abs(sum(momentum_signals)) / len(momentum_signals)
            
            # Calculate confidence based on momentum and volume
            momentum_strength = abs(momentum_1m) * 100  # Convert to percentage
            volume_boost = min(0.3, (volume_ratio - 1.5) * 0.1)
            base_confidence = min(0.8, momentum_strength * 20 + volume_boost)
            
            # Require strong momentum consistency for high confidence
            if momentum_consistency < 0.67:  # At least 2/3 signals agree
                base_confidence *= 0.5
            
            # Long signal
            if (momentum_1m > 0.003 and momentum_3m > 0.001 and 
                volume_ratio > self.min_volume_ratio and 
                price_vs_vwap > -0.005 and  # Not too far below VWAP
                base_confidence > 0.6):
                
                # Tight intraday stops and targets
                stop_loss = current_price * 0.995  # 0.5% stop loss
                take_profit = current_price * 1.008  # 0.8% target
                
                return TradeSignal(
                    agent_id=self.agent_id,
                    symbol=symbol,
                    signal_type=TradeSignalType.BUY,
                    confidence=base_confidence,
                    quantity=int(200 * base_confidence),  # Larger size for scalping
                    price=current_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    reasoning=f"Momentum scalp long: 1m={momentum_1m:.4f}, vol={volume_ratio:.1f}x, VWAP+{price_vs_vwap:.3f}"
                )
            
            # Short signal
            elif (momentum_1m < -0.003 and momentum_3m < -0.001 and 
                  volume_ratio > self.min_volume_ratio and 
                  price_vs_vwap < 0.005 and  # Not too far above VWAP
                  base_confidence > 0.6):
                
                stop_loss = current_price * 1.005  # 0.5% stop loss
                take_profit = current_price * 0.992  # 0.8% target
                
                return TradeSignal(
                    agent_id=self.agent_id,
                    symbol=symbol,
                    signal_type=TradeSignalType.SELL,
                    confidence=base_confidence,
                    quantity=int(200 * base_confidence),
                    price=current_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    reasoning=f"Momentum scalp short: 1m={momentum_1m:.4f}, vol={volume_ratio:.1f}x, VWAP{price_vs_vwap:.3f}"
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error in momentum scalper agent signal generation: {e}")
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


class GapTradingAgent(TradingAgent):
    """Gap trading agent for morning gap opportunities"""
    
    def __init__(self, agent_id: str = "gap_trading_agent"):
        super().__init__(agent_id, "gap_trading")
        self.min_gap_percentage = 0.5  # Minimum 0.5% gap
        self.max_gap_percentage = 5.0  # Maximum 5% gap
        self.gap_trading_window = 30  # First 30 minutes only
    
    def generate_signal(self, market_data: Dict[str, Any]) -> Optional[TradeSignal]:
        """Generate gap trading signal for morning opportunities"""
        try:
            df = market_data.get('price_data')
            symbol = market_data.get('symbol', 'UNKNOWN')
            current_time = market_data.get('current_time', datetime.now())
            
            if df is None or len(df) < 5:
                return None
            
            # Only trade gaps in first 30 minutes (9:15-9:45 AM)
            hour = current_time.hour
            minute = current_time.minute
            
            if not (hour == 9 and 15 <= minute <= 45):
                return None  # Outside gap trading window
            
            # Get previous day's close and current open
            if len(df) < 2:
                return None
            
            prev_close = df['close'].iloc[-2]  # Previous period close
            current_open = df['open'].iloc[-1] if 'open' in df.columns else df['close'].iloc[-1]
            current_price = df['close'].iloc[-1]
            current_volume = df['volume'].iloc[-1]
            
            # Calculate gap percentage
            gap_percentage = ((current_open - prev_close) / prev_close) * 100
            
            # Check if gap is within tradeable range
            if abs(gap_percentage) < self.min_gap_percentage or abs(gap_percentage) > self.max_gap_percentage:
                return None
            
            # Volume confirmation
            avg_volume = df['volume'].tail(10).mean()
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
            
            if volume_ratio < 1.2:  # Need at least 20% above average volume
                return None
            
            # Calculate confidence based on gap size and volume
            gap_strength = min(abs(gap_percentage) / 3.0, 1.0)  # Normalize to 0-1
            volume_boost = min((volume_ratio - 1.0) * 0.3, 0.3)
            confidence = min(0.85, gap_strength * 0.7 + volume_boost)
            
            # Gap up - look for continuation or fade
            if gap_percentage > self.min_gap_percentage:
                # Check if price is holding above gap level
                gap_hold_ratio = (current_price - prev_close) / (current_open - prev_close)
                
                if gap_hold_ratio > 0.7:  # Gap holding strong - continuation play
                    return TradeSignal(
                        agent_id=self.agent_id,
                        symbol=symbol,
                        signal_type=TradeSignalType.BUY,
                        confidence=confidence,
                        quantity=int(150 * confidence),
                        price=current_price,
                        stop_loss=current_price * 0.995,  # Tight 0.5% stop
                        take_profit=current_price * (1 + gap_percentage * 0.01 * 0.5),  # Half gap size target
                        reasoning=f"Gap up continuation: {gap_percentage:.2f}% gap, vol {volume_ratio:.1f}x, holding {gap_hold_ratio:.2f}"
                    )
                
                elif gap_hold_ratio < 0.3:  # Gap fading - fade play
                    return TradeSignal(
                        agent_id=self.agent_id,
                        symbol=symbol,
                        signal_type=TradeSignalType.SELL,
                        confidence=confidence * 0.8,  # Lower confidence for fade
                        quantity=int(100 * confidence),
                        price=current_price,
                        stop_loss=current_open * 1.002,  # Stop above gap open
                        take_profit=prev_close * 1.002,  # Target near previous close
                        reasoning=f"Gap up fade: {gap_percentage:.2f}% gap fading, vol {volume_ratio:.1f}x"
                    )
            
            # Gap down - look for continuation or bounce
            elif gap_percentage < -self.min_gap_percentage:
                gap_hold_ratio = (prev_close - current_price) / (prev_close - current_open)
                
                if gap_hold_ratio > 0.7:  # Gap down holding - continuation
                    return TradeSignal(
                        agent_id=self.agent_id,
                        symbol=symbol,
                        signal_type=TradeSignalType.SELL,
                        confidence=confidence,
                        quantity=int(150 * confidence),
                        price=current_price,
                        stop_loss=current_price * 1.005,  # Tight 0.5% stop
                        take_profit=current_price * (1 + gap_percentage * 0.01 * 0.5),  # Half gap size target
                        reasoning=f"Gap down continuation: {gap_percentage:.2f}% gap, vol {volume_ratio:.1f}x, holding {gap_hold_ratio:.2f}"
                    )
                
                elif gap_hold_ratio < 0.3:  # Gap filling - bounce play
                    return TradeSignal(
                        agent_id=self.agent_id,
                        symbol=symbol,
                        signal_type=TradeSignalType.BUY,
                        confidence=confidence * 0.8,
                        quantity=int(100 * confidence),
                        price=current_price,
                        stop_loss=current_open * 0.998,  # Stop below gap open
                        take_profit=prev_close * 0.998,  # Target near previous close
                        reasoning=f"Gap down bounce: {gap_percentage:.2f}% gap filling, vol {volume_ratio:.1f}x"
                    )
            
            return None
            
        except Exception as e:
            logger.error(f"Error in gap trading agent signal generation: {e}")
            return None


class MACrossoverAgent(TradingAgent):
    """Moving Average Crossover agent for intraday signals"""
    
    def __init__(self, agent_id: str = "ma_crossover_agent"):
        super().__init__(agent_id, "ma_crossover")
        self.fast_ma = 5  # 5-period MA for intraday
        self.slow_ma = 20  # 20-period MA for intraday
        self.min_volume_ratio = 1.2  # Minimum volume confirmation
    
    def generate_signal(self, market_data: Dict[str, Any]) -> Optional[TradeSignal]:
        """Generate MA crossover signal for intraday trading"""
        try:
            df = market_data.get('price_data')
            symbol = market_data.get('symbol', 'UNKNOWN')
            current_time = market_data.get('current_time', datetime.now())
            
            if df is None or len(df) < self.slow_ma + 5:
                return None
            
            # Check trading hours - avoid lunch and late afternoon
            hour = current_time.hour
            minute = current_time.minute
            
            if (hour == 11 and minute >= 30) or (hour == 12) or (hour == 0) or (hour >= 15 and minute >= 15):
                return None
            
            # Calculate moving averages
            fast_ma = df['close'].rolling(self.fast_ma).mean()
            slow_ma = df['close'].rolling(self.slow_ma).mean()
            
            if fast_ma.isna().any() or slow_ma.isna().any():
                return None
            
            current_price = df['close'].iloc[-1]
            current_fast_ma = fast_ma.iloc[-1]
            current_slow_ma = slow_ma.iloc[-1]
            prev_fast_ma = fast_ma.iloc[-2]
            prev_slow_ma = slow_ma.iloc[-2]
            
            # Volume confirmation
            current_volume = df['volume'].iloc[-1]
            avg_volume = df['volume'].rolling(20).mean().iloc[-1]
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
            
            if volume_ratio < self.min_volume_ratio:
                return None
            
            # VWAP for additional confirmation
            vwap = (df['close'] * df['volume']).sum() / df['volume'].sum()
            price_vs_vwap = (current_price - vwap) / vwap
            
            # Detect crossover
            bullish_crossover = (prev_fast_ma <= prev_slow_ma and current_fast_ma > current_slow_ma)
            bearish_crossover = (prev_fast_ma >= prev_slow_ma and current_fast_ma < current_slow_ma)
            
            # Calculate MA separation for confidence
            ma_separation = abs(current_fast_ma - current_slow_ma) / current_slow_ma
            
            # Base confidence from MA separation and volume
            base_confidence = min(0.8, ma_separation * 100 + (volume_ratio - 1.2) * 0.2)
            
            # Bullish crossover
            if (bullish_crossover and 
                current_price > current_fast_ma and  # Price above fast MA
                price_vs_vwap > -0.01 and  # Not too far below VWAP
                base_confidence > 0.6):
                
                # Tight intraday stops
                stop_loss = current_slow_ma * 0.998  # Just below slow MA
                take_profit = current_price * (1 + ma_separation * 3)  # Target based on MA separation
                
                return TradeSignal(
                    agent_id=self.agent_id,
                    symbol=symbol,
                    signal_type=TradeSignalType.BUY,
                    confidence=base_confidence,
                    quantity=int(150 * base_confidence),
                    price=current_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    reasoning=f"Bullish MA crossover: {self.fast_ma}MA > {self.slow_ma}MA, vol {volume_ratio:.1f}x, sep {ma_separation:.3f}"
                )
            
            # Bearish crossover
            elif (bearish_crossover and 
                  current_price < current_fast_ma and  # Price below fast MA
                  price_vs_vwap < 0.01 and  # Not too far above VWAP
                  base_confidence > 0.6):
                
                stop_loss = current_slow_ma * 1.002  # Just above slow MA
                take_profit = current_price * (1 - ma_separation * 3)
                
                return TradeSignal(
                    agent_id=self.agent_id,
                    symbol=symbol,
                    signal_type=TradeSignalType.SELL,
                    confidence=base_confidence,
                    quantity=int(150 * base_confidence),
                    price=current_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    reasoning=f"Bearish MA crossover: {self.fast_ma}MA < {self.slow_ma}MA, vol {volume_ratio:.1f}x, sep {ma_separation:.3f}"
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error in MA crossover agent signal generation: {e}")
            return None


def create_specialized_agents() -> List[TradingAgent]:
    """Create intraday-focused specialized trading agents"""
    agents = [
        BreakoutAgent("intraday_breakout_agent"),
        MomentumScalperAgent("momentum_scalper_agent"),
        ScalpingAgent("enhanced_scalping_agent"),
        GapTradingAgent("gap_trading_agent"),
        MACrossoverAgent("ma_crossover_agent")
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