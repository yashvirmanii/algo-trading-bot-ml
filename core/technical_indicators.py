"""
Enhanced Technical Indicators Module

This module implements comprehensive technical indicators for the trading system:
- Volume Weighted Average Price (VWAP)
- Enhanced RSI with multiple timeframes
- Enhanced MACD with signal line and histogram
- SuperTrend indicator
- Multiple Exponential Moving Averages (9, 21, 50)
- Additional momentum and volatility indicators

All indicators are optimized for real-time processing and multi-agent usage.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Try to import TA-Lib for optimized calculations
try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    logging.warning("TA-Lib not available, using custom implementations")

logger = logging.getLogger(__name__)


@dataclass
class TechnicalIndicatorConfig:
    """Configuration for technical indicators"""
    # RSI settings
    rsi_periods: List[int] = field(default_factory=lambda: [14, 21])
    rsi_overbought: float = 70.0
    rsi_oversold: float = 30.0
    
    # MACD settings
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    
    # EMA settings
    ema_periods: List[int] = field(default_factory=lambda: [9, 21, 50])
    
    # SuperTrend settings
    supertrend_period: int = 10
    supertrend_multiplier: float = 3.0
    
    # VWAP settings
    vwap_reset_period: str = 'daily'  # 'daily', 'weekly', 'session'
    
    # Bollinger Bands
    bb_period: int = 20
    bb_std: float = 2.0
    
    # ATR settings
    atr_period: int = 14


class EnhancedTechnicalIndicators:
    """
    Enhanced technical indicators with optimized calculations
    """
    
    def __init__(self, config: TechnicalIndicatorConfig = None):
        self.config = config or TechnicalIndicatorConfig()
        logger.info("EnhancedTechnicalIndicators initialized")
    
    def calculate_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all technical indicators for a given DataFrame
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with all technical indicators added
        """
        if df.empty or len(df) < 2:
            return df
        
        try:
            # Make a copy to avoid modifying original
            result_df = df.copy()
            
            # 1. Volume Weighted Average Price (VWAP)
            result_df = self.add_vwap(result_df)
            
            # 2. Enhanced RSI (multiple timeframes)
            result_df = self.add_enhanced_rsi(result_df)
            
            # 3. Enhanced MACD
            result_df = self.add_enhanced_macd(result_df)
            
            # 4. SuperTrend
            result_df = self.add_supertrend(result_df)
            
            # 5. Multiple EMAs
            result_df = self.add_multiple_emas(result_df)
            
            # 6. Additional indicators
            result_df = self.add_bollinger_bands(result_df)
            result_df = self.add_atr(result_df)
            result_df = self.add_momentum_indicators(result_df)
            
            # 7. Signal generation
            result_df = self.add_trading_signals(result_df)
            
            logger.debug(f"Calculated indicators for {len(result_df)} data points")
            return result_df
            
        except Exception as e:
            logger.error(f"Error calculating technical indicators: {e}")
            return df
    
    def add_vwap(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add Volume Weighted Average Price (VWAP)
        
        VWAP = Cumulative(Price * Volume) / Cumulative(Volume)
        """
        try:
            if 'volume' not in df.columns:
                logger.warning("Volume data not available for VWAP calculation")
                df['vwap'] = df['close']
                df['price_vs_vwap'] = 0.0
                df['vwap_signal'] = 'neutral'
                return df
            
            # Calculate typical price
            df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
            
            # Calculate VWAP
            df['price_volume'] = df['typical_price'] * df['volume']
            df['cumulative_pv'] = df['price_volume'].cumsum()
            df['cumulative_volume'] = df['volume'].cumsum()
            
            # Avoid division by zero
            df['vwap'] = df['cumulative_pv'] / df['cumulative_volume'].replace(0, 1)
            
            # Price vs VWAP analysis
            df['price_vs_vwap'] = ((df['close'] / df['vwap']) - 1) * 100
            
            # VWAP signals
            df['vwap_signal'] = 'neutral'
            df.loc[df['close'] > df['vwap'], 'vwap_signal'] = 'bullish'
            df.loc[df['close'] < df['vwap'], 'vwap_signal'] = 'bearish'
            
            # VWAP strength (how far from VWAP)
            df['vwap_strength'] = abs(df['price_vs_vwap'])
            
            # Clean up temporary columns
            df.drop(['typical_price', 'price_volume', 'cumulative_pv', 'cumulative_volume'], 
                   axis=1, inplace=True, errors='ignore')
            
            logger.debug("VWAP calculated successfully")
            return df
            
        except Exception as e:
            logger.error(f"Error calculating VWAP: {e}")
            return df
    
    def add_enhanced_rsi(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add Enhanced RSI with multiple timeframes
        """
        try:
            for period in self.config.rsi_periods:
                if len(df) >= period:
                    rsi_col = f'rsi_{period}'
                    
                    if TALIB_AVAILABLE:
                        df[rsi_col] = talib.RSI(df['close'].values, timeperiod=period)
                    else:
                        # Custom RSI calculation
                        delta = df['close'].diff()
                        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
                        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
                        rs = gain / loss.replace(0, 1)
                        df[rsi_col] = 100 - (100 / (1 + rs))
                    
                    # RSI signals
                    df[f'rsi_{period}_signal'] = 'neutral'
                    df.loc[df[rsi_col] > self.config.rsi_overbought, f'rsi_{period}_signal'] = 'overbought'
                    df.loc[df[rsi_col] < self.config.rsi_oversold, f'rsi_{period}_signal'] = 'oversold'
                    
                    # RSI momentum
                    df[f'rsi_{period}_momentum'] = df[rsi_col].diff()
            
            # Combined RSI signal (using primary RSI_14)
            if 'rsi_14' in df.columns:
                df['rsi_combined_signal'] = df['rsi_14_signal']
                
                # RSI divergence detection (simplified)
                if len(df) >= 20:
                    price_trend = df['close'].rolling(10).apply(lambda x: 1 if x.iloc[-1] > x.iloc[0] else -1)
                    rsi_trend = df['rsi_14'].rolling(10).apply(lambda x: 1 if x.iloc[-1] > x.iloc[0] else -1)
                    df['rsi_divergence'] = (price_trend != rsi_trend).astype(int)
            
            logger.debug("Enhanced RSI calculated successfully")
            return df
            
        except Exception as e:
            logger.error(f"Error calculating enhanced RSI: {e}")
            return df
    
    def add_enhanced_macd(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add Enhanced MACD with signal line and histogram
        """
        try:
            if len(df) < self.config.macd_slow:
                return df
            
            if TALIB_AVAILABLE:
                macd_line, macd_signal, macd_hist = talib.MACD(
                    df['close'].values, 
                    fastperiod=self.config.macd_fast,
                    slowperiod=self.config.macd_slow,
                    signalperiod=self.config.macd_signal
                )
                df['macd_line'] = macd_line
                df['macd_signal'] = macd_signal
                df['macd_histogram'] = macd_hist
            else:
                # Custom MACD calculation
                ema_fast = df['close'].ewm(span=self.config.macd_fast).mean()
                ema_slow = df['close'].ewm(span=self.config.macd_slow).mean()
                df['macd_line'] = ema_fast - ema_slow
                df['macd_signal'] = df['macd_line'].ewm(span=self.config.macd_signal).mean()
                df['macd_histogram'] = df['macd_line'] - df['macd_signal']
            
            # MACD signals
            df['macd_trend'] = 'neutral'
            df.loc[df['macd_line'] > df['macd_signal'], 'macd_trend'] = 'bullish'
            df.loc[df['macd_line'] < df['macd_signal'], 'macd_trend'] = 'bearish'
            
            # MACD crossover signals
            df['macd_crossover'] = 0
            macd_cross = (df['macd_line'] > df['macd_signal']) & (df['macd_line'].shift(1) <= df['macd_signal'].shift(1))
            macd_cross_down = (df['macd_line'] < df['macd_signal']) & (df['macd_line'].shift(1) >= df['macd_signal'].shift(1))
            
            df.loc[macd_cross, 'macd_crossover'] = 1  # Bullish crossover
            df.loc[macd_cross_down, 'macd_crossover'] = -1  # Bearish crossover
            
            # MACD momentum
            df['macd_momentum'] = df['macd_histogram'].diff()
            
            logger.debug("Enhanced MACD calculated successfully")
            return df
            
        except Exception as e:
            logger.error(f"Error calculating enhanced MACD: {e}")
            return df
    
    def add_supertrend(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add SuperTrend indicator
        """
        try:
            if len(df) < self.config.supertrend_period:
                return df
            
            # Calculate ATR first
            if 'atr' not in df.columns:
                df = self.add_atr(df)
            
            # Calculate basic upper and lower bands
            hl2 = (df['high'] + df['low']) / 2
            upper_band = hl2 + (self.config.supertrend_multiplier * df['atr'])
            lower_band = hl2 - (self.config.supertrend_multiplier * df['atr'])
            
            # Initialize SuperTrend arrays
            supertrend = pd.Series(index=df.index, dtype=float)
            supertrend_direction = pd.Series(index=df.index, dtype=int)
            
            # Calculate SuperTrend
            for i in range(1, len(df)):
                # Upper band calculation
                if upper_band.iloc[i] < upper_band.iloc[i-1] or df['close'].iloc[i-1] > upper_band.iloc[i-1]:
                    upper_band.iloc[i] = upper_band.iloc[i]
                else:
                    upper_band.iloc[i] = upper_band.iloc[i-1]
                
                # Lower band calculation
                if lower_band.iloc[i] > lower_band.iloc[i-1] or df['close'].iloc[i-1] < lower_band.iloc[i-1]:
                    lower_band.iloc[i] = lower_band.iloc[i]
                else:
                    lower_band.iloc[i] = lower_band.iloc[i-1]
                
                # SuperTrend calculation
                if i == 1:
                    supertrend.iloc[i] = upper_band.iloc[i]
                    supertrend_direction.iloc[i] = 1
                else:
                    if supertrend.iloc[i-1] == upper_band.iloc[i-1] and df['close'].iloc[i] <= upper_band.iloc[i]:
                        supertrend.iloc[i] = upper_band.iloc[i]
                        supertrend_direction.iloc[i] = 1
                    elif supertrend.iloc[i-1] == upper_band.iloc[i-1] and df['close'].iloc[i] > upper_band.iloc[i]:
                        supertrend.iloc[i] = lower_band.iloc[i]
                        supertrend_direction.iloc[i] = -1
                    elif supertrend.iloc[i-1] == lower_band.iloc[i-1] and df['close'].iloc[i] >= lower_band.iloc[i]:
                        supertrend.iloc[i] = lower_band.iloc[i]
                        supertrend_direction.iloc[i] = -1
                    elif supertrend.iloc[i-1] == lower_band.iloc[i-1] and df['close'].iloc[i] < lower_band.iloc[i]:
                        supertrend.iloc[i] = upper_band.iloc[i]
                        supertrend_direction.iloc[i] = 1
                    else:
                        supertrend.iloc[i] = supertrend.iloc[i-1]
                        supertrend_direction.iloc[i] = supertrend_direction.iloc[i-1]
            
            df['supertrend'] = supertrend
            df['supertrend_direction'] = supertrend_direction
            
            # SuperTrend signals
            df['supertrend_signal'] = 'neutral'
            df.loc[df['close'] > df['supertrend'], 'supertrend_signal'] = 'uptrend'
            df.loc[df['close'] < df['supertrend'], 'supertrend_signal'] = 'downtrend'
            
            # Price vs SuperTrend
            df['price_vs_supertrend'] = ((df['close'] / df['supertrend']) - 1) * 100
            
            logger.debug("SuperTrend calculated successfully")
            return df
            
        except Exception as e:
            logger.error(f"Error calculating SuperTrend: {e}")
            return df
    
    def add_multiple_emas(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add multiple Exponential Moving Averages
        """
        try:
            for period in self.config.ema_periods:
                if len(df) >= period:
                    ema_col = f'ema_{period}'
                    
                    if TALIB_AVAILABLE:
                        df[ema_col] = talib.EMA(df['close'].values, timeperiod=period)
                    else:
                        df[ema_col] = df['close'].ewm(span=period).mean()
                    
                    # Price vs EMA
                    df[f'price_vs_ema_{period}'] = ((df['close'] / df[ema_col]) - 1) * 100
            
            # EMA crossover signals
            if 'ema_9' in df.columns and 'ema_21' in df.columns:
                df['ema_9_21_cross'] = 0
                bullish_cross = (df['ema_9'] > df['ema_21']) & (df['ema_9'].shift(1) <= df['ema_21'].shift(1))
                bearish_cross = (df['ema_9'] < df['ema_21']) & (df['ema_9'].shift(1) >= df['ema_21'].shift(1))
                
                df.loc[bullish_cross, 'ema_9_21_cross'] = 1
                df.loc[bearish_cross, 'ema_9_21_cross'] = -1
            
            if 'ema_21' in df.columns and 'ema_50' in df.columns:
                df['ema_21_50_cross'] = 0
                bullish_cross = (df['ema_21'] > df['ema_50']) & (df['ema_21'].shift(1) <= df['ema_50'].shift(1))
                bearish_cross = (df['ema_21'] < df['ema_50']) & (df['ema_21'].shift(1) >= df['ema_50'].shift(1))
                
                df.loc[bullish_cross, 'ema_21_50_cross'] = 1
                df.loc[bearish_cross, 'ema_21_50_cross'] = -1
            
            # EMA trend alignment
            if all(f'ema_{p}' in df.columns for p in [9, 21, 50]):
                df['ema_alignment'] = 'neutral'
                bullish_alignment = (df['ema_9'] > df['ema_21']) & (df['ema_21'] > df['ema_50'])
                bearish_alignment = (df['ema_9'] < df['ema_21']) & (df['ema_21'] < df['ema_50'])
                
                df.loc[bullish_alignment, 'ema_alignment'] = 'bullish'
                df.loc[bearish_alignment, 'ema_alignment'] = 'bearish'
            
            logger.debug("Multiple EMAs calculated successfully")
            return df
            
        except Exception as e:
            logger.error(f"Error calculating multiple EMAs: {e}")
            return df
    
    def add_bollinger_bands(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add Bollinger Bands
        """
        try:
            if len(df) < self.config.bb_period:
                return df
            
            if TALIB_AVAILABLE:
                upper, middle, lower = talib.BBANDS(
                    df['close'].values, 
                    timeperiod=self.config.bb_period,
                    nbdevup=self.config.bb_std,
                    nbdevdn=self.config.bb_std
                )
                df['bb_upper'] = upper
                df['bb_middle'] = middle
                df['bb_lower'] = lower
            else:
                # Custom Bollinger Bands calculation
                df['bb_middle'] = df['close'].rolling(window=self.config.bb_period).mean()
                bb_std = df['close'].rolling(window=self.config.bb_period).std()
                df['bb_upper'] = df['bb_middle'] + (bb_std * self.config.bb_std)
                df['bb_lower'] = df['bb_middle'] - (bb_std * self.config.bb_std)
            
            # Bollinger Band position
            df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
            df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle'] * 100
            
            # BB signals
            df['bb_signal'] = 'neutral'
            df.loc[df['close'] > df['bb_upper'], 'bb_signal'] = 'overbought'
            df.loc[df['close'] < df['bb_lower'], 'bb_signal'] = 'oversold'
            
            logger.debug("Bollinger Bands calculated successfully")
            return df
            
        except Exception as e:
            logger.error(f"Error calculating Bollinger Bands: {e}")
            return df
    
    def add_atr(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add Average True Range (ATR)
        """
        try:
            if len(df) < self.config.atr_period:
                return df
            
            if TALIB_AVAILABLE:
                df['atr'] = talib.ATR(
                    df['high'].values, 
                    df['low'].values, 
                    df['close'].values, 
                    timeperiod=self.config.atr_period
                )
            else:
                # Custom ATR calculation
                high_low = df['high'] - df['low']
                high_close = abs(df['high'] - df['close'].shift())
                low_close = abs(df['low'] - df['close'].shift())
                
                true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
                df['atr'] = true_range.rolling(window=self.config.atr_period).mean()
            
            # ATR as percentage of price
            df['atr_percent'] = (df['atr'] / df['close']) * 100
            
            # Volatility regime based on ATR
            df['volatility_regime'] = 'normal'
            atr_mean = df['atr_percent'].rolling(50).mean()
            atr_std = df['atr_percent'].rolling(50).std()
            
            df.loc[df['atr_percent'] > (atr_mean + atr_std), 'volatility_regime'] = 'high'
            df.loc[df['atr_percent'] < (atr_mean - atr_std), 'volatility_regime'] = 'low'
            
            logger.debug("ATR calculated successfully")
            return df
            
        except Exception as e:
            logger.error(f"Error calculating ATR: {e}")
            return df
    
    def add_momentum_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add additional momentum indicators
        """
        try:
            # Williams %R
            if len(df) >= 14:
                high_14 = df['high'].rolling(14).max()
                low_14 = df['low'].rolling(14).min()
                df['williams_r'] = ((high_14 - df['close']) / (high_14 - low_14)) * -100
            
            # Stochastic Oscillator
            if len(df) >= 14:
                low_14 = df['low'].rolling(14).min()
                high_14 = df['high'].rolling(14).max()
                df['stoch_k'] = ((df['close'] - low_14) / (high_14 - low_14)) * 100
                df['stoch_d'] = df['stoch_k'].rolling(3).mean()
            
            # Rate of Change (ROC)
            for period in [5, 10, 20]:
                if len(df) > period:
                    df[f'roc_{period}'] = ((df['close'] / df['close'].shift(period)) - 1) * 100
            
            logger.debug("Momentum indicators calculated successfully")
            return df
            
        except Exception as e:
            logger.error(f"Error calculating momentum indicators: {e}")
            return df
    
    def add_trading_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add composite trading signals based on multiple indicators
        """
        try:
            # Initialize signal columns
            df['composite_signal'] = 'neutral'
            df['signal_strength'] = 0.0
            df['bullish_signals'] = 0
            df['bearish_signals'] = 0
            
            # Count bullish signals
            bullish_conditions = [
                df.get('vwap_signal') == 'bullish',
                df.get('supertrend_signal') == 'uptrend',
                df.get('macd_trend') == 'bullish',
                df.get('ema_alignment') == 'bullish',
                df.get('rsi_14', 50) < 70,  # Not overbought
                df.get('bb_signal') != 'overbought'
            ]
            
            # Count bearish signals
            bearish_conditions = [
                df.get('vwap_signal') == 'bearish',
                df.get('supertrend_signal') == 'downtrend',
                df.get('macd_trend') == 'bearish',
                df.get('ema_alignment') == 'bearish',
                df.get('rsi_14', 50) > 30,  # Not oversold
                df.get('bb_signal') != 'oversold'
            ]
            
            # Count signals
            for condition in bullish_conditions:
                if isinstance(condition, pd.Series):
                    df['bullish_signals'] += condition.astype(int)
            
            for condition in bearish_conditions:
                if isinstance(condition, pd.Series):
                    df['bearish_signals'] += condition.astype(int)
            
            # Generate composite signal
            signal_diff = df['bullish_signals'] - df['bearish_signals']
            df['signal_strength'] = signal_diff / len(bullish_conditions)
            
            df.loc[signal_diff >= 3, 'composite_signal'] = 'strong_bullish'
            df.loc[(signal_diff >= 1) & (signal_diff < 3), 'composite_signal'] = 'bullish'
            df.loc[signal_diff <= -3, 'composite_signal'] = 'strong_bearish'
            df.loc[(signal_diff <= -1) & (signal_diff > -3), 'composite_signal'] = 'bearish'
            
            logger.debug("Trading signals calculated successfully")
            return df
            
        except Exception as e:
            logger.error(f"Error calculating trading signals: {e}")
            return df
    
    def get_agent_specific_signals(self, df: pd.DataFrame, agent_type: str) -> Dict[str, Any]:
        """
        Get agent-specific signals based on indicator combinations
        
        Args:
            df: DataFrame with calculated indicators
            agent_type: Type of agent ('breakout', 'trend', 'scalping', 'volatility')
            
        Returns:
            Dictionary with agent-specific signals and confidence
        """
        if df.empty:
            return {'signal': 'neutral', 'confidence': 0.0, 'reasoning': []}
        
        latest = df.iloc[-1]
        signals = {'signal': 'neutral', 'confidence': 0.0, 'reasoning': []}
        
        try:
            if agent_type == 'breakout':
                # Breakout Agent: VWAP breakout + volume confirmation
                if (latest.get('vwap_signal') == 'bullish' and 
                    latest.get('price_vs_vwap', 0) > 1.0 and
                    latest.get('volume', 0) > df['volume'].rolling(20).mean().iloc[-1]):
                    signals['signal'] = 'bullish'
                    signals['confidence'] = 0.8
                    signals['reasoning'].append("VWAP breakout with volume confirmation")
                
            elif agent_type == 'trend':
                # Trend Agent: EMA alignment + SuperTrend confirmation
                if (latest.get('ema_alignment') == 'bullish' and 
                    latest.get('supertrend_signal') == 'uptrend'):
                    signals['signal'] = 'bullish'
                    signals['confidence'] = 0.85
                    signals['reasoning'].append("EMA alignment with SuperTrend uptrend")
                elif (latest.get('ema_alignment') == 'bearish' and 
                      latest.get('supertrend_signal') == 'downtrend'):
                    signals['signal'] = 'bearish'
                    signals['confidence'] = 0.85
                    signals['reasoning'].append("EMA alignment with SuperTrend downtrend")
                
            elif agent_type == 'scalping':
                # Scalping Agent: Short-term EMA + RSI for quick entries
                if (latest.get('ema_9_21_cross', 0) == 1 and 
                    30 < latest.get('rsi_14', 50) < 70):
                    signals['signal'] = 'bullish'
                    signals['confidence'] = 0.7
                    signals['reasoning'].append("EMA 9/21 bullish cross with RSI in range")
                elif (latest.get('ema_9_21_cross', 0) == -1 and 
                      30 < latest.get('rsi_14', 50) < 70):
                    signals['signal'] = 'bearish'
                    signals['confidence'] = 0.7
                    signals['reasoning'].append("EMA 9/21 bearish cross with RSI in range")
                
            elif agent_type == 'volatility':
                # Volatility Agent: MACD histogram + volatility regime
                if (latest.get('macd_momentum', 0) > 0 and 
                    latest.get('volatility_regime') == 'high'):
                    signals['signal'] = 'bullish'
                    signals['confidence'] = 0.75
                    signals['reasoning'].append("MACD momentum with high volatility")
            
            return signals
            
        except Exception as e:
            logger.error(f"Error getting agent-specific signals: {e}")
            return {'signal': 'neutral', 'confidence': 0.0, 'reasoning': []}


# Convenience function for easy integration
def calculate_enhanced_indicators(df: pd.DataFrame, config: TechnicalIndicatorConfig = None) -> pd.DataFrame:
    """
    Convenience function to calculate all enhanced technical indicators
    
    Args:
        df: DataFrame with OHLCV data
        config: Optional configuration
        
    Returns:
        DataFrame with all indicators calculated
    """
    calculator = EnhancedTechnicalIndicators(config)
    return calculator.calculate_all_indicators(df)


# Example usage and testing
if __name__ == "__main__":
    # Create sample data for testing
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    np.random.seed(42)
    
    # Generate sample OHLCV data
    close_prices = 100 + np.cumsum(np.random.randn(100) * 0.5)
    high_prices = close_prices + np.random.rand(100) * 2
    low_prices = close_prices - np.random.rand(100) * 2
    open_prices = close_prices + np.random.randn(100) * 0.5
    volumes = np.random.randint(10000, 100000, 100)
    
    sample_df = pd.DataFrame({
        'datetime': dates,
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'volume': volumes
    })
    
    # Calculate indicators
    config = TechnicalIndicatorConfig()
    calculator = EnhancedTechnicalIndicators(config)
    
    result_df = calculator.calculate_all_indicators(sample_df)
    
    print("Enhanced Technical Indicators Test Results:")
    print(f"Original columns: {len(sample_df.columns)}")
    print(f"Enhanced columns: {len(result_df.columns)}")
    print(f"New indicators added: {len(result_df.columns) - len(sample_df.columns)}")
    
    # Show some key indicators
    key_indicators = ['vwap', 'rsi_14', 'macd_line', 'supertrend', 'ema_9', 'ema_21', 'composite_signal']
    available_indicators = [col for col in key_indicators if col in result_df.columns]
    
    print(f"\nSample of calculated indicators:")
    print(result_df[available_indicators].tail())
    
    # Test agent-specific signals
    for agent_type in ['breakout', 'trend', 'scalping', 'volatility']:
        signals = calculator.get_agent_specific_signals(result_df, agent_type)
        print(f"\n{agent_type.title()} Agent Signals:")
        print(f"Signal: {signals['signal']}, Confidence: {signals['confidence']:.2f}")
        if signals['reasoning']:
            print(f"Reasoning: {signals['reasoning'][0]}")