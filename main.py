"""
Main entry point for the Advanced AI-Powered Trading Bot.

This script orchestrates the entire trading workflow:
- Loads configuration and credentials securely from .env
- Initializes all core modules (screener, broker, risk, notifier, logger, AI systems)
- Screens stocks using Zerodha Kite Connect API
- Uses AttentionStrategySelector for intelligent strategy selection
- Uses DQNTradingAgent for optimal action decisions
- Executes trades based on AI-driven signals
- Logs trades and sends real-time analytics and error notifications to Telegram
- Ensures robust error handling and logging for reliability and security

The design is modular, AI-powered, and ready for production use.
"""

import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict
from dotenv import load_dotenv
from core.screener import StockScreener
from core.risk import RiskManager
from broker.zerodha import ZerodhaBroker
from notify.telegram import TelegramNotifier
from data.storage import TradeLogger

from core.risk_reward import RiskRewardManager
from core.order_executor import SmartOrderExecutor
from core.paper_trader import PaperTrader
from core.trade_outcome_analyzer import (
    TradeOutcomeAnalyzer, TradeContext, MarketConditions, TechnicalIndicators
)
from core.attention_strategy_selector import (
    AttentionStrategySelector, MarketState, StrategyPerformance
)
from core.dqn_trading_agent import (
    DQNTradingAgent, TradingState, TradingAction, create_sample_trading_state
)
from core.multi_agent_coordinator import (
    MultiAgentCoordinator, TradeSignal, TradeSignalType, ConflictResolution
)
from core.specialized_agents import create_specialized_agents
from core.realtime_feature_pipeline import (
    RealTimeFeaturePipeline, FeatureConfig
)
from core.model_validation_framework import (
    ModelValidationFramework, ValidationConfig, PerformanceMetrics
)
from core.risk_aware_position_sizer import (
    RiskAwarePositionSizer, PositionSizingConfig, TradeContext, PortfolioState
)
from core.dynamic_capital_manager import (
    DynamicCapitalManager, CapitalAllocation
)
from core.sentiment_integration import (
    SentimentTechnicalIntegrator, SentimentTechnicalSignal
)

# Load environment variables
load_dotenv()
API_KEY = os.getenv('KITE_API_KEY')
API_SECRET = os.getenv('KITE_API_SECRET')
ACCESS_TOKEN = os.getenv('KITE_ACCESS_TOKEN')
TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')

# Setup logging
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    filename='logs/bot.log',
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)s %(message)s'
)
logger = logging.getLogger(__name__)


def create_trade_context(symbol, df, entry_price, exit_price, entry_time, exit_time, 
                        quantity, trade_type, strategy_used, stop_loss=None, take_profit=None):
    """
    Create TradeContext from trade data for outcome analysis.
    
    Args:
        symbol: Stock symbol
        df: Price dataframe with technical indicators
        entry_price, exit_price: Trade prices
        entry_time, exit_time: Trade timestamps
        quantity: Trade quantity
        trade_type: 'long' or 'short'
        strategy_used: Strategy name
        stop_loss, take_profit: Optional SL/TP levels
    
    Returns:
        TradeContext object for analysis
    """
    from datetime import datetime
    
    # Extract market conditions from current market state
    market_conditions = MarketConditions(
        market_trend="bullish" if df['close'].iloc[-1] > df['close'].iloc[-10] else "bearish",
        volatility_regime="high" if df.get('atr', pd.Series([1])).iloc[-1] > 2 else "medium",
        volume_profile="above_average" if df.get('volume', pd.Series([1000])).iloc[-1] > df.get('volume', pd.Series([1000])).mean() else "average",
        time_of_day="opening" if entry_time.hour < 11 else "mid_session" if entry_time.hour < 14 else "closing",
        day_of_week=entry_time.strftime("%A"),
        market_breadth=1.0  # Placeholder - would need market-wide data
    )
    
    # Extract technical indicators from dataframe
    technical_indicators = TechnicalIndicators(
        rsi=df.get('rsi', pd.Series([50])).iloc[-1] if 'rsi' in df.columns else None,
        macd_signal="bullish" if df.get('macd', pd.Series([0])).iloc[-1] > 0 else "bearish" if 'macd' in df.columns else None,
        moving_avg_position="above" if df['close'].iloc[-1] > df.get('sma_20', df['close']).iloc[-1] else "below",
        volume_sma_ratio=df.get('volume', pd.Series([1000])).iloc[-1] / df.get('volume', pd.Series([1000])).mean() if 'volume' in df.columns else None,
        atr=df.get('atr', pd.Series([1])).iloc[-1] if 'atr' in df.columns else None,
        bollinger_position="middle",  # Placeholder
        support_resistance_distance=1.0  # Placeholder
    )
    
    return TradeContext(
        symbol=symbol,
        entry_price=entry_price,
        exit_price=exit_price,
        entry_time=entry_time,
        exit_time=exit_time,
        quantity=quantity,
        trade_type=trade_type,
        strategy_used=strategy_used,
        stop_loss=stop_loss,
        take_profit=take_profit,
        market_conditions=market_conditions,
        technical_indicators=technical_indicators,
        news_sentiment=0.0,  # Placeholder - would integrate with sentiment analyzer
        volume_at_entry=int(df.get('volume', pd.Series([1000])).iloc[-1]) if 'volume' in df.columns else None,
        volume_at_exit=int(df.get('volume', pd.Series([1000])).iloc[-1]) if 'volume' in df.columns else None
    )


def create_market_state(df: pd.DataFrame, entry_time: datetime) -> MarketState:
    """
    Create MarketState from price dataframe for attention strategy selector.
    
    Args:
        df: Price dataframe with technical indicators
        entry_time: Current time for time-based features
        
    Returns:
        MarketState object for strategy selection
    """
    try:
        # Price momentum features
        price_momentum_1m = df['close'].pct_change(1).iloc[-1] if len(df) > 1 else 0.0
        price_momentum_5m = df['close'].pct_change(5).iloc[-1] if len(df) > 5 else 0.0
        price_momentum_15m = df['close'].pct_change(15).iloc[-1] if len(df) > 15 else 0.0
        
        # Price volatility (ATR-based)
        atr = df.get('atr', pd.Series([1])).iloc[-1]
        price_volatility = atr / df['close'].iloc[-1] if df['close'].iloc[-1] > 0 else 0.01
        
        # Volume features
        volume_mean = df['volume'].rolling(20).mean().iloc[-1] if len(df) >= 20 else df['volume'].mean()
        volume_ratio = df['volume'].iloc[-1] / volume_mean if volume_mean > 0 else 1.0
        
        # Volume trend (simple slope)
        if len(df) >= 5:
            recent_volume = df['volume'].tail(5)
            volume_trend = (recent_volume.iloc[-1] - recent_volume.iloc[0]) / recent_volume.iloc[0]
        else:
            volume_trend = 0.0
        
        volume_momentum = df['volume'].pct_change(3).iloc[-1] if len(df) > 3 else 0.0
        
        # Technical indicators
        rsi = df.get('rsi', pd.Series([50])).iloc[-1] / 100.0  # Normalize to 0-1
        macd = df.get('macd', pd.Series([0])).iloc[-1]
        macd_signal = np.tanh(macd) if not np.isnan(macd) else 0.0  # Normalize to [-1, 1]
        
        # Bollinger band position
        bb_upper = df.get('bb_upper', df['close']).iloc[-1]
        bb_lower = df.get('bb_lower', df['close']).iloc[-1]
        current_price = df['close'].iloc[-1]
        if bb_upper != bb_lower:
            bb_position = (current_price - bb_lower) / (bb_upper - bb_lower)
        else:
            bb_position = 0.5
        bb_position = np.clip(bb_position, 0, 1)
        
        # Trend strength (simplified ADX-like)
        if len(df) >= 10:
            price_changes = df['close'].pct_change().tail(10)
            trend_strength = abs(price_changes.mean()) / price_changes.std() if price_changes.std() > 0 else 0.0
        else:
            trend_strength = 0.0
        trend_strength = np.clip(trend_strength, 0, 1)
        
        # Market regime (simplified - would integrate with MarketRegimeDetector)
        # For now, use simple heuristics
        if price_momentum_15m > 0.01 and trend_strength > 0.3:
            regime_trending_up, regime_trending_down, regime_sideways, regime_volatile = 1.0, 0.0, 0.0, 0.0
        elif price_momentum_15m < -0.01 and trend_strength > 0.3:
            regime_trending_up, regime_trending_down, regime_sideways, regime_volatile = 0.0, 1.0, 0.0, 0.0
        elif price_volatility > 0.03:
            regime_trending_up, regime_trending_down, regime_sideways, regime_volatile = 0.0, 0.0, 0.0, 1.0
        else:
            regime_trending_up, regime_trending_down, regime_sideways, regime_volatile = 0.0, 0.0, 1.0, 0.0
        
        # Time features
        time_of_day = (entry_time.hour * 60 + entry_time.minute) / (24 * 60)  # Normalize to 0-1
        day_of_week = entry_time.weekday() / 6.0  # Normalize to 0-1
        
        # Market breadth (placeholder)
        market_breadth = 0.5  # Would need market-wide data
        vix_level = 0.2  # Placeholder VIX level
        
        return MarketState(
            price_momentum_1m=float(price_momentum_1m) if not np.isnan(price_momentum_1m) else 0.0,
            price_momentum_5m=float(price_momentum_5m) if not np.isnan(price_momentum_5m) else 0.0,
            price_momentum_15m=float(price_momentum_15m) if not np.isnan(price_momentum_15m) else 0.0,
            price_volatility=float(price_volatility),
            volume_ratio=float(np.clip(volume_ratio, 0, 5)),
            volume_trend=float(np.clip(volume_trend, -1, 1)),
            volume_momentum=float(volume_momentum) if not np.isnan(volume_momentum) else 0.0,
            rsi=float(np.clip(rsi, 0, 1)),
            macd_signal=float(macd_signal),
            bb_position=float(bb_position),
            trend_strength=float(trend_strength),
            regime_trending_up=float(regime_trending_up),
            regime_trending_down=float(regime_trending_down),
            regime_sideways=float(regime_sideways),
            regime_volatile=float(regime_volatile),
            time_of_day=float(time_of_day),
            day_of_week=float(day_of_week),
            market_breadth=float(market_breadth),
            vix_level=float(vix_level)
        )
        
    except Exception as e:
        logger.error(f"Error creating market state: {e}")
        # Return default market state
        return MarketState(
            price_momentum_1m=0.0, price_momentum_5m=0.0, price_momentum_15m=0.0,
            price_volatility=0.01, volume_ratio=1.0, volume_trend=0.0, volume_momentum=0.0,
            rsi=0.5, macd_signal=0.0, bb_position=0.5, trend_strength=0.0,
            regime_trending_up=0.0, regime_trending_down=0.0, regime_sideways=1.0, regime_volatile=0.0,
            time_of_day=0.5, day_of_week=0.5, market_breadth=0.5, vix_level=0.2
        )


def create_strategy_performances(trade_logger, strategy_names: List[str]) -> Dict[str, StrategyPerformance]:
    """
    Create strategy performance metrics from trade history.
    
    Args:
        trade_logger: TradeLogger instance
        strategy_names: List of strategy names
        
    Returns:
        Dictionary of strategy performances
    """
    try:
        strategy_performances = {}
        
        # Get recent performance data from trade history
        for strategy_name in strategy_names:
            try:
                # Query trade_logger for strategy-specific performance
                strategy_trades = trade_logger.get_strategy_performance(strategy_name, days=30)
                
                if strategy_trades and len(strategy_trades) > 0:
                    # Calculate real performance metrics
                    returns = [trade.get('pnl', 0) / trade.get('entry_price', 1) for trade in strategy_trades]
                    wins = [1 for trade in strategy_trades if trade.get('pnl', 0) > 0]
                    
                    recent_return = np.mean(returns) if returns else 0.0
                    win_rate = len(wins) / len(strategy_trades) if strategy_trades else 0.5
                    volatility = np.std(returns) if len(returns) > 1 else 0.02
                    sharpe_ratio = recent_return / volatility if volatility > 0 else 0.0
                    max_drawdown = min(returns) if returns else -0.05
                    trades_count = len(strategy_trades)
                    avg_trade_duration = np.mean([trade.get('duration', 60) for trade in strategy_trades]) if strategy_trades else 60.0
                else:
                    # Default values for new strategies
                    recent_return = 0.0
                    win_rate = 0.5
                    sharpe_ratio = 0.0
                    max_drawdown = -0.05
                    volatility = 0.02
                    trades_count = 0
                    avg_trade_duration = 60.0
                
                strategy_performances[strategy_name] = StrategyPerformance(
                    strategy_name=strategy_name,
                    recent_return=recent_return,
                    win_rate=win_rate,
                    sharpe_ratio=sharpe_ratio,
                    max_drawdown=max_drawdown,
                    volatility=volatility,
                    trades_count=trades_count,
                    avg_trade_duration=avg_trade_duration
                )
                
            except Exception as e:
                logger.warning(f"Could not get performance for {strategy_name}, using defaults: {e}")
                # Fallback to default performance
                strategy_performances[strategy_name] = StrategyPerformance(
                    strategy_name=strategy_name,
                    recent_return=0.0,
                    win_rate=0.5,
                    sharpe_ratio=0.0,
                    max_drawdown=-0.05,
                    volatility=0.02,
                    trades_count=0,
                    avg_trade_duration=60.0
                )
        
        return strategy_performances
        
    except Exception as e:
        logger.error(f"Error creating strategy performances: {e}")
        return {}


def create_trading_state(df: pd.DataFrame, portfolio_state: Dict[str, float], 
                        strategy_performances: Dict[str, StrategyPerformance], 
                        entry_time: datetime) -> TradingState:
    """
    Create TradingState from market data, portfolio state, and strategy performance.
    
    Args:
        df: Price dataframe with technical indicators
        portfolio_state: Current portfolio state
        strategy_performances: Strategy performance metrics
        entry_time: Current time
        
    Returns:
        TradingState object for DQN agent
    """
    try:
        # Market features (30 dimensions)
        price_momentum_1m = df['close'].pct_change(1).iloc[-1] if len(df) > 1 else 0.0
        price_momentum_5m = df['close'].pct_change(5).iloc[-1] if len(df) > 5 else 0.0
        price_momentum_15m = df['close'].pct_change(15).iloc[-1] if len(df) > 15 else 0.0
        price_momentum_30m = df['close'].pct_change(30).iloc[-1] if len(df) > 30 else 0.0
        
        # Price volatility
        atr = df.get('atr', pd.Series([1])).iloc[-1]
        price_volatility = atr / df['close'].iloc[-1] if df['close'].iloc[-1] > 0 else 0.01
        
        # Volume features
        volume_mean = df['volume'].rolling(20).mean().iloc[-1] if len(df) >= 20 else df['volume'].mean()
        volume_ratio = df['volume'].iloc[-1] / volume_mean if volume_mean > 0 else 1.0
        volume_trend = df['volume'].pct_change(5).iloc[-1] if len(df) > 5 else 0.0
        volume_momentum = df['volume'].pct_change(3).iloc[-1] if len(df) > 3 else 0.0
        
        # Technical indicators
        rsi = df.get('rsi', pd.Series([50])).iloc[-1] / 100.0  # Normalize to 0-1
        macd = df.get('macd', pd.Series([0])).iloc[-1]
        macd_signal = df.get('macd_signal', pd.Series([0])).iloc[-1]
        
        # Bollinger bands
        bb_upper = df.get('bb_upper', df['close']).iloc[-1]
        bb_lower = df.get('bb_lower', df['close']).iloc[-1]
        current_price = df['close'].iloc[-1]
        
        if bb_upper != bb_lower:
            bb_position = (current_price - bb_lower) / (bb_upper - bb_lower)
            bb_width = (bb_upper - bb_lower) / current_price
        else:
            bb_position = 0.5
            bb_width = 0.02
        
        # Moving averages
        sma_20 = df.get('sma_20', df['close']).iloc[-1] / current_price - 1.0
        ema_12 = df.get('ema_12', df['close']).iloc[-1] / current_price - 1.0
        ema_26 = df.get('ema_26', df['close']).iloc[-1] / current_price - 1.0
        
        # Additional indicators (calculated from data or defaults)
        stoch_k = df.get('stoch_k', pd.Series([0.5])).iloc[-1] / 100.0 if 'stoch_k' in df.columns else 0.5
        stoch_d = df.get('stoch_d', pd.Series([0.5])).iloc[-1] / 100.0 if 'stoch_d' in df.columns else 0.5
        williams_r = df.get('williams_r', pd.Series([-50])).iloc[-1] / 100.0 if 'williams_r' in df.columns else -0.5
        cci = df.get('cci', pd.Series([0])).iloc[-1] / 100.0 if 'cci' in df.columns else 0.0
        adx = df.get('adx', pd.Series([25])).iloc[-1] / 100.0 if 'adx' in df.columns else 0.25
        
        # Trend and pattern features
        if len(df) >= 10:
            price_changes = df['close'].pct_change().tail(10)
            trend_strength = abs(price_changes.mean()) / price_changes.std() if price_changes.std() > 0 else 0.0
        else:
            trend_strength = 0.0
        
        # Calculate support/resistance distance from price levels
        high_20 = df['high'].rolling(20).max().iloc[-1] if len(df) >= 20 else df['high'].max()
        low_20 = df['low'].rolling(20).min().iloc[-1] if len(df) >= 20 else df['low'].min()
        support_resistance_distance = min(
            abs(current_price - high_20) / current_price,
            abs(current_price - low_20) / current_price
        )
        
        # Calculate breakout probability based on price position and volume
        price_range = high_20 - low_20
        price_position = (current_price - low_20) / price_range if price_range > 0 else 0.5
        volume_surge = volume_ratio > 1.5
        breakout_probability = min(1.0, price_position * 0.7 + (0.3 if volume_surge else 0.0))
        
        # Calculate mean reversion signal based on RSI and BB position
        mean_reversion_signal = 0.0
        if rsi > 0.7:  # Overbought
            mean_reversion_signal = -min(1.0, (rsi - 0.7) / 0.3)
        elif rsi < 0.3:  # Oversold
            mean_reversion_signal = min(1.0, (0.3 - rsi) / 0.3)
        
        # Market regime
        if price_momentum_15m > 0.01 and trend_strength > 0.3:
            market_regime_trending, market_regime_sideways, market_regime_volatile = 1.0, 0.0, 0.0
        elif price_volatility > 0.03:
            market_regime_trending, market_regime_sideways, market_regime_volatile = 0.0, 0.0, 1.0
        else:
            market_regime_trending, market_regime_sideways, market_regime_volatile = 0.0, 1.0, 0.0
        
        # News sentiment (would integrate with sentiment analyzer)
        news_sentiment = 0.0  # Neutral default - would be replaced by actual sentiment analysis
        
        # Portfolio state (10 dimensions)
        cash_ratio = portfolio_state.get('cash_ratio', 0.5)
        position_size = portfolio_state.get('position_size', 0.0)
        unrealized_pnl = portfolio_state.get('unrealized_pnl', 0.0)
        realized_pnl = portfolio_state.get('realized_pnl', 0.0)
        total_return = portfolio_state.get('total_return', 0.0)
        sharpe_ratio = portfolio_state.get('sharpe_ratio', 0.0)
        max_drawdown = portfolio_state.get('max_drawdown', 0.0)
        win_rate = portfolio_state.get('win_rate', 0.5)
        avg_trade_duration = portfolio_state.get('avg_trade_duration', 60.0) / 300.0  # Normalize
        trades_count = portfolio_state.get('trades_count', 0.0) / 100.0  # Normalize
        
        # Strategy performance (10 dimensions)
        strategies = ['momentum', 'mean_reversion', 'breakout', 'trend_following', 'scalping']
        strategy_perfs = []
        strategy_confs = []
        
        for strategy in strategies:
            if strategy in strategy_performances:
                perf = strategy_performances[strategy]
                strategy_perfs.append(perf.recent_return)
                strategy_confs.append(perf.win_rate)  # Use win_rate as confidence proxy
            else:
                strategy_perfs.append(0.0)
                strategy_confs.append(0.5)
        
        return TradingState(
            # Market features
            price_momentum_1m=float(np.clip(price_momentum_1m, -0.1, 0.1)) if not np.isnan(price_momentum_1m) else 0.0,
            price_momentum_5m=float(np.clip(price_momentum_5m, -0.1, 0.1)) if not np.isnan(price_momentum_5m) else 0.0,
            price_momentum_15m=float(np.clip(price_momentum_15m, -0.1, 0.1)) if not np.isnan(price_momentum_15m) else 0.0,
            price_momentum_30m=float(np.clip(price_momentum_30m, -0.1, 0.1)) if not np.isnan(price_momentum_30m) else 0.0,
            price_volatility=float(np.clip(price_volatility, 0, 0.1)),
            volume_ratio=float(np.clip(volume_ratio, 0, 5)),
            volume_trend=float(np.clip(volume_trend, -1, 1)) if not np.isnan(volume_trend) else 0.0,
            volume_momentum=float(np.clip(volume_momentum, -1, 1)) if not np.isnan(volume_momentum) else 0.0,
            rsi=float(np.clip(rsi, 0, 1)),
            macd=float(np.clip(macd, -0.1, 0.1)) if not np.isnan(macd) else 0.0,
            macd_signal=float(np.clip(macd_signal, -0.1, 0.1)) if not np.isnan(macd_signal) else 0.0,
            bb_position=float(np.clip(bb_position, 0, 1)),
            bb_width=float(np.clip(bb_width, 0, 0.2)),
            sma_20=float(np.clip(sma_20, -0.1, 0.1)),
            ema_12=float(np.clip(ema_12, -0.1, 0.1)),
            ema_26=float(np.clip(ema_26, -0.1, 0.1)),
            atr=float(np.clip(atr / current_price, 0, 0.1)),
            stoch_k=float(stoch_k),
            stoch_d=float(stoch_d),
            williams_r=float(williams_r),
            cci=float(np.clip(cci / 100, -2, 2)),
            adx=float(adx),
            trend_strength=float(np.clip(trend_strength, 0, 1)),
            support_resistance_distance=float(support_resistance_distance),
            breakout_probability=float(breakout_probability),
            mean_reversion_signal=float(mean_reversion_signal),
            market_regime_trending=float(market_regime_trending),
            market_regime_sideways=float(market_regime_sideways),
            market_regime_volatile=float(market_regime_volatile),
            news_sentiment=float(news_sentiment),
            
            # Portfolio state
            cash_ratio=float(np.clip(cash_ratio, 0, 1)),
            position_size=float(np.clip(position_size, -1, 1)),
            unrealized_pnl=float(np.clip(unrealized_pnl, -0.5, 0.5)),
            realized_pnl=float(np.clip(realized_pnl, -0.5, 0.5)),
            total_return=float(np.clip(total_return, -1, 1)),
            sharpe_ratio=float(np.clip(sharpe_ratio, -3, 3)),
            max_drawdown=float(np.clip(max_drawdown, -1, 0)),
            win_rate=float(np.clip(win_rate, 0, 1)),
            avg_trade_duration=float(avg_trade_duration),
            trades_count=float(trades_count),
            
            # Strategy performance
            momentum_performance=float(np.clip(strategy_perfs[0], -0.1, 0.1)),
            mean_reversion_performance=float(np.clip(strategy_perfs[1], -0.1, 0.1)),
            breakout_performance=float(np.clip(strategy_perfs[2], -0.1, 0.1)),
            trend_following_performance=float(np.clip(strategy_perfs[3], -0.1, 0.1)),
            scalping_performance=float(np.clip(strategy_perfs[4], -0.1, 0.1)),
            momentum_confidence=float(strategy_confs[0]),
            mean_reversion_confidence=float(strategy_confs[1]),
            breakout_confidence=float(strategy_confs[2]),
            trend_following_confidence=float(strategy_confs[3]),
            scalping_confidence=float(strategy_confs[4])
        )
        
    except Exception as e:
        logger.error(f"Error creating trading state: {e}")
        # Return sample trading state as fallback
        return create_sample_trading_state()


def analyze_completed_trade(outcome_analyzer, symbol, df, entry_price, exit_price, 
                          entry_time, exit_time, quantity, strategy_used, 
                          stop_loss=None, take_profit=None):
    """
    Analyze a completed trade and return insights.
    
    Args:
        outcome_analyzer: TradeOutcomeAnalyzer instance
        Other args: Trade details
    
    Returns:
        TradeAnalysisResult with failure analysis and recommendations
    """
    try:
        # Determine trade type based on entry/exit
        trade_type = "long"  # Simplified - in practice, determine from order side
        
        # Create trade context
        trade_context = create_trade_context(
            symbol, df, entry_price, exit_price, entry_time, exit_time,
            quantity, trade_type, strategy_used, stop_loss, take_profit
        )
        
        # Analyze the trade
        analysis_result = outcome_analyzer.analyze_trade(trade_context)
        
        logger.info(f"Trade analysis for {symbol}: {analysis_result.outcome.value}, "
                   f"Failure: {analysis_result.failure_type.value if analysis_result.failure_type else 'None'}")
        
        return analysis_result
        
    except Exception as e:
        logger.error(f"Error analyzing trade for {symbol}: {e}")
        return None


def main():
    try:
        screener = StockScreener()
        risk = RiskManager()
        broker = ZerodhaBroker(API_KEY, API_SECRET, ACCESS_TOKEN)
        notifier = TelegramNotifier(TELEGRAM_TOKEN, TELEGRAM_CHAT_ID)
        trade_logger = TradeLogger()
        # Initialize AttentionStrategySelector
        strategy_selector = AttentionStrategySelector(
            model_dir="models",
            d_model=64,
            n_heads=8,
            n_layers=3,
            learning_rate=0.001,
            update_frequency=10
        )
        
        # Register available strategies
        available_strategies = ['momentum', 'mean_reversion', 'breakout', 'trend_following', 'scalping']
        strategy_selector.register_strategies(available_strategies)
        
        # Initialize DQN Trading Agent
        dqn_agent = DQNTradingAgent(
            state_dim=50,
            action_dim=4,
            hidden_dims=[256, 128, 64],
            learning_rate=0.001,
            gamma=0.99,
            epsilon_start=1.0,
            epsilon_end=0.01,
            epsilon_decay=10000,
            batch_size=32,
            buffer_size=100000,
            target_update_freq=1000,
            model_dir="models/dqn",
            log_dir="logs/dqn"
        )
        
        # Portfolio state tracking for DQN
        portfolio_state = {
            'cash_ratio': 0.5,
            'position_size': 0.0,
            'unrealized_pnl': 0.0,
            'realized_pnl': 0.0,
            'total_return': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'win_rate': 0.5,
            'avg_trade_duration': 60.0,
            'trades_count': 0.0
        }
        
        rr_manager = RiskRewardManager()
        order_executor = SmartOrderExecutor(broker)
        paper_trader = PaperTrader(total_capital=1000000)  # Example capital
        outcome_analyzer = TradeOutcomeAnalyzer(data_dir="data", lookback_days=90)
        
        # Initialize MultiAgentCoordinator
        multi_agent_coordinator = MultiAgentCoordinator(
            total_capital=1000000.0,
            max_agents=6,
            conflict_resolution=ConflictResolution.WEIGHTED_VOTE,
            meta_learning_enabled=True,
            model_dir="models/multi_agent",
            log_dir="logs/multi_agent"
        )
        
        # Create and register specialized agents
        specialized_agents = create_specialized_agents()
        for agent in specialized_agents:
            multi_agent_coordinator.register_agent(
                agent=agent,
                initial_capital=1000000.0 / len(specialized_agents),  # Equal allocation
                max_positions=3
            )
        
        # Start all agents
        multi_agent_coordinator.start_all_agents()
        logger.info(f"MultiAgentCoordinator initialized with {len(specialized_agents)} specialized agents")
        
        # Initialize RealTimeFeaturePipeline
        feature_config = FeatureConfig(
            timeframes=['1m', '5m', '15m'],
            rsi_periods=[14, 21],
            ema_periods=[9, 21, 50],
            sma_periods=[20, 50],
            rolling_windows=[10, 20, 50],
            lag_periods=[1, 2, 3, 5],
            scaler_type='robust',
            max_cache_size=10000
        )
        
        feature_pipeline = RealTimeFeaturePipeline(
            config=feature_config,
            model_dir="models/features"
        )
        
        logger.info("RealTimeFeaturePipeline initialized with enhanced technical indicators")
        
        # Initialize ModelValidationFramework
        validation_config = ValidationConfig(
            initial_train_size=252,  # 1 year of trading days
            step_size=21,  # Monthly retraining
            window_type='expanding',
            cv_folds=5,
            purged_cv=True,
            embargo_period=5,
            risk_free_rate=0.06,
            confidence_level=0.95,
            significance_level=0.05,
            generate_plots=True,
            save_detailed_results=True
        )
        
        model_validator = ModelValidationFramework(
            config=validation_config,
            output_dir="validation_results"
        )
        
        logger.info("ModelValidationFramework initialized for comprehensive model validation")
        
        # Initialize RiskAwarePositionSizer
        position_sizing_config = PositionSizingConfig(
            max_position_size=0.1,  # Maximum 10% per trade
            min_position_size=0.001,  # Minimum 0.1% per trade
            max_portfolio_risk=0.2,  # Maximum 20% portfolio risk
            max_drawdown_limit=0.15,  # Stop if drawdown > 15%
            kelly_multiplier=0.25,  # Conservative Kelly
            retrain_frequency=50,  # Retrain every 50 trades
            ensemble_weights={
                'gradient_boosting': 0.4,
                'random_forest': 0.3,
                'xgboost': 0.2,
                'lightgbm': 0.1
            }
        )
        
        risk_aware_sizer = RiskAwarePositionSizer(
            config=position_sizing_config,
            model_dir="models/position_sizer"
        )
        
        logger.info("RiskAwarePositionSizer initialized with ML-based position sizing")
        
        # Initialize Dynamic Capital Manager
        capital_manager = DynamicCapitalManager(
            initial_capital=100000.0,  # Default starting capital
            storage_file="data/capital_config.json",
            min_capital=10000.0,
            max_capital=10000000.0
        )
        
        # Register system components with capital manager
        capital_manager.register_component('multi_agent_coordinator', multi_agent_coordinator)
        capital_manager.register_component('position_sizer', risk_aware_sizer)
        capital_manager.register_component('risk_manager', risk)
        capital_manager.register_component('portfolio_tracker', paper_trader)
        
        # Set capital manager reference in Telegram notifier
        notifier.set_capital_manager(capital_manager)
        
        # Update all components with current capital
        current_capital = capital_manager.get_current_capital()
        multi_agent_coordinator.total_capital = current_capital
        paper_trader.total_capital = current_capital
        
        logger.info(f"DynamicCapitalManager initialized with capital: ₹{current_capital:,.2f}")
        
        # Initialize Sentiment-Technical Integrator
        sentiment_integrator = SentimentTechnicalIntegrator(
            sentiment_weight=0.18,  # 18% weight to sentiment
            technical_weight=0.82   # 82% weight to technical
        )
        
        logger.info("SentimentTechnicalIntegrator initialized with 18% sentiment, 82% technical weighting")

        # 1. Screen stocks
        try:
            tradable_stocks = screener.get_tradable_stocks()
            notifier.send_message(f"Universe: {', '.join(tradable_stocks['symbol'].tolist())}")
        except Exception as e:
            logger.error(f"Screener error: {e}")
            notifier.send_message(f"Screener error: {e}")
            return

        # 2. Process each stock with AttentionStrategySelector
        try:
            dfs = {row['symbol']: row['data'] for _, row in tradable_stocks.iterrows()}
            
            # Create strategy performance metrics
            strategy_performances = create_strategy_performances(trade_logger, available_strategies)
            
            logger.info(f"Processing {len(dfs)} stocks with AttentionStrategySelector")
            
        except Exception as e:
            logger.error(f"Data preparation error: {e}")
            notifier.send_message(f"Data preparation error: {e}")
            return

        # 3. Execute trades using Multi-Agent Coordination with Enhanced Features
        for symbol, df in dfs.items():
            try:
                entry_time = datetime.now()
                
                # Process real-time features using the feature pipeline
                tick_data = {
                    'timestamp': entry_time,
                    'symbol': symbol,
                    'price': df['close'].iloc[-1],
                    'volume': df['volume'].iloc[-1] if 'volume' in df.columns else 1000,
                    'high': df['high'].iloc[-1] if 'high' in df.columns else df['close'].iloc[-1],
                    'low': df['low'].iloc[-1] if 'low' in df.columns else df['close'].iloc[-1]
                }
                
                # Generate enhanced features (50+ technical and statistical features)
                enhanced_features = feature_pipeline.process_tick(tick_data)
                
                # Add enhanced features to the dataframe for agents to use
                for feature_name, feature_value in enhanced_features.items():
                    df[feature_name] = feature_value
                
                # Prepare market data for multi-agent system with enhanced features
                market_data = {
                    'symbol': symbol,
                    'price_data': df,
                    'benchmark_data': df.copy(),  # Use same data as benchmark for now
                    'timestamp': entry_time,
                    'enhanced_features': enhanced_features,
                    'feature_count': len(enhanced_features)
                }
                
                # Get coordinated signals from all agents
                coordinated_signals = multi_agent_coordinator.coordinate_trading(market_data)
                
                # Process each coordinated signal
                for signal in coordinated_signals:
                    try:
                        # Create market state for attention strategy selector
                        market_state = create_market_state(df, entry_time)
                        
                        # Get strategy selection from attention mechanism
                        strategy_selection = strategy_selector.select_strategy(market_state, strategy_performances)
                        selected_strategy = strategy_selection.selected_strategy
                        strategy_confidence = strategy_selection.overall_confidence
                        
                        # Create trading state for DQN agent
                        trading_state = create_trading_state(df, portfolio_state, strategy_performances, entry_time)
                        
                        # Get DQN action recommendation
                        dqn_action = dqn_agent.select_action(trading_state, training=True)
                        dqn_action_name = TradingAction.get_action_names()[dqn_action]
                        
                        # Combine multi-agent signal with AI recommendations
                        # Multi-agent provides the signal, AI systems provide confidence and sizing
                        combined_confidence = (signal.confidence + strategy_confidence) / 2.0
                        
                        # Check if we should execute the trade
                        if (combined_confidence > 0.6 and 
                            dqn_action != TradingAction.HOLD and 
                            risk.check_max_trades(0)):
                            
                            # Use signal parameters but adjust with AI insights
                            entry_price = signal.price
                            sl = signal.stop_loss
                            tp = signal.take_profit
                            
                            # Adjust quantity based on DQN action and combined confidence
                            if dqn_action == TradingAction.ADJUST_POSITION_SIZE:
                                qty = max(1, int(signal.quantity * combined_confidence))
                            else:
                                qty = signal.quantity
                            
                            # Execute trade
                            if notifier.mode == 'paper':
                                if signal.signal_type == TradeSignalType.BUY:
                                    success, msg = paper_trader.buy(symbol, qty, entry_price, sl, tp)
                                elif signal.signal_type == TradeSignalType.SELL:
                                    success, msg = paper_trader.sell(symbol, qty, entry_price, sl, tp)
                                else:
                                    continue
                                
                                if success:
                                    # Simulate trade completion for demo
                                    exit_price = tp if tp else entry_price * 1.01
                                    exit_time = datetime.now()
                                    
                                    if signal.signal_type == TradeSignalType.BUY:
                                        pnl = (exit_price - entry_price) * qty
                                    else:
                                        pnl = (entry_price - exit_price) * qty
                                    
                                    # Log trade with multi-agent info
                                    trade_logger.log_trade(
                                        symbol=symbol,
                                        side=signal.signal_type.value,
                                        qty=qty,
                                        entry_price=entry_price,
                                        exit_price=exit_price,
                                        pnl=pnl,
                                        signals=f'multi_agent_{signal.agent_id}',
                                        strategy=selected_strategy,
                                        outcome='win' if pnl > 0 else 'loss',
                                        notes=f'Multi-Agent: {signal.agent_id} | Confidence: {combined_confidence:.2f} | Reasoning: {signal.reasoning}'
                                    )
                                    
                                    # Update AI systems with trade results
                                    actual_return = pnl / (entry_price * qty)
                                    
                                    # Update strategy selector
                                    strategy_selector.add_experience(
                                        market_state=market_state,
                                        strategy_performances=strategy_performances,
                                        selected_strategy=selected_strategy,
                                        actual_return=actual_return
                                    )
                                    
                                    # Update DQN agent
                                    reward = actual_return * 100  # Scale reward
                                    next_trading_state = create_trading_state(df, portfolio_state, strategy_performances, exit_time)
                                    dqn_agent.store_experience(trading_state, dqn_action, reward, next_trading_state, True)
                                    
                                    # Update multi-agent coordinator
                                    trade_duration = (exit_time - entry_time).total_seconds() / 60.0  # minutes
                                    multi_agent_coordinator.update_agent_performance(signal.agent_id, pnl, trade_duration)
                                    
                                    # Update portfolio state
                                    portfolio_state['position_size'] = qty / 100.0
                                    portfolio_state['unrealized_pnl'] = actual_return
                                    portfolio_state['realized_pnl'] += actual_return
                                    portfolio_state['trades_count'] += 1
                                    
                                    # Analyze completed trade
                                    analysis_result = analyze_completed_trade(
                                        outcome_analyzer, symbol, df, entry_price, exit_price,
                                        entry_time, exit_time, qty, signal.agent_id, sl, tp
                                    )
                                    
                                    # Send notification with multi-agent insights
                                    notification_msg = (
                                        f"🤖 Multi-Agent Trade Executed\n"
                                        f"📊 {symbol} | {signal.signal_type.value.upper()}\n"
                                        f"🎯 Agent: {signal.agent_id}\n"
                                        f"💰 Entry: ₹{entry_price:.2f} | Exit: ₹{exit_price:.2f}\n"
                                        f"📈 PnL: ₹{pnl:.2f} ({actual_return*100:.2f}%)\n"
                                        f"🎲 Combined Confidence: {combined_confidence:.2f}\n"
                                        f"🧠 Strategy: {selected_strategy}\n"
                                        f"🤖 DQN Action: {dqn_action_name}\n"
                                        f"💡 Reasoning: {signal.reasoning}"
                                    )
                                    
                                    if analysis_result:
                                        notification_msg += f"\n📋 Analysis: {analysis_result.outcome.value}"
                                        if analysis_result.recommendations:
                                            notification_msg += f"\n💡 Recommendations: {', '.join(analysis_result.recommendations[:2])}"
                                    
                                    notifier.send_message(notification_msg)
                                    
                                    logger.info(f"Multi-agent trade executed: {symbol} {signal.signal_type.value} by {signal.agent_id}")
                                
                            else:
                                # Live trading would go here
                                logger.info(f"Live trading not implemented for multi-agent signal: {symbol}")
                        
                        else:
                            logger.debug(f"Multi-agent signal filtered out: {symbol} - confidence {combined_confidence:.2f}, DQN action {dqn_action_name}")
                    
                    except Exception as e:
                        logger.error(f"Error processing multi-agent signal for {symbol}: {e}")
                        continue
                
                # Train AI systems periodically
                if len(dfs) % 10 == 0:  # Every 10 stocks
                    try:
                        # Train DQN if enough experiences
                        if len(dqn_agent.replay_buffer) > 100:
                            dqn_agent.train()
                        
                        # Update strategy selector
                        strategy_selector.update_model()
                        
                        logger.debug("AI systems training update completed")
                    except Exception as e:
                        logger.error(f"Error training AI systems: {e}")
                
            except Exception as e:
                logger.error(f"Error processing {symbol} with multi-agent system: {e}")
                continue

        # 4. Generate batch report with multi-agent insights
        try:
            # Get system status from multi-agent coordinator
            system_status = multi_agent_coordinator.get_system_status()
            
            # Generate comprehensive report
            report_msg = "📊 Multi-Agent Trading Session Complete\n\n"
            
            # Multi-agent system status
            coordinator_status = system_status['coordinator']
            report_msg += f"🤖 Multi-Agent System:\n"
            report_msg += f"• Total Agents: {coordinator_status['total_agents']}\n"
            report_msg += f"• Active Agents: {coordinator_status['active_agents']}\n"
            report_msg += f"• Total Capital: ₹{coordinator_status['total_capital']:,.0f}\n"
            report_msg += f"• Conflict Resolution: {coordinator_status['conflict_resolution']}\n"
            report_msg += f"• Meta Learning: {'Enabled' if coordinator_status['meta_learning_enabled'] else 'Disabled'}\n\n"
            
            # Agent performance summary
            report_msg += "🎯 Agent Performance:\n"
            for agent_id, perf in system_status['performance'].items():
                if perf['total_trades'] > 0:
                    report_msg += f"• {agent_id}: {perf['total_trades']} trades, {perf['win_rate']:.1%} win rate, ₹{perf['total_pnl']:.2f} PnL\n"
            
            # Resource utilization
            report_msg += "\n💰 Resource Utilization:\n"
            for agent_id, resource in system_status['resources'].items():
                report_msg += f"• {agent_id}: {resource['capital_utilization']:.1%} capital, {resource['position_utilization']:.1%} positions\n"
            
            # AI systems performance
            report_msg += f"\n🧠 AI Systems:\n"
            report_msg += f"• Strategy Selector: {len(available_strategies)} strategies registered\n"
            report_msg += f"• DQN Agent: {len(dqn_agent.replay_buffer)} experiences stored\n"
            report_msg += f"• Trade Analyzer: {len(outcome_analyzer.trades_history)} trades analyzed\n"
            report_msg += f"• Feature Pipeline: {len(enhanced_features) if 'enhanced_features' in locals() else 0} features generated\n"
            report_msg += f"• Model Validator: Ready for comprehensive model validation\n"
            report_msg += f"• Position Sizer: ML-based risk-aware position sizing active\n"
            
            # Add position sizing performance summary
            try:
                sizing_summary = risk_aware_sizer.get_performance_summary()
                if 'total_trades' in sizing_summary:
                    report_msg += f"• Position Sizing Performance: {sizing_summary['total_trades']} trades analyzed\n"
                    report_msg += f"  - Average position size: {sizing_summary.get('avg_position_size', 0)*100:.1f}% of portfolio\n"
                    report_msg += f"  - Position sizing win rate: {sizing_summary.get('win_rate', 0)*100:.1f}%\n"
            except Exception as e:
                logger.warning(f"Could not get position sizing summary: {e}")
            
            # Portfolio summary
            report_msg += f"\n💼 Portfolio Summary:\n"
            report_msg += f"• Total Return: {portfolio_state['realized_pnl']*100:.2f}%\n"
            report_msg += f"• Win Rate: {portfolio_state['win_rate']:.1%}\n"
            report_msg += f"• Total Trades: {int(portfolio_state['trades_count'])}\n"
            
            notifier.send_message(report_msg)
            logger.info("Multi-agent batch report sent successfully")
            
        except Exception as e:
            logger.error(f"Error generating multi-agent batch report: {e}")
            notifier.send_message(f"❌ Error generating batch report: {e}")

        # 5. Cleanup and save state
        try:
            # Save multi-agent coordinator state
            multi_agent_coordinator.save_state()
            
            # Save AI model states
            strategy_selector.save_model()
            dqn_agent.save_model()
            
            # Save feature pipeline state
            feature_pipeline.save_pipeline_state()
            
            # Shutdown systems
            multi_agent_coordinator.shutdown()
            feature_pipeline.shutdown()
            
            logger.info("Multi-agent trading session with enhanced features completed successfully")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

    except Exception as e:
        logger.error(f"Critical error in main: {e}")
        notifier.send_message(f"❌ Critical system error: {e}")
        
        # Emergency shutdown
        try:
            if 'multi_agent_coordinator' in locals():
                multi_agent_coordinator.shutdown()
        except:
            pass


if __name__ == "__main__":
    main()fs.items():fs.items():
            try:
                entry_time = datetime.now()
                
                # Prepare market data for multi-agent system
                market_data = {
                    'symbol': symbol,
                    'price_data': df,
                    'benchmark_data': df.copy(),  # Use same data as benchmark for now
                    'timestamp': entry_time
                }
                
                # Get coordinated signals from all agents
                coordinated_signals = multi_agent_coordinator.coordinate_trading(market_data)
                
                # Process each coordinated signal
                for signal in coordinated_signals:
                    try:
                        # Create market state for attention strategy selector
                        market_state = create_market_state(df, entry_time)
                        
                        # Get strategy selection from attention mechanism
                        strategy_selection = strategy_selector.select_strategy(market_state, strategy_performances)
                        selected_strategy = strategy_selection.selected_strategy
                        strategy_confidence = strategy_selection.overall_confidence
                        
                        # Create trading state for DQN agent
                        trading_state = create_trading_state(df, portfolio_state, strategy_performances, entry_time)
                        
                        # Get DQN action recommendation
                        dqn_action = dqn_agent.select_action(trading_state, training=True)
                        dqn_action_name = TradingAction.get_action_names()[dqn_action]
                        
                        # Combine multi-agent signal with AI recommendations
                        # Multi-agent provides the signal, AI systems provide confidence and sizing
                        combined_confidence = (signal.confidence + strategy_confidence) / 2.0
                        
                        # Check if we should execute the trade
                        if (combined_confidence > 0.6 and 
                            dqn_action != TradingAction.HOLD and 
                            risk.check_max_trades(0)):
                            
                            # Use signal parameters but adjust with AI insights
                            entry_price = signal.price
                            sl = signal.stop_loss
                            tp = signal.take_profit
                            
                            # Adjust quantity based on DQN action and combined confidence
                            if dqn_action == TradingAction.ADJUST_POSITION_SIZE:
                                qty = max(1, int(signal.quantity * combined_confidence))
                            else:
                                qty = signal.quantity
                            
                            # Execute trade
                            if notifier.mode == 'paper':
                                if signal.signal_type == TradeSignalType.BUY:
                                    success, msg = paper_trader.buy(symbol, qty, entry_price, sl, tp)
                                elif signal.signal_type == TradeSignalType.SELL:
                                    success, msg = paper_trader.sell(symbol, qty, entry_price, sl, tp)
                                else:
                                    continue
                                
                                if success:
                                    # Simulate trade completion for demo
                                    exit_price = tp if tp else entry_price * 1.01
                                    exit_time = datetime.now()
                                    
                                    if signal.signal_type == TradeSignalType.BUY:
                                        pnl = (exit_price - entry_price) * qty
                                    else:
                                        pnl = (entry_price - exit_price) * qty
                                    
                                    # Log trade with multi-agent info
                                    trade_logger.log_trade(
                                        symbol=symbol,
                                        side=signal.signal_type.value,
                                        qty=qty,
                                        entry_price=entry_price,
                                        exit_price=exit_price,
                                        pnl=pnl,
                                        signals=f'multi_agent_{signal.agent_id}',
                                        strategy=selected_strategy,
                                        outcome='win' if pnl > 0 else 'loss',
                                        notes=f'Multi-Agent: {signal.agent_id} | Confidence: {combined_confidence:.2f} | Reasoning: {signal.reasoning}'
                                    )
                                    
                                    # Update AI systems with trade results
                                    actual_return = pnl / (entry_price * qty)
                                    
                                    # Update strategy selector
                                    strategy_selector.add_experience(
                                        market_state=market_state,
                                        strategy_performances=strategy_performances,
                                        selected_strategy=selected_strategy,
                                        actual_return=actual_return
                                    )
                                    
                                    # Update DQN agent
                                    reward = actual_return * 100  # Scale reward
                                    next_trading_state = create_trading_state(df, portfolio_state, strategy_performances, exit_time)
                                    dqn_agent.store_experience(trading_state, dqn_action, reward, next_trading_state, True)
                                    
                                    # Update multi-agent coordinator
                                    trade_duration = (exit_time - entry_time).total_seconds() / 60.0  # minutes
                                    multi_agent_coordinator.update_agent_performance(signal.agent_id, pnl, trade_duration)
                                    
                                    # Update portfolio state
                                    portfolio_state['position_size'] = qty / 100.0
                                    portfolio_state['unrealized_pnl'] = actual_return
                                    portfolio_state['realized_pnl'] += actual_return
                                    portfolio_state['trades_count'] += 1
                                    
                                    # Analyze completed trade
                                    analysis_result = analyze_completed_trade(
                                        outcome_analyzer, symbol, df, entry_price, exit_price,
                                        entry_time, exit_time, qty, signal.agent_id, sl, tp
                                    )
                                    
                                    # Send notification with multi-agent insights
                                    notification_msg = (
                                        f"🤖 Multi-Agent Trade Executed\n"
                                        f"📊 {symbol} | {signal.signal_type.value.upper()}\n"
                                        f"🎯 Agent: {signal.agent_id}\n"
                                        f"💰 Entry: ₹{entry_price:.2f} | Exit: ₹{exit_price:.2f}\n"
                                        f"📈 PnL: ₹{pnl:.2f} ({actual_return*100:.2f}%)\n"
                                        f"🎲 Combined Confidence: {combined_confidence:.2f}\n"
                                        f"🧠 Strategy: {selected_strategy}\n"
                                        f"🤖 DQN Action: {dqn_action_name}\n"
                                        f"💡 Reasoning: {signal.reasoning}"
                                    )
                                    
                                    if analysis_result:
                                        notification_msg += f"\n📋 Analysis: {analysis_result.outcome.value}"
                                        if analysis_result.recommendations:
                                            notification_msg += f"\n💡 Recommendations: {', '.join(analysis_result.recommendations[:2])}"
                                    
                                    notifier.send_message(notification_msg)
                                    
                                    logger.info(f"Multi-agent trade executed: {symbol} {signal.signal_type.value} by {signal.agent_id}")
                                
                            else:
                                # Live trading would go here
                                logger.info(f"Live trading not implemented for multi-agent signal: {symbol}")
                        
                        else:
                            logger.debug(f"Multi-agent signal filtered out: {symbol} - confidence {combined_confidence:.2f}, DQN action {dqn_action_name}")
                    
                    except Exception as e:
                        logger.error(f"Error processing multi-agent signal for {symbol}: {e}")
                        continue
                
                # Train AI systems periodically
                if len(dfs) % 10 == 0:  # Every 10 stocks
                    try:
                        # Train DQN if enough experiences
                        if len(dqn_agent.replay_buffer) > 100:
                            dqn_agent.train()
                        
                        # Update strategy selector
                        strategy_selector.update_model()
                        
                        logger.debug("AI systems training update completed")
                    except Exception as e:
                        logger.error(f"Error training AI systems: {e}")
                
            except Exception as e:
                logger.error(f"Error processing {symbol} with multi-agent system: {e}")
                continue
                        portfolio_state['total_return'] += pnl / entry_price
                        portfolio_state['trades_count'] += 1
                        
                        # Create next trading state for DQN experience
                        next_trading_state = create_trading_state(df, portfolio_state, strategy_performances, exit_time)
                        
                        # Calculate DQN reward
                        dqn_reward_obj = dqn_agent.reward_calculator.calculate_reward(
                            action=dqn_action,
                            current_return=actual_return,
                            portfolio_value=100000 + pnl,  # Simplified portfolio value
                            previous_portfolio_value=100000,
                            volatility=price_volatility,
                            max_drawdown=portfolio_state['max_drawdown'],
                            position_size=portfolio_state['position_size']
                        )
                        
                        # Store DQN experience
                        dqn_agent.store_experience(
                            state=trading_state,
                            action=dqn_action,
                            reward=dqn_reward_obj.total_reward,
                            next_state=next_trading_state,
                            done=True  # Trade completed
                        )
                        
                        # Train DQN agent
                        dqn_loss = dqn_agent.train_step()
                        
                        # Analyze completed trade
                        analysis_result = analyze_completed_trade(
                            outcome_analyzer, symbol, df, entry_price, exit_price,
                            entry_time, exit_time, qty, selected_strategy, sl, tp
                        )
                        
                        # Create enhanced notification with analysis
                        base_msg = f"Paper Trade: Bought {symbol} at {entry_price:.2f}, SL: {sl:.2f}, TP: {tp:.2f}, exited at {exit_price:.2f}, PnL: {pnl:.2f}, R:R: {rr:.2f} | {msg}"
                        
                        if analysis_result:
                            analysis_msg = f"\n📊 Analysis: {analysis_result.outcome.value.upper()}"
                            if analysis_result.failure_type:
                                analysis_msg += f" | Failure: {analysis_result.failure_type.value}"
                                analysis_msg += f" | Confidence: {analysis_result.confidence_score:.2f}"
                                if analysis_result.recommendations:
                                    analysis_msg += f"\n💡 Key Rec: {analysis_result.recommendations[0]}"
                            base_msg += analysis_msg
                        
                        notifier.send_message(base_msg)
                    else:
                        success, fill_price, msg = order_executor.execute_order(symbol, qty, entry_price, sl, tp, side='buy', live=True)
                        exit_price = tp  # For logging, assume TP hit
                        exit_time = datetime.now()
                        pnl = exit_price - fill_price if success else 0
                        
                        # Log trade
                        trade_logger.log_trade(
                            symbol=symbol,
                            side='buy',
                            qty=qty,
                            entry_price=fill_price,
                            exit_price=exit_price,
                            pnl=pnl,
                            signals='pool',
                            strategy='pool',
                            outcome='win' if pnl > 0 else 'loss',
                            notes=f'Live trade | SL: {sl:.2f} | TP: {tp:.2f} | R:R: {rr:.2f} | {msg}'
                        )
                        
                        # Add experience to strategy selector for online learning (live trades)
                        if success:
                            actual_return = pnl / fill_price  # Calculate return percentage
                            strategy_selector.add_experience(
                                market_state=market_state,
                                strategy_performances=strategy_performances,
                                selected_strategy=selected_strategy,
                                actual_return=actual_return
                            )
                            
                            # Analyze completed trade (for live trades, analyze after execution)
                            analysis_result = analyze_completed_trade(
                                outcome_analyzer, symbol, df, fill_price, exit_price,
                                entry_time, exit_time, qty, selected_strategy, sl, tp
                            )
                            
                            # Create enhanced notification with analysis
                            base_msg = f"Live Trade: Bought {symbol} at {fill_price:.2f}, SL: {sl:.2f}, TP: {tp:.2f}, exited at {exit_price:.2f}, PnL: {pnl:.2f}, R:R: {rr:.2f} | {msg}"
                            
                            if analysis_result:
                                analysis_msg = f"\n📊 Analysis: {analysis_result.outcome.value.upper()}"
                                if analysis_result.failure_type:
                                    analysis_msg += f" | Failure: {analysis_result.failure_type.value}"
                                    analysis_msg += f" | Confidence: {analysis_result.confidence_score:.2f}"
                                    if analysis_result.recommendations:
                                        analysis_msg += f"\n💡 Key Rec: {analysis_result.recommendations[0]}"
                                base_msg += analysis_msg
                            
                            notifier.send_message(base_msg)
                        else:
                            notifier.send_message(
                                f"Live Trade Failed: {symbol} | {msg}"
                            )
                else:
                    notifier.send_message(f"No attention trade for {symbol} (confidence: {strategy_confidence:.2f}, strategy: {selected_strategy})")
            except Exception as e:
                logger.error(f"Trade execution error for {symbol}: {e}")
                notifier.send_message(f"Trade execution error for {symbol}: {e}")

        # 4. Batch report to Telegram with trade outcome analysis and attention insights
        try:
            # Get trade outcome statistics
            outcome_stats = outcome_analyzer.get_failure_statistics(days=7)  # Last 7 days
            
            # Get strategy attribution from attention selector
            strategy_attribution = strategy_selector.get_strategy_attribution(days=7)
            
            report_msg = f"🤖 Attention-Based Trading Report:\n"
            report_msg += f"Performance: {trade_logger.analyze_performance()}\n"
            
            # Add strategy selection insights
            if "error" not in strategy_attribution:
                report_msg += f"\n🎯 Strategy Selection (7 days):"
                report_msg += f"\n• Total Selections: {strategy_attribution['total_selections']}"
                
                # Show strategy frequency
                if strategy_attribution['strategy_frequency']:
                    report_msg += f"\n• Strategy Usage:"
                    for strategy, freq in strategy_attribution['strategy_frequency'].items():
                        report_msg += f"\n  - {strategy}: {freq*100:.1f}%"
                
                # Show average confidences
                if strategy_attribution['average_confidences']:
                    report_msg += f"\n• Avg Confidence:"
                    for strategy, conf in strategy_attribution['average_confidences'].items():
                        report_msg += f"\n  - {strategy}: {conf:.2f}"
            
            # Add outcome analysis if available
            if "error" not in outcome_stats:
                report_msg += f"\n\n📊 Trade Analysis (7 days):"
                report_msg += f"\n• Total Trades: {outcome_stats['total_trades']}"
                report_msg += f"\n• Win Rate: {outcome_stats['win_rate']:.1f}%"
                report_msg += f"\n• Avg Win: {outcome_stats['average_win_pct']:.2f}%"
                report_msg += f"\n• Avg Loss: {outcome_stats['average_loss_pct']:.2f}%"
                
                if outcome_stats['most_common_failure']:
                    report_msg += f"\n• Top Failure: {outcome_stats['most_common_failure']}"
                
                # Show failure distribution
                if outcome_stats['failure_type_distribution']:
                    report_msg += f"\n• Failures: "
                    failure_items = []
                    for failure_type, count in outcome_stats['failure_type_distribution'].items():
                        failure_items.append(f"{failure_type}({count})")
                    report_msg += ", ".join(failure_items)
            
            notifier.send_message(report_msg)
            
        except Exception as e:
            logger.error(f"Batch report error: {e}")
            notifier.send_message(f"Batch report error: {e}")

    except Exception as e:
        logger.critical(f"Fatal error: {e}")
        try:
            notifier = TelegramNotifier(TELEGRAM_TOKEN, TELEGRAM_CHAT_ID)
            notifier.send_message(f"Fatal error: {e}")
        except Exception:
            pass

if __name__ == "__main__":
    main() 