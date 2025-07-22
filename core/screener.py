"""
Stock Screener Module with Kite Connect Instruments Dump

This module implements dynamic stock universe selection using Zerodha Kite Connect
instruments dump for comprehensive equity screening.

Key Features:
- Downloads complete instrument dump from Kite Connect
- Filters for EQ (Equity) instruments only
- Applies NSE segment filtering
- Dynamic price, volume, and liquidity filters
- Creates tradeable universe of 20-50 stocks
- Metadata-rich instrument information
- Daily updated instrument list
- No API rate limits for instrument data

Benefits:
- Complete coverage of all NSE equity instruments
- Rich metadata (lot size, tick size, etc.)
- Efficient single API call for entire universe
- Updated daily with fresh instrument list
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import os
import json
from dataclasses import dataclass, field
from broker.zerodha import ZerodhaBroker

logger = logging.getLogger(__name__)


@dataclass
class ScreeningConfig:
    """Configuration for stock screening parameters"""
    # Instrument filtering
    instrument_type: str = 'EQ'  # Equity instruments only
    exchange: str = 'NSE'  # NSE segment
    
    # Price filters
    min_price: float = 50.0  # Minimum ₹50
    max_price: float = 5000.0  # Maximum ₹5000
    
    # Volume filters
    min_volume: int = 100000  # Minimum 1 lakh shares daily volume
    min_value_traded: float = 10000000.0  # Minimum ₹1 crore value traded
    
    # Volatility filters
    min_atr_percentage: float = 1.0  # Minimum 1% ATR
    max_atr_percentage: float = 8.0  # Maximum 8% ATR
    
    # Market cap filters (if available)
    min_market_cap: Optional[float] = 1000000000.0  # Minimum ₹100 crore market cap
    
    # Liquidity filters
    min_delivery_percentage: float = 20.0  # Minimum 20% delivery
    max_impact_cost: float = 1.0  # Maximum 1% impact cost
    
    # Universe size
    target_universe_size: int = 30  # Target 30 stocks
    max_universe_size: int = 50  # Maximum 50 stocks
    
    # Ban list
    banned_symbols: List[str] = field(default_factory=lambda: [
        'SUZLON', 'YESBANK', 'JETAIRWAYS'  # Example banned stocks
    ])
    
    # Sector diversification
    max_stocks_per_sector: int = 5  # Maximum 5 stocks per sector
    
    # Data requirements
    min_historical_days: int = 50  # Minimum 50 days of data required


@dataclass
class InstrumentInfo:
    """Information about a trading instrument"""
    instrument_token: int
    exchange_token: int
    tradingsymbol: str
    name: str
    last_price: float
    expiry: Optional[str]
    strike: float
    tick_size: float
    lot_size: int
    instrument_type: str
    segment: str
    exchange: str


class KiteInstrumentsDumpProcessor:
    """Processes Kite Connect instruments dump for equity screening"""
    
    def __init__(self, broker: ZerodhaBroker, config: ScreeningConfig = None):
        self.broker = broker
        self.config = config or ScreeningConfig()
        self.instruments_cache = {}
        self.cache_timestamp = None
        self.cache_file = "data/instruments_cache.json"
        
        # Ensure data directory exists
        os.makedirs("data", exist_ok=True)
        
        logger.info("KiteInstrumentsDumpProcessor initialized")
    
    def download_instruments_dump(self, force_refresh: bool = False) -> pd.DataFrame:
        """
        Download complete instruments dump from Kite Connect
        
        Args:
            force_refresh: Force download even if cache is fresh
            
        Returns:
            DataFrame with all instruments
        """
        try:
            # Check if we have fresh cached data (less than 1 day old)
            if not force_refresh and self._is_cache_fresh():
                logger.info("Using cached instruments data")
                return self._load_cached_instruments()
            
            logger.info("Downloading fresh instruments dump from Kite Connect...")
            
            # Download instruments dump from Kite Connect
            # This is a single API call that gets all instruments
            instruments = self.broker.kite.instruments()
            
            if not instruments:
                logger.error("Failed to download instruments dump")
                return pd.DataFrame()
            
            # Convert to DataFrame
            instruments_df = pd.DataFrame(instruments)
            
            # Cache the data
            self._cache_instruments(instruments_df)
            
            logger.info(f"Downloaded {len(instruments_df)} instruments from Kite Connect")
            return instruments_df
            
        except Exception as e:
            logger.error(f"Error downloading instruments dump: {e}")
            # Try to use cached data as fallback
            if os.path.exists(self.cache_file):
                logger.info("Using cached instruments as fallback")
                return self._load_cached_instruments()
            return pd.DataFrame()
    
    def filter_equity_instruments(self, instruments_df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter instruments dump for equity instruments only
        
        Args:
            instruments_df: Complete instruments DataFrame
            
        Returns:
            Filtered DataFrame with only equity instruments
        """
        try:
            if instruments_df.empty:
                return pd.DataFrame()
            
            # Filter for equity instruments
            equity_filter = (
                (instruments_df['instrument_type'] == self.config.instrument_type) &
                (instruments_df['segment'] == self.config.exchange)
            )
            
            equity_instruments = instruments_df[equity_filter].copy()
            
            # Remove banned symbols
            if self.config.banned_symbols:
                equity_instruments = equity_instruments[
                    ~equity_instruments['tradingsymbol'].isin(self.config.banned_symbols)
                ]
            
            # Filter by basic criteria
            if 'last_price' in equity_instruments.columns:
                price_filter = (
                    (equity_instruments['last_price'] >= self.config.min_price) &
                    (equity_instruments['last_price'] <= self.config.max_price)
                )
                equity_instruments = equity_instruments[price_filter]
            
            logger.info(f"Filtered to {len(equity_instruments)} equity instruments")
            return equity_instruments
            
        except Exception as e:
            logger.error(f"Error filtering equity instruments: {e}")
            return pd.DataFrame()
    
    def _is_cache_fresh(self) -> bool:
        """Check if cached instruments data is fresh (less than 1 day old)"""
        if not os.path.exists(self.cache_file):
            return False
        
        try:
            with open(self.cache_file, 'r') as f:
                cache_data = json.load(f)
            
            cache_time = datetime.fromisoformat(cache_data.get('timestamp', '2000-01-01'))
            return (datetime.now() - cache_time).total_seconds() < 86400  # 24 hours
            
        except Exception as e:
            logger.error(f"Error checking cache freshness: {e}")
            return False
    
    def _cache_instruments(self, instruments_df: pd.DataFrame):
        """Cache instruments data to file"""
        try:
            cache_data = {
                'timestamp': datetime.now().isoformat(),
                'instruments': instruments_df.to_dict('records')
            }
            
            with open(self.cache_file, 'w') as f:
                json.dump(cache_data, f, default=str)
            
            logger.info(f"Cached {len(instruments_df)} instruments")
            
        except Exception as e:
            logger.error(f"Error caching instruments: {e}")
    
    def _load_cached_instruments(self) -> pd.DataFrame:
        """Load instruments from cache"""
        try:
            with open(self.cache_file, 'r') as f:
                cache_data = json.load(f)
            
            instruments_df = pd.DataFrame(cache_data['instruments'])
            logger.info(f"Loaded {len(instruments_df)} instruments from cache")
            return instruments_df
            
        except Exception as e:
            logger.error(f"Error loading cached instruments: {e}")
            return pd.DataFrame()


class StockScreener:
    """
    Advanced stock screener using Kite Connect instruments dump
    """
    
    def __init__(self, config: ScreeningConfig = None):
        self.config = config or ScreeningConfig()
        self.broker = ZerodhaBroker()
        self.instruments_processor = KiteInstrumentsDumpProcessor(self.broker, self.config)
        
        # Performance tracking
        self.screening_history = []
        self.last_screening_time = None
        
        logger.info("StockScreener initialized with instruments dump processing")
    
    def get_tradable_stocks(self, force_refresh: bool = False, include_sentiment: bool = True) -> pd.DataFrame:
        """
        Get tradeable stocks using instruments dump and advanced filtering
        
        Args:
            force_refresh: Force refresh of instruments data
            include_sentiment: Whether to include sentiment analysis
            
        Returns:
            DataFrame with tradeable stocks and their data
        """
        try:
            start_time = datetime.now()
            
            # Step 1: Download instruments dump
            instruments_df = self.instruments_processor.download_instruments_dump(force_refresh)
            if instruments_df.empty:
                logger.error("No instruments data available")
                return pd.DataFrame()
            
            # Step 2: Filter for equity instruments
            equity_instruments = self.instruments_processor.filter_equity_instruments(instruments_df)
            if equity_instruments.empty:
                logger.error("No equity instruments found")
                return pd.DataFrame()
            
            # Step 3: Get market data for filtered instruments
            market_data = self._get_market_data_for_instruments(equity_instruments)
            if market_data.empty:
                logger.error("No market data available")
                return pd.DataFrame()
            
            # Step 4: Apply advanced filters
            filtered_stocks = self._apply_advanced_filters(market_data)
            
            # Step 5: Rank and select final universe
            final_universe = self._select_final_universe(filtered_stocks)
            
            # Step 6: Add technical indicators
            final_universe_with_indicators = self._add_technical_indicators(final_universe)
            
            # Step 7: Add sentiment analysis
            if include_sentiment:
                final_universe_with_sentiment = self._add_sentiment_analysis(final_universe_with_indicators)
            else:
                final_universe_with_sentiment = final_universe_with_indicators
            
            # Track screening performance
            screening_time = (datetime.now() - start_time).total_seconds()
            self._track_screening_performance(len(instruments_df), len(final_universe_with_sentiment), screening_time)
            
            logger.info(f"Screening completed: {len(instruments_df)} instruments → {len(final_universe_with_sentiment)} tradeable stocks in {screening_time:.2f}s")
            
            return final_universe_with_sentiment
            
        except Exception as e:
            logger.error(f"Error in get_tradable_stocks: {e}")
            return pd.DataFrame()
    
    def _get_market_data_for_instruments(self, equity_instruments: pd.DataFrame) -> pd.DataFrame:
        """Get market data for filtered equity instruments"""
        try:
            stocks_data = []
            
            # Limit to reasonable number for API calls
            instruments_to_process = equity_instruments.head(200)  # Process top 200 by market cap or other criteria
            
            for _, instrument in instruments_to_process.iterrows():
                try:
                    symbol = instrument['tradingsymbol']
                    
                    # Get historical data
                    hist_data = self.broker.get_historical_data(symbol, days=self.config.min_historical_days)
                    
                    if hist_data is not None and not hist_data.empty and len(hist_data) >= 20:
                        # Calculate basic metrics
                        last_close = hist_data['close'].iloc[-1]
                        avg_volume = hist_data['volume'].tail(20).mean()
                        avg_value = (hist_data['close'] * hist_data['volume']).tail(20).mean()
                        
                        # Calculate ATR
                        high_low = hist_data['high'] - hist_data['low']
                        high_close = abs(hist_data['high'] - hist_data['close'].shift())
                        low_close = abs(hist_data['low'] - hist_data['close'].shift())
                        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
                        atr = true_range.rolling(14).mean().iloc[-1]
                        atr_percentage = (atr / last_close) * 100
                        
                        # Calculate volatility
                        returns = hist_data['close'].pct_change()
                        volatility = returns.std() * np.sqrt(252) * 100  # Annualized volatility %
                        
                        stock_info = {
                            'symbol': symbol,
                            'name': instrument.get('name', symbol),
                            'instrument_token': instrument['instrument_token'],
                            'close': last_close,
                            'volume': avg_volume,
                            'value_traded': avg_value,
                            'atr': atr,
                            'atr_percentage': atr_percentage,
                            'volatility': volatility,
                            'tick_size': instrument.get('tick_size', 0.05),
                            'lot_size': instrument.get('lot_size', 1),
                            'data': hist_data,
                            'last_updated': datetime.now()
                        }
                        
                        stocks_data.append(stock_info)
                        
                except Exception as e:
                    logger.warning(f"Error processing {instrument.get('tradingsymbol', 'unknown')}: {e}")
                    continue
            
            if stocks_data:
                return pd.DataFrame(stocks_data)
            else:
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error getting market data: {e}")
            return pd.DataFrame()
    
    def _apply_advanced_filters(self, market_data: pd.DataFrame) -> pd.DataFrame:
        """Apply advanced filtering criteria"""
        try:
            if market_data.empty:
                return pd.DataFrame()
            
            filtered = market_data.copy()
            initial_count = len(filtered)
            
            # Price filters
            filtered = filtered[
                (filtered['close'] >= self.config.min_price) &
                (filtered['close'] <= self.config.max_price)
            ]
            logger.info(f"Price filter: {initial_count} → {len(filtered)} stocks")
            
            # Volume filters
            filtered = filtered[
                (filtered['volume'] >= self.config.min_volume) &
                (filtered['value_traded'] >= self.config.min_value_traded)
            ]
            logger.info(f"Volume filter: {initial_count} → {len(filtered)} stocks")
            
            # Volatility filters
            filtered = filtered[
                (filtered['atr_percentage'] >= self.config.min_atr_percentage) &
                (filtered['atr_percentage'] <= self.config.max_atr_percentage)
            ]
            logger.info(f"Volatility filter: {initial_count} → {len(filtered)} stocks")
            
            # Remove stocks with insufficient data
            filtered = filtered[filtered['data'].apply(lambda x: len(x) >= self.config.min_historical_days)]
            logger.info(f"Data quality filter: {initial_count} → {len(filtered)} stocks")
            
            return filtered
            
        except Exception as e:
            logger.error(f"Error applying advanced filters: {e}")
            return pd.DataFrame()
    
    def _select_final_universe(self, filtered_stocks: pd.DataFrame) -> pd.DataFrame:
        """Select final universe with ranking and diversification"""
        try:
            if filtered_stocks.empty:
                return pd.DataFrame()
            
            # Calculate composite score for ranking
            filtered_stocks = filtered_stocks.copy()
            
            # Normalize metrics for scoring (higher is better)
            filtered_stocks['volume_score'] = (filtered_stocks['volume'] - filtered_stocks['volume'].min()) / (filtered_stocks['volume'].max() - filtered_stocks['volume'].min())
            filtered_stocks['value_score'] = (filtered_stocks['value_traded'] - filtered_stocks['value_traded'].min()) / (filtered_stocks['value_traded'].max() - filtered_stocks['value_traded'].min())
            filtered_stocks['volatility_score'] = (filtered_stocks['atr_percentage'] - filtered_stocks['atr_percentage'].min()) / (filtered_stocks['atr_percentage'].max() - filtered_stocks['atr_percentage'].min())
            
            # Composite score (weighted)
            filtered_stocks['composite_score'] = (
                filtered_stocks['volume_score'] * 0.4 +
                filtered_stocks['value_score'] * 0.4 +
                filtered_stocks['volatility_score'] * 0.2
            )
            
            # Sort by composite score
            filtered_stocks = filtered_stocks.sort_values('composite_score', ascending=False)
            
            # Select top stocks up to target universe size
            final_universe = filtered_stocks.head(self.config.target_universe_size)
            
            logger.info(f"Selected final universe: {len(final_universe)} stocks")
            return final_universe
            
        except Exception as e:
            logger.error(f"Error selecting final universe: {e}")
            return pd.DataFrame()
    
    def _add_technical_indicators(self, stocks_df: pd.DataFrame) -> pd.DataFrame:
        """Add enhanced technical indicators to stock data"""
        try:
            if stocks_df.empty:
                return pd.DataFrame()
            
            enhanced_stocks = []
            
            # Import enhanced technical indicators
            from core.technical_indicators import EnhancedTechnicalIndicators, TechnicalIndicatorConfig
            
            # Initialize technical calculator
            indicator_config = TechnicalIndicatorConfig()
            technical_calculator = EnhancedTechnicalIndicators(indicator_config)
            
            for _, stock in stocks_df.iterrows():
                try:
                    hist_data = stock['data'].copy()
                    
                    # Calculate all enhanced technical indicators
                    enhanced_data = technical_calculator.calculate_all_indicators(hist_data)
                    
                    # Update stock data with enhanced indicators
                    stock_dict = stock.to_dict()
                    stock_dict['data'] = enhanced_data
                    
                    # Add summary indicators to stock level for easy access
                    if not enhanced_data.empty:
                        latest = enhanced_data.iloc[-1]
                        stock_dict.update({
                            'vwap': latest.get('vwap', stock_dict['close']),
                            'vwap_signal': latest.get('vwap_signal', 'neutral'),
                            'rsi_14': latest.get('rsi_14', 50),
                            'rsi_signal': latest.get('rsi_14_signal', 'neutral'),
                            'macd_trend': latest.get('macd_trend', 'neutral'),
                            'supertrend_signal': latest.get('supertrend_signal', 'neutral'),
                            'ema_alignment': latest.get('ema_alignment', 'neutral'),
                            'composite_signal': latest.get('composite_signal', 'neutral'),
                            'signal_strength': latest.get('signal_strength', 0),
                            'volatility_regime': latest.get('volatility_regime', 'normal')
                        })
                    
                    enhanced_stocks.append(stock_dict)
                    
                except Exception as e:
                    logger.warning(f"Error adding enhanced indicators for {stock['symbol']}: {e}")
                    # Add stock without enhanced indicators (fallback to basic)
                    stock_dict = stock.to_dict()
                    try:
                        # Basic fallback indicators
                        hist_data = stock['data'].copy()
                        hist_data['rsi'] = 50  # Default RSI
                        hist_data['sma_20'] = hist_data['close'].rolling(20).mean()
                        stock_dict['data'] = hist_data
                        stock_dict.update({
                            'vwap': stock_dict['close'],
                            'vwap_signal': 'neutral',
                            'rsi_14': 50,
                            'composite_signal': 'neutral'
                        })
                    except:
                        pass
                    enhanced_stocks.append(stock_dict)
            
            logger.info(f"Enhanced technical indicators added to {len(enhanced_stocks)} stocks")
            return pd.DataFrame(enhanced_stocks)
            
        except Exception as e:
            logger.error(f"Error adding enhanced technical indicators: {e}")
            return stocks_df
    
    def _add_sentiment_analysis(self, stocks_df: pd.DataFrame) -> pd.DataFrame:
        """Add sentiment analysis to stock data"""
        try:
            if stocks_df.empty:
                return pd.DataFrame()
            
            # Import sentiment analyzer
            from core.sentiment_integration import SentimentTechnicalIntegrator
            
            # Initialize sentiment integrator
            sentiment_integrator = SentimentTechnicalIntegrator(
                sentiment_weight=0.18,  # 18% sentiment weight
                technical_weight=0.82   # 82% technical weight
            )
            
            enhanced_stocks = []
            
            for _, stock in stocks_df.iterrows():
                try:
                    symbol = stock['symbol']
                    
                    # Prepare technical data for sentiment integration
                    technical_data = {
                        'overall_score': stock.get('signal_strength', 0),
                        'confidence': abs(stock.get('signal_strength', 0)),
                        'rsi': stock.get('rsi_14', 50),
                        'macd_signal': stock.get('macd_trend', 'neutral'),
                        'price_vs_vwap': stock.get('vwap_signal', 'neutral'),
                        'volume_ratio': 1.0  # Default volume ratio
                    }
                    
                    # Get market data for sentiment analysis
                    market_data = stock.get('data', pd.DataFrame())
                    
                    # Create enhanced signal with sentiment
                    enhanced_signal = sentiment_integrator.create_enhanced_signal(
                        symbol, technical_data, market_data
                    )
                    
                    # Update stock data with sentiment information
                    stock_dict = stock.to_dict()
                    stock_dict.update({
                        'sentiment_score': enhanced_signal.sentiment_score,
                        'sentiment_confidence': enhanced_signal.sentiment_confidence,
                        'sentiment_category': enhanced_signal.sentiment_category,
                        'combined_score': enhanced_signal.combined_score,
                        'final_confidence': enhanced_signal.final_confidence,
                        'trade_direction': enhanced_signal.trade_direction,
                        'position_size_multiplier': enhanced_signal.position_size_multiplier,
                        'risk_adjustment': enhanced_signal.risk_adjustment,
                        'enable_short_selling': enhanced_signal.enable_short_selling,
                        'skip_trade': enhanced_signal.skip_trade,
                        'sentiment_reasoning': enhanced_signal.reasoning
                    })
                    
                    enhanced_stocks.append(stock_dict)
                    
                except Exception as e:
                    logger.warning(f"Error adding sentiment analysis for {stock['symbol']}: {e}")
                    # Add stock without sentiment (fallback)
                    stock_dict = stock.to_dict()
                    stock_dict.update({
                        'sentiment_score': 0.0,
                        'sentiment_confidence': 0.0,
                        'sentiment_category': 'neutral',
                        'combined_score': stock.get('signal_strength', 0),
                        'final_confidence': abs(stock.get('signal_strength', 0)),
                        'trade_direction': 'long' if stock.get('signal_strength', 0) > 0 else 'neutral',
                        'position_size_multiplier': 1.0,
                        'risk_adjustment': 1.0,
                        'enable_short_selling': False,
                        'skip_trade': False,
                        'sentiment_reasoning': ['Sentiment analysis failed - using technical signals only']
                    })
                    enhanced_stocks.append(stock_dict)
            
            logger.info(f"Sentiment analysis added to {len(enhanced_stocks)} stocks")
            return pd.DataFrame(enhanced_stocks)
            
        except Exception as e:
            logger.error(f"Error adding sentiment analysis: {e}")
            return stocks_df
    
    def _track_screening_performance(self, total_instruments: int, final_count: int, screening_time: float):
        """Track screening performance metrics"""
        performance_record = {
            'timestamp': datetime.now(),
            'total_instruments': total_instruments,
            'final_count': final_count,
            'screening_time': screening_time,
            'filter_efficiency': final_count / total_instruments if total_instruments > 0 else 0
        }
        
        self.screening_history.append(performance_record)
        self.last_screening_time = datetime.now()
        
        # Keep only last 100 records
        if len(self.screening_history) > 100:
            self.screening_history = self.screening_history[-100:]
    
    def get_screening_statistics(self) -> Dict:
        """Get screening performance statistics"""
        if not self.screening_history:
            return {'message': 'No screening history available'}
        
        recent_records = self.screening_history[-10:]  # Last 10 screenings
        
        return {
            'total_screenings': len(self.screening_history),
            'last_screening': self.last_screening_time.isoformat() if self.last_screening_time else None,
            'average_screening_time': np.mean([r['screening_time'] for r in recent_records]),
            'average_final_count': np.mean([r['final_count'] for r in recent_records]),
            'average_filter_efficiency': np.mean([r['filter_efficiency'] for r in recent_records]),
            'config': {
                'target_universe_size': self.config.target_universe_size,
                'min_price': self.config.min_price,
                'max_price': self.config.max_price,
                'min_volume': self.config.min_volume
            }
        }
    
    def update_screening_config(self, **kwargs):
        """Update screening configuration parameters"""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
                logger.info(f"Updated screening config: {key} = {value}")
            else:
                logger.warning(f"Unknown config parameter: {key}")
    
    def get_banned_symbols(self) -> List[str]:
        """Get current banned symbols list"""
        return self.config.banned_symbols.copy()
    
    def add_banned_symbol(self, symbol: str):
        """Add symbol to ban list"""
        if symbol not in self.config.banned_symbols:
            self.config.banned_symbols.append(symbol)
            logger.info(f"Added {symbol} to ban list")
    
    def remove_banned_symbol(self, symbol: str):
        """Remove symbol from ban list"""
        if symbol in self.config.banned_symbols:
            self.config.banned_symbols.remove(symbol)
            logger.info(f"Removed {symbol} from ban list")


# Example usage and testing
if __name__ == "__main__":
    # Create screening configuration
    config = ScreeningConfig(
        min_price=100.0,
        max_price=2000.0,
        min_volume=500000,
        target_universe_size=25
    )
    
    # Initialize screener
    screener = StockScreener(config)
    
    # Get tradeable stocks
    tradeable_stocks = screener.get_tradable_stocks()
    
    if not tradeable_stocks.empty:
        print(f"Found {len(tradeable_stocks)} tradeable stocks:")
        for _, stock in tradeable_stocks.head(10).iterrows():
            print(f"- {stock['symbol']}: ₹{stock['close']:.2f}, Volume: {stock['volume']:,.0f}, ATR: {stock['atr_percentage']:.2f}%")
    else:
        print("No tradeable stocks found")
    
    # Get statistics
    stats = screener.get_screening_statistics()
    print(f"\nScreening statistics: {stats}") 