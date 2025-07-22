"""
Zerodha Broker Integration with Enhanced Technical Indicators

This module provides a wrapper around the Zerodha Kite Connect API for:
- Authentication and secure credential management
- Fetching the full instrument list and stock universe
- Downloading historical OHLCV data for any symbol
- Enhanced technical indicator calculations (VWAP, RSI, MACD, SuperTrend, EMAs)
- Order placement and management (limit, market, SL/TP)

All API keys and tokens are loaded securely from environment variables. This module is the single point of contact for all broker-related operations in the trading bot.
"""

import os
import pandas as pd
import logging
from kiteconnect import KiteConnect
from dotenv import load_dotenv
from datetime import datetime, timedelta
from core.technical_indicators import EnhancedTechnicalIndicators, TechnicalIndicatorConfig

logger = logging.getLogger(__name__)

class ZerodhaBroker:
    def __init__(self, api_key=None, api_secret=None, access_token=None):
        load_dotenv()
        self.api_key = api_key or os.getenv('KITE_API_KEY')
        self.api_secret = api_secret or os.getenv('KITE_API_SECRET')
        self.access_token = access_token or os.getenv('KITE_ACCESS_TOKEN')
        self.kite = None
        self.authenticate()
        self.instruments = self.kite.instruments('NSE')
        
        # Initialize enhanced technical indicators
        self.indicator_config = TechnicalIndicatorConfig()
        self.technical_calculator = EnhancedTechnicalIndicators(self.indicator_config)
        
        logger.info("ZerodhaBroker initialized with enhanced technical indicators")

    def authenticate(self):
        self.kite = KiteConnect(api_key=self.api_key)
        self.kite.set_access_token(self.access_token)

    def get_stock_universe(self):
        nifty100 = [row['tradingsymbol'] for row in self.instruments if row.get('index') == 'NIFTY 100' or row.get('name') == 'NIFTY 100']
        if not nifty100:
            nifty100 = [row['tradingsymbol'] for row in self.instruments[:50]]
        return nifty100

    def get_historical_data(self, symbol, interval='day', days=7):
        token = None
        for row in self.instruments:
            if row['tradingsymbol'] == symbol:
                token = row['instrument_token']
                break
        if not token:
            raise Exception(f"Symbol {symbol} not found in instruments")
        to_date = datetime.now()
        from_date = to_date - timedelta(days=days)
        data = self.kite.historical_data(token, from_date, to_date, interval)
        df = pd.DataFrame(data)
        if not df.empty:
            df.rename(columns={'date': 'datetime'}, inplace=True)
            df.set_index('datetime', inplace=True)
        return df
    
    def get_enhanced_historical_data(self, symbol, interval='day', days=7, include_indicators=True):
        """
        Get historical data with enhanced technical indicators
        
        Args:
            symbol: Stock symbol
            interval: Data interval ('minute', '5minute', '15minute', 'day')
            days: Number of days of historical data
            include_indicators: Whether to calculate technical indicators
            
        Returns:
            DataFrame with OHLCV data and technical indicators
        """
        try:
            # Get basic historical data
            df = self.get_historical_data(symbol, interval, days)
            
            if df.empty:
                logger.warning(f"No historical data available for {symbol}")
                return df
            
            # Add technical indicators if requested
            if include_indicators:
                df = self.technical_calculator.calculate_all_indicators(df)
                logger.debug(f"Enhanced technical indicators calculated for {symbol}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error getting enhanced historical data for {symbol}: {e}")
            return pd.DataFrame()
    
    def get_agent_signals(self, symbol, agent_type, interval='15minute', days=30):
        """
        Get agent-specific trading signals based on technical indicators
        
        Args:
            symbol: Stock symbol
            agent_type: Type of agent ('breakout', 'trend', 'scalping', 'volatility')
            interval: Data interval for analysis
            days: Number of days of historical data
            
        Returns:
            Dictionary with agent-specific signals and analysis
        """
        try:
            # Get enhanced historical data
            df = self.get_enhanced_historical_data(symbol, interval, days, include_indicators=True)
            
            if df.empty:
                return {
                    'symbol': symbol,
                    'agent_type': agent_type,
                    'signal': 'neutral',
                    'confidence': 0.0,
                    'reasoning': ['No data available'],
                    'timestamp': datetime.now()
                }
            
            # Get agent-specific signals
            signals = self.technical_calculator.get_agent_specific_signals(df, agent_type)
            
            # Add metadata
            signals.update({
                'symbol': symbol,
                'agent_type': agent_type,
                'timestamp': datetime.now(),
                'data_points': len(df),
                'latest_price': df['close'].iloc[-1] if 'close' in df.columns else 0
            })
            
            return signals
            
        except Exception as e:
            logger.error(f"Error getting agent signals for {symbol} ({agent_type}): {e}")
            return {
                'symbol': symbol,
                'agent_type': agent_type,
                'signal': 'neutral',
                'confidence': 0.0,
                'reasoning': [f'Error: {str(e)}'],
                'timestamp': datetime.now()
            }
    
    def get_multi_agent_analysis(self, symbol, interval='15minute', days=30):
        """
        Get comprehensive multi-agent analysis for a symbol
        
        Args:
            symbol: Stock symbol
            interval: Data interval for analysis
            days: Number of days of historical data
            
        Returns:
            Dictionary with analysis from all agent types
        """
        try:
            agent_types = ['breakout', 'trend', 'scalping', 'volatility']
            analysis = {
                'symbol': symbol,
                'timestamp': datetime.now(),
                'agents': {}
            }
            
            # Get enhanced data once for all agents
            df = self.get_enhanced_historical_data(symbol, interval, days, include_indicators=True)
            
            if df.empty:
                logger.warning(f"No data available for multi-agent analysis of {symbol}")
                return analysis
            
            # Analyze with each agent type
            for agent_type in agent_types:
                signals = self.technical_calculator.get_agent_specific_signals(df, agent_type)
                analysis['agents'][agent_type] = signals
            
            # Calculate consensus
            bullish_count = sum(1 for agent in analysis['agents'].values() if agent['signal'] == 'bullish')
            bearish_count = sum(1 for agent in analysis['agents'].values() if agent['signal'] == 'bearish')
            
            if bullish_count > bearish_count:
                consensus = 'bullish'
                consensus_strength = bullish_count / len(agent_types)
            elif bearish_count > bullish_count:
                consensus = 'bearish'
                consensus_strength = bearish_count / len(agent_types)
            else:
                consensus = 'neutral'
                consensus_strength = 0.5
            
            analysis['consensus'] = {
                'signal': consensus,
                'strength': consensus_strength,
                'bullish_agents': bullish_count,
                'bearish_agents': bearish_count,
                'neutral_agents': len(agent_types) - bullish_count - bearish_count
            }
            
            # Add current market data
            latest = df.iloc[-1]
            analysis['market_data'] = {
                'current_price': latest['close'],
                'vwap': latest.get('vwap', latest['close']),
                'rsi_14': latest.get('rsi_14', 50),
                'macd_signal': latest.get('macd_trend', 'neutral'),
                'supertrend_signal': latest.get('supertrend_signal', 'neutral'),
                'volatility_regime': latest.get('volatility_regime', 'normal'),
                'composite_signal': latest.get('composite_signal', 'neutral')
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error in multi-agent analysis for {symbol}: {e}")
            return {
                'symbol': symbol,
                'timestamp': datetime.now(),
                'error': str(e),
                'agents': {},
                'consensus': {'signal': 'neutral', 'strength': 0.0}
            }
    
    def get_technical_summary(self, symbol, interval='day', days=50):
        """
        Get technical analysis summary for a symbol
        
        Args:
            symbol: Stock symbol
            interval: Data interval
            days: Number of days of data
            
        Returns:
            Dictionary with technical analysis summary
        """
        try:
            df = self.get_enhanced_historical_data(symbol, interval, days, include_indicators=True)
            
            if df.empty:
                return {'symbol': symbol, 'error': 'No data available'}
            
            latest = df.iloc[-1]
            
            summary = {
                'symbol': symbol,
                'timestamp': datetime.now(),
                'price_data': {
                    'current_price': latest['close'],
                    'change_pct': ((latest['close'] / df['close'].iloc[-2]) - 1) * 100 if len(df) > 1 else 0,
                    'high_52w': df['high'].max(),
                    'low_52w': df['low'].min(),
                    'volume': latest.get('volume', 0)
                },
                'technical_indicators': {
                    'vwap': {
                        'value': latest.get('vwap', latest['close']),
                        'signal': latest.get('vwap_signal', 'neutral'),
                        'price_vs_vwap': latest.get('price_vs_vwap', 0)
                    },
                    'rsi_14': {
                        'value': latest.get('rsi_14', 50),
                        'signal': latest.get('rsi_14_signal', 'neutral')
                    },
                    'macd': {
                        'line': latest.get('macd_line', 0),
                        'signal': latest.get('macd_signal', 0),
                        'histogram': latest.get('macd_histogram', 0),
                        'trend': latest.get('macd_trend', 'neutral')
                    },
                    'supertrend': {
                        'value': latest.get('supertrend', latest['close']),
                        'signal': latest.get('supertrend_signal', 'neutral'),
                        'price_vs_supertrend': latest.get('price_vs_supertrend', 0)
                    },
                    'emas': {
                        'ema_9': latest.get('ema_9', latest['close']),
                        'ema_21': latest.get('ema_21', latest['close']),
                        'ema_50': latest.get('ema_50', latest['close']),
                        'alignment': latest.get('ema_alignment', 'neutral')
                    }
                },
                'signals': {
                    'composite_signal': latest.get('composite_signal', 'neutral'),
                    'signal_strength': latest.get('signal_strength', 0),
                    'bullish_signals': latest.get('bullish_signals', 0),
                    'bearish_signals': latest.get('bearish_signals', 0)
                },
                'volatility': {
                    'atr': latest.get('atr', 0),
                    'atr_percent': latest.get('atr_percent', 0),
                    'regime': latest.get('volatility_regime', 'normal')
                }
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting technical summary for {symbol}: {e}")
            return {'symbol': symbol, 'error': str(e)}

    def place_order(self, symbol, qty, order_type, price=None):
        # Place a limit or market order using Kite Connect API
        params = {
            'tradingsymbol': symbol,
            'exchange': self.kite.EXCHANGE_NSE,
            'transaction_type': self.kite.TRANSACTION_TYPE_BUY,
            'quantity': qty,
            'order_type': self.kite.ORDER_TYPE_MARKET if order_type == 'MARKET' else self.kite.ORDER_TYPE_LIMIT,
            'product': self.kite.PRODUCT_MIS,
            'variety': self.kite.VARIETY_REGULAR,
            'validity': self.kite.VALIDITY_DAY
        }
        if order_type == 'LIMIT' and price is not None:
            params['price'] = price
        order_id = self.kite.place_order(**params)
        return order_id

    def check_order_status(self, order_id):
        # Check order status using Kite Connect API
        orders = self.kite.orders()
        for order in orders:
            if order['order_id'] == order_id:
                return order['status']
        return 'UNKNOWN'

    def cancel_order(self, order_id):
        # Cancel order using Kite Connect API
        self.kite.cancel_order(variety=self.kite.VARIETY_REGULAR, order_id=order_id)

    def get_order_fill_price(self, order_id):
        # Get the average fill price for an order
        orders = self.kite.orders()
        for order in orders:
            if order['order_id'] == order_id:
                return order.get('average_price', None)
        return None

    def place_sl_tp_orders(self, symbol, qty, sl, tp, side='buy'):
        # Place SL and TP orders (bracket order logic)
        # Note: Bracket orders are not available for all products; fallback to separate SL/TP orders
        # Place SL order
        sl_order_id = self.kite.place_order(
            tradingsymbol=symbol,
            exchange=self.kite.EXCHANGE_NSE,
            transaction_type=self.kite.TRANSACTION_TYPE_SELL if side == 'buy' else self.kite.TRANSACTION_TYPE_BUY,
            quantity=qty,
            order_type=self.kite.ORDER_TYPE_SLM,
            price=None,
            trigger_price=sl,
            product=self.kite.PRODUCT_MIS,
            variety=self.kite.VARIETY_REGULAR,
            validity=self.kite.VALIDITY_DAY
        )
        # Place TP order (limit)
        tp_order_id = self.kite.place_order(
            tradingsymbol=symbol,
            exchange=self.kite.EXCHANGE_NSE,
            transaction_type=self.kite.TRANSACTION_TYPE_SELL if side == 'buy' else self.kite.TRANSACTION_TYPE_BUY,
            quantity=qty,
            order_type=self.kite.ORDER_TYPE_LIMIT,
            price=tp,
            product=self.kite.PRODUCT_MIS,
            variety=self.kite.VARIETY_REGULAR,
            validity=self.kite.VALIDITY_DAY
        )
        return sl_order_id, tp_order_id

    def get_portfolio(self):
        # TODO: Implement portfolio fetch logic
        pass 