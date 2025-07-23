"""
WebSocket Manager for Real-Time Stock Price Streaming

This module implements WebSocket connection to Zerodha Kite Connect for real-time
price streaming of active positions only. It handles connection management,
auto-reconnection, and data validation.

Key Features:
- Real-time price streaming for portfolio positions only
- Automatic reconnection with exponential backoff
- Data validation and error handling
- Efficient subscription management
- Thread-safe operations
"""

import asyncio
import websocket
import threading
import time
import json
import logging
from typing import Dict, List, Callable, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict, deque
import struct
from enum import Enum

logger = logging.getLogger(__name__)


class SubscriptionMode(Enum):
    """WebSocket subscription modes"""
    LTP = "ltp"          # Last Traded Price
    QUOTE = "quote"      # Bid, Ask, Volume, OHLC
    FULL = "full"        # Complete market depth


@dataclass
class TickData:
    """Real-time tick data structure"""
    instrument_token: int
    symbol: str
    last_price: float
    change: float
    change_percent: float
    volume: int
    timestamp: datetime
    bid_price: float = 0.0
    ask_price: float = 0.0
    bid_quantity: int = 0
    ask_quantity: int = 0
    open_price: float = 0.0
    high_price: float = 0.0
    low_price: float = 0.0
    close_price: float = 0.0


@dataclass
class WebSocketConfig:
    """WebSocket configuration"""
    api_key: str
    access_token: str
    reconnect_attempts: int = 10
    reconnect_delay: float = 1.0
    max_reconnect_delay: float = 60.0
    ping_interval: int = 30
    ping_timeout: int = 10
    subscription_mode: SubscriptionMode = SubscriptionMode.LTP
    max_subscriptions: int = 100


class WebSocketManager:
    """
    Manages WebSocket connection to Zerodha Kite Connect for real-time data
    """
    
    def __init__(self, config: WebSocketConfig):
        self.config = config
        self.ws = None
        self.is_connected = False
        self.is_running = False
        
        # Subscription management
        self.subscribed_tokens = set()
        self.token_symbol_map = {}  # token -> symbol mapping
        self.symbol_token_map = {}  # symbol -> token mapping
        
        # Data callbacks
        self.tick_callbacks: List[Callable[[TickData], None]] = []
        self.connection_callbacks: List[Callable[[bool], None]] = []
        self.error_callbacks: List[Callable[[str], None]] = []
        
        # Connection management
        self.reconnect_count = 0
        self.last_ping_time = 0
        self.connection_thread = None
        self.lock = threading.RLock()
        
        # Data processing
        self.tick_buffer = deque(maxlen=1000)
        self.last_tick_time = {}
        
        logger.info("WebSocketManager initialized")
    
    def add_tick_callback(self, callback: Callable[[TickData], None]):
        """Add callback for tick data"""
        self.tick_callbacks.append(callback)
    
    def add_connection_callback(self, callback: Callable[[bool], None]):
        """Add callback for connection status changes"""
        self.connection_callbacks.append(callback)
    
    def add_error_callback(self, callback: Callable[[str], None]):
        """Add callback for errors"""
        self.error_callbacks.append(callback)
    
    def start(self):
        """Start WebSocket connection"""
        if self.is_running:
            logger.warning("WebSocket already running")
            return
        
        self.is_running = True
        self.connection_thread = threading.Thread(target=self._connection_loop, daemon=True)
        self.connection_thread.start()
        logger.info("WebSocket connection started")
    
    def stop(self):
        """Stop WebSocket connection"""
        self.is_running = False
        if self.ws:
            self.ws.close()
        
        if self.connection_thread and self.connection_thread.is_alive():
            self.connection_thread.join(timeout=5)
        
        logger.info("WebSocket connection stopped")
    
    def subscribe(self, instruments: Dict[int, str]):
        """
        Subscribe to instruments for real-time data
        
        Args:
            instruments: Dict of {instrument_token: symbol}
        """
        with self.lock:
            if len(instruments) > self.config.max_subscriptions:
                logger.warning(f"Too many subscriptions requested: {len(instruments)}")
                # Take only the first max_subscriptions
                instruments = dict(list(instruments.items())[:self.config.max_subscriptions])
            
            # Update mappings
            self.token_symbol_map.update(instruments)
            self.symbol_token_map.update({v: k for k, v in instruments.items()})
            
            # Add to subscription set
            new_tokens = set(instruments.keys()) - self.subscribed_tokens
            self.subscribed_tokens.update(new_tokens)
            
            # Send subscription if connected
            if self.is_connected and new_tokens:
                self._send_subscription(list(new_tokens))
            
            logger.info(f"Subscribed to {len(new_tokens)} new instruments, total: {len(self.subscribed_tokens)}")
    
    def unsubscribe(self, symbols: List[str]):
        """Unsubscribe from symbols"""
        with self.lock:
            tokens_to_remove = []
            for symbol in symbols:
                if symbol in self.symbol_token_map:
                    token = self.symbol_token_map[symbol]
                    tokens_to_remove.append(token)
                    
                    # Remove from mappings
                    del self.symbol_token_map[symbol]
                    if token in self.token_symbol_map:
                        del self.token_symbol_map[token]
            
            # Remove from subscription set
            self.subscribed_tokens -= set(tokens_to_remove)
            
            # Send unsubscription if connected
            if self.is_connected and tokens_to_remove:
                self._send_unsubscription(tokens_to_remove)
            
            logger.info(f"Unsubscribed from {len(tokens_to_remove)} instruments")
    
    def get_subscribed_symbols(self) -> List[str]:
        """Get list of currently subscribed symbols"""
        return list(self.symbol_token_map.keys())
    
    def _connection_loop(self):
        """Main connection loop with reconnection logic"""
        while self.is_running:
            try:
                self._connect()
                
                # Keep connection alive
                while self.is_running and self.is_connected:
                    time.sleep(1)
                    self._check_connection_health()
                
            except Exception as e:
                logger.error(f"Connection error: {e}")
                self._notify_error(str(e))
                
                if self.is_running:
                    self._handle_reconnection()
            
            if not self.is_running:
                break
    
    def _connect(self):
        """Establish WebSocket connection"""
        # Construct WebSocket URL
        ws_url = f"wss://ws.kite.trade/?api_key={self.config.api_key}&access_token={self.config.access_token}"
        
        logger.info("Connecting to Kite WebSocket...")
        
        # Create WebSocket connection
        self.ws = websocket.WebSocketApp(
            ws_url,
            on_open=self._on_open,
            on_message=self._on_message,
            on_error=self._on_error,
            on_close=self._on_close
        )
        
        # Run WebSocket
        self.ws.run_forever(
            ping_interval=self.config.ping_interval,
            ping_timeout=self.config.ping_timeout
        )
    
    def _on_open(self, ws):
        """WebSocket connection opened"""
        logger.info("WebSocket connection established")
        self.is_connected = True
        self.reconnect_count = 0
        self.last_ping_time = time.time()
        
        # Subscribe to instruments if any
        if self.subscribed_tokens:
            self._send_subscription(list(self.subscribed_tokens))
        
        # Notify connection callbacks
        self._notify_connection(True)
    
    def _on_message(self, ws, message):
        """Process incoming WebSocket message"""
        try:
            # Parse binary message from Kite WebSocket
            tick_data = self._parse_binary_message(message)
            
            if tick_data:
                # Add to buffer
                self.tick_buffer.append(tick_data)
                self.last_tick_time[tick_data.instrument_token] = time.time()
                
                # Notify callbacks
                for callback in self.tick_callbacks:
                    try:
                        callback(tick_data)
                    except Exception as e:
                        logger.error(f"Error in tick callback: {e}")
                        
        except Exception as e:
            logger.error(f"Error processing message: {e}")
    
    def _on_error(self, ws, error):
        """WebSocket error occurred"""
        logger.error(f"WebSocket error: {error}")
        self._notify_error(str(error))
    
    def _on_close(self, ws, close_status_code, close_msg):
        """WebSocket connection closed"""
        logger.info(f"WebSocket connection closed: {close_status_code} - {close_msg}")
        self.is_connected = False
        self._notify_connection(False)
    
    def _send_subscription(self, tokens: List[int]):
        """Send subscription message"""
        if not self.is_connected or not self.ws:
            return
        
        try:
            # Kite WebSocket subscription format
            subscription_message = {
                "a": "subscribe",
                "v": tokens
            }
            
            # Set mode based on configuration
            if self.config.subscription_mode == SubscriptionMode.QUOTE:
                subscription_message["a"] = "mode"
                subscription_message["v"] = [self.config.subscription_mode.value, tokens]
            elif self.config.subscription_mode == SubscriptionMode.FULL:
                subscription_message["a"] = "mode"
                subscription_message["v"] = [self.config.subscription_mode.value, tokens]
            
            self.ws.send(json.dumps(subscription_message))
            logger.info(f"Sent subscription for {len(tokens)} tokens")
            
        except Exception as e:
            logger.error(f"Error sending subscription: {e}")
    
    def _send_unsubscription(self, tokens: List[int]):
        """Send unsubscription message"""
        if not self.is_connected or not self.ws:
            return
        
        try:
            unsubscription_message = {
                "a": "unsubscribe",
                "v": tokens
            }
            
            self.ws.send(json.dumps(unsubscription_message))
            logger.info(f"Sent unsubscription for {len(tokens)} tokens")
            
        except Exception as e:
            logger.error(f"Error sending unsubscription: {e}")
    
    def _parse_binary_message(self, message: bytes) -> Optional[TickData]:
        """Parse binary message from Kite WebSocket"""
        try:
            if len(message) < 4:
                return None
            
            # Read number of packets
            num_packets = struct.unpack(">H", message[:2])[0]
            
            if num_packets == 0:
                return None
            
            # Parse first packet (LTP mode)
            offset = 2
            packet_length = struct.unpack(">H", message[offset:offset+2])[0]
            offset += 2
            
            if packet_length < 8:
                return None
            
            # Parse instrument token
            instrument_token = struct.unpack(">I", message[offset:offset+4])[0]
            offset += 4
            
            # Get symbol from mapping
            symbol = self.token_symbol_map.get(instrument_token, f"TOKEN_{instrument_token}")
            
            # Parse LTP
            last_price = struct.unpack(">I", message[offset:offset+4])[0] / 100.0
            offset += 4
            
            # Create tick data
            tick_data = TickData(
                instrument_token=instrument_token,
                symbol=symbol,
                last_price=last_price,
                change=0.0,  # Will be calculated if needed
                change_percent=0.0,  # Will be calculated if needed
                volume=0,  # Not available in LTP mode
                timestamp=datetime.now()
            )
            
            # Parse additional data if available (QUOTE/FULL mode)
            if packet_length > 8 and len(message) > offset:
                try:
                    # Parse volume if available
                    if len(message) >= offset + 4:
                        tick_data.volume = struct.unpack(">I", message[offset:offset+4])[0]
                        offset += 4
                    
                    # Parse OHLC if available
                    if len(message) >= offset + 16:
                        tick_data.open_price = struct.unpack(">I", message[offset:offset+4])[0] / 100.0
                        tick_data.high_price = struct.unpack(">I", message[offset+4:offset+8])[0] / 100.0
                        tick_data.low_price = struct.unpack(">I", message[offset+8:offset+12])[0] / 100.0
                        tick_data.close_price = struct.unpack(">I", message[offset+12:offset+16])[0] / 100.0
                        
                except struct.error:
                    # Ignore parsing errors for additional data
                    pass
            
            return tick_data
            
        except Exception as e:
            logger.error(f"Error parsing binary message: {e}")
            return None
    
    def _check_connection_health(self):
        """Check connection health and send ping if needed"""
        current_time = time.time()
        
        # Check if we've received any data recently
        if self.last_tick_time:
            last_data_time = max(self.last_tick_time.values())
            if current_time - last_data_time > 60:  # No data for 60 seconds
                logger.warning("No data received for 60 seconds, connection may be stale")
        
        # Send ping if needed
        if current_time - self.last_ping_time > self.config.ping_interval:
            try:
                if self.ws:
                    self.ws.ping()
                    self.last_ping_time = current_time
            except Exception as e:
                logger.error(f"Error sending ping: {e}")
    
    def _handle_reconnection(self):
        """Handle reconnection with exponential backoff"""
        if self.reconnect_count >= self.config.reconnect_attempts:
            logger.error("Maximum reconnection attempts reached, stopping")
            self.is_running = False
            return
        
        self.reconnect_count += 1
        delay = min(
            self.config.reconnect_delay * (2 ** (self.reconnect_count - 1)),
            self.config.max_reconnect_delay
        )
        
        logger.info(f"Reconnecting in {delay:.1f} seconds (attempt {self.reconnect_count})")
        time.sleep(delay)
    
    def _notify_connection(self, connected: bool):
        """Notify connection status callbacks"""
        for callback in self.connection_callbacks:
            try:
                callback(connected)
            except Exception as e:
                logger.error(f"Error in connection callback: {e}")
    
    def _notify_error(self, error_message: str):
        """Notify error callbacks"""
        for callback in self.error_callbacks:
            try:
                callback(error_message)
            except Exception as e:
                logger.error(f"Error in error callback: {e}")
    
    def get_connection_stats(self) -> Dict[str, Any]:
        """Get connection statistics"""
        return {
            "is_connected": self.is_connected,
            "is_running": self.is_running,
            "subscribed_instruments": len(self.subscribed_tokens),
            "reconnect_count": self.reconnect_count,
            "buffer_size": len(self.tick_buffer),
            "last_tick_times": dict(self.last_tick_time),
            "symbols": list(self.symbol_token_map.keys())
        }


# Example usage and testing
if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    
    # Configuration
    config = WebSocketConfig(
        api_key=os.getenv('KITE_API_KEY'),
        access_token=os.getenv('KITE_ACCESS_TOKEN'),
        subscription_mode=SubscriptionMode.LTP
    )
    
    # Create WebSocket manager
    ws_manager = WebSocketManager(config)
    
    # Add callbacks
    def on_tick(tick_data: TickData):
        print(f"Tick: {tick_data.symbol} - ₹{tick_data.last_price:.2f}")
    
    def on_connection(connected: bool):
        print(f"Connection: {'Connected' if connected else 'Disconnected'}")
    
    def on_error(error: str):
        print(f"Error: {error}")
    
    ws_manager.add_tick_callback(on_tick)
    ws_manager.add_connection_callback(on_connection)
    ws_manager.add_error_callback(on_error)
    
    # Subscribe to some instruments (example tokens)
    instruments = {
        738561: "RELIANCE",  # Example token
        408065: "TCS",       # Example token
    }
    
    ws_manager.subscribe(instruments)
    
    # Start WebSocket
    ws_manager.start()
    
    try:
        # Keep running
        while True:
            time.sleep(1)
            stats = ws_manager.get_connection_stats()
            if stats["is_connected"]:
                print(f"Connected - Buffer: {stats['buffer_size']} ticks")
    except KeyboardInterrupt:
        print("Stopping WebSocket...")
        ws_manager.stop()