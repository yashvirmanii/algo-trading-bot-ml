"""
Position Monitor - Real-Time Position Tracking

This module integrates WebSocket real-time data with trailing stop-loss management
to provide comprehensive position monitoring and automatic trade execution.

Key Features:
- Real-time position tracking using WebSocket data
- Integration with trailing stop-loss manager
- Automatic order execution when stops are hit
- Portfolio-level risk monitoring
- Performance analytics and reporting
"""

import logging
import threading
import time
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict
import json
import os

from core.websocket_manager import WebSocketManager, WebSocketConfig, TickData, SubscriptionMode
from core.trailing_stop_manager import (
    TrailingStopManager, TrailingConfig, TrailingStopEvent, 
    Position, PositionSide, TrailingType
)

logger = logging.getLogger(__name__)


@dataclass
class PositionMonitorConfig:
    """Configuration for position monitor"""
    # WebSocket configuration
    websocket_config: WebSocketConfig
    
    # Trailing stop configuration
    trailing_config: TrailingConfig
    
    # Monitoring settings
    position_update_interval: float = 1.0  # seconds
    portfolio_update_interval: float = 5.0  # seconds
    
    # Risk management
    max_portfolio_loss_percent: float = 5.0  # Stop all trading if portfolio down 5%
    max_position_loss_percent: float = 10.0  # Individual position max loss
    
    # Execution settings
    auto_execute_stops: bool = True
    execution_delay_seconds: float = 0.5  # Delay before executing stops
    
    # Logging and storage
    log_all_ticks: bool = False
    save_position_history: bool = True
    history_file: str = "data/position_history.json"


@dataclass
class PortfolioMetrics:
    """Portfolio-level metrics"""
    total_positions: int = 0
    total_invested: float = 0.0
    total_current_value: float = 0.0
    total_unrealized_pnl: float = 0.0
    total_unrealized_pnl_percent: float = 0.0
    
    # Risk metrics
    largest_loss: float = 0.0
    largest_gain: float = 0.0
    positions_in_profit: int = 0
    positions_in_loss: int = 0
    
    # Trailing metrics
    total_trails_active: int = 0
    total_trail_updates_today: int = 0
    
    last_updated: datetime = field(default_factory=datetime.now)


class PositionMonitor:
    """
    Comprehensive position monitoring system with real-time WebSocket integration
    """
    
    def __init__(self, config: PositionMonitorConfig):
        self.config = config
        
        # Initialize WebSocket manager
        self.ws_manager = WebSocketManager(config.websocket_config)
        
        # Initialize trailing stop manager
        self.trail_manager = TrailingStopManager(config.trailing_config)
        
        # Position tracking
        self.active_positions: Dict[str, Dict] = {}
        self.position_history: List[Dict] = []
        
        # Portfolio metrics
        self.portfolio_metrics = PortfolioMetrics()
        
        # Event callbacks
        self.position_callbacks: List[Callable[[str, Dict], None]] = []
        self.portfolio_callbacks: List[Callable[[PortfolioMetrics], None]] = []
        self.execution_callbacks: List[Callable[[str, str, float], None]] = []
        
        # Monitoring state
        self.is_monitoring = False
        self.monitoring_thread = None
        
        # Statistics
        self.stats = {
            'ticks_processed': 0,
            'trails_updated': 0,
            'stops_executed': 0,
            'start_time': datetime.now()
        }
        
        # Setup callbacks
        self._setup_callbacks()
        
        logger.info("PositionMonitor initialized")
    
    def _setup_callbacks(self):
        """Setup callbacks between components"""
        # WebSocket tick data callback
        self.ws_manager.add_tick_callback(self._on_tick_data)
        
        # WebSocket connection callback
        self.ws_manager.add_connection_callback(self._on_connection_change)
        
        # Trailing stop callbacks
        self.trail_manager.add_stop_loss_callback(self._on_stop_loss_hit)
        self.trail_manager.add_trail_update_callback(self._on_trail_update)
    
    def add_position(self, symbol: str, side: str, quantity: int, entry_price: float,
                    stop_loss: float, take_profit: float = None, 
                    instrument_token: int = None) -> bool:
        """
        Add a new position to monitor
        
        Args:
            symbol: Stock symbol
            side: 'long' or 'short'
            quantity: Number of shares
            entry_price: Entry price
            stop_loss: Initial stop-loss level
            take_profit: Optional take-profit level
            instrument_token: Zerodha instrument token for WebSocket
            
        Returns:
            True if position added successfully
        """
        try:
            # Convert side to enum
            position_side = PositionSide.LONG if side.lower() == 'long' else PositionSide.SHORT
            
            # Add to trailing stop manager
            success = self.trail_manager.add_position(
                symbol=symbol,
                side=position_side,
                quantity=quantity,
                entry_price=entry_price,
                initial_stop_loss=stop_loss,
                take_profit=take_profit
            )
            
            if not success:
                return False
            
            # Store position details
            position_data = {
                'symbol': symbol,
                'side': side,
                'quantity': quantity,
                'entry_price': entry_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'instrument_token': instrument_token,
                'entry_time': datetime.now(),
                'status': 'active'
            }
            
            self.active_positions[symbol] = position_data
            
            # Subscribe to WebSocket if token provided
            if instrument_token:
                self.ws_manager.subscribe({instrument_token: symbol})
            
            # Update portfolio metrics
            self._update_portfolio_metrics()
            
            logger.info(f"Added position: {symbol} {side} {quantity}@₹{entry_price:.2f}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding position {symbol}: {e}")
            return False
    
    def remove_position(self, symbol: str, reason: str = "manual") -> bool:
        """Remove position from monitoring"""
        try:
            # Remove from trailing stop manager
            self.trail_manager.remove_position(symbol)
            
            # Remove from active positions
            if symbol in self.active_positions:
                position_data = self.active_positions[symbol].copy()
                position_data['exit_time'] = datetime.now()
                position_data['exit_reason'] = reason
                position_data['status'] = 'closed'
                
                # Add to history
                self.position_history.append(position_data)
                
                # Remove from active
                del self.active_positions[symbol]
                
                # Unsubscribe from WebSocket
                self.ws_manager.unsubscribe([symbol])
                
                # Update portfolio metrics
                self._update_portfolio_metrics()
                
                logger.info(f"Removed position: {symbol} (reason: {reason})")
                return True
            else:
                logger.warning(f"Position not found: {symbol}")
                return False
                
        except Exception as e:
            logger.error(f"Error removing position {symbol}: {e}")
            return False
    
    def start_monitoring(self):
        """Start real-time monitoring"""
        if self.is_monitoring:
            logger.warning("Position monitoring already started")
            return
        
        try:
            # Start WebSocket connection
            self.ws_manager.start()
            
            # Start trailing stop monitoring
            self.trail_manager.start_monitoring()
            
            # Start position monitoring thread
            self.is_monitoring = True
            self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
            self.monitoring_thread.start()
            
            logger.info("Position monitoring started")
            
        except Exception as e:
            logger.error(f"Error starting position monitoring: {e}")
            self.stop_monitoring()
    
    def stop_monitoring(self):
        """Stop real-time monitoring"""
        try:
            self.is_monitoring = False
            
            # Stop WebSocket
            self.ws_manager.stop()
            
            # Stop trailing stop monitoring
            self.trail_manager.stop_monitoring()
            
            # Stop monitoring thread
            if self.monitoring_thread and self.monitoring_thread.is_alive():
                self.monitoring_thread.join(timeout=5)
            
            # Save position history
            if self.config.save_position_history:
                self._save_position_history()
            
            logger.info("Position monitoring stopped")
            
        except Exception as e:
            logger.error(f"Error stopping position monitoring: {e}")
    
    def _on_tick_data(self, tick_data: TickData):
        """Handle incoming tick data from WebSocket"""
        try:
            self.stats['ticks_processed'] += 1
            
            # Log tick if enabled
            if self.config.log_all_ticks:
                logger.debug(f"Tick: {tick_data.symbol} ₹{tick_data.last_price:.2f}")
            
            # Update trailing stop manager
            self.trail_manager.process_tick_data(tick_data)
            
            # Update position data
            if tick_data.symbol in self.active_positions:
                self.active_positions[tick_data.symbol]['current_price'] = tick_data.last_price
                self.active_positions[tick_data.symbol]['last_update'] = datetime.now()
            
        except Exception as e:
            logger.error(f"Error processing tick data: {e}")
    
    def _on_connection_change(self, connected: bool):
        """Handle WebSocket connection changes"""
        if connected:
            logger.info("WebSocket connected - real-time monitoring active")
        else:
            logger.warning("WebSocket disconnected - monitoring degraded")
    
    def _on_stop_loss_hit(self, event: TrailingStopEvent):
        """Handle stop-loss hit events"""
        try:
            logger.warning(f"STOP LOSS HIT: {event.symbol} at ₹{event.current_price:.2f}")
            
            # Execute stop-loss order if auto-execution enabled
            if self.config.auto_execute_stops:
                self._execute_stop_order(event)
            
            # Update statistics
            self.stats['stops_executed'] += 1
            
            # Notify callbacks
            for callback in self.execution_callbacks:
                try:
                    callback(event.symbol, 'stop_loss', event.current_price)
                except Exception as e:
                    logger.error(f"Error in execution callback: {e}")
                    
        except Exception as e:
            logger.error(f"Error handling stop-loss event: {e}")
    
    def _on_trail_update(self, event: TrailingStopEvent):
        """Handle trailing stop update events"""
        try:
            logger.info(f"Trail updated: {event.symbol} SL ₹{event.old_stop_loss:.2f} → ₹{event.new_stop_loss:.2f}")
            
            # Update statistics
            self.stats['trails_updated'] += 1
            
            # Update position data
            if event.symbol in self.active_positions:
                self.active_positions[event.symbol]['stop_loss'] = event.new_stop_loss
                self.active_positions[event.symbol]['trail_count'] = self.active_positions[event.symbol].get('trail_count', 0) + 1
                
        except Exception as e:
            logger.error(f"Error handling trail update: {e}")
    
    def _execute_stop_order(self, event: TrailingStopEvent):
        """Execute stop-loss order"""
        try:
            # Add execution delay to avoid whipsaws
            if self.config.execution_delay_seconds > 0:
                time.sleep(self.config.execution_delay_seconds)
            
            # Here you would integrate with your order execution system
            # For now, we'll just log and remove the position
            logger.info(f"Executing stop order for {event.symbol} at ₹{event.current_price:.2f}")
            
            # Remove position from monitoring
            self.remove_position(event.symbol, f"stop_loss_at_{event.current_price:.2f}")
            
        except Exception as e:
            logger.error(f"Error executing stop order for {event.symbol}: {e}")
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        last_portfolio_update = 0
        
        while self.is_monitoring:
            try:
                current_time = time.time()
                
                # Update portfolio metrics periodically
                if current_time - last_portfolio_update > self.config.portfolio_update_interval:
                    self._update_portfolio_metrics()
                    self._check_portfolio_risk()
                    last_portfolio_update = current_time
                
                # Sleep for position update interval
                time.sleep(self.config.position_update_interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(1)
    
    def _update_portfolio_metrics(self):
        """Update portfolio-level metrics"""
        try:
            metrics = PortfolioMetrics()
            
            # Get position statuses from trailing manager
            for symbol in self.active_positions:
                position_status = self.trail_manager.get_position_status(symbol)
                
                if 'error' not in position_status:
                    metrics.total_positions += 1
                    
                    # Calculate invested amount
                    invested = position_status['entry_price'] * position_status['quantity']
                    current_value = position_status['current_price'] * position_status['quantity']
                    
                    metrics.total_invested += invested
                    metrics.total_current_value += current_value
                    metrics.total_unrealized_pnl += position_status['unrealized_pnl']
                    
                    # Track largest gains/losses
                    pnl = position_status['unrealized_pnl']
                    if pnl > 0:
                        metrics.positions_in_profit += 1
                        metrics.largest_gain = max(metrics.largest_gain, pnl)
                    elif pnl < 0:
                        metrics.positions_in_loss += 1
                        metrics.largest_loss = min(metrics.largest_loss, pnl)
                    
                    # Count active trails
                    if position_status.get('trail_count', 0) > 0:
                        metrics.total_trails_active += 1
            
            # Calculate portfolio percentage
            if metrics.total_invested > 0:
                metrics.total_unrealized_pnl_percent = (metrics.total_unrealized_pnl / metrics.total_invested) * 100
            
            metrics.total_trail_updates_today = self.stats['trails_updated']
            metrics.last_updated = datetime.now()
            
            self.portfolio_metrics = metrics
            
            # Notify portfolio callbacks
            for callback in self.portfolio_callbacks:
                try:
                    callback(metrics)
                except Exception as e:
                    logger.error(f"Error in portfolio callback: {e}")
                    
        except Exception as e:
            logger.error(f"Error updating portfolio metrics: {e}")
    
    def _check_portfolio_risk(self):
        """Check portfolio-level risk limits"""
        try:
            # Check maximum portfolio loss
            if (self.portfolio_metrics.total_unrealized_pnl_percent < 
                -self.config.max_portfolio_loss_percent):
                
                logger.critical(f"Portfolio loss limit exceeded: {self.portfolio_metrics.total_unrealized_pnl_percent:.2f}%")
                # Here you could implement emergency stop-all functionality
            
            # Check individual position losses
            for symbol in self.active_positions:
                position_status = self.trail_manager.get_position_status(symbol)
                
                if ('error' not in position_status and 
                    position_status['unrealized_pnl_percent'] < -self.config.max_position_loss_percent):
                    
                    logger.warning(f"Position {symbol} loss limit exceeded: {position_status['unrealized_pnl_percent']:.2f}%")
                    
        except Exception as e:
            logger.error(f"Error checking portfolio risk: {e}")
    
    def _save_position_history(self):
        """Save position history to file"""
        try:
            os.makedirs(os.path.dirname(self.config.history_file), exist_ok=True)
            
            history_data = {
                'positions': self.position_history,
                'stats': self.stats,
                'saved_at': datetime.now().isoformat()
            }
            
            with open(self.config.history_file, 'w') as f:
                json.dump(history_data, f, indent=2, default=str)
                
            logger.info(f"Saved position history: {len(self.position_history)} positions")
            
        except Exception as e:
            logger.error(f"Error saving position history: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive status"""
        return {
            'monitoring_active': self.is_monitoring,
            'websocket_connected': self.ws_manager.is_connected,
            'active_positions': len(self.active_positions),
            'portfolio_metrics': {
                'total_positions': self.portfolio_metrics.total_positions,
                'total_invested': self.portfolio_metrics.total_invested,
                'total_pnl': self.portfolio_metrics.total_unrealized_pnl,
                'total_pnl_percent': self.portfolio_metrics.total_unrealized_pnl_percent,
                'positions_in_profit': self.portfolio_metrics.positions_in_profit,
                'positions_in_loss': self.portfolio_metrics.positions_in_loss
            },
            'statistics': self.stats,
            'websocket_stats': self.ws_manager.get_connection_stats()
        }
    
    def get_position_details(self, symbol: str = None) -> Dict[str, Any]:
        """Get detailed position information"""
        if symbol:
            if symbol in self.active_positions:
                position_data = self.active_positions[symbol].copy()
                trail_status = self.trail_manager.get_position_status(symbol)
                
                if 'error' not in trail_status:
                    position_data.update(trail_status)
                
                return position_data
            else:
                return {'error': f'Position {symbol} not found'}
        else:
            # Return all positions
            all_positions = {}
            for sym in self.active_positions:
                all_positions[sym] = self.get_position_details(sym)
            
            return {
                'positions': all_positions,
                'portfolio_metrics': self.portfolio_metrics.__dict__
            }
    
    # Callback registration methods
    def add_position_callback(self, callback: Callable[[str, Dict], None]):
        """Add callback for position updates"""
        self.position_callbacks.append(callback)
    
    def add_portfolio_callback(self, callback: Callable[[PortfolioMetrics], None]):
        """Add callback for portfolio updates"""
        self.portfolio_callbacks.append(callback)
    
    def add_execution_callback(self, callback: Callable[[str, str, float], None]):
        """Add callback for trade executions"""
        self.execution_callbacks.append(callback)


# Example usage and testing
if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    
    # Configuration
    ws_config = WebSocketConfig(
        api_key=os.getenv('KITE_API_KEY'),
        access_token=os.getenv('KITE_ACCESS_TOKEN'),
        subscription_mode=SubscriptionMode.LTP
    )
    
    trail_config = TrailingConfig(
        trailing_type=TrailingType.PERCENTAGE,
        trail_percentage=2.0,
        min_profit_to_trail=1.0
    )
    
    monitor_config = PositionMonitorConfig(
        websocket_config=ws_config,
        trailing_config=trail_config,
        auto_execute_stops=False  # Set to False for testing
    )
    
    # Create position monitor
    monitor = PositionMonitor(monitor_config)
    
    # Add callbacks
    def on_portfolio_update(metrics: PortfolioMetrics):
        print(f"Portfolio: {metrics.total_positions} positions, P&L: ₹{metrics.total_unrealized_pnl:.2f}")
    
    def on_execution(symbol: str, order_type: str, price: float):
        print(f"EXECUTION: {symbol} {order_type} at ₹{price:.2f}")
    
    monitor.add_portfolio_callback(on_portfolio_update)
    monitor.add_execution_callback(on_execution)
    
    # Add test positions
    monitor.add_position(
        symbol="RELIANCE",
        side="long",
        quantity=100,
        entry_price=2500.0,
        stop_loss=2450.0,
        take_profit=2600.0,
        instrument_token=738561  # Example token
    )
    
    # Start monitoring
    monitor.start_monitoring()
    
    try:
        # Keep running
        while True:
            time.sleep(5)
            status = monitor.get_status()
            print(f"Status: {status['active_positions']} positions, WebSocket: {status['websocket_connected']}")
            
    except KeyboardInterrupt:
        print("Stopping position monitor...")
        monitor.stop_monitoring()