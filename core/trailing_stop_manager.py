"""
Trailing Stop-Loss Manager

This module implements intelligent trailing stop-loss functionality that dynamically
adjusts stop-loss and take-profit levels as prices move favorably. It supports
multiple trailing algorithms and integrates with real-time WebSocket data.

Key Features:
- Dynamic SL/TP adjustment based on price movement
- Multiple trailing algorithms (fixed, percentage, ATR-based)
- Minimum profit lock-in before trailing begins
- Maximum trail distance to prevent over-trailing
- Volatility-adjusted trailing for different market conditions
- Real-time monitoring and execution
"""

import logging
import threading
import time
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import numpy as np
import pandas as pd
from collections import defaultdict, deque

from core.websocket_manager import TickData

logger = logging.getLogger(__name__)


class TrailingType(Enum):
    """Types of trailing stop-loss algorithms"""
    FIXED_AMOUNT = "fixed_amount"      # Trail by fixed rupee amount
    PERCENTAGE = "percentage"          # Trail by percentage
    ATR_BASED = "atr_based"           # Trail based on Average True Range
    HYBRID = "hybrid"                  # Combination of methods


class PositionSide(Enum):
    """Position side"""
    LONG = "long"    # Buy first, sell later
    SHORT = "short"  # Sell first, buy later


@dataclass
class TrailingConfig:
    """Configuration for trailing stop-loss"""
    # Trailing algorithm
    trailing_type: TrailingType = TrailingType.PERCENTAGE
    
    # Fixed amount trailing (in rupees)
    fixed_trail_amount: float = 5.0
    
    # Percentage trailing
    trail_percentage: float = 2.0  # 2% trailing
    
    # ATR-based trailing
    atr_multiplier: float = 2.0    # 2x ATR
    atr_period: int = 14           # ATR calculation period
    
    # Minimum profit before trailing starts
    min_profit_to_trail: float = 1.0  # Minimum 1% profit
    
    # Maximum trail distance (safety limit)
    max_trail_distance: float = 10.0  # Maximum 10% trail
    
    # Update frequency
    update_frequency_seconds: float = 1.0  # Update every second
    
    # Volatility adjustment
    volatility_adjustment: bool = True
    high_volatility_multiplier: float = 1.5
    low_volatility_multiplier: float = 0.7
@da
taclass
class Position:
    """Represents a trading position"""
    symbol: str
    side: PositionSide
    quantity: int
    entry_price: float
    current_price: float
    entry_time: datetime
    
    # Stop-loss and take-profit levels
    stop_loss: float
    take_profit: Optional[float] = None
    
    # Trailing parameters
    trailing_enabled: bool = True
    highest_price: float = 0.0  # For long positions
    lowest_price: float = float('inf')  # For short positions
    
    # Profit tracking
    unrealized_pnl: float = 0.0
    unrealized_pnl_percent: float = 0.0
    
    # Trailing history
    trail_history: List[Dict] = field(default_factory=list)
    
    # Metadata
    last_updated: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """Initialize calculated fields"""
        if self.side == PositionSide.LONG:
            self.highest_price = max(self.entry_price, self.current_price)
        else:
            self.lowest_price = min(self.entry_price, self.current_price)
        
        self.update_pnl()
    
    def update_price(self, new_price: float):
        """Update current price and recalculate P&L"""
        self.current_price = new_price
        
        if self.side == PositionSide.LONG:
            self.highest_price = max(self.highest_price, new_price)
        else:
            self.lowest_price = min(self.lowest_price, new_price)
        
        self.update_pnl()
        self.last_updated = datetime.now()
    
    def update_pnl(self):
        """Update unrealized P&L"""
        if self.side == PositionSide.LONG:
            self.unrealized_pnl = (self.current_price - self.entry_price) * self.quantity
            self.unrealized_pnl_percent = ((self.current_price / self.entry_price) - 1) * 100
        else:
            self.unrealized_pnl = (self.entry_price - self.current_price) * self.quantity
            self.unrealized_pnl_percent = ((self.entry_price / self.current_price) - 1) * 100
    
    def is_profitable(self) -> bool:
        """Check if position is currently profitable"""
        return self.unrealized_pnl > 0
    
    def get_profit_percent(self) -> float:
        """Get current profit percentage"""
        return self.unrealized_pnl_percent


@dataclass
class TrailingStopEvent:
    """Event triggered when trailing stop conditions are met"""
    symbol: str
    event_type: str  # 'stop_loss_hit', 'take_profit_hit', 'trail_updated'
    old_stop_loss: float
    new_stop_loss: float
    current_price: float
    profit_percent: float
    timestamp: datetime = field(default_factory=datetime.now)


class TrailingStopManager:
    """
    Manages trailing stop-loss for active positions with real-time price updates
    """
    
    def __init__(self, config: TrailingConfig = None):
        self.config = config or TrailingConfig()
        
        # Position tracking
        self.positions: Dict[str, Position] = {}
        self.lock = threading.RLock()
        
        # Price history for ATR calculation
        self.price_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        
        # Event callbacks
        self.stop_loss_callbacks: List[Callable[[TrailingStopEvent], None]] = []
        self.trail_update_callbacks: List[Callable[[TrailingStopEvent], None]] = []
        
        # Monitoring thread
        self.monitoring_thread = None
        self.is_monitoring = False
        
        logger.info("TrailingStopManager initialized")
    
    def add_position(self, symbol: str, side: PositionSide, quantity: int, 
                    entry_price: float, initial_stop_loss: float, 
                    take_profit: float = None) -> bool:
        """
        Add a new position to track
        
        Args:
            symbol: Stock symbol
            side: Position side (LONG/SHORT)
            quantity: Number of shares
            entry_price: Entry price
            initial_stop_loss: Initial stop-loss level
            take_profit: Optional take-profit level
            
        Returns:
            True if position added successfully
        """
        try:
            with self.lock:
                position = Position(
                    symbol=symbol,
                    side=side,
                    quantity=quantity,
                    entry_price=entry_price,
                    current_price=entry_price,
                    entry_time=datetime.now(),
                    stop_loss=initial_stop_loss,
                    take_profit=take_profit
                )
                
                self.positions[symbol] = position
                
                # Initialize price history
                self.price_history[symbol].append({
                    'price': entry_price,
                    'timestamp': datetime.now()
                })
                
                logger.info(f"Added position: {symbol} {side.value} {quantity}@₹{entry_price:.2f}")
                return True
                
        except Exception as e:
            logger.error(f"Error adding position {symbol}: {e}")
            return False
    
    def remove_position(self, symbol: str) -> bool:
        """Remove position from tracking"""
        try:
            with self.lock:
                if symbol in self.positions:
                    del self.positions[symbol]
                    if symbol in self.price_history:
                        del self.price_history[symbol]
                    logger.info(f"Removed position: {symbol}")
                    return True
                else:
                    logger.warning(f"Position not found: {symbol}")
                    return False
                    
        except Exception as e:
            logger.error(f"Error removing position {symbol}: {e}")
            return False
    
    def update_price(self, symbol: str, new_price: float) -> Optional[TrailingStopEvent]:
        """
        Update price for a position and check trailing conditions
        
        Args:
            symbol: Stock symbol
            new_price: New price
            
        Returns:
            TrailingStopEvent if any action is triggered
        """
        try:
            with self.lock:
                if symbol not in self.positions:
                    return None
                
                position = self.positions[symbol]
                old_stop_loss = position.stop_loss
                
                # Update position price
                position.update_price(new_price)
                
                # Add to price history
                self.price_history[symbol].append({
                    'price': new_price,
                    'timestamp': datetime.now()
                })
                
                # Check if stop-loss or take-profit is hit
                if self._is_stop_loss_hit(position):
                    event = TrailingStopEvent(
                        symbol=symbol,
                        event_type='stop_loss_hit',
                        old_stop_loss=old_stop_loss,
                        new_stop_loss=position.stop_loss,
                        current_price=new_price,
                        profit_percent=position.get_profit_percent()
                    )
                    self._notify_stop_loss(event)
                    return event
                
                if position.take_profit and self._is_take_profit_hit(position):
                    event = TrailingStopEvent(
                        symbol=symbol,
                        event_type='take_profit_hit',
                        old_stop_loss=old_stop_loss,
                        new_stop_loss=position.stop_loss,
                        current_price=new_price,
                        profit_percent=position.get_profit_percent()
                    )
                    self._notify_stop_loss(event)
                    return event
                
                # Update trailing stop if conditions are met
                if self._should_update_trailing_stop(position):
                    new_stop_loss = self._calculate_new_stop_loss(position)
                    
                    if new_stop_loss != position.stop_loss:
                        # Record trail history
                        trail_record = {
                            'timestamp': datetime.now(),
                            'price': new_price,
                            'old_stop_loss': position.stop_loss,
                            'new_stop_loss': new_stop_loss,
                            'profit_percent': position.get_profit_percent()
                        }
                        position.trail_history.append(trail_record)
                        
                        # Update stop-loss
                        position.stop_loss = new_stop_loss
                        
                        event = TrailingStopEvent(
                            symbol=symbol,
                            event_type='trail_updated',
                            old_stop_loss=old_stop_loss,
                            new_stop_loss=new_stop_loss,
                            current_price=new_price,
                            profit_percent=position.get_profit_percent()
                        )
                        
                        self._notify_trail_update(event)
                        logger.info(f"Trail updated {symbol}: SL ₹{old_stop_loss:.2f} → ₹{new_stop_loss:.2f}")
                        return event
                
                return None
                
        except Exception as e:
            logger.error(f"Error updating price for {symbol}: {e}")
            return None
    
    def process_tick_data(self, tick_data: TickData):
        """Process real-time tick data from WebSocket"""
        if tick_data.symbol in self.positions:
            self.update_price(tick_data.symbol, tick_data.last_price)
    
    def _should_update_trailing_stop(self, position: Position) -> bool:
        """Check if trailing stop should be updated"""
        if not position.trailing_enabled:
            return False
        
        # Check minimum profit requirement
        if position.get_profit_percent() < self.config.min_profit_to_trail:
            return False
        
        return True
    
    def _calculate_new_stop_loss(self, position: Position) -> float:
        """Calculate new stop-loss level based on trailing algorithm"""
        try:
            if self.config.trailing_type == TrailingType.FIXED_AMOUNT:
                return self._calculate_fixed_amount_trail(position)
            elif self.config.trailing_type == TrailingType.PERCENTAGE:
                return self._calculate_percentage_trail(position)
            elif self.config.trailing_type == TrailingType.ATR_BASED:
                return self._calculate_atr_trail(position)
            elif self.config.trailing_type == TrailingType.HYBRID:
                return self._calculate_hybrid_trail(position)
            else:
                return position.stop_loss
                
        except Exception as e:
            logger.error(f"Error calculating new stop-loss for {position.symbol}: {e}")
            return position.stop_loss
    
    def _calculate_fixed_amount_trail(self, position: Position) -> float:
        """Calculate trailing stop using fixed amount"""
        trail_amount = self.config.fixed_trail_amount
        
        # Adjust for volatility if enabled
        if self.config.volatility_adjustment:
            volatility_multiplier = self._get_volatility_multiplier(position.symbol)
            trail_amount *= volatility_multiplier
        
        if position.side == PositionSide.LONG:
            # For long positions, trail stop-loss up as price goes up
            new_stop_loss = position.highest_price - trail_amount
            return max(new_stop_loss, position.stop_loss)  # Never lower the stop-loss
        else:
            # For short positions, trail stop-loss down as price goes down
            new_stop_loss = position.lowest_price + trail_amount
            return min(new_stop_loss, position.stop_loss)  # Never raise the stop-loss
    
    def _calculate_percentage_trail(self, position: Position) -> float:
        """Calculate trailing stop using percentage"""
        trail_percent = self.config.trail_percentage / 100.0
        
        # Adjust for volatility if enabled
        if self.config.volatility_adjustment:
            volatility_multiplier = self._get_volatility_multiplier(position.symbol)
            trail_percent *= volatility_multiplier
        
        if position.side == PositionSide.LONG:
            # For long positions
            new_stop_loss = position.highest_price * (1 - trail_percent)
            return max(new_stop_loss, position.stop_loss)
        else:
            # For short positions
            new_stop_loss = position.lowest_price * (1 + trail_percent)
            return min(new_stop_loss, position.stop_loss)
    
    def _calculate_atr_trail(self, position: Position) -> float:
        """Calculate trailing stop using ATR (Average True Range)"""
        atr = self._calculate_atr(position.symbol)
        if atr == 0:
            # Fallback to percentage method if ATR calculation fails
            return self._calculate_percentage_trail(position)
        
        trail_distance = atr * self.config.atr_multiplier
        
        # Adjust for volatility if enabled
        if self.config.volatility_adjustment:
            volatility_multiplier = self._get_volatility_multiplier(position.symbol)
            trail_distance *= volatility_multiplier
        
        if position.side == PositionSide.LONG:
            new_stop_loss = position.highest_price - trail_distance
            return max(new_stop_loss, position.stop_loss)
        else:
            new_stop_loss = position.lowest_price + trail_distance
            return min(new_stop_loss, position.stop_loss)
    
    def _calculate_hybrid_trail(self, position: Position) -> float:
        """Calculate trailing stop using hybrid approach"""
        # Combine percentage and ATR methods
        percentage_stop = self._calculate_percentage_trail(position)
        atr_stop = self._calculate_atr_trail(position)
        
        if position.side == PositionSide.LONG:
            # Use the higher of the two (more conservative)
            return max(percentage_stop, atr_stop)
        else:
            # Use the lower of the two (more conservative)
            return min(percentage_stop, atr_stop)
    
    def _calculate_atr(self, symbol: str) -> float:
        """Calculate Average True Range for a symbol"""
        try:
            price_data = list(self.price_history[symbol])
            if len(price_data) < self.config.atr_period:
                return 0.0
            
            # Simple ATR calculation using price data
            # Note: This is simplified - in production, you'd want OHLC data
            prices = [p['price'] for p in price_data[-self.config.atr_period:]]
            
            if len(prices) < 2:
                return 0.0
            
            # Calculate true ranges (simplified using only close prices)
            true_ranges = []
            for i in range(1, len(prices)):
                tr = abs(prices[i] - prices[i-1])
                true_ranges.append(tr)
            
            # Return average true range
            return np.mean(true_ranges) if true_ranges else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating ATR for {symbol}: {e}")
            return 0.0
    
    def _get_volatility_multiplier(self, symbol: str) -> float:
        """Get volatility multiplier for adjustment"""
        try:
            price_data = list(self.price_history[symbol])
            if len(price_data) < 10:
                return 1.0
            
            # Calculate recent volatility
            prices = [p['price'] for p in price_data[-20:]]
            if len(prices) < 2:
                return 1.0
            
            returns = [prices[i]/prices[i-1] - 1 for i in range(1, len(prices))]
            volatility = np.std(returns) if returns else 0.0
            
            # Classify volatility and return multiplier
            if volatility > 0.03:  # High volatility (>3% daily moves)
                return self.config.high_volatility_multiplier
            elif volatility < 0.01:  # Low volatility (<1% daily moves)
                return self.config.low_volatility_multiplier
            else:
                return 1.0  # Normal volatility
                
        except Exception as e:
            logger.error(f"Error calculating volatility multiplier for {symbol}: {e}")
            return 1.0
    
    def _is_stop_loss_hit(self, position: Position) -> bool:
        """Check if stop-loss is hit"""
        if position.side == PositionSide.LONG:
            return position.current_price <= position.stop_loss
        else:
            return position.current_price >= position.stop_loss
    
    def _is_take_profit_hit(self, position: Position) -> bool:
        """Check if take-profit is hit"""
        if not position.take_profit:
            return False
        
        if position.side == PositionSide.LONG:
            return position.current_price >= position.take_profit
        else:
            return position.current_price <= position.take_profit
    
    def add_stop_loss_callback(self, callback: Callable[[TrailingStopEvent], None]):
        """Add callback for stop-loss events"""
        self.stop_loss_callbacks.append(callback)
    
    def add_trail_update_callback(self, callback: Callable[[TrailingStopEvent], None]):
        """Add callback for trail update events"""
        self.trail_update_callbacks.append(callback)
    
    def _notify_stop_loss(self, event: TrailingStopEvent):
        """Notify stop-loss callbacks"""
        for callback in self.stop_loss_callbacks:
            try:
                callback(event)
            except Exception as e:
                logger.error(f"Error in stop-loss callback: {e}")
    
    def _notify_trail_update(self, event: TrailingStopEvent):
        """Notify trail update callbacks"""
        for callback in self.trail_update_callbacks:
            try:
                callback(event)
            except Exception as e:
                logger.error(f"Error in trail update callback: {e}")
    
    def get_position_status(self, symbol: str = None) -> Dict[str, Any]:
        """Get status of positions"""
        with self.lock:
            if symbol:
                if symbol in self.positions:
                    position = self.positions[symbol]
                    return {
                        'symbol': position.symbol,
                        'side': position.side.value,
                        'quantity': position.quantity,
                        'entry_price': position.entry_price,
                        'current_price': position.current_price,
                        'stop_loss': position.stop_loss,
                        'take_profit': position.take_profit,
                        'unrealized_pnl': position.unrealized_pnl,
                        'unrealized_pnl_percent': position.unrealized_pnl_percent,
                        'trail_count': len(position.trail_history),
                        'last_updated': position.last_updated.isoformat()
                    }
                else:
                    return {'error': f'Position {symbol} not found'}
            else:
                # Return all positions
                positions_status = {}
                for sym, pos in self.positions.items():
                    positions_status[sym] = {
                        'side': pos.side.value,
                        'quantity': pos.quantity,
                        'entry_price': pos.entry_price,
                        'current_price': pos.current_price,
                        'stop_loss': pos.stop_loss,
                        'unrealized_pnl': pos.unrealized_pnl,
                        'unrealized_pnl_percent': pos.unrealized_pnl_percent,
                        'trail_count': len(pos.trail_history)
                    }
                
                return {
                    'total_positions': len(self.positions),
                    'positions': positions_status
                }
    
    def start_monitoring(self):
        """Start monitoring thread for periodic updates"""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        logger.info("Started trailing stop monitoring")
    
    def stop_monitoring(self):
        """Stop monitoring thread"""
        self.is_monitoring = False
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=5)
        logger.info("Stopped trailing stop monitoring")
    
    def _monitoring_loop(self):
        """Monitoring loop for periodic checks"""
        while self.is_monitoring:
            try:
                # Perform periodic maintenance
                self._cleanup_old_price_history()
                time.sleep(self.config.update_frequency_seconds)
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(1)
    
    def _cleanup_old_price_history(self):
        """Clean up old price history data"""
        cutoff_time = datetime.now() - timedelta(hours=1)
        
        for symbol in self.price_history:
            # Remove old entries
            while (self.price_history[symbol] and 
                   self.price_history[symbol][0]['timestamp'] < cutoff_time):
                self.price_history[symbol].popleft()


# Example usage and testing
if __name__ == "__main__":
    # Configuration
    config = TrailingConfig(
        trailing_type=TrailingType.PERCENTAGE,
        trail_percentage=2.0,
        min_profit_to_trail=1.0
    )
    
    # Create trailing stop manager
    trail_manager = TrailingStopManager(config)
    
    # Add callbacks
    def on_stop_loss(event: TrailingStopEvent):
        print(f"STOP LOSS HIT: {event.symbol} at ₹{event.current_price:.2f}")
    
    def on_trail_update(event: TrailingStopEvent):
        print(f"TRAIL UPDATE: {event.symbol} SL: ₹{event.old_stop_loss:.2f} → ₹{event.new_stop_loss:.2f}")
    
    trail_manager.add_stop_loss_callback(on_stop_loss)
    trail_manager.add_trail_update_callback(on_trail_update)
    
    # Add a test position
    trail_manager.add_position(
        symbol="RELIANCE",
        side=PositionSide.LONG,
        quantity=100,
        entry_price=2500.0,
        initial_stop_loss=2450.0,
        take_profit=2600.0
    )
    
    # Simulate price updates
    test_prices = [2500, 2510, 2520, 2530, 2525, 2535, 2540, 2530, 2520]
    
    for price in test_prices:
        print(f"\nPrice update: ₹{price}")
        event = trail_manager.update_price("RELIANCE", price)
        if event:
            print(f"Event: {event.event_type}")
        
        status = trail_manager.get_position_status("RELIANCE")
        print(f"Status: P&L: ₹{status['unrealized_pnl']:.2f} ({status['unrealized_pnl_percent']:.2f}%)")
        print(f"Stop Loss: ₹{status['stop_loss']:.2f}")
        
        time.sleep(0.5)