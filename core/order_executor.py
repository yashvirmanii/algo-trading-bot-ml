"""
Order Executor Module

Implements smart order execution:
- Tries limit order near bid/ask first
- Converts to market after X seconds or if unfilled
- Controls slippage
- Places SL/TP immediately after entry
"""
import time
import logging

class SmartOrderExecutor:
    def __init__(self, broker, slippage_tolerance=0.002, limit_wait=5):
        self.broker = broker
        self.slippage_tolerance = slippage_tolerance  # e.g. 0.2%
        self.limit_wait = limit_wait  # seconds to wait for limit fill

    def execute_order(self, symbol, qty, entry_price, sl, tp, side='buy', live=True):
        """
        Attempts limit order, then market if not filled. Places SL/TP immediately.
        Returns: (success, fill_price, message)
        """
        try:
            if not live:
                return True, entry_price, 'Paper trade (no real order)'
            # 1. Place limit order
            limit_price = entry_price * (1 - self.slippage_tolerance) if side == 'buy' else entry_price * (1 + self.slippage_tolerance)
            order_id = self.broker.place_order(symbol, qty, 'LIMIT', price=limit_price)
            start = time.time()
            filled = False
            fill_price = limit_price
            # 2. Wait for fill or timeout
            while time.time() - start < self.limit_wait:
                status = self.broker.check_order_status(order_id)
                if status == 'FILLED':
                    filled = True
                    break
                time.sleep(1)
            # 3. If not filled, convert to market
            if not filled:
                self.broker.cancel_order(order_id)
                order_id = self.broker.place_order(symbol, qty, 'MARKET')
                fill_price = self.broker.get_order_fill_price(order_id)
            # 4. Place SL/TP orders
            self.broker.place_sl_tp_orders(symbol, qty, sl, tp, side)
            return True, fill_price, f'Order executed at {fill_price:.2f} (SL: {sl:.2f}, TP: {tp:.2f})'
        except Exception as e:
            logging.error(f"Order execution error: {e}")
            return False, None, f'Order execution error: {e}' 