"""
Zerodha Broker Integration

This module provides a wrapper around the Zerodha Kite Connect API for:
- Authentication and secure credential management
- Fetching the full instrument list and stock universe
- Downloading historical OHLCV data for any symbol
- Order placement and management (limit, market, SL/TP)

All API keys and tokens are loaded securely from environment variables. This module is the single point of contact for all broker-related operations in the trading bot.
"""

import os
import pandas as pd
from kiteconnect import KiteConnect
from dotenv import load_dotenv
from datetime import datetime, timedelta

class ZerodhaBroker:
    def __init__(self, api_key=None, api_secret=None, access_token=None):
        load_dotenv()
        self.api_key = api_key or os.getenv('KITE_API_KEY')
        self.api_secret = api_secret or os.getenv('KITE_API_SECRET')
        self.access_token = access_token or os.getenv('KITE_ACCESS_TOKEN')
        self.kite = None
        self.authenticate()
        self.instruments = self.kite.instruments('NSE')

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