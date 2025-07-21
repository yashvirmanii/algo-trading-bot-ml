"""
Zerodha Broker Integration

This module provides a wrapper around the Zerodha Kite Connect API for:
- Authentication and secure credential management
- Fetching the full instrument list and stock universe
- Downloading historical OHLCV data for any symbol
- (Planned) Order placement and portfolio management

All API keys and tokens are loaded securely from environment variables. This module is the single point of contact for all broker-related operations in the trading bot.
"""

import os
import pandas as pd
from kiteconnect import KiteConnect
from dotenv import load_dotenv

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
        # Example: NIFTY 100 stocks
        nifty100 = [row['tradingsymbol'] for row in self.instruments if row.get('index') == 'NIFTY 100' or row.get('name') == 'NIFTY 100']
        if not nifty100:
            # fallback: top 50 by market cap
            nifty100 = [row['tradingsymbol'] for row in self.instruments[:50]]
        return nifty100

    def get_historical_data(self, symbol, interval='day', days=7):
        # Find instrument token
        token = None
        for row in self.instruments:
            if row['tradingsymbol'] == symbol:
                token = row['instrument_token']
                break
        if not token:
            raise Exception(f"Symbol {symbol} not found in instruments")
        from datetime import datetime, timedelta
        to_date = datetime.now()
        from_date = to_date - timedelta(days=days)
        data = self.kite.historical_data(token, from_date, to_date, interval)
        df = pd.DataFrame(data)
        if not df.empty:
            df.rename(columns={'date': 'datetime'}, inplace=True)
            df.set_index('datetime', inplace=True)
        return df

    def place_order(self, symbol, qty, order_type, price=None):
        # TODO: Implement order placement logic
        pass

    def get_portfolio(self):
        # TODO: Implement portfolio fetch logic
        pass 