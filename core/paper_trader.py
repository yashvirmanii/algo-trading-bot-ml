"""
Paper Trader Module

Simulates trades, PnL, and capital usage for paper trading mode. No real orders are placed.
"""

class PaperTrader:
    def __init__(self, total_capital):
        self.total_capital = total_capital
        self.available_capital = total_capital
        self.positions = []
        self.pnl = 0
        self.trades = []

    def buy(self, symbol, qty, price, sl, tp):
        cost = qty * price
        if cost > self.available_capital:
            return False, 'Insufficient capital'
        self.available_capital -= cost
        self.positions.append({'symbol': symbol, 'qty': qty, 'entry': price, 'sl': sl, 'tp': tp})
        self.trades.append({'symbol': symbol, 'side': 'buy', 'qty': qty, 'price': price, 'sl': sl, 'tp': tp})
        return True, 'Paper buy executed'

    def sell(self, symbol, qty, price):
        for pos in self.positions:
            if pos['symbol'] == symbol and pos['qty'] == qty:
                pnl = (price - pos['entry']) * qty
                self.pnl += pnl
                self.available_capital += qty * price
                self.positions.remove(pos)
                self.trades.append({'symbol': symbol, 'side': 'sell', 'qty': qty, 'price': price, 'pnl': pnl})
                return True, f'Paper sell executed, PnL: {pnl:.2f}'
        return False, 'Position not found'

    def get_pnl(self):
        return self.pnl

    def get_capital(self):
        return self.available_capital

    def get_trades(self):
        return self.trades 