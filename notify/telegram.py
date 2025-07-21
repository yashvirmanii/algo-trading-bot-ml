"""
Telegram Notification Module

This module provides:
- Real-time notifications and controls via Telegram bot
- Command handling for status, mode, PnL, universe, weights, sentiment, and more
- Error and analytics reporting to the user

All Telegram credentials are loaded securely, and only authorized users can control the bot.
"""

from telegram import Bot, Update
from telegram.ext import Updater, CommandHandler, CallbackContext
import logging

logging.basicConfig(level=logging.INFO)

class TelegramNotifier:
    def __init__(self, token, chat_id):
        self.token = token
        self.chat_id = chat_id
        self.bot = Bot(token=self.token)
        self.updater = Updater(token=self.token, use_context=True)
        self.dispatcher = self.updater.dispatcher
        self._register_handlers()
        self.is_running = True
        self.mode = 'paper'  # Default mode

    def _register_handlers(self):
        self.dispatcher.add_handler(CommandHandler('start', self.start))
        self.dispatcher.add_handler(CommandHandler('stop', self.stop))
        self.dispatcher.add_handler(CommandHandler('status', self.status))
        self.dispatcher.add_handler(CommandHandler('help', self.help))
        self.dispatcher.add_handler(CommandHandler('mode', self.mode_cmd))
        self.dispatcher.add_handler(CommandHandler('pnl', self.pnl))
        self.dispatcher.add_handler(CommandHandler('universe', self.universe))
        self.dispatcher.add_handler(CommandHandler('weights', self.weights))
        self.dispatcher.add_handler(CommandHandler('sentiment', self.sentiment))

    def send_message(self, text):
        try:
            self.bot.send_message(chat_id=self.chat_id, text=text)
        except Exception as e:
            print(f"Failed to send Telegram message: {e}")

    def start(self, update: Update, context: CallbackContext):
        if str(update.effective_chat.id) != str(self.chat_id):
            return
        context.bot.send_message(chat_id=self.chat_id, text="🤖 Trading bot started! Use /help to see commands.")

    def stop(self, update: Update, context: CallbackContext):
        if str(update.effective_chat.id) != str(self.chat_id):
            return
        self.is_running = False
        context.bot.send_message(chat_id=self.chat_id, text="🛑 Trading bot stopped by user (kill switch activated).")

    def status(self, update: Update, context: CallbackContext):
        if str(update.effective_chat.id) != str(self.chat_id):
            return
        # Placeholder: Replace with real status
        context.bot.send_message(chat_id=self.chat_id, text=f"Bot is running. Mode: {self.mode}. Portfolio: [placeholder]")

    def help(self, update: Update, context: CallbackContext):
        if str(update.effective_chat.id) != str(self.chat_id):
            return
        help_text = (
            "/start - Start the bot\n"
            "/stop - Stop all trading activity\n"
            "/status - Get current bot status\n"
            "/mode [live/paper] - Toggle between live and paper trading mode\n"
            "/pnl - Get current day's PnL and trade summary\n"
            "/universe - Show current stock universe being traded\n"
            "/weights - Show current strategy/indicator weights\n"
            "/sentiment - Show current sentiment scores for universe\n"
            "/help - List all commands\n"
        )
        context.bot.send_message(chat_id=self.chat_id, text=help_text)

    def mode_cmd(self, update: Update, context: CallbackContext):
        """/mode [live/paper] - Toggle between live and paper trading mode."""
        if str(update.effective_chat.id) != str(self.chat_id):
            return
        if context.args and context.args[0].lower() in ['live', 'paper']:
            self.mode = context.args[0].lower()
            context.bot.send_message(chat_id=self.chat_id, text=f"Trading mode set to: {self.mode}")
        else:
            context.bot.send_message(chat_id=self.chat_id, text="Usage: /mode [live/paper]")

    def pnl(self, update: Update, context: CallbackContext):
        if str(update.effective_chat.id) != str(self.chat_id):
            return
        # Placeholder: Replace with real PnL logic
        context.bot.send_message(chat_id=self.chat_id, text="Today's PnL: ₹0.00 | Trades: 0 | Win rate: 0%")

    def universe(self, update: Update, context: CallbackContext):
        if str(update.effective_chat.id) != str(self.chat_id):
            return
        # Placeholder: Replace with real universe
        context.bot.send_message(chat_id=self.chat_id, text="Universe: [RELIANCE, TCS, INFY, ...]")

    def weights(self, update: Update, context: CallbackContext):
        if str(update.effective_chat.id) != str(self.chat_id):
            return
        # Placeholder: Replace with real weights
        context.bot.send_message(chat_id=self.chat_id, text="Weights: RSI=20%, MACD=30%, ATR=50%")

    def sentiment(self, update: Update, context: CallbackContext):
        if str(update.effective_chat.id) != str(self.chat_id):
            return
        # Placeholder: Replace with real sentiment
        context.bot.send_message(chat_id=self.chat_id, text="Sentiment: RELIANCE=Bullish, TCS=Neutral, INFY=Bearish")

    def run_bot(self):
        self.updater.start_polling()
        self.updater.idle() 