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
        self.capital_manager = None  # Will be set during initialization

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
        self.dispatcher.add_handler(CommandHandler('setcapital', self.set_capital))
        self.dispatcher.add_handler(CommandHandler('capitalstatus', self.capital_status))
        self.dispatcher.add_handler(CommandHandler('capitalhistory', self.capital_history))

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
            "🤖 **Trading Bot Commands**\n\n"
            "**Basic Controls:**\n"
            "/start - Start the bot\n"
            "/stop - Stop all trading activity\n"
            "/status - Get current bot status\n"
            "/mode [live/paper] - Toggle between live and paper trading mode\n"
            "/help - List all commands\n\n"
            "**Trading Information:**\n"
            "/pnl - Get current day's PnL and trade summary\n"
            "/universe - Show current stock universe being traded\n"
            "/weights - Show current strategy/indicator weights\n"
            "/sentiment - Show current sentiment scores for universe\n\n"
            "**💰 Capital Management:**\n"
            "/setcapital <amount> - Update trading capital (e.g., /setcapital 100000)\n"
            "/capitalstatus - Show current capital allocation and status\n"
            "/capitalhistory [limit] - Show capital change history\n\n"
            "**Examples:**\n"
            "• /setcapital 50000 - Set capital to ₹50,000\n"
            "• /capitalhistory 5 - Show last 5 capital changes\n"
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

    def set_capital(self, update: Update, context: CallbackContext):
        """Handle /setcapital command for dynamic capital management"""
        if str(update.effective_chat.id) != str(self.chat_id):
            return
        
        if not self.capital_manager:
            context.bot.send_message(chat_id=self.chat_id, text="❌ Capital manager not initialized")
            return
        
        try:
            if not context.args:
                current_capital = self.capital_manager.get_current_capital()
                message = (
                    f"💰 Current Capital: ₹{current_capital:,.2f}\n\n"
                    "Usage: /setcapital <amount>\n"
                    "Example: /setcapital 100000"
                )
                context.bot.send_message(chat_id=self.chat_id, text=message)
                return
            
            # Parse capital amount
            try:
                new_capital = float(context.args[0])
            except ValueError:
                context.bot.send_message(chat_id=self.chat_id, text="❌ Invalid amount. Please enter a valid number.")
                return
            
            # Get user info
            user_id = str(update.effective_user.id) if update.effective_user else None
            username = update.effective_user.username if update.effective_user else "Unknown"
            
            # Update capital
            result = self.capital_manager.update_capital(
                new_capital=new_capital,
                reason=f"Telegram command by {username}",
                user_id=user_id,
                source="telegram"
            )
            
            if result['success']:
                message = (
                    f"✅ Capital Updated Successfully!\n\n"
                    f"💰 Old Capital: ₹{result['old_capital']:,.2f}\n"
                    f"💰 New Capital: ₹{result['new_capital']:,.2f}\n"
                    f"📈 Change: ₹{result['change_amount']:+,.2f} ({result['change_percentage']:+.1f}%)\n\n"
                    f"🔄 All system components have been reallocated automatically."
                )
            else:
                message = f"❌ Error updating capital: {result['error']}"
            
            context.bot.send_message(chat_id=self.chat_id, text=message)
                
        except Exception as e:
            logging.error(f"Error in set_capital command: {e}")
            context.bot.send_message(chat_id=self.chat_id, text=f"❌ Error processing command: {str(e)}")

    def capital_status(self, update: Update, context: CallbackContext):
        """Handle /capitalstatus command"""
        if str(update.effective_chat.id) != str(self.chat_id):
            return
        
        if not self.capital_manager:
            context.bot.send_message(chat_id=self.chat_id, text="❌ Capital manager not initialized")
            return
        
        try:
            allocation = self.capital_manager.get_capital_allocation()
            stats = self.capital_manager.get_capital_statistics()
            
            message = f"💰 **Capital Status Report**\n\n"
            message += f"🏦 **Current Capital**: ₹{allocation.total_capital:,.2f}\n"
            message += f"💵 Available: ₹{allocation.available_capital:,.2f}\n"
            message += f"📊 Allocated: ₹{allocation.allocated_capital:,.2f}\n"
            message += f"🛡️ Reserved: ₹{allocation.reserved_capital:,.2f}\n\n"
            
            if allocation.agent_allocations:
                message += "🤖 **Agent Allocations**:\n"
                for agent_id, amount in allocation.agent_allocations.items():
                    message += f"• {agent_id}: ₹{amount:,.2f}\n"
                message += "\n"
            
            message += f"📈 **Statistics**:\n"
            message += f"• Total Changes: {stats['total_changes']}\n"
            message += f"• Average Change: ₹{stats.get('average_change', 0):+,.2f}\n"
            
            if stats.get('last_change'):
                message += f"• Last Updated: {stats['last_change'][:19]}\n"
            
            context.bot.send_message(chat_id=self.chat_id, text=message)
            
        except Exception as e:
            logging.error(f"Error in capital_status command: {e}")
            context.bot.send_message(chat_id=self.chat_id, text=f"❌ Error getting capital status: {str(e)}")

    def capital_history(self, update: Update, context: CallbackContext):
        """Handle /capitalhistory command"""
        if str(update.effective_chat.id) != str(self.chat_id):
            return
        
        if not self.capital_manager:
            context.bot.send_message(chat_id=self.chat_id, text="❌ Capital manager not initialized")
            return
        
        try:
            limit = 10
            if context.args:
                try:
                    limit = min(int(context.args[0]), 20)  # Max 20 records for Telegram
                except ValueError:
                    pass
            
            history = self.capital_manager.get_capital_history(limit)
            
            if not history:
                context.bot.send_message(chat_id=self.chat_id, text="📊 No capital change history available.")
                return
            
            message = f"📊 **Capital Change History** (Last {len(history)} changes)\n\n"
            
            for record in reversed(history):  # Most recent first
                change_emoji = "📈" if record.change_amount > 0 else "📉"
                message += (
                    f"{change_emoji} {record.timestamp.strftime('%Y-%m-%d %H:%M')}\n"
                    f"   ₹{record.old_capital:,.0f} → ₹{record.new_capital:,.0f} "
                    f"({record.change_percentage:+.1f}%)\n"
                    f"   Reason: {record.reason}\n\n"
                )
            
            context.bot.send_message(chat_id=self.chat_id, text=message)
            
        except Exception as e:
            logging.error(f"Error in capital_history command: {e}")
            context.bot.send_message(chat_id=self.chat_id, text=f"❌ Error getting capital history: {str(e)}")

    def set_capital_manager(self, capital_manager):
        """Set the capital manager reference"""
        self.capital_manager = capital_manager

    def run_bot(self):
        self.updater.start_polling()
        self.updater.idle() 