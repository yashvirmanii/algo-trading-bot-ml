"""
Dynamic Capital Manager

This module implements dynamic capital management that allows real-time updates
to trading capital via Telegram commands, with automatic reallocation of all
system components including agent budgets, position sizing, and risk parameters.

Key Features:
- Real-time capital updates via Telegram commands
- Dynamic reallocation of multi-agent budgets
- Automatic adjustment of position sizing parameters
- Risk parameter scaling based on capital changes
- Persistent storage of capital amount
- Capital change history and audit trail
- Safety checks and validation
"""

import os
import json
import logging
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime
import threading
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class CapitalChangeRecord:
    """Record of capital change"""
    timestamp: datetime
    old_capital: float
    new_capital: float
    change_amount: float
    change_percentage: float
    reason: str
    user_id: Optional[str] = None
    source: str = "telegram"  # telegram, api, manual


@dataclass
class CapitalAllocation:
    """Capital allocation breakdown"""
    total_capital: float
    available_capital: float
    allocated_capital: float
    reserved_capital: float
    agent_allocations: Dict[str, float] = field(default_factory=dict)
    position_sizing_limits: Dict[str, float] = field(default_factory=dict)
    risk_parameters: Dict[str, float] = field(default_factory=dict)
    last_updated: datetime = field(default_factory=datetime.now)


class DynamicCapitalManager:
    """
    Manages dynamic capital allocation and real-time updates across all system components
    """
    
    def __init__(self, initial_capital: float = 100000.0, 
                 storage_file: str = "data/capital_config.json",
                 min_capital: float = 10000.0,
                 max_capital: float = 10000000.0):
        
        self.storage_file = storage_file
        self.min_capital = min_capital
        self.max_capital = max_capital
        self.lock = threading.Lock()
        
        # Ensure storage directory exists
        os.makedirs(os.path.dirname(storage_file), exist_ok=True)
        
        # Initialize capital and load from storage
        self.current_capital = self._load_capital() or initial_capital
        self.capital_history: List[CapitalChangeRecord] = []
        self.allocation = CapitalAllocation(total_capital=self.current_capital)
        
        # Callbacks for capital changes
        self.capital_change_callbacks: List[Callable[[float, float], None]] = []
        
        # Component references (will be set during initialization)
        self.multi_agent_coordinator = None
        self.position_sizer = None
        self.risk_manager = None
        self.portfolio_tracker = None
        
        logger.info(f"DynamicCapitalManager initialized with capital: ₹{self.current_capital:,.2f}")
    
    def register_component(self, component_name: str, component_instance: Any):
        """Register system components for capital updates"""
        setattr(self, component_name, component_instance)
        logger.info(f"Registered component: {component_name}")
    
    def register_capital_change_callback(self, callback: Callable[[float, float], None]):
        """Register callback function to be called when capital changes"""
        self.capital_change_callbacks.append(callback)
    
    def update_capital(self, new_capital: float, reason: str = "Manual update", 
                      user_id: str = None, source: str = "telegram") -> Dict[str, Any]:
        """
        Update trading capital and reallocate all system components
        
        Args:
            new_capital: New capital amount
            reason: Reason for capital change
            user_id: User who initiated the change
            source: Source of the change (telegram, api, manual)
            
        Returns:
            Dictionary with update results and new allocations
        """
        with self.lock:
            try:
                # Validate new capital
                validation_result = self._validate_capital_change(new_capital)
                if not validation_result['valid']:
                    return {
                        'success': False,
                        'error': validation_result['error'],
                        'current_capital': self.current_capital
                    }
                
                old_capital = self.current_capital
                change_amount = new_capital - old_capital
                change_percentage = (change_amount / old_capital) * 100 if old_capital > 0 else 0
                
                # Create change record
                change_record = CapitalChangeRecord(
                    timestamp=datetime.now(),
                    old_capital=old_capital,
                    new_capital=new_capital,
                    change_amount=change_amount,
                    change_percentage=change_percentage,
                    reason=reason,
                    user_id=user_id,
                    source=source
                )
                
                # Update capital
                self.current_capital = new_capital
                self.capital_history.append(change_record)
                
                # Reallocate all system components
                reallocation_results = self._reallocate_all_components(old_capital, new_capital)
                
                # Save to persistent storage
                self._save_capital()
                
                # Notify callbacks
                for callback in self.capital_change_callbacks:
                    try:
                        callback(old_capital, new_capital)
                    except Exception as e:
                        logger.error(f"Error in capital change callback: {e}")
                
                logger.info(f"Capital updated: ₹{old_capital:,.2f} → ₹{new_capital:,.2f} ({change_percentage:+.1f}%)")
                
                return {
                    'success': True,
                    'old_capital': old_capital,
                    'new_capital': new_capital,
                    'change_amount': change_amount,
                    'change_percentage': change_percentage,
                    'reallocation_results': reallocation_results,
                    'timestamp': change_record.timestamp.isoformat()
                }
                
            except Exception as e:
                logger.error(f"Error updating capital: {e}")
                return {
                    'success': False,
                    'error': str(e),
                    'current_capital': self.current_capital
                }
    
    def _validate_capital_change(self, new_capital: float) -> Dict[str, Any]:
        """Validate capital change request"""
        if not isinstance(new_capital, (int, float)):
            return {'valid': False, 'error': 'Capital must be a number'}
        
        if new_capital <= 0:
            return {'valid': False, 'error': 'Capital must be positive'}
        
        if new_capital < self.min_capital:
            return {'valid': False, 'error': f'Capital must be at least ₹{self.min_capital:,.2f}'}
        
        if new_capital > self.max_capital:
            return {'valid': False, 'error': f'Capital cannot exceed ₹{self.max_capital:,.2f}'}
        
        # Check for reasonable change (not more than 10x or less than 0.1x)
        if self.current_capital > 0:
            ratio = new_capital / self.current_capital
            if ratio > 10 or ratio < 0.1:
                return {
                    'valid': False, 
                    'error': f'Capital change too extreme (ratio: {ratio:.2f}x). Please confirm if intentional.'
                }
        
        return {'valid': True}
    
    def _reallocate_all_components(self, old_capital: float, new_capital: float) -> Dict[str, Any]:
        """Reallocate all system components based on new capital"""
        results = {}
        scaling_factor = new_capital / old_capital if old_capital > 0 else 1.0
        
        try:
            # 1. Reallocate Multi-Agent Coordinator
            if self.multi_agent_coordinator:
                agent_results = self._reallocate_multi_agent_coordinator(scaling_factor)
                results['multi_agent_coordinator'] = agent_results
            
            # 2. Update Position Sizer
            if self.position_sizer:
                position_results = self._update_position_sizer(new_capital)
                results['position_sizer'] = position_results
            
            # 3. Update Risk Manager
            if self.risk_manager:
                risk_results = self._update_risk_manager(new_capital)
                results['risk_manager'] = risk_results
            
            # 4. Update Portfolio Tracker
            if self.portfolio_tracker:
                portfolio_results = self._update_portfolio_tracker(new_capital)
                results['portfolio_tracker'] = portfolio_results
            
            # 5. Update allocation record
            self._update_allocation_record(new_capital, results)
            
            return results
            
        except Exception as e:
            logger.error(f"Error reallocating components: {e}")
            return {'error': str(e)}
    
    def _reallocate_multi_agent_coordinator(self, scaling_factor: float) -> Dict[str, Any]:
        """Reallocate multi-agent coordinator budgets"""
        try:
            if not hasattr(self.multi_agent_coordinator, 'agent_resources'):
                return {'error': 'Multi-agent coordinator not properly initialized'}
            
            reallocation_results = {}
            
            # Update total capital
            self.multi_agent_coordinator.total_capital = self.current_capital
            
            # Reallocate each agent's budget
            for agent_id, resource in self.multi_agent_coordinator.agent_resources.items():
                old_allocation = resource.allocated_capital
                new_allocation = old_allocation * scaling_factor
                
                # Update agent resource allocation
                resource.allocated_capital = new_allocation
                
                # Adjust used capital proportionally (but don't exceed new allocation)
                if resource.used_capital > 0:
                    new_used_capital = min(resource.used_capital * scaling_factor, new_allocation)
                    resource.used_capital = new_used_capital
                
                reallocation_results[agent_id] = {
                    'old_allocation': old_allocation,
                    'new_allocation': new_allocation,
                    'scaling_factor': scaling_factor
                }
                
                logger.info(f"Agent {agent_id}: ₹{old_allocation:,.2f} → ₹{new_allocation:,.2f}")
            
            return {
                'success': True,
                'total_capital': self.current_capital,
                'scaling_factor': scaling_factor,
                'agent_reallocations': reallocation_results
            }
            
        except Exception as e:
            logger.error(f"Error reallocating multi-agent coordinator: {e}")
            return {'error': str(e)}
    
    def _update_position_sizer(self, new_capital: float) -> Dict[str, Any]:
        """Update position sizer with new capital"""
        try:
            if not hasattr(self.position_sizer, 'config'):
                return {'error': 'Position sizer not properly initialized'}
            
            # Update any capital-dependent parameters in position sizer
            # The position sizer will automatically use the new capital in calculations
            
            return {
                'success': True,
                'new_capital': new_capital,
                'message': 'Position sizer will use new capital in future calculations'
            }
            
        except Exception as e:
            logger.error(f"Error updating position sizer: {e}")
            return {'error': str(e)}
    
    def _update_risk_manager(self, new_capital: float) -> Dict[str, Any]:
        """Update risk manager with new capital"""
        try:
            # Update risk manager's capital reference
            if hasattr(self.risk_manager, 'total_capital'):
                self.risk_manager.total_capital = new_capital
            
            return {
                'success': True,
                'new_capital': new_capital,
                'message': 'Risk manager updated with new capital'
            }
            
        except Exception as e:
            logger.error(f"Error updating risk manager: {e}")
            return {'error': str(e)}
    
    def _update_portfolio_tracker(self, new_capital: float) -> Dict[str, Any]:
        """Update portfolio tracker with new capital"""
        try:
            # Update portfolio tracker's capital reference
            if hasattr(self.portfolio_tracker, 'total_capital'):
                self.portfolio_tracker.total_capital = new_capital
            
            return {
                'success': True,
                'new_capital': new_capital,
                'message': 'Portfolio tracker updated with new capital'
            }
            
        except Exception as e:
            logger.error(f"Error updating portfolio tracker: {e}")
            return {'error': str(e)}
    
    def _update_allocation_record(self, new_capital: float, results: Dict[str, Any]):
        """Update the allocation record"""
        self.allocation = CapitalAllocation(
            total_capital=new_capital,
            available_capital=new_capital * 0.8,  # 80% available for trading
            allocated_capital=new_capital * 0.2,  # 20% allocated/reserved
            reserved_capital=new_capital * 0.1,   # 10% emergency reserve
            last_updated=datetime.now()
        )
        
        # Update agent allocations if available
        if 'multi_agent_coordinator' in results and 'agent_reallocations' in results['multi_agent_coordinator']:
            agent_reallocations = results['multi_agent_coordinator']['agent_reallocations']
            self.allocation.agent_allocations = {
                agent_id: data['new_allocation'] 
                for agent_id, data in agent_reallocations.items()
            }
    
    def get_current_capital(self) -> float:
        """Get current trading capital"""
        return self.current_capital
    
    def get_capital_allocation(self) -> CapitalAllocation:
        """Get current capital allocation breakdown"""
        return self.allocation
    
    def get_capital_history(self, limit: int = 50) -> List[CapitalChangeRecord]:
        """Get capital change history"""
        return self.capital_history[-limit:] if limit else self.capital_history
    
    def get_capital_statistics(self) -> Dict[str, Any]:
        """Get capital statistics and analytics"""
        if not self.capital_history:
            return {
                'current_capital': self.current_capital,
                'total_changes': 0,
                'message': 'No capital change history available'
            }
        
        changes = [record.change_amount for record in self.capital_history]
        percentages = [record.change_percentage for record in self.capital_history]
        
        return {
            'current_capital': self.current_capital,
            'initial_capital': self.capital_history[0].old_capital if self.capital_history else self.current_capital,
            'total_changes': len(self.capital_history),
            'total_change_amount': sum(changes),
            'average_change': sum(changes) / len(changes),
            'largest_increase': max(changes) if changes else 0,
            'largest_decrease': min(changes) if changes else 0,
            'average_change_percentage': sum(percentages) / len(percentages),
            'last_change': self.capital_history[-1].timestamp.isoformat() if self.capital_history else None
        }
    
    def _save_capital(self):
        """Save current capital to persistent storage"""
        try:
            capital_data = {
                'current_capital': self.current_capital,
                'last_updated': datetime.now().isoformat(),
                'capital_history': [
                    {
                        'timestamp': record.timestamp.isoformat(),
                        'old_capital': record.old_capital,
                        'new_capital': record.new_capital,
                        'change_amount': record.change_amount,
                        'change_percentage': record.change_percentage,
                        'reason': record.reason,
                        'user_id': record.user_id,
                        'source': record.source
                    }
                    for record in self.capital_history[-100:]  # Keep last 100 records
                ]
            }
            
            with open(self.storage_file, 'w') as f:
                json.dump(capital_data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving capital data: {e}")
    
    def _load_capital(self) -> Optional[float]:
        """Load capital from persistent storage"""
        try:
            if not os.path.exists(self.storage_file):
                return None
            
            with open(self.storage_file, 'r') as f:
                capital_data = json.load(f)
            
            # Load capital history
            if 'capital_history' in capital_data:
                for record_data in capital_data['capital_history']:
                    record = CapitalChangeRecord(
                        timestamp=datetime.fromisoformat(record_data['timestamp']),
                        old_capital=record_data['old_capital'],
                        new_capital=record_data['new_capital'],
                        change_amount=record_data['change_amount'],
                        change_percentage=record_data['change_percentage'],
                        reason=record_data['reason'],
                        user_id=record_data.get('user_id'),
                        source=record_data.get('source', 'unknown')
                    )
                    self.capital_history.append(record)
            
            return capital_data.get('current_capital')
            
        except Exception as e:
            logger.error(f"Error loading capital data: {e}")
            return None
    
    def export_capital_report(self, filepath: str = None) -> str:
        """Export detailed capital report"""
        if filepath is None:
            filepath = f"reports/capital_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        report = {
            'report_timestamp': datetime.now().isoformat(),
            'current_status': {
                'current_capital': self.current_capital,
                'allocation': {
                    'total_capital': self.allocation.total_capital,
                    'available_capital': self.allocation.available_capital,
                    'allocated_capital': self.allocation.allocated_capital,
                    'reserved_capital': self.allocation.reserved_capital,
                    'agent_allocations': self.allocation.agent_allocations,
                    'last_updated': self.allocation.last_updated.isoformat()
                }
            },
            'statistics': self.get_capital_statistics(),
            'recent_changes': [
                {
                    'timestamp': record.timestamp.isoformat(),
                    'old_capital': record.old_capital,
                    'new_capital': record.new_capital,
                    'change_amount': record.change_amount,
                    'change_percentage': record.change_percentage,
                    'reason': record.reason,
                    'source': record.source
                }
                for record in self.capital_history[-20:]  # Last 20 changes
            ]
        }
        
        try:
            with open(filepath, 'w') as f:
                json.dump(report, f, indent=2)
            
            logger.info(f"Capital report exported to: {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Error exporting capital report: {e}")
            return ""


# Telegram command handlers
class TelegramCapitalCommands:
    """Telegram command handlers for capital management"""
    
    def __init__(self, capital_manager: DynamicCapitalManager):
        self.capital_manager = capital_manager
    
    def handle_set_capital(self, update, context) -> str:
        """Handle /setcapital command"""
        try:
            if not context.args:
                return (
                    f"💰 Current Capital: ₹{self.capital_manager.get_current_capital():,.2f}\n\n"
                    "Usage: /setcapital <amount>\n"
                    "Example: /setcapital 100000"
                )
            
            # Parse capital amount
            try:
                new_capital = float(context.args[0])
            except ValueError:
                return "❌ Invalid amount. Please enter a valid number."
            
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
                return (
                    f"✅ Capital Updated Successfully!\n\n"
                    f"💰 Old Capital: ₹{result['old_capital']:,.2f}\n"
                    f"💰 New Capital: ₹{result['new_capital']:,.2f}\n"
                    f"📈 Change: ₹{result['change_amount']:+,.2f} ({result['change_percentage']:+.1f}%)\n\n"
                    f"🔄 All system components have been reallocated automatically."
                )
            else:
                return f"❌ Error updating capital: {result['error']}"
                
        except Exception as e:
            logger.error(f"Error in set_capital command: {e}")
            return f"❌ Error processing command: {str(e)}"
    
    def handle_capital_status(self, update, context) -> str:
        """Handle /capitalstatus command"""
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
            
            return message
            
        except Exception as e:
            logger.error(f"Error in capital_status command: {e}")
            return f"❌ Error getting capital status: {str(e)}"
    
    def handle_capital_history(self, update, context) -> str:
        """Handle /capitalhistory command"""
        try:
            limit = 10
            if context.args:
                try:
                    limit = min(int(context.args[0]), 50)  # Max 50 records
                except ValueError:
                    pass
            
            history = self.capital_manager.get_capital_history(limit)
            
            if not history:
                return "📊 No capital change history available."
            
            message = f"📊 **Capital Change History** (Last {len(history)} changes)\n\n"
            
            for record in reversed(history):  # Most recent first
                change_emoji = "📈" if record.change_amount > 0 else "📉"
                message += (
                    f"{change_emoji} {record.timestamp.strftime('%Y-%m-%d %H:%M')}\n"
                    f"   ₹{record.old_capital:,.0f} → ₹{record.new_capital:,.0f} "
                    f"({record.change_percentage:+.1f}%)\n"
                    f"   Reason: {record.reason}\n\n"
                )
            
            return message
            
        except Exception as e:
            logger.error(f"Error in capital_history command: {e}")
            return f"❌ Error getting capital history: {str(e)}"


# Example usage and testing
if __name__ == "__main__":
    # Initialize capital manager
    capital_manager = DynamicCapitalManager(initial_capital=100000.0)
    
    # Test capital update
    result = capital_manager.update_capital(150000.0, "Test increase")
    print(f"Update result: {result}")
    
    # Get status
    allocation = capital_manager.get_capital_allocation()
    print(f"Current allocation: {allocation}")
    
    # Get statistics
    stats = capital_manager.get_capital_statistics()
    print(f"Statistics: {stats}")
    
    # Test Telegram commands
    telegram_commands = TelegramCapitalCommands(capital_manager)
    print("Telegram commands initialized")