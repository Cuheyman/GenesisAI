from datetime import datetime, timedelta
import logging
import asyncio
import time
import config

class RiskManager:
    def __init__(self, initial_equity):
        self.initial_equity = initial_equity
        self.total_equity = initial_equity
        self.equity_high_water_mark = initial_equity
        self.current_drawdown = 0.0
        self.max_drawdown_recorded = 0.0
        self.daily_high_equity = initial_equity
        self.position_count = 0
        self.position_values = {}
        self.day_start_time = time.time()
        self.equity_history = []
        self.risk_level = "normal"  # normal, reduced, minimal
        self.day_profit_target_reached = False
        
        # Use drawdown thresholds from config
        self.drawdown_thresholds = config.DRAWDOWN_THRESHOLDS
        
        # For new users or conservative settings, use more conservative thresholds
        if getattr(config, 'CONSERVATIVE_THRESHOLDS_ENABLED', False):
            self.drawdown_thresholds = config.CONSERVATIVE_THRESHOLDS
        
        # Recovery mode settings
        self.recovery_mode = False
        self.severe_recovery_mode = False
        self.recovery_start_equity = 0.0
        self.recovery_start_time = None
        self.drawdown_actions_taken = set()  # Track which actions have been taken
        
        # Risk parameters (can be adjusted based on drawdown)
        self.position_size_multiplier = 1.0
        self.max_positions_multiplier = 1.0
        
        # Initialize logging
        logging.info(f"Risk Manager initialized with equity protection: Warning at {self.drawdown_thresholds['warning']*100}%, "
                   f"Risk Reduction at {self.drawdown_thresholds['reduce_risk']*100}%, "
                   f"High Alert at {self.drawdown_thresholds['high_alert']*100}%, "
                   f"Emergency at {self.drawdown_thresholds['emergency']*100}%")
    
    def update_equity(self, current_equity):
        """Update equity tracking and check drawdown protection"""
        self.total_equity = current_equity
        self.equity_history.append((time.time(), current_equity))
        
        # Update high water mark if new high
        if current_equity > self.equity_high_water_mark:
            old_hwm = self.equity_high_water_mark
            self.equity_high_water_mark = current_equity
            logging.info(f"New equity high water mark: ${current_equity:.2f} (previous: ${old_hwm:.2f})")
        
        # Track daily high
        seconds_in_day = 86400
        day_elapsed = (time.time() - self.day_start_time)
        
        if day_elapsed > seconds_in_day:
            # Reset for new day
            self.day_start_time = time.time()
            self.daily_high_equity = current_equity
            self.day_profit_target_reached = False
        elif current_equity > self.daily_high_equity:
            self.daily_high_equity = current_equity
        
        # Calculate drawdown
        self.current_drawdown = self._calculate_drawdown()
        
        # Update max drawdown record if needed
        if self.current_drawdown > self.max_drawdown_recorded:
            self.max_drawdown_recorded = self.current_drawdown
            logging.warning(f"New maximum drawdown recorded: {self.max_drawdown_recorded*100:.2f}%")
            
        # Check drawdown thresholds and take action if needed
        self._check_drawdown_protection()
            
        # Update risk level based on drawdown
        if self.current_drawdown > self.drawdown_thresholds['high_alert']:
            self.risk_level = "minimal"
        elif self.current_drawdown > self.drawdown_thresholds['reduce_risk']:
            self.risk_level = "reduced"
        else:
            self.risk_level = "normal"
            
        # Calculate daily ROI
        daily_roi = self.calculate_daily_roi()
        
        return {
            "equity": current_equity,
            "drawdown": self.current_drawdown * 100,  # Convert to percentage
            "daily_roi": daily_roi,
            "risk_level": self.risk_level,
            "recovery_mode": self.recovery_mode,
            "severe_recovery_mode": self.severe_recovery_mode
        }
    
    def _calculate_drawdown(self):
        """Calculate current drawdown percentage (as decimal)"""
        if self.equity_high_water_mark <= 0:
            return 0
            
        drawdown = (self.equity_high_water_mark - self.total_equity) / self.equity_high_water_mark
        return max(0, drawdown)  # Ensure non-negative
    
    def calculate_daily_roi(self):
        """Calculate daily ROI percentage"""
        day_start_equity = self.initial_equity
        
        # Find equity at start of day
        for timestamp, equity in self.equity_history:
            if timestamp >= self.day_start_time:
                break
            day_start_equity = equity
            
        # Calculate daily ROI
        if day_start_equity > 0:
            daily_roi = ((self.total_equity / day_start_equity) - 1) * 100
            return daily_roi
        return 0
    
    def _check_drawdown_protection(self):
        """Check drawdown thresholds and take action if needed"""
        drawdown_pct = self.current_drawdown * 100  # Convert to percentage
        
        # Skip if no drawdown
        if drawdown_pct <= 0:
            return
            
        # Check against thresholds
        if drawdown_pct >= self.drawdown_thresholds['emergency'] * 100:
            # Emergency level - most severe response
            if 'emergency' not in self.drawdown_actions_taken:
                logging.warning(f"EMERGENCY DRAWDOWN LEVEL REACHED ({drawdown_pct:.2f}%)!")
                logging.warning("Taking emergency measures - risk significantly reduced")
                
                # Enter severe recovery mode
                self.recovery_mode = True
                self.severe_recovery_mode = True
                self.recovery_start_equity = self.total_equity
                self.recovery_start_time = time.time()
                
                # Reduce position sizing dramatically
                self.position_size_multiplier = 0.3  # 70% reduction
                
                # Reduce max positions significantly
                self.max_positions_multiplier = 0.3  # Only 30% of normal max positions
                
                # Record action taken
                self.drawdown_actions_taken.add('emergency')
                
                # Alert for risk reduction
                return "EMERGENCY_RISK_REDUCTION"
                
        elif drawdown_pct >= self.drawdown_thresholds['high_alert'] * 100:
            # High alert - reduce risk substantially
            if 'high_alert' not in self.drawdown_actions_taken:
                logging.warning(f"HIGH ALERT DRAWDOWN LEVEL REACHED ({drawdown_pct:.2f}%)!")
                
                # Enter recovery mode
                self.recovery_mode = True
                self.recovery_start_equity = self.total_equity
                self.recovery_start_time = time.time()
                
                # Reduce position sizing
                self.position_size_multiplier = 0.5  # 50% reduction
                
                # Reduce max positions
                self.max_positions_multiplier = 0.5  # Only 50% of normal max positions
                
                # Record action taken
                self.drawdown_actions_taken.add('high_alert')
                
                # Alert for risk reduction
                return "HIGH_ALERT_RISK_REDUCTION"
                
        elif drawdown_pct >= self.drawdown_thresholds['reduce_risk'] * 100:
            # Risk reduction - scale back trading
            if 'reduce_risk' not in self.drawdown_actions_taken:
                logging.warning(f"RISK REDUCTION DRAWDOWN LEVEL REACHED ({drawdown_pct:.2f}%)!")
                
                # Reduce position sizing
                self.position_size_multiplier = 0.7  # 30% reduction
                
                # Reduce max positions slightly
                self.max_positions_multiplier = 0.7  # 30% fewer positions
                
                # Record action taken
                self.drawdown_actions_taken.add('reduce_risk')
                
                # Alert for minor risk reduction
                return "REDUCE_RISK"
                
        elif drawdown_pct >= self.drawdown_thresholds['warning'] * 100:
            # Warning level - take precautionary measures
            if 'warning' not in self.drawdown_actions_taken:
                logging.warning(f"WARNING DRAWDOWN LEVEL REACHED ({drawdown_pct:.2f}%)!")
                
                # Make minor adjustments to risk parameters
                self.position_size_multiplier = 0.9  # 10% reduction
                
                # Record action taken
                self.drawdown_actions_taken.add('warning')
                
                # Alert for drawdown warning
                return "WARNING"
                
        # Check if we're in recovery mode and have recovered
        self._check_recovery_mode_exit()
        
        return None
    
    def _check_recovery_mode_exit(self):
        """Check if we should exit recovery mode due to equity recovery"""
        if not self.recovery_mode:
            return
            
        try:
            # Only proceed if we have recovery start data
            if not self.recovery_start_equity or not self.recovery_start_time:
                return
                
            # Calculate how much we've recovered
            recovery_pct = ((self.total_equity - self.recovery_start_equity) / 
                          self.recovery_start_equity * 100)
            
            # Get configuration settings for recovery
            recovery_target = getattr(config, 'RECOVERY_PROFIT_TARGET', 0.05) * 100  # Convert to percentage
            recovery_timeout_days = getattr(config, 'RECOVERY_TIMEOUT_DAYS', 14)
            
            # Check if we've recovered enough
            if recovery_pct >= recovery_target:
                if self.severe_recovery_mode:
                    # Exit severe recovery but stay in regular recovery
                    self.severe_recovery_mode = False
                    logging.info(f"Exiting SEVERE recovery mode after {recovery_pct:.2f}% recovery")
                    
                    # Adjust multipliers
                    self.position_size_multiplier = 0.5  # Increase from 0.3 to 0.5
                    self.max_positions_multiplier = 0.5  # Increase from 0.3 to 0.5
                else:
                    # Exit recovery mode completely
                    self.recovery_mode = False
                    self.drawdown_actions_taken.clear()  # Reset actions for next drawdown
                    logging.info(f"Exiting recovery mode after {recovery_pct:.2f}% recovery")
                    
                    # Reset multipliers
                    self.position_size_multiplier = 1.0
                    self.max_positions_multiplier = 1.0
                    
            # Check if recovery is taking too long
            elif (time.time() - self.recovery_start_time) > (recovery_timeout_days * 24 * 3600):
                if self.severe_recovery_mode:
                    # After timeout in severe recovery, step down to regular recovery
                    self.severe_recovery_mode = False
                    logging.info(f"Downgrading from SEVERE to regular recovery mode after {recovery_timeout_days} days")
                    
                    # Adjust multipliers
                    self.position_size_multiplier = 0.5  # Increase from 0.3 to 0.5
                    self.max_positions_multiplier = 0.5  # Increase from 0.3 to 0.5
                    
        except Exception as e:
            logging.error(f"Error checking recovery mode exit: {str(e)}")
    
    def should_take_trade(self, pair, signal_strength, correlation_data=None):
        """Determine if a new trade should be taken based on risk management"""
        # Check if we're in recovery mode with severe restrictions
        if self.severe_recovery_mode and self.position_count >= 3:
            return False, "Maximum positions reached for severe recovery mode"
            
        # Check if we're in recovery mode with moderate restrictions
        if self.recovery_mode and self.position_count >= (5 * self.max_positions_multiplier):
            return False, "Maximum positions reached for recovery mode"
            
        # Check risk level based on drawdown
        if self.risk_level == "minimal" and self.position_count >= 3:
            return False, "Maximum positions reached for minimal risk level"
            
        if self.risk_level == "reduced" and self.position_count >= (7 * self.max_positions_multiplier):
            return False, "Maximum positions reached for reduced risk level"
            
        # Normal max positions check (15 positions by default)
        max_positions = config.MAX_POSITIONS
        if self.position_count >= (max_positions * self.max_positions_multiplier):
            return False, "Maximum positions reached"
            
        # Get correlation data if available
        is_diversified = True
        if correlation_data:
            is_diversified = correlation_data.get('is_diversified', True)
            
        # Skip highly correlated assets in high-risk conditions
        if not is_diversified and (self.recovery_mode or self.risk_level != "normal"):
            return False, "Avoiding correlated assets during elevated risk conditions"
            
        # Calculate position size based on risk level and signal strength
        position_size = self._calculate_position_size(signal_strength)
        
        return True, {
            "approved": True,
            "position_size": position_size,
            "risk_level": self.risk_level,
            "recovery_mode": self.recovery_mode
        }
    
    def _calculate_position_size(self, signal_strength):
        """Calculate position size based on risk level and signal strength"""
        # Base size as percentage of equity (5-10% for $1000 account)
        if signal_strength > 0.8:  # Very strong signal
            base_percent = 0.10  # 10% of equity ($100)
        elif signal_strength > 0.6:  # Strong signal
            base_percent = 0.08  # 8% of equity ($80)
        elif signal_strength > 0.4:  # Moderate signal
            base_percent = 0.06  # 6% of equity ($60)
        else:  # Weaker signal
            base_percent = 0.05  # 5% of equity ($50)
            
        # Calculate base position size
        base_size = self.total_equity * base_percent
        
        # Apply drawdown-based position size multiplier
        position_size = base_size * self.position_size_multiplier
        
        # Ensure minimum viable trade size
        min_trade = 50.0  # $50 minimum
        if position_size < min_trade:
            position_size = min_trade
            
        # Cap at maximum position size (15% of equity)
        max_position = self.total_equity * 0.15  # Max $150 per position
        if position_size > max_position:
            position_size = max_position
            
        return position_size
    
    def add_position(self, pair, position_size):
        """Track a new position"""
        self.position_count += 1
        self.position_values[pair] = position_size
        
    def remove_position(self, pair):
        """Remove a position from tracking"""
        if pair in self.position_values:
            self.position_count -= 1
            del self.position_values[pair]
    
    def should_take_profit(self, pair, current_profit, position_age_hours):
        """Determine if profit should be taken based on targets and position age"""
        # Adjust profit targets based on recovery mode and risk level
        if self.severe_recovery_mode:
            # Take profits more aggressively in severe recovery
            if current_profit >= 1.0:  # 1% or higher is good enough
                return True
                
            if position_age_hours > 12 and current_profit >= 0.5:  # 0.5% after 12 hours
                return True
                
        elif self.recovery_mode:
            # Take profits somewhat aggressively in normal recovery
            if current_profit >= 1.5:  # 1.5% or higher
                return True
                
            if position_age_hours > 24 and current_profit >= 0.8:  # 0.8% after 24 hours
                return True
                
        elif self.risk_level == "reduced":
            # Take profits at moderate thresholds
            if current_profit >= 2.0:  # 2% or higher
                return True
                
            if position_age_hours > 36 and current_profit >= 1.0:  # 1% after 36 hours
                return True
                
        else:
            # Normal profit targets
            if current_profit >= 3.0:  # 3% or higher
                return True
                
            if position_age_hours > 24 and current_profit >= 1.5:  # 1.5% after 24 hours
                return True
                
            if position_age_hours > 48 and current_profit >= 1.0:  # 1% after 48 hours
                return True
                
        return False
    
    def get_max_positions(self):
        """Get maximum allowed positions based on current risk level"""
        base_max = config.MAX_POSITIONS  # Default max positions
        
        # Apply multiplier based on risk level
        return int(base_max * self.max_positions_multiplier)