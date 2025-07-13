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

    def should_take_trade(self, pair, signal_strength, correlation_data=None):
        """Enhanced trade decision with strict position limits"""
        try:
            # CRITICAL: Check if we've reached maximum allocation (5 positions × 20% = 100%)
            if self.position_count >= 5:
                return False, f"Maximum allocation reached (5 positions × 20% = 100% of available cash)"
            
            # Check if we already have this position
            if pair in self.position_values:
                return False, f"Already have position in {pair}"
            
            # Calculate total allocated percentage
            total_allocated = self.position_count * 20  # Each position is 20%
            remaining_allocation = 100 - total_allocated
            
            if remaining_allocation < 20:
                return False, f"Insufficient allocation remaining ({remaining_allocation}% < 20% required)"
            
            logging.info(f"Position check passed: {self.position_count}/5 positions, {total_allocated}% allocated, {remaining_allocation}% remaining")
            
            # Proceed with original checks
            if self.severe_recovery_mode:
                return False, "Severe recovery mode active - no new trades"
            
            if signal_strength < 0.3:
                return False, f"Signal strength too low: {signal_strength:.2f}"
                
            return True, "Trade approved"
            
        except Exception as e:
            logging.error(f"Error in trade decision: {str(e)}")
            return False, f"Error: {str(e)}"

    def can_open_position(self, pair=None, signal_strength=0.5, correlation_data=None):
        """Enhanced position approval with quick mode considerations"""
        
        # Check if we're in quick profit mode
        is_quick_mode = getattr(config, 'QUICK_PROFIT_MODE', False)
        
        if is_quick_mode:
            # Quick mode: stricter position limits
            max_positions = getattr(config, 'QUICK_MODE_MAX_POSITIONS', 2)  # Max 2 positions
            min_signal = getattr(config, 'QUICK_MODE_MIN_SIGNAL', 0.5)
            
            # Check signal strength requirement
            if signal_strength < min_signal:
                return False, f"Quick mode requires signal strength >= {min_signal}, got {signal_strength:.2f}"
            
            # Check position count
            if self.position_count >= max_positions:
                return False, f"Quick mode maximum positions reached ({max_positions})"
                
        else:
            # Normal mode: existing logic
            max_positions = getattr(config, 'NORMAL_MODE_MAX_POSITIONS', 10)
            if self.position_count >= max_positions:
                return False, f"Normal mode maximum positions reached ({max_positions})"
        
        # Continue with existing risk checks...
        if self.severe_recovery_mode:
            return False, "Severe recovery mode active - no new positions"
            
        # Check if we're in recovery mode with moderate restrictions
        if self.recovery_mode and self.position_count >= (5 * self.max_positions_multiplier):
            return False, "Maximum positions reached for recovery mode"
            
        # Check risk level based on drawdown
        if self.risk_level == "minimal" and self.position_count >= 3:
            return False, "Maximum positions reached for minimal risk level"
            
        if self.risk_level == "reduced" and self.position_count >= (7 * self.max_positions_multiplier):
            return False, "Maximum positions reached for reduced risk level"
            
        # Get correlation data if available
        is_diversified = True
        if correlation_data:
            is_diversified = correlation_data.get('is_diversified', True)
            
        # Skip highly correlated assets in high-risk conditions
        if not is_diversified and (self.recovery_mode or self.risk_level != "normal"):
            return False, "Avoiding correlated assets during elevated risk conditions"
            
        # Calculate position size based on mode and risk level
        position_size = self._calculate_position_size(signal_strength)
        
        mode_info = "QUICK" if is_quick_mode else "NORMAL"
        logging.info(f"{mode_info} MODE: Approved position for {pair} - Size: ${position_size:.2f}, Signal: {signal_strength:.2f}")
        
        return True, {
            "approved": True,
            "position_size": position_size,
            "risk_level": self.risk_level,
            "recovery_mode": self.recovery_mode,
            "trading_mode": mode_info
        }
    
    def _calculate_position_size(self, signal_strength, confidence=None, market_regime=None):
        """Calculate position size based on STRICT 20% rule - NO LEVERAGE"""
        
        # FIXED SIZE: Each position is exactly 20% of available cash (base equity)
        # This prevents leverage and ensures bot only trades with available cash
        base_percent = 0.20  # Fixed 20% per position
        base_size = self.total_equity * base_percent
        
        # CRITICAL FIX: Apply drawdown protection multiplier (was missing!)
        # If bot is in risk reduction mode, reduce position sizes accordingly
        base_size = base_size * self.position_size_multiplier
        
        logging.info(f"STRICT SIZING: Using 20% of base equity (${self.total_equity:.2f}) × {self.position_size_multiplier:.1f} multiplier = ${base_size:.2f} per position")
        
        # Apply confidence filter - reject if too low
        if confidence is not None:
            confidence_decimal = confidence / 100 if confidence > 1 else confidence
            min_threshold = getattr(config, 'MIN_SIGNAL_CONFIDENCE', 20) / 100
            
            if confidence_decimal < min_threshold:
                logging.info(f"Rejecting signal - confidence {confidence_decimal:.1%} below {min_threshold:.1%} threshold")
                return 0
        
        # Ensure minimum viable trade size even with multipliers
        min_trade = 50.0  # $50 minimum
        if base_size < min_trade:
            base_size = min_trade
            logging.info(f"Position size adjusted to minimum: ${base_size:.2f} (overriding multiplier for viability)")
            
        # Ensure maximum position size doesn't exceed 20% of original equity (before multiplier)
        max_position = self.total_equity * 0.20  # Strict 20% limit on original equity
        if base_size > max_position:
            base_size = max_position
            logging.info(f"Position size capped at 20% limit: ${base_size:.2f}")
            
        return base_size
    
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
        # Adjust profit targets based on position age and risk level
        return False
    
    def get_status(self):
        """Get current risk management status"""
        current_equity = self.total_equity
        
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
            return None
            
        # Check against thresholds
        if drawdown_pct >= self.drawdown_thresholds['emergency'] * 100:
            # Emergency level - most severe response
            if 'emergency' not in self.drawdown_actions_taken:
                logging.error(f"EMERGENCY DRAWDOWN LEVEL REACHED ({drawdown_pct:.2f}%)!")
                
                # Enter severe recovery mode
                self.severe_recovery_mode = True
                self.recovery_mode = True
                self.recovery_start_equity = self.total_equity
                self.recovery_start_time = time.time()
                
                # Drastically reduce risk
                self.position_size_multiplier = 0.3  # 70% reduction
                self.max_positions_multiplier = 0.3  # 70% fewer positions
                
                # Record action taken
                self.drawdown_actions_taken.add('emergency')
                
                # Alert for emergency drawdown
                return "EMERGENCY"
                
        elif drawdown_pct >= self.drawdown_thresholds['high_alert'] * 100:
            # High alert level - severe response
            if 'high_alert' not in self.drawdown_actions_taken:
                logging.error(f"HIGH ALERT DRAWDOWN LEVEL REACHED ({drawdown_pct:.2f}%)!")
                
                # Enter recovery mode
                self.recovery_mode = True
                self.recovery_start_equity = self.total_equity
                self.recovery_start_time = time.time()
                
                # Significantly reduce risk
                self.position_size_multiplier = 0.5  # 50% reduction
                self.max_positions_multiplier = 0.5  # 50% fewer positions
                
                # Record action taken
                self.drawdown_actions_taken.add('high_alert')
                
                # Alert for high alert
                return "HIGH_ALERT"
                
        elif drawdown_pct >= self.drawdown_thresholds['reduce_risk'] * 100:
            # Risk reduction level - moderate response
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
                    self.position_size_multiplier = 0.7
                    self.max_positions_multiplier = 0.7
                else:
                    # Exit recovery mode completely
                    self.recovery_mode = False
                    logging.info(f"Exiting recovery mode after {recovery_pct:.2f}% recovery")
                    
                    # Reset multipliers
                    self.position_size_multiplier = 1.0
                    self.max_positions_multiplier = 1.0
                    
                # Clear actions taken to allow re-triggering if needed
                self.drawdown_actions_taken.clear()
                
            # Check for timeout
            time_in_recovery = time.time() - self.recovery_start_time
            if time_in_recovery > (recovery_timeout_days * 24 * 3600):
                logging.warning(f"Recovery mode timeout reached after {recovery_timeout_days} days")
                self.recovery_mode = False
                self.severe_recovery_mode = False
                
                # Reset multipliers gradually
                self.position_size_multiplier = 0.8
                self.max_positions_multiplier = 0.8
                
                # Clear actions taken
                self.drawdown_actions_taken.clear()
                
        except Exception as e:
            logging.error(f"Error in recovery mode exit check: {str(e)}")
    
    def reset_to_normal_trading(self):
        """Reset risk manager to normal trading mode (like successful bot)"""
        self.position_size_multiplier = 1.0
        self.max_positions_multiplier = 1.0
        self.recovery_mode = False
        self.severe_recovery_mode = False
        self.drawdown_actions_taken.clear()
        logging.info("RESET: Risk manager reset to normal trading mode - full $200 positions enabled")
    
    def reset_daily_targets(self):
        """Reset daily tracking for new day"""
        self.day_start_time = time.time()
        self.day_profit_target_reached = False
        self.daily_high_equity = self.total_equity
        logging.info("Daily targets reset for new trading day")