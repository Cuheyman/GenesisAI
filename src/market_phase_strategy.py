import logging
import asyncio
import time
from typing import Dict, List, Any, Optional, Tuple
import config

class MarketPhaseStrategyHandler:
    """
    Handles strategy execution based on market phases from the enhanced API
    Implements different execution strategies for ACCUMULATION, DISTRIBUTION, MARKUP, MARKDOWN, and CONSOLIDATION phases
    """
    
    def __init__(self, bot_instance):
        self.bot = bot_instance
        self.execution_methods = {
            'ACCUMULATION': self.execute_accumulation_strategy,
            'DISTRIBUTION': self.execute_distribution_strategy,
            'MARKUP': self.execute_markup_strategy,
            'MARKDOWN': self.execute_markdown_strategy,
            'CONSOLIDATION': self.execute_consolidation_strategy
        }
        
        # Strategy configuration
        self.accumulation_orders = 3  # Number of orders for accumulation
        self.accumulation_intervals = [0, 15*60, 30*60]  # Intervals in seconds
        self.accumulation_price_offsets = [0, -0.01, -0.02]  # Price offsets for each order
        
        logging.info("Market Phase Strategy Handler initialized")
    
    async def execute_phase_strategy(self, pair: str, signal: Dict[str, Any]) -> bool:
        """
        Execute strategy based on market phase
        
        Args:
            pair: Trading pair symbol
            signal: Enhanced signal with market phase data
            
        Returns:
            bool: True if strategy executed successfully
        """
        try:
            api_data = signal.get('api_data', {})
            market_phase = api_data.get('market_phase', 'CONSOLIDATION')
            signal_action = api_data.get('signal', 'HOLD')
            confidence = api_data.get('confidence', 50)
            strategy_type = api_data.get('strategy_type', 'Standard Entry')
            
            logging.info(f"Executing {market_phase} strategy for {pair}: {signal_action} "
                        f"(confidence: {confidence}%, strategy: {strategy_type})")
            
            # Get the appropriate strategy handler
            strategy_handler = self.execution_methods.get(market_phase, self.execute_standard_strategy)
            
            # Execute the strategy
            result = await strategy_handler(pair, signal)
            
            if result:
                logging.info(f"{market_phase} strategy executed successfully for {pair}")
            else:
                logging.warning(f"{market_phase} strategy execution failed for {pair}")
            
            return result
            
        except Exception as e:
            logging.error(f"Error executing phase strategy for {pair}: {str(e)}")
            return False
    
    async def execute_accumulation_strategy(self, pair: str, signal: Dict[str, Any]) -> bool:
        """
        ACCUMULATION Phase: Gradual position building with scaled entries
        - Low volatility, building positions
        - Split orders over time with small price differences
        """
        try:
            api_data = signal.get('api_data', {})
            signal_action = api_data.get('signal', 'HOLD')
            
            if signal_action != 'BUY':
                logging.info(f"ACCUMULATION: {pair} signal is {signal_action}, no action needed")
                return True
            
            # Get position sizing
            total_position_size = api_data.get('position_size_percent', 3.5)
            entry_price = api_data.get('entry_price', 0)
            
            if not entry_price:
                entry_price = await self.bot.get_current_price(pair)
            
            # Split into multiple orders (40%, 35%, 25%)
            order_sizes = [0.4, 0.35, 0.25]
            
            logging.info(f"ACCUMULATION: Executing scaled entry for {pair} with {len(order_sizes)} orders")
            
            success_count = 0
            for i, size_ratio in enumerate(order_sizes):
                try:
                    # Calculate order details
                    order_size = total_position_size * size_ratio
                    order_price = entry_price * (1 + self.accumulation_price_offsets[i])
                    delay = self.accumulation_intervals[i]
                    
                    # Wait for the specified delay
                    if delay > 0:
                        logging.info(f"ACCUMULATION: Waiting {delay}s before order {i+1}")
                        await asyncio.sleep(delay)
                    
                    # Create modified signal for this order
                    order_signal = signal.copy()
                    order_signal['api_data'] = api_data.copy()
                    order_signal['api_data']['position_size_percent'] = order_size
                    order_signal['api_data']['entry_price'] = order_price
                    order_signal['signal_strength'] = signal.get('signal_strength', 0.5)
                    
                    # Execute the order
                    logging.info(f"ACCUMULATION: Executing order {i+1}: {order_size:.1f}% at ${order_price:.4f}")
                    
                    if config.TEST_MODE:
                        # In test mode, execute through the bot's buy method
                        result = await self.bot._execute_api_enhanced_buy(pair, order_signal)
                    else:
                        # In live mode, use limit orders
                        result = await self.execute_limit_order(pair, order_signal, order_price)
                    
                    if result:
                        success_count += 1
                        logging.info(f"ACCUMULATION order {i+1} executed successfully")
                    else:
                        logging.warning(f"ACCUMULATION order {i+1} failed")
                        
                except Exception as e:
                    logging.error(f"Error executing accumulation order {i+1} for {pair}: {str(e)}")
            
            # Consider successful if at least one order executed
            return success_count > 0
            
        except Exception as e:
            logging.error(f"Error in accumulation strategy for {pair}: {str(e)}")
            return False
    
    async def execute_distribution_strategy(self, pair: str, signal: Dict[str, Any]) -> bool:
        """
        DISTRIBUTION Phase: Risk-off positioning, profit taking
        - High volatility, decreasing volume, topping patterns
        - Quick exits or short opportunities
        """
        try:
            api_data = signal.get('api_data', {})
            signal_action = api_data.get('signal', 'HOLD')
            confidence = api_data.get('confidence', 50)
            
            logging.info(f"DISTRIBUTION: Processing {signal_action} signal for {pair} (confidence: {confidence}%)")
            
            if signal_action == 'SELL':
                # Quick exit or short opportunity
                if confidence > 60:
                    logging.info(f"DISTRIBUTION: High confidence SELL - executing immediate order for {pair}")
                    return await self.bot._execute_api_enhanced_buy(pair, signal) if signal.get('sell_signal') else False
                else:
                    logging.info(f"DISTRIBUTION: Moderate confidence SELL - reducing existing positions for {pair}")
                    return await self.reduce_existing_positions(pair, 0.5)  # Reduce by 50%
            
            elif signal_action == 'HOLD':
                # Defensive positioning - reduce exposure
                logging.info(f"DISTRIBUTION: HOLD signal - implementing defensive positioning for {pair}")
                return await self.reduce_existing_positions(pair, 0.3)  # Reduce by 30%
            
            else:
                logging.info(f"DISTRIBUTION: {signal_action} signal not suitable for distribution phase")
                return True
            
        except Exception as e:
            logging.error(f"Error in distribution strategy for {pair}: {str(e)}")
            return False
    
    async def execute_markup_strategy(self, pair: str, signal: Dict[str, Any]) -> bool:
        """
        MARKUP Phase: Trend following with momentum confirmation
        - Rising prices, strong momentum
        - Immediate entries for strong signals, pullback entries for weaker ones
        """
        try:
            api_data = signal.get('api_data', {})
            signal_action = api_data.get('signal', 'HOLD')
            confidence = api_data.get('confidence', 50)
            
            if signal_action != 'BUY':
                if signal_action == 'HOLD':
                    # Trail existing positions
                    return await self.update_trailing_stops(pair)
                else:
                    logging.info(f"MARKUP: {pair} signal is {signal_action}, no action needed")
                    return True
            
            logging.info(f"MARKUP: Processing BUY signal for {pair} (confidence: {confidence}%)")
            
            if confidence > 70:
                # Strong signal - immediate market entry
                logging.info(f"MARKUP: High confidence - executing immediate entry for {pair}")
                return await self.bot._execute_api_enhanced_buy(pair, signal)
            else:
                # Moderate signal - wait for pullback
                logging.info(f"MARKUP: Moderate confidence - executing pullback entry for {pair}")
                return await self.execute_pullback_entry(pair, signal)
            
        except Exception as e:
            logging.error(f"Error in markup strategy for {pair}: {str(e)}")
            return False
    
    async def execute_markdown_strategy(self, pair: str, signal: Dict[str, Any]) -> bool:
        """
        MARKDOWN Phase: Short opportunities or defensive holds
        - Falling prices, bearish momentum
        - Short opportunities or oversold bounce plays
        """
        try:
            api_data = signal.get('api_data', {})
            signal_action = api_data.get('signal', 'HOLD')
            confidence = api_data.get('confidence', 50)
            
            logging.info(f"MARKDOWN: Processing {signal_action} signal for {pair} (confidence: {confidence}%)")
            
            if signal_action == 'SELL':
                # Short opportunity or exit longs
                if confidence > 65:
                    logging.info(f"MARKDOWN: High confidence SELL - executing short strategy for {pair}")
                    return await self.execute_short_strategy(pair, signal)
                else:
                    logging.info(f"MARKDOWN: Moderate confidence SELL - exiting longs for {pair}")
                    return await self.exit_long_positions(pair)
            
            elif signal_action == 'BUY':
                # Oversold bounce opportunity
                if confidence > 70:
                    logging.info(f"MARKDOWN: High confidence BUY - executing contrarian entry for {pair}")
                    return await self.execute_contrarian_entry(pair, signal)
                else:
                    logging.info(f"MARKDOWN: Low confidence BUY in markdown phase - skipping")
                    return True
            
            else:
                # Hold - defensive positioning
                logging.info(f"MARKDOWN: HOLD signal - maintaining defensive stance for {pair}")
                return True
            
        except Exception as e:
            logging.error(f"Error in markdown strategy for {pair}: {str(e)}")
            return False
    
    async def execute_consolidation_strategy(self, pair: str, signal: Dict[str, Any]) -> bool:
        """
        CONSOLIDATION Phase: Range trading strategy
        - Sideways movement, range trading
        - Buy at support, sell at resistance, wait for breakouts
        """
        try:
            api_data = signal.get('api_data', {})
            signal_action = api_data.get('signal', 'HOLD')
            confidence = api_data.get('confidence', 50)
            
            logging.info(f"CONSOLIDATION: Processing {signal_action} signal for {pair} (confidence: {confidence}%)")
            
            if signal_action == 'BUY':
                # Buy at support with limit order
                logging.info(f"CONSOLIDATION: BUY signal - executing support level entry for {pair}")
                entry_price = api_data.get('entry_price', 0)
                if not entry_price:
                    entry_price = await self.bot.get_current_price(pair)
                
                # Place limit order slightly below current price
                limit_price = entry_price * 0.999  # 0.1% below
                return await self.execute_limit_order(pair, signal, limit_price)
            
            elif signal_action == 'SELL':
                # Sell at resistance
                logging.info(f"CONSOLIDATION: SELL signal - executing resistance level exit for {pair}")
                return await self.bot.execute_sell(pair, {"sell_signal": True, "signal_strength": 0.8})
            
            elif signal_action == 'HOLD':
                # Wait for breakout - setup breakout orders
                logging.info(f"CONSOLIDATION: HOLD signal - setting up breakout monitoring for {pair}")
                return await self.setup_breakout_orders(pair, signal)
            
            else:
                return True
            
        except Exception as e:
            logging.error(f"Error in consolidation strategy for {pair}: {str(e)}")
            return False
    
    async def execute_standard_strategy(self, pair: str, signal: Dict[str, Any]) -> bool:
        """
        Standard strategy for unknown or undefined market phases
        """
        try:
            logging.info(f"STANDARD: Executing standard strategy for {pair}")
            
            if signal.get('buy_signal'):
                return await self.bot._execute_api_enhanced_buy(pair, signal)
            elif signal.get('sell_signal'):
                return await self.bot.execute_sell(pair, signal)
            else:
                return True
                
        except Exception as e:
            logging.error(f"Error in standard strategy for {pair}: {str(e)}")
            return False
    
    # Helper methods for specific execution types
    
    async def execute_limit_order(self, pair: str, signal: Dict[str, Any], limit_price: float) -> bool:
        """Execute a limit order at specified price"""
        try:
            # Modify signal for limit order execution
            modified_signal = signal.copy()
            if 'api_data' in modified_signal:
                modified_signal['api_data']['entry_price'] = limit_price
            
            # For now, execute as market order in test mode
            # In live mode, this would place actual limit orders
            return await self.bot._execute_api_enhanced_buy(pair, modified_signal)
            
        except Exception as e:
            logging.error(f"Error executing limit order for {pair}: {str(e)}")
            return False
    
    async def execute_pullback_entry(self, pair: str, signal: Dict[str, Any]) -> bool:
        """Execute entry on pullback (0.5% below current price)"""
        try:
            current_price = await self.bot.get_current_price(pair)
            pullback_price = current_price * 0.995  # 0.5% below current
            
            return await self.execute_limit_order(pair, signal, pullback_price)
            
        except Exception as e:
            logging.error(f"Error executing pullback entry for {pair}: {str(e)}")
            return False
    
    async def execute_contrarian_entry(self, pair: str, signal: Dict[str, Any]) -> bool:
        """Execute contrarian entry with smaller position size"""
        try:
            # Reduce position size for contrarian plays
            modified_signal = signal.copy()
            if 'api_data' in modified_signal:
                original_size = modified_signal['api_data'].get('position_size_percent', 3.5)
                modified_signal['api_data']['position_size_percent'] = original_size * 0.5  # 50% of normal size
            
            return await self.bot._execute_api_enhanced_buy(pair, modified_signal)
            
        except Exception as e:
            logging.error(f"Error executing contrarian entry for {pair}: {str(e)}")
            return False
    
    async def execute_short_strategy(self, pair: str, signal: Dict[str, Any]) -> bool:
        """Execute short strategy (for now, just exit longs)"""
        try:
            # For now, just exit existing long positions
            # In future, could implement actual short selling
            return await self.exit_long_positions(pair)
            
        except Exception as e:
            logging.error(f"Error executing short strategy for {pair}: {str(e)}")
            return False
    
    async def reduce_existing_positions(self, pair: str, reduction_ratio: float) -> bool:
        """Reduce existing positions by specified ratio"""
        try:
            if pair in self.bot.active_positions:
                position = self.bot.active_positions[pair]
                current_quantity = position['quantity']
                reduction_quantity = current_quantity * reduction_ratio
                
                logging.info(f"Reducing {pair} position by {reduction_ratio*100:.0f}% "
                           f"({reduction_quantity:.6f} of {current_quantity:.6f})")
                
                return await self.bot.execute_partial_sell(pair, reduction_quantity, 
                                                         f"risk_reduction_{reduction_ratio*100:.0f}%")
            else:
                logging.info(f"No existing position to reduce for {pair}")
                return True
                
        except Exception as e:
            logging.error(f"Error reducing positions for {pair}: {str(e)}")
            return False
    
    async def exit_long_positions(self, pair: str) -> bool:
        """Exit all long positions for the pair"""
        try:
            if pair in self.bot.active_positions:
                logging.info(f"Exiting long position for {pair}")
                return await self.bot.execute_sell(pair, {"sell_signal": True, "signal_strength": 0.9})
            else:
                logging.info(f"No long position to exit for {pair}")
                return True
                
        except Exception as e:
            logging.error(f"Error exiting long positions for {pair}: {str(e)}")
            return False
    
    async def update_trailing_stops(self, pair: str) -> bool:
        """Update trailing stops for existing positions"""
        try:
            if pair in self.bot.active_positions:
                current_price = await self.bot.get_current_price(pair)
                if current_price:
                    await self.bot.update_trailing_stop(pair, current_price)
                    logging.info(f"Updated trailing stop for {pair}")
                    return True
            else:
                logging.info(f"No position to update trailing stop for {pair}")
                return True
                
        except Exception as e:
            logging.error(f"Error updating trailing stops for {pair}: {str(e)}")
            return False
    
    async def setup_breakout_orders(self, pair: str, signal: Dict[str, Any]) -> bool:
        """Setup breakout monitoring (placeholder for future implementation)"""
        try:
            logging.info(f"Breakout monitoring setup for {pair} (not yet implemented)")
            # Placeholder for future breakout order implementation
            return True
            
        except Exception as e:
            logging.error(f"Error setting up breakout orders for {pair}: {str(e)}")
            return False
    
    def get_strategy_summary(self) -> Dict[str, Any]:
        """Get summary of strategy handler status"""
        return {
            "available_phases": list(self.execution_methods.keys()),
            "accumulation_config": {
                "orders": self.accumulation_orders,
                "intervals": self.accumulation_intervals,
                "price_offsets": self.accumulation_price_offsets
            }
        } 


        