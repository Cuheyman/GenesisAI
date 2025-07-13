import asyncio
import logging
import time
from datetime import datetime, timedelta
import os
import json
import random
import traceback
from binance.client import Client
import numpy as np
import pandas as pd

from market_analysis import MarketAnalysis
from order_book import OrderBookAnalysis
from db_manager import DatabaseManager
from asset_selection import AssetSelection, DynamicAssetSelection
from correlation_analysis import CorrelationAnalysis
from enhanced_strategy import EnhancedStrategy
from risk_manager import RiskManager
from performance_tracker import PerformanceTracker
from opportunity_scanner import OpportunityScanner
from enhanced_strategy_api import EnhancedSignalAPIClient
from market_phase_strategy import MarketPhaseStrategyHandler
# APIEnhancedStrategy removed - using EnhancedStrategy for API-only mode
import config


import config

class HybridTradingBot:
    def __init__(self):
        """Initialize the hybrid trading bot with API-only components"""
        # Set up logging
        self._setup_logging()
        
        # Initialize Binance client with timestamp offset handling
        self.binance_client = Client(
            config.BINANCE_API_KEY, 
            config.BINANCE_API_SECRET,
            testnet=config.TEST_MODE
        )
        
        # Fix timestamp synchronization issue
        try:
            # Get server time to calculate offset
            server_time = self.binance_client.get_server_time()
            local_time = int(time.time() * 1000)
            time_offset = server_time['serverTime'] - local_time
            
            # Set timestamp offset if needed (>500ms difference)
            if abs(time_offset) > 500:
                logging.info(f"Detected time offset: {time_offset}ms, adjusting...")
                # Recreate client with timestamp offset
                self.binance_client = Client(
                    config.BINANCE_API_KEY, 
                    config.BINANCE_API_SECRET,
                    testnet=config.TEST_MODE
                )
                # Set the timestamp offset
                self.binance_client.timestamp_offset = time_offset
                logging.info(f"Timestamp offset set to {time_offset}ms")
            else:
                logging.info("System time is synchronized with Binance servers")
        except Exception as e:
            logging.warning(f"Could not sync timestamp: {e}. Trying manual offset...")
            # Apply a small negative offset to ensure we're not ahead
            self.binance_client.timestamp_offset = -2000  # 2 seconds behind
            logging.info("Applied manual timestamp offset: -2000ms")

        # Initialize database manager
        self.db_manager = DatabaseManager(config.DB_PATH)
        
        # Update database schema
        self.db_manager.update_schema()
        
        # Initialize market analysis module
        self.market_analysis = MarketAnalysis(self.binance_client)
        
        # Initialize order book analysis
        self.order_book = OrderBookAnalysis(self.binance_client)
        
        # Initialize asset selection with dynamic capabilities
        self.asset_selection = DynamicAssetSelection(self.binance_client, self.market_analysis)
        
        # Enable symbol validation if configured
        if getattr(config, 'SYMBOL_VALIDATION_ENABLED', True):
            self.asset_selection.validate_symbols = True
            logging.info("Symbol validation enabled for asset selection")
        else:
            self.asset_selection.validate_symbols = False
            logging.info("Symbol validation disabled")
        
        # Initialize correlation analysis
        self.correlation = CorrelationAnalysis(self.market_analysis)
        
        # Get initial equity
        self.initial_equity = self.get_total_equity()
        
        # API-ENHANCED STRATEGY INITIALIZATION:
        self.strategy = EnhancedStrategy(
            self.binance_client, 
            None,  # No AI client needed for API-only mode
            self.market_analysis, 
            self.order_book
        )
        
        # Initialize API strategy
        self.api_strategy_initialized = False
        
        # Initialize global API client for rate limiting consistency
        self.global_api_client = None
        
        # API monitoring
        self.api_health_checks = 0
        self.api_failures = 0
        self.last_api_health_check = 0
        
        # Initialize opportunity scanner (simplified)
        self.opportunity_scanner = OpportunityScanner(self.binance_client, None)
        
        # Quick trade settings for momentum catching
        self.quick_trade_mode = True
        self.momentum_trades = {}  # Track momentum-based trades

        # Initialize risk manager
        self.risk_manager = RiskManager(self.initial_equity)
        
        # CRITICAL: Reset to normal trading mode (like successful bot)
        self.risk_manager.reset_to_normal_trading()
        
        # Initialize performance tracker
        self.performance_tracker = PerformanceTracker(config.DB_PATH)
        
        # Initialize market phase strategy handler
        self.phase_strategy = MarketPhaseStrategyHandler(self)
        
        # Trading state
        self.active_positions = {}
        self.pending_orders = {}
        self.analyzed_pairs = {}
        self.trailing_stops = {}
        self.trailing_tp_data = {}
        self.position_scales = {}
        
        # Market state tracking
        self.market_state = {
            'breadth': 0.5,  # Default to neutral (0-1 scale)
            'volatility': 'normal',  # high, normal, low
            'trend': 'neutral',  # bullish, neutral, bearish
            'liquidity': 'normal',  # high, normal, low
            'sentiment': 'neutral',  # bullish, neutral, bearish
            'regime': 'NEUTRAL'  # BULL_TRENDING, BULL_VOLATILE, BEAR_TRENDING, BEAR_VOLATILE, NEUTRAL
        }
        self.recently_traded = {}
        
        # Initialize session for API calls
        self.session = None
        
        # API call rate limit management
        self.api_call_tracker = {}
        self.api_semaphore = asyncio.Semaphore(10)  # Limit concurrent API calls
        
        # Initialize statistics tracking
        self.api_stats = {
            'total_analyses': 0,
            'api_signals_used': 0,
            'fallback_used': 0,
            'api_success_rate': 0.0,
            'api_cache_hits': 0,
            'api_available': True,
            'api_errors': 0,
            'successful_api_calls': 0
        }
        
        logging.info("Hybrid Trading Bot initialized - API-Only Mode")
        logging.info(f"TEST_MODE: {config.TEST_MODE}")
        logging.info(f"API URL: {getattr(config, 'SIGNAL_API_URL', 'not configured')}")
        logging.info(f"API enabled: {getattr(config, 'ENABLE_ENHANCED_API', False)}")
        logging.info(f"Mode: {'TEST' if config.TEST_MODE else 'LIVE'} trading")
        logging.info(f"Initial equity: ${self.initial_equity:.2f}")
        logging.info("External APIs: Disabled (Nebula, CoinGecko, Taapi.io removed)")
        logging.info("Signal Source: Enhanced Signal API only")
    
    def _setup_logging(self):
        """Set up logging configuration"""
        log_dir = 'logs'
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        log_file = os.path.join(log_dir, f'hybrid_bot_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        # Create separate trade signal logger
        self.trade_logger = logging.getLogger('trade_signals')
        self.trade_logger.setLevel(logging.INFO)
        trade_log_file = os.path.join(log_dir, f'trades_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        self.trade_logger.addHandler(logging.FileHandler(trade_log_file))
        self.trade_logger.propagate = False  # Prevent duplicate logging
    

    def get_total_equity(self):
        """Get total account equity"""
        logging.info(f"Getting total equity - TEST_MODE: {config.TEST_MODE}")
        
        if config.TEST_MODE:
            try:
                # Get equity from database in test mode
                equity_query = "SELECT value FROM bot_stats WHERE key='total_equity' LIMIT 1"
                result = self.db_manager.execute_query_sync(equity_query, fetch_one=True)
                
                if result:
                    equity = float(result[0])
                    logging.info(f"Test mode: Retrieved equity from database: ${equity:.2f}")
                    return equity
                else:
                    # Set initial equity if not found
                    self.db_manager.execute_query_sync(
                        "INSERT INTO bot_stats (key, value, last_updated) VALUES (?, ?, ?)",
                        ('total_equity', 1000.0, datetime.now().strftime('%Y-%m-%d %H:%M:%S')),
                        commit=True
                    )
                    logging.info("Test mode: Set initial equity to $1000.00")
                    return 1000.0  # Default initial equity
            except Exception as e:
                logging.warning(f"Error accessing database in test mode: {e}")
                logging.info("Test mode: Using fallback equity of $1000.00")
                return 1000.0  # Fallback for test mode
        
        # Use real account data for live mode
        logging.info("Live mode: Getting real account data from Binance")
        account = self.binance_client.get_account()
        
        total = 0
        for balance in account['balances']:
            asset = balance['asset']
            try:
                free_amount = float(balance['free']) if balance['free'] else 0.0
                locked_amount = float(balance['locked']) if balance['locked'] else 0.0
                total_amount = free_amount + locked_amount
            except (ValueError, TypeError) as e:
                logging.debug(f"Could not convert balance for {asset}: free={balance['free']}, locked={balance['locked']}")
                continue
            
            if total_amount > 0:
                if asset == 'USDT':
                    total += total_amount
                else:
                    # Get asset price in USDT
                    try:
                        ticker = self.binance_client.get_symbol_ticker(symbol=f"{asset}USDT")
                        price = float(ticker['price']) if ticker['price'] else 0.0
                        total += total_amount * price
                    except (ValueError, TypeError, Exception) as e:
                        # Skip if cannot get price or convert
                        logging.debug(f"Could not get price for {asset}USDT: {str(e)}")
                        pass
        
                return total

    async def get_real_time_equity(self):
        """Get real-time equity including current position values and unrealized P&L"""
        try:
            # Start with base equity (cash + non-position assets)
            base_equity = self.get_total_equity()
            
            if not self.active_positions:
                return base_equity
            
            # Calculate current value of all positions
            total_position_value = 0
            total_unrealized_pnl = 0
            
            for pair, position in self.active_positions.items():
                try:
                    # Get current price
                    current_price = await self.get_current_price(pair)
                    if not current_price:
                        logging.warning(f"Could not get current price for {pair}, using entry price")
                        current_price = position['entry_price']
                    
                    # Calculate current position value
                    current_value = position['quantity'] * current_price
                    initial_value = position['quantity'] * position['entry_price']
                    unrealized_pnl = current_value - initial_value
                    
                    total_position_value += current_value
                    total_unrealized_pnl += unrealized_pnl
                    
                    logging.debug(f"{pair}: Current value ${current_value:.2f}, "
                                f"Initial ${initial_value:.2f}, P&L ${unrealized_pnl:.2f}")
                    
                except Exception as e:
                    logging.error(f"Error calculating position value for {pair}: {str(e)}")
                    # Fallback to entry price if current price unavailable
                    fallback_value = position['quantity'] * position['entry_price']
                    total_position_value += fallback_value
            
            # Real-time equity = base equity + unrealized P&L
            # Note: In test mode, base_equity includes cash, position values are additional
            if config.TEST_MODE:
                # In test mode, we need to be more careful about double counting
                # The base equity from database should already account for position costs
                real_time_equity = base_equity + total_unrealized_pnl
            else:
                # In live mode, base equity is account balance, add position values
                real_time_equity = base_equity + total_unrealized_pnl
            
            logging.info(f"Real-time equity: ${real_time_equity:.2f} "
                        f"(Base: ${base_equity:.2f}, Position Value: ${total_position_value:.2f}, "
                        f"Unrealized P&L: ${total_unrealized_pnl:+.2f})")
            
            return real_time_equity
            
        except Exception as e:
            logging.error(f"Error calculating real-time equity: {str(e)}")
            # Fallback to base equity
            return self.get_total_equity()

    async def update_equity_in_db(self, new_equity):
        """Update equity in database for test mode"""
        if config.TEST_MODE:
            try:
                self.db_manager.execute_query_sync(
                    "UPDATE bot_stats SET value = ?, last_updated = ? WHERE key = 'total_equity'",
                    (new_equity, datetime.now().strftime('%Y-%m-%d %H:%M:%S')),
                    commit=True
                )
                logging.debug(f"Updated equity in database: ${new_equity:.2f}")
            except Exception as e:
                logging.error(f"Error updating equity in database: {str(e)}")

    async def run(self):
        try:
            # Test API connections
            logging.info("Testing API connections...")
            
            logging.info("Initializing Enhanced Signal API...")
            # Initialize global API client
            from enhanced_strategy_api import EnhancedSignalAPIClient
            self.global_api_client = EnhancedSignalAPIClient()
            await self.global_api_client.initialize()
            self.api_strategy_initialized = True
            
            if self.api_strategy_initialized:
                logging.info("Enhanced Signal API initialized successfully")
                logging.info("Using AI-powered signals with on-chain analysis")
            else:
                logging.warning("API initialization failed, using fallback mode")

            # Test Binance connection
            server_time = self.binance_client.get_server_time()
            logging.info(f"Binance server time: {datetime.fromtimestamp(server_time['serverTime']/1000)}")
            
            # API-only mode - no external API tests needed
            logging.info("API-only mode - skipping external API tests")
            
            # Preload historical data for analysis
            logging.info("Preloading historical data for analysis...")
            await self.preload_historical_data()
            logging.info("Historical data preloading complete")
            
            # Create tasks for different components
            tasks = [
                self.market_monitor_task(),
                self.trading_task(),
                self.position_monitor_task(),
                self.performance_track_task(),
                
            ]
            
            # Run all tasks concurrently
            await asyncio.gather(*tasks)
            
        except Exception as e:
            logging.error(f"Critical error running bot: {str(e)}")
            traceback.print_exc()
            raise


    async def _monitor_api_health(self):
        """Monitor API health and log status"""
        current_time = time.time()
        
        # Check every 5 minutes
        if current_time - self.last_api_health_check > 300:
            self.api_health_checks += 1
            
            if hasattr(self.strategy, 'api_client'):
                api_available = await self.strategy.api_client.check_api_health()
                
                if not api_available:
                    self.api_failures += 1
                    logging.warning(f"API health check failed ({self.api_failures} total failures)")
                else:
                    logging.debug("API health check passed")
            
            self.last_api_health_check = current_time

    async def _log_api_statistics(self):
        """Log comprehensive API usage statistics"""
        try:
            # Calculate success rate
            total_calls = self.api_stats['successful_api_calls'] + self.api_stats['api_errors']
            success_rate = (self.api_stats['successful_api_calls'] / total_calls * 100) if total_calls > 0 else 0.0
            
            # Calculate usage percentages
            total_analyses = self.api_stats['total_analyses']
            api_used_pct = (self.api_stats['api_signals_used'] / total_analyses * 100) if total_analyses > 0 else 0.0
            fallback_pct = (self.api_stats['fallback_used'] / total_analyses * 100) if total_analyses > 0 else 0.0
            
            # Get actual cache hits from API client
            actual_cache_hits = 0
            if hasattr(self, 'global_api_client') and self.global_api_client:
                actual_cache_hits = getattr(self.global_api_client, 'cache_hits', 0)
            
            logging.info("API STRATEGY STATISTICS:")
            logging.info(f"  • Total Analyses: {self.api_stats['total_analyses']}")
            logging.info(f"  • API Signals Used: {self.api_stats['api_signals_used']} ({api_used_pct:.1f}%)")
            logging.info(f"  • Fallback Used: {self.api_stats['fallback_used']} ({fallback_pct:.1f}%)")
            logging.info(f"  • API Success Rate: {success_rate:.1f}%")
            logging.info(f"  • API Cache Hits: {actual_cache_hits}")
            logging.info(f"  • API Available: {self.api_stats['api_available']}")
                
        except Exception as e:
            logging.error(f"Error logging API statistics: {str(e)}")

    def _get_correlation_data(self, pair: str) -> dict[str, any]:
        """Get correlation data for pair"""
        try:
            active_positions = list(self.active_positions.keys())
            return {
                'portfolio_correlation': self.correlation.get_portfolio_correlation(pair, active_positions),
                'is_diversified': self.correlation.are_pairs_diversified(pair, active_positions)
            }
        except Exception as e:
            logging.debug(f"Error getting correlation data: {str(e)}")
            return {'portfolio_correlation': 0, 'is_diversified': True}

    async def _set_api_stop_loss(self, pair: str, entry_price: float, stop_loss_price: float):
        """Set stop loss based on API recommendation"""
        try:
            if not hasattr(self, 'api_stop_losses'):
                self.api_stop_losses = {}
            
            self.api_stop_losses[pair] = {
                'entry_price': entry_price,
                'stop_loss_price': stop_loss_price,
                'set_time': time.time(),
                'source': 'api_recommendation'
            }
            
            logging.info(f"API Stop Loss set for {pair}: ${stop_loss_price:.4f}")
            
        except Exception as e:
            logging.error(f"Error setting API stop loss: {str(e)}")

    async def _set_api_take_profits(self, pair: str, entry_price: float, exit_levels: dict[str, any]):
        """Set take profit levels based on API recommendations"""
        try:
            if not hasattr(self, 'api_take_profits'):
                self.api_take_profits = {}
            
            take_profits = []
            for i in range(1, 4):
                tp_key = f'take_profit_{i}'
                if tp_key in exit_levels and exit_levels[tp_key]:
                    take_profits.append({
                        'level': i,
                        'price': exit_levels[tp_key],
                        'percentage': 33.33 if i < 3 else 34  # Split position 33/33/34
                    })
            
            if take_profits:
                self.api_take_profits[pair] = {
                    'entry_price': entry_price,
                    'take_profits': take_profits,
                    'set_time': time.time(),
                    'source': 'api_recommendation'
                }
                
                tp_prices = [f"${tp['price']:.4f}" for tp in take_profits]
                logging.info(f"API Take Profits set for {pair}: {', '.join(tp_prices)}")
                
        except Exception as e:
            logging.error(f"Error setting API take profits: {str(e)}")
    
    async def preload_historical_data(self):
        """Preload sufficient historical data for all needed timeframes"""
        logging.info("Preloading historical data for analysis...")

        # Get pairs to preload using dynamic selection
        pairs = await self.asset_selection.get_optimal_trading_pairs(max_pairs=15)
        # Always include major pairs as fallback
        major_pairs = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'ADAUSDT']
        for major in major_pairs:
            if major not in pairs:
                pairs.append(major)
        pairs = list(set(pairs))  # Remove duplicates

        # Timeframes to load
        timeframes = ['5m', '15m', '1h', '4h', '1d']

        # Days of history needed
        days_of_history = {
            '5m': 5,    # 5 days of 5-minute data
            '15m': 10,  # 10 days of 15-minute data
            '1h': 30,   # 30 days of hourly data
            '4h': 60,   # 60 days of 4-hour data
            '1d': 200   # 200 days of daily data
        }

        # Process pairs in smaller batches to avoid rate limits
        chunk_size = 5  # Process 5 pairs at a time
        for i in range(0, len(pairs), chunk_size):
            pair_chunk = pairs[i:i+chunk_size]
            logging.info(f"Preloading data for pairs: {', '.join(pair_chunk)}")
            
            for tf in timeframes:
                for pair in pair_chunk:
                    try:
                        await self.market_analysis.get_klines(
                            pair, 
                            int(time.time() * 1000) - (days_of_history[tf] * 24 * 60 * 60 * 1000),
                            interval=tf
                        )
                    except Exception as e:
                        logging.warning(f"Error preloading {tf} data for {pair}: {str(e)}")
                
                # Pause between timeframes to avoid rate limits
                await asyncio.sleep(1)
            
            # Pause between batches to avoid rate limits
            await asyncio.sleep(2)
    
    # Add these methods to your hybrid_bot.py class

    # OLD METHOD REMOVED - Now using DynamicAssetSelection.get_optimal_trading_pairs()
    # The new system scans ALL available pairs and scores them dynamically based on:
    # - Volume and liquidity
    # - Volatility and momentum 
    # - Range position and breakout potential
    # - Market cap category
    # - Diversification across sectors



    async def should_take_profit(self, pair, position, current_price):
        """FIXED: More aggressive profit-taking logic like successful trading bot"""
        profit_pct = ((current_price - position['entry_price']) / position['entry_price']) * 100
        hold_time_minutes = (time.time() - position['entry_time']) / 60
        
        # Get API recommendation if available
        api_data = position.get('api_data', {})
        
        # CRITICAL FIX: Check partial profit levels MORE AGGRESSIVELY
        if config.ENABLE_PARTIAL_PROFITS:
            for level in config.PARTIAL_PROFIT_LEVELS:
                if profit_pct >= level['profit_pct']:
                    # Check if this level hasn't been executed yet
                    completed_partials = position.get('completed_partials', [])
                    level_key = f"partial_{level['profit_pct']}"
                    
                    if level_key not in completed_partials:
                        # Calculate partial sell quantity
                        partial_qty = position['quantity'] * level['sell_pct']
                        
                        logging.info(f"PROFIT TAKING: {pair} at {profit_pct:.2f}% profit - "
                                   f"Selling {level['sell_pct']*100:.0f}% at {level['profit_pct']:.2f}% target")
                        
                        # Execute partial sell
                        try:
                            await self.execute_partial_sell(pair, partial_qty, 
                                                          reason=f"partial_tp_{level['profit_pct']}")
                            
                            # Mark this level as completed
                            if 'completed_partials' not in position:
                                position['completed_partials'] = []
                            position['completed_partials'].append(level_key)
                            
                        except Exception as e:
                            logging.error(f"Error executing partial sell for {pair}: {str(e)}")
        
        # ULTRA-AGGRESSIVE: Take profits much faster than before
        # Adjust targets based on market conditions
        if self.market_state.get('volatility') == 'high':
            # Take profits EVEN QUICKER in volatile markets
            target_multiplier = 0.5  # 50% of normal targets
        elif self.market_state.get('trend') == 'bullish' and position.get('type') == 'buy':
            # Still let some winners run, but not as much
            target_multiplier = 0.8  # 80% of normal targets
        else:
            target_multiplier = 0.7  # 70% of normal targets (more aggressive)
        
        # Check time-based targets with AGGRESSIVE multipliers
        targets = config.MOMENTUM_TAKE_PROFIT if position.get('momentum_trade') else config.REGULAR_TAKE_PROFIT
        
        for target in targets:
            # AGGRESSIVE CHANGE: Reduce time requirement by 50%
            required_time = target['minutes'] * 0.5  # Half the time requirement
            
            if hold_time_minutes >= required_time:
                adjusted_target = target['profit_pct'] * target_multiplier
                if profit_pct >= adjusted_target:
                    logging.info(f"AGGRESSIVE TIME TARGET: {pair} - "
                               f"{profit_pct:.2f}% profit after {hold_time_minutes:.0f}min "
                               f"(target: {adjusted_target:.2f}% in {required_time:.0f}min)")
                    return True
        
        # IMMEDIATE profit taking for good gains (NEW - like successful bot)
        if profit_pct >= 0.8:  # Take 0.8%+ profits immediately
            logging.info(f"IMMEDIATE PROFIT TAKE: {pair} at {profit_pct:.2f}% - exceeds 0.8% threshold")
            return True
        
        # Quick profit taking for decent gains after short time
        if hold_time_minutes >= 5 and profit_pct >= 0.5:  # 0.5% after 5 minutes
            logging.info(f"QUICK PROFIT TAKE: {pair} at {profit_pct:.2f}% after {hold_time_minutes:.0f}min")
            return True
        
        # Check API take profit levels
        if api_data:
            if current_price >= api_data.get('take_profit_1', float('inf')):
                logging.info(f"API take profit 1 reached for {pair}")
                return True
        
        # REDUCED maximum hold time - take ANY profit if held too long
        max_hold_minutes = config.API_MAX_HOLD_TIME_HOURS * 60 * 0.5  # Half the max hold time
        if hold_time_minutes > max_hold_minutes:
            if profit_pct > 0.1:  # Take even 0.1% profit if held too long
                logging.info(f"MAX HOLD EXCEEDED: {pair} - taking {profit_pct:.2f}% profit after {hold_time_minutes:.0f}min")
                return True
        
        return False

    async def execute_smart_entry(self, pair, analysis):
        """Smart entry with multiple confirmation checks"""
        try:
            # Get current market microstructure
            order_book = self.binance_client.get_order_book(symbol=pair, limit=20)
            best_bid = float(order_book['bids'][0][0])
            best_ask = float(order_book['asks'][0][0])
            spread_pct = ((best_ask - best_bid) / best_bid) * 100
            
            # Check spread
            if spread_pct > config.MAX_SPREAD_PCT:
                logging.info(f"Spread too wide for {pair}: {spread_pct:.3f}%")
                return False
            
            # Check recent price action (no huge spikes)
            klines = self.binance_client.get_klines(
                symbol=pair, 
                interval='1m', 
                limit=5
            )
            recent_move = abs(float(klines[-1][4]) - float(klines[0][1])) / float(klines[0][1])
            
            if recent_move > config.MAX_SINGLE_CANDLE_MOVEMENT:
                logging.info(f"Recent price movement too extreme for {pair}: {recent_move:.1%}")
                return False
            
            # Calculate entry price (limit order)
            # Enter slightly below ask for better fill
            entry_price = best_ask * 0.9995  # 0.05% below ask
            
            # Extract signal strength from analysis
            signal_strength = analysis.get('signal_strength', 0.5)
            
            # Extract confidence for enhanced position sizing
            confidence = analysis.get('confidence', analysis.get('api_data', {}).get('confidence', 50))
            
            # Proceed with position sizing and execution
            # Now calling with correct parameters: pair, signal_strength, current_price, confidence
            quantity, position_size = await self.calculate_dynamic_position_size(
                pair, signal_strength, entry_price, confidence=confidence
            )
            
            if quantity > 0 and position_size > 0:
                # Execute the buy with all our enhancements
                success = await self._execute_api_enhanced_buy(pair, analysis)
                
                if success:
                    # Set initial trailing stop
                    await self.set_smart_trailing_stop(pair, entry_price)
                    
                return success
                
        except Exception as e:
            logging.error(f"Error in smart entry for {pair}: {str(e)}")
            return False

    async def set_smart_trailing_stop(self, pair, entry_price):
        """Set intelligent trailing stop based on ATR"""
        try:
            # Get recent price data for ATR
            klines = self.binance_client.get_klines(
                symbol=pair,
                interval='5m',
                limit=50
            )
            
            # Calculate ATR
            highs = [float(k[2]) for k in klines]
            lows = [float(k[3]) for k in klines]
            closes = [float(k[4]) for k in klines]
            
            atr = self.calculate_atr(highs, lows, closes)
            
            # Set trailing stop at 2x ATR
            trailing_distance = atr * 2
            trailing_pct = (trailing_distance / entry_price) * 100
            
            # Apply limits
            trailing_pct = max(1.0, min(3.0, trailing_pct))  # 1-3% range
            
            self.trailing_stops[pair] = {
                'activation_price': entry_price * (1 + config.TRAILING_ACTIVATION_THRESHOLD),
                'trailing_pct': trailing_pct,
                'highest_price': entry_price,
                'stop_price': entry_price * (1 - trailing_pct/100)
            }
            
            logging.info(f"Smart trailing stop set for {pair}: {trailing_pct:.2f}% "
                        f"(ATR-based), activation at {config.TRAILING_ACTIVATION_THRESHOLD*100}% profit")
            
        except Exception as e:
            logging.error(f"Error setting smart trailing stop: {str(e)}")

    def calculate_atr(self, highs, lows, closes, period=14):
        """Calculate Average True Range"""
        true_ranges = []
        
        for i in range(1, len(closes)):
            high_low = highs[i] - lows[i]
            high_close = abs(highs[i] - closes[i-1])
            low_close = abs(lows[i] - closes[i-1])
            true_range = max(high_low, high_close, low_close)
            true_ranges.append(true_range)
        
        if len(true_ranges) >= period:
            atr = sum(true_ranges[-period:]) / period
        else:
            atr = sum(true_ranges) / len(true_ranges) if true_ranges else closes[-1] * 0.02
        
        return atr

    def log_advanced_indicators_stats(self):
        """Log statistics about advanced indicators usage"""
        if self.advanced_indicators:
            try:
                stats = self.advanced_indicators.get_stats()
                logging.info(f"Advanced Indicators Stats - API calls: {stats['api_calls_made']}, "
                           f"Cache hits: {stats['cache_hits']}, "
                           f"Cache items: {stats['cache_stats']['cached_items']}, "
                           f"Client: {stats['client_type']}")
            except Exception as e:
                logging.error(f"Error getting advanced indicators stats: {str(e)}")


    def _get_neutral_analysis(self):
       
        return {
            "mtf_trend": 0,
            "mtf_momentum": 0,
            "mtf_volatility": 0.5,
            "mtf_volume": 0.5,
            "overall_score": 0,
            "timeframes_analyzed": 0,
            "data_source": "neutral_fallback"
        }

    async def market_monitor_task(self):
        """Task to monitor market conditions"""
        try:
            while True:
                try:
                    # Update market breadth
                    self.market_state['breadth'] = await self.market_analysis.calculate_market_breadth()
                    
                    # Update correlation matrix with dynamically selected pairs
                    pairs = await self.asset_selection.get_optimal_trading_pairs(max_pairs=30)
                    await self.correlation.update_correlation_matrix(pairs)
                    
                    # Determine market trend
                    btc_analysis = await self.market_analysis.get_multi_timeframe_analysis('BTCUSDT')
                    eth_analysis = await self.market_analysis.get_multi_timeframe_analysis('ETHUSDT')
                    
                    # Average the trend scores
                    avg_trend = (btc_analysis['mtf_trend'] + eth_analysis['mtf_trend']) / 2
                    
                    if avg_trend > 0.3:
                        self.market_state['trend'] = 'bullish'
                    elif avg_trend < -0.3:
                        self.market_state['trend'] = 'bearish'
                    else:
                        self.market_state['trend'] = 'neutral'
                    
                    # Determine volatility
                    # Get current volatility vs historical
                    btc_volatility = btc_analysis['mtf_volatility']
                    eth_volatility = eth_analysis['mtf_volatility']
                    avg_volatility = (btc_volatility + eth_volatility) / 2
                    
                    if avg_volatility > 0.7:
                        self.market_state['volatility'] = 'high'
                    elif avg_volatility < 0.3:
                        self.market_state['volatility'] = 'low'
                    else:
                        self.market_state['volatility'] = 'normal'
                    
                    # Determine market regime
                    if self.market_state['trend'] == 'bullish' and self.market_state['volatility'] == 'low':
                        regime = "BULL_TRENDING"
                    elif self.market_state['trend'] == 'bullish' and self.market_state['volatility'] == 'high':
                        regime = "BULL_VOLATILE"
                    elif self.market_state['trend'] == 'bearish' and self.market_state['volatility'] == 'low':
                        regime = "BEAR_TRENDING"
                    elif self.market_state['trend'] == 'bearish' and self.market_state['volatility'] == 'high':
                        regime = "BEAR_VOLATILE"
                    else:
                        regime = "NEUTRAL"
                    
                    # Update market regime
                    if regime != self.market_state['regime']:
                        logging.info(f"Market regime changed: {self.market_state['regime']} -> {regime}")
                        self.market_state['regime'] = regime
                    
                    # Log current market state
                    logging.info(f"Market state: Regime={regime}, Trend={self.market_state['trend']}, " +
                               f"Volatility={self.market_state['volatility']}, Breadth={self.market_state['breadth']:.2f}")
                
                except Exception as inner_e:
                    logging.error(f"Error in market monitor cycle: {str(inner_e)}")
                
                # Wait before next update (30 minutes)
                await asyncio.sleep(1800)
                
        except Exception as e:
            logging.error(f"Error in market monitor task: {str(e)}")
            # Restart the task after a delay
            await asyncio.sleep(60)
            asyncio.create_task(self.market_monitor_task())
    
    # 1. UPDATE THE TRADING TASK - Replace pair selection logic
    async def trading_task(self):
        """FIXED: Enhanced trading task with new methods"""
        try:
            while True:
                try:
                    start_time = time.time()
                    
                    # Log API health stats periodically
                    await self._monitor_api_health()
                    await self._log_api_statistics()
                    
                    # Update risk manager with BASE EQUITY ONLY (available cash, no leverage)
                    base_equity = self.get_total_equity()  # Available cash only
                    self.risk_manager.update_equity(base_equity)
                    
                    # Get real-time equity for performance tracking only  
                    current_equity = await self.get_real_time_equity()  # Total portfolio value for tracking
                    
                    # Get current portfolio status
                    risk_status = self.risk_manager.get_status()
                    
                    # Check if we should be trading (simple check for recovery mode)
                    if risk_status.get('severe_recovery_mode', False):
                        logging.info("Trading paused due to severe recovery mode")
                        await asyncio.sleep(60)
                        continue
                    
                    # Log current status
                    logging.info(f"Current equity: ${risk_status['equity']:.2f}, " +
                            f"Daily ROI: {risk_status['daily_roi']:.2f}%, " + 
                            f"Drawdown: {risk_status['drawdown']:.2f}%, " +
                            f"Risk level: {risk_status['risk_level']}, " +
                            f"Positions: {self.risk_manager.position_count}/{config.MAX_POSITIONS} " +
                            f"Recovery mode: {risk_status['recovery_mode']}")
                    
                    # Get high-priority opportunities from scanner
                    high_priority_pairs = []
                    if hasattr(self, 'latest_opportunities') and self.latest_opportunities:
                        # Get fresh opportunities from the scanner
                        high_priority_pairs = [opp['symbol'] for opp in self.latest_opportunities[:3]]
                    
                    # ===== DYNAMIC ASSET SELECTION: SCAN ALL MARKETS =====
                    # Use new fully dynamic selection that scans ALL available pairs
                    pairs_to_analyze = await self.asset_selection.get_optimal_trading_pairs(max_pairs=20)
                    
                    # Process fewer pairs in high volatility
                    if self.market_state.get('volatility') == 'high':
                        pairs_to_analyze = pairs_to_analyze[:10]  # Limit to 10 in high volatility
                        logging.info("High volatility - limiting analysis to 10 pairs")
                    
                    # Combine priority and regular pairs
                    final_pairs = high_priority_pairs[:3] + [p for p in pairs_to_analyze if p not in high_priority_pairs][:15]
                    
                    if high_priority_pairs:
                        logging.info(f"High priority opportunities: {', '.join(high_priority_pairs[:3])}")
                    
                    logging.info(f"Selected {len(final_pairs)} pairs for trading: {final_pairs[:10]}")
                    
                    # Execute analysis on selected pairs - INCREASED FROM 10 TO 15-20
                    analysis_tasks = []
                    for pair in final_pairs[:20]:  # Increased from 10 to 20 pairs
                        try:
                            analysis_tasks.append(self.analyze_and_trade(pair))
                        except Exception as e:
                            logging.error(f"Error creating analysis task for {pair}: {str(e)}")
                    
                    if analysis_tasks:
                        try:
                            await asyncio.gather(*analysis_tasks, return_exceptions=True)
                        except Exception as e:
                            logging.error(f"Error in analysis tasks: {str(e)}")
                    
                    # Check if we've reached daily goal
                    if risk_status and risk_status.get('daily_roi', 0) >= config.TARGET_DAILY_ROI_MIN * 100:
                        logging.info("Daily ROI target achieved! Securing profits...")
                        await self.secure_profits()
                    
                    # Log current daily ROI vs target
                    logging.info(f"Current daily ROI: {risk_status.get('daily_roi', 0):.2f}% "
                            f"(target: {config.TARGET_DAILY_ROI_MIN*100}% - {config.TARGET_DAILY_ROI_MAX*100}%)")
                    
                    # Calculate sleep time (AGGRESSIVE)
                    processing_time = time.time() - start_time
                    sleep_time = max(10, 20 - processing_time)  # Run every 20 seconds for high-frequency
                    await asyncio.sleep(sleep_time)
                    
                except Exception as inner_e:
                    logging.error(f"Error in trading cycle: {str(inner_e)}")
                    await asyncio.sleep(60)
                    
        except Exception as e:
            logging.error(f"Error in trading task: {str(e)}")
            await asyncio.sleep(60)
            asyncio.create_task(self.trading_task())

    
    async def position_monitor_task(self):
        """Task to monitor open positions"""
        try:
            while True:
                try:
                    # Monitor open positions for stop loss, take profit, etc.
                    await self.monitor_positions()
                except Exception as inner_e:
                    logging.error(f"Error in position monitoring cycle: {str(inner_e)}")
                
                # Wait before next check (30 seconds)
                await asyncio.sleep(30)
                
        except Exception as e:
            logging.error(f"Error in position monitor task: {str(e)}")
            # Restart the task after a delay
            await asyncio.sleep(60)
            asyncio.create_task(self.position_monitor_task())
    
    async def performance_track_task(self):
        """Task to track and report performance"""
        try:
            while True:
                try:
                    # Get real-time equity including current position values
                    current_equity = await self.get_real_time_equity()
                    
                    # Calculate daily metrics
                    daily_roi = self.risk_manager.calculate_daily_roi()
                    drawdown = self.risk_manager.current_drawdown * 100
                    
                    # Record equity and performance metrics
                    await self.performance_tracker.record_equity(
                        current_equity,
                        daily_roi=daily_roi, 
                        drawdown=drawdown
                    )
                    
                    # Generate and log performance report hourly
                    current_hour = datetime.now().hour
                    if not hasattr(self, 'last_report_hour') or self.last_report_hour != current_hour:
                        self.last_report_hour = current_hour
                        report = await self.performance_tracker.generate_performance_report()
                        logging.info(f"Hourly performance report: {json.dumps(report, indent=2)}")
                        
                except Exception as inner_e:
                    logging.error(f"Error in performance tracking cycle: {str(inner_e)}")
                
                # Wait before next update (5 minutes)
                await asyncio.sleep(300)
                
        except Exception as e:
            logging.error(f"Error in performance track task: {str(e)}")
            # Restart the task after a delay
            await asyncio.sleep(60)
            asyncio.create_task(self.performance_track_task())
    


    async def calculate_dynamic_position_size(self, pair, signal_strength, current_price, confidence=None):
        """
        Calculate dynamic position size that ensures minimum order requirements are met
        """
        try:
            # Get symbol info for precision and limits
            info = self.binance_client.get_symbol_info(pair)
            if not info:
                logging.error(f"Could not get symbol info for {pair}")
                return 0, 0
            
            # Extract filters
            min_notional = 10.0  # Default minimum notional value ($10)
            min_qty = 0.0
            max_qty = float('inf')
            step_size = 0.00001
            
            for f in info['filters']:
                if f['filterType'] == 'MIN_NOTIONAL':
                    min_notional = float(f['minNotional'])
                elif f['filterType'] == 'LOT_SIZE':
                    min_qty = float(f['minQty'])
                    max_qty = float(f['maxQty'])
                    step_size = float(f['stepSize'])
            
            # Get current market regime for enhanced position sizing
            market_regime = self.market_state.get('regime', 'NEUTRAL')
            
            # Get base position size from risk manager with enhanced parameters
            base_position_size = self.risk_manager._calculate_position_size(
                signal_strength, 
                confidence=confidence, 
                market_regime=market_regime
            )
            
            # Return 0 if confidence filtering rejected the signal
            if base_position_size == 0:
                return 0, 0
            
            # Apply dynamic adjustments based on signal strength
            if signal_strength > 0.8:
                multiplier = 1.5  # 50% larger for very strong signals
            elif signal_strength > 0.6:
                multiplier = 1.2  # 20% larger for strong signals  
            elif signal_strength > 0.4:
                multiplier = 1.0  # Normal size
            else:
                multiplier = 0.8  # 20% smaller for weak signals
            
            adjusted_position_size = base_position_size * multiplier
            
            # CRITICAL: Ensure minimum notional value is met with buffer
            min_required = min_notional * 1.2
            if adjusted_position_size < min_required:
                adjusted_position_size = min_required
                logging.info(f"Adjusted position size to ${adjusted_position_size:.2f} to meet minimum notional")
            
            # Calculate quantity
            quantity = adjusted_position_size / current_price
            
            # Ensure minimum quantity is met
            if quantity < min_qty:
                quantity = min_qty
                adjusted_position_size = quantity * current_price
                logging.info(f"Adjusted quantity to meet minimum: {quantity}")
            
            # Round to step size properly
            precision = 0
            temp_step = step_size
            while temp_step < 1:
                temp_step *= 10
                precision += 1
            
            # Proper rounding to avoid going below minimum
            quantity = round(quantity - (quantity % step_size), precision)
            
            # Final validation - this is the key fix
            final_notional = quantity * current_price
            if final_notional < min_notional:
                # Force to minimum with extra buffer
                quantity = (min_notional * 1.2) / current_price
                quantity = round(quantity + step_size, precision)  # Round UP, not down
                adjusted_position_size = quantity * current_price
                logging.info(f"Final adjustment: quantity={quantity}, value=${adjusted_position_size:.2f}")
            
            return quantity, adjusted_position_size
            
        except Exception as e:
            logging.error(f"Error calculating dynamic position size: {str(e)}")
            return 0, 0


    async def analyze_and_trade(self, pair: str, analysis: dict = None) -> bool:
        
        try:
            # Skip if already have position
            if pair in self.active_positions:
                return False
            
            # CRITICAL: Validate position sync and enforce strict limits
            actual_position_count = self._validate_position_sync()
            
            # ENFORCE STRICT 5-POSITION LIMIT (like successful trading bot)
            if actual_position_count >= 5:
                logging.debug(f"Skipping {pair} - STRICT LIMIT: {actual_position_count}/5 positions (100% allocation)")
                return False
            
            # Double-check with config limit (redundant safety)
            if actual_position_count >= config.MAX_POSITIONS:
                logging.debug(f"Skipping {pair} - config limit: {actual_position_count}/{config.MAX_POSITIONS}")
                return False
            
            # Track that we're analyzing this pair
            self.api_stats['total_analyses'] += 1
            
            # Get API signal using global client
            signal = await self.strategy.get_api_signal(pair, self.global_api_client)
            
            if not signal:
                logging.debug(f"No API signal for {pair}")
                self.api_stats['api_errors'] += 1
                return False
            
            # Track successful API call
            self.api_stats['successful_api_calls'] += 1
            self.api_stats['api_signals_used'] += 1
            
            # Check signal quality
            confidence = signal.get('confidence', 0) / 100
            signal_type = signal.get('signal', 'HOLD').upper()
            reason = signal.get('reasoning', 'unknown')
            
            # ===== ENHANCED: BLEND CONFIDENCE WITH ASSET SCORES =====
            # Get asset selection score for this pair
            asset_score = await self._get_asset_score(pair)
            
            # SAFETY: Validate symbol for live trading
            is_safe_for_trading = await self._validate_symbol_for_trading(pair)
            if not is_safe_for_trading:
                logging.warning(f"Skipping {pair} - failed live trading safety validation")
                return False
            
            # Blend API confidence with asset selection score
            blended_confidence = self._calculate_blended_confidence(confidence, asset_score, signal)
            
            # Use blended confidence for decision making
            min_confidence = getattr(config, 'MIN_SIGNAL_CONFIDENCE', 25) / 100  # Convert to decimal
            if blended_confidence < min_confidence:
                logging.info(f"Ignoring {pair} signal - blended confidence {blended_confidence:.1%} below {min_confidence:.1%} threshold (API: {confidence:.1%}, Asset Score: {asset_score:.2f})")
                return False
            
            # Log the enhanced confidence
            logging.info(f"Enhanced signal for {pair}: {signal_type} (API: {confidence:.1%}, Asset Score: {asset_score:.2f}, Blended: {blended_confidence:.1%}, reason: {reason})")
            
            # Use blended confidence for strategy execution
            strategy_confidence = blended_confidence
            
            # ===== ENHANCED: USE MARKET PHASE STRATEGY =====
            if signal_type == 'BUY' and strategy_confidence > 0.15:  # Use blended confidence
                logging.info(f"EXECUTING MARKET PHASE STRATEGY for {pair} - Blended confidence {strategy_confidence:.1%}")
                
                # Convert API signal to analysis format with enhanced data
                analysis = {
                    'buy_signal': True,
                    'signal_strength': strategy_confidence,  # Use blended confidence
                    'confidence': blended_confidence * 100,  # Convert back to percentage for consistency
                    'asset_score': asset_score,  # Include asset score
                    'api_data': signal,
                    'source': 'enhanced_api_signal'
                }
                
                # Use market phase strategy handler
                success = await self.phase_strategy.execute_phase_strategy(pair, analysis)
                return success
                
            elif signal_type == 'SELL' and strategy_confidence > 0.15:  # Use blended confidence
                # Check if we actually have a position to sell
                if pair not in self.active_positions:
                    logging.debug(f"SELL signal for {pair} ignored - no active position")
                    return False
                    
                logging.info(f"EXECUTING MARKET PHASE SELL STRATEGY for {pair} - Blended confidence {strategy_confidence:.1%}")
                
                analysis = {
                    'sell_signal': True,
                    'signal_strength': strategy_confidence,  # Use blended confidence
                    'asset_score': asset_score,  # Include asset score
                    'api_data': signal,
                    'source': 'enhanced_api_signal'
                }
                
                # Use market phase strategy handler for sell signals too
                success = await self.phase_strategy.execute_phase_strategy(pair, analysis)
                return success
                
            else:
                logging.debug(f"HOLDING {pair} - {reason} (blended confidence {strategy_confidence:.1%} too low)")
                return False
                
        except Exception as e:
            logging.error(f"Error in analyze_and_trade for {pair}: {str(e)}")
            self.api_stats['api_errors'] += 1
            return False
    
    async def _get_asset_score(self, pair: str) -> float:
        """Get the asset selection score for a trading pair"""
        try:
            # Check if asset selection has cached scores (should be very recent)
            if hasattr(self.asset_selection, 'cached_scores') and pair in self.asset_selection.cached_scores:
                cache_time = getattr(self.asset_selection, 'cached_scores_time', 0)
                if time.time() - cache_time < 600:  # 10 minute cache
                    return self.asset_selection.cached_scores[pair]
            
            # If no cached scores, trigger a fresh selection to populate cache
            await self.asset_selection.get_optimal_trading_pairs(max_pairs=30)
            
            # Try to get score from cache again
            if hasattr(self.asset_selection, 'cached_scores') and pair in self.asset_selection.cached_scores:
                return self.asset_selection.cached_scores[pair]
            
            # Return default score if not found
            return 3.0  # Default neutral score
            
        except Exception as e:
            logging.error(f"Error getting asset score for {pair}: {str(e)}")
            return 3.0  # Default neutral score
    
    def _calculate_blended_confidence(self, api_confidence: float, asset_score: float, signal: dict) -> float:
        """Calculate blended confidence using API confidence, asset score, and market regime"""
        try:
            # Get market regime data from signal
            market_regime = signal.get('market_regime', {})
            regime_confidence = market_regime.get('regime_confidence', 0.5)
            regime_strength = market_regime.get('regime_strength', 0.5)
            
            # Normalize asset score to 0-1 range (assuming scores are 0-10)
            normalized_asset_score = min(asset_score / 10.0, 1.0)
            
            # Get blending weights from config
            ml_weight = getattr(config, 'ML_CONFIDENCE_WEIGHT', 0.7)
            regime_weight = getattr(config, 'REGIME_CONFIDENCE_WEIGHT', 0.3)
            
            # Calculate base blended confidence
            blended = (api_confidence * ml_weight) + (normalized_asset_score * 0.5 * regime_weight)
            
            # Apply regime strength boost if enabled
            if getattr(config, 'USE_REGIME_CONFIDENCE_BOOST', True):
                # Boost confidence for strong market regimes
                regime_boost = (regime_confidence * regime_strength) * 0.1  # Max 10% boost
                blended += regime_boost
            
            # Apply asset score multiplier for high-scoring opportunities
            if asset_score >= 5.0:  # High-scoring assets
                asset_boost = (asset_score - 5.0) * 0.02  # 2% boost per point above 5.0
                blended += asset_boost
                
            # Cap at maximum confidence
            blended = min(blended, 0.95)  # Max 95% confidence
            
            return blended
            
        except Exception as e:
            logging.error(f"Error calculating blended confidence: {str(e)}")
            return api_confidence  # Fallback to API confidence
    
    async def _validate_symbol_for_trading(self, symbol: str) -> bool:
        """Validate symbol for live trading safety using API endpoint"""
        try:
            # Skip validation in test mode unless explicitly enabled
            if config.TEST_MODE and not getattr(config, 'VALIDATE_SYMBOLS_IN_TEST', False):
                return True
            
            # Check cache first (symbols don't change validity often)
            cache_key = f"symbol_validation_{symbol}"
            if hasattr(self, 'symbol_validation_cache'):
                cached_result = self.symbol_validation_cache.get(cache_key)
                if cached_result is not None:
                    cache_time = cached_result.get('timestamp', 0)
                    if time.time() - cache_time < 3600:  # 1 hour cache
                        return cached_result['is_valid']
            
            # Initialize cache if needed
            if not hasattr(self, 'symbol_validation_cache'):
                self.symbol_validation_cache = {}
            
            # Make API request to validate symbol
            import aiohttp
            import asyncio
            
            api_base_url = getattr(config, 'API_BASE_URL', 'http://localhost:3001')
            api_key = getattr(config, 'API_KEY', '')
            
            if not api_key:
                logging.warning(f"No API key configured, skipping symbol validation for {symbol}")
                return True  # Default to safe if no API key
            
            url = f"{api_base_url}/api/v1/validate-symbol/{symbol}"
            headers = {
                'Authorization': f'Bearer {api_key}',
                'Content-Type': 'application/json'
            }
            
            timeout = aiohttp.ClientTimeout(total=10)  # 10 second timeout
            
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(url, headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        is_safe = data.get('safe_for_live_trading', False)
                        
                        # Cache the result
                        self.symbol_validation_cache[cache_key] = {
                            'is_valid': is_safe,
                            'timestamp': time.time(),
                            'response': data
                        }
                        
                        if is_safe:
                            logging.debug(f"✅ {symbol} validated safe for live trading")
                        else:
                            logging.warning(f"⚠️ {symbol} marked as UNSAFE for live trading")
                            
                        return is_safe
                        
                    elif response.status == 400:
                        # Symbol doesn't exist or invalid
                        data = await response.json()
                        logging.warning(f"❌ {symbol} validation failed: {data.get('message', 'Invalid symbol')}")
                        
                        # Cache negative result
                        self.symbol_validation_cache[cache_key] = {
                            'is_valid': False,
                            'timestamp': time.time(),
                            'response': data
                        }
                        return False
                        
                    else:
                        # API error - log but don't block trading
                        logging.warning(f"Symbol validation API error for {symbol}: HTTP {response.status}")
                        return True  # Default to safe on API errors
                        
        except asyncio.TimeoutError:
            logging.warning(f"Symbol validation timeout for {symbol}, defaulting to safe")
            return True
        except Exception as e:
            logging.error(f"Error validating symbol {symbol}: {str(e)}")
            return True  # Default to safe on errors
    
    async def get_current_price(self, pair: str) -> float:
        """FIXED: Get current price with comprehensive error handling"""
        try:
            # Check cache first
            cache_key = f"price_{pair}"
            current_time = time.time()
            
            if (hasattr(self, 'data_cache') and cache_key in self.data_cache and 
                hasattr(self, 'cache_expiry') and self.cache_expiry.get(cache_key, 0) > current_time):
                return self.data_cache[cache_key]
                
            # Initialize cache if needed
            if not hasattr(self, 'data_cache'):
                self.data_cache = {}
            if not hasattr(self, 'cache_expiry'):
                self.cache_expiry = {}
                
            # Track API call
            try:
                self.track_api_call("get_symbol_ticker")
            except:
                pass

            # FIXED: Safe API call with fallback
            price = 0.0
            try:
                async with self.api_semaphore:
                    ticker = self.binance_client.get_symbol_ticker(symbol=pair)
                    if ticker and isinstance(ticker, dict) and 'price' in ticker:
                        price = float(ticker['price'])
                    else:
                        logging.warning(f"Invalid ticker response for {pair}: {ticker}")
                        return 0.0
            except Exception as api_e:
                logging.error(f"API error getting price for {pair}: {str(api_e)}")
                # Try to return cached price if available
                if hasattr(self, 'data_cache') and cache_key in self.data_cache:
                    cached_price = self.data_cache[cache_key]
                    logging.info(f"Using cached price for {pair}: {cached_price}")
                    return cached_price
                return 0.0

            # FIXED: Validate price before caching
            if price > 0:
                # Cache the price
                self.data_cache[cache_key] = price
                self.cache_expiry[cache_key] = current_time + 30
                return price
            else:
                logging.warning(f"Invalid price for {pair}: {price}")
                return 0.0
                
        except Exception as e:
            logging.error(f"Error getting current price for {pair}: {str(e)}")
            # Return cached price if available
            try:
                cache_key = f"price_{pair}"
                if hasattr(self, 'data_cache') and cache_key in self.data_cache:
                    return self.data_cache[cache_key]
            except:
                pass
            return 0.0
    
    def track_api_call(self, endpoint: str):
        """Track API call frequency for performance monitoring and rate limiting"""
        try:
            current_time = time.time()
            current_hour = int(current_time / 3600)
            
            # Initialize tracking structures if they don't exist
            if not hasattr(self, 'api_call_tracker'):
                self.api_call_tracker = {}
                
            # Initialize endpoint tracking if needed
            if endpoint not in self.api_call_tracker:
                self.api_call_tracker[endpoint] = {}
                
            # Count this hour's calls
            if current_hour not in self.api_call_tracker[endpoint]:
                self.api_call_tracker[endpoint][current_hour] = 1
            else:
                self.api_call_tracker[endpoint][current_hour] += 1
                
            # Get the current count
            hourly_calls = self.api_call_tracker[endpoint][current_hour]
            
            # Clean up old data (keep only current hour plus previous hour)
            hours_to_keep = [current_hour, current_hour - 1]
            for hour in list(self.api_call_tracker[endpoint].keys()):
                if hour not in hours_to_keep:
                    del self.api_call_tracker[endpoint][hour]
                    
            return hourly_calls
            
        except Exception as e:
            logging.error(f"Error tracking API call for {endpoint}: {str(e)}")
            return 0

    async def debug_position_sizing(self, pair):
        """Debug method to check position sizing for a pair"""
        try:
            current_price = await self.get_current_price(pair)
            info = self.binance_client.get_symbol_info(pair)
            
            min_notional = 10.0
            min_qty = 0.0
            step_size = 0.00001
            
            for f in info['filters']:
                if f['filterType'] == 'MIN_NOTIONAL':
                    min_notional = float(f['minNotional'])
                elif f['filterType'] == 'LOT_SIZE':
                    min_qty = float(f['minQty'])
                    step_size = float(f['stepSize'])
            
            logging.info(f"=== {pair} Sizing Debug ===")
            logging.info(f"Current Price: ${current_price}")
            logging.info(f"Min Notional: ${min_notional}")
            logging.info(f"Min Quantity: {min_qty}")
            logging.info(f"Step Size: {step_size}")
            
            # Test different signal strengths
            for signal in [0.3, 0.5, 0.7, 0.9]:
                qty, value = await self.calculate_dynamic_position_size(pair, signal, current_price)
                logging.info(f"Signal {signal}: Qty={qty}, Value=${value:.2f}")
                
        except Exception as e:
            logging.error(f"Debug error: {str(e)}")
    



    async def _execute_api_enhanced_buy(self, pair: str, analysis: dict) -> bool:
        """Execute buy order with API-enhanced parameters"""
        try:
            # Get API data
            api_data = analysis.get('api_data', {})
            
            # Get current price
            current_price = await self.get_current_price(pair)
            if not current_price:
                return False
            
            # ===== CHANGE 5: USE DYNAMIC POSITION SIZING =====
            # Extract signal strength from analysis
            signal_strength = analysis.get('signal_strength', 0.5)
            
            # Extract confidence for enhanced position sizing
            confidence = analysis.get('confidence', api_data.get('confidence', 50))
            
            # Call with correct parameters: pair, signal_strength, current_price, confidence
            quantity, position_value = await self.calculate_dynamic_position_size(
                pair, signal_strength, current_price, confidence=confidence
            )
            
            if quantity <= 0 or position_value <= 0:
                logging.info(f"Position size 0 for {pair} (qty: {quantity}, value: {position_value}), skipping trade")
                return False
            
            # Enhanced logging with all confidence factors
            api_confidence = api_data.get('confidence', 50)
            blended_confidence = analysis.get('confidence', 50)
            asset_score = analysis.get('asset_score', 3.0)
            api_reason = api_data.get('reasoning', 'technical_analysis')
            
            # Log the enhanced trade setup
            logging.info(f"ENHANCED BUY: {pair} - Qty: {quantity:.6f}, Value: ${position_value:.2f}, "
                        f"API: {api_confidence:.1f}%, Blended: {blended_confidence:.1f}%, "
                        f"Asset Score: {asset_score:.2f}, Reason: {api_reason}")
            
            # Execute the order (test mode or live)
            if config.TEST_MODE:
                # Simulate successful order
                order_id = f"test_{int(time.time())}"
                
                # Record position using synchronized method
                position_data = {
                    'entry_price': current_price,
                    'quantity': quantity,
                    'entry_time': time.time(),
                    'position_value': position_value,  # Use consistent key name
                    'signal_source': 'api_enhanced',
                    'signal_strength': analysis.get('signal_strength', 0),
                    'api_data': api_data,
                    'order_id': order_id,
                    'momentum_trade': analysis.get('momentum_trade', False)
                }
                
                # Use synchronized position management to prevent 6/5 position bugs
                self._add_position_synchronized(pair, position_data)
                
                # Update equity in database to reflect position cost
                if config.TEST_MODE:
                    current_equity = await self.get_real_time_equity()
                    await self.update_equity_in_db(current_equity)
                
                # ===== CHANGE 6: SET SMART TRAILING STOP =====
                await self.set_smart_trailing_stop(pair, current_price)
                
                # Set API-based stop loss and take profits if available
                if api_data:
                    # Calculate stop loss price (1.5% below entry)
                    stop_loss_price = current_price * (1 + config.QUICK_STOP_LOSS / 100)
                    await self._set_api_stop_loss(pair, current_price, stop_loss_price)
                    
                    # Set take profits if API provides exit levels
                    if isinstance(api_data, dict) and 'exit_levels' in api_data:
                        await self._set_api_take_profits(pair, current_price, api_data['exit_levels'])
                    else:
                        # Set default take profit levels
                        default_exit_levels = {
                            'take_profit_1': current_price * 1.015,  # 1.5% profit
                            'take_profit_2': current_price * 1.025,  # 2.5% profit
                            'take_profit_3': current_price * 1.035   # 3.5% profit
                        }
                        await self._set_api_take_profits(pair, current_price, default_exit_levels)
                
                logging.info(f"TEST: Buy order for {pair} simulated successfully")
                return True
                
            else:
                # Execute real order
                try:
                    order = await self.binance_client.create_order(
                        symbol=pair,
                        side='BUY',
                        type='MARKET',
                        quantity=quantity
                    )
                    
                    if order and order.get('status') == 'FILLED':
                        # Get actual fill price
                        fill_price = float(order.get('fills', [{}])[0].get('price', current_price))
                        
                        # Record position using synchronized method
                        position_data = {
                            'entry_price': fill_price,
                            'quantity': float(order['executedQty']),
                            'entry_time': time.time(),
                            'position_value': float(order['cummulativeQuoteQty']),
                            'signal_source': 'api_enhanced',
                            'signal_strength': analysis.get('signal_strength', 0),
                            'api_data': api_data,
                            'order_id': order['orderId'],
                            'momentum_trade': analysis.get('momentum_trade', False)
                        }
                        
                        # Use synchronized position management to prevent 6/5 position bugs
                        self._add_position_synchronized(pair, position_data)
                        
                        # Update equity in database to reflect position cost
                        if config.TEST_MODE:
                            current_equity = await self.get_real_time_equity()
                            await self.update_equity_in_db(current_equity)
                        
                        # ===== CHANGE 7: SET SMART TRAILING STOP =====
                        await self.set_smart_trailing_stop(pair, fill_price)
                        
                        logging.info(f"LIVE: Buy order for {pair} executed successfully")
                        return True
                    else:
                        logging.error(f"Buy order failed: {order}")
                        return False
                except Exception as e:
                    logging.error(f"Error executing buy order: {str(e)}")
                    return False
        
        except Exception as e:
            logging.error(f"Error executing API-enhanced buy for {pair}: {str(e)}")
            return False


    async def _execute_buy_with_levels(self, pair: str, quantity: float, position_size: float, 
                                      analysis: dict[str, any], exit_levels: dict[str, any]) -> bool:
        
        try:
            current_price = await self.get_current_price(pair)
            
            # Enhanced logging with API data
            confidence = analysis.get('confidence', 0.0)
            reason = analysis.get('reason', 'API signal')
            log_message = (f"API-ENHANCED BUY: {pair} - "
                          f"Qty: {quantity:.6f}, Value: ${position_size:.2f}, "
                          f"API Confidence: {confidence:.1%}, "
                          f"Reason: {reason}")
            
            logging.info(log_message)
            self.trade_logger.info(log_message)
            
            # Execute order (existing logic)
            trade_id = f"{int(time.time())}-{random.randint(1000, 9999)}"
            
            if config.TEST_MODE:
                # Record enhanced position data
                self.active_positions[pair] = {
                    "id": trade_id,
                    "entry_price": current_price,
                    "quantity": quantity,
                    "entry_time": time.time(),
                    "position_size": position_size,
                    "signal_source": "enhanced_api",
                    "signal_strength": analysis.get('confidence', 0),
                    "exit_levels": exit_levels,
                    "api_confidence": analysis.get('confidence', 0),
                    "reason": analysis.get('reason', 'API signal')
                }
                
                # Set API-recommended stop loss and take profits
                if exit_levels.get('stop_loss'):
                    await self._set_api_stop_loss(pair, current_price, exit_levels['stop_loss'])
                
                if exit_levels.get('take_profit_1'):
                    await self._set_api_take_profits(pair, current_price, exit_levels)
                
                return True
                
            else:
                # KOMPLET: Execute real order for live trading
                try:
                    order = self.binance_client.order_market_buy(
                        symbol=pair,
                        quantity=quantity
                    )
                    
                    if order and order.get('status') == 'FILLED':
                        # Get actual execution details
                        actual_price = float(order['fills'][0]['price'])
                        actual_quantity = float(order['executedQty'])
                        actual_value = actual_price * actual_quantity
                        
                        # Record the trade in database
                        await self.db_manager.execute_query(
                            """
                            INSERT INTO trades
                            (trade_id, pair, type, price, quantity, value, timestamp)
                            VALUES (?, ?, ?, ?, ?, ?, ?)
                            """,
                            (
                                trade_id, pair, 'buy', actual_price, actual_quantity,
                                actual_value, datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                            ),
                            commit=True
                        )
                        
                        # Record the position
                        position_id = await self.db_manager.execute_query(
                            """
                            INSERT INTO positions
                            (pair, entry_price, quantity, entry_time, status, signal_source, signal_strength)
                            VALUES (?, ?, ?, ?, ?, ?, ?)
                            """,
                            (
                                pair, actual_price, actual_quantity,
                                datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 
                                'open', analysis.get('source', 'enhanced_api'),
                                analysis.get('confidence', 0)
                            ),
                            commit=True
                        )
                        
                        # Record in active positions dictionary
                        self.active_positions[pair] = {
                            "id": position_id,
                            "entry_price": actual_price,
                            "quantity": actual_quantity,
                            "entry_time": time.time(),
                            "position_size": actual_value,
                            "order_id": order.get('orderId'),
                            "signal_source": analysis.get('source', 'enhanced_api'),
                            "signal_strength": analysis.get('confidence', 0)
                        }
                        
                        # Initialize stop loss if needed
                        await self.initialize_stop_loss(pair, actual_price)
                        
                        # Initialize trailing take profit if enabled
                        if config.ENABLE_TRAILING_TP:
                            await self.initialize_trailing_take_profit(pair, actual_price)
                        
                        # TILFØJ: Mark pair as recently traded for diversification
                        if hasattr(self, 'asset_selection'):
                            self.asset_selection.mark_recently_traded(pair)
                        
                        logging.info(f"LIVE: Buy order for {pair} executed successfully at ${actual_price:.4f}")
                        return True
                    else:
                        logging.error(f"Buy order failed: {order}")
                        return False
                except Exception as e:
                    logging.error(f"Error executing buy order: {str(e)}")
                    return False
        
        except Exception as e:
            logging.error(f"Error executing buy for {pair}: {str(e)}")
            return False

    async def execute_sell(self, pair, analysis):
        """Execute a sell order with enhanced tracking and recently traded marking"""
        try:
            if pair not in self.active_positions:
                logging.warning(f"Cannot sell {pair}: no active position")
                return False
                
            position = self.active_positions[pair]
            quantity = position['quantity']
            
            # Get current price
            current_price = await self.get_current_price(pair)
            if current_price <= 0:
                logging.error(f"Invalid price for {pair}: {current_price}")
                return False
                
            # Calculate profit/loss
            entry_price = position['entry_price']
            profit_loss = (current_price - entry_price) * quantity
            profit_percent = ((current_price / entry_price) - 1) * 100
            
            # Log the order details
            log_message = f"SELL ORDER: {pair} - {quantity} at approx. ${current_price}, " + \
                        f"P/L: ${profit_loss:.2f} ({profit_percent:.2f}%)"
            logging.info(log_message)
            self.trade_logger.info(log_message)
            
            # Generate a trade ID
            trade_id = f"{int(time.time())}-{random.randint(1000, 9999)}"
            
            # Track successful sells
            success = False
            
            if config.TEST_MODE:
                # Simulate order in test mode
                order_id = f"test_{int(time.time())}"
                
                # Record the trade in database
                await self.db_manager.execute_query(
                    """
                    INSERT INTO trades
                    (trade_id, pair, type, price, quantity, value, profit_loss, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        trade_id, pair, 'sell', current_price, quantity,
                        current_price * quantity, profit_loss,
                        datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    ),
                    commit=True
                )
                
                # Update position status
                if 'id' in position:
                    await self.db_manager.execute_query(
                        """
                        UPDATE positions
                        SET status = 'closed', exit_price = ?, exit_time = ?, profit_loss = ?
                        WHERE id = ?
                        """,
                        (
                            current_price, datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            profit_loss, position['id']
                        ),
                        commit=True
                    )
                
                # Record the trade data
                trade_data = {
                    "pair": pair,
                    "direction": "SELL",
                    "entry_price": position['entry_price'],
                    "exit_price": current_price,
                    "quantity": quantity,
                    "profit_loss": profit_loss,
                    "profit_percent": profit_percent,
                    "trade_duration": time.time() - position['entry_time'],
                    "signal_source": position.get('signal_source', 'combined'),
                    "signal_strength": position.get('signal_strength', 0)
                }
                
                # Record the trade in performance tracker
                await self.performance_tracker.record_trade(trade_data)
                
                # Clean up trailing stops and take profit data
                if pair in self.trailing_stops:
                    del self.trailing_stops[pair]
                
                if pair in self.trailing_tp_data:
                    del self.trailing_tp_data[pair]
                    
                # Clean up position scaling data
                if pair in self.position_scales:
                    del self.position_scales[pair]
                
                # Remove the position
                del self.active_positions[pair]
                
                # Update risk manager
                self.risk_manager.remove_position(pair)
                
                # Update equity in database to reflect realized P&L
                if config.TEST_MODE:
                    new_equity = await self.get_real_time_equity()
                    await self.update_equity_in_db(new_equity)
                
                success = True
                
            else:
                # Execute real order for live trading
                try:
                    order = self.binance_client.order_market_sell(
                        symbol=pair,
                        quantity=quantity
                    )
                    
                    if order and order.get('status') == 'FILLED':
                        # Get actual execution details
                        actual_price = float(order['fills'][0]['price'])
                        actual_quantity = float(order['executedQty'])
                        actual_value = actual_price * actual_quantity
                        actual_profit_loss = (actual_price - entry_price) * actual_quantity
                        actual_profit_percent = ((actual_price / entry_price) - 1) * 100
                        
                        # Record the trade in database
                        await self.db_manager.execute_query(
                            """
                            INSERT INTO trades
                            (trade_id, pair, type, price, quantity, value, profit_loss, timestamp)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                            (
                                trade_id, pair, 'sell', actual_price, actual_quantity,
                                actual_value, actual_profit_loss,
                                datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                            ),
                            commit=True
                        )
                        
                        # Update position status
                        if 'id' in position:
                            await self.db_manager.execute_query(
                                """
                                UPDATE positions
                                SET status = 'closed', exit_price = ?, exit_time = ?, profit_loss = ?
                                WHERE id = ?
                                """,
                                (
                                    actual_price, datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                    actual_profit_loss, position['id']
                                ),
                                commit=True
                            )
                        
                        # Record the trade data
                        trade_data = {
                            "pair": pair,
                            "direction": "SELL",
                            "entry_price": position['entry_price'],
                            "exit_price": actual_price,
                            "quantity": actual_quantity,
                            "profit_loss": actual_profit_loss,
                            "profit_percent": actual_profit_percent,
                            "trade_duration": time.time() - position['entry_time'],
                            "signal_source": position.get('signal_source', 'combined'),
                            "signal_strength": position.get('signal_strength', 0)
                        }
                        
                        # Record the trade in performance tracker
                        await self.performance_tracker.record_trade(trade_data)
                        
                        # Clean up trailing stops and take profit data
                        if pair in self.trailing_stops:
                            del self.trailing_stops[pair]
                        
                        if pair in self.trailing_tp_data:
                            del self.trailing_tp_data[pair]
                            
                        # Clean up position scaling data
                        if pair in self.position_scales:
                            del self.position_scales[pair]
                        
                        # Remove the position
                        del self.active_positions[pair]

                        # Update risk manager
                        self.risk_manager.remove_position(pair)
                        
                        # Update equity in database to reflect realized P&L
                        if config.TEST_MODE:
                            new_equity = await self.get_real_time_equity()
                            await self.update_equity_in_db(new_equity)
                        
                        logging.info(f"LIVE: Sell order for {pair} executed successfully at ${actual_price:.4f}")
                        success = True
                    else:
                        logging.error(f"Sell order failed: {order}")
                        success = False
                except Exception as e:
                    logging.error(f"Error executing sell order: {str(e)}")
                    success = False
            
            # RETTET: Improved recently traded tracking
            if success:
                # TILFØJ: Mark pair as recently traded using asset_selection method
                if hasattr(self, 'asset_selection'):
                    self.asset_selection.mark_recently_traded(pair)
                
                # Session stats tracking (simplified)
                if not hasattr(self, 'session_stats'):
                    self.session_stats = {
                        'total_trades': 0,
                        'winning_trades': 0,
                        'total_profit': 0
                    }
                
                self.session_stats['total_trades'] += 1
                if profit_loss > 0:
                    self.session_stats['winning_trades'] += 1
                self.session_stats['total_profit'] += profit_loss
                
                # Log session stats
                win_rate = (self.session_stats['winning_trades'] / self.session_stats['total_trades'] * 100) if self.session_stats['total_trades'] > 0 else 0
                logging.info(f"Session stats: {self.session_stats['total_trades']} trades, "
                            f"{win_rate:.1f}% win rate, ${self.session_stats['total_profit']:.2f} profit")
            
            return success
        
        except Exception as e:
            logging.error(f"Error executing sell for {pair}: {str(e)}")
            return False
            

        
    def calculate_profit_percent(self, pair):
        """FIXED: Calculate profit percentage with safe error handling"""
        try:
            if pair not in self.active_positions:
                return 0
                
            position = self.active_positions[pair]
            entry_price = position.get('entry_price', 0)
            
            if entry_price <= 0:
                logging.warning(f"Invalid entry price for {pair}: {entry_price}")
                return 0
            
            # Get current price safely
            try:
                ticker = self.binance_client.get_symbol_ticker(symbol=pair)
                if ticker and isinstance(ticker, dict) and 'price' in ticker:
                    current_price = float(ticker['price'])
                    if current_price > 0:
                        profit_percent = ((current_price / entry_price) - 1) * 100
                        return profit_percent
                    else:
                        logging.warning(f"Invalid current price for {pair}: {current_price}")
                        return 0
                else:
                    logging.warning(f"Invalid ticker for {pair}: {ticker}")
                    return 0
            except Exception as price_e:
                logging.error(f"Error getting current price for profit calculation {pair}: {str(price_e)}")
                return 0
                
        except Exception as e:
            logging.error(f"Error calculating profit for {pair}: {str(e)}")
            return 0
    
    async def monitor_positions(self):
        """Monitor and manage open positions with enhanced logic"""
        if not self.active_positions:
            return
        
        try:
            positions_to_close = []
            
            for pair, position in self.active_positions.items():
                try:
                    # Get current price
                    current_price = await self.get_current_price(pair)
                    if not current_price:
                        continue
                    
                    # Calculate metrics
                    entry_price = position['entry_price']
                    profit_loss = (current_price - entry_price) * position['quantity']
                    profit_percent = ((current_price - entry_price) / entry_price) * 100
                    position_age_minutes = (time.time() - position['entry_time']) / 60
                    
                    # Get position info
                    has_api_data = 'api_data' in position and position['api_data']
                    is_momentum = position.get('momentum_trade', False)
                    
                    # Log position status
                    position_type = "MOMENTUM" if is_momentum else "REGULAR"
                    api_status = "API" if has_api_data else "No API"
                    
                    logging.info(f"{pair} status: {profit_percent:+.1f}% after {position_age_minutes:.0f}min "
                            f"[{position_type}] ({api_status})")
                    
                    # ===== CHANGE 3: USE NEW PROFIT TAKING LOGIC =====
                    should_sell = await self.should_take_profit(pair, position, current_price)
                    
                    if should_sell:
                        positions_to_close.append((pair, {"sell_signal": True, "signal_strength": 0.9}))
                        continue
                    
                    # Check stop loss (keep existing logic but wider stops)
                    stop_loss_pct = config.MOMENTUM_STOP_LOSS if is_momentum else config.QUICK_STOP_LOSS
                    if profit_percent <= stop_loss_pct:
                        logging.info(f"Stop loss triggered for {pair} at {profit_percent:.2f}%")
                        positions_to_close.append((pair, {"sell_signal": True, "signal_strength": 0.95}))
                        continue
                    
                    # ===== CHANGE 4: UPDATE TRAILING STOPS =====
                    if config.ENABLE_TRAILING_STOPS and pair in self.trailing_stops:
                        trail_data = self.trailing_stops[pair]
                        
                        # Update highest price
                        if current_price > trail_data['highest_price']:
                            trail_data['highest_price'] = current_price
                            # Update stop price
                            trail_data['stop_price'] = current_price * (1 - trail_data['trailing_pct']/100)
                            logging.debug(f"Updated trailing stop for {pair}: ${trail_data['stop_price']:.4f}")
                        
                        # Check if trailing stop hit
                        if current_price <= trail_data['stop_price'] and current_price > trail_data['activation_price']:
                            logging.info(f"Trailing stop hit for {pair} at {profit_percent:.2f}%")
                            positions_to_close.append((pair, {"sell_signal": True, "signal_strength": 0.9}))
                    
                except Exception as e:
                    logging.error(f"Error monitoring position {pair}: {str(e)}")
            
            # Execute all closes
            for pair, analysis in positions_to_close:
                try:
                    await self.execute_sell(pair, analysis)
                except Exception as e:
                    logging.error(f"Error closing position {pair}: {str(e)}")
            
        except Exception as e:
            logging.error(f"Error in position monitoring: {str(e)}")

    async def cleanup(self):
        try:
            # Close global API client first
            if hasattr(self, 'global_api_client') and self.global_api_client:
                await self.global_api_client.close()
                logging.info("Global API client closed")
            
            if hasattr(self, 'strategy') and hasattr(self.strategy, 'close'):
                await self.strategy.close()
        except Exception as e:
            logging.error(f"Error during bot cleanup: {str(e)}")



    async def _check_api_stop_loss(self, pair: str, current_price: float) -> bool:
        """Check if API-recommended stop loss should trigger"""
        if not hasattr(self, 'api_stop_losses') or pair not in self.api_stop_losses:
            return False
        
        stop_data = self.api_stop_losses[pair]
        stop_price = stop_data['stop_loss_price']
        
        # Check if stop loss triggered
        position = self.active_positions.get(pair, {})
        is_buy_position = True  # Assuming buy positions for now
        
        if is_buy_position and current_price <= stop_price:
            return True
        elif not is_buy_position and current_price >= stop_price:
            return True
        
        return False

    async def _check_api_take_profits(self, pair: str, current_price: float, profit_percent: float) -> bool:
        """Check and execute API-recommended take profits"""
        if not hasattr(self, 'api_take_profits') or pair not in self.api_take_profits:
            return False
        
        tp_data = self.api_take_profits[pair]
        take_profits = tp_data['take_profits']
        
        for tp in take_profits:
            if not tp.get('executed', False) and current_price >= tp['price']:
                # Execute partial take profit
                percentage = tp['percentage']
                logging.info(f"API Take Profit {tp['level']} triggered for {pair}: "
                           f"${tp['price']:.4f} ({percentage}% of position)")
                
                # Mark as executed
                tp['executed'] = True
                tp['execution_time'] = time.time()
                tp['execution_price'] = current_price
                
                # Execute partial sell (implementation depends on your partial sell logic)
                if percentage < 100:
                    await self._execute_partial_sell_api(pair, percentage / 100)
                else:
                    await self.execute_sell(pair, {"sell_signal": True, "signal_strength": 0.8})
                    return True
        
        return False

    async def _execute_partial_sell_api(self, pair: str, percentage: float):
        """Execute partial sell based on API take profit recommendation"""
        try:
            if pair not in self.active_positions:
                return False
            
            position = self.active_positions[pair]
            total_quantity = position['quantity']
            sell_quantity = total_quantity * percentage
            
            # Execute partial sell (simplified - use your existing partial sell logic)
            logging.info(f"Executing {percentage*100:.1f}% partial sell for {pair}")
            
            # Update position quantity
            remaining_quantity = total_quantity - sell_quantity
            self.active_positions[pair]['quantity'] = remaining_quantity
            
            return True
            
        except Exception as e:
            logging.error(f"Error executing API partial sell: {str(e)}")
            return False
        
    async def opportunity_scan_task(self):
        """Dedicated task for continuous opportunity scanning"""
        try:
            while True:
                try:
                    # Run opportunity scan
                    opportunities = await self.opportunity_scanner.scan_for_opportunities()
                    
                    if opportunities:
                        top_opps = opportunities[:5]
                        logging.info(f"Top opportunities: {[(o['symbol'], f'{o['score']:.2f}') for o in top_opps]}")

                        
                        # Store opportunities for the trading task to process
                        self.latest_opportunities = opportunities
                        self.last_opportunity_scan = time.time()
                    
                    # Wait before next scan (more frequent than regular analysis)
                    await asyncio.sleep(45)  # Scan every 45 seconds
                    
                except Exception as inner_e:
                    logging.error(f"Error in opportunity scan: {str(inner_e)}")
                    await asyncio.sleep(60)
                    
        except Exception as e:
            logging.error(f"Error in opportunity scan task: {str(e)}")
            await asyncio.sleep(60)
            asyncio.create_task(self.opportunity_scan_task())


    async def secure_profits(self, take_top=2):
        """Take partial profits when daily target is reached"""
        # Only proceed if we have positions
        if not self.active_positions:
            return
            
        logging.info("Securing profits from open positions...")
        
        # Sort positions by profit (highest first)
        profit_sorted = []
        for pair, position in self.active_positions.items():
            profit_percent = self.calculate_profit_percent(pair)
            profit_sorted.append((pair, profit_percent))
            
        profit_sorted.sort(key=lambda x: x[1], reverse=True)
        
        # Take profits from the most profitable positions
        profits_taken = 0
        for pair, profit in profit_sorted:
            if profit > 1.0:  # Only take profits from positions in profit
                logging.info(f"Taking profits from {pair} at {profit:.2f}%")
                await self.execute_sell(pair, {"sell_signal": True, "signal_strength": 0.9})
                profits_taken += 1
                
                # Stop after taking profits from specified number of positions
                if profits_taken >= take_top:
                    break
                    
                # Brief pause between orders
                await asyncio.sleep(1)
        
        logging.info(f"Took profits on {profits_taken} positions")
    
    async def handle_drawdown_protection(self, protection_level):
        """Handle drawdown protection events"""
        if protection_level == "EMERGENCY_RISK_REDUCTION":
            # Most severe drawdown - liquidate 75% of all positions
            await self.reduce_risk_exposure(0.75)
            logging.warning("EMERGENCY RISK REDUCTION: Liquidated 75% of positions")
            
        elif protection_level == "HIGH_ALERT_RISK_REDUCTION":
            # Severe drawdown - liquidate 50% of all positions
            await self.reduce_risk_exposure(0.5)
            logging.warning("HIGH ALERT: Liquidated 50% of positions")
            
        elif protection_level == "REDUCE_RISK":
            # Moderate drawdown - liquidate 30% of all positions
            await self.reduce_risk_exposure(0.3)
            logging.warning("RISK REDUCTION: Liquidated 30% of positions")
            
        elif protection_level == "WARNING":
            # Warning level - take partial profits from best performing positions
            await self.secure_profits(take_top=1)  # Take profit from top performing position
            logging.warning("DRAWDOWN WARNING: Taking partial profits")
    
    async def reduce_risk_exposure(self, reduction_pct):
        """Reduce risk exposure by selling a portion of positions"""
        if not self.active_positions:
            logging.info("No active positions to reduce")
            return
            
        logging.warning(f"Reducing risk exposure by {reduction_pct*100:.0f}% across all positions")
        
        # Sort positions by profit (worst first to preserve better positions)
        positions_by_profit = []
        for pair, position in self.active_positions.items():
            profit_percent = self.calculate_profit_percent(pair)
            positions_by_profit.append((pair, profit_percent, position))
            
        # Sort ascending (worst first)
        positions_by_profit.sort(key=lambda x: x[1])
        
        total_positions = len(positions_by_profit)
        positions_to_close = int(total_positions * reduction_pct)
        
        # Close entire positions rather than partial sells for worst performers
        for i in range(min(positions_to_close, total_positions)):
            pair, profit, position = positions_by_profit[i]
            logging.info(f"Liquidating {pair} position at {profit:.2f}% profit")
            await self.execute_sell(pair, {"sell_signal": True, "signal_strength": 0.9})
            
        # If we need to close partial positions for remaining positions
        if positions_to_close < 1 and total_positions > 0:
            # Close partial amount of a single position
            pair, profit, position = positions_by_profit[0]  # Worst position
            quantity = position['quantity'] * reduction_pct
            
            logging.info(f"Reducing {pair} position by {reduction_pct*100:.0f}% at {profit:.2f}% profit")
            
            # Execute partial sell
            current_price = await self.get_current_price(pair)
            if current_price:
                # Calculate the amount to sell
                amount_to_sell = position['quantity'] * reduction_pct
                # Try to execute the partial sell
                await self.execute_partial_sell(pair, amount_to_sell)
    
    async def execute_partial_sell(self, pair: str, quantity: float, reason: str = "partial_tp"):
        """Execute a partial sell of a position"""
        try:
            if pair not in self.active_positions:
                logging.warning(f"No position found for {pair}")
                return False
            
            position = self.active_positions[pair]
            current_price = await self.get_current_price(pair)
            
            if not current_price:
                return False
            
            # Ensure we don't sell more than we have
            quantity = min(quantity, position['quantity'])
            
            if config.TEST_MODE:
                # Simulate partial sell
                position['quantity'] -= quantity
                position['value'] = position['quantity'] * current_price
                
                # Calculate profit for this partial
                profit = (current_price - position['entry_price']) * quantity
                
                logging.info(f"TEST: Partial sell {quantity:.6f} {pair} at ${current_price:.4f}, "
                            f"Profit: ${profit:.2f}, Remaining: {position['quantity']:.6f}")
                
                # If position is fully closed
                if position['quantity'] <= 0:
                    del self.active_positions[pair]
                    self.risk_manager.remove_position(pair)
                
                return True
                
            else:
                # Execute real partial sell
                order = await self.binance_client.create_order(
                    symbol=pair,
                    side='SELL',
                    type='MARKET',
                    quantity=quantity
                )
                
                if order and order.get('status') == 'FILLED':
                    # Update position
                    position['quantity'] -= float(order['executedQty'])
                    position['value'] = position['quantity'] * current_price
                    
                    # Log the partial sell
                    profit = (current_price - position['entry_price']) * float(order['executedQty'])
                    
                    logging.info(f"LIVE: Partial sell {order['executedQty']} {pair} executed, "
                                f"Profit: ${profit:.2f}, Remaining: {position['quantity']:.6f}")
                    
                    # If position is fully closed
                    if position['quantity'] <= 0:
                        del self.active_positions[pair]
                        self.risk_manager.remove_position(pair)
                    
                    return True
                    
        except Exception as e:
            logging.error(f"Error executing partial sell for {pair}: {str(e)}")
            return False
        
    async def initialize_stop_loss(self, pair, entry_price):
        """Initialize stop loss for a position"""
        try:
            # Calculate initial stop loss (2.5% below entry)
            stop_price = entry_price * 0.975  # 2.5% below entry
            
            # Store in trailing stops dictionary
            if not hasattr(self, 'trailing_stops'):
                self.trailing_stops = {}
                
            self.trailing_stops[pair] = {
                'entry_price': entry_price,
                'current_stop': stop_price,
                'initial_stop': stop_price,
                'highest_price': entry_price,
                'activated': False
            }
            
            logging.info(f"Stop loss for {pair} initialized at {stop_price:.6f} (2.5% below entry)")
            
        except Exception as e:
            logging.error(f"Error initializing stop loss for {pair}: {str(e)}")
    
    async def update_trailing_stop(self, pair, current_price):
        """Update trailing stop as price moves up"""
        if pair not in self.trailing_stops:
            return None
            
        stop_data = self.trailing_stops[pair]
        entry_price = stop_data['entry_price']
        current_stop = stop_data['current_stop']
        highest_price = stop_data['highest_price']
        activated = stop_data['activated']
        
        # Calculate current gain percentage
        current_gain_pct = ((current_price / entry_price) - 1) * 100
        
        # If price has moved up by activation threshold, activate trailing
        if not activated and current_gain_pct >= config.TRAILING_ACTIVATION_THRESHOLD * 100:
            self.trailing_stops[pair]['activated'] = True
            activated = True
            logging.info(f"Trailing stop activated for {pair} at {current_gain_pct:.2f}% gain")
        
        # If price is higher than previous highest, update the stop
        if current_price > highest_price:
            # Update highest seen price
            self.trailing_stops[pair]['highest_price'] = current_price
            
            # Only trail if we're activated
            if activated:
                # Calculate new stop price
                trailing_distance_pct = config.TRAILING_TP_PERCENTAGE  # 3% trailing distance
                new_stop = current_price * (1 - trailing_distance_pct)
                
                # Only move stop up, never down
                if new_stop > current_stop:
                    self.trailing_stops[pair]['current_stop'] = new_stop
                    logging.debug(f"Updated trailing stop for {pair} to {new_stop:.6f}")
        
        # Return current stop level
        return self.trailing_stops[pair]['current_stop']
    
    async def check_trailing_stop(self, pair, current_price):
        """Check if trailing stop has been triggered"""
        if pair not in self.trailing_stops:
            return False
            
        stop_price = self.trailing_stops[pair]['current_stop']
        
        # Update trailing stop first
        await self.update_trailing_stop(pair, current_price)
        
        # Get updated stop price
        stop_price = self.trailing_stops[pair]['current_stop']
        
        # Check if price has dropped below stop
        if current_price <= stop_price:
            logging.info(f"Trailing stop triggered for {pair} at {current_price:.6f} (stop: {stop_price:.6f})")
            return True
            
        return False
    
    async def initialize_trailing_take_profit(self, pair, entry_price):
        """Initialize trailing take profit for a position"""
        if not hasattr(self, 'trailing_tp_data'):
            self.trailing_tp_data = {}
            
        # Use standard trailing take profit settings
        activation_threshold = config.TRAILING_ACTIVATION_THRESHOLD
        trailing_percentage = config.TRAILING_TP_PERCENTAGE
        
        # Initialize data
        self.trailing_tp_data[pair] = {
            'activated': False,
            'initial_threshold': activation_threshold,
            'highest_price': entry_price,
            'trailing_stop_price': 0,
            'trailing_percentage': trailing_percentage,
            'entry_time': time.time(),
            'entry_price': entry_price
        }
        
        logging.info(f"Trailing take profit for {pair} initialized (activates at {activation_threshold*100:.1f}%, " +
                   f"trails at {trailing_percentage*100:.1f}%)")
    
    async def manage_trailing_take_profit(self, pair, current_price):
        """Manage trailing take profit for a position"""
        try:
            # Skip if trailing take-profit is disabled
            if not config.ENABLE_TRAILING_TP:
                return False
                
            # Skip if we don't have data for this pair
            if pair not in self.trailing_tp_data:
                return False
                
            # Get trailing take profit data
            tp_data = self.trailing_tp_data[pair]
            entry_price = tp_data['entry_price']
            
            # Calculate current profit percentage
            profit_pct = (current_price - entry_price) / entry_price
            
            # If trailing take-profit not yet activated, check if we've hit the threshold
            if not tp_data['activated']:
                # Get the current activation threshold
                activation_threshold = tp_data['initial_threshold']
                
                if profit_pct >= activation_threshold:
                    # Activate trailing take-profit
                    tp_data['activated'] = True
                    tp_data['highest_price'] = current_price
                    tp_data['trailing_stop_price'] = current_price * (1 - tp_data['trailing_percentage'])
                    
                    logging.info(f"ACTIVATED trailing take-profit for {pair} at {profit_pct*100:.2f}% profit")
                    
                    # We've just activated, not triggered yet
                    return False
            else:
                # Trailing take-profit is active, update if we have a new highest price
                if current_price > tp_data['highest_price']:
                    old_stop = tp_data['trailing_stop_price']
                    tp_data['highest_price'] = current_price
                    tp_data['trailing_stop_price'] = current_price * (1 - tp_data['trailing_percentage'])
                    
                    # Only log if the stop price change is significant
                    if tp_data['trailing_stop_price'] - old_stop > 0.001 * old_stop:
                        logging.info(f"Updated {pair} trailing take-profit stop to ${tp_data['trailing_stop_price']:.4f}")
                
                # Check if current price has dropped below trailing stop
                if current_price <= tp_data['trailing_stop_price']:
                    logging.info(f"TRIGGERED trailing take-profit for {pair} at ${current_price:.4f}")
                    
                    # Clean up data
                    del self.trailing_tp_data[pair]
                    
                    # Signal to execute take-profit
                    return True
                    
            return False
            
        except Exception as e:
            logging.error(f"Error in trailing take-profit management for {pair}: {str(e)}")
            return False
    
    async def should_scale_position(self, pair, current_price, profit_percent):
        """Check if a position should be scaled up"""
        try:
            # Skip if position scaling is disabled
            if not config.ENABLE_POSITION_SCALING:
                return False, None
                
            # Skip if we don't have position data
            if pair not in self.active_positions:
                return False, None
                
            position = self.active_positions[pair]
            
            # Check if we've already scaled this position too many times
            if not hasattr(self, 'position_scales'):
                self.position_scales = {}
                
            if pair not in self.position_scales:
                self.position_scales[pair] = {
                    'scale_count': 0,
                    'last_scale_time': 0,
                    'original_size': position['position_size'],
                    'total_added': 0.0
                }
                
            # Don't scale if we've reached the maximum number of scale-ups
            if self.position_scales[pair]['scale_count'] >= config.MAX_SCALE_COUNT:
                return False, None
                
            # Don't scale more frequently than every 12 hours
            if (time.time() - self.position_scales[pair]['last_scale_time']) < (12 * 3600):
                return False, None
                
            # Only scale positions that are in profit
            if profit_percent < config.POSITION_SCALE_THRESHOLD * 100:
                return False, None
                
            # Calculate amount to add - 50% of the original position size
            scale_amount = self.position_scales[pair]['original_size'] * config.SCALE_FACTOR
            
            # Limit total scaling to 200% of original position
            if (self.position_scales[pair]['total_added'] + scale_amount) > (self.position_scales[pair]['original_size'] * 2):
                return False, None
                
            # Check market trend score from multi-timeframe analysis
            mtf_analysis = await self.market_analysis.get_multi_timeframe_analysis(pair)
            market_trend_score = mtf_analysis.get('mtf_trend', 0)
            
            # Only scale in favorable trend conditions
            if market_trend_score < config.MIN_MARKET_TREND_SCORE:
                return False, None
                
            # Prepare scale info
            scale_info = {
                'scale_amount': scale_amount,
                'market_trend_score': market_trend_score,
                'profit_percent': profit_percent
            }
            
            return True, scale_info
            
        except Exception as e:
            logging.error(f"Error checking position scaling for {pair}: {str(e)}")
            return False, None
    
    async def scale_up_position(self, pair, scale_info):
        """Scale up an existing position"""
        try:
            scale_amount = scale_info['scale_amount']
            
            # Get current price
            current_price = await self.get_current_price(pair)
            if current_price <= 0:
                return False
                
            # Calculate quantity to add
            quantity_to_add = scale_amount / current_price
            
            # Round to appropriate precision
            info = self.binance_client.get_symbol_info(pair)
            step_size = 0.00001  # Default
            for f in info['filters']:
                if f['filterType'] == 'LOT_SIZE':
                    step_size = float(f['stepSize'])
                    break
                    
            precision = 0
            while step_size < 1:
                step_size *= 10
                precision += 1
                
            quantity_to_add = round(quantity_to_add - (quantity_to_add % float(step_size)), precision)
            
            if quantity_to_add <= 0:
                logging.warning(f"Calculated scale-up quantity too small for {pair}: {quantity_to_add}")
                return False
                
            # Log the scale-up
            log_message = f"SCALING UP: {pair} by adding {quantity_to_add} units (${scale_amount:.2f})"
            logging.info(log_message)
            self.trade_logger.info(log_message)
            
            # Generate a trade ID
            trade_id = f"scale_{int(time.time())}"
            
            if config.TEST_MODE:
                # Simulate order in test mode
                # Update position with increased quantity
                position = self.active_positions[pair]
                new_quantity = position['quantity'] + quantity_to_add
                self.active_positions[pair]['quantity'] = new_quantity
                
                # Record the scale-up trade
                await self.db_manager.execute_query(
                    """
                    INSERT INTO trades
                    (trade_id, pair, type, price, quantity, value, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        trade_id, pair, 'scale_up', current_price, quantity_to_add,
                        scale_amount, datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    ),
                    commit=True
                )
                
                # Update position in database
                if 'id' in position:
                    await self.db_manager.execute_query(
                        """
                        UPDATE positions
                        SET quantity = ?
                        WHERE id = ?
                        """,
                        (new_quantity, position['id']),
                        commit=True
                    )
                
                # Update scaling data
                self.position_scales[pair]['scale_count'] += 1
                self.position_scales[pair]['last_scale_time'] = time.time()
                self.position_scales[pair]['total_added'] += scale_amount
                
                logging.info(f"Successfully scaled up {pair} position ({self.position_scales[pair]['scale_count']}/{config.MAX_SCALE_COUNT})")
                return True
                
            else:
                # Execute real order
                try:
                    order = self.binance_client.order_market_buy(
                        symbol=pair,
                        quantity=quantity_to_add
                    )
                    
                    if order and order.get('status') == 'FILLED':
                        # Get actual execution details
                        actual_price = float(order['fills'][0]['price'])
                        actual_quantity = float(order['executedQty'])
                        actual_value = actual_price * actual_quantity
                        
                        # Record the scale-up trade
                        await self.db_manager.execute_query(
                            """
                            INSERT INTO trades
                            (trade_id, pair, type, price, quantity, value, timestamp)
                            VALUES (?, ?, ?, ?, ?, ?, ?)
                            """,
                            (
                                trade_id, pair, 'scale_up', actual_price, actual_quantity,
                                actual_value, datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                            ),
                            commit=True
                        )
                        
                        # Update position with increased quantity
                        position = self.active_positions[pair]
                        new_quantity = position['quantity'] + actual_quantity
                        self.active_positions[pair]['quantity'] = new_quantity
                        
                        # Update position in database
                        if 'id' in position:
                            await self.db_manager.execute_query(
                                """
                                UPDATE positions
                                SET quantity = ?
                                WHERE id = ?
                                """,
                                (new_quantity, position['id']),
                                commit=True
                            )
                        
                        # Update scaling data
                        self.position_scales[pair]['scale_count'] += 1
                        self.position_scales[pair]['last_scale_time'] = time.time()
                        self.position_scales[pair]['total_added'] += actual_value
                        
                        logging.info(f"Successfully scaled up {pair} position ({self.position_scales[pair]['scale_count']}/{config.MAX_SCALE_COUNT})")
                        return True
                    else:
                        logging.error(f"Scale-up order failed: {order}")
                        return False
                except Exception as e:
                    logging.error(f"Error executing scale-up order: {str(e)}")
                    return False
                    
        except Exception as e:
            logging.error(f"Error scaling up position for {pair}: {str(e)}")
            return False
        
    def _add_position_synchronized(self, pair: str, position_data: dict):
        """Add position with synchronized tracking to prevent count mismatches"""
        try:
            # Critical: Add to both tracking systems atomically
            self.active_positions[pair] = position_data
            position_value = position_data.get('position_value', position_data.get('value', 0))
            self.risk_manager.add_position(pair, position_value)
            
            # Log for debugging
            logging.info(f"SYNC: Added position {pair}, Active: {len(self.active_positions)}, Risk Manager: {self.risk_manager.position_count}")
            
        except Exception as e:
            # If any error occurs, ensure both systems stay consistent
            logging.error(f"Error adding position {pair}: {str(e)}")
            if pair in self.active_positions:
                del self.active_positions[pair]
            self.risk_manager.remove_position(pair)
    
    def _remove_position_synchronized(self, pair: str):
        """Remove position with synchronized tracking to prevent count mismatches"""
        try:
            # Critical: Remove from both tracking systems atomically
            if pair in self.active_positions:
                del self.active_positions[pair]
            self.risk_manager.remove_position(pair)
            
            # Log for debugging
            logging.info(f"SYNC: Removed position {pair}, Active: {len(self.active_positions)}, Risk Manager: {self.risk_manager.position_count}")
            
        except Exception as e:
            logging.error(f"Error removing position {pair}: {str(e)}")
    
    def _validate_position_sync(self):
        """Validate that both position tracking systems are in sync"""
        active_count = len(self.active_positions)
        risk_count = self.risk_manager.position_count
        
        if active_count != risk_count:
            logging.error(f"POSITION SYNC MISMATCH: Active={active_count}, Risk Manager={risk_count}")
            logging.error(f"Active positions: {list(self.active_positions.keys())}")
            logging.error(f"Risk manager positions: {list(self.risk_manager.position_values.keys())}")
            
            # Force synchronization - trust active_positions as source of truth
            self.risk_manager.position_count = active_count
            self.risk_manager.position_values = {
                pair: pos.get('position_value', pos.get('value', 0)) 
                for pair, pos in self.active_positions.items()
            }
            logging.info(f"FORCED SYNC: Both systems now at {active_count} positions")
            
        return active_count



    async def _execute_api_enhanced_buy(self, pair: str, analysis: dict) -> bool:
        """Execute buy order with API-enhanced parameters"""
        try:
            # Get API data
            api_data = analysis.get('api_data', {})
            
            # Get current price
            current_price = await self.get_current_price(pair)
            if not current_price:
                return False
            
            # ===== CHANGE 5: USE DYNAMIC POSITION SIZING =====
            # Extract signal strength from analysis
            signal_strength = analysis.get('signal_strength', 0.5)
            
            # Extract confidence for enhanced position sizing
            confidence = analysis.get('confidence', api_data.get('confidence', 50))
            
            # Call with correct parameters: pair, signal_strength, current_price, confidence
            quantity, position_value = await self.calculate_dynamic_position_size(
                pair, signal_strength, current_price, confidence=confidence
            )
            
            if quantity <= 0 or position_value <= 0:
                logging.info(f"Position size 0 for {pair} (qty: {quantity}, value: {position_value}), skipping trade")
                return False
            
            # Enhanced logging with all confidence factors
            api_confidence = api_data.get('confidence', 50)
            blended_confidence = analysis.get('confidence', 50)
            asset_score = analysis.get('asset_score', 3.0)
            api_reason = api_data.get('reasoning', 'technical_analysis')
            
            # Log the enhanced trade setup
            logging.info(f"ENHANCED BUY: {pair} - Qty: {quantity:.6f}, Value: ${position_value:.2f}, "
                        f"API: {api_confidence:.1f}%, Blended: {blended_confidence:.1f}%, "
                        f"Asset Score: {asset_score:.2f}, Reason: {api_reason}")
            
            # Execute the order (test mode or live)
            if config.TEST_MODE:
                # Simulate successful order
                order_id = f"test_{int(time.time())}"
                
                # Record position using synchronized method
                position_data = {
                    'entry_price': current_price,
                    'quantity': quantity,
                    'entry_time': time.time(),
                    'position_value': position_value,  # Use consistent key name
                    'signal_source': 'api_enhanced',
                    'signal_strength': analysis.get('signal_strength', 0),
                    'api_data': api_data,
                    'order_id': order_id,
                    'momentum_trade': analysis.get('momentum_trade', False)
                }
                
                # Use synchronized position management to prevent 6/5 position bugs
                self._add_position_synchronized(pair, position_data)
                
                # Update risk manager
                self.risk_manager.add_position(pair, position_value)
                
                # Update equity in database to reflect position cost
                if config.TEST_MODE:
                    current_equity = await self.get_real_time_equity()
                    await self.update_equity_in_db(current_equity)
                
                # ===== CHANGE 6: SET SMART TRAILING STOP =====
                await self.set_smart_trailing_stop(pair, current_price)
                
                # Set API-based stop loss and take profits if available
                if api_data:
                    # Calculate stop loss price (1.5% below entry)
                    stop_loss_price = current_price * (1 + config.QUICK_STOP_LOSS / 100)
                    await self._set_api_stop_loss(pair, current_price, stop_loss_price)
                    
                    # Set take profits if API provides exit levels
                    if isinstance(api_data, dict) and 'exit_levels' in api_data:
                        await self._set_api_take_profits(pair, current_price, api_data['exit_levels'])
                    else:
                        # Set default take profit levels
                        default_exit_levels = {
                            'take_profit_1': current_price * 1.015,  # 1.5% profit
                            'take_profit_2': current_price * 1.025,  # 2.5% profit
                            'take_profit_3': current_price * 1.035   # 3.5% profit
                        }
                        await self._set_api_take_profits(pair, current_price, default_exit_levels)
                
                logging.info(f"TEST: Buy order for {pair} simulated successfully")
                return True
                
            else:
                # Execute real order
                try:
                    order = await self.binance_client.create_order(
                        symbol=pair,
                        side='BUY',
                        type='MARKET',
                        quantity=quantity
                    )
                    
                    if order and order.get('status') == 'FILLED':
                        # Get actual fill price
                        fill_price = float(order.get('fills', [{}])[0].get('price', current_price))
                        
                        # Record position using synchronized method
                        position_data = {
                            'entry_price': fill_price,
                            'quantity': float(order['executedQty']),
                            'entry_time': time.time(),
                            'position_value': float(order['cummulativeQuoteQty']),
                            'signal_source': 'api_enhanced',
                            'signal_strength': analysis.get('signal_strength', 0),
                            'api_data': api_data,
                            'order_id': order['orderId'],
                            'momentum_trade': analysis.get('momentum_trade', False)
                        }
                        
                        # Use synchronized position management to prevent 6/5 position bugs
                        self._add_position_synchronized(pair, position_data)
                        
                        # Update risk manager
                        self.risk_manager.add_position(pair, float(order['cummulativeQuoteQty']))
                        
                        # Update equity in database to reflect position cost
                        if config.TEST_MODE:
                            current_equity = await self.get_real_time_equity()
                            await self.update_equity_in_db(current_equity)
                        
                        # ===== CHANGE 7: SET SMART TRAILING STOP =====
                        await self.set_smart_trailing_stop(pair, fill_price)
                        
                        logging.info(f"LIVE: Buy order for {pair} executed successfully")
                        return True
                    else:
                        logging.error(f"Buy order failed: {order}")
                        return False
                except Exception as e:
                    logging.error(f"Error executing buy order: {str(e)}")
                    return False
        
        except Exception as e:
            logging.error(f"Error executing API-enhanced buy for {pair}: {str(e)}")
            return False



