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

from advanced_indicators import AdvancedIndicators
from coin_gecko_ai import CoinGeckoAI, DummyCoinGeckoAI 
from market_analysis import MarketAnalysis
from order_book import OrderBookAnalysis
from db_manager import DatabaseManager
from asset_selection import AssetSelection
from correlation_analysis import CorrelationAnalysis
from enhanced_strategy import EnhancedStrategy
from risk_manager import RiskManager
from performance_tracker import PerformanceTracker
from opportunity_scanner import OpportunityScanner

import config

class HybridTradingBot:
    def __init__(self):
        """Initialize the hybrid trading bot with all components"""
        # Set up logging
        self._setup_logging()
        
        # Initialize Binance client
        self.binance_client = Client(
            config.BINANCE_API_KEY, 
            config.BINANCE_API_SECRET,
            testnet=config.TEST_MODE
        )
        
        # Initialize CoinGecko AI client
        if hasattr(config, 'ENABLE_COINGECKO') and config.ENABLE_COINGECKO:
            try:
                self.ai_client = CoinGeckoAI(config.COINGECKO_API_KEY)
                logging.info("CoinGecko AI client initialized")
            except Exception as e:
                logging.warning(f"Failed to initialize CoinGecko: {str(e)} - using fallback mode")
                self.ai_client = DummyCoinGeckoAI()
        else:
            # Create a dummy AI client
            self.ai_client = DummyCoinGeckoAI()
            logging.info("CoinGecko AI analysis disabled - using dummy client")


        self.opportunity_scanner = OpportunityScanner(self.binance_client, self.ai_client)
        
        # Quick trade settings for momentum catching
        self.quick_trade_mode = True
        self.momentum_trades = {}  # Track momentum-based trades

        # Keep nebula reference for backward compatibility with existing code
        self.nebula = self.ai_client
        
        # Initialize database manager
        self.db_manager = DatabaseManager(config.DB_PATH)
        
        # Update database schema
        self.db_manager.update_schema()
        
        # Initialize market analysis module
        self.market_analysis = MarketAnalysis(self.binance_client)
        
        # Initialize order book analysis
        self.order_book = OrderBookAnalysis(self.binance_client)
        
        # Initialize asset selection
        self.asset_selection = AssetSelection(self.binance_client, self.market_analysis)
        
        # Initialize correlation analysis
        self.correlation = CorrelationAnalysis(self.market_analysis)
        
        # Get initial equity
        self.initial_equity = self.get_total_equity()
        
        # Initialize strategy
        self.strategy = EnhancedStrategy(
            self.binance_client, 
            self.ai_client,  # Use ai_client instead of nebula
            self.market_analysis, 
            self.order_book
        )
        
         # Initialize advanced indicators (NEW)
        try:
            self.advanced_indicators = AdvancedIndicators()
            logging.info("Advanced indicators initialized successfully")
        except Exception as e:
            logging.warning(f"Failed to initialize advanced indicators: {str(e)}")
            self.advanced_indicators = None


        # Initialize risk manager
        self.risk_manager = RiskManager(self.initial_equity)
        
        # Initialize performance tracker
        self.performance_tracker = PerformanceTracker(config.DB_PATH)
        
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
        
        logging.info(f"Hybrid Trading Bot initialized with {self.initial_equity:.2f} equity")
        logging.info(f"Mode: {'TEST' if config.TEST_MODE else 'LIVE'} trading")
        logging.info(f"AI Provider: {'CoinGecko' if config.ENABLE_COINGECKO else 'Disabled'}")
        logging.info(f"Advanced Indicators: {'Taapi.io' if config.ENABLE_TAAPI and self.advanced_indicators else 'Disabled'}")
    
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
        if config.TEST_MODE:
            # Get equity from database in test mode
            equity_query = "SELECT value FROM bot_stats WHERE key='total_equity' LIMIT 1"
            result = self.db_manager.execute_query_sync(equity_query, fetch_one=True)
            
            if result:
                return float(result[0])
            else:
                # Set initial equity if not found
                self.db_manager.execute_query_sync(
                    "INSERT INTO bot_stats (key, value, last_updated) VALUES (?, ?, ?)",
                    ('total_equity', 1000.0, datetime.now().strftime('%Y-%m-%d %H:%M:%S')),
                    commit=True
                )
                return 1000.0  # Default initial equity
        
        # Use real account data for live mode
        account = self.binance_client.get_account()
        
        total = 0
        for balance in account['balances']:
            asset = balance['asset']
            free_amount = float(balance['free'])
            locked_amount = float(balance['locked'])
            total_amount = free_amount + locked_amount
            
            if total_amount > 0:
                if asset == 'USDT':
                    total += total_amount
                else:
                    # Get asset price in USDT
                    try:
                        ticker = self.binance_client.get_symbol_ticker(symbol=f"{asset}USDT")
                        price = float(ticker['price'])
                        total += total_amount * price
                    except:
                        # Skip if cannot get price
                        pass
        
        return total
    
    async def run(self):
        try:
            # Test API connections
            logging.info("Testing API connections...")
            
            # Test Binance connection
            server_time = self.binance_client.get_server_time()
            logging.info(f"Binance server time: {datetime.fromtimestamp(server_time['serverTime']/1000)}")
            
            # Test CoinGecko connection
            try:
                if hasattr(self.ai_client, 'api_available') and self.ai_client.api_available:
                    token = "ETH"
                    sentiment = await self.ai_client.get_sentiment_analysis(token)
                    if sentiment:
                        logging.info(f"CoinGecko AI connection successful: {sentiment.get('sentiment_score', 'N/A')}")
                    else:
                        logging.warning("CoinGecko API connected but no data returned - using fallback mode")
                else:
                    logging.warning("CoinGecko API not available - using fallback mode")
            except Exception as e:
                logging.warning(f"CoinGecko connection test failed: {str(e)} - using fallback mode")
            
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
    
    async def preload_historical_data(self):
        """Preload sufficient historical data for all needed timeframes"""
        logging.info("Preloading historical data for analysis...")

        # Get pairs to preload
        pairs = await self.asset_selection.get_trending_cryptos(limit=20)
        pairs = pairs + ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'ADAUSDT']  # Always include major pairs
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




    async def market_monitor_task(self):
        """Task to monitor market conditions"""
        try:
            while True:
                try:
                    # Update market breadth
                    self.market_state['breadth'] = await self.market_analysis.calculate_market_breadth()
                    
                    # Update correlation matrix
                    pairs = await self.asset_selection.get_available_pairs()
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
    
    async def trading_task(self):
        """Task to handle trading decisions"""
        try:
            while True:
                try:
                    start_time = time.time()
                    
                    # Update equity and risk parameters
                    current_equity = self.get_total_equity()
                    risk_status = self.risk_manager.update_equity(current_equity)
                    
                
                    open_positions = len(self.active_positions)
                    total_position_value = sum(pos.get('position_size', 0) for pos in self.active_positions.values())
                    position_percentage = (total_position_value / current_equity * 100) if current_equity > 0 else 0

                    # Format position details
                    position_details = []
                    for pair, pos in self.active_positions.items():
                        current_price = await self.get_current_price(pair)
                        entry_price = pos['entry_price']
                        profit_pct = ((current_price / entry_price) - 1) * 100 if entry_price > 0 else 0
                        position_details.append(f"{pair.replace('USDT', '')}:{profit_pct:+.1f}%")

                    positions_str = ", ".join(position_details) if position_details else "None"

                    # Log current status with position information
                    logging.info(f"Current equity: ${current_equity:.2f}, " +
                            f"Daily ROI: {risk_status['daily_roi']:.2f}%, " +
                            f"Drawdown: {risk_status['drawdown']:.2f}%, " +
                            f"Risk level: {risk_status['risk_level']}, " +
                            f"Positions: {open_positions}/{config.MAX_POSITIONS} " +
                            f"(${total_position_value:.2f}, {position_percentage:.1f}% of equity)")

                    if open_positions > 0:
                        logging.info(f"Active positions: {positions_str}")
                              
                    # Check for drawdown protection events
                    protection_level = self.risk_manager._check_drawdown_protection()
                    if protection_level:
                        await self.handle_drawdown_protection(protection_level)
                        # After taking protective actions, pause trading for a cycle
                        await asyncio.sleep(300)  # 5 minute pause after drawdown action
                        continue
                    
                    # Check daily ROI target
                    roi_status, roi_message = self.performance_tracker.check_daily_roi_target(
                        current_equity, self.initial_equity
                    )
                    
                    logging.info(roi_message)
                    
                    # If in severe recovery mode, skip trading this cycle
                    if risk_status.get('severe_recovery_mode', False):
                        logging.info("In SEVERE recovery mode - skipping trading cycle")
                        await asyncio.sleep(60)
                        continue
                        
                    # Get pairs to analyze - use optimal asset selection
                    pairs_to_analyze = await self.asset_selection.select_optimal_assets(limit=15)
                    
                    # Process fewer pairs in high volatility to avoid overtrading
                    if self.market_state['volatility'] == 'high':
                        pairs_to_analyze = pairs_to_analyze[:10]
                        logging.info("High volatility - limiting analysis to 10 pairs")
                    
                    # Analyze pairs in parallel
                    analysis_tasks = []
                    for pair in pairs_to_analyze:
                        analysis_tasks.append(self.analyze_and_trade(pair))
                    
                    # Wait for all analyses to complete
                    await asyncio.gather(*analysis_tasks)
                    
                    # Check if we've reached daily goal
                    if roi_status and risk_status['daily_roi'] >= config.TARGET_DAILY_ROI_MIN * 100:
                        logging.info("Daily ROI target achieved! Reducing risk exposure.")
                        # Take partial profits
                        await self.secure_profits()
                    
                    # Calculate processing time and log
                    processing_time = time.time() - start_time
                    logging.info(f"Trading cycle completed in {processing_time:.2f} seconds")
                    
                    # Adaptive sleep time based on processing
                    sleep_time = max(30, 60 - processing_time)
                    await asyncio.sleep(sleep_time)
                    
                except Exception as inner_e:
                    logging.error(f"Error in trading cycle: {str(inner_e)}")
                    await asyncio.sleep(60)

                opportunities = await self.opportunity_scanner.scan_for_opportunities()
                    
                    # Process high-score opportunities immediately
                high_priority_pairs = []
                for opp in opportunities:
                        if opp['score'] > 2.0:  # High priority threshold
                            high_priority_pairs.append(opp['symbol'])
                            # Mark as momentum trade
                            self.momentum_trades[opp['symbol']] = {
                                'type': opp['sources'][0],
                                'score': opp['score'],
                                'entry_time': time.time()
                            }
                    
                    # Combine with regular asset selection
                        regular_pairs = await self.asset_selection.select_optimal_assets(limit=10)
                    
                    # Prioritize opportunity pairs
                        pairs_to_analyze = high_priority_pairs[:10] + [p for p in regular_pairs if p not in high_priority_pairs][:5]
                        
                        # Log what we're analyzing
                        if high_priority_pairs:
                            logging.info(f"High priority opportunities: {', '.join(high_priority_pairs[:5])}")
                        
                        # Analyze pairs in parallel with priority
                        analysis_tasks = []
                        
                        # High priority pairs first
                        for pair in high_priority_pairs[:10]:
                            analysis_tasks.append(self.analyze_and_trade(pair, priority='high'))
                        
                        # Then regular pairs
                        for pair in regular_pairs[:5]:
                            if pair not in high_priority_pairs:
                                analysis_tasks.append(self.analyze_and_trade(pair, priority='normal'))
                        
                        # Wait for all analyses to complete
                        await asyncio.gather(*analysis_tasks)

                        # Faster cycle for momentum trading
                        processing_time = time.time() - start_time
                        sleep_time = max(15, 30 - processing_time)  # Faster 15-30 second cycles
                        await asyncio.sleep(sleep_time)
                        
                    
                    
        except Exception as e:
            logging.error(f"Error in trading task: {str(e)}")
            # Restart the task after a delay
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
                    # Get current equity
                    current_equity = self.get_total_equity()
                    
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
    


    async def calculate_dynamic_position_size(self, pair, signal_strength, current_price):
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
            
            # Get base position size from risk manager
            base_position_size = self.risk_manager._calculate_position_size(signal_strength)
            
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
            min_required = min_notional * 1.2  # 20% buffer above minimum
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




    async def analyze_and_trade(self, pair, priority='normal'):
        
        try:
            # Skip if this pair was recently processed (within 5 minutes)
            cache_key = f"processed_{pair}"
            current_time = time.time()
            cache_duration = 60  # 5 minutes

            #if hasattr(self, 'cache_expiry') and cache_key in self.cache_expiry:
             #   if self.cache_expiry[cache_key] > current_time:
              #      return
                
            # Initialize cache if needed
            if not hasattr(self, 'cache_expiry'):
                self.cache_expiry = {}

            is_momentum_trade = pair in self.momentum_trades
            momentum_data = self.momentum_trades.get(pair, {})
                
            # Check if we already have a position
            have_position = pair in self.active_positions
            
            # Extract token from pair
            token = pair.replace("USDT", "")
            
            # Get trading signal from AI client (CoinGecko)
            ai_signal = await self.ai_client.get_trading_signal(token)
            
            # Run multi-timeframe analysis
            mtf_analysis = await self.market_analysis.get_multi_timeframe_analysis(pair)
            
            # Get order book data
            order_book_data = await self.order_book.get_order_book_data(pair)
            
            # Get correlation data
            active_positions = list(self.active_positions.keys())
            correlation_data = {
                'portfolio_correlation': self.correlation.get_portfolio_correlation(pair, active_positions),
                'is_diversified': self.correlation.are_pairs_diversified(pair, active_positions)
            }
            
            # Get combined analysis from all sources
            analysis = await self.strategy.analyze_pair(
                pair,
                mtf_analysis=mtf_analysis,
                order_book_data=order_book_data,
                correlation_data=correlation_data,
                market_state=self.market_state,
                nebula_signal=ai_signal  # Pass the AI signal to the strategy (keep parameter name for compatibility)
            )
            
    
            # Log the analysis
            signal_type = "BUY" if analysis['buy_signal'] else ("SELL" if analysis['sell_signal'] else "NEUTRAL")
            logging.info(f"{pair} analysis: {signal_type} signal with {analysis['signal_strength']:.2f} strength")

            # Add detailed logging for weak signals
            if analysis['signal_strength'] > 0 and analysis['signal_strength'] < config.MIN_SIGNAL_STRENGTH:
                logging.info(f"  -> Signal too weak to trade: {analysis['signal_strength']:.2f} < {config.MIN_SIGNAL_STRENGTH}")

            # Store analysis
            self.analyzed_pairs[pair] = {
                "timestamp": time.time(),
                "analysis": analysis,
                "ai_signal": ai_signal
            }

            if is_momentum_trade:
                momentum_score = momentum_data.get('score', 0)
                if momentum_score > 2.0:
                    # Boost signal strength for high-score opportunities
                    analysis['signal_strength'] = min(0.95, analysis['signal_strength'] * 1.3)
                    analysis['source'] += f",momentum_{momentum_data.get('type', 'unknown')}"
            
            # ENHANCED: More aggressive entry for high priority
            if priority == 'high' and analysis['buy_signal']:
                # Lower threshold for high priority opportunities
                if analysis['signal_strength'] > 0.5:  # Lower from 0.6
                    analysis['signal_strength'] = min(0.9, analysis['signal_strength'] * 1.2)
            
            # Execute trades based on signals
            if analysis['buy_signal'] and not have_position:
                # Check risk management with adjusted parameters for momentum
                can_trade, trade_details = self.risk_manager.should_take_trade(
                    pair, analysis['signal_strength'], correlation_data
                )
                
                if can_trade:
                    # Adjust position size for momentum trades
                    position_size = trade_details['position_size']
                    if is_momentum_trade and momentum_score > 2.5:
                        position_size *= 1.2  # 20% larger for strong momentum
                    
                    success = await self.execute_buy(pair, position_size, analysis)
                    
                    if success:
                        self.risk_manager.add_position(pair, position_size)
                        
                        # Track momentum trade entry
                        if is_momentum_trade:
                            self.active_positions[pair]['momentum_trade'] = True
                            self.active_positions[pair]['momentum_type'] = momentum_data.get('type')
            
            elif analysis['sell_signal'] and have_position:
                # Check if we should take profit
                position = self.active_positions[pair]
                profit_percent = self.calculate_profit_percent(pair)
                position_age_hours = (time.time() - position['entry_time']) / 3600
                
                should_sell = self.risk_manager.should_take_profit(
                    pair, profit_percent, position_age_hours
                )
                
                if should_sell:
                    # Execute sell
                    success = await self.execute_sell(pair, analysis)
                    
                    if success:
                        # Update risk manager
                        self.risk_manager.remove_position(pair)
                else:
                    logging.info(f"Holding {pair} position with {profit_percent:.2f}% profit")
                    
            # Mark this pair as processed
            self.cache_expiry[cache_key] = current_time + 15  # Don't process again for 15 seconds
            
        except Exception as e:
            logging.error(f"Error analyzing {pair}: {str(e)}")
    
 
    
    async def get_current_price(self, pair: str) -> float:
        """Get current price of a trading pair with caching"""
        try:
            # Check cache first
            cache_key = f"price_{pair}"
            current_time = time.time()
            if hasattr(self, 'data_cache') and cache_key in self.data_cache and hasattr(self, 'cache_expiry') and self.cache_expiry.get(cache_key, 0) > current_time:
                return self.data_cache[cache_key]
                
            # Initialize cache if needed
            if not hasattr(self, 'data_cache'):
                self.data_cache = {}
            if not hasattr(self, 'cache_expiry'):
                self.cache_expiry = {}
                
            # Track API call
            self.track_api_call("get_symbol_ticker")

            # Get ticker info from Binance
            async with self.api_semaphore:
                ticker = self.binance_client.get_symbol_ticker(symbol=pair)
                price = float(ticker['price'])

            # Cache the price for a brief period
            self.data_cache[cache_key] = price
            self.cache_expiry[cache_key] = current_time + 30  # 30 second cache

            return price
        except Exception as e:
            logging.error(f"Error getting current price for {pair}: {str(e)}")
            # If available, return last known price from cache
            if hasattr(self, 'data_cache') and cache_key in self.data_cache:
                return self.data_cache[cache_key]
            # If all else fails, return 0
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
    
    async def execute_buy(self, pair, position_size, analysis):
        
        try:
            # Get current price
            current_price = await self.get_current_price(pair)
            if current_price <= 0:
                logging.error(f"Invalid price for {pair}: {current_price}")
                return False
            
            # Get signal strength from analysis
            signal_strength = analysis.get('signal_strength', 0.5)
            
            # Use dynamic position sizing
            quantity, actual_position_size = await self.calculate_dynamic_position_size(
                pair, signal_strength, current_price
            )
            
            if quantity <= 0:
                logging.error(f"Could not calculate valid quantity for {pair}")
                return False
            
            # Log the order details
            log_message = f"DYNAMIC BUY: {pair} - Qty: {quantity}, Price: ${current_price:.4f}, Value: ${actual_position_size:.2f}, Signal: {signal_strength:.2f}"
            logging.info(log_message)
            self.trade_logger.info(log_message)
            
            # Generate a trade ID
            trade_id = f"{int(time.time())}-{random.randint(1000, 9999)}"
            
            if config.TEST_MODE:
                # Simulate order in test mode
                order_id = f"test_{int(time.time())}"
                
                # Record the trade in database
                await self.db_manager.execute_query(
                    """
                    INSERT INTO trades
                    (trade_id, pair, type, price, quantity, value, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        trade_id, pair, 'buy', current_price, quantity,
                        actual_position_size, datetime.now().strftime('%Y-%m-%d %H:%M:%S')
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
                        pair, current_price, quantity,
                        datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 
                        'open', analysis.get('source', 'combined'),
                        signal_strength
                    ),
                    commit=True
                )
                
                # Record in active positions dictionary
                self.active_positions[pair] = {
                    "id": position_id,
                    "entry_price": current_price,
                    "quantity": quantity,
                    "entry_time": time.time(),
                    "position_size": actual_position_size,
                    "order_id": order_id,
                    "signal_source": analysis.get('source', 'combined'),
                    "signal_strength": signal_strength
                }
                
                # Initialize stop loss if needed
                await self.initialize_stop_loss(pair, current_price)
                
                # Initialize trailing take profit if enabled
                if config.ENABLE_TRAILING_TP:
                    await self.initialize_trailing_take_profit(pair, current_price)
                
                logging.info(f"TEST MODE: Simulated buy for {pair} executed successfully")
                return True
            else:
                # Execute real order
                try:
                    order = self.binance_client.order_market_buy(
                        symbol=pair,
                        quantity=quantity
                    )
                    
                    if order and order.get('status') == 'FILLED':
                        # ... rest of your existing real order handling code ...
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
        """Execute a sell order with enhanced tracking"""
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
                
                logging.info(f"TEST MODE: Simulated sell for {pair} executed successfully")
                success = True
                
            else:
                # Execute real order
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
                        
                        logging.info(f"Sell order for {pair} executed successfully")
                        success = True
                    else:
                        logging.error(f"Sell order failed: {order}")
                        success = False
                except Exception as e:
                    logging.error(f"Error executing sell order: {str(e)}")
                    success = False
            
            # TRACK RECENTLY TRADED - This is the key addition
            if success:
                # Initialize recently_traded if it doesn't exist
                if not hasattr(self, 'recently_traded'):
                    self.recently_traded = {}
                    
                # Mark this pair as recently traded
                self.recently_traded[pair] = time.time()
                
                # Clean old entries (keep only last hour)
                current_time = time.time()
                self.recently_traded = {
                    p: t for p, t in self.recently_traded.items() 
                    if current_time - t < 3600  # Keep for 1 hour
                }
                
                # Log recently traded pairs for debugging
                logging.debug(f"Recently traded pairs: {list(self.recently_traded.keys())}")
                
                # Also track profit/loss for this session
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
        """Calculate current profit percentage for a position"""
        if pair not in self.active_positions:
            return 0
            
        position = self.active_positions[pair]
        entry_price = position['entry_price']
        
        # Get current price
        try:
            ticker = self.binance_client.get_symbol_ticker(symbol=pair)
            current_price = float(ticker['price'])
            
            profit_percent = ((current_price / entry_price) - 1) * 100
            return profit_percent
        except:
            return 0
    
    async def monitor_positions(self):
        """Enhanced position monitoring with ultra-quick profit taking"""
        for pair, position in list(self.active_positions.items()):
            try:
                current_price = await self.get_current_price(pair)
                if current_price <= 0:
                    continue
                
                entry_price = position['entry_price']
                profit_percent = ((current_price / entry_price) - 1) * 100
                position_age_minutes = (time.time() - position['entry_time']) / 60
                
                # Check if this is a momentum trade
                is_momentum = position.get('momentum_trade', False)
                
                # ULTRA-QUICK PROFIT TAKING - Take ANY profit fast
                if profit_percent >= 0.3 and position_age_minutes < 5:
                    logging.info(f"Ultra-quick profit exit for {pair} at {profit_percent:.2f}% (5 min)")
                    await self.execute_sell(pair, {"sell_signal": True, "signal_strength": 0.95})
                    continue
                
                if profit_percent >= 0.5 and position_age_minutes < 10:
                    logging.info(f"Quick profit exit for {pair} at {profit_percent:.2f}% (10 min)")
                    await self.execute_sell(pair, {"sell_signal": True, "signal_strength": 0.9})
                    continue
                
                # MOMENTUM TRADES - Even quicker exits
                if is_momentum:
                    if profit_percent >= 1.0:  # 1% for momentum trades
                        logging.info(f"Momentum trade profit target hit for {pair} at {profit_percent:.2f}%")
                        await self.execute_sell(pair, {"sell_signal": True, "signal_strength": 0.9})
                        continue
                    elif position_age_minutes > 20 and profit_percent >= 0.5:  # 0.5% after 20 min
                        logging.info(f"Momentum trade time-based exit for {pair} at {profit_percent:.2f}%")
                        await self.execute_sell(pair, {"sell_signal": True, "signal_strength": 0.8})
                        continue
                    elif position_age_minutes > 45 and profit_percent >= 0:  # Break even after 45 min
                        logging.info(f"Momentum trade break-even exit for {pair} at {profit_percent:.2f}%")
                        await self.execute_sell(pair, {"sell_signal": True, "signal_strength": 0.7})
                        continue
                    
                    # Tighter stop loss for momentum trades
                    if profit_percent <= -1.0:  # 1% stop loss
                        logging.info(f"Momentum trade stop loss for {pair} at {profit_percent:.2f}%")
                        await self.execute_sell(pair, {"sell_signal": True, "signal_strength": 0.9})
                        continue
                
                # REGULAR TRADES - Still quick but slightly more patient
                else:
                    # Scaled profit targets based on time
                    if profit_percent >= 2.0:  # Take 2% anytime
                        logging.info(f"Profit target reached for {pair} at {profit_percent:.2f}%")
                        await self.execute_sell(pair, {"sell_signal": True, "signal_strength": 0.9})
                        continue
                    elif position_age_minutes > 30 and profit_percent >= 1.0:  # 1% after 30 min
                        logging.info(f"Time-based profit taking for {pair} at {profit_percent:.2f}%")
                        await self.execute_sell(pair, {"sell_signal": True, "signal_strength": 0.8})
                        continue
                    elif position_age_minutes > 60 and profit_percent >= 0.5:  # 0.5% after 1 hour
                        logging.info(f"Minimum profit exit for {pair} at {profit_percent:.2f}%")
                        await self.execute_sell(pair, {"sell_signal": True, "signal_strength": 0.7})
                        continue
                    elif position_age_minutes > 120 and profit_percent >= 0:  # Break even after 2 hours
                        logging.info(f"Break even exit for {pair} at {profit_percent:.2f}%")
                        await self.execute_sell(pair, {"sell_signal": True, "signal_strength": 0.6})
                        continue
                    elif position_age_minutes > 180 and profit_percent >= -0.5:  # Small loss after 3 hours
                        logging.info(f"Time stop for {pair} at {profit_percent:.2f}%")
                        await self.execute_sell(pair, {"sell_signal": True, "signal_strength": 0.6})
                        continue
                    
                    # Tighter stop loss
                    if profit_percent <= -1.5:  # 1.5% stop loss
                        logging.info(f"Stop loss triggered for {pair} at {profit_percent:.2f}%")
                        await self.execute_sell(pair, {"sell_signal": True, "signal_strength": 0.9})
                        continue
                
                # TRAILING STOP for positions in profit
                if profit_percent > 0.5:
                    # Check if price is dropping from recent high
                    if not hasattr(position, 'highest_profit'):
                        position['highest_profit'] = profit_percent
                    elif profit_percent > position['highest_profit']:
                        position['highest_profit'] = profit_percent
                    
                    # If profit dropped 0.3% from high, exit
                    if position.get('highest_profit', 0) - profit_percent > 0.3:
                        logging.info(f"Trailing stop triggered for {pair} at {profit_percent:.2f}% (high was {position['highest_profit']:.2f}%)")
                        await self.execute_sell(pair, {"sell_signal": True, "signal_strength": 0.8})
                        continue
                
                # Clean up momentum trade data if expired
                if is_momentum and pair in self.momentum_trades:
                    if time.time() - self.momentum_trades[pair]['entry_time'] > 3600:  # 1 hour
                        del self.momentum_trades[pair]
                
                # Mark pair as recently traded after selling to prevent immediate rebuy
                if not hasattr(self, 'recently_traded'):
                    self.recently_traded = {}
                
            except Exception as e:
                logging.error(f"Error monitoring position {pair}: {str(e)}")
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
    
    async def execute_partial_sell(self, pair, amount_to_sell):
        """Sell part of a position"""
        if pair not in self.active_positions:
            logging.warning(f"Cannot execute partial sell for {pair}: position not found")
            return False
            
        position = self.active_positions[pair]
        total_quantity = position['quantity']
        
        if amount_to_sell >= total_quantity:
            # If selling almost everything, sell entire position
            return await self.execute_sell(pair, {"sell_signal": True, "signal_strength": 0.9})
        
        # Get current price
        current_price = await self.get_current_price(pair)
        if current_price <= 0:
            logging.error(f"Invalid price for {pair}: {current_price}")
            return False
        
        # Calculate new quantity after partial sell
        remaining_quantity = total_quantity - amount_to_sell
        
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
            
        amount_to_sell = round(amount_to_sell, precision)
        remaining_quantity = round(remaining_quantity, precision)
        
        # Log the order details
        log_message = f"PARTIAL SELL: {pair} - {amount_to_sell} of {total_quantity} at approx. ${current_price}"
        logging.info(log_message)
        self.trade_logger.info(log_message)
        
        # Generate a trade ID
        trade_id = f"{int(time.time())}-{random.randint(1000, 9999)}"
        
        if config.TEST_MODE:
            # Simulate partial sell in test mode
            # Calculate profit/loss for this portion
            entry_price = position['entry_price']
            partial_profit_loss = (current_price - entry_price) * amount_to_sell
            
            # Record the trade in database
            await self.db_manager.execute_query(
                """
                INSERT INTO trades
                (trade_id, pair, type, price, quantity, value, profit_loss, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    trade_id, pair, 'partial_sell', current_price, amount_to_sell,
                    current_price * amount_to_sell, partial_profit_loss,
                    datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                ),
                commit=True
            )
            
            # Update position with reduced quantity
            self.active_positions[pair]['quantity'] = remaining_quantity
            
            # Update position in database
            if 'id' in position:
                await self.db_manager.execute_query(
                    """
                    UPDATE positions
                    SET quantity = ?
                    WHERE id = ?
                    """,
                    (remaining_quantity, position['id']),
                    commit=True
                )
            
            logging.info(f"TEST MODE: Simulated partial sell for {pair} executed successfully")
            return True
        else:
            # Execute real order for the partial amount
            try:
                order = self.binance_client.order_market_sell(
                    symbol=pair,
                    quantity=amount_to_sell
                )
                
                if order and order.get('status') == 'FILLED':
                    # Get actual execution details
                    actual_price = float(order['fills'][0]['price'])
                    actual_quantity = float(order['executedQty'])
                    actual_value = actual_price * actual_quantity
                    
                    # Calculate profit/loss for this portion
                    entry_price = position['entry_price']
                    partial_profit_loss = (actual_price - entry_price) * actual_quantity
                    
                    # Record the trade in database
                    await self.db_manager.execute_query(
                        """
                        INSERT INTO trades
                        (trade_id, pair, type, price, quantity, value, profit_loss, timestamp)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            trade_id, pair, 'partial_sell', actual_price, actual_quantity,
                            actual_value, partial_profit_loss,
                            datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        ),
                        commit=True
                    )
                    
                    # Update the position with the new quantity
                    new_quantity = total_quantity - actual_quantity
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
                    
                    logging.info(f"Partial sell order for {pair} executed successfully")
                    return True
                else:
                    logging.error(f"Partial sell order failed: {order}")
                    return False
            except Exception as e:
                logging.error(f"Error executing partial sell: {str(e)}")
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