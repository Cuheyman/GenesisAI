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
import aiohttp
from typing import Dict, List, Any, Optional, Tuple

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

# Momentum system imports
try:
    from momentum.enhanced_momentum_taapi import EnhancedMomentumTaapiClient
    from momentum.high_winrate_entry_filter import HighWinRateEntryFilter, EntrySignalStrength
    from momentum.momentum_strategy_config import momentum_config
    from momentum.momentum_performance_optimizer import PerformanceOptimizer
    from momentum.momentum_bot_integration import MomentumTradingOrchestrator
    MOMENTUM_SYSTEM_AVAILABLE = True
    print("Momentum system imported successfully")
except ImportError as e:
    print(f"Momentum system import failed: {str(e)}")
    print("Falling back to original bot functionality")
    MOMENTUM_SYSTEM_AVAILABLE = False

import config


class HybridTradingBot:
    def __init__(self, enable_momentum=True):
        """Initialize the hybrid trading bot with momentum system"""
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
            server_time = self.binance_client.get_server_time()
            local_time = int(time.time() * 1000)
            time_offset = server_time['serverTime'] - local_time
            
            if abs(time_offset) > 500:
                logging.info(f"Detected time offset: {time_offset}ms, adjusting...")
                self.binance_client = Client(
                    config.BINANCE_API_KEY, 
                    config.BINANCE_API_SECRET,
                    testnet=config.TEST_MODE
                )
                self.binance_client.timestamp_offset = time_offset
                logging.info(f"Timestamp offset set to {time_offset}ms")
            else:
                logging.info("System time is synchronized with Binance servers")
        except Exception as e:
            logging.warning(f"Could not sync timestamp: {e}. Trying manual offset...")
            self.binance_client.timestamp_offset = -2000
            logging.info("Applied manual timestamp offset: -2000ms")

        # Initialize database manager
        self.db_manager = DatabaseManager(config.DB_PATH)
        self.db_manager.update_schema()

        # Add momentum system initialization
        self.enable_momentum = enable_momentum and MOMENTUM_SYSTEM_AVAILABLE
        self.momentum_orchestrator = None
        self.momentum_performance_optimizer = None
        self.momentum_initialized = False
        self.momentum_trades = {}
        
        # TAAPI configuration for live trading
        self.taapi_config = {
            'api_secret': os.getenv('TAAPI_API_SECRET'),
            'base_url': 'https://api.taapi.io',
            'rate_limit_delay': 1.2,
            'timeout': 15,
            'max_retries': 3
        }
        
        if self.enable_momentum:
            logging.info("Momentum system enabled - will initialize on first run")
        else:
            logging.info("Using original trading strategy")
        
        # Initialize market analysis module
        self.market_analysis = MarketAnalysis(self.binance_client)
        
        # Initialize order book analysis
        self.order_book = OrderBookAnalysis(self.binance_client)
        
        # Initialize asset selection with dynamic capabilities
        self.asset_selection = DynamicAssetSelection(self.binance_client, self.market_analysis)
        
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
        
        # API-ENHANCED STRATEGY INITIALIZATION
        self.strategy = EnhancedStrategy(
            self.binance_client, 
            None,
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
        
        # Initialize opportunity scanner
        self.opportunity_scanner = OpportunityScanner(self.binance_client, None)
        
        # Quick trade settings for momentum catching
        self.quick_trade_mode = True

        # Initialize risk manager
        self.risk_manager = RiskManager(self.initial_equity)
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
            'breadth': 0.5,
            'volatility': 'normal',
            'trend': 'neutral',
            'liquidity': 'normal',
            'sentiment': 'neutral',
            'regime': 'NEUTRAL'
        }
        self.recently_traded = {}
        
        # Initialize session for API calls
        self.session = None
        
        # API call rate limit management
        self.api_call_tracker = {}
        self.api_semaphore = asyncio.Semaphore(10)
        
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
        logging.info(f"Mode: {'TEST' if config.TEST_MODE else 'LIVE'} trading")
        logging.info(f"Initial equity: ${self.initial_equity:.2f}")
        if self.enable_momentum:
            logging.info("Danish momentum strategy enabled")
    
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
        self.trade_logger.propagate = False

    async def initialize_momentum_system(self):
        """Initialize the momentum trading system"""
        if not self.enable_momentum or self.momentum_initialized:
            return True
        
        try:
            if not self.taapi_config['api_secret']:
                logging.error("TAAPI_API_SECRET not found - momentum system disabled")
                self.enable_momentum = False
                return False
            
            # Configuration for Danish strategy
            config_override = {
                'MIN_CONFLUENCE_SCORE': 75,
                'MIN_CONFIDENCE_SCORE': 80,
                'REQUIRE_VOLUME_CONFIRMATION': True,
                'REQUIRE_BREAKOUT_CONFIRMATION': True,
                'IGNORE_BEARISH_SIGNALS': True,
                'ONLY_BULLISH_ENTRIES': True,
                'TAAPI_RATE_LIMIT_DELAY': 1.2,
            }
            
            # Initialize momentum orchestrator
            self.momentum_orchestrator = MomentumTradingOrchestrator(
                bot_instance=self,
                config_override=config_override
            )
            
            # Initialize performance optimizer
            self.momentum_performance_optimizer = PerformanceOptimizer("momentum_performance.db")
            
            self.momentum_initialized = True
            logging.info("Momentum system initialized successfully")
            logging.info("Target win rate: 75-90% through selective entries")
            logging.info("🇩🇰 Danish strategy: Only bullish momentum with volume confirmation")
            
            return True
            
        except Exception as e:
            logging.error(f"Failed to initialize momentum system: {str(e)}")
            self.enable_momentum = False
            return False

    async def get_live_taapi_data(self, pair: str) -> Dict[str, Any]:
        """Get real TAAPI data using bulk queries for live trading"""
        try:
            bulk_query = {
                "secret": self.taapi_config['api_secret'],
                "construct": {
                    "exchange": "binance",
                    "symbol": pair.replace("USDT", "/USDT"),
                    "interval": "1h",
                    "indicators": [
                        {"indicator": "rsi", "period": 14, "id": "rsi_1h"},
                        {"indicator": "macd", "fastPeriod": 12, "slowPeriod": 26, "signalPeriod": 9, "id": "macd_1h"},
                        {"indicator": "ema", "period": 20, "id": "ema20_1h"},
                        {"indicator": "ema", "period": 50, "id": "ema50_1h"},
                        {"indicator": "ema", "period": 200, "id": "ema200_1h"},
                        {"indicator": "bbands", "period": 20, "nbdevup": 2, "nbdevdn": 2, "id": "bbands_1h"},
                        {"indicator": "atr", "period": 14, "id": "atr_1h"},
                        {"indicator": "adx", "period": 14, "id": "adx_1h"},
                        {"indicator": "stochrsi", "fastk": 3, "fastd": 3, "rsi_period": 14, "id": "stochrsi_1h"},
                        {"indicator": "mfi", "period": 14, "id": "mfi_1h"},
                        {"indicator": "obv", "id": "obv_1h"},
                        {"indicator": "vwap", "id": "vwap_1h"}
                    ]
                }
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.taapi_config['base_url']}/bulk",
                    json=bulk_query,
                    timeout=self.taapi_config['timeout']
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        return self._parse_taapi_bulk_response(data)
                    else:
                        logging.error(f"TAAPI API error: {response.status}")
                        return {}
                        
        except Exception as e:
            logging.error(f"Error getting TAAPI data for {pair}: {str(e)}")
            return {}

    def _parse_taapi_bulk_response(self, taapi_response: Dict) -> Dict[str, Any]:
        """Parse TAAPI bulk response into structured format"""
        parsed_data = {
            'primary': {},
            'short_term': {},
            'long_term': {}
        }
        
        try:
            if 'data' in taapi_response:
                for item in taapi_response['data']:
                    indicator_id = item.get('id', '')
                    result = item.get('result', {})
                    
                    if '_1h' in indicator_id:
                        indicator_name = indicator_id.replace('_1h', '')
                        
                        if indicator_name == 'rsi':
                            parsed_data['primary']['rsi'] = result.get('value', 0)
                        elif indicator_name == 'macd':
                            parsed_data['primary']['macd'] = {
                                'valueMACD': result.get('valueMACD', 0),
                                'valueMACDSignal': result.get('valueMACDSignal', 0),
                                'valueMACDHist': result.get('valueMACDHist', 0)
                            }
                        elif indicator_name == 'ema20':
                            parsed_data['primary']['ema20'] = result.get('value', 0)
                        elif indicator_name == 'ema50':
                            parsed_data['primary']['ema50'] = result.get('value', 0)
                        elif indicator_name == 'ema200':
                            parsed_data['primary']['ema200'] = result.get('value', 0)
                        elif indicator_name == 'bbands':
                            parsed_data['primary']['bbands'] = {
                                'valueUpperBand': result.get('valueUpperBand', 0),
                                'valueMiddleBand': result.get('valueMiddleBand', 0),
                                'valueLowerBand': result.get('valueLowerBand', 0)
                            }
                        elif indicator_name == 'atr':
                            parsed_data['primary']['atr'] = result.get('value', 0)
                        elif indicator_name == 'adx':
                            parsed_data['primary']['adx'] = result.get('value', 0)
                        elif indicator_name == 'stochrsi':
                            parsed_data['primary']['stochrsi'] = {
                                'valueFastK': result.get('valueFastK', 0),
                                'valueFastD': result.get('valueFastD', 0)
                            }
                        elif indicator_name == 'mfi':
                            parsed_data['primary']['mfi'] = result.get('value', 0)
                        elif indicator_name == 'obv':
                            parsed_data['primary']['obv'] = result.get('value', 0)
                        elif indicator_name == 'vwap':
                            parsed_data['primary']['vwap'] = result.get('value', 0)
            
            return parsed_data
            
        except Exception as e:
            logging.error(f"Error parsing TAAPI response: {str(e)}")
            return {'primary': {}, 'short_term': {}, 'long_term': {}}

    async def get_live_market_data(self, pair: str) -> Dict[str, Any]:
        """Get real market data for momentum analysis"""
        try:
            current_price = await self.get_current_price_live(pair)
            ticker_24h = await self.get_24h_ticker(pair)
            volume_analysis = await self.calculate_volume_analysis(pair)
            price_momentum = await self.calculate_price_momentum(pair)
            
            return {
                'current_price': current_price,
                'volume_analysis': volume_analysis,
                'price_momentum': price_momentum,
                'ticker_24h': ticker_24h,
                'market_hours': 'regular',
                'resistance_levels': {
                    'nearest_resistance': current_price * 1.02
                }
            }
            
        except Exception as e:
            logging.error(f"Error getting market data for {pair}: {str(e)}")
            return {'current_price': 0, 'volume_analysis': {}, 'price_momentum': {}}

    async def get_current_price_live(self, pair: str) -> float:
        """Get current live price from Binance"""
        try:
            if hasattr(self, 'binance_client') and self.binance_client:
                ticker = self.binance_client.get_symbol_ticker(symbol=pair)
                return float(ticker['price'])
            else:
                async with aiohttp.ClientSession() as session:
                    url = f"https://api.binance.com/api/v3/ticker/price?symbol={pair}"
                    async with session.get(url) as response:
                        if response.status == 200:
                            data = await response.json()
                            return float(data['price'])
                return 0.0
        except Exception as e:
            logging.error(f"Error getting current price for {pair}: {str(e)}")
            return 0.0

    async def get_24h_ticker(self, pair: str) -> Dict[str, Any]:
        """Get 24h ticker statistics"""
        try:
            if hasattr(self, 'binance_client') and self.binance_client:
                ticker = self.binance_client.get_ticker(symbol=pair)
                return {
                    'priceChangePercent': float(ticker.get('priceChangePercent', 0)),
                    'volume': float(ticker.get('volume', 0)),
                    'quoteVolume': float(ticker.get('quoteVolume', 0)),
                    'high': float(ticker.get('highPrice', 0)),
                    'low': float(ticker.get('lowPrice', 0))
                }
            return {}
        except Exception as e:
            logging.error(f"Error getting 24h ticker for {pair}: {str(e)}")
            return {}

    async def calculate_volume_analysis(self, pair: str) -> Dict[str, Any]:
        """Calculate real volume analysis"""
        try:
            if hasattr(self, 'binance_client') and self.binance_client:
                klines = self.binance_client.get_klines(
                    symbol=pair,
                    interval='1h',
                    limit=24
                )
                
                volumes = [float(k[5]) for k in klines]
                
                if len(volumes) >= 10:
                    recent_volume = np.mean(volumes[-3:])
                    baseline_volume = np.mean(volumes[:-3])
                    
                    volume_spike_ratio = recent_volume / baseline_volume if baseline_volume > 0 else 1.0
                    
                    return {
                        'volume_spike_ratio': volume_spike_ratio,
                        'recent_volume': recent_volume,
                        'baseline_volume': baseline_volume,
                        'breakout_volume_ratio': volume_spike_ratio
                    }
            
            return {'volume_spike_ratio': 1.0, 'breakout_volume_ratio': 1.0}
            
        except Exception as e:
            logging.error(f"Error calculating volume analysis for {pair}: {str(e)}")
            return {'volume_spike_ratio': 1.0, 'breakout_volume_ratio': 1.0}

    async def calculate_price_momentum(self, pair: str) -> Dict[str, Any]:
        """Calculate real price momentum"""
        try:
            if hasattr(self, 'binance_client') and self.binance_client:
                klines_1h = self.binance_client.get_klines(symbol=pair, interval='1h', limit=5)
                klines_4h = self.binance_client.get_klines(symbol=pair, interval='4h', limit=2)
                
                momentum = {}
                
                if len(klines_1h) >= 2:
                    price_1h_ago = float(klines_1h[-2][4])
                    current_price = float(klines_1h[-1][4])
                    momentum['1h'] = ((current_price / price_1h_ago) - 1) * 100
                
                if len(klines_4h) >= 2:
                    price_4h_ago = float(klines_4h[-2][4])
                    current_price = float(klines_4h[-1][4])
                    momentum['4h'] = ((current_price / price_4h_ago) - 1) * 100
                
                return momentum
            
            return {'1h': 0.0, '4h': 0.0}
            
        except Exception as e:
            logging.error(f"Error calculating price momentum for {pair}: {str(e)}")
            return {'1h': 0.0, '4h': 0.0}

    async def analyze_pair_with_momentum(self, pair: str, mtf_analysis=None, order_book_data=None, 
                                       correlation_data=None, market_state=None, nebula_signal=None, 
                                       global_api_client=None):
        """LIVE momentum analysis using real TAAPI data - no mocks"""
        
        if not await self.initialize_momentum_system():
            return await self.analyze_pair_original(pair, mtf_analysis, order_book_data, 
                                                   correlation_data, market_state, nebula_signal, 
                                                   global_api_client)
        
        try:
            # Get live momentum signal
            momentum_signal = await self.momentum_orchestrator.taapi_client.get_momentum_optimized_signal(pair)
            
            # Danish strategy: Only BUY signals allowed
            if momentum_signal.action != 'BUY':
                return {
                    'signal': 'hold',
                    'confidence': 0,
                    'reason': f'Danish strategy: Not bullish signal ({momentum_signal.action})',
                    'momentum_data': {
                        'action': momentum_signal.action,
                        'confidence': momentum_signal.confidence,
                        'momentum_strength': momentum_signal.momentum_strength,
                        'entry_quality': momentum_signal.entry_quality
                    }
                }
            
            # Get real market data
            market_data = await self.get_live_market_data(pair)
            
            # Get real TAAPI data
            taapi_data = await self.get_live_taapi_data(pair)
            
            # Import and use entry filter with real data
            entry_filter = HighWinRateEntryFilter(momentum_config)
            entry_metrics = await entry_filter.evaluate_entry_quality(pair, taapi_data, market_data)
            
            # High probability requirement for 75-90% win rate
            if not entry_metrics.is_high_probability:
                return {
                    'signal': 'hold',
                    'confidence': entry_metrics.overall_score,
                    'reason': f'Entry quality insufficient for high win rate - score: {entry_metrics.overall_score:.1f}',
                    'momentum_data': {
                        'overall_score': entry_metrics.overall_score,
                        'entry_quality': entry_metrics.entry_quality,
                        'signal_strength': entry_metrics.signal_strength.value,
                        'volume_confirmed': entry_metrics.has_volume_confirmation,
                        'breakout_confirmed': entry_metrics.has_breakout_confirmation,
                        'risk_factors': entry_metrics.risk_factors
                    }
                }
            
            # Create live trading signal
            current_price = market_data.get('current_price', 0)
            
            live_signal = {
                'signal': 'buy',
                'confidence': momentum_signal.confidence,
                'signal_strength': momentum_signal.momentum_strength.lower(),
                'entry_quality': entry_metrics.entry_quality.lower(),
                
                'momentum_data': {
                    'overall_score': entry_metrics.overall_score,
                    'breakout_type': momentum_signal.breakout_type,
                    'volume_confirmation': entry_metrics.has_volume_confirmation,
                    'momentum_confirmation': entry_metrics.has_momentum_confirmation,
                    'breakout_confirmation': entry_metrics.has_breakout_confirmation,
                    'indicators_aligned': momentum_signal.indicators_aligned,
                    'risk_reward_ratio': entry_metrics.risk_reward_ratio,
                    'entry_timing': entry_metrics.entry_timing,
                    'market_phase_fit': entry_metrics.market_phase_fit,
                    'is_high_probability': entry_metrics.is_high_probability,
                    'current_price': current_price,
                    'volume_spike_ratio': market_data.get('volume_analysis', {}).get('volume_spike_ratio', 1.0)
                },
                
                'api_data': {
                    'signal': 'BUY',
                    'confidence': momentum_signal.confidence,
                    'strength': momentum_signal.momentum_strength,
                    'strategy_type': 'Live Momentum Strategy',
                    'market_phase': entry_metrics.market_phase_fit,
                    'risk_reward_ratio': entry_metrics.risk_reward_ratio,
                    'enhanced_by': 'live_momentum_system',
                    'taapi_enabled': True,
                    'entry_price': current_price,
                    'stop_loss': current_price * 0.96,
                    'take_profit_1': current_price * 1.05,
                    'take_profit_2': current_price * 1.10,
                    'take_profit_3': current_price * 1.15,
                    'position_size_percent': 0.06 if entry_metrics.entry_quality.lower() == 'excellent' else 0.04
                },
                
                'reasoning': momentum_signal.reasons[:3],
                'timestamp': momentum_signal.timestamp.isoformat(),
                'buy_signal': True,
                'sell_signal': False,
                'hold_signal': False,
                'high_probability_entry': entry_metrics.is_high_probability,
                'momentum_confirmed': entry_metrics.has_momentum_confirmation,
                'danish_strategy_compliant': True,
                'source': 'live_momentum_system'
            }
            
            logging.info(f"""
            🚀 LIVE MOMENTUM SIGNAL for {pair}
            Action: {momentum_signal.action}
            Confidence: {momentum_signal.confidence:.1f}%
            Entry Quality: {entry_metrics.entry_quality}
            Overall Score: {entry_metrics.overall_score:.1f}
            Current Price: ${current_price:.2f}
            Volume Spike: {market_data.get('volume_analysis', {}).get('volume_spike_ratio', 1.0):.1f}x
            High Probability: {entry_metrics.is_high_probability}
            """)
            
            return live_signal
            
        except Exception as e:
            logging.error(f"Error in live momentum analysis for {pair}: {str(e)}")
            return await self.analyze_pair_original(pair, mtf_analysis, order_book_data, 
                                                   correlation_data, market_state, nebula_signal, 
                                                   global_api_client)

    async def analyze_pair_original(self, pair: str, mtf_analysis=None, order_book_data=None, 
                                   correlation_data=None, market_state=None, nebula_signal=None, 
                                   global_api_client=None):
        """Original analyze_pair method as fallback"""
        
        try:
            # Track that we're analyzing this pair
            self.api_stats['total_analyses'] += 1
            
            # Get API signal using global client
            signal = await self.strategy.get_api_signal(pair, self.global_api_client)
            
            if not signal:
                logging.debug(f"No API signal for {pair}")
                self.api_stats['api_errors'] += 1
                return {'signal': 'hold', 'confidence': 0, 'reason': 'No API signal'}
            
            # Track successful API call
            self.api_stats['successful_api_calls'] += 1
            self.api_stats['api_signals_used'] += 1
            
            # Check signal quality
            confidence = signal.get('confidence', 0) / 100
            signal_type = signal.get('signal', 'HOLD').upper()
            reason = signal.get('reasoning', 'unknown')
            
            min_confidence = getattr(config, 'MIN_SIGNAL_CONFIDENCE', 25) / 100
            if confidence < min_confidence:
                logging.info(f"Ignoring {pair} signal - confidence {confidence:.1%} below {min_confidence:.1%} threshold")
                return {'signal': 'hold', 'confidence': confidence * 100, 'reason': f'Confidence too low: {confidence:.1%}'}
            
            logging.info(f"Signal for {pair}: {signal_type} (confidence: {confidence:.1%}, reason: {reason})")
            
            # Convert to analysis format
            analysis = {
                'buy_signal': signal_type == 'BUY',
                'sell_signal': signal_type == 'SELL',
                'signal_strength': confidence,
                'confidence': confidence * 100,
                'api_data': signal,
                'source': 'api_signal',
                'reasoning': [reason]
            }
            
            return analysis
                
        except Exception as e:
            logging.error(f"Error in original analyze_pair for {pair}: {str(e)}")
            self.api_stats['api_errors'] += 1
            return {'signal': 'hold', 'confidence': 0, 'reason': f'Error: {str(e)}'}

    # Main analyze_pair method that chooses momentum or original
    async def analyze_pair(self, pair: str, mtf_analysis=None, order_book_data=None, 
                          correlation_data=None, market_state=None, nebula_signal=None, 
                          global_api_client=None):
        """Main analyze_pair - uses live momentum when enabled"""
        if self.enable_momentum:
            return await self.analyze_pair_with_momentum(
                pair, mtf_analysis, order_book_data, correlation_data, 
                market_state, nebula_signal, global_api_client
            )
        else:
            return await self.analyze_pair_original(
                pair, mtf_analysis, order_book_data, correlation_data, 
                market_state, nebula_signal, global_api_client
            )

    async def execute_live_momentum_buy(self, pair: str, signal: Dict[str, Any]) -> bool:
        """Execute live buy with performance tracking"""
        try:
            entry_price = signal.get('momentum_data', {}).get('current_price', 0)
            if not entry_price:
                entry_price = await self.get_current_price_live(pair)
            
            # Prepare live tracking data
            entry_data = {
                'entry_price': entry_price,
                'confidence': signal.get('confidence', 0),
                'entry_quality_score': signal.get('momentum_data', {}).get('overall_score', 0),
                'signal_strength': signal.get('momentum_data', {}).get('breakout_type', 'UNKNOWN'),
                'breakout_type': signal.get('momentum_data', {}).get('breakout_type', 'NONE'),
                'volume_confirmed': signal.get('momentum_data', {}).get('volume_confirmation', False),
                'momentum_confirmed': signal.get('momentum_data', {}).get('momentum_confirmation', False),
                'is_high_probability': signal.get('momentum_data', {}).get('is_high_probability', False),
                'risk_reward_ratio': signal.get('momentum_data', {}).get('risk_reward_ratio', 2.0),
                'market_phase': signal.get('momentum_data', {}).get('market_phase_fit', 'UNKNOWN'),
                'volume_spike_ratio': signal.get('momentum_data', {}).get('volume_spike_ratio', 1.0)
            }
            
            # Execute using existing buy method
            success = await self._execute_api_enhanced_buy(pair, signal)
            
            if success and self.momentum_performance_optimizer:
                # Track for live performance optimization
                trade_id = await self.momentum_performance_optimizer.track_trade_entry(pair, entry_data)
                
                self.momentum_trades[pair] = {
                    'trade_id': trade_id,
                    'entry_time': datetime.now(),
                    'entry_data': entry_data,
                    'signal': signal
                }
                
                logging.info(f"📊 LIVE buy executed and tracked for {pair}, trade_id: {trade_id}")
            
            return success
            
        except Exception as e:
            logging.error(f"Error in live momentum buy for {pair}: {str(e)}")
            return await self._execute_api_enhanced_buy(pair, signal)

    async def execute_live_momentum_sell(self, pair: str, sell_data: Dict, exit_reason: str = "MANUAL") -> bool:
        """Execute live sell with performance tracking"""
        try:
            exit_price = await self.get_current_price_live(pair)
            success = await self.execute_sell(pair, sell_data)
            
            if success and pair in self.momentum_trades and self.momentum_performance_optimizer:
                trade_info = self.momentum_trades[pair]
                await self.momentum_performance_optimizer.track_trade_exit(
                    trade_info['trade_id'], 
                    exit_price, 
                    exit_reason
                )
                
                entry_price = trade_info['entry_data']['entry_price']
                pnl_percent = ((exit_price - entry_price) / entry_price) * 100 if entry_price > 0 else 0
                
                logging.info(f"📊 LIVE sell executed for {pair}: {pnl_percent:.2f}% PnL")
                
                del self.momentum_trades[pair]
                await self.check_live_performance()
            
            return success
            
        except Exception as e:
            logging.error(f"Error in live momentum sell for {pair}: {str(e)}")
            return await self.execute_sell(pair, sell_data)

    async def check_live_performance(self):
        """Check live performance and alert if needed"""
        if not self.momentum_performance_optimizer:
            return
        
        try:
            metrics = await self.momentum_performance_optimizer.get_current_performance()
            
            if metrics.total_trades >= 20:
                if metrics.win_rate < 0.70:
                    logging.warning(f"🚨 LIVE ALERT: Win rate {metrics.win_rate*100:.1f}% below 70%")
                
                if metrics.max_consecutive_losses >= 4:
                    logging.warning(f"🚨 LIVE ALERT: {metrics.max_consecutive_losses} consecutive losses")
                
                if metrics.total_trades % 10 == 0:
                    logging.info(f"""
                    📊 LIVE PERFORMANCE UPDATE
                    Total Trades: {metrics.total_trades}
                    Win Rate: {metrics.win_rate*100:.1f}%
                    High Probability Win Rate: {metrics.high_prob_win_rate*100:.1f}%
                    """)
            
        except Exception as e:
            logging.error(f"Error checking live performance: {str(e)}")

    def get_total_equity(self):
        """Get total account equity"""
        logging.info(f"Getting total equity - TEST_MODE: {config.TEST_MODE}")
        
        if config.TEST_MODE:
            try:
                equity_query = "SELECT value FROM bot_stats WHERE key='total_equity' LIMIT 1"
                result = self.db_manager.execute_query_sync(equity_query, fetch_one=True)
                
                if result:
                    equity = float(result[0])
                    logging.info(f"Test mode: Retrieved equity from database: ${equity:.2f}")
                    return equity
                else:
                    self.db_manager.execute_query_sync(
                        "INSERT INTO bot_stats (key, value, last_updated) VALUES (?, ?, ?)",
                        ('total_equity', 1000.0, datetime.now().strftime('%Y-%m-%d %H:%M:%S')),
                        commit=True
                    )
                    logging.info("Test mode: Set initial equity to $1000.00")
                    return 1000.0
            except Exception as e:
                logging.warning(f"Error accessing database in test mode: {e}")
                logging.info("Test mode: Using fallback equity of $1000.00")
                return 1000.0
        
        # Live mode
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
                    try:
                        ticker = self.binance_client.get_symbol_ticker(symbol=f"{asset}USDT")
                        price = float(ticker['price']) if ticker['price'] else 0.0
                        total += total_amount * price
                    except (ValueError, TypeError, Exception) as e:
                        logging.debug(f"Could not get price for {asset}USDT: {str(e)}")
                        pass
        
        return total

    async def get_current_price(self, pair: str) -> float:
        """Get current price with comprehensive error handling"""
        try:
            cache_key = f"price_{pair}"
            current_time = time.time()
            
            if (hasattr(self, 'data_cache') and cache_key in self.data_cache and 
                hasattr(self, 'cache_expiry') and self.cache_expiry.get(cache_key, 0) > current_time):
                return self.data_cache[cache_key]
                
            if not hasattr(self, 'data_cache'):
                self.data_cache = {}
            if not hasattr(self, 'cache_expiry'):
                self.cache_expiry = {}
                
            try:
                self.track_api_call("get_symbol_ticker")
            except:
                pass

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
                if hasattr(self, 'data_cache') and cache_key in self.data_cache:
                    cached_price = self.data_cache[cache_key]
                    logging.info(f"Using cached price for {pair}: {cached_price}")
                    return cached_price
                return 0.0

            if price > 0:
                self.data_cache[cache_key] = price
                self.cache_expiry[cache_key] = current_time + 30
                return price
            else:
                logging.warning(f"Invalid price for {pair}: {price}")
                return 0.0
                
        except Exception as e:
            logging.error(f"Error getting current price for {pair}: {str(e)}")
            try:
                cache_key = f"price_{pair}"
                if hasattr(self, 'data_cache') and cache_key in self.data_cache:
                    return self.data_cache[cache_key]
            except:
                pass
            return 0.0

    def track_api_call(self, endpoint: str):
        """Track API call frequency"""
        try:
            current_time = time.time()
            current_hour = int(current_time / 3600)
            
            if not hasattr(self, 'api_call_tracker'):
                self.api_call_tracker = {}
                
            if endpoint not in self.api_call_tracker:
                self.api_call_tracker[endpoint] = {}
                
            if current_hour not in self.api_call_tracker[endpoint]:
                self.api_call_tracker[endpoint][current_hour] = 1
            else:
                self.api_call_tracker[endpoint][current_hour] += 1
                
            hourly_calls = self.api_call_tracker[endpoint][current_hour]
            
            hours_to_keep = [current_hour, current_hour - 1]
            for hour in list(self.api_call_tracker[endpoint].keys()):
                if hour not in hours_to_keep:
                    del self.api_call_tracker[endpoint][hour]
                    
            return hourly_calls
            
        except Exception as e:
            logging.error(f"Error tracking API call for {endpoint}: {str(e)}")
            return 0

    async def calculate_dynamic_position_size(self, pair, signal_strength, current_price, confidence=None):
        """Calculate dynamic position size that ensures minimum order requirements are met"""
        try:
            info = self.binance_client.get_symbol_info(pair)
            if not info:
                logging.error(f"Could not get symbol info for {pair}")
                return 0, 0
            
            min_notional = 10.0
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
            
            market_regime = self.market_state.get('regime', 'NEUTRAL')
            
            base_position_size = self.risk_manager._calculate_position_size(
                signal_strength, 
                confidence=confidence, 
                market_regime=market_regime
            )
            
            if base_position_size == 0:
                return 0, 0
            
            if signal_strength > 0.8:
                multiplier = 1.5
            elif signal_strength > 0.6:
                multiplier = 1.2
            elif signal_strength > 0.4:
                multiplier = 1.0
            else:
                multiplier = 0.8
            
            adjusted_position_size = base_position_size * multiplier
            
            min_required = min_notional * 1.2
            if adjusted_position_size < min_required:
                adjusted_position_size = min_required
                logging.info(f"Adjusted position size to ${adjusted_position_size:.2f} to meet minimum notional")
            
            quantity = adjusted_position_size / current_price
            
            if quantity < min_qty:
                quantity = min_qty
                adjusted_position_size = quantity * current_price
                logging.info(f"Adjusted quantity to meet minimum: {quantity}")
            
            precision = 0
            temp_step = step_size
            while temp_step < 1:
                temp_step *= 10
                precision += 1
            
            quantity = round(quantity - (quantity % step_size), precision)
            
            final_notional = quantity * current_price
            if final_notional < min_notional:
                quantity = (min_notional * 1.2) / current_price
                quantity = round(quantity + step_size, precision)
                adjusted_position_size = quantity * current_price
                logging.info(f"Final adjustment: quantity={quantity}, value=${adjusted_position_size:.2f}")
            
            return quantity, adjusted_position_size
            
        except Exception as e:
            logging.error(f"Error calculating dynamic position size: {str(e)}")
            return 0, 0

    async def _execute_api_enhanced_buy(self, pair: str, analysis: dict) -> bool:
        """Execute buy order with API-enhanced parameters"""
        try:
            api_data = analysis.get('api_data', {})
            current_price = await self.get_current_price(pair)
            if not current_price:
                return False
            
            signal_strength = analysis.get('signal_strength', 0.5)
            confidence = analysis.get('confidence', api_data.get('confidence', 50))
            
            quantity, position_value = await self.calculate_dynamic_position_size(
                pair, signal_strength, current_price, confidence=confidence
            )
            
            if quantity <= 0 or position_value <= 0:
                logging.info(f"Position size 0 for {pair} (qty: {quantity}, value: {position_value}), skipping trade")
                return False
            
            api_confidence = api_data.get('confidence', 50)
            blended_confidence = analysis.get('confidence', 50)
            asset_score = analysis.get('asset_score', 3.0)
            api_reason = api_data.get('reasoning', 'technical_analysis')
            
            logging.info(f"ENHANCED BUY: {pair} - Qty: {quantity:.6f}, Value: ${position_value:.2f}, "
                        f"API: {api_confidence:.1f}%, Blended: {blended_confidence:.1f}%, "
                        f"Asset Score: {asset_score:.2f}, Reason: {api_reason}")
            
            if config.TEST_MODE:
                order_id = f"test_{int(time.time())}"
                
                position_data = {
                    'entry_price': current_price,
                    'quantity': quantity,
                    'entry_time': time.time(),
                    'position_value': position_value,
                    'signal_source': 'api_enhanced',
                    'signal_strength': analysis.get('signal_strength', 0),
                    'api_data': api_data,
                    'order_id': order_id,
                    'momentum_trade': analysis.get('momentum_trade', False)
                }
                
                self._add_position_synchronized(pair, position_data)
                
                if config.TEST_MODE:
                    current_equity = await self.get_real_time_equity()
                    await self.update_equity_in_db(current_equity)
                
                await self.set_smart_trailing_stop(pair, current_price)
                
                if api_data:
                    stop_loss_price = current_price * (1 + config.QUICK_STOP_LOSS / 100)
                    await self._set_api_stop_loss(pair, current_price, stop_loss_price)
                    
                    if isinstance(api_data, dict) and 'exit_levels' in api_data:
                        await self._set_api_take_profits(pair, current_price, api_data['exit_levels'])
                    else:
                        default_exit_levels = {
                            'take_profit_1': current_price * 1.015,
                            'take_profit_2': current_price * 1.025,
                            'take_profit_3': current_price * 1.035
                        }
                        await self._set_api_take_profits(pair, current_price, default_exit_levels)
                
                logging.info(f"TEST: Buy order for {pair} simulated successfully")
                return True
                
            else:
                try:
                    order = await self.binance_client.create_order(
                        symbol=pair,
                        side='BUY',
                        type='MARKET',
                        quantity=quantity
                    )
                    
                    if order and order.get('status') == 'FILLED':
                        fill_price = float(order.get('fills', [{}])[0].get('price', current_price))
                        
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
                        
                        self._add_position_synchronized(pair, position_data)
                        
                        if config.TEST_MODE:
                            current_equity = await self.get_real_time_equity()
                            await self.update_equity_in_db(current_equity)
                        
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

    async def execute_sell(self, pair, analysis):
        """Execute a sell order with enhanced tracking"""
        try:
            if pair not in self.active_positions:
                logging.warning(f"Cannot sell {pair}: no active position")
                return False
                
            position = self.active_positions[pair]
            quantity = position['quantity']
            
            current_price = await self.get_current_price(pair)
            if current_price <= 0:
                logging.error(f"Invalid price for {pair}: {current_price}")
                return False
                
            entry_price = position['entry_price']
            profit_loss = (current_price - entry_price) * quantity
            profit_percent = ((current_price / entry_price) - 1) * 100
            
            log_message = f"SELL ORDER: {pair} - {quantity} at approx. ${current_price}, " + \
                        f"P/L: ${profit_loss:.2f} ({profit_percent:.2f}%)"
            logging.info(log_message)
            self.trade_logger.info(log_message)
            
            trade_id = f"{int(time.time())}-{random.randint(1000, 9999)}"
            success = False
            
            if config.TEST_MODE:
                order_id = f"test_{int(time.time())}"
                
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
                
                await self.performance_tracker.record_trade(trade_data)
                
                if pair in self.trailing_stops:
                    del self.trailing_stops[pair]
                
                if pair in self.trailing_tp_data:
                    del self.trailing_tp_data[pair]
                    
                if pair in self.position_scales:
                    del self.position_scales[pair]
                
                del self.active_positions[pair]
                self.risk_manager.remove_position(pair)
                
                if config.TEST_MODE:
                    new_equity = await self.get_real_time_equity()
                    await self.update_equity_in_db(new_equity)
                
                success = True
                
            else:
                try:
                    order = self.binance_client.order_market_sell(
                        symbol=pair,
                        quantity=quantity
                    )
                    
                    if order and order.get('status') == 'FILLED':
                        actual_price = float(order['fills'][0]['price'])
                        actual_quantity = float(order['executedQty'])
                        actual_value = actual_price * actual_quantity
                        actual_profit_loss = (actual_price - entry_price) * actual_quantity
                        actual_profit_percent = ((actual_price / entry_price) - 1) * 100
                        
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
                        
                        await self.performance_tracker.record_trade(trade_data)
                        
                        if pair in self.trailing_stops:
                            del self.trailing_stops[pair]
                        
                        if pair in self.trailing_tp_data:
                            del self.trailing_tp_data[pair]
                            
                        if pair in self.position_scales:
                            del self.position_scales[pair]
                        
                        del self.active_positions[pair]
                        self.risk_manager.remove_position(pair)
                        
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
            
            if success:
                if hasattr(self, 'asset_selection'):
                    self.asset_selection.mark_recently_traded(pair)
                
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
                
                win_rate = (self.session_stats['winning_trades'] / self.session_stats['total_trades'] * 100) if self.session_stats['total_trades'] > 0 else 0
                logging.info(f"Session stats: {self.session_stats['total_trades']} trades, "
                            f"{win_rate:.1f}% win rate, ${self.session_stats['total_profit']:.2f} profit")
            
            return success
        
        except Exception as e:
            logging.error(f"Error executing sell for {pair}: {str(e)}")
            return False

    async def execute_buy(self, pair: str, signal: Dict[str, Any]) -> bool:
        """Execute buy with live momentum tracking"""
        if self.enable_momentum and 'momentum_data' in signal:
            return await self.execute_live_momentum_buy(pair, signal)
        else:
            return await self._execute_api_enhanced_buy(pair, signal)

    def _add_position_synchronized(self, pair: str, position_data: dict):
        """Add position with synchronized tracking"""
        try:
            self.active_positions[pair] = position_data
            position_value = position_data.get('position_value', position_data.get('value', 0))
            self.risk_manager.add_position(pair, position_value)
            
            logging.info(f"SYNC: Added position {pair}, Active: {len(self.active_positions)}, Risk Manager: {self.risk_manager.position_count}")
            
        except Exception as e:
            logging.error(f"Error adding position {pair}: {str(e)}")
            if pair in self.active_positions:
                del self.active_positions[pair]
            self.risk_manager.remove_position(pair)

    def _validate_position_sync(self):
        """Validate that both position tracking systems are in sync"""
        active_count = len(self.active_positions)
        risk_count = self.risk_manager.position_count
        
        if active_count != risk_count:
            logging.error(f"POSITION SYNC MISMATCH: Active={active_count}, Risk Manager={risk_count}")
            logging.error(f"Active positions: {list(self.active_positions.keys())}")
            logging.error(f"Risk manager positions: {list(self.risk_manager.position_values.keys())}")
            
            self.risk_manager.position_count = active_count
            self.risk_manager.position_values = {
                pair: pos.get('position_value', pos.get('value', 0)) 
                for pair, pos in self.active_positions.items()
            }
            logging.info(f"FORCED SYNC: Both systems now at {active_count} positions")
            
        return active_count

    async def get_real_time_equity(self):
        """Get real-time equity including current position values and unrealized P&L"""
        try:
            base_equity = self.get_total_equity()
            
            if not self.active_positions:
                return base_equity
            
            total_position_value = 0
            total_unrealized_pnl = 0
            
            for pair, position in self.active_positions.items():
                try:
                    current_price = await self.get_current_price(pair)
                    if not current_price:
                        logging.warning(f"Could not get current price for {pair}, using entry price")
                        current_price = position['entry_price']
                    
                    current_value = position['quantity'] * current_price
                    initial_value = position['quantity'] * position['entry_price']
                    unrealized_pnl = current_value - initial_value
                    
                    total_position_value += current_value
                    total_unrealized_pnl += unrealized_pnl
                    
                    logging.debug(f"{pair}: Current value ${current_value:.2f}, "
                                f"Initial ${initial_value:.2f}, P&L ${unrealized_pnl:.2f}")
                    
                except Exception as e:
                    logging.error(f"Error calculating position value for {pair}: {str(e)}")
                    fallback_value = position['quantity'] * position['entry_price']
                    total_position_value += fallback_value
            
            if config.TEST_MODE:
                real_time_equity = base_equity + total_unrealized_pnl
            else:
                real_time_equity = base_equity + total_unrealized_pnl
            
            logging.info(f"Real-time equity: ${real_time_equity:.2f} "
                        f"(Base: ${base_equity:.2f}, Position Value: ${total_position_value:.2f}, "
                        f"Unrealized P&L: ${total_unrealized_pnl:+.2f})")
            
            return real_time_equity
            
        except Exception as e:
            logging.error(f"Error calculating real-time equity: {str(e)}")
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

    async def preload_historical_data(self):
        """Preload sufficient historical data for all needed timeframes"""
        logging.info("Preloading historical data for analysis...")

        pairs = await self.asset_selection.get_optimal_trading_pairs(max_pairs=15)
        major_pairs = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'ADAUSDT']
        for major in major_pairs:
            if major not in pairs:
                pairs.append(major)
        pairs = list(set(pairs))

        timeframes = ['5m', '15m', '1h', '4h', '1d']

        days_of_history = {
            '5m': 5,
            '15m': 10,
            '1h': 30,
            '4h': 60,
            '1d': 200
        }

        chunk_size = 5
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
                
                await asyncio.sleep(1)
            
            await asyncio.sleep(2)

    async def should_take_profit(self, pair, position, current_price):
        """More aggressive profit-taking logic"""
        profit_pct = ((current_price - position['entry_price']) / position['entry_price']) * 100
        hold_time_minutes = (time.time() - position['entry_time']) / 60
        
        api_data = position.get('api_data', {})
        
        if config.ENABLE_PARTIAL_PROFITS:
            for level in config.PARTIAL_PROFIT_LEVELS:
                if profit_pct >= level['profit_pct']:
                    completed_partials = position.get('completed_partials', [])
                    level_key = f"partial_{level['profit_pct']}"
                    
                    if level_key not in completed_partials:
                        partial_qty = position['quantity'] * level['sell_pct']
                        
                        logging.info(f"PROFIT TAKING: {pair} at {profit_pct:.2f}% profit - "
                                   f"Selling {level['sell_pct']*100:.0f}% at {level['profit_pct']:.2f}% target")
                        
                        try:
                            await self.execute_partial_sell(pair, partial_qty, 
                                                          reason=f"partial_tp_{level['profit_pct']}")
                            
                            if 'completed_partials' not in position:
                                position['completed_partials'] = []
                            position['completed_partials'].append(level_key)
                            
                        except Exception as e:
                            logging.error(f"Error executing partial sell for {pair}: {str(e)}")
        
        if self.market_state.get('volatility') == 'high':
            target_multiplier = 0.5
        elif self.market_state.get('trend') == 'bullish' and position.get('type') == 'buy':
            target_multiplier = 0.8
        else:
            target_multiplier = 0.7
        
        targets = config.MOMENTUM_TAKE_PROFIT if position.get('momentum_trade') else config.REGULAR_TAKE_PROFIT
        
        for target in targets:
            required_time = target['minutes'] * 0.5
            
            if hold_time_minutes >= required_time:
                adjusted_target = target['profit_pct'] * target_multiplier
                if profit_pct >= adjusted_target:
                    logging.info(f"AGGRESSIVE TIME TARGET: {pair} - "
                               f"{profit_pct:.2f}% profit after {hold_time_minutes:.0f}min "
                               f"(target: {adjusted_target:.2f}% in {required_time:.0f}min)")
                    return True
        
        if profit_pct >= 0.8:
            logging.info(f"IMMEDIATE PROFIT TAKE: {pair} at {profit_pct:.2f}% - exceeds 0.8% threshold")
            return True
        
        if hold_time_minutes >= 5 and profit_pct >= 0.5:
            logging.info(f"QUICK PROFIT TAKE: {pair} at {profit_pct:.2f}% after {hold_time_minutes:.0f}min")
            return True
        
        if api_data:
            if current_price >= api_data.get('take_profit_1', float('inf')):
                logging.info(f"API take profit 1 reached for {pair}")
                return True
        
        max_hold_minutes = config.API_MAX_HOLD_TIME_HOURS * 60 * 0.5
        if hold_time_minutes > max_hold_minutes:
            if profit_pct > 0.1:
                logging.info(f"MAX HOLD EXCEEDED: {pair} - taking {profit_pct:.2f}% profit after {hold_time_minutes:.0f}min")
                return True
        
        return False

    async def set_smart_trailing_stop(self, pair, entry_price):
        """Set intelligent trailing stop based on ATR"""
        try:
            klines = self.binance_client.get_klines(
                symbol=pair,
                interval='5m',
                limit=50
            )
            
            highs = [float(k[2]) for k in klines]
            lows = [float(k[3]) for k in klines]
            closes = [float(k[4]) for k in klines]
            
            atr = self.calculate_atr(highs, lows, closes)
            
            trailing_distance = atr * 2
            trailing_pct = (trailing_distance / entry_price) * 100
            
            trailing_pct = max(1.0, min(3.0, trailing_pct))
            
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

    async def _set_api_take_profits(self, pair: str, entry_price: float, exit_levels: dict):
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
                        'percentage': 33.33 if i < 3 else 34
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

    async def analyze_and_trade(self, pair: str, analysis: dict = None) -> bool:
        """Analyze and potentially trade a pair"""
        try:
            if pair in self.active_positions:
                return False
            
            actual_position_count = self._validate_position_sync()
            
            if actual_position_count >= 5:
                logging.debug(f"Skipping {pair} - STRICT LIMIT: {actual_position_count}/5 positions")
                return False
            
            if actual_position_count >= config.MAX_POSITIONS:
                logging.debug(f"Skipping {pair} - config limit: {actual_position_count}/{config.MAX_POSITIONS}")
                return False
            
            self.api_stats['total_analyses'] += 1
            
            signal = await self.analyze_pair(pair)
            
            if not signal:
                logging.debug(f"No signal for {pair}")
                return False
            
            signal_type = signal.get('signal', 'hold').upper()
            confidence = signal.get('confidence', 0)
            
            min_confidence = getattr(config, 'MIN_SIGNAL_CONFIDENCE', 25)
            if confidence < min_confidence:
                logging.info(f"Ignoring {pair} signal - confidence {confidence:.1f}% below {min_confidence}% threshold")
                return False
            
            logging.info(f"Signal for {pair}: {signal_type} (confidence: {confidence:.1f}%)")
            
            if signal_type == 'BUY' and confidence > 15:
                logging.info(f"EXECUTING BUY for {pair} - confidence {confidence:.1f}%")
                success = await self.execute_buy(pair, signal)
                return success
                
            elif signal_type == 'SELL' and confidence > 15:
                if pair not in self.active_positions:
                    logging.debug(f"SELL signal for {pair} ignored - no active position")
                    return False
                    
                logging.info(f"EXECUTING SELL for {pair} - confidence {confidence:.1f}%")
                success = await self.execute_sell(pair, signal)
                return success
                
            else:
                logging.debug(f"HOLDING {pair} - confidence {confidence:.1f}% too low")
                return False
                
        except Exception as e:
            logging.error(f"Error in analyze_and_trade for {pair}: {str(e)}")
            return False

    async def run(self):
        """Main run method"""
        try:
            logging.info("Testing API connections...")
            
            logging.info("Initializing Enhanced Signal API...")
            self.global_api_client = EnhancedSignalAPIClient()
            await self.global_api_client.initialize()
            self.api_strategy_initialized = True
            
            if self.api_strategy_initialized:
                logging.info("Enhanced Signal API initialized successfully")
            else:
                logging.warning("API initialization failed, using fallback mode")

            server_time = self.binance_client.get_server_time()
            logging.info(f"Binance server time: {datetime.fromtimestamp(server_time['serverTime']/1000)}")
            
            logging.info("API-only mode - skipping external API tests")
            
            logging.info("Preloading historical data for analysis...")
            await self.preload_historical_data()
            logging.info("Historical data preloading complete")
            
            tasks = [
                self.market_monitor_task(),
                self.trading_task(),
                self.position_monitor_task(),
                self.performance_track_task(),
            ]
            
            await asyncio.gather(*tasks)
            
        except Exception as e:
            logging.error(f"Critical error running bot: {str(e)}")
            traceback.print_exc()
            raise

    async def market_monitor_task(self):
        """Task to monitor market conditions"""
        try:
            while True:
                try:
                    self.market_state['breadth'] = await self.market_analysis.calculate_market_breadth()
                    
                    pairs = await self.asset_selection.get_optimal_trading_pairs(max_pairs=30)
                    await self.correlation.update_correlation_matrix(pairs)
                    
                    btc_analysis = await self.market_analysis.get_multi_timeframe_analysis('BTCUSDT')
                    eth_analysis = await self.market_analysis.get_multi_timeframe_analysis('ETHUSDT')
                    
                    avg_trend = (btc_analysis['mtf_trend'] + eth_analysis['mtf_trend']) / 2
                    
                    if avg_trend > 0.3:
                        self.market_state['trend'] = 'bullish'
                    elif avg_trend < -0.3:
                        self.market_state['trend'] = 'bearish'
                    else:
                        self.market_state['trend'] = 'neutral'
                    
                    btc_volatility = btc_analysis['mtf_volatility']
                    eth_volatility = eth_analysis['mtf_volatility']
                    avg_volatility = (btc_volatility + eth_volatility) / 2
                    
                    if avg_volatility > 0.7:
                        self.market_state['volatility'] = 'high'
                    elif avg_volatility < 0.3:
                        self.market_state['volatility'] = 'low'
                    else:
                        self.market_state['volatility'] = 'normal'
                    
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
                    
                    if regime != self.market_state['regime']:
                        logging.info(f"Market regime changed: {self.market_state['regime']} -> {regime}")
                        self.market_state['regime'] = regime
                    
                    logging.info(f"Market state: Regime={regime}, Trend={self.market_state['trend']}, " +
                               f"Volatility={self.market_state['volatility']}, Breadth={self.market_state['breadth']:.2f}")
                
                except Exception as inner_e:
                    logging.error(f"Error in market monitor cycle: {str(inner_e)}")
                
                await asyncio.sleep(1800)
                
        except Exception as e:
            logging.error(f"Error in market monitor task: {str(e)}")
            await asyncio.sleep(60)
            asyncio.create_task(self.market_monitor_task())

    async def trading_task(self):
        """Enhanced trading task with momentum system"""
        try:
            while True:
                try:
                    start_time = time.time()
                    
                    base_equity = self.get_total_equity()
                    self.risk_manager.update_equity(base_equity)
                    
                    current_equity = await self.get_real_time_equity()
                    
                    risk_status = self.risk_manager.get_status()
                    
                    if risk_status.get('severe_recovery_mode', False):
                        logging.info("Trading paused due to severe recovery mode")
                        await asyncio.sleep(60)
                        continue
                    
                    logging.info(f"Current equity: ${risk_status['equity']:.2f}, " +
                            f"Daily ROI: {risk_status['daily_roi']:.2f}%, " + 
                            f"Drawdown: {risk_status['drawdown']:.2f}%, " +
                            f"Risk level: {risk_status['risk_level']}, " +
                            f"Positions: {self.risk_manager.position_count}/{config.MAX_POSITIONS}")
                    
                    high_priority_pairs = []
                    if hasattr(self, 'latest_opportunities') and self.latest_opportunities:
                        high_priority_pairs = [opp['symbol'] for opp in self.latest_opportunities[:3]]
                    
                    pairs_to_analyze = await self.asset_selection.get_optimal_trading_pairs(max_pairs=20)
                    
                    if self.market_state.get('volatility') == 'high':
                        pairs_to_analyze = pairs_to_analyze[:10]
                        logging.info("High volatility - limiting analysis to 10 pairs")
                    
                    final_pairs = high_priority_pairs[:3] + [p for p in pairs_to_analyze if p not in high_priority_pairs][:15]
                    
                    if high_priority_pairs:
                        logging.info(f"High priority opportunities: {', '.join(high_priority_pairs[:3])}")
                    
                    logging.info(f"Selected {len(final_pairs)} pairs for trading: {final_pairs[:10]}")
                    
                    analysis_tasks = []
                    for pair in final_pairs[:20]:
                        try:
                            analysis_tasks.append(self.analyze_and_trade(pair))
                        except Exception as e:
                            logging.error(f"Error creating analysis task for {pair}: {str(e)}")
                    
                    if analysis_tasks:
                        try:
                            await asyncio.gather(*analysis_tasks, return_exceptions=True)
                        except Exception as e:
                            logging.error(f"Error in analysis tasks: {str(e)}")
                    
                    if risk_status and risk_status.get('daily_roi', 0) >= config.TARGET_DAILY_ROI_MIN * 100:
                        logging.info("Daily ROI target achieved! Securing profits...")
                        await self.secure_profits()
                    
                    logging.info(f"Current daily ROI: {risk_status.get('daily_roi', 0):.2f}% "
                            f"(target: {config.TARGET_DAILY_ROI_MIN*100}% - {config.TARGET_DAILY_ROI_MAX*100}%)")
                    
                    processing_time = time.time() - start_time
                    sleep_time = max(10, 20 - processing_time)
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
                    await self.monitor_positions()
                except Exception as inner_e:
                    logging.error(f"Error in position monitoring cycle: {str(inner_e)}")
                
                await asyncio.sleep(30)
                
        except Exception as e:
            logging.error(f"Error in position monitor task: {str(e)}")
            await asyncio.sleep(60)
            asyncio.create_task(self.position_monitor_task())

    async def performance_track_task(self):
        """Task to track and report performance"""
        try:
            while True:
                try:
                    current_equity = await self.get_real_time_equity()
                    
                    daily_roi = self.risk_manager.calculate_daily_roi()
                    drawdown = self.risk_manager.current_drawdown * 100
                    
                    await self.performance_tracker.record_equity(
                        current_equity,
                        daily_roi=daily_roi, 
                        drawdown=drawdown
                    )
                    
                    current_hour = datetime.now().hour
                    if not hasattr(self, 'last_report_hour') or self.last_report_hour != current_hour:
                        self.last_report_hour = current_hour
                        report = await self.performance_tracker.generate_performance_report()
                        logging.info(f"Hourly performance report: {json.dumps(report, indent=2)}")
                        
                except Exception as inner_e:
                    logging.error(f"Error in performance tracking cycle: {str(inner_e)}")
                
                await asyncio.sleep(300)
                
        except Exception as e:
            logging.error(f"Error in performance track task: {str(e)}")
            await asyncio.sleep(60)
            asyncio.create_task(self.performance_track_task())

    async def monitor_positions(self):
        """Monitor and manage open positions with enhanced logic"""
        if not self.active_positions:
            return
        
        try:
            positions_to_close = []
            
            for pair, position in self.active_positions.items():
                try:
                    current_price = await self.get_current_price(pair)
                    if not current_price:
                        continue
                    
                    entry_price = position['entry_price']
                    profit_loss = (current_price - entry_price) * position['quantity']
                    profit_percent = ((current_price - entry_price) / entry_price) * 100
                    position_age_minutes = (time.time() - position['entry_time']) / 60
                    
                    has_api_data = 'api_data' in position and position['api_data']
                    is_momentum = position.get('momentum_trade', False)
                    
                    position_type = "MOMENTUM" if is_momentum else "REGULAR"
                    api_status = "API" if has_api_data else "No API"
                    
                    logging.info(f"{pair} status: {profit_percent:+.1f}% after {position_age_minutes:.0f}min "
                            f"[{position_type}] ({api_status})")
                    
                    should_sell = await self.should_take_profit(pair, position, current_price)
                    
                    if should_sell:
                        positions_to_close.append((pair, {"sell_signal": True, "signal_strength": 0.9}))
                        continue
                    
                    stop_loss_pct = config.MOMENTUM_STOP_LOSS if is_momentum else config.QUICK_STOP_LOSS
                    if profit_percent <= stop_loss_pct:
                        logging.info(f"Stop loss triggered for {pair} at {profit_percent:.2f}%")
                        positions_to_close.append((pair, {"sell_signal": True, "signal_strength": 0.95}))
                        continue
                    
                    if config.ENABLE_TRAILING_STOPS and pair in self.trailing_stops:
                        trail_data = self.trailing_stops[pair]
                        
                        if current_price > trail_data['highest_price']:
                            trail_data['highest_price'] = current_price
                            trail_data['stop_price'] = current_price * (1 - trail_data['trailing_pct']/100)
                            logging.debug(f"Updated trailing stop for {pair}: ${trail_data['stop_price']:.4f}")
                        
                        if current_price <= trail_data['stop_price'] and current_price > trail_data['activation_price']:
                            logging.info(f"Trailing stop hit for {pair} at {profit_percent:.2f}%")
                            positions_to_close.append((pair, {"sell_signal": True, "signal_strength": 0.9}))
                    
                except Exception as e:
                    logging.error(f"Error monitoring position {pair}: {str(e)}")
            
            for pair, analysis in positions_to_close:
                try:
                    await self.execute_sell(pair, analysis)
                except Exception as e:
                    logging.error(f"Error closing position {pair}: {str(e)}")
            
        except Exception as e:
            logging.error(f"Error in position monitoring: {str(e)}")

    async def secure_profits(self, take_top=2):
        """Take partial profits when daily target is reached"""
        if not self.active_positions:
            return
            
        logging.info("Securing profits from open positions...")
        
        profit_sorted = []
        for pair, position in self.active_positions.items():
            profit_percent = self.calculate_profit_percent(pair)
            profit_sorted.append((pair, profit_percent))
            
        profit_sorted.sort(key=lambda x: x[1], reverse=True)
        
        profits_taken = 0
        for pair, profit in profit_sorted:
            if profit > 1.0:
                logging.info(f"Taking profits from {pair} at {profit:.2f}%")
                await self.execute_sell(pair, {"sell_signal": True, "signal_strength": 0.9})
                profits_taken += 1
                
                if profits_taken >= take_top:
                    break
                    
                await asyncio.sleep(1)
        
        logging.info(f"Took profits on {profits_taken} positions")

    def calculate_profit_percent(self, pair):
        """Calculate profit percentage with safe error handling"""
        try:
            if pair not in self.active_positions:
                return 0
                
            position = self.active_positions[pair]
            entry_price = position.get('entry_price', 0)
            
            if entry_price <= 0:
                logging.warning(f"Invalid entry price for {pair}: {entry_price}")
                return 0
            
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

    async def cleanup(self):
        """Cleanup bot resources"""
        try:
            # Close global API client first
            if hasattr(self, 'global_api_client') and self.global_api_client:
                await self.global_api_client.close()
                logging.info("Global API client closed")
            
            if hasattr(self, 'strategy') and hasattr(self.strategy, 'close'):
                await self.strategy.close()
                
            # Close momentum system resources
            if hasattr(self, 'momentum_orchestrator') and self.momentum_orchestrator:
                try:
                    if hasattr(self.momentum_orchestrator, 'cleanup'):
                        await self.momentum_orchestrator.cleanup()
                except Exception as e:
                    logging.error(f"Error cleaning up momentum orchestrator: {str(e)}")
                    
            if hasattr(self, 'momentum_performance_optimizer') and self.momentum_performance_optimizer:
                try:
                    if hasattr(self.momentum_performance_optimizer, 'close'):
                        await self.momentum_performance_optimizer.close()
                except Exception as e:
                    logging.error(f"Error cleaning up momentum performance optimizer: {str(e)}")
                    
            logging.info("Bot cleanup completed successfully")
            
        except Exception as e:
            logging.error(f"Error during bot cleanup: {str(e)}")

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

    def _get_correlation_data(self, pair: str) -> dict:
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

    def log_advanced_indicators_stats(self):
        """Log statistics about advanced indicators usage"""
        if hasattr(self, 'advanced_indicators') and self.advanced_indicators:
            try:
                stats = self.advanced_indicators.get_stats()
                logging.info(f"Advanced Indicators Stats - API calls: {stats['api_calls_made']}, "
                           f"Cache hits: {stats['cache_hits']}, "
                           f"Cache items: {stats['cache_stats']['cached_items']}, "
                           f"Client: {stats['client_type']}")
            except Exception as e:
                logging.error(f"Error getting advanced indicators stats: {str(e)}")

    def _get_neutral_analysis(self):
        """Return neutral analysis when no data available"""
        return {
            "mtf_trend": 0,
            "mtf_momentum": 0,
            "mtf_volatility": 0.5,
            "mtf_volume": 0.5,
            "overall_score": 0,
            "timeframes_analyzed": 0,
            "data_source": "neutral_fallback"
        }

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
                    'original_size': position['position_value'],
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

    async def _execute_buy_with_levels(self, pair: str, quantity: float, position_size: float, 
                                      analysis: dict, exit_levels: dict) -> bool:
        """Execute buy with API-recommended levels"""
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
                # Execute real order for live trading
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
                        
                        # Mark pair as recently traded for diversification
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

    async def _execute_momentum_buy(self, pair: str, signal: Dict[str, Any]) -> bool:
        """Enhanced buy execution with momentum performance tracking"""
        try:
            # Prepare tracking data
            entry_data = {
                'entry_price': signal.get('api_data', {}).get('entry_price', 0) or await self.get_current_price(pair),
                'confidence': signal.get('confidence', 0),
                'entry_quality_score': signal.get('momentum_data', {}).get('overall_score', 0),
                'signal_strength': signal.get('momentum_data', {}).get('breakout_type', 'UNKNOWN'),
                'breakout_type': signal.get('momentum_data', {}).get('breakout_type', 'NONE'),
                'volume_confirmed': signal.get('momentum_data', {}).get('volume_confirmation', False),
                'momentum_confirmed': signal.get('momentum_data', {}).get('momentum_confirmation', False),
                'is_high_probability': signal.get('momentum_data', {}).get('is_high_probability', False),
                'risk_reward_ratio': signal.get('momentum_data', {}).get('risk_reward_ratio', 2.0),
                'market_phase': signal.get('momentum_data', {}).get('market_phase_fit', 'UNKNOWN'),
                'rsi_at_entry': None,  # Could extract from signal if available
                'volume_spike_ratio': None  # Could extract from signal if available
            }
            
            # Execute the trade using your existing method
            success = await self._execute_api_enhanced_buy(pair, signal)
            
            if success and self.momentum_performance_optimizer:
                # Track trade entry for performance optimization
                trade_id = await self.momentum_performance_optimizer.track_trade_entry(pair, entry_data)
                
                # Store trade ID for later exit tracking
                self.momentum_trades[pair] = {
                    'trade_id': trade_id,
                    'entry_time': datetime.now(),
                    'entry_data': entry_data,
                    'signal': signal
                }
                
                logging.info(f"Buy executed and tracked for {pair}, trade_id: {trade_id}")
            
            return success
            
        except Exception as e:
            logging.error(f"Error in momentum buy execution for {pair}: {str(e)}")
            # Fallback to original buy method
            return await self._execute_api_enhanced_buy(pair, signal)

    async def _execute_momentum_sell(self, pair: str, sell_data: Dict, exit_reason: str = "MANUAL") -> bool:
        """Enhanced sell execution with momentum performance tracking"""
        try:
            # Get current price for tracking
            current_price = await self.get_current_price(pair)
            
            # Execute the sell using your existing method
            success = await self.execute_sell(pair, sell_data)
            
            if success and pair in self.momentum_trades and self.momentum_performance_optimizer:
                # Track trade exit
                trade_info = self.momentum_trades[pair]
                await self.momentum_performance_optimizer.track_trade_exit(
                    trade_info['trade_id'], 
                    current_price, 
                    exit_reason
                )
                
                # Remove from active trades
                del self.momentum_trades[pair]
                
                logging.info(f"Sell executed and tracked for {pair}")
                
                # Check if we need performance review
                await self._check_momentum_performance()
            
            return success
            
        except Exception as e:
            logging.error(f"Error in momentum sell execution for {pair}: {str(e)}")
            # Fallback to original sell method
            return await self.execute_sell(pair, sell_data)

    async def _check_momentum_performance(self):
        """Check momentum system performance and adjust if needed"""
        if not self.momentum_performance_optimizer:
            return
        
        try:
            current_metrics = await self.momentum_performance_optimizer.get_current_performance()
            
            # Log performance every 10 trades
            if current_metrics.total_trades > 0 and current_metrics.total_trades % 10 == 0:
                logging.info(f"""
                MOMENTUM PERFORMANCE UPDATE
                Total Trades: {current_metrics.total_trades}
                Win Rate: {current_metrics.win_rate*100:.1f}% (Target: 75-90%)
                High Prob Win Rate: {current_metrics.high_prob_win_rate*100:.1f}%
                Volume Confirmed Win Rate: {current_metrics.volume_confirmed_win_rate*100:.1f}%
                """)
            
            # Alert if performance is concerning
            if current_metrics.total_trades >= 20:  # Need sufficient sample size
                if current_metrics.win_rate < 0.70:
                    logging.warning(f"Win rate ({current_metrics.win_rate*100:.1f}%) below 70% - consider review")
                
                if current_metrics.max_consecutive_losses >= 4:
                    logging.warning(f"{current_metrics.max_consecutive_losses} consecutive losses - review strategy")
            
        except Exception as e:
            logging.error(f"Error checking momentum performance: {str(e)}")

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
                            logging.debug(f"{symbol} validated safe for live trading")
                        else:
                            logging.warning(f"{symbol} marked as UNSAFE for live trading")
                            
                        return is_safe
                        
                    elif response.status == 400:
                        # Symbol doesn't exist or invalid
                        data = await response.json()
                        logging.warning(f"{symbol} validation failed: {data.get('message', 'Invalid symbol')}")
                        
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

    async def _prepare_taapi_data_for_momentum(self, pair: str, momentum_signal):
        """Prepare TAAPI data from momentum signal for entry filter"""
        try:
            # Extract TAAPI data from momentum signal if available
            taapi_data = getattr(momentum_signal, 'taapi_data', {})
            
            if not taapi_data:
                # Get fresh TAAPI data if not in signal
                taapi_data = await self.get_live_taapi_data(pair)
            
            return taapi_data
            
        except Exception as e:
            logging.error(f"Error preparing TAAPI data for {pair}: {str(e)}")
            return {}

    async def _get_market_data_for_momentum(self, pair: str):
        """Get market data formatted for momentum analysis"""
        try:
            # Get live market data
            market_data = await self.get_live_market_data(pair)
            
            # Ensure required fields are present
            if 'current_price' not in market_data:
                market_data['current_price'] = await self.get_current_price_live(pair)
            
            if 'volume_analysis' not in market_data:
                market_data['volume_analysis'] = await self.calculate_volume_analysis(pair)
            
            if 'price_momentum' not in market_data:
                market_data['price_momentum'] = await self.calculate_price_momentum(pair)
            
            return market_data
            
        except Exception as e:
            logging.error(f"Error getting market data for momentum {pair}: {str(e)}")
            return {'current_price': 0, 'volume_analysis': {}, 'price_momentum': {}}