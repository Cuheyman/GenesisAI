#!/usr/bin/env python3
"""
Complete Integration System for Momentum-Based High Win Rate Trading
Integrates Enhanced TAAPI, Entry Filters, and Configuration for 75-90% Win Rate

This is the main integration script that connects all components:
- EnhancedMomentumTaapiClient (advanced TAAPI.io integration)
- HighWinRateEntryFilter (quality assessment system)
- MomentumStrategyConfig (configuration management)
- Your existing bot infrastructure

Author: Enhanced for Danish momentum strategy
Goal: 75-90% win rate through selective, high-quality bullish entries only
"""

import asyncio
import logging
import time
import json
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from dataclasses import asdict

# Import our enhanced components
from momentum import enhanced_momentum_taapi
from momentum import high_winrate_entry_filter
from momentum import momentum_strategy_config

# Your existing imports (adapt to your actual module names)
# from hybrid_bot import HybridBot
# from enhanced_strategy import EnhancedStrategy
# from risk_manager import RiskManager

class MomentumTradingOrchestrator:
    """
    Main orchestrator for momentum-based trading with high win rate focus
    Implements Danish strategy: only bullish entries with volume/breakout confirmation
    """
    
    def __init__(self, bot_instance=None, config_override: Dict = None):
        """
        Initialize the momentum trading orchestrator
        
        Args:
            bot_instance: Your existing bot instance
            config_override: Dictionary to override default config values
        """
        
        # Configuration setup
        self.config = momentum_strategy_config.momentum_config
        if config_override:
            self._apply_config_overrides(config_override)
        
        # Validate configuration
        validation = momentum_strategy_config.validate_config()
        if not validation['valid']:
            raise ValueError(f"Configuration validation failed: {validation['errors']}")
        
        # Initialize components
        self.taapi_client = enhanced_momentum_taapi.EnhancedMomentumTaapiClient(self.config.TAAPI_API_SECRET)
        self.entry_filter = high_winrate_entry_filter.HighWinRateEntryFilter(self.config)
        self.strategy_integration = enhanced_momentum_taapi.MomentumStrategyIntegration(self.taapi_client)
        
        # Bot integration
        self.bot = bot_instance
        
        # Performance tracking
        self.trade_history = []
        self.win_rate_tracker = {
            'total_trades': 0,
            'winning_trades': 0,
            'current_win_rate': 0.0,
            'target_win_rate': self.config.PERFORMANCE_TRACKING['target_win_rate']
        }
        
        # Active monitoring
        self.active_pairs = set()
        self.monitoring_active = False
        self.last_scan_time = 0
        
        # Logging setup
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        self.logger.info("Momentum Trading Orchestrator initialized")
        self.logger.info(f"Target win rate: {self.win_rate_tracker['target_win_rate']*100:.1f}%")
        
    async def start_momentum_trading(self, pairs: List[str] = None):
        """
        Start the momentum trading system
        
        Args:
            pairs: List of trading pairs to monitor (uses config default if None)
        """
        
        if not pairs:
            pairs = self.config.SYMBOL_FILTERS['preferred_pairs']
        
        self.active_pairs = set(pairs)
        self.monitoring_active = True
        
        self.logger.info(f"Starting momentum trading for {len(pairs)} pairs")
        self.logger.info(f"Pairs: {', '.join(pairs)}")
        
        # Start main trading loop
        await self._run_trading_loop()
    
    async def _run_trading_loop(self):
        """Main trading loop with momentum detection and high win rate filtering"""
        
        while self.monitoring_active:
            try:
                loop_start_time = time.time()
                
                # Scan all active pairs
                scan_results = await self._scan_pairs_for_opportunities()
                
                # Process high-quality opportunities
                await self._process_trading_opportunities(scan_results)
                
                # Performance review
                await self._review_performance()
                
                # Wait for next scan cycle
                loop_duration = time.time() - loop_start_time
                sleep_time = max(60 - loop_duration, 10)  # Scan every minute, minimum 10s
                
                self.logger.debug(f"Trading loop completed in {loop_duration:.2f}s, sleeping {sleep_time:.1f}s")
                await asyncio.sleep(sleep_time)
                
            except Exception as e:
                self.logger.error(f"Error in trading loop: {str(e)}")
                await asyncio.sleep(30)  # Wait 30s before retry
    
    async def _scan_pairs_for_opportunities(self) -> List[Dict[str, Any]]:
        """Scan all active pairs for momentum opportunities"""
        
        opportunities = []
        
        for pair in self.active_pairs:
            try:
                # Get enhanced momentum signal
                momentum_signal = await self.taapi_client.get_momentum_optimized_signal(pair)
                
                # Skip if not a buy signal (following your Danish strategy)
                if momentum_signal.action != 'BUY':
                    continue
                
                # Get current market data (you'd implement this based on your data source)
                market_data = await self._get_market_data(pair)
                
                # Convert to TAAPI data format for entry filter
                taapi_data = await self._prepare_taapi_data_for_filter(pair, momentum_signal)
                
                # Evaluate entry quality
                entry_metrics = await self.entry_filter.evaluate_entry_quality(
                    pair, taapi_data, market_data
                )
                
                # Create opportunity record
                opportunity = {
                    'pair': pair,
                    'timestamp': datetime.now(),
                    'momentum_signal': momentum_signal,
                    'entry_metrics': entry_metrics,
                    'market_data': market_data,
                    'should_trade': self._should_execute_trade(momentum_signal, entry_metrics)
                }
                
                opportunities.append(opportunity)
                
                # Rate limiting
                await asyncio.sleep(self.config.TAAPI_RATE_LIMIT_DELAY)
                
            except Exception as e:
                self.logger.warning(f"Error scanning {pair}: {str(e)}")
                continue
        
        # Sort by quality (best opportunities first)
        opportunities.sort(key=lambda x: x['entry_metrics'].overall_score, reverse=True)
        
        self.logger.info(f"Scanned {len(self.active_pairs)} pairs, found {len(opportunities)} opportunities")
        
        return opportunities
    
    async def _process_trading_opportunities(self, opportunities: List[Dict[str, Any]]):
        """Process trading opportunities according to high win rate strategy"""
        
        high_quality_ops = [op for op in opportunities if op['should_trade']]
        
        if not high_quality_ops:
            self.logger.debug("No high-quality opportunities found this cycle")
            return
        
        self.logger.info(f"Processing {len(high_quality_ops)} high-quality opportunities")
        
        for opportunity in high_quality_ops[:3]:  # Limit to top 3 opportunities per cycle
            await self._execute_momentum_trade(opportunity)
    
    async def _execute_momentum_trade(self, opportunity: Dict[str, Any]):
        """Execute a momentum trade based on opportunity analysis"""
        
        pair = opportunity['pair']
        momentum_signal = opportunity['momentum_signal']
        entry_metrics = opportunity['entry_metrics']
        
        try:
            # Log opportunity details
            self.logger.info(f"""
            EXECUTING MOMENTUM TRADE
            Pair: {pair}
            Signal Strength: {momentum_signal.momentum_strength}
            Confidence: {momentum_signal.confidence:.1f}%
            Entry Quality: {entry_metrics.entry_quality}
            Overall Score: {entry_metrics.overall_score:.1f}
            Risk-Reward: {entry_metrics.risk_reward_ratio:.1f}
            Breakout Type: {momentum_signal.breakout_type}
            Volume Confirmed: {entry_metrics.has_volume_confirmation}
            High Probability: {entry_metrics.is_high_probability}
            """)
            
            # Convert to format compatible with your existing bot
            enhanced_signal = await self.strategy_integration.get_enhanced_signal_for_pair(pair)
            
            # Add our quality metrics to the signal
            enhanced_signal.update({
                'momentum_orchestrator': {
                    'entry_quality_score': entry_metrics.overall_score,
                    'signal_strength': momentum_signal.momentum_strength,
                    'is_high_probability': entry_metrics.is_high_probability,
                    'breakout_type': momentum_signal.breakout_type,
                    'risk_reward_ratio': entry_metrics.risk_reward_ratio,
                    'entry_timing': entry_metrics.entry_timing,
                    'market_phase_fit': entry_metrics.market_phase_fit
                }
            })
            
            # Execute trade through your existing bot
            success = await self._execute_through_bot(pair, enhanced_signal, entry_metrics)
            
            # Track trade
            self._track_trade_execution(pair, enhanced_signal, entry_metrics, success)
            
        except Exception as e:
            self.logger.error(f"Error executing trade for {pair}: {str(e)}")
    
    async def _execute_through_bot(self, pair: str, signal: Dict, metrics) -> bool:
        """Execute trade through your existing bot infrastructure"""
        
        try:
            if self.bot:
                # Use your existing bot's execution method
                # Example integration with your hybrid_bot.py
                if hasattr(self.bot, '_execute_api_enhanced_buy'):
                    result = await self.bot._execute_api_enhanced_buy(pair, signal)
                    return result
                else:
                    self.logger.warning("Bot doesn't have expected execution method")
                    return False
            else:
                # Simulation mode - just log what would happen
                self.logger.info(f"SIMULATION: Would execute BUY for {pair}")
                self.logger.info(f"Signal: {json.dumps(signal, indent=2, default=str)}")
                return True
                
        except Exception as e:
            self.logger.error(f"Error in bot execution for {pair}: {str(e)}")
            return False
    
    def _should_execute_trade(self, momentum_signal, entry_metrics) -> bool:
        """Determine if trade should be executed based on high win rate criteria"""
        
        # Primary quality gate
        if not entry_metrics.is_high_probability:
            return False
        
        # Signal strength requirement
        if entry_metrics.signal_strength not in [high_winrate_entry_filter.EntrySignalStrength.STRONG, high_winrate_entry_filter.EntrySignalStrength.EXCELLENT]:
            return False
        
        # Confidence threshold
        if momentum_signal.confidence < self.config.MIN_CONFIDENCE_SCORE:
            return False
        
        # Volume confirmation requirement (critical for your strategy)
        if self.config.REQUIRE_VOLUME_CONFIRMATION and not entry_metrics.has_volume_confirmation:
            return False
        
        # Breakout confirmation requirement (critical for your strategy)
        if self.config.REQUIRE_BREAKOUT_CONFIRMATION and not entry_metrics.has_breakout_confirmation:
            return False
        
        # Risk factors check
        critical_risks = ["RSI overbought", "Weak money flow", "High volatility"]
        for risk in entry_metrics.risk_factors:
            if any(critical in risk for critical in critical_risks):
                return False
        
        # Configuration-based final check
        return momentum_strategy_config.should_take_trade(
            entry_metrics.overall_score, 
            entry_metrics.entry_quality, 
            entry_metrics.market_phase_fit.lower()
        )
    
    async def _get_market_data(self, pair: str) -> Dict[str, Any]:
        """Get current market data for a pair (implement based on your data source)"""
        
        # This is a placeholder - implement based on your actual data source
        # You might use Binance API, your existing market data service, etc.
        
        try:
            # Example structure - adapt to your actual implementation
            return {
                'current_price': 50000.0,  # Get from your price feed
                'volume_analysis': {
                    'volume_spike_ratio': 1.5,
                    'breakout_volume_ratio': 2.0
                },
                'resistance_levels': {
                    'nearest_resistance': 51000.0
                },
                'price_momentum': {
                    '1h': 0.8,
                    '4h': 1.2
                },
                'market_hours': 'regular'
            }
        except Exception as e:
            self.logger.warning(f"Error getting market data for {pair}: {str(e)}")
            return {}
    
    async def _prepare_taapi_data_for_filter(self, pair: str, momentum_signal) -> Dict[str, Any]:
        """Prepare TAAPI data in format expected by entry filter"""
        
        # This converts the momentum signal data to the format expected by the entry filter
        # In a full implementation, you'd structure this based on your actual TAAPI responses
        
        return {
            'primary': {
                'rsi': 55.0,  # These would come from actual TAAPI responses
                'macd': {'valueMACD': 100, 'valueMACDSignal': 90, 'valueMACDHist': 10},
                'ema20': 49000,
                'ema50': 48000,
                'ema200': 45000,
                'bbands': {'valueUpperBand': 51000, 'valueMiddleBand': 50000, 'valueLowerBand': 49000},
                'atr': 1000,
                'adx': 30,
                'stochrsi': {'valueFastK': 60, 'valueFastD': 55},
                'supertrend': 49500,
                'squeeze': 0.1,
                'vwap': 49800,
                'obv': 1000000,
                'mfi': 60,
                'cdlhammer': None,
                'cdlengulfing': None,
                'cdlmorningstar': None
            },
            'short_term': {
                'rsi_15m': 58.0,
                'macd_15m': {'valueMACD': 80, 'valueMACDSignal': 75, 'valueMACDHist': 5}
            },
            'long_term': {
                'rsi_4h': 52.0,
                'macd_4h': {'valueMACD': 120, 'valueMACDSignal': 110, 'valueMACDHist': 10}
            }
        }
    
    def _track_trade_execution(self, pair: str, signal: Dict, metrics, success: bool):
        """Track trade execution for performance analysis"""
        
        trade_record = {
            'pair': pair,
            'timestamp': datetime.now(),
            'signal_confidence': signal.get('confidence', 0),
            'entry_quality_score': metrics.overall_score,
            'signal_strength': metrics.signal_strength.value,
            'is_high_probability': metrics.is_high_probability,
            'execution_success': success,
            'entry_timing': metrics.entry_timing,
            'breakout_type': signal.get('momentum_data', {}).get('breakout_type', 'UNKNOWN'),
            'risk_reward_ratio': metrics.risk_reward_ratio
        }
        
        self.trade_history.append(trade_record)
        
        # Update win rate tracking (simplified - in real implementation, track actual outcomes)
        if success:
            self.win_rate_tracker['total_trades'] += 1
            # In real implementation, you'd track the actual trade outcome
            # For now, assume high-probability trades have better outcomes
            if metrics.is_high_probability:
                self.win_rate_tracker['winning_trades'] += 1
        
        # Calculate current win rate
        if self.win_rate_tracker['total_trades'] > 0:
            self.win_rate_tracker['current_win_rate'] = (
                self.win_rate_tracker['winning_trades'] / 
                self.win_rate_tracker['total_trades']
            )
        
        # Keep only last 100 trades
        if len(self.trade_history) > 100:
            self.trade_history = self.trade_history[-100:]
    
    async def _review_performance(self):
        """Review performance and adjust strategy if needed"""
        
        # Only review if we have sufficient data
        min_trades = self.config.PERFORMANCE_TRACKING['min_sample_size']
        if self.win_rate_tracker['total_trades'] < min_trades:
            return
        
        current_wr = self.win_rate_tracker['current_win_rate']
        target_wr = self.win_rate_tracker['target_win_rate']
        adjustment_threshold = self.config.PERFORMANCE_TRACKING['adjustment_threshold']
        
        if current_wr < adjustment_threshold:
            self.logger.warning(f"""
            PERFORMANCE REVIEW WARNING
            Current win rate: {current_wr*100:.1f}%
            Target win rate: {target_wr*100:.1f}%
            Adjustment threshold: {adjustment_threshold*100:.1f}%
            
            Consider increasing quality thresholds or reviewing strategy parameters.
            """)
            
            # Auto-adjust thresholds to be more selective
            self._auto_adjust_thresholds()
    
    def _auto_adjust_thresholds(self):
        """Automatically adjust thresholds to improve win rate"""
        
        # Increase minimum confluence score by 5 points
        current_min = self.config.MIN_CONFLUENCE_SCORE
        self.config.MIN_CONFLUENCE_SCORE = min(85, current_min + 5)
        
        # Increase minimum confidence score by 5 points
        current_conf = self.config.MIN_CONFIDENCE_SCORE
        self.config.MIN_CONFIDENCE_SCORE = min(90, current_conf + 5)
        
        self.logger.info(f"""
        AUTO-ADJUSTED THRESHOLDS
        Min confluence: {current_min} -> {self.config.MIN_CONFLUENCE_SCORE}
        Min confidence: {current_conf} -> {self.config.MIN_CONFIDENCE_SCORE}
        """)
    
    def _apply_config_overrides(self, overrides: Dict):
        """Apply configuration overrides"""
        for key, value in overrides.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
                self.logger.info(f"Config override: {key} = {value}")
            else:
                self.logger.warning(f"Unknown config key: {key}")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        
        return {
            'win_rate_tracking': self.win_rate_tracker.copy(),
            'entry_filter_performance': self.entry_filter.get_performance_summary(),
            'recent_trades': len(self.trade_history),
            'configuration': {
                'min_confluence_score': self.config.MIN_CONFLUENCE_SCORE,
                'min_confidence_score': self.config.MIN_CONFIDENCE_SCORE,
                'target_win_rate': self.config.PERFORMANCE_TRACKING['target_win_rate']
            },
            'quality_distribution': self._get_quality_distribution()
        }
    
    def _get_quality_distribution(self) -> Dict[str, int]:
        """Get distribution of trade qualities"""
        distribution = {}
        for trade in self.trade_history[-50:]:  # Last 50 trades
            strength = trade.get('signal_strength', 'UNKNOWN')
            distribution[strength] = distribution.get(strength, 0) + 1
        return distribution
    
    async def stop_trading(self):
        """Stop the trading system gracefully"""
        self.monitoring_active = False
        self.logger.info("Momentum trading system stopped")

# Usage Examples and Integration Guide

async def example_standalone_usage():
    """Example of using the orchestrator standalone"""
    
    # Configuration overrides for your specific needs
    config_overrides = {
        'MIN_CONFLUENCE_SCORE': 75,  # More selective
        'MIN_CONFIDENCE_SCORE': 80,  # Higher confidence requirement
        'TAAPI_RATE_LIMIT_DELAY': 1.0  # Faster if you have higher rate limits
    }
    
    # Initialize orchestrator
    orchestrator = MomentumTradingOrchestrator(
        bot_instance=None,  # None for simulation mode
        config_override=config_overrides
    )
    
    # Define pairs to monitor
    pairs_to_monitor = ['BTCUSDT', 'ETHUSDT', 'ADAUSDT', 'SOLUSDT']
    
    # Start trading
    try:
        await orchestrator.start_momentum_trading(pairs_to_monitor)
    except KeyboardInterrupt:
        await orchestrator.stop_trading()
        print("Trading stopped by user")

async def example_bot_integration():
    """Example of integrating with your existing bot"""
    
    # Assuming you have your bot instance
    # your_bot = HybridBot(...)  # Your existing bot initialization
    
    # Initialize orchestrator with your bot
    orchestrator = MomentumTradingOrchestrator(
        bot_instance=None,  # Replace with your_bot
        config_override={
            'MIN_CONFLUENCE_SCORE': 70,
            'REQUIRE_VOLUME_CONFIRMATION': True,
            'REQUIRE_BREAKOUT_CONFIRMATION': True
        }
    )
    
    # Start monitoring
    await orchestrator.start_momentum_trading()

def single_pair_analysis_example():
    """Example of analyzing a single pair"""
    
    async def analyze_pair():
        orchestrator = MomentumTradingOrchestrator()
        
        # Get momentum signal for a specific pair
        momentum_signal = await orchestrator.taapi_client.get_momentum_optimized_signal('BTCUSDT')
        
        print(f"Signal: {momentum_signal.action}")
        print(f"Confidence: {momentum_signal.confidence:.1f}%")
        print(f"Momentum: {momentum_signal.momentum_strength}")
        print(f"Breakout: {momentum_signal.breakout_type}")
        print(f"Entry Quality: {momentum_signal.entry_quality}")
        print(f"Reasons: {', '.join(momentum_signal.reasons[:3])}")
        
        return momentum_signal
    
    return asyncio.run(analyze_pair())

if __name__ == "__main__":
    """
    Main execution - choose your mode:
    1. Standalone simulation mode
    2. Integration with existing bot
    3. Single pair analysis
    """
    
    import sys
    
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
        
        if mode == 'simulate':
            print("Running in simulation mode...")
            asyncio.run(example_standalone_usage())
        elif mode == 'integrate':
            print("Running with bot integration...")
            asyncio.run(example_bot_integration())
        elif mode == 'analyze':
            pair = sys.argv[2] if len(sys.argv) > 2 else 'BTCUSDT'
            print(f"Analyzing {pair}...")
            signal = single_pair_analysis_example()
            print(f"Analysis complete for {pair}")
        else:
            print("Unknown mode. Use: simulate, integrate, or analyze [PAIR]")
    else:
        print("""
        Momentum Trading Orchestrator
        
        Usage:
        python momentum_bot_integration.py simulate    # Run simulation mode
        python momentum_bot_integration.py integrate   # Integrate with existing bot
        python momentum_bot_integration.py analyze BTCUSDT  # Analyze specific pair
        
        For integration with your existing bot:
        1. Import: from momentum_bot_integration import MomentumTradingOrchestrator
        2. Initialize with your bot instance
        3. Start monitoring with your preferred pairs
        """)