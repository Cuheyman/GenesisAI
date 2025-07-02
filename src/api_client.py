import logging
import asyncio
import time
from typing import Dict, List, Any, Optional
import config
from enhanced_strategy_api import EnhancedSignalAPIClient, MockAPIClient

class APIEnhancedStrategy:
    """
    Enhanced trading strategy that relies primarily on the Enhanced Signal API
    Falls back to internal analysis when API is unavailable
    """
    
    def __init__(self, binance_client, market_analysis, order_book):
        self.binance_client = binance_client
        self.market_analysis = market_analysis
        self.order_book = order_book
        
        # Initialize API client
        if getattr(config, 'ENABLE_ENHANCED_API', True):
            self.api_client = EnhancedSignalAPIClient(
                api_url=getattr(config, 'SIGNAL_API_URL', 'http://localhost:3000/api'),
                api_key=getattr(config, 'SIGNAL_API_KEY', '')
            )
        else:
            self.api_client = MockAPIClient()
            
        self.api_initialized = False
        
        # Strategy settings
        self.min_api_confidence = getattr(config, 'API_MIN_CONFIDENCE', 50)
        self.override_confidence = getattr(config, 'API_OVERRIDE_CONFIDENCE', 80)
        self.use_api_position_sizing = getattr(config, 'USE_API_POSITION_SIZING', True)
        self.use_api_stop_loss = getattr(config, 'USE_API_STOP_LOSS', True)
        self.use_api_take_profit = getattr(config, 'USE_API_TAKE_PROFIT', True)
        
        # Signal weighting
        self.api_weight = getattr(config, 'API_SIGNAL_WEIGHT', 0.8)
        self.internal_weight = getattr(config, 'INTERNAL_SIGNAL_WEIGHT', 0.2)
        
        # Fallback settings
        self.fallback_enabled = getattr(config, 'API_FALLBACK_ENABLED', True)
        
        # Statistics
        self.api_signals_used = 0
        self.fallback_signals_used = 0
        self.total_analyses = 0
        
        # Cache for recent API responses
        self.recent_signals = {}
        self.signal_cache_duration = 120  # 2 minutes
        
        logging.info(f"API Enhanced Strategy initialized")
        logging.info(f"API enabled: {getattr(config, 'ENABLE_ENHANCED_API', True)}")
        logging.info(f"API URL: {getattr(config, 'SIGNAL_API_URL', 'not configured')}")
        logging.info(f"Fallback enabled: {self.fallback_enabled}")
    
    async def initialize(self):
        """Initialize the API client"""
        try:
            self.api_initialized = await self.api_client.initialize()
            if self.api_initialized:
                logging.info("Enhanced Signal API initialized successfully")
            else:
                logging.warning("API initialization failed, using fallback mode")
            return self.api_initialized
        except Exception as e:
            logging.error(f"Error initializing API client: {str(e)}")
            self.api_initialized = False
            return False
    
    async def analyze_pair(self, pair: str, mtf_analysis=None, order_book_data=None, 
                          correlation_data=None, market_state=None, nebula_signal=None) -> Dict[str, Any]:
        """
        Analyze a trading pair using Enhanced Signal API with fallback to internal analysis
        
        Args:
            pair: Trading pair (e.g., 'BTCUSDT')
            mtf_analysis: Multi-timeframe analysis (for fallback)
            order_book_data: Order book data (for fallback)
            correlation_data: Correlation data
            market_state: Current market state
            nebula_signal: Legacy nebula signal (ignored in API mode)
            
        Returns:
            Dictionary containing trading signal and analysis
        """
        
        self.total_analyses += 1
        
        try:
            # Initialize API if needed
            if not self.api_initialized:
                await self.initialize()
            
            # Try to get signal from API first
            api_signal = await self._get_api_signal(pair, market_state)
            
            if api_signal and self._is_valid_api_signal(api_signal):
                # Use API signal
                converted_signal = self.api_client.convert_to_internal_format(api_signal)
                
                # Enhance with additional data
                enhanced_signal = await self._enhance_api_signal(
                    converted_signal, pair, correlation_data, market_state
                )
                
                self.api_signals_used += 1
                
                # Cache the signal
                self._cache_signal(pair, enhanced_signal)
                
                # Log API signal usage
                confidence = api_signal.get('confidence', 0)
                signal_type = api_signal.get('signal', 'UNKNOWN')
                onchain_score = api_signal.get('onchain_score', 0)
                
                logging.info(f"API signal for {pair}: {signal_type} "
                           f"(confidence: {confidence}%, onchain: {onchain_score}/100)")
                
                return enhanced_signal
                
            elif self.fallback_enabled:
                # Fall back to internal analysis
                logging.info(f"Using fallback analysis for {pair} (API unavailable/invalid)")
                fallback_signal = await self._get_fallback_signal(
                    pair, mtf_analysis, order_book_data, correlation_data, market_state
                )
                
                self.fallback_signals_used += 1
                return fallback_signal
                
            else:
                # No fallback - return neutral signal
                logging.warning(f"No valid signal available for {pair}, returning neutral")
                return self._get_neutral_signal(pair)
                
        except Exception as e:
            logging.error(f"Error analyzing {pair}: {str(e)}")
            
            if self.fallback_enabled:
                try:
                    fallback_signal = await self._get_fallback_signal(
                        pair, mtf_analysis, order_book_data, correlation_data, market_state
                    )
                    self.fallback_signals_used += 1
                    return fallback_signal
                except Exception as fallback_error:
                    logging.error(f"Fallback analysis also failed for {pair}: {str(fallback_error)}")
            
            return self._get_neutral_signal(pair)
    
    async def _get_api_signal(self, pair: str, market_state: Dict = None) -> Optional[Dict[str, Any]]:
        """Get trading signal from the Enhanced Signal API"""
        try:
            # Determine analysis parameters based on market state
            timeframe = self._determine_optimal_timeframe(market_state)
            analysis_depth = self._determine_analysis_depth(pair)
            risk_level = self._determine_risk_level(market_state)
            
            # Get signal from API
            api_signal = await self.api_client.get_trading_signal(
                symbol=pair,
                timeframe=timeframe,
                analysis_depth=analysis_depth,
                risk_level=risk_level
            )
            
            return api_signal
            
        except Exception as e:
            logging.error(f"Error getting API signal for {pair}: {str(e)}")
            return None
    
    def _is_valid_api_signal(self, api_signal: Dict[str, Any]) -> bool:
        """Validate API signal quality and confidence"""
        if not api_signal:
            return False
        
        # Check required fields
        required_fields = ['signal', 'confidence']
        for field in required_fields:
            if field not in api_signal:
                logging.warning(f"API signal missing required field: {field}")
                return False
        
        # Check confidence threshold
        confidence = api_signal.get('confidence', 0)
        if confidence < self.min_api_confidence:
            logging.debug(f"API signal confidence too low: {confidence}% < {self.min_api_confidence}%")
            return False
        
        # Check signal validity
        signal = api_signal.get('signal', '').upper()
        if signal not in ['BUY', 'SELL', 'HOLD']:
            logging.warning(f"Invalid API signal: {signal}")
            return False
        
        return True
    
    async def _enhance_api_signal(self, converted_signal: Dict[str, Any], pair: str,
                                 correlation_data: Dict = None, market_state: Dict = None) -> Dict[str, Any]:
        """Enhance API signal with additional context and adjustments"""
        
        # Apply correlation adjustments
        if correlation_data and not correlation_data.get('is_diversified', True):
            # Reduce signal strength for highly correlated assets
            if converted_signal.get('signal_strength', 0) > 0:
                converted_signal['signal_strength'] *= 0.85
                converted_signal['source'] += ",correlation_adjusted"
        
        # Apply market regime adjustments
        if market_state:
            regime = market_state.get('regime', 'NEUTRAL')
            
            # Boost signals in trending markets
            if regime in ['BULL_TRENDING', 'BEAR_TRENDING']:
                if converted_signal.get('buy_signal') and regime == 'BULL_TRENDING':
                    converted_signal['signal_strength'] = min(0.95, converted_signal['signal_strength'] * 1.1)
                elif converted_signal.get('sell_signal') and regime == 'BEAR_TRENDING':
                    converted_signal['signal_strength'] = min(0.95, converted_signal['signal_strength'] * 1.1)
            
            # Reduce signals in volatile markets
            elif regime in ['BULL_VOLATILE', 'BEAR_VOLATILE']:
                converted_signal['signal_strength'] *= 0.9
                
        # Add API-specific metadata
        converted_signal.update({
            'analysis_method': 'enhanced_api',
            'api_available': self.api_client.api_available,
            'enhanced_with_ai': True,
            'onchain_analysis': True,
            'whale_tracking': True,
            'defi_integration': True,
            'multi_chain_analysis': True
        })
        
        return converted_signal
    
    async def _get_fallback_signal(self, pair: str, mtf_analysis=None, order_book_data=None,
                                  correlation_data=None, market_state=None) -> Dict[str, Any]:
        """Generate fallback signal using internal analysis when API is unavailable"""
        
        # Use basic internal analysis as fallback
        from enhanced_strategy import EnhancedStrategy
        
        # Create fallback strategy instance
        fallback_strategy = EnhancedStrategy(
            self.binance_client, 
            None,  # No AI client for fallback
            self.market_analysis, 
            self.order_book
        )
        
        # Get internal analysis
        internal_signal = await fallback_strategy.analyze_pair(
            pair, mtf_analysis, order_book_data, correlation_data, market_state
        )
        
        # Mark as fallback
        internal_signal['source'] = 'fallback_internal'
        internal_signal['api_available'] = False
        internal_signal['fallback_mode'] = True
        
        return internal_signal
    
    def _get_neutral_signal(self, pair: str) -> Dict[str, Any]:
        """Return neutral signal when no analysis is possible"""
        return {
            "buy_signal": False,
            "sell_signal": False,
            "signal_strength": 0,
            "source": "neutral_fallback",
            "pair": pair,
            "api_available": False,
            "error": "No valid signal available",
            "timestamp": time.time()
        }
    
    def _determine_optimal_timeframe(self, market_state: Dict = None) -> str:
        """Determine optimal timeframe based on market conditions"""
        if not market_state:
            return '1h'  # Default
        
        volatility = market_state.get('volatility', 'normal')
        regime = market_state.get('regime', 'NEUTRAL')
        
        # Use shorter timeframes in volatile markets
        if volatility == 'high' or 'VOLATILE' in regime:
            return '15m'
        elif volatility == 'low':
            return '4h'
        else:
            return '1h'
    
    def _determine_analysis_depth(self, pair: str) -> str:
        """Determine analysis depth based on pair importance"""
        major_pairs = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'ADAUSDT']
        
        if pair in major_pairs:
            return 'comprehensive'
        else:
            return 'advanced'
    
    def _determine_risk_level(self, market_state: Dict = None) -> str:
        """Determine risk level based on market conditions"""
        if not market_state:
            return 'moderate'
        
        volatility = market_state.get('volatility', 'normal')
        
        if volatility == 'high':
            return 'conservative'
        elif volatility == 'low':
            return 'aggressive'
        else:
            return 'moderate'
    
    def _cache_signal(self, pair: str, signal: Dict[str, Any]):
        """Cache recent signal for pair"""
        self.recent_signals[pair] = {
            'signal': signal,
            'timestamp': time.time()
        }
        
        # Clean up old cache entries
        current_time = time.time()
        expired_pairs = [
            p for p, data in self.recent_signals.items()
            if current_time - data['timestamp'] > self.signal_cache_duration
        ]
        for p in expired_pairs:
            del self.recent_signals[p]
    
    async def get_position_sizing_recommendation(self, pair: str, signal_data: Dict[str, Any]) -> Dict[str, Any]:
        """Get position sizing recommendation from API signal or fallback calculation"""
        
        if not self.use_api_position_sizing:
            return self._calculate_default_position_size(signal_data)
        
        api_data = signal_data.get('api_data', {})
        
        if api_data:
            # Use API recommendations
            position_size_pct = api_data.get('position_size_percent', 5)
            risk_reward_ratio = api_data.get('risk_reward_ratio', 2.0)
            max_drawdown_pct = api_data.get('max_drawdown_percent', 5)
            
            return {
                'position_size_percent': position_size_pct,
                'risk_reward_ratio': risk_reward_ratio,
                'max_drawdown_percent': max_drawdown_pct,
                'source': 'api_recommendation'
            }
        else:
            # Use internal calculation
            return self._calculate_default_position_size(signal_data)
    
    def _calculate_default_position_size(self, signal_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate default position size when API data unavailable"""
        signal_strength = signal_data.get('signal_strength', 0.5)
        
        # Base position size on signal strength
        if signal_strength > 0.8:
            position_size_pct = 8
        elif signal_strength > 0.6:
            position_size_pct = 6
        elif signal_strength > 0.4:
            position_size_pct = 4
        else:
            position_size_pct = 3
        
        return {
            'position_size_percent': position_size_pct,
            'risk_reward_ratio': 2.0,
            'max_drawdown_percent': 3.0,
            'source': 'internal_calculation'
        }
    
    async def get_exit_levels(self, pair: str, entry_price: float, signal_data: Dict[str, Any]) -> Dict[str, Any]:
        """Get stop loss and take profit levels from API or calculate defaults"""
        
        api_data = signal_data.get('api_data', {})
        
        if api_data and self.use_api_stop_loss and self.use_api_take_profit:
            # Use API recommendations
            stop_loss = api_data.get('stop_loss', 0)
            take_profits = api_data.get('take_profits', [])
            
            # Filter valid take profit levels
            valid_tps = [tp for tp in take_profits if tp > 0]
            
            return {
                'stop_loss': stop_loss,
                'take_profit_1': valid_tps[0] if len(valid_tps) > 0 else entry_price * 1.02,
                'take_profit_2': valid_tps[1] if len(valid_tps) > 1 else entry_price * 1.04,
                'take_profit_3': valid_tps[2] if len(valid_tps) > 2 else entry_price * 1.06,
                'source': 'api_levels'
            }
        else:
            # Calculate default levels
            return self._calculate_default_exit_levels(entry_price, signal_data)
    
    def _calculate_default_exit_levels(self, entry_price: float, signal_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate default exit levels when API data unavailable"""
        is_buy = signal_data.get('buy_signal', False)
        signal_strength = signal_data.get('signal_strength', 0.5)
        
        # Calculate stop loss (1.5% for buy, -1.5% for sell)
        stop_loss_pct = 0.015
        if is_buy:
            stop_loss = entry_price * (1 - stop_loss_pct)
            tp1 = entry_price * (1 + 0.02)  # 2%
            tp2 = entry_price * (1 + 0.04)  # 4%
            tp3 = entry_price * (1 + 0.06)  # 6%
        else:
            stop_loss = entry_price * (1 + stop_loss_pct)
            tp1 = entry_price * (1 - 0.02)  # -2%
            tp2 = entry_price * (1 - 0.04)  # -4%
            tp3 = entry_price * (1 - 0.06)  # -6%
        
        # Adjust based on signal strength
        if signal_strength > 0.8:
            # More aggressive targets for strong signals
            multiplier = 1.2
            tp1 *= multiplier if is_buy else (2 - multiplier)
            tp2 *= multiplier if is_buy else (2 - multiplier)
            tp3 *= multiplier if is_buy else (2 - multiplier)
        
        return {
            'stop_loss': stop_loss,
            'take_profit_1': tp1,
            'take_profit_2': tp2,
            'take_profit_3': tp3,
            'source': 'default_calculation'
        }
    
    async def cleanup(self):
        """Cleanup resources"""
        if self.api_client:
            await self.api_client.close()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get strategy statistics"""
        api_stats = self.api_client.get_statistics() if self.api_client else {}
        
        api_usage_pct = (self.api_signals_used / self.total_analyses * 100) if self.total_analyses > 0 else 0
        fallback_usage_pct = (self.fallback_signals_used / self.total_analyses * 100) if self.total_analyses > 0 else 0
        
        return {
            'strategy_type': 'api_enhanced',
            'total_analyses': self.total_analyses,
            'api_signals_used': self.api_signals_used,
            'fallback_signals_used': self.fallback_signals_used,
            'api_usage_percent': round(api_usage_pct, 2),
            'fallback_usage_percent': round(fallback_usage_pct, 2),
            'api_initialized': self.api_initialized,
            'api_client_stats': api_stats,
            'cached_signals': len(self.recent_signals),
            'settings': {
                'min_api_confidence': self.min_api_confidence,
                'use_api_position_sizing': self.use_api_position_sizing,
                'api_weight': self.api_weight,
                'fallback_enabled': self.fallback_enabled
            }
        }