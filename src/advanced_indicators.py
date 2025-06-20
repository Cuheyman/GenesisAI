import logging
import asyncio
import numpy as np
from typing import Dict, Any, Optional, List
from taapi_client import TaapiClient, DummyTaapiClient
import config

class AdvancedIndicators:
    """
    Advanced indicators using Taapi.io API for sophisticated technical analysis
    Integrates with existing enhanced_strategy.py to provide additional signal sources
    """
    
    def __init__(self):
        # Initialize Taapi.io client based on configuration
        if hasattr(config, 'ENABLE_TAAPI') and config.ENABLE_TAAPI:
            try:
                self.taapi_client = TaapiClient(config.TAAPI_API_SECRET)
                logging.info("Advanced indicators initialized with Taapi.io")
            except Exception as e:
                logging.warning(f"Failed to initialize Taapi.io client: {str(e)} - using dummy client")
                self.taapi_client = DummyTaapiClient()
        else:
            self.taapi_client = DummyTaapiClient()
            logging.info("Advanced indicators disabled - using dummy client")
        
        # Cache for indicator results
        self.cache = {}
        
        # Track API usage for debugging
        self.api_calls_made = 0
        self.cache_hits = 0
    
    async def get_advanced_signals(self, pair: str) -> Dict[str, Any]:
        """
        Get comprehensive advanced signals for a trading pair
        Returns signals that can be integrated with enhanced_strategy.py
        """
        try:
            # Extract base symbol (remove USDT)
            symbol = pair.replace("USDT", "")
            
            # Initialize results
            advanced_signals = {
                'ichimoku': None,
                'supertrend': None, 
                'tdsequential': None,
                'fisher_transform': None,
                'choppiness_index': None,
                'candlestick_patterns': None,
                'overall_signal': {
                    'buy_signal': False,
                    'sell_signal': False,
                    'signal_strength': 0,
                    'confidence': 0,
                    'source': 'advanced_indicators'
                }
            }
            
            # Get individual indicators (with rate limiting built into client)
            tasks = []
            
            # Ichimoku Cloud (4h timeframe for trend analysis)
            tasks.append(self._get_ichimoku_signal(symbol, config.TAAPI_ICHIMOKU_TIMEFRAME))
            
            # Supertrend (1h timeframe for trend following)
            tasks.append(self._get_supertrend_signal(symbol, config.TAAPI_SUPERTREND_TIMEFRAME))
            
            # TD Sequential (1d timeframe for reversal signals)
            tasks.append(self._get_tdsequential_signal(symbol, config.TAAPI_TDSEQUENTIAL_TIMEFRAME))
            
            # Fisher Transform (1h timeframe for price reversals)
            tasks.append(self._get_fisher_signal(symbol, config.TAAPI_FISHER_TIMEFRAME))
            
            # Choppiness Index (4h timeframe for market condition)
            tasks.append(self._get_choppiness_signal(symbol, config.TAAPI_CHOPPINESS_TIMEFRAME))
            
            # Candlestick patterns (1h timeframe)
            tasks.append(self._get_candlestick_signals(symbol, config.TAAPI_CANDLESTICK_TIMEFRAME))
            
            # Execute all tasks concurrently but respect rate limiting in client
            results = await self._execute_with_rate_limiting(tasks)
            
            # Parse results
            advanced_signals['ichimoku'] = results[0] if len(results) > 0 else None
            advanced_signals['supertrend'] = results[1] if len(results) > 1 else None
            advanced_signals['tdsequential'] = results[2] if len(results) > 2 else None
            advanced_signals['fisher_transform'] = results[3] if len(results) > 3 else None
            advanced_signals['choppiness_index'] = results[4] if len(results) > 4 else None
            advanced_signals['candlestick_patterns'] = results[5] if len(results) > 5 else None
            
            # Combine signals into overall assessment
            overall_signal = self._combine_advanced_signals(advanced_signals)
            advanced_signals['overall_signal'] = overall_signal
            
            return advanced_signals
            
        except Exception as e:
            logging.error(f"Error getting advanced signals for {pair}: {str(e)}")
            return self._get_neutral_advanced_signals()
    
    async def _execute_with_rate_limiting(self, tasks: List) -> List[Any]:
        """Execute tasks with built-in rate limiting from TaapiClient"""
        results = []
        for task in tasks:
            try:
                result = await task
                results.append(result)
            except Exception as e:
                logging.error(f"Advanced indicator task failed: {str(e)}")
                results.append(None)
        return results
    
    async def _get_ichimoku_signal(self, symbol: str, interval: str) -> Optional[Dict]:
        """Get Ichimoku Cloud signal"""
        try:
            ichimoku_data = await self.taapi_client.get_ichimoku(symbol, interval)
            if not ichimoku_data:
                return None
            
            self.api_calls_made += 1
            
            # Extract Ichimoku values
            tenkan_sen = ichimoku_data.get('tenkanSen', 0)
            kijun_sen = ichimoku_data.get('kijunSen', 0)
            senkou_span_a = ichimoku_data.get('senkouSpanA', 0)
            senkou_span_b = ichimoku_data.get('senkouSpanB', 0)
            chikou_span = ichimoku_data.get('chikouSpan', 0)
            
            # Current price (approximate from tenkan_sen)
            current_price = tenkan_sen  # This is an approximation
            
            # Ichimoku signal logic
            buy_signal = False
            sell_signal = False
            signal_strength = 0
            
            # Signal 1: Tenkan-sen vs Kijun-sen crossover
            if tenkan_sen > kijun_sen:
                signal_strength += 0.3
                buy_signal = True
            elif tenkan_sen < kijun_sen:
                signal_strength += 0.3
                sell_signal = True
            
            # Signal 2: Price vs Cloud
            cloud_top = max(senkou_span_a, senkou_span_b)
            cloud_bottom = min(senkou_span_a, senkou_span_b)
            
            if current_price > cloud_top:
                signal_strength += 0.4
                buy_signal = True
            elif current_price < cloud_bottom:
                signal_strength += 0.4
                sell_signal = True
            
            # Signal 3: Chikou span confirmation
            if chikou_span > current_price and buy_signal:
                signal_strength += 0.3
            elif chikou_span < current_price and sell_signal:
                signal_strength += 0.3
            
            return {
                'buy_signal': buy_signal and not sell_signal,
                'sell_signal': sell_signal and not buy_signal,
                'signal_strength': min(signal_strength, 1.0),
                'indicator_data': ichimoku_data,
                'source': 'ichimoku'
            }
            
        except Exception as e:
            logging.error(f"Error getting Ichimoku signal for {symbol}: {str(e)}")
            return None
    
    async def _get_supertrend_signal(self, symbol: str, interval: str) -> Optional[Dict]:
        """Get Supertrend signal"""
        try:
            supertrend_data = await self.taapi_client.get_supertrend(symbol, interval)
            if not supertrend_data:
                return None
            
            self.api_calls_made += 1
            
            # Extract Supertrend values
            supertrend_value = supertrend_data.get('valueAbovePrice', 0)
            is_uptrend = supertrend_data.get('valueAbovePrice', 0) == 0  # If valueAbovePrice is 0, it's uptrend
            
            # Supertrend signal logic
            buy_signal = is_uptrend
            sell_signal = not is_uptrend
            signal_strength = 0.8  # Supertrend gives strong signals
            
            return {
                'buy_signal': buy_signal,
                'sell_signal': sell_signal,
                'signal_strength': signal_strength,
                'indicator_data': supertrend_data,
                'source': 'supertrend'
            }
            
        except Exception as e:
            logging.error(f"Error getting Supertrend signal for {symbol}: {str(e)}")
            return None
    
    async def _get_tdsequential_signal(self, symbol: str, interval: str) -> Optional[Dict]:
        """Get TD Sequential signal"""
        try:
            td_data = await self.taapi_client.get_tdsequential(symbol, interval)
            if not td_data:
                return None
            
            self.api_calls_made += 1
            
            # Extract TD Sequential values
            buy_countdown = td_data.get('buyCountdown', 0)
            sell_countdown = td_data.get('sellCountdown', 0)
            buy_setup = td_data.get('buySetup', 0)
            sell_setup = td_data.get('sellSetup', 0)
            
            # TD Sequential signal logic
            buy_signal = False
            sell_signal = False
            signal_strength = 0
            
            # Buy signals: high countdown numbers indicate potential reversal
            if buy_countdown >= 12 or buy_setup >= 8:
                buy_signal = True
                signal_strength = min(0.9, (buy_countdown + buy_setup) / 20)
            
            # Sell signals: high countdown numbers indicate potential reversal  
            if sell_countdown >= 12 or sell_setup >= 8:
                sell_signal = True
                signal_strength = min(0.9, (sell_countdown + sell_setup) / 20)
            
            return {
                'buy_signal': buy_signal,
                'sell_signal': sell_signal,
                'signal_strength': signal_strength,
                'indicator_data': td_data,
                'source': 'tdsequential'
            }
            
        except Exception as e:
            logging.error(f"Error getting TD Sequential signal for {symbol}: {str(e)}")
            return None
    
    async def _get_fisher_signal(self, symbol: str, interval: str) -> Optional[Dict]:
        """Get Fisher Transform signal"""
        try:
            fisher_data = await self.taapi_client.get_fisher_transform(symbol, interval)
            if not fisher_data:
                return None
            
            self.api_calls_made += 1
            
            # Extract Fisher Transform values
            fisher_value = fisher_data.get('value', 0)
            
            # Fisher Transform signal logic
            buy_signal = fisher_value < -1.5  # Oversold, potential reversal
            sell_signal = fisher_value > 1.5   # Overbought, potential reversal
            
            # Signal strength based on how extreme the Fisher value is
            signal_strength = min(abs(fisher_value) / 3.0, 1.0)
            
            return {
                'buy_signal': buy_signal,
                'sell_signal': sell_signal,
                'signal_strength': signal_strength,
                'indicator_data': fisher_data,
                'source': 'fisher_transform'
            }
            
        except Exception as e:
            logging.error(f"Error getting Fisher Transform signal for {symbol}: {str(e)}")
            return None
    
    async def _get_choppiness_signal(self, symbol: str, interval: str) -> Optional[Dict]:
        """Get Choppiness Index signal"""
        try:
            choppiness_data = await self.taapi_client.get_choppiness_index(symbol, interval)
            if not choppiness_data:
                return None
            
            self.api_calls_made += 1
            
            # Extract Choppiness Index value
            choppiness_value = choppiness_data.get('value', 50)
            
            # Choppiness Index interpretation
            # > 61.8: Market is choppy, avoid trend-following strategies
            # < 38.2: Market is trending, good for trend-following
            
            is_trending = choppiness_value < 38.2
            is_choppy = choppiness_value > 61.8
            
            # This indicator doesn't give direct buy/sell signals
            # Instead, it indicates market condition for strategy adjustment
            return {
                'is_trending': is_trending,
                'is_choppy': is_choppy,
                'choppiness_value': choppiness_value,
                'signal_modifier': 1.2 if is_trending else 0.8 if is_choppy else 1.0,
                'indicator_data': choppiness_data,
                'source': 'choppiness_index'
            }
            
        except Exception as e:
            logging.error(f"Error getting Choppiness Index signal for {symbol}: {str(e)}")
            return None
    
    async def _get_candlestick_signals(self, symbol: str, interval: str) -> Optional[Dict]:
        """Get candlestick pattern signals"""
        try:
            # Get multiple candlestick patterns
            patterns = ['doji', 'hammer', 'engulfing']
            if interval == '1d':  # Only check star patterns on daily timeframe
                patterns.extend(['morningstar', 'eveningstar'])
            
            pattern_results = await self.taapi_client.get_multiple_indicators(symbol, interval, patterns)
            
            if not pattern_results:
                return None
            
            # Count API calls
            self.api_calls_made += len([r for r in pattern_results.values() if r is not None])
            
            # Analyze patterns
            bullish_patterns = 0
            bearish_patterns = 0
            
            for pattern_name, pattern_data in pattern_results.items():
                if pattern_data and pattern_data.get('value') == 100:  # Pattern detected
                    if pattern_name in ['hammer', 'morningstar']:
                        bullish_patterns += 1
                    elif pattern_name in ['eveningstar']:
                        bearish_patterns += 1
                    elif pattern_name == 'engulfing':
                        # Engulfing can be bullish or bearish - need more data to determine
                        # For now, treat as neutral
                        pass
                    elif pattern_name == 'doji':
                        # Doji indicates indecision - no clear signal
                        pass
            
            # Generate signals based on pattern count
            buy_signal = bullish_patterns > bearish_patterns and bullish_patterns > 0
            sell_signal = bearish_patterns > bullish_patterns and bearish_patterns > 0
            signal_strength = min((abs(bullish_patterns - bearish_patterns) / 3.0), 0.8)
            
            return {
                'buy_signal': buy_signal,
                'sell_signal': sell_signal,
                'signal_strength': signal_strength,
                'bullish_patterns': bullish_patterns,
                'bearish_patterns': bearish_patterns,
                'pattern_results': pattern_results,
                'source': 'candlestick_patterns'
            }
            
        except Exception as e:
            logging.error(f"Error getting candlestick patterns for {symbol}: {str(e)}")
            return None
    
    def _combine_advanced_signals(self, signals: Dict[str, Any]) -> Dict[str, Any]:
        """Combine all advanced signals into overall assessment"""
        try:
            buy_score = 0
            sell_score = 0
            total_weight = 0
            confidence_factors = []
            
            # Process each signal type with its configured weight
            signal_weights = {
                'ichimoku': config.TAAPI_ICHIMOKU_WEIGHT,
                'supertrend': config.TAAPI_SUPERTREND_WEIGHT,
                'tdsequential': config.TAAPI_TDSEQUENTIAL_WEIGHT,
                'fisher_transform': config.TAAPI_FISHER_WEIGHT,
                'candlestick_patterns': config.TAAPI_CANDLESTICK_WEIGHT
            }
            
            for signal_name, weight in signal_weights.items():
                signal_data = signals.get(signal_name)
                if signal_data and isinstance(signal_data, dict):
                    strength = signal_data.get('signal_strength', 0)
                    
                    if signal_data.get('buy_signal', False):
                        buy_score += strength * weight
                        confidence_factors.append(strength)
                    elif signal_data.get('sell_signal', False):
                        sell_score += strength * weight
                        confidence_factors.append(strength)
                    
                    total_weight += weight
            
            # Apply choppiness index modifier
            choppiness_data = signals.get('choppiness_index')
            signal_modifier = 1.0
            if choppiness_data and isinstance(choppiness_data, dict):
                signal_modifier = choppiness_data.get('signal_modifier', 1.0)
            
            # Apply modifier to scores
            buy_score *= signal_modifier
            sell_score *= signal_modifier
            
            # Determine final signal
            final_buy = buy_score > sell_score and buy_score > 0.3
            final_sell = sell_score > buy_score and sell_score > 0.3
            
            # Calculate final strength and confidence
            final_strength = max(buy_score, sell_score) if total_weight > 0 else 0
            confidence = np.mean(confidence_factors) if confidence_factors else 0
            
            return {
                'buy_signal': final_buy,
                'sell_signal': final_sell,
                'signal_strength': min(final_strength, 1.0),
                'confidence': confidence,
                'buy_score': buy_score,
                'sell_score': sell_score,
                'signal_modifier': signal_modifier,
                'source': 'advanced_indicators_combined'
            }
            
        except Exception as e:
            logging.error(f"Error combining advanced signals: {str(e)}")
            return {
                'buy_signal': False,
                'sell_signal': False,
                'signal_strength': 0,
                'confidence': 0,
                'source': 'advanced_indicators_error'
            }
    
    def _get_neutral_advanced_signals(self) -> Dict[str, Any]:
        """Return neutral signals when errors occur"""
        return {
            'ichimoku': None,
            'supertrend': None,
            'tdsequential': None,
            'fisher_transform': None,
            'choppiness_index': None,
            'candlestick_patterns': None,
            'overall_signal': {
                'buy_signal': False,
                'sell_signal': False,
                'signal_strength': 0,
                'confidence': 0,
                'source': 'advanced_indicators_neutral'
            }
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about advanced indicators usage"""
        cache_stats = self.taapi_client.get_cache_stats()
        
        return {
            'api_calls_made': self.api_calls_made,
            'cache_hits': self.cache_hits,
            'cache_stats': cache_stats,
            'client_type': type(self.taapi_client).__name__
        }