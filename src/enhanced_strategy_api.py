import aiohttp
import asyncio
import logging
import time
import json
from typing import Dict, List, Any, Optional
import config

class EnhancedSignalAPIClient:
    """
    Client for communicating with the Enhanced Crypto Signal API
    Provides AI-powered trading signals using Claude + Nebula AI + Taapi.io
    """
    
    def __init__(self, api_url: str = None, api_key: str = None):
        self.api_url = api_url or getattr(config, 'SIGNAL_API_URL', 'http://localhost:3000/api')
        self.api_key = api_key or getattr(config, 'SIGNAL_API_KEY', '')
        self.session = None
        self.request_timeout = getattr(config, 'API_REQUEST_TIMEOUT', 15)
        
        # Rate limiting - 120 REQUESTS PER HOUR (1 request every 30 seconds)
        self.last_request_time = 0
        self.min_request_interval = getattr(config, 'API_MIN_INTERVAL', 30.0)  # 30 seconds = 120 requests per hour
        
        # Cache for recent signals - SHORTER CACHE FOR MORE OPPORTUNITIES
        self.signal_cache = {}
        self.cache_duration = getattr(config, 'API_CACHE_DURATION', 120)  # 120 seconds (2 minutes) cache
        
        # Statistics
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.cache_hits = 0
        
        # API status
        self.api_available = True
        self.last_health_check = 0
        self.health_check_interval = getattr(config, 'API_HEALTH_CHECK_INTERVAL', 600)  # INCREASED from 300 to 600 seconds (10 minutes)
        
        # Enhanced features tracking
        self.taapi_available = True
        self.entries_avoided = 0
        self.quality_filtered = 0
        
        logging.info(f"Enhanced Signal API Client initialized: {self.api_url}")
        logging.info(f"Rate limiting: {self.min_request_interval}s interval, {self.cache_duration}s cache")
        logging.info("Enhanced features: Taapi integration, Entry avoidance, Signal quality scoring")
    
    async def initialize(self):
        """Initialize the HTTP session and check API health"""
        if not self.session:
            connector = aiohttp.TCPConnector(limit=10, limit_per_host=5)
            timeout = aiohttp.ClientTimeout(total=self.request_timeout)
            
            self.session = aiohttp.ClientSession(
                connector=connector,
                timeout=timeout,
                headers={
                    'Authorization': f'Bearer {self.api_key}',
                    'Content-Type': 'application/json',
                    'User-Agent': 'TradingBot/2.0-Enhanced'
                }
            )
        
        # Check API health and Taapi status
        await self.check_api_health()
        await self.check_taapi_health()
        
        return self.api_available
    
    async def close(self):
        """Close the HTTP session"""
        if self.session:
            await self.session.close()
            self.session = None
    
    async def check_api_health(self) -> bool:
        """Check if the API is healthy and responsive"""
        current_time = time.time()
        
        # Skip if recently checked
        if current_time - self.last_health_check < self.health_check_interval:
            return self.api_available
        
        try:
            if not self.session:
                await self.initialize()
            
            async with self.session.get(f"{self.api_url}/health") as response:
                if response.status == 200:
                    data = await response.json()
                    self.api_available = data.get('status') == 'healthy'
                    self.last_health_check = current_time
                    
                    if self.api_available:
                        logging.info("API health check passed")
                    else:
                        logging.warning("API health check failed - service unhealthy")
                        
                    return self.api_available
                else:
                    logging.warning(f"API health check failed with status {response.status}")
                    self.api_available = False
                    return False
                    
        except Exception as e:
            logging.error(f"API health check failed: {str(e)}")
            self.api_available = False
            return False

    async def check_taapi_health(self) -> bool:
        """Check if Taapi.io integration is available"""
        try:
            if not self.session:
                await self.initialize()
            
            async with self.session.get(f"{self.api_url}/v1/taapi/health") as response:
                if response.status == 200:
                    data = await response.json()
                    self.taapi_available = data.get('taapi_status') == 'healthy'
                    
                    if self.taapi_available:
                        logging.info("Taapi.io integration available")
                    else:
                        logging.warning("Taapi.io integration unavailable - using base signals only")
                        
                    return self.taapi_available
                else:
                    logging.warning(f"Taapi health check failed with status {response.status}")
                    self.taapi_available = False
                    return False
                    
        except Exception as e:
            logging.warning(f"Taapi health check failed: {str(e)} - using base signals only")
            self.taapi_available = False
            return False
    
    async def get_trading_signal(self, symbol: str, timeframe: str = '1h', 
                                analysis_depth: str = 'comprehensive',
                                risk_level: str = 'moderate',
                                wallet_address: str = None,
                                use_taapi: bool = True,
                                avoid_bad_entries: bool = True,
                                include_reasoning: bool = True) -> Optional[Dict[str, Any]]:
        """
        Get enhanced trading signal from the API with Taapi integration and entry avoidance
        
        Args:
            symbol: Trading pair (e.g., 'BTCUSDT')
            timeframe: Time period ('1m', '5m', '15m', '1h', '4h', '1d')
            analysis_depth: Analysis level ('basic', 'advanced', 'comprehensive')
            risk_level: Risk tolerance ('conservative', 'moderate', 'aggressive')
            wallet_address: Optional wallet address for personalized analysis
            use_taapi: Enable Taapi.io real-time indicators (default: True)
            avoid_bad_entries: Enable entry avoidance system (default: True)
            include_reasoning: Include detailed reasoning (default: True)
            
        Returns:
            Dictionary containing enhanced trading signal with Taapi analysis or None if failed
        """
        
        # Check cache first
        cache_key = f"{symbol}_{timeframe}_{analysis_depth}_{risk_level}_{use_taapi}_{avoid_bad_entries}"
        cached_signal = self._get_cached_signal(cache_key)
        if cached_signal:
            self.cache_hits += 1
            logging.info(f"Using cached enhanced signal for {symbol} (cache hit)")
            return cached_signal

        # Check if rate limiting would require a long wait
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.min_request_interval:
            wait_time = self.min_request_interval - time_since_last
            if wait_time > 15:  # Don't wait more than 15 seconds
                logging.warning(f"Rate limit requires {wait_time:.0f}s wait for {symbol}, returning cached or None")
                # Try to return any cached signal, even if slightly expired
                expired_cache = self._get_cached_signal(cache_key, allow_expired=True)
                if expired_cache:
                    logging.info(f"Using expired cached signal for {symbol} due to rate limiting")
                    return expired_cache
                return None

        # Check API availability
        if not self.api_available:
            await self.check_api_health()
            if not self.api_available:
                logging.warning("API unavailable, cannot get trading signal")
                return None

        try:
            # Rate limiting (with reasonable timeout)
            await self._enforce_rate_limit()
            
            if not self.session:
                await self.initialize()
            
            # Prepare enhanced request payload
            payload = {
                'symbol': symbol,
                'timeframe': timeframe,
                'risk_level': risk_level,
                'use_taapi': use_taapi and self.taapi_available,  # Only use if available
                'avoid_bad_entries': avoid_bad_entries,
                'include_reasoning': include_reasoning
            }
            
            # Legacy compatibility fields
            if analysis_depth:
                payload['analysis_depth'] = analysis_depth
            if wallet_address:
                payload['wallet_address'] = wallet_address
            
            # Make API request to ENHANCED endpoint
            self.total_requests += 1
            start_time = time.time()
            
            async with self.session.post(
                f"{self.api_url}/v1/enhanced-signal",  # ← CHANGED FROM /v1/signal
                json=payload
            ) as response:
                
                request_time = time.time() - start_time
                
                if response.status == 200:
                    data = await response.json()
                    
                    # Handle enhanced signal response format
                    if data.get('success'):
                        signal_data = data['data']
                    elif 'signal' in data or 'confidence' in data:
                        signal_data = data
                    else:
                        logging.error(f"Enhanced API request failed: {data.get('error', 'Unknown response format')}")
                        self.failed_requests += 1
                        return None
                    
                    self.successful_requests += 1
                    
                    # Log enhanced features
                    self._log_enhanced_features(symbol, signal_data)
                    
                    # Check if entry was avoided
                    if signal_data.get('signal') == 'AVOID':
                        self.entries_avoided += 1
                        logging.warning(f"Entry AVOIDED for {symbol}: {signal_data.get('reasoning', 'No reason provided')}")
                        
                        # Cache the avoidance signal
                        self._cache_signal(cache_key, signal_data)
                        return signal_data
                    
                    # Check signal quality
                    quality_grade = signal_data.get('signal_quality', {}).get('overall_grade', 'C')
                    if quality_grade in ['D', 'F']:
                        self.quality_filtered += 1
                        logging.info(f"Signal quality too low for {symbol}: Grade {quality_grade}")
                    
                    # Cache the signal
                    self._cache_signal(cache_key, signal_data)
                    
                    # Log successful request with enhanced info
                    taapi_info = ""
                    if signal_data.get('taapi_analysis'):
                        bullish = signal_data['taapi_analysis'].get('bullish_signals', 0)
                        bearish = signal_data['taapi_analysis'].get('bearish_signals', 0)
                        strength = signal_data['taapi_analysis'].get('market_strength', 'UNKNOWN')
                        taapi_info = f", Taapi: {bullish}B/{bearish}B, Strength: {strength}"
                    
                    quality_info = ""
                    if signal_data.get('signal_quality'):
                        grade = signal_data['signal_quality'].get('overall_grade', 'N/A')
                        score = signal_data['signal_quality'].get('overall_score', 0)
                        quality_info = f", Quality: {grade} ({score})"
                    
                    logging.info(f"Enhanced API signal received for {symbol}: "
                               f"{signal_data.get('signal', 'UNKNOWN')} "
                               f"(confidence: {signal_data.get('confidence', 0)}%"
                               f"{taapi_info}{quality_info}, "
                               f"request_time: {request_time:.2f}s)")
                    
                    return signal_data
                    
                elif response.status == 429:
                    # Rate limited
                    logging.warning("Enhanced API rate limit exceeded")
                    self.failed_requests += 1
                    return None
                    
                elif response.status == 401:
                    # Authentication failed
                    logging.error("Enhanced API authentication failed - check API key")
                    self.api_available = False
                    self.failed_requests += 1
                    return None
                    
                else:
                    error_text = await response.text()
                    logging.error(f"Enhanced API request failed with status {response.status}: {error_text}")
                    self.failed_requests += 1
                    return None
                    
        except asyncio.TimeoutError:
            logging.error(f"Enhanced API request timeout for {symbol}")
            self.failed_requests += 1
            return None
            
        except Exception as e:
            logging.error(f"Enhanced API request error for {symbol}: {str(e)}")
            self.failed_requests += 1
            return None

    def _log_enhanced_features(self, symbol: str, signal_data: Dict[str, Any]):
        """Log enhanced features from the signal"""
        try:
            # Log Taapi analysis if available
            if signal_data.get('taapi_analysis'):
                taapi = signal_data['taapi_analysis']
                logging.info(f"[TAAPI] {symbol}: "
                           f"Bullish:{taapi.get('bullish_signals', 0)} "
                           f"Bearish:{taapi.get('bearish_signals', 0)} "
                           f"Strength:{taapi.get('market_strength', 'UNKNOWN')} "
                           f"Trend:{taapi.get('trend_direction', 'UNKNOWN')}")
            
            # Log signal quality if available
            if signal_data.get('signal_quality'):
                quality = signal_data['signal_quality']
                logging.info(f"[QUALITY] {symbol}: "
                           f"Grade:{quality.get('overall_grade', 'N/A')} "
                           f"Score:{quality.get('overall_score', 0)} "
                           f"Risk:{quality.get('risk_level', {}).get('level', 'UNKNOWN')}")
            
            # Log validation results if available
            if signal_data.get('validation'):
                validation = signal_data['validation']
                confirmations = len(validation.get('confirmations', []))
                warnings = len(validation.get('warnings', []))
                logging.info(f"[VALIDATION] {symbol}: "
                           f"Score:{validation.get('score', 0)} "
                           f"Confirmations:{confirmations} "
                           f"Warnings:{warnings} "
                           f"Rec:{validation.get('recommendation', 'UNKNOWN')}")
            
            # Log risk factors if any
            risk_factors = signal_data.get('risk_factors', [])
            if risk_factors:
                logging.warning(f"[RISK] {symbol}: Risk factors detected: {', '.join(risk_factors)}")
            
            # Log warnings if any
            warnings = signal_data.get('warnings', [])
            if warnings:
                logging.warning(f"[WARNING] {symbol}: {', '.join(warnings)}")
                
        except Exception as e:
            logging.debug(f"Error logging enhanced features for {symbol}: {str(e)}")
    
    async def get_batch_signals(self, symbols: List[str], timeframe: str = '1h',
                               analysis_depth: str = 'advanced',
                               risk_level: str = 'moderate',
                               use_taapi: bool = True,
                               avoid_bad_entries: bool = True) -> Dict[str, Any]:
        """
        Get enhanced trading signals for multiple symbols using individual requests
        
        Args:
            symbols: List of trading pairs (max 10)
            timeframe: Time period
            analysis_depth: Analysis level
            risk_level: Risk tolerance
            use_taapi: Enable Taapi.io integration
            avoid_bad_entries: Enable entry avoidance
            
        Returns:
            Dictionary mapping symbols to their enhanced signals
        """
        
        if len(symbols) > 10:
            logging.warning("Too many symbols for batch request, limiting to 10")
            symbols = symbols[:10]
        
        results = {}
        avoided_count = 0
        success_count = 0
        
        # Process each symbol individually with enhanced features
        for symbol in symbols:
            try:
                signal = await self.get_trading_signal(
                    symbol=symbol,
                    timeframe=timeframe,
                    analysis_depth=analysis_depth,
                    risk_level=risk_level,
                    use_taapi=use_taapi,
                    avoid_bad_entries=avoid_bad_entries,
                    include_reasoning=False  # Reduce payload size for batch
                )
                
                if signal:
                    if signal.get('signal') == 'AVOID':
                        avoided_count += 1
                    else:
                        success_count += 1
                        
                    results[symbol] = {
                        'symbol': symbol,
                        'success': True,
                        **signal
                    }
                else:
                    results[symbol] = {
                        'symbol': symbol,
                        'success': False,
                        'error': 'Failed to get enhanced signal'
                    }
                    
            except Exception as e:
                logging.error(f"Failed to get enhanced signal for {symbol}: {str(e)}")
                results[symbol] = {
                    'symbol': symbol,
                    'success': False,
                    'error': str(e)
                }
        
        logging.info(f"Enhanced batch signals completed: {success_count} signals, {avoided_count} avoided, {len(symbols)-success_count-avoided_count} failed")
        
        return results
    
    def _get_cached_signal(self, cache_key: str, allow_expired: bool = False) -> Optional[Dict[str, Any]]:
        """Get signal from cache if still valid or if expired cache is allowed"""
        if cache_key in self.signal_cache:
            cached_data = self.signal_cache[cache_key]
            cache_age = time.time() - cached_data['timestamp']
            
            if cache_age < self.cache_duration:
                # Fresh cache
                return cached_data['signal']
            elif allow_expired and cache_age < (self.cache_duration * 2):  # Allow cache up to 2x duration when expired
                # Expired but still usable in rate-limited situations
                logging.debug(f"Using expired cache (age: {cache_age:.0f}s) for {cache_key}")
                return cached_data['signal']
            else:
                # Remove very old cache entry
                del self.signal_cache[cache_key]
        return None
    
    def _cache_signal(self, cache_key: str, signal_data: Dict[str, Any]):
        """Cache signal data"""
        self.signal_cache[cache_key] = {
            'signal': signal_data,
            'timestamp': time.time()
        }
        
        # Clean up old cache entries
        current_time = time.time()
        expired_keys = [
            key for key, data in self.signal_cache.items()
            if current_time - data['timestamp'] > self.cache_duration
        ]
        for key in expired_keys:
            del self.signal_cache[key]
    
    async def _enforce_rate_limit(self):
        """Enforce minimum interval between API requests"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.min_request_interval:
            sleep_time = self.min_request_interval - time_since_last
            # Don't sleep for more than 15 seconds to avoid timeouts
            if sleep_time > 15:
                logging.warning(f"Rate limit requires {sleep_time:.0f}s wait, skipping API request")
                raise asyncio.TimeoutError(f"Rate limit wait too long: {sleep_time:.0f}s")
            else:
                logging.info(f"Rate limiting: waiting {sleep_time:.1f}s before enhanced API request")
                await asyncio.sleep(sleep_time)
        
        self.last_request_time = time.time()

    async def get_bot_instructions(self, symbol: str, bot_type: str = 'python', 
                                  risk_level: str = 'moderate') -> Optional[Dict[str, Any]]:
        """
        Get detailed bot execution instructions from the enhanced API
        
        Args:
            symbol: Trading pair (e.g., 'BTCUSDT')
            bot_type: Type of bot ('python', 'ninjatrader', 'metatrader')
            risk_level: Risk tolerance ('conservative', 'moderate', 'aggressive')
            
        Returns:
            Dictionary containing detailed execution instructions or None if failed
        """
        try:
            if not self.session:
                await self.initialize()
            
            payload = {
                'symbol': symbol,
                'bot_type': bot_type,
                'risk_level': risk_level
            }
            
            async with self.session.post(
                f"{self.api_url}/v1/bot-instructions",
                json=payload
            ) as response:
                
                if response.status == 200:
                    data = await response.json()
                    
                    if data.get('success'):
                        instructions = data['data']
                        logging.info(f"Enhanced bot instructions received for {symbol}")
                        return instructions
                    else:
                        logging.error(f"Bot instructions request failed: {data.get('error', 'Unknown error')}")
                        return None
                else:
                    logging.error(f"Bot instructions request failed with status {response.status}")
                    return None
                    
        except Exception as e:
            logging.error(f"Error getting bot instructions for {symbol}: {str(e)}")
            return None

    async def validate_strategy(self, signal: Dict[str, Any], risk_params: Dict[str, Any], 
                               account_balance: float) -> Optional[Dict[str, Any]]:
        """
        Validate strategy before execution using the enhanced API
        
        Args:
            signal: Trading signal to validate
            risk_params: Risk management parameters
            account_balance: Current account balance
            
        Returns:
            Dictionary containing validation results or None if failed
        """
        try:
            if not self.session:
                await self.initialize()
            
            payload = {
                'signal': signal,
                'risk_params': risk_params,
                'account_balance': account_balance
            }
            
            async with self.session.post(
                f"{self.api_url}/v1/validate-strategy",
                json=payload
            ) as response:
                
                if response.status == 200:
                    data = await response.json()
                    
                    if data.get('success'):
                        validation = data['data']
                        logging.info(f"Enhanced strategy validation completed: {validation.get('valid', False)}")
                        return validation
                    else:
                        logging.error(f"Strategy validation failed: {data.get('error', 'Unknown error')}")
                        return None
                else:
                    logging.error(f"Strategy validation request failed with status {response.status}")
                    return None
                    
        except Exception as e:
            logging.error(f"Error validating strategy: {str(e)}")
            return None

    def is_signal_quality_acceptable(self, signal: Dict[str, Any], min_grade: str = 'C') -> bool:
        """
        Check if enhanced signal quality is acceptable for trading
        
        Args:
            signal: Enhanced signal data
            min_grade: Minimum acceptable grade ('A+', 'A', 'B+', 'B', 'C', 'D', 'F')
            
        Returns:
            True if signal quality is acceptable
        """
        try:
            # Grade hierarchy
            grade_values = {'A+': 95, 'A': 85, 'B+': 75, 'B': 65, 'C': 55, 'D': 45, 'F': 0}
            min_value = grade_values.get(min_grade, 55)
            
            # Check signal quality grade
            quality = signal.get('signal_quality', {})
            overall_grade = quality.get('overall_grade', 'F')
            overall_score = quality.get('overall_score', 0)
            
            grade_acceptable = grade_values.get(overall_grade, 0) >= min_value
            score_acceptable = overall_score >= min_value
            
            # Check validation recommendation
            validation = signal.get('validation', {})
            recommendation = validation.get('recommendation', 'AVOID')
            validation_acceptable = recommendation in ['PROCEED']
            
            # Check if entry should be avoided
            entry_quality = signal.get('entry_quality', {})
            should_avoid = entry_quality.get('should_avoid', True)
            
            acceptable = grade_acceptable and score_acceptable and validation_acceptable and not should_avoid
            
            if not acceptable:
                reasons = []
                if not grade_acceptable:
                    reasons.append(f"Grade {overall_grade} below {min_grade}")
                if not score_acceptable:
                    reasons.append(f"Score {overall_score} below {min_value}")
                if not validation_acceptable:
                    reasons.append(f"Validation: {recommendation}")
                if should_avoid:
                    reasons.append("Entry avoidance triggered")
                
                logging.info(f"Signal quality not acceptable: {', '.join(reasons)}")
            
            return acceptable
            
        except Exception as e:
            logging.error(f"Error checking signal quality: {str(e)}")
            return False

    def convert_to_internal_format(self, api_signal: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert enhanced API signal format to internal bot format
        
        Args:
            api_signal: Enhanced signal data from API
            
        Returns:
            Signal data in internal format with enhanced features
        """
        try:
            # Map API signal to internal format
            signal_action = api_signal.get('signal', 'HOLD')
            confidence = api_signal.get('confidence', 50) / 100.0  # Convert to 0-1 scale
            
            # Convert to buy/sell signals
            buy_signal = signal_action == 'BUY'
            sell_signal = signal_action == 'SELL'
            avoid_signal = signal_action == 'AVOID'
            
            # Get additional data
            entry_price = api_signal.get('entry_price', 0)
            stop_loss = api_signal.get('stop_loss', 0)
            take_profit_1 = api_signal.get('take_profit_1', 0)
            take_profit_2 = api_signal.get('take_profit_2', 0)
            take_profit_3 = api_signal.get('take_profit_3', 0)
            position_size_pct = api_signal.get('position_size_percent', 5)
            
            # Enhanced data from API
            taapi_analysis = api_signal.get('taapi_analysis', {})
            signal_quality = api_signal.get('signal_quality', {})
            validation = api_signal.get('validation', {})
            entry_quality = api_signal.get('entry_quality', {})
            risk_factors = api_signal.get('risk_factors', [])
            
            # Extract market context data
            market_context = api_signal.get('market_context', {})
            market_regime = market_context.get('market_regime', {})
            risk_environment = market_context.get('risk_environment', {})
            risk_management = api_signal.get('risk_management', {})
            
            return {
                "buy_signal": buy_signal and not avoid_signal,
                "sell_signal": sell_signal and not avoid_signal,
                "avoid_signal": avoid_signal,
                "signal_strength": confidence,
                "source": "enhanced_api_v2",
                
                # Enhanced features
                "enhanced_data": {
                    "taapi_analysis": taapi_analysis,
                    "signal_quality": signal_quality,
                    "validation": validation,
                    "entry_quality": entry_quality,
                    "quality_grade": signal_quality.get('overall_grade', 'C'),
                    "quality_score": signal_quality.get('overall_score', 50),
                    "validation_score": validation.get('score', 50),
                    "recommendation": validation.get('recommendation', 'PROCEED'),
                    "should_avoid": entry_quality.get('should_avoid', False),
                    "avoidance_factors": entry_quality.get('avoidance_factors', []),
                    "market_strength": taapi_analysis.get('market_strength', 'UNKNOWN'),
                    "trend_direction": taapi_analysis.get('trend_direction', 'UNKNOWN'),
                    "bullish_signals": taapi_analysis.get('bullish_signals', 0),
                    "bearish_signals": taapi_analysis.get('bearish_signals', 0),
                    "risk_factors": risk_factors,
                    "enhanced_by": api_signal.get('metadata', {}).get('enhanced_by', 'unknown'),
                    "taapi_enabled": api_signal.get('metadata', {}).get('taapi_enabled', False)
                },
                
                "api_data": {
                    "signal": signal_action,
                    "confidence": api_signal.get('confidence', 50),
                    "strength": api_signal.get('strength', 'MODERATE'),
                    "entry_price": entry_price,
                    "stop_loss": stop_loss,
                    "take_profit_1": take_profit_1,
                    "take_profit_2": take_profit_2,
                    "take_profit_3": take_profit_3,
                    "position_size_percent": position_size_pct,
                    "risk_reward_ratio": api_signal.get('risk_reward_ratio', 2.0),
                    
                    # Enhanced market phase analysis
                    "market_phase": market_regime.get('market_phase', 'CONSOLIDATION'),
                    "primary_trend": market_regime.get('primary_trend', 'NEUTRAL'),
                    "volatility_regime": market_regime.get('volatility_regime', 'NORMAL_VOLATILITY'),
                    "strategy_type": market_context.get('strategy_type', 'Standard Entry'),
                    
                    # Risk environment
                    "risk_score": risk_environment.get('risk_score', 50),
                    "risk_environment": risk_environment.get('risk_environment', 'MODERATE'),
                    
                    # Risk management
                    "risk_level": risk_management.get('risk_level', 'moderate'),
                    "max_position_size": risk_management.get('position_sizing', {}).get('max_allowed', 5.0),
                    
                    # Legacy fields for backward compatibility
                    "timeframe": api_signal.get('timeframe', 'SWING'),
                    "market_sentiment": api_signal.get('market_sentiment', 'NEUTRAL'),
                    "volatility_rating": api_signal.get('volatility_rating', 'MEDIUM'),
                    "onchain_score": api_signal.get('onchain_score', 50),
                    "whale_influence": api_signal.get('whale_influence', 'NEUTRAL'),
                    "defi_impact": api_signal.get('defi_impact', 'NEUTRAL'),
                    "technical_score": api_signal.get('technical_score', 50),
                    "momentum_score": api_signal.get('momentum_score', 50),
                    "trend_score": api_signal.get('trend_score', 50),
                    "risk_factors": risk_factors,
                    "catalysts": api_signal.get('catalysts', []),
                    "probability_success": api_signal.get('probability_success', 50),
                    "time_horizon_hours": api_signal.get('time_horizon_hours', 24),
                    "max_drawdown_percent": api_signal.get('max_drawdown_percent', 5),
                    "reasoning": api_signal.get('reasoning', 'Enhanced AI-generated signal'),
                    "market_structure": api_signal.get('market_structure', 'RANGING'),
                    "institutional_flow": api_signal.get('institutional_flow', 'NEUTRAL')
                },
                "details": {
                    "api_confidence": api_signal.get('confidence', 50),
                    "enhanced_features": {
                        "taapi_integrated": bool(taapi_analysis),
                        "quality_scored": bool(signal_quality),
                        "entry_validated": bool(validation),
                        "risk_assessed": bool(risk_factors)
                    },
                    "onchain_analysis": {
                        "whale_influence": api_signal.get('whale_influence', 'NEUTRAL'),
                        "defi_impact": api_signal.get('defi_impact', 'NEUTRAL'),
                        "score": api_signal.get('onchain_score', 50)
                    },
                    "technical_analysis": {
                        "score": api_signal.get('technical_score', 50),
                        "momentum": api_signal.get('momentum_score', 50),
                        "trend": api_signal.get('trend_score', 50)
                    },
                    "risk_assessment": {
                        "factors": risk_factors,
                        "max_drawdown": api_signal.get('max_drawdown_percent', 5),
                        "probability": api_signal.get('probability_success', 50)
                    }
                },
                "timestamp": time.time()
            }
            
        except Exception as e:
            logging.error(f"Error converting enhanced API signal format: {str(e)}")
            return {
                "buy_signal": False,
                "sell_signal": False,
                "avoid_signal": True,
                "signal_strength": 0,
                "source": "api_error",
                "error": str(e)
            }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get enhanced API client statistics"""
        total = self.total_requests
        success_rate = (self.successful_requests / total * 100) if total > 0 else 0
        
        return {
            "total_requests": total,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "success_rate_percent": round(success_rate, 2),
            "cache_hits": self.cache_hits,
            "api_available": self.api_available,
            "taapi_available": self.taapi_available,
            "cached_signals": len(self.signal_cache),
            "last_health_check": self.last_health_check,
            
            # Enhanced features statistics
            "enhanced_features": {
                "entries_avoided": self.entries_avoided,
                "quality_filtered": self.quality_filtered,
                "enhancement_active": self.taapi_available
            }
        }


class MockAPIClient:
    """
    Mock API client for testing or when API is unavailable
    """
    
    def __init__(self):
        self.api_available = False
        self.taapi_available = False
        logging.info("Mock API client initialized - using fallback signals")
    
    async def initialize(self):
        return False
    
    async def close(self):
        pass
    
    async def get_trading_signal(self, symbol: str, **kwargs) -> Optional[Dict[str, Any]]:
        """Return mock enhanced signal"""
        import random
        
        signals = ['BUY', 'SELL', 'HOLD']
        signal = random.choice(signals)
        confidence = random.randint(30, 85)
        
        # Mock enhanced features
        quality_grades = ['A+', 'A', 'B+', 'B', 'C']
        quality_grade = random.choice(quality_grades)
        quality_score = random.randint(50, 95)
        
        return {
            "signal": signal,
            "confidence": confidence,
            "strength": "MODERATE",
            "entry_price": 40000 + random.randint(-5000, 5000),
            "stop_loss": 38000,
            "take_profit_1": 42000,
            "reasoning": "Mock enhanced signal for testing",
            "onchain_score": random.randint(30, 80),
            "whale_influence": "NEUTRAL",
            "source": "mock_enhanced_api",
            
            # Mock enhanced features
            "signal_quality": {
                "overall_grade": quality_grade,
                "overall_score": quality_score
            },
            "taapi_analysis": {
                "bullish_signals": random.randint(0, 8),
                "bearish_signals": random.randint(0, 8),
                "market_strength": random.choice(["WEAK", "MODERATE", "STRONG"]),
                "trend_direction": random.choice(["BULLISH", "BEARISH", "SIDEWAYS"])
            },
            "validation": {
                "score": random.randint(40, 90),
                "recommendation": random.choice(["PROCEED", "CAUTION", "AVOID"])
            },
            "entry_quality": {
                "should_avoid": random.choice([True, False]),
                "quality_grade": quality_grade,
                "recommendation": "PROCEED"
            },
            "metadata": {
                "enhanced_by": "mock_system",
                "taapi_enabled": False
            }
        }
    
    async def get_batch_signals(self, symbols: List[str], **kwargs) -> Dict[str, Any]:
        """Return mock batch enhanced signals"""
        results = {}
        for symbol in symbols:
            signal = await self.get_trading_signal(symbol)
            if signal:
                results[symbol] = {
                    'symbol': symbol,
                    'success': True,
                    **signal
                }
        return results

    def is_signal_quality_acceptable(self, signal: Dict[str, Any], min_grade: str = 'C') -> bool:
        """Mock quality check"""
        return signal.get('signal_quality', {}).get('overall_grade', 'F') in ['A+', 'A', 'B+', 'B', 'C']
    
    def convert_to_internal_format(self, api_signal: Dict[str, Any]) -> Dict[str, Any]:
        """Convert mock signal to internal format"""
        return {
            "buy_signal": api_signal.get('signal') == 'BUY',
            "sell_signal": api_signal.get('signal') == 'SELL', 
            "avoid_signal": api_signal.get('signal') == 'AVOID',
            "signal_strength": api_signal.get('confidence', 50) / 100.0,
            "source": "mock_enhanced_api",
            "enhanced_data": {
                "quality_grade": api_signal.get('signal_quality', {}).get('overall_grade', 'C'),
                "taapi_enabled": False
            }
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        return {
            "total_requests": 0,
            "api_available": False,
            "taapi_available": False,
            "mock_mode": True,
            "enhanced_features": {
                "entries_avoided": 0,
                "quality_filtered": 0,
                "enhancement_active": False
            }
        }