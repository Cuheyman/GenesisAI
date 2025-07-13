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
    Provides AI-powered trading signals using Claude + Nebula AI
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
        
        logging.info(f"Enhanced Signal API Client initialized: {self.api_url}")
        logging.info(f"Rate limiting: {self.min_request_interval}s interval, {self.cache_duration}s cache")
    
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
                    'User-Agent': 'TradingBot/2.0'
                }
            )
        
        # Check API health
        await self.check_api_health()
        
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
    
    async def get_trading_signal(self, symbol: str, timeframe: str = '1h', 
                                analysis_depth: str = 'comprehensive',
                                risk_level: str = 'moderate',
                                wallet_address: str = None) -> Optional[Dict[str, Any]]:
        """
        Get enhanced trading signal from the API with market phase analysis
        
        Args:
            symbol: Trading pair (e.g., 'BTCUSDT')
            timeframe: Time period ('1m', '5m', '15m', '1h', '4h', '1d')
            analysis_depth: Analysis level ('basic', 'advanced', 'comprehensive')
            risk_level: Risk tolerance ('conservative', 'moderate', 'aggressive')
            wallet_address: Optional wallet address for personalized analysis
            
        Returns:
            Dictionary containing enhanced trading signal with market phase analysis or None if failed
        """
        
        # Check cache first
        cache_key = f"{symbol}_{timeframe}_{analysis_depth}_{risk_level}"
        cached_signal = self._get_cached_signal(cache_key)
        if cached_signal:
            self.cache_hits += 1
            logging.info(f"Using cached signal for {symbol} (cache hit)")
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
            
            # Prepare request payload
            payload = {
                'symbol': symbol,
                'timeframe': timeframe,
                'analysis_depth': analysis_depth,
                'risk_level': risk_level
            }
            
            if wallet_address:
                payload['wallet_address'] = wallet_address
            
            # Make API request
            self.total_requests += 1
            start_time = time.time()
            
            async with self.session.post(
                f"{self.api_url}/v1/signal",
                json=payload
            ) as response:
                
                request_time = time.time() - start_time
                
                if response.status == 200:
                    data = await response.json()
                    
                    # Check if it's the expected format with 'success' field
                    if data.get('success'):
                        signal_data = data['data']
                    # Handle direct signal response format (your API's actual format)
                    elif 'signal' in data or 'confidence' in data:
                        signal_data = data
                    else:
                        logging.error(f"API request failed: {data.get('error', 'Unknown response format')}")
                        self.failed_requests += 1
                        return None
                    
                    self.successful_requests += 1
                    
                    # Cache the signal
                    self._cache_signal(cache_key, signal_data)
                    
                    # Log successful request
                    logging.info(f"API signal received for {symbol}: "
                               f"{signal_data.get('signal', 'UNKNOWN')} "
                               f"(confidence: {signal_data.get('confidence', 0)}%, "
                               f"request_time: {request_time:.2f}s)")
                    
                    return signal_data
                    
                elif response.status == 429:
                    # Rate limited
                    logging.warning("API rate limit exceeded")
                    self.failed_requests += 1
                    return None
                    
                elif response.status == 401:
                    # Authentication failed
                    logging.error("API authentication failed - check API key")
                    self.api_available = False
                    self.failed_requests += 1
                    return None
                    
                else:
                    error_text = await response.text()
                    logging.error(f"API request failed with status {response.status}: {error_text}")
                    self.failed_requests += 1
                    return None
                    
        except asyncio.TimeoutError:
            logging.error(f"API request timeout for {symbol}")
            self.failed_requests += 1
            return None
            
        except Exception as e:
            logging.error(f"API request error for {symbol}: {str(e)}")
            self.failed_requests += 1
            return None
    
    async def get_batch_signals(self, symbols: List[str], timeframe: str = '1h',
                               analysis_depth: str = 'advanced',
                               risk_level: str = 'moderate') -> Dict[str, Any]:
        """
        Get trading signals for multiple symbols using individual requests
        (since the API doesn't have a batch endpoint)
        
        Args:
            symbols: List of trading pairs (max 10)
            timeframe: Time period
            analysis_depth: Analysis level
            risk_level: Risk tolerance
            
        Returns:
            Dictionary mapping symbols to their signals
        """
        
        if len(symbols) > 10:
            logging.warning("Too many symbols for batch request, limiting to 10")
            symbols = symbols[:10]
        
        results = {}
        
        # Process each symbol individually
        for symbol in symbols:
            try:
                signal = await self.get_trading_signal(
                    symbol=symbol,
                    timeframe=timeframe,
                    analysis_depth=analysis_depth,
                    risk_level=risk_level
                )
                
                if signal:
                    results[symbol] = {
                        'symbol': symbol,
                        'success': True,
                        **signal
                    }
                else:
                    results[symbol] = {
                        'symbol': symbol,
                        'success': False,
                        'error': 'Failed to get signal'
                    }
                    
            except Exception as e:
                logging.error(f"Failed to get signal for {symbol}: {str(e)}")
                results[symbol] = {
                    'symbol': symbol,
                    'success': False,
                    'error': str(e)
                }
        
        successful_count = sum(1 for r in results.values() if r.get('success'))
        logging.info(f"Batch signals completed: {successful_count}/{len(symbols)} successful")
        
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
                logging.info(f"Rate limiting: waiting {sleep_time:.1f}s before API request")
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
                        logging.info(f"Bot instructions received for {symbol}")
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
                        logging.info(f"Strategy validation completed: {validation.get('valid', False)}")
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
    
    def convert_to_internal_format(self, api_signal: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert API signal format to internal bot format
        
        Args:
            api_signal: Signal data from API
            
        Returns:
            Signal data in internal format
        """
        try:
            # Map API signal to internal format
            signal_action = api_signal.get('signal', 'HOLD')
            confidence = api_signal.get('confidence', 50) / 100.0  # Convert to 0-1 scale
            
            # Convert to buy/sell signals
            buy_signal = signal_action == 'BUY'
            sell_signal = signal_action == 'SELL'
            
            # Get additional data
            entry_price = api_signal.get('entry_price', 0)
            stop_loss = api_signal.get('stop_loss', 0)
            take_profit_1 = api_signal.get('take_profit_1', 0)
            take_profit_2 = api_signal.get('take_profit_2', 0)
            take_profit_3 = api_signal.get('take_profit_3', 0)
            position_size_pct = api_signal.get('position_size_percent', 5)
            
            # Enhanced data from API
            onchain_score = api_signal.get('onchain_score', 50)
            whale_influence = api_signal.get('whale_influence', 'NEUTRAL')
            defi_impact = api_signal.get('defi_impact', 'NEUTRAL')
            risk_factors = api_signal.get('risk_factors', [])
            catalysts = api_signal.get('catalysts', [])
            
            # Extract market context data
            market_context = api_signal.get('market_context', {})
            market_regime = market_context.get('market_regime', {})
            risk_environment = market_context.get('risk_environment', {})
            risk_management = api_signal.get('risk_management', {})
            
            return {
                "buy_signal": buy_signal,
                "sell_signal": sell_signal,
                "signal_strength": confidence,
                "source": "enhanced_api",
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
                    "onchain_score": onchain_score,
                    "whale_influence": whale_influence,
                    "defi_impact": defi_impact,
                    "technical_score": api_signal.get('technical_score', 50),
                    "momentum_score": api_signal.get('momentum_score', 50),
                    "trend_score": api_signal.get('trend_score', 50),
                    "risk_factors": risk_factors,
                    "catalysts": catalysts,
                    "probability_success": api_signal.get('probability_success', 50),
                    "time_horizon_hours": api_signal.get('time_horizon_hours', 24),
                    "max_drawdown_percent": api_signal.get('max_drawdown_percent', 5),
                    "reasoning": api_signal.get('reasoning', 'AI-generated signal'),
                    "market_structure": api_signal.get('market_structure', 'RANGING'),
                    "institutional_flow": api_signal.get('institutional_flow', 'NEUTRAL')
                },
                "details": {
                    "api_confidence": api_signal.get('confidence', 50),
                    "onchain_analysis": {
                        "whale_influence": whale_influence,
                        "defi_impact": defi_impact,
                        "score": onchain_score
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
            logging.error(f"Error converting API signal format: {str(e)}")
            return {
                "buy_signal": False,
                "sell_signal": False,
                "signal_strength": 0,
                "source": "api_error",
                "error": str(e)
            }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get API client statistics"""
        total = self.total_requests
        success_rate = (self.successful_requests / total * 100) if total > 0 else 0
        
        return {
            "total_requests": total,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "success_rate_percent": round(success_rate, 2),
            "cache_hits": self.cache_hits,
            "api_available": self.api_available,
            "cached_signals": len(self.signal_cache),
            "last_health_check": self.last_health_check
        }


class MockAPIClient:
    """
    Mock API client for testing or when API is unavailable
    """
    
    def __init__(self):
        self.api_available = False
        logging.info("Mock API client initialized - using fallback signals")
    
    async def initialize(self):
        return False
    
    async def close(self):
        pass
    
    async def get_trading_signal(self, symbol: str, **kwargs) -> Optional[Dict[str, Any]]:
        """Return mock signal"""
        import random
        
        signals = ['BUY', 'SELL', 'HOLD']
        signal = random.choice(signals)
        confidence = random.randint(30, 85)
        
        return {
            "signal": signal,
            "confidence": confidence,
            "strength": "MODERATE",
            "entry_price": 40000 + random.randint(-5000, 5000),
            "stop_loss": 38000,
            "take_profit_1": 42000,
            "reasoning": "Mock signal for testing",
            "onchain_score": random.randint(30, 80),
            "whale_influence": "NEUTRAL",
            "source": "mock_api"
        }
    
    async def get_batch_signals(self, symbols: List[str], **kwargs) -> Dict[str, Any]:
        """Return mock batch signals"""
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
    
    def convert_to_internal_format(self, api_signal: Dict[str, Any]) -> Dict[str, Any]:
        """Convert mock signal to internal format"""
        return {
            "buy_signal": api_signal.get('signal') == 'BUY',
            "sell_signal": api_signal.get('signal') == 'SELL', 
            "signal_strength": api_signal.get('confidence', 50) / 100.0,
            "source": "mock_api"
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        return {
            "total_requests": 0,
            "api_available": False,
            "mock_mode": True
        }