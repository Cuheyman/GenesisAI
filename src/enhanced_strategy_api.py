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
        self.request_timeout = getattr(config, 'API_REQUEST_TIMEOUT', 30)
        
        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = getattr(config, 'API_MIN_INTERVAL', 2.0)  # 2 seconds between requests
        
        # Cache for recent signals
        self.signal_cache = {}
        self.cache_duration = getattr(config, 'API_CACHE_DURATION', 60)  # 1 minute cache
        
        # Statistics
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.cache_hits = 0
        
        # API status
        self.api_available = True
        self.last_health_check = 0
        self.health_check_interval = 300  # 5 minutes
        
        logging.info(f"Enhanced Signal API Client initialized: {self.api_url}")
    
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
        Get enhanced trading signal from the API
        
        Args:
            symbol: Trading pair (e.g., 'BTCUSDT')
            timeframe: Time period ('1m', '5m', '15m', '1h', '4h', '1d')
            analysis_depth: Analysis level ('basic', 'advanced', 'comprehensive')
            risk_level: Risk tolerance ('conservative', 'moderate', 'aggressive')
            wallet_address: Optional wallet address for personalized analysis
            
        Returns:
            Dictionary containing enhanced trading signal or None if failed
        """
        
        # Check cache first
        cache_key = f"{symbol}_{timeframe}_{analysis_depth}_{risk_level}"
        cached_signal = self._get_cached_signal(cache_key)
        if cached_signal:
            self.cache_hits += 1
            return cached_signal
        
        # Rate limiting
        await self._enforce_rate_limit()
        
        # Check API availability
        if not self.api_available:
            await self.check_api_health()
            if not self.api_available:
                logging.warning("API unavailable, cannot get trading signal")
                return None
        
        try:
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
                f"{self.api_url}/v1/signals/generate",
                json=payload
            ) as response:
                
                request_time = time.time() - start_time
                
                if response.status == 200:
                    data = await response.json()
                    
                    if data.get('success'):
                        signal_data = data['data']
                        self.successful_requests += 1
                        
                        # Cache the signal
                        self._cache_signal(cache_key, signal_data)
                        
                        # Log successful request
                        logging.info(f"API signal received for {symbol}: "
                                   f"{signal_data.get('signal', 'UNKNOWN')} "
                                   f"(confidence: {signal_data.get('confidence', 0)}%, "
                                   f"request_time: {request_time:.2f}s)")
                        
                        return signal_data
                    else:
                        logging.error(f"API request failed: {data.get('error', 'Unknown error')}")
                        self.failed_requests += 1
                        return None
                        
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
        Get trading signals for multiple symbols in a single request
        
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
        
        # Rate limiting
        await self._enforce_rate_limit()
        
        if not self.api_available:
            await self.check_api_health()
            if not self.api_available:
                return {}
        
        try:
            if not self.session:
                await self.initialize()
            
            payload = {
                'symbols': symbols,
                'timeframe': timeframe,
                'analysis_depth': analysis_depth,
                'risk_level': risk_level
            }
            
            self.total_requests += 1
            
            async with self.session.post(
                f"{self.api_url}/v1/signals/batch",
                json=payload
            ) as response:
                
                if response.status == 200:
                    data = await response.json()
                    
                    if data.get('success'):
                        results = {}
                        for result in data.get('results', []):
                            symbol = result.get('symbol')
                            if result.get('success') and symbol:
                                results[symbol] = result
                                
                                # Cache individual signals
                                cache_key = f"{symbol}_{timeframe}_{analysis_depth}_{risk_level}"
                                self._cache_signal(cache_key, result)
                        
                        self.successful_requests += 1
                        logging.info(f"Batch signals received for {len(results)}/{len(symbols)} symbols")
                        return results
                    else:
                        logging.error(f"Batch API request failed: {data.get('error')}")
                        self.failed_requests += 1
                        return {}
                else:
                    error_text = await response.text()
                    logging.error(f"Batch API request failed with status {response.status}: {error_text}")
                    self.failed_requests += 1
                    return {}
                    
        except Exception as e:
            logging.error(f"Batch API request error: {str(e)}")
            self.failed_requests += 1
            return {}
    
    def _get_cached_signal(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get signal from cache if still valid"""
        if cache_key in self.signal_cache:
            cached_data = self.signal_cache[cache_key]
            if time.time() - cached_data['timestamp'] < self.cache_duration:
                return cached_data['signal']
            else:
                # Remove expired cache entry
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
            await asyncio.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
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
            
            return {
                "buy_signal": buy_signal,
                "sell_signal": sell_signal,
                "signal_strength": confidence,
                "source": "enhanced_api",
                "api_data": {
                    "signal": signal_action,
                    "confidence": api_signal.get('confidence', 50),
                    "strength": api_signal.get('strength', 'MODERATE'),
                    "timeframe": api_signal.get('timeframe', 'SWING'),
                    "entry_price": entry_price,
                    "stop_loss": stop_loss,
                    "take_profits": [take_profit_1, take_profit_2, take_profit_3],
                    "position_size_percent": position_size_pct,
                    "risk_reward_ratio": api_signal.get('risk_reward_ratio', 2.0),
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