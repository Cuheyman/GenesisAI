import requests
import logging
import time
import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
import hashlib
import threading
from functools import wraps
import config

class RateLimiter:
    """Rate limiter for Taapi.io free tier: 1 request per 15 seconds"""
    def __init__(self, max_requests: int = 1, time_window: int = 15):
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests = []
        self.lock = threading.Lock()
    
    def wait_if_needed(self):
        with self.lock:
            now = time.time()
            # Remove old requests outside the time window
            self.requests = [req_time for req_time in self.requests if now - req_time < self.time_window]
            
            if len(self.requests) >= self.max_requests:
                # Calculate how long to wait
                oldest_request = min(self.requests)
                wait_time = self.time_window - (now - oldest_request) + 0.1  # Add small buffer
                if wait_time > 0:
                    logging.debug(f"Rate limit reached, waiting {wait_time:.1f} seconds")
                    time.sleep(wait_time)
                    # Clean up after waiting
                    now = time.time()
                    self.requests = [req_time for req_time in self.requests if now - req_time < self.time_window]
            
            # Record this request
            self.requests.append(now)

def retry_with_backoff(max_retries: int = 3, backoff_factor: float = 2):
    """Decorator for retry logic with exponential backoff"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except requests.exceptions.RequestException as e:
                    if attempt == max_retries - 1:
                        raise e
                    wait_time = backoff_factor ** attempt
                    logging.warning(f"Taapi.io request failed (attempt {attempt + 1}), retrying in {wait_time}s: {str(e)}")
                    time.sleep(wait_time)
            return None
        return wrapper
    return decorator

class TaapiCache:
    """Simple in-memory cache with TTL for Taapi.io responses"""
    def __init__(self):
        self.cache = {}
        self.ttl_cache = {}
    
    def _get_cache_key(self, indicator: str, exchange: str, symbol: str, interval: str, **kwargs) -> str:
        """Generate unique cache key for request"""
        key_data = f"{indicator}:{exchange}:{symbol}:{interval}:{json.dumps(sorted(kwargs.items()), sort_keys=True)}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _get_ttl_for_interval(self, interval: str) -> int:
        """Get appropriate TTL based on timeframe"""
        ttl_mapping = {
            '1m': 60,      # 1 minute
            '5m': 250,     # ~4 minutes
            '15m': 800,    # ~13 minutes  
            '30m': 1600,   # ~27 minutes
            '1h': 3300,    # ~55 minutes
            '2h': 6600,    # ~110 minutes
            '4h': 13200,   # ~220 minutes
            '6h': 19800,   # ~330 minutes
            '12h': 39600,  # ~660 minutes
            '1d': 79200,   # ~22 hours
            '1w': 518400   # ~6 days
        }
        return ttl_mapping.get(interval, 300)  # Default 5 minutes
    
    def get(self, indicator: str, exchange: str, symbol: str, interval: str, **kwargs) -> Optional[Any]:
        """Get cached result if valid"""
        cache_key = self._get_cache_key(indicator, exchange, symbol, interval, **kwargs)
        
        if cache_key in self.cache:
            expiry_time = self.ttl_cache.get(cache_key, 0)
            if time.time() < expiry_time:
                logging.debug(f"Cache hit for {indicator} {symbol} {interval}")
                return self.cache[cache_key]
            else:
                # Remove expired entry
                del self.cache[cache_key]
                if cache_key in self.ttl_cache:
                    del self.ttl_cache[cache_key]
        
        return None
    
    def set(self, indicator: str, exchange: str, symbol: str, interval: str, value: Any, **kwargs):
        """Cache the result with appropriate TTL"""
        cache_key = self._get_cache_key(indicator, exchange, symbol, interval, **kwargs)
        ttl = self._get_ttl_for_interval(interval)
        
        self.cache[cache_key] = value
        self.ttl_cache[cache_key] = time.time() + ttl
        
        logging.debug(f"Cached {indicator} {symbol} {interval} for {ttl}s")

class TaapiClient:
    """Client for Taapi.io API with rate limiting and caching"""
    
    def __init__(self, api_secret: str = None):
        self.api_secret = api_secret or getattr(config, 'TAAPI_API_SECRET', '')
        self.base_url = "https://api.taapi.io"
        self.session = requests.Session()
        self.session.headers.update({
            'Accept': 'application/json',
            'User-Agent': 'TradingBot/1.0'
        })
        
        # Rate limiter for free tier (1 request per 15 seconds)
        self.rate_limiter = RateLimiter(max_requests=1, time_window=15)
        
        # Cache for responses
        self.cache = TaapiCache()
        
        # Track API availability
        self.api_available = True
        self.last_error_time = 0
        self.error_backoff = 300  # 5 minutes backoff after errors
        
        logging.info("Taapi.io client initialized (Free tier: 1 request/15s)")
    
    def _is_api_available(self) -> bool:
        """Check if API is available (not in error backoff period)"""
        if not self.api_available:
            if time.time() - self.last_error_time > self.error_backoff:
                self.api_available = True
                logging.info("Taapi.io API available again after backoff period")
        return self.api_available
    
    def _handle_error(self, error: Exception):
        """Handle API errors and set backoff if needed"""
        if isinstance(error, requests.exceptions.RequestException):
            self.api_available = False
            self.last_error_time = time.time()
            logging.warning(f"Taapi.io API error, backing off for {self.error_backoff}s: {str(error)}")
    
    @retry_with_backoff(max_retries=2, backoff_factor=1.5)
    def _make_request(self, indicator: str, params: Dict[str, Any]) -> Optional[Dict]:
        """Make request to Taapi.io API with error handling"""
        if not self._is_api_available():
            return None
        
        if not self.api_secret:
            logging.error("Taapi.io API secret not configured")
            return None
        
        # Apply rate limiting
        self.rate_limiter.wait_if_needed()
        
        # Add API secret to params
        params['secret'] = self.api_secret
        
        try:
            url = f"{self.base_url}/{indicator}"
            response = self.session.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 429:
                logging.warning("Taapi.io rate limit exceeded")
                self._handle_error(requests.exceptions.RequestException("Rate limit exceeded"))
                return None
            else:
                logging.error(f"Taapi.io API error {response.status_code}: {response.text}")
                self._handle_error(requests.exceptions.RequestException(f"HTTP {response.status_code}"))
                return None
                
        except Exception as e:
            logging.error(f"Taapi.io request failed: {str(e)}")
            self._handle_error(e)
            return None
    
    async def get_indicator(self, indicator: str, symbol: str, interval: str, 
                          exchange: str = "binance", **kwargs) -> Optional[Dict]:
        """Get indicator data from Taapi.io API"""
        
        # Check cache first
        cached_result = self.cache.get(indicator, exchange, symbol, interval, **kwargs)
        if cached_result is not None:
            return cached_result
        
        # Prepare parameters
        params = {
            'exchange': exchange,
            'symbol': symbol,
            'interval': interval,
            **kwargs
        }
        
        # Make request in thread pool to avoid blocking async loop
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, self._make_request, indicator, params)
        
        if result is not None:
            # Cache successful result
            self.cache.set(indicator, exchange, symbol, interval, result, **kwargs)
        
        return result
    
    # Advanced Indicators Methods
    
    async def get_ichimoku(self, symbol: str, interval: str = "4h") -> Optional[Dict]:
        """Get Ichimoku Cloud data"""
        return await self.get_indicator("ichimoku", symbol, interval)
    
    async def get_tdsequential(self, symbol: str, interval: str = "1d") -> Optional[Dict]:
        """Get TD Sequential countdown"""
        return await self.get_indicator("tdsequential", symbol, interval)
    
    async def get_supertrend(self, symbol: str, interval: str = "1h", period: int = 10, multiplier: float = 3.0) -> Optional[Dict]:
        """Get Supertrend indicator"""
        return await self.get_indicator("supertrend", symbol, interval, period=period, multiplier=multiplier)
    
    async def get_choppiness_index(self, symbol: str, interval: str = "4h", period: int = 14) -> Optional[Dict]:
        """Get Choppiness Index - measures market trendiness vs choppiness"""
        return await self.get_indicator("chop", symbol, interval, period=period)
    
    async def get_fisher_transform(self, symbol: str, interval: str = "1h", period: int = 9) -> Optional[Dict]:
        """Get Fisher Transform for identifying price reversals"""
        return await self.get_indicator("fisher", symbol, interval, period=period)
    
    async def get_trix(self, symbol: str, interval: str = "4h", period: int = 14) -> Optional[Dict]:
        """Get TRIX indicator for momentum analysis"""
        return await self.get_indicator("trix", symbol, interval, period=period)
    
    async def get_vortex_indicator(self, symbol: str, interval: str = "1h", period: int = 14) -> Optional[Dict]:
        """Get Vortex Indicator for trend detection"""
        return await self.get_indicator("vortex", symbol, interval, period=period)
        
    # Candlestick Pattern Methods
    
    async def get_doji(self, symbol: str, interval: str = "1h") -> Optional[Dict]:
        """Get Doji candlestick pattern"""
        return await self.get_indicator("doji", symbol, interval)
    
    async def get_hammer(self, symbol: str, interval: str = "1h") -> Optional[Dict]:
        """Get Hammer candlestick pattern"""
        return await self.get_indicator("hammer", symbol, interval)
    
    async def get_engulfing(self, symbol: str, interval: str = "1h") -> Optional[Dict]:
        """Get Engulfing candlestick pattern"""
        return await self.get_indicator("engulfing", symbol, interval)
    
    async def get_morning_star(self, symbol: str, interval: str = "1d") -> Optional[Dict]:
        """Get Morning Star pattern (bullish reversal)"""
        return await self.get_indicator("morningstar", symbol, interval)
    
    async def get_evening_star(self, symbol: str, interval: str = "1d") -> Optional[Dict]:
        """Get Evening Star pattern (bearish reversal)"""
        return await self.get_indicator("eveningstar", symbol, interval)
    
    # Volume Indicators
    
    async def get_klinger_oscillator(self, symbol: str, interval: str = "1h") -> Optional[Dict]:
        """Get Klinger Volume Oscillator"""
        return await self.get_indicator("kvo", symbol, interval)
    
    
    
    # Utility Methods
    
    async def get_multiple_indicators(self, symbol: str, interval: str, indicators: List[str]) -> Dict[str, Optional[Dict]]:
        """Get multiple indicators with proper rate limiting"""
        results = {}
        
        for indicator in indicators:
            try:
                result = await self.get_indicator(indicator, symbol, interval)
                results[indicator] = result
                
                # Small delay between requests to be respectful to API
                if len(indicators) > 1:
                    await asyncio.sleep(0.5)
                    
            except Exception as e:
                logging.error(f"Failed to get {indicator} for {symbol}: {str(e)}")
                results[indicator] = None
        
        return results
    
    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics"""
        return {
            'cached_items': len(self.cache.cache),
            'ttl_items': len(self.cache.ttl_cache)
        }

# Dummy client for when Taapi.io is disabled or unavailable
class DummyTaapiClient:
    """Dummy client that returns None for all indicators"""
    
    def __init__(self, *args, **kwargs):
        logging.info("Taapi.io client disabled - using dummy client")
    
    async def get_indicator(self, *args, **kwargs):
        return None
    
    async def get_ichimoku(self, *args, **kwargs):
        return None
    
    async def get_tdsequential(self, *args, **kwargs):
        return None
    
    async def get_supertrend(self, *args, **kwargs):
        return None
    
    async def get_choppiness_index(self, *args, **kwargs):
        return None
    
    async def get_fisher_transform(self, *args, **kwargs):
        return None
    
    async def get_trix(self, *args, **kwargs):
        return None
    
    async def get_vortex_indicator(self, *args, **kwargs):
        return None
    
    async def get_doji(self, *args, **kwargs):
        return None
    
    async def get_hammer(self, *args, **kwargs):
        return None
    
    async def get_engulfing(self, *args, **kwargs):
        return None
    
    async def get_morning_star(self, *args, **kwargs):
        return None
    
    async def get_evening_star(self, *args, **kwargs):
        return None
    
    async def get_klinger_oscillator(self, *args, **kwargs):
        return None
    
    
    
    async def get_multiple_indicators(self, *args, **kwargs):
        return {}
    
    def get_cache_stats(self):
        return {'cached_items': 0, 'ttl_items': 0}