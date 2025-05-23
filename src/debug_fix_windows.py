#!/usr/bin/env python3
"""
Debug and Fix CoinGecko Issues - Windows Compatible
=================================================

This script will debug and fix the configuration and CoinGecko connection issues.
Windows-compatible version without emoji characters.
"""

import os
import sys

def debug_configuration():
    """Debug configuration issues"""
    print("Debugging Configuration...")
    
    try:
        import config
        
        print("Config imported successfully")
        print(f"BINANCE_API_KEY: {'SET' if config.BINANCE_API_KEY else 'NOT SET'}")
        print(f"BINANCE_API_SECRET: {'SET' if config.BINANCE_API_SECRET else 'NOT SET'}")
        print(f"COINGECKO_API_KEY: {'SET' if config.COINGECKO_API_KEY else 'NOT SET'}")
        print(f"ENABLE_COINGECKO: {config.ENABLE_COINGECKO}")
        print(f"TEST_MODE: {config.TEST_MODE}")
        
        # Check the actual values (first 10 chars only for security)
        if config.BINANCE_API_KEY:
            print(f"BINANCE_API_KEY starts with: {config.BINANCE_API_KEY[:10]}...")
        if config.BINANCE_API_SECRET:
            print(f"BINANCE_API_SECRET starts with: {config.BINANCE_API_SECRET[:10]}...")
        if config.COINGECKO_API_KEY:
            print(f"COINGECKO_API_KEY starts with: {config.COINGECKO_API_KEY[:10]}...")
            
        return True
    except Exception as e:
        print(f"Config debug failed: {e}")
        return False

def fix_config_validation():
    """Fix the configuration validation function"""
    print("Fixing configuration validation...")
    
    if not os.path.exists("config.py"):
        print("config.py not found")
        return False
    
    with open("config.py", "r", encoding='utf-8') as f:
        content = f.read()
    
    # Replace the problematic validation function
    old_validation = '''def validate_config():
    """Validate configuration settings"""
    errors = []
    
    # Check required API keys
    if not BINANCE_API_KEY or not BINANCE_API_SECRET:
        errors.append("Binance API key and secret are required")'''
    
    new_validation = '''def validate_config():
    """Validate configuration settings"""
    errors = []
    
    # Check required API keys - Fixed validation
    binance_key_valid = BINANCE_API_KEY and BINANCE_API_KEY.strip() != '' and len(BINANCE_API_KEY.strip()) > 10
    binance_secret_valid = BINANCE_API_SECRET and BINANCE_API_SECRET.strip() != '' and len(BINANCE_API_SECRET.strip()) > 10
    
    if not binance_key_valid or not binance_secret_valid:
        errors.append("Binance API key and secret are required and must be valid")'''
    
    if old_validation in content:
        content = content.replace(old_validation, new_validation)
        
        with open("config.py", "w", encoding='utf-8') as f:
            f.write(content)
        
        print("Configuration validation function updated")
        return True
    else:
        print("Configuration validation function not found or already updated")
        return True

def create_full_coingecko_client():
    """Create the full CoinGecko client implementation"""
    print("Creating full CoinGecko client...")
    
    full_coingecko_content = '''import requests
import logging
import time
import asyncio
import json
from functools import lru_cache
from typing import Dict, Any, Optional
import numpy as np

import config

class CoinGeckoAI:
    """CoinGecko API integration for crypto trading signals and analysis"""
    
    def __init__(self, api_key=None):
        self.api_key = api_key or getattr(config, 'COINGECKO_API_KEY', '')
        self.base_url = "https://api.coingecko.com/api/v3"
        self.pro_url = "https://pro-api.coingecko.com/api/v3"
        self.cache = {}
        self.cache_expiry = {}
        
        # Rate limiting - CoinGecko has different limits for free vs pro
        self.last_request_time = 0
        self.min_request_interval = 1.0 if not self.api_key else 0.2  # Free: 1 req/sec, Pro: 5 req/sec
        
        # Request timeout
        self.request_timeout = 10  # seconds
        
        # Check if API is available
        self.api_available = self._check_api_connection()
        
        # Create coin ID mapping cache
        self.coin_id_cache = {}
        
        if self.api_available:
            logging.info("CoinGecko AI client initialized successfully")
        else:
            logging.warning("CoinGecko AI client initialized but API unavailable - will use fallback mode")
    
    def _check_api_connection(self):
        """Check if the CoinGecko API is available"""
        try:
            print("Testing CoinGecko connection...")
            print(f"Using API key: {'YES' if self.api_key else 'NO (free tier)'}")
            
            # Test with a simple endpoint
            url = f"{self._get_base_url()}/ping"
            headers = self._get_headers()
            
            print(f"Testing URL: {url}")
            
            response = requests.get(url, headers=headers, timeout=5)
            
            print(f"Response status: {response.status_code}")
            
            if response.status_code == 200:
                try:
                    data = response.json()
                    print(f"CoinGecko ping successful: {data}")
                    return True
                except:
                    print("CoinGecko ping successful (no JSON response)")
                    return True
            else:
                print(f"CoinGecko ping failed with status {response.status_code}")
                print(f"Response: {response.text[:200]}")
                return False
                
        except requests.exceptions.Timeout:
            print("CoinGecko connection timeout")
            return False
        except requests.exceptions.ConnectionError:
            print("CoinGecko connection error - check internet connection")
            return False
        except Exception as e:
            print(f"CoinGecko connection check failed: {str(e)}")
            return False
    
    def _get_base_url(self):
        """Get the appropriate base URL (pro or free)"""
        return self.pro_url if self.api_key else self.base_url
    
    def _get_headers(self):
        """Get headers for API requests"""
        headers = {"accept": "application/json"}
        if self.api_key:
            headers["x-cg-pro-api-key"] = self.api_key
        return headers
    
    async def _make_api_request(self, endpoint: str, params: Dict[str, Any] = None) -> Optional[Dict]:
        """Make API request with rate limiting and error handling"""
        if not self.api_available:
            logging.debug(f"Skipping CoinGecko request to {endpoint} - API not available")
            return None
        
        # Ensure minimum time between requests
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.min_request_interval:
            await asyncio.sleep(self.min_request_interval - time_since_last)
        
        try:
            url = f"{self._get_base_url()}/{endpoint}"
            response = requests.get(
                url,
                params=params or {},
                headers=self._get_headers(),
                timeout=self.request_timeout
            )
            
            self.last_request_time = time.time()
            
            if response.status_code == 200:
                data = response.json()
                return data
            elif response.status_code == 429:
                logging.warning("CoinGecko rate limit hit, backing off...")
                await asyncio.sleep(5)
                return None
            else:
                logging.error(f"CoinGecko API error: {response.status_code} - {response.text[:200]}")
                return None
                
        except requests.exceptions.Timeout:
            logging.error(f"CoinGecko API timeout after {self.request_timeout} seconds")
            return None
        except requests.exceptions.RequestException as e:
            logging.error(f"CoinGecko API request error: {str(e)}")
            return None
        except Exception as e:
            logging.error(f"Error making CoinGecko API request: {str(e)}")
            return None
    
    async def _get_coin_id(self, symbol: str) -> str:
        """Get CoinGecko coin ID from symbol"""
        symbol = symbol.upper().replace("USDT", "")
        
        # Check cache first
        if symbol in self.coin_id_cache:
            return self.coin_id_cache[symbol]
        
        # Common mappings
        symbol_to_id = {
            "BTC": "bitcoin",
            "ETH": "ethereum", 
            "BNB": "binancecoin",
            "SOL": "solana",
            "ADA": "cardano",
            "XRP": "ripple",
            "DOGE": "dogecoin",
            "AVAX": "avalanche-2",
            "LINK": "chainlink",
            "DOT": "polkadot",
            "MATIC": "polygon",
            "UNI": "uniswap",
            "LTC": "litecoin"
        }
        
        if symbol in symbol_to_id:
            coin_id = symbol_to_id[symbol]
            self.coin_id_cache[symbol] = coin_id
            return coin_id
        
        # Fallback to lowercase symbol
        coin_id = symbol.lower()
        self.coin_id_cache[symbol] = coin_id
        return coin_id
    
    async def get_coin_data(self, token: str) -> Optional[Dict]:
        """Get comprehensive coin data from CoinGecko"""
        cache_key = f"coin_data_{token}"
        
        # Check cache
        cached_data = self._get_cached_data(cache_key, 300)  # 5 minute cache
        if cached_data:
            return cached_data
        
        try:
            coin_id = await self._get_coin_id(token)
            
            # Get detailed coin data
            params = {
                "localization": "false",
                "tickers": "false",
                "market_data": "true",
                "community_data": "true",
                "developer_data": "false",
                "sparkline": "false"
            }
            
            result = await self._make_api_request(f"coins/{coin_id}", params)
            
            if result:
                self._cache_data(cache_key, result, 300)
                return result
            
        except Exception as e:
            logging.error(f"Error getting coin data for {token}: {str(e)}")
        
        return None
    
    async def get_market_prediction(self, token: str, timeframe: str = "4h") -> Dict:
        """Get market prediction based on CoinGecko metrics"""
        if not self.api_available:
            return {"prediction": {"direction": "neutral", "confidence": 0.5}}
        
        try:
            coin_data = await self.get_coin_data(token)
            if not coin_data:
                return {"prediction": {"direction": "neutral", "confidence": 0.5}}
            
            market_data = coin_data.get("market_data", {})
            
            # Extract key metrics
            price_change_24h = market_data.get("price_change_percentage_24h", 0)
            price_change_7d = market_data.get("price_change_percentage_7d", 0)
            market_cap_rank = market_data.get("market_cap_rank", 999)
            
            # Simple prediction logic
            direction = "neutral"
            confidence = 0.5
            
            if price_change_24h > 5:
                direction = "bullish"
                confidence = 0.7
            elif price_change_24h < -5:
                direction = "bearish"
                confidence = 0.7
            elif price_change_7d > 10:
                direction = "bullish"
                confidence = 0.6
            elif price_change_7d < -10:
                direction = "bearish"
                confidence = 0.6
            
            return {
                "prediction": {
                    "direction": direction,
                    "confidence": confidence,
                    "analysis": f"24h: {price_change_24h:.1f}%, 7d: {price_change_7d:.1f}%"
                }
            }
            
        except Exception as e:
            logging.error(f"Error in CoinGecko market prediction: {str(e)}")
            return {"prediction": {"direction": "neutral", "confidence": 0.5}}
    
    async def get_sentiment_analysis(self, token: str) -> Dict:
        """Get sentiment analysis from CoinGecko community and market data"""
        if not self.api_available:
            return {"sentiment_score": 0}
        
        try:
            coin_data = await self.get_coin_data(token)
            if not coin_data:
                return {"sentiment_score": 0}
            
            market_data = coin_data.get("market_data", {})
            price_change_24h = market_data.get("price_change_percentage_24h", 0)
            
            # Simple sentiment based on price changes
            sentiment_score = np.tanh(price_change_24h / 10)  # Normalize to -1 to 1
            
            return {
                "sentiment_score": sentiment_score,
                "analysis": f"Based on 24h price change: {price_change_24h:.1f}%"
            }
            
        except Exception as e:
            logging.error(f"Error in CoinGecko sentiment analysis: {str(e)}")
            return {"sentiment_score": 0}
    
    async def get_whale_activity(self, token: str) -> Dict:
        """Analyze whale activity using CoinGecko volume and market data"""
        if not self.api_available:
            return {"accumulation_score": 0}
        
        try:
            coin_data = await self.get_coin_data(token)
            if not coin_data:
                return {"accumulation_score": 0}
            
            market_data = coin_data.get("market_data", {})
            volume_24h = market_data.get("total_volume", {}).get("usd", 0)
            market_cap = market_data.get("market_cap", {}).get("usd", 0)
            
            # Simple whale analysis based on volume/market cap ratio
            if market_cap > 0:
                volume_ratio = volume_24h / market_cap
                accumulation_score = min(1, max(-1, (volume_ratio - 0.1) * 10))
            else:
                accumulation_score = 0
            
            return {
                "accumulation_score": accumulation_score,
                "analysis": f"Volume/MCap ratio: {volume_ratio:.3f}" if market_cap > 0 else "No market cap data"
            }
            
        except Exception as e:
            logging.error(f"Error in CoinGecko whale activity analysis: {str(e)}")
            return {"accumulation_score": 0}
    
    async def get_smart_money_positions(self, token: str) -> Dict:
        """Analyze smart money positions using CoinGecko market data"""
        if not self.api_available:
            return {"position": "neutral"}
        
        try:
            coin_data = await self.get_coin_data(token)
            if not coin_data:
                return {"position": "neutral"}
            
            market_data = coin_data.get("market_data", {})
            market_cap_rank = market_data.get("market_cap_rank", 999)
            price_change_7d = market_data.get("price_change_percentage_7d", 0)
            
            # Simple smart money analysis
            position = "neutral"
            if market_cap_rank <= 50 and price_change_7d > 5:
                position = "bullish"
            elif market_cap_rank <= 50 and price_change_7d < -5:
                position = "bearish"
            
            return {
                "position": position,
                "analysis": f"Rank: {market_cap_rank}, 7d: {price_change_7d:.1f}%"
            }
            
        except Exception as e:
            logging.error(f"Error in CoinGecko smart money analysis: {str(e)}")
            return {"position": "neutral"}
    
    async def get_trading_signal(self, token: str) -> Dict:
        """Get consolidated trading signal from CoinGecko data"""
        try:
            insights = await self.get_consolidated_insights(token)
            signal_strength = self.get_signal_strength(insights)
            
            metrics = insights.get("metrics", {})
            prediction_direction = metrics.get("prediction_direction", "neutral")
            
            if prediction_direction == "bullish" and signal_strength > 0.6:
                action = "buy"
            elif prediction_direction == "bearish" and signal_strength > 0.6:
                action = "sell"
            else:
                action = "hold"
            
            return {
                "action": action,
                "strength": signal_strength,
                "insights": metrics
            }
            
        except Exception as e:
            logging.error(f"Error getting trading signal: {str(e)}")
            return {"action": "hold", "strength": 0.5}
    
    async def get_consolidated_insights(self, token: str, max_wait_time: int = 15) -> Dict:
        """Get consolidated insights from all CoinGecko data sources"""
        try:
            # Get all analyses
            prediction = await self.get_market_prediction(token)
            sentiment = await self.get_sentiment_analysis(token)
            whale_activity = await self.get_whale_activity(token)
            smart_money = await self.get_smart_money_positions(token)
            
            # Extract metrics
            metrics = {
                "overall_sentiment": sentiment.get("sentiment_score", 0),
                "prediction_direction": prediction.get("prediction", {}).get("direction", "neutral"),
                "prediction_confidence": prediction.get("prediction", {}).get("confidence", 0.5),
                "whale_accumulation": whale_activity.get("accumulation_score", 0),
                "smart_money_direction": smart_money.get("position", "neutral")
            }
            
            return {
                "market_prediction": prediction,
                "sentiment": sentiment,
                "whale_activity": whale_activity,
                "smart_money": smart_money,
                "timestamp": time.time(),
                "metrics": metrics
            }
            
        except Exception as e:
            logging.error(f"Error in get_consolidated_insights: {str(e)}")
            return self._get_fallback_insights(token)
    
    def get_signal_strength(self, insights: Dict) -> float:
        """Calculate signal strength from CoinGecko insights"""
        try:
            metrics = insights.get('metrics', {})
            if not metrics:
                return 0.5
            
            # Simple signal strength calculation
            prediction_conf = metrics.get('prediction_confidence', 0.5)
            sentiment = abs(metrics.get('overall_sentiment', 0))
            whale_activity = abs(metrics.get('whale_accumulation', 0))
            
            # Combine factors
            strength = (prediction_conf * 0.5 + sentiment * 0.3 + whale_activity * 0.2)
            return min(1.0, max(0.0, strength))
            
        except Exception as e:
            logging.error(f"Error calculating signal strength: {str(e)}")
            return 0.5
    
    def _get_cached_data(self, cache_key: str, max_age: int) -> Optional[Any]:
        """Get data from cache if not expired"""
        current_time = time.time()
        if cache_key in self.cache and self.cache_expiry.get(cache_key, 0) > current_time:
            return self.cache[cache_key]
        return None
    
    def _cache_data(self, cache_key: str, data: Any, ttl: int):
        """Cache data with expiration"""
        self.cache[cache_key] = data
        self.cache_expiry[cache_key] = time.time() + ttl
    
    def _get_fallback_insights(self, token: str) -> Dict:
        """Provide fallback insights data when CoinGecko is unavailable"""
        return {
            "metrics": {
                "overall_sentiment": 0,
                "prediction_direction": "neutral",
                "prediction_confidence": 0.5,
                "whale_accumulation": 0,
                "smart_money_direction": "neutral"
            }
        }


class DummyCoinGeckoAI:
    """Fallback class when CoinGecko API is disabled or unavailable"""
    def __init__(self):
        logging.info("Dummy CoinGecko AI client initialized (fallback mode)")
    
    @property
    def api_available(self):
        return False
        
    async def get_market_prediction(self, *args, **kwargs):
        return {"prediction": {"direction": "neutral", "confidence": 0.5}}
        
    async def get_sentiment_analysis(self, *args, **kwargs):
        return {"sentiment_score": 0}
        
    async def get_whale_activity(self, *args, **kwargs):
        return {"accumulation_score": 0}
        
    async def get_smart_money_positions(self, *args, **kwargs):
        return {"position": "neutral"}
    
    async def get_trading_signal(self, *args, **kwargs):
        return {"action": "hold", "strength": 0.5}
        
    async def get_consolidated_insights(self, *args, **kwargs):
        return {
            "metrics": {
                "overall_sentiment": 0,
                "prediction_direction": "neutral",
                "prediction_confidence": 0.5,
                "whale_accumulation": 0,
                "smart_money_direction": "neutral"
            }
        }
    
    def get_signal_strength(self, *args, **kwargs):
        return 0.5
'''
    
    with open("coingecko_ai.py", "w", encoding='utf-8') as f:
        f.write(full_coingecko_content)
    
    print("Full CoinGecko client created")

def test_fixes():
    """Test if the fixes work"""
    print("Testing fixes...")
    
    try:
        # Reload modules
        if 'config' in sys.modules:
            del sys.modules['config']
        if 'coingecko_ai' in sys.modules:
            del sys.modules['coingecko_ai']
        
        import config
        from coingecko_ai import CoinGeckoAI
        
        print("Modules imported successfully")
        
        # Test config validation
        try:
            config.validate_config()
            print("Configuration validation passed")
        except Exception as e:
            print(f"Configuration validation failed: {e}")
        
        # Test CoinGecko connection
        client = CoinGeckoAI(config.COINGECKO_API_KEY)
        print(f"CoinGecko client created, API available: {client.api_available}")
        
        return True
    except Exception as e:
        print(f"Test failed: {e}")
        return False

def main():
    """Main debug and fix function"""
    print("CoinGecko Debug and Fix")
    print("=" * 40)
    
    # Step 1: Debug configuration
    debug_configuration()
    
    # Step 2: Fix configuration validation
    fix_config_validation()
    
    # Step 3: Create full CoinGecko client
    create_full_coingecko_client()
    
    # Step 4: Test fixes
    test_fixes()
    
    print("\\n" + "=" * 40)
    print("Debug and fix completed!")
    print("Next steps:")
    print("   1. Run: python test_coingecko.py")
    print("   2. If tests pass, run: python main.py")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Debug script failed: {str(e)}")
        import traceback
        traceback.print_exc()