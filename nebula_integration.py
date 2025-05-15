import requests
import logging
import time
import asyncio
from web3 import Web3
import json
from functools import lru_cache

import config

class NebulaAI:
    def __init__(self, api_key=None):
        self.api_key = api_key or config.THIRDWEB_API_KEY
        self.nebula_endpoint = "https://api.thirdweb.com/nebula"
        self.web3 = Web3(Web3.HTTPProvider(config.WEB3_PROVIDER_URL))
        self.cache = {}
        self.cache_expiry = {}
        
        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 0.5  # seconds
        
        logging.info("Thirdweb Nebula AI client initialized")
    
    async def _make_api_request(self, endpoint, data):
        """Make API request with rate limiting"""
        # Ensure minimum time between requests
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.min_request_interval:
            await asyncio.sleep(self.min_request_interval - time_since_last)
        
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            response = requests.post(
                f"{self.nebula_endpoint}/{endpoint}", 
                json=data,
                headers=headers
            )
            
            self.last_request_time = time.time()
            
            if response.status_code == 200:
                return response.json()
            else:
                logging.error(f"Nebula API error: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            logging.error(f"Error making Nebula API request: {str(e)}")
            return None
    
    @lru_cache(maxsize=50)
    async def get_cached_data(self, cache_key, expiry_seconds, fetch_func, *args):
        """Get data with caching to reduce API calls"""
        current_time = time.time()
        
        # Check if we have cached data
        if cache_key in self.cache and self.cache_expiry.get(cache_key, 0) > current_time:
            return self.cache[cache_key]
            
        # Fetch new data
        result = await fetch_func(*args)
        
        # Cache the result
        if result:
            self.cache[cache_key] = result
            self.cache_expiry[cache_key] = current_time + expiry_seconds
            
        return result
    
    async def get_market_prediction(self, token, timeframe="4h"):
        """Get AI-powered market prediction from Nebula"""
        # Clean token symbol
        token = token.replace("USDT", "").lower()
        
        # Create cache key
        cache_key = f"market_prediction_{token}_{timeframe}"
        
        # Define data fetching function
        async def fetch_prediction():
            request_data = {
                "model": config.NEBULA_MODELS["market_prediction"],
                "inputs": {
                    "token": token,
                    "timeframe": timeframe
                }
            }
            return await self._make_api_request("predict", request_data)
        
        # Get data with caching (30 minute expiry)
        return await self.get_cached_data(cache_key, 1800, fetch_prediction)
    
    async def get_sentiment_analysis(self, token):
        """Get on-chain sentiment analysis"""
        token = token.replace("USDT", "").lower()
        cache_key = f"sentiment_{token}"
        
        async def fetch_sentiment():
            request_data = {
                "model": config.NEBULA_MODELS["sentiment"],
                "inputs": {
                    "token": token,
                    "timeframe": "24h"
                }
            }
            return await self._make_api_request("analyze", request_data)
        
        # Get data with caching (1 hour expiry)
        return await self.get_cached_data(cache_key, 3600, fetch_sentiment)
    
    async def get_whale_activity(self, token):
        """Track whale wallet activity for the token"""
        token = token.replace("USDT", "").lower()
        cache_key = f"whale_{token}"
        
        async def fetch_whale_data():
            request_data = {
                "model": config.NEBULA_MODELS["whale_tracking"],
                "inputs": {
                    "token": token,
                    "lookback_hours": 24
                }
            }
            return await self._make_api_request("analyze", request_data)
        
        # Get data with caching (15 minute expiry)
        return await self.get_cached_data(cache_key, 900, fetch_whale_data)
    
    async def get_smart_money_positions(self, token):
        """Analyze positions of 'smart money' wallets"""
        token = token.replace("USDT", "").lower()
        cache_key = f"smart_money_{token}"
        
        async def fetch_smart_money():
            request_data = {
                "model": config.NEBULA_MODELS["smart_money"],
                "inputs": {
                    "token": token
                }
            }
            return await self._make_api_request("analyze", request_data)
        
        # Get data with caching (30 minute expiry)
        return await self.get_cached_data(cache_key, 1800, fetch_smart_money)
    
    async def get_consolidated_insights(self, token):
        """Get consolidated insights from all Nebula models"""
        token = token.replace("USDT", "").lower()
        
        # Gather data from all models in parallel
        tasks = [
            self.get_market_prediction(token),
            self.get_sentiment_analysis(token),
            self.get_whale_activity(token),
            self.get_smart_money_positions(token)
        ]
        
        results = await asyncio.gather(*tasks)
        
        # Consolidate results
        consolidated = {
            "market_prediction": results[0],
            "sentiment": results[1],
            "whale_activity": results[2],
            "smart_money": results[3],
            "timestamp": time.time()
        }
        
        # Extract key metrics for easier access
        metrics = {
            "overall_sentiment": self._extract_sentiment(results[1]),
            "prediction_direction": self._extract_prediction(results[0]),
            "prediction_confidence": self._extract_confidence(results[0]),
            "whale_accumulation": self._extract_whale_trend(results[2]),
            "smart_money_direction": self._extract_smart_money(results[3])
        }
        
        consolidated["metrics"] = metrics
        
        return consolidated
    
    def _extract_sentiment(self, sentiment_data):
        """Extract sentiment score from sentiment data"""
        if not sentiment_data:
            return 0
        return sentiment_data.get("sentiment_score", 0)
    
    def _extract_prediction(self, prediction_data):
        """Extract prediction direction from prediction data"""
        if not prediction_data:
            return "neutral"
        return prediction_data.get("prediction", {}).get("direction", "neutral")
    
    def _extract_confidence(self, prediction_data):
        """Extract confidence from prediction data"""
        if not prediction_data:
            return 0.5
        return prediction_data.get("prediction", {}).get("confidence", 0.5)
    
    def _extract_whale_trend(self, whale_data):
        """Extract whale trend (accumulation/distribution)"""
        if not whale_data:
            return 0
        return whale_data.get("accumulation_score", 0)
    
    def _extract_smart_money(self, smart_money_data):
        """Extract smart money direction"""
        if not smart_money_data:
            return "neutral"
        return smart_money_data.get("position", "neutral")