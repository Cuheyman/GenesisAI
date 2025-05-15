import requests
import logging
import time
import asyncio
import json
import re
from functools import lru_cache

import config

class NebulaAI:
    def __init__(self, api_key=None):
        self.api_key = api_key or config.THIRDWEB_API_KEY
        
        # Use local proxy or fallback to direct endpoint (which likely won't work)
        self.use_proxy = getattr(config, 'USE_NEBULA_PROXY', True)
        self.proxy_url = getattr(config, 'NEBULA_PROXY_URL', "http://localhost:3000")
        self.direct_url = "https://nebula-api.thirdweb.com"
        
        self.nebula_endpoint = f"{self.proxy_url}/nebula" if self.use_proxy else self.direct_url
        self.cache = {}
        self.cache_expiry = {}
        
        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 0.5  # seconds
        
        # Session ID for conversation continuity
        self.session_id = None
        
        # Request timeout (critical to prevent hanging)
        self.request_timeout = 10  # seconds
        
        # Check if proxy is available if we're using it
        self.proxy_available = self._check_proxy_connection() if self.use_proxy else False
        
        if self.use_proxy and self.proxy_available:
            logging.info("Thirdweb Nebula AI client initialized via local proxy")
        elif self.use_proxy and not self.proxy_available:
            logging.warning("Thirdweb Nebula AI proxy not available - will use fallback mode")
        else:
            logging.info("Thirdweb Nebula AI client initialized (direct mode)")
    
    def _check_proxy_connection(self):
        """Check if the Nebula proxy is available"""
        try:
            response = requests.get(f"{self.proxy_url}/health", timeout=2)
            return response.status_code == 200
        except Exception as e:
            logging.debug(f"Proxy connection check failed: {str(e)}")
            return False
    
    async def _make_api_request(self, endpoint, data):
        """Make API request with rate limiting and timeout protection"""
        # Skip if we're using proxy and it's not available
        if self.use_proxy and not self.proxy_available:
            logging.debug(f"Skipping Nebula request to {endpoint} - proxy not available")
            return None
        
        # Ensure minimum time between requests
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.min_request_interval:
            await asyncio.sleep(self.min_request_interval - time_since_last)
        
        try:
            # Headers differ between proxy and direct mode
            if self.use_proxy:
                headers = {
                    "Content-Type": "application/json"
                }
            else:
                headers = {
                    "x-secret-key": self.api_key,
                    "Content-Type": "application/json"
                }
            
            response = requests.post(
                f"{self.nebula_endpoint}/{endpoint}", 
                json=data,
                headers=headers,
                timeout=self.request_timeout
            )
            
            self.last_request_time = time.time()
            
            if response.status_code == 200:
                return response.json()
            else:
                logging.error(f"Nebula API error: {response.status_code} - {response.text[:200]}")
                return None
                
        except requests.exceptions.Timeout:
            logging.error(f"Nebula API timeout after {self.request_timeout} seconds")
            return None
        except requests.exceptions.RequestException as e:
            logging.error(f"Nebula API request error: {str(e)}")
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
    
    async def send_chat_message(self, message, user_id="default-user"):
        """Send a message to Nebula chat API"""
        # Skip if proxy is unavailable when in proxy mode
        if self.use_proxy and not self.proxy_available:
            return None
            
        request_data = {
            "message": message,
            "user_id": user_id,
            "stream": False
        }
        
        return await self._make_api_request("chat", request_data)
    
    async def get_market_prediction(self, token, timeframe="4h"):
        """Get AI-powered market prediction from Nebula"""
        # Skip if proxy is unavailable when in proxy mode
        if self.use_proxy and not self.proxy_available:
            return {"prediction": {"direction": "neutral", "confidence": 0.5}}
            
        # Clean token symbol
        token = token.replace("USDT", "").lower()
        
        # Create cache key
        cache_key = f"market_prediction_{token}_{timeframe}"
        
        # Define data fetching function
        async def fetch_prediction():
            if self.use_proxy:
                # Use proxy format
                request_data = {
                    "inputs": {
                        "token": token,
                        "timeframe": timeframe
                    }
                }
                return await self._make_api_request("predict", request_data)
            else:
                # Use direct chat format
                message = f"Analyze price prediction for {token} in {timeframe} timeframe. Format the response as JSON with prediction direction and confidence."
                response = await self.send_chat_message(message)
                
                # Try to extract structured information from response
                try:
                    if response and 'content' in response:
                        content = response['content']
                        # Try to find and parse JSON in the response
                        json_match = re.search(r'\{.*\}', content, re.DOTALL)
                        if json_match:
                            prediction_data = json.loads(json_match.group(0))
                            return {
                                "prediction": prediction_data
                            }
                except Exception as e:
                    logging.error(f"Error parsing prediction response: {str(e)}")
                
                # Fallback to returning raw response or default values
                if response:
                    return response
                else:
                    return {"prediction": {"direction": "neutral", "confidence": 0.5}}
        
        # Get data with caching (30 minute expiry)
        result = await self.get_cached_data(cache_key, 1800, fetch_prediction)
        
        # Ensure we always return something valid
        if not result:
            return {"prediction": {"direction": "neutral", "confidence": 0.5}}
        return result
    
    async def get_sentiment_analysis(self, token):
        """Get on-chain sentiment analysis"""
        # Skip if proxy is unavailable when in proxy mode
        if self.use_proxy and not self.proxy_available:
            return {"sentiment_score": 0}
            
        token = token.replace("USDT", "").lower()
        cache_key = f"sentiment_{token}"
        
        async def fetch_sentiment():
            if self.use_proxy:
                # Use proxy format
                request_data = {
                    "inputs": {
                        "token": token,
                        "timeframe": "24h"
                    }
                }
                return await self._make_api_request("analyze", request_data)
            else:
                # Use direct chat format
                message = f"Analyze on-chain sentiment for {token}. Format the response as JSON with sentiment_score from -1 to 1."
                response = await self.send_chat_message(message)
                
                # Try to extract structured information from response
                try:
                    if response and 'content' in response:
                        content = response['content']
                        # Try to find and parse JSON
                        json_match = re.search(r'\{.*\}', content, re.DOTALL)
                        if json_match:
                            sentiment_data = json.loads(json_match.group(0))
                            return sentiment_data
                except Exception as e:
                    logging.error(f"Error parsing sentiment response: {str(e)}")
                
                # Fallback
                return {"sentiment_score": 0}
        
        # Get data with caching (1 hour expiry)
        result = await self.get_cached_data(cache_key, 3600, fetch_sentiment)
        
        # Ensure we always return something valid
        if not result:
            return {"sentiment_score": 0}
        return result
    
    async def get_whale_activity(self, token):
        """Track whale wallet activity for the token"""
        # Skip if proxy is unavailable when in proxy mode
        if self.use_proxy and not self.proxy_available:
            return {"accumulation_score": 0}
            
        token = token.replace("USDT", "").lower()
        cache_key = f"whale_{token}"
        
        async def fetch_whale_data():
            if self.use_proxy:
                # Use proxy format
                request_data = {
                    "inputs": {
                        "token": token,
                        "lookback_hours": 24
                    }
                }
                return await self._make_api_request("whale-tracking", request_data)
            else:
                # Use direct chat format
                message = f"Analyze whale activity for {token} in the last 24 hours. Format the response as JSON with accumulation_score from -1 to 1."
                response = await self.send_chat_message(message)
                
                # Process response
                try:
                    if response and 'content' in response:
                        content = response['content']
                        json_match = re.search(r'\{.*\}', content, re.DOTALL)
                        if json_match:
                            whale_data = json.loads(json_match.group(0))
                            return whale_data
                except Exception as e:
                    logging.error(f"Error parsing whale activity response: {str(e)}")
                
                # Fallback
                return {"accumulation_score": 0}
        
        # Get data with caching (15 minute expiry)
        result = await self.get_cached_data(cache_key, 900, fetch_whale_data)
        
        # Ensure we always return something valid
        if not result:
            return {"accumulation_score": 0}
        return result
    
    async def get_smart_money_positions(self, token):
        """Analyze positions of 'smart money' wallets"""
        # Skip if proxy is unavailable when in proxy mode
        if self.use_proxy and not self.proxy_available:
            return {"position": "neutral"}
            
        token = token.replace("USDT", "").lower()
        cache_key = f"smart_money_{token}"
        
        async def fetch_smart_money():
            if self.use_proxy:
                # Use proxy format
                request_data = {
                    "inputs": {
                        "token": token
                    }
                }
                return await self._make_api_request("smart-money", request_data)
            else:
                # Use direct chat format
                message = f"Analyze smart money positions for {token}. Format the response as JSON with position as either bullish, bearish, or neutral."
                response = await self.send_chat_message(message)
                
                # Process response
                try:
                    if response and 'content' in response:
                        content = response['content']
                        json_match = re.search(r'\{.*\}', content, re.DOTALL)
                        if json_match:
                            smart_money_data = json.loads(json_match.group(0))
                            return smart_money_data
                except Exception as e:
                    logging.error(f"Error parsing smart money response: {str(e)}")
                
                # Fallback
                return {"position": "neutral"}
        
        # Get data with caching (30 minute expiry)
        result = await self.get_cached_data(cache_key, 1800, fetch_smart_money)
        
        # Ensure we always return something valid
        if not result:
            return {"position": "neutral"}
        return result
    
    async def get_consolidated_insights(self, token, max_wait_time=15):
        """Get consolidated insights from all Nebula models with timeout protection"""
        token = token.replace("USDT", "").lower()
        
        # If proxy is unavailable when in proxy mode, use fallback
        if self.use_proxy and not self.proxy_available:
            return self._get_fallback_insights(token)
        
        try:
            # Use a timeout to prevent hanging
            # Gather data from all models in parallel with timeout protection
            tasks = [
                asyncio.create_task(self.get_market_prediction(token)),
                asyncio.create_task(self.get_sentiment_analysis(token)),
                asyncio.create_task(self.get_whale_activity(token)),
                asyncio.create_task(self.get_smart_money_positions(token))
            ]
            
            # Wait for all tasks to complete with timeout
            done, pending = await asyncio.wait(tasks, timeout=max_wait_time)
            
            # Cancel any pending tasks that didn't complete within the timeout
            for task in pending:
                task.cancel()
                
            # Get results from completed tasks
            results = []
            for i, task in enumerate(tasks):
                try:
                    if task in done:
                        results.append(task.result())
                    else:
                        # Use default values for tasks that didn't complete
                        if i == 0:  # market prediction
                            results.append({"prediction": {"direction": "neutral", "confidence": 0.5}})
                        elif i == 1:  # sentiment
                            results.append({"sentiment_score": 0})
                        elif i == 2:  # whale activity
                            results.append({"accumulation_score": 0})
                        else:  # smart money
                            results.append({"position": "neutral"})
                except Exception as e:
                    logging.error(f"Error getting task result: {str(e)}")
                    # Use appropriate default value based on task index
                    if i == 0:  # market prediction
                        results.append({"prediction": {"direction": "neutral", "confidence": 0.5}})
                    elif i == 1:  # sentiment
                        results.append({"sentiment_score": 0})
                    elif i == 2:  # whale activity
                        results.append({"accumulation_score": 0})
                    else:  # smart money
                        results.append({"position": "neutral"})
            
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
            
        except Exception as e:
            logging.error(f"Error in get_consolidated_insights: {str(e)}")
            # Return default data in case of any error
            return self._get_fallback_insights(token)
    
    def _get_fallback_insights(self, token):
        """Provide fallback insights data when Nebula is unavailable"""
        return {
            "market_prediction": {},
            "sentiment": {},
            "whale_activity": {},
            "smart_money": {},
            "timestamp": time.time(),
            "metrics": {
                "overall_sentiment": 0,
                "prediction_direction": "neutral",
                "prediction_confidence": 0.5,
                "whale_accumulation": 0,
                "smart_money_direction": "neutral"
            }
        }
    
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

class DummyNebulaAI:
    """Fallback class when Nebula API is disabled or unavailable"""
    def __init__(self):
        logging.info("Dummy Nebula AI client initialized (fallback mode)")
        
    async def get_market_prediction(self, *args, **kwargs):
        return {"prediction": {"direction": "neutral", "confidence": 0.5}}
        
    async def get_sentiment_analysis(self, *args, **kwargs):
        return {"sentiment_score": 0}
        
    async def get_whale_activity(self, *args, **kwargs):
        return {"accumulation_score": 0}
        
    async def get_smart_money_positions(self, *args, **kwargs):
        return {"position": "neutral"}
        
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