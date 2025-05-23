import requests
import logging
import time
import asyncio
import json
from functools import lru_cache
from typing import Dict, Any, Optional

import config

class LunarCrushAI:
    """LunarCrush API integration for crypto trading signals and analysis"""
    
    def __init__(self, api_key=None):
        self.api_key = api_key or getattr(config, 'LUNARCRUSH_API_KEY', '')
        self.base_url = "https://lunarcrush.com/api3"
        self.cache = {}
        self.cache_expiry = {}
        
        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 0.5  # seconds
        
        # Request timeout
        self.request_timeout = 10  # seconds
        
        # Check if API is available
        self.api_available = self._check_api_connection()
        
        if self.api_available:
            logging.info("LunarCrush AI client initialized successfully")
        else:
            logging.warning("LunarCrush AI client initialized but API unavailable - will use fallback mode")
    
    def _check_api_connection(self):
        """Check if the LunarCrush API is available"""
        try:
            # Test with a simple endpoint
            response = requests.get(
                f"{self.base_url}/coins/list",
                params={"key": self.api_key},
                timeout=5
            )
            return response.status_code == 200
        except Exception as e:
            logging.debug(f"LunarCrush connection check failed: {str(e)}")
            return False
    
    async def _make_api_request(self, endpoint: str, params: Dict[str, Any]) -> Optional[Dict]:
        """Make API request with rate limiting and error handling"""
        if not self.api_available:
            logging.debug(f"Skipping LunarCrush request to {endpoint} - API not available")
            return None
        
        # Ensure minimum time between requests
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.min_request_interval:
            await asyncio.sleep(self.min_request_interval - time_since_last)
        
        try:
            # Add API key to params
            params["key"] = self.api_key
            
            response = requests.get(
                f"{self.base_url}/{endpoint}",
                params=params,
                timeout=self.request_timeout
            )
            
            self.last_request_time = time.time()
            
            if response.status_code == 200:
                data = response.json()
                if data.get("data"):
                    return data
                else:
                    logging.error(f"LunarCrush API returned no data for {endpoint}")
                    return None
            else:
                logging.error(f"LunarCrush API error: {response.status_code} - {response.text[:200]}")
                return None
                
        except requests.exceptions.Timeout:
            logging.error(f"LunarCrush API timeout after {self.request_timeout} seconds")
            return None
        except requests.exceptions.RequestException as e:
            logging.error(f"LunarCrush API request error: {str(e)}")
            return None
        except Exception as e:
            logging.error(f"Error making LunarCrush API request: {str(e)}")
            return None
    
    async def get_asset_data(self, token: str) -> Optional[Dict]:
        """Get comprehensive asset data from LunarCrush"""
        cache_key = f"asset_data_{token}"
        
        # Check cache
        cached_data = self._get_cached_data(cache_key, 300)  # 5 minute cache
        if cached_data:
            return cached_data
        
        # Make API request
        params = {
            "symbol": token.upper(),
            "data_points": 1,
            "interval": "hour"
        }
        
        result = await self._make_api_request("assets", params)
        
        if result and result.get("data"):
            asset_data = result["data"][0] if isinstance(result["data"], list) else result["data"]
            self._cache_data(cache_key, asset_data, 300)
            return asset_data
        
        return None
    
    async def get_market_prediction(self, token: str, timeframe: str = "4h") -> Dict:
        """Get market prediction based on LunarCrush metrics"""
        if not self.api_available:
            return {"prediction": {"direction": "neutral", "confidence": 0.5}}
        
        try:
            # Get asset data
            asset_data = await self.get_asset_data(token)
            if not asset_data:
                return {"prediction": {"direction": "neutral", "confidence": 0.5}}
            
            # Analyze key metrics
            galaxy_score = asset_data.get("galaxy_score", 50) / 100  # Normalize to 0-1
            alt_rank = asset_data.get("alt_rank", 100)
            price_score = asset_data.get("price_score", 50) / 100
            social_score = asset_data.get("social_score", 50) / 100
            
            # Calculate direction based on multiple factors
            bull_signals = 0
            bear_signals = 0
            
            # Galaxy score analysis
            if galaxy_score > 0.7:
                bull_signals += 2
            elif galaxy_score > 0.5:
                bull_signals += 1
            elif galaxy_score < 0.3:
                bear_signals += 2
            elif galaxy_score < 0.5:
                bear_signals += 1
            
            # Alt rank analysis (lower is better)
            if alt_rank <= 20:
                bull_signals += 2
            elif alt_rank <= 50:
                bull_signals += 1
            elif alt_rank >= 100:
                bear_signals += 1
            
            # Price score analysis
            if price_score > 0.7:
                bull_signals += 1
            elif price_score < 0.3:
                bear_signals += 1
            
            # Social score analysis
            if social_score > 0.7:
                bull_signals += 1
            elif social_score < 0.3:
                bear_signals += 1
            
            # Determine direction and confidence
            if bull_signals > bear_signals + 2:
                direction = "bullish"
                confidence = min(0.9, 0.5 + (bull_signals - bear_signals) * 0.1)
            elif bear_signals > bull_signals + 2:
                direction = "bearish"
                confidence = min(0.9, 0.5 + (bear_signals - bull_signals) * 0.1)
            else:
                direction = "neutral"
                confidence = 0.5
            
            return {
                "prediction": {
                    "direction": direction,
                    "confidence": confidence,
                    "analysis": f"Galaxy Score: {galaxy_score:.2f}, Alt Rank: {alt_rank}, Price Score: {price_score:.2f}"
                }
            }
            
        except Exception as e:
            logging.error(f"Error in LunarCrush market prediction: {str(e)}")
            return {"prediction": {"direction": "neutral", "confidence": 0.5}}
    
    async def get_sentiment_analysis(self, token: str) -> Dict:
        """Get sentiment analysis from LunarCrush social data"""
        if not self.api_available:
            return {"sentiment_score": 0}
        
        try:
            # Get asset data
            asset_data = await self.get_asset_data(token)
            if not asset_data:
                return {"sentiment_score": 0}
            
            # Extract sentiment metrics
            sentiment_relative = asset_data.get("sentiment_relative", 50) / 100  # Normalize to 0-1
            sentiment_absolute = asset_data.get("sentiment_absolute", 50) / 100
            social_score = asset_data.get("social_score", 50) / 100
            correlation_rank = asset_data.get("correlation_rank", 0.5)
            
            # Calculate weighted sentiment score (-1 to 1)
            # Convert 0-1 scores to -1 to 1 range
            sentiment_rel_normalized = (sentiment_relative - 0.5) * 2
            sentiment_abs_normalized = (sentiment_absolute - 0.5) * 2
            social_normalized = (social_score - 0.5) * 2
            
            # Weight the components
            sentiment_score = (
                sentiment_rel_normalized * 0.4 +
                sentiment_abs_normalized * 0.3 +
                social_normalized * 0.3
            )
            
            # Adjust based on correlation rank (higher correlation is better)
            if correlation_rank > 0.7:
                sentiment_score *= 1.2
            elif correlation_rank < 0.3:
                sentiment_score *= 0.8
            
            # Clamp to -1 to 1 range
            sentiment_score = max(-1, min(1, sentiment_score))
            
            return {
                "sentiment_score": sentiment_score,
                "analysis": f"Relative: {sentiment_relative:.2f}, Absolute: {sentiment_absolute:.2f}, Social: {social_score:.2f}"
            }
            
        except Exception as e:
            logging.error(f"Error in LunarCrush sentiment analysis: {str(e)}")
            return {"sentiment_score": 0}
    
    async def get_whale_activity(self, token: str) -> Dict:
        """Analyze whale activity using LunarCrush volume and market data"""
        if not self.api_available:
            return {"accumulation_score": 0}
        
        try:
            # Get asset data
            asset_data = await self.get_asset_data(token)
            if not asset_data:
                return {"accumulation_score": 0}
            
            # Get volume and market metrics
            volume_24h = asset_data.get("volume_24h", 0)
            volume_24h_rank = asset_data.get("volume_24h_rank", 100)
            market_cap_rank = asset_data.get("market_cap_rank", 100)
            percent_change_24h = asset_data.get("percent_change_24h", 0)
            
            # Get average volume (if available)
            avg_volume = asset_data.get("average_volume_24h", volume_24h)
            
            # Calculate volume ratio
            volume_ratio = volume_24h / avg_volume if avg_volume > 0 else 1
            
            # Initialize accumulation score
            accumulation_score = 0
            
            # High volume with positive price movement suggests accumulation
            if volume_ratio > 1.5 and percent_change_24h > 0:
                accumulation_score += 0.4
            elif volume_ratio > 1.2 and percent_change_24h > 0:
                accumulation_score += 0.2
            
            # High volume with negative price movement suggests distribution
            elif volume_ratio > 1.5 and percent_change_24h < 0:
                accumulation_score -= 0.4
            elif volume_ratio > 1.2 and percent_change_24h < 0:
                accumulation_score -= 0.2
            
            # Volume rank analysis (lower rank = higher volume)
            if volume_24h_rank <= 20:
                accumulation_score += 0.2
            elif volume_24h_rank <= 50:
                accumulation_score += 0.1
            elif volume_24h_rank >= 100:
                accumulation_score -= 0.1
            
            # Market cap rank consideration
            if market_cap_rank <= 50 and volume_ratio > 1.3:
                accumulation_score += 0.1  # Large cap with high volume
            
            # Clamp to -1 to 1 range
            accumulation_score = max(-1, min(1, accumulation_score))
            
            return {
                "accumulation_score": accumulation_score,
                "analysis": f"Volume Ratio: {volume_ratio:.2f}, Price Change: {percent_change_24h:.2f}%"
            }
            
        except Exception as e:
            logging.error(f"Error in LunarCrush whale activity analysis: {str(e)}")
            return {"accumulation_score": 0}
    
    async def get_smart_money_positions(self, token: str) -> Dict:
        """Analyze smart money positions using LunarCrush correlation and market data"""
        if not self.api_available:
            return {"position": "neutral"}
        
        try:
            # Get asset data
            asset_data = await self.get_asset_data(token)
            if not asset_data:
                return {"position": "neutral"}
            
            # Get relevant metrics
            correlation_rank = asset_data.get("correlation_rank", 0.5)
            galaxy_score = asset_data.get("galaxy_score", 50) / 100
            market_dominance = asset_data.get("market_dominance", 0)
            percent_change_7d = asset_data.get("percent_change_7d", 0)
            
            # Analyze smart money indicators
            position_score = 0
            
            # High correlation rank suggests smart money interest
            if correlation_rank > 0.8:
                position_score += 0.4
            elif correlation_rank > 0.6:
                position_score += 0.2
            elif correlation_rank < 0.2:
                position_score -= 0.3
            
            # Galaxy score analysis
            if galaxy_score > 0.7:
                position_score += 0.3
            elif galaxy_score < 0.3:
                position_score -= 0.3
            
            # Market dominance trends
            if market_dominance > 0.1 and percent_change_7d > 5:
                position_score += 0.2
            elif market_dominance < 0.05 and percent_change_7d < -5:
                position_score -= 0.2
            
            # Determine position
            if position_score > 0.3:
                position = "bullish"
            elif position_score < -0.3:
                position = "bearish"
            else:
                position = "neutral"
            
            return {
                "position": position,
                "analysis": f"Correlation: {correlation_rank:.2f}, Galaxy: {galaxy_score:.2f}, Score: {position_score:.2f}"
            }
            
        except Exception as e:
            logging.error(f"Error in LunarCrush smart money analysis: {str(e)}")
            return {"position": "neutral"}
    
    async def get_trading_signal(self, token: str) -> Dict:
        """Get consolidated trading signal from LunarCrush data"""
        try:
            # Get all insights
            insights = await self.get_consolidated_insights(token)
            
            # Extract metrics
            metrics = insights.get("metrics", {})
            
            # Calculate signal strength
            signal_strength = self.get_signal_strength(insights)
            
            # Determine action based on metrics
            if metrics.get("prediction_direction") == "bullish" and signal_strength > 0.6:
                action = "buy"
            elif metrics.get("prediction_direction") == "bearish" and signal_strength > 0.6:
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
        """Get consolidated insights from all LunarCrush data sources"""
        token = token.replace("USDT", "").upper()
        
        if not self.api_available:
            logging.warning(f"LunarCrush API unavailable for {token} analysis - using fallback data")
            return self._get_fallback_insights(token)
        
        try:
            # Gather data from all analyses in parallel
            tasks = [
                asyncio.create_task(self.get_market_prediction(token)),
                asyncio.create_task(self.get_sentiment_analysis(token)),
                asyncio.create_task(self.get_whale_activity(token)),
                asyncio.create_task(self.get_smart_money_positions(token))
            ]
            
            # Wait for all tasks with timeout
            done, pending = await asyncio.wait(tasks, timeout=max_wait_time)
            
            # Cancel any pending tasks
            for task in pending:
                task.cancel()
            
            # Get results
            results = []
            for task in tasks:
                try:
                    if task in done:
                        results.append(task.result())
                    else:
                        # Use default values for timed out tasks
                        if len(results) == 0:  # prediction
                            results.append({"prediction": {"direction": "neutral", "confidence": 0.5}})
                        elif len(results) == 1:  # sentiment
                            results.append({"sentiment_score": 0})
                        elif len(results) == 2:  # whale activity
                            results.append({"accumulation_score": 0})
                        else:  # smart money
                            results.append({"position": "neutral"})
                except Exception as e:
                    logging.error(f"Error getting LunarCrush data: {str(e)}")
                    # Add default values
                    if len(results) == 0:
                        results.append({"prediction": {"direction": "neutral", "confidence": 0.5}})
                    elif len(results) == 1:
                        results.append({"sentiment_score": 0})
                    elif len(results) == 2:
                        results.append({"accumulation_score": 0})
                    else:
                        results.append({"position": "neutral"})
            
            # Consolidate results
            market_prediction, sentiment, whale_activity, smart_money = results
            
            consolidated = {
                "market_prediction": market_prediction,
                "sentiment": sentiment,
                "whale_activity": whale_activity,
                "smart_money": smart_money,
                "timestamp": time.time()
            }
            
            # Extract key metrics
            metrics = {
                "overall_sentiment": sentiment.get("sentiment_score", 0),
                "prediction_direction": market_prediction.get("prediction", {}).get("direction", "neutral"),
                "prediction_confidence": market_prediction.get("prediction", {}).get("confidence", 0.5),
                "whale_accumulation": whale_activity.get("accumulation_score", 0),
                "smart_money_direction": smart_money.get("position", "neutral")
            }
            
            consolidated["metrics"] = metrics
            
            # Log successful data retrieval
            logging.info(f"LunarCrush insights for {token}: {metrics['prediction_direction']} "
                        f"({metrics['prediction_confidence']:.2f} confidence), "
                        f"sentiment: {metrics['overall_sentiment']:.2f}, "
                        f"whale: {metrics['whale_accumulation']:.2f}, "
                        f"smart money: {metrics['smart_money_direction']}")
            
            return consolidated
            
        except Exception as e:
            logging.error(f"Error in get_consolidated_insights: {str(e)}")
            return self._get_fallback_insights(token)
    
    def get_signal_strength(self, insights: Dict) -> float:
        """Calculate signal strength from LunarCrush insights"""
        if not insights or not isinstance(insights, dict):
            return 0.5
        
        try:
            metrics = insights.get('metrics', {})
            if not metrics:
                return 0.5
            
            # Get individual metrics
            prediction_conf = metrics.get('prediction_confidence', 0.5)
            prediction_dir = metrics.get('prediction_direction', 'neutral')
            sentiment = metrics.get('overall_sentiment', 0)
            whale_acc = metrics.get('whale_accumulation', 0)
            smart_money = metrics.get('smart_money_direction', 'neutral')
            
            # Get weights from config
            weights = getattr(config, 'LUNARCRUSH_WEIGHTS', {
                'PREDICTION': 0.3,
                'SENTIMENT': 0.25,
                'WHALE': 0.25,
                'SMART_MONEY': 0.2
            })
            
            # Calculate direction multipliers (-1 to 1)
            pred_mult = 1 if prediction_dir == 'bullish' else (-1 if prediction_dir == 'bearish' else 0)
            sm_mult = 1 if smart_money == 'bullish' else (-1 if smart_money == 'bearish' else 0)
            
            # Calculate weighted components
            prediction_comp = pred_mult * prediction_conf * weights['PREDICTION']
            sentiment_comp = sentiment * weights['SENTIMENT']
            whale_comp = whale_acc * weights['WHALE']
            sm_comp = sm_mult * weights['SMART_MONEY'] * 0.6  # Slightly reduced weight
            
            # Calculate final signal (-1 to 1 range)
            signal = prediction_comp + sentiment_comp + whale_comp + sm_comp
            
            # Convert to 0-1 range
            normalized_signal = (signal + 1) / 2
            
            return normalized_signal
            
        except Exception as e:
            logging.error(f"Error calculating LunarCrush signal strength: {str(e)}")
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
        """Provide fallback insights data when LunarCrush is unavailable"""
        return {
            "market_prediction": {"prediction": {"direction": "neutral", "confidence": 0.5}},
            "sentiment": {"sentiment_score": 0},
            "whale_activity": {"accumulation_score": 0},
            "smart_money": {"position": "neutral"},
            "timestamp": time.time(),
            "metrics": {
                "overall_sentiment": 0,
                "prediction_direction": "neutral",
                "prediction_confidence": 0.5,
                "whale_accumulation": 0,
                "smart_money_direction": "neutral"
            }
        }


class DummyNebulaAI:
    """Fallback class when LunarCrush API is disabled or unavailable"""
    def __init__(self):
        logging.info("Dummy AI client initialized (fallback mode)")
    
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