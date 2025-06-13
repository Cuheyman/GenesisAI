import requests
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
            if self.api_key and self.api_key.startswith('CG-'):
                logging.info("CoinGecko AI client initialized with demo API key")
            else:
                logging.info("CoinGecko AI client initialized successfully")
        else:
            # This should rarely happen now
            logging.warning("CoinGecko AI client initialized - connection check failed but will retry")
    
    def _check_api_connection(self):
        """Check if the CoinGecko API is available"""
        try:
            # Always test with the free tier ping endpoint
            test_url = "https://api.coingecko.com/api/v3/ping"
            headers = {"accept": "application/json"}
            
            # Add demo API key if available
            if self.api_key and self.api_key.startswith('CG-'):
                headers["x-cg-demo-api-key"] = self.api_key
            
            response = requests.get(test_url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                try:
                    data = response.json()
                    if 'gecko_says' in data:
                        logging.info(f"CoinGecko API connected: {data['gecko_says']}")
                        return True
                except:
                    # Even without JSON, 200 status means success
                    return True
            else:
                logging.warning(f"CoinGecko API returned status {response.status_code}")
                # Don't completely fail - allow retry later
                return True  # Return True to allow retry on actual requests
                
        except requests.exceptions.ConnectionError:
            logging.warning("CoinGecko connection error - will retry later")
            return True  # Allow retry
        except requests.exceptions.Timeout:
            logging.warning("CoinGecko request timed out - will retry later")
            return True  # Allow retry
        except Exception as e:
            logging.warning(f"CoinGecko connection check error: {str(e)} - will retry later")
            return True  # Allow retry
    def _get_base_url(self):
        """Get the appropriate base URL (pro or free)"""
        # Always use free tier URL - pro tier requires special paid API keys
        # Demo keys should use the free tier endpoint
        return self.base_url  # Always use api.coingecko.com
    
    def _get_headers(self):
        """Get headers for API requests"""
        headers = {"accept": "application/json"}
        # For demo/free keys, use the demo API key header
        if self.api_key:
            # Demo keys use x-cg-demo-api-key header
            if self.api_key.startswith('CG-'):
                headers["x-cg-demo-api-key"] = self.api_key
            else:
                # Legacy pro key format
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
            "LTC": "litecoin",
            "BCH": "bitcoin-cash",
            "ALGO": "algorand",
            "ATOM": "cosmos",
            "FIL": "filecoin",
            "VET": "vechain",
            "ICP": "internet-computer",
            "THETA": "theta-network",
            "XLM": "stellar",
            "TRX": "tron",
            "ETC": "ethereum-classic",
            "HBAR": "hedera-hashgraph",
            "NEAR": "near",
            "APT": "aptos",
            "SUI": "sui",
            "ARB": "arbitrum",
            "OP": "optimism",
            "PEPE": "pepe",
            "SHIB": "shiba-inu"
        }
        
        if symbol in symbol_to_id:
            coin_id = symbol_to_id[symbol]
            self.coin_id_cache[symbol] = coin_id
            return coin_id
        
        # If not in common mappings, try to find via API
        try:
            coins_list = await self._make_api_request("coins/list")
            if coins_list:
                for coin in coins_list:
                    if coin.get("symbol", "").upper() == symbol:
                        coin_id = coin["id"]
                        self.coin_id_cache[symbol] = coin_id
                        return coin_id
        except Exception as e:
            logging.error(f"Error getting coin ID for {symbol}: {str(e)}")
        
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
    
    async def get_market_data(self, token: str) -> Optional[Dict]:
        """Get market data for analysis"""
        cache_key = f"market_data_{token}"
        
        # Check cache
        cached_data = self._get_cached_data(cache_key, 180)  # 3 minute cache
        if cached_data:
            return cached_data
        
        try:
            coin_id = await self._get_coin_id(token)
            
            # Get market data
            params = {
                "ids": coin_id,
                "vs_currencies": "usd",
                "include_market_cap": "true",
                "include_24hr_vol": "true",
                "include_24hr_change": "true",
                "include_last_updated_at": "true"
            }
            
            result = await self._make_api_request("simple/price", params)
            
            if result and coin_id in result:
                market_data = result[coin_id]
                self._cache_data(cache_key, market_data, 180)
                return market_data
            
        except Exception as e:
            logging.error(f"Error getting market data for {token}: {str(e)}")
        
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
            
            # Extract key metrics with safe None handling
            price_change_24h = market_data.get("price_change_percentage_24h") or 0
            price_change_7d = market_data.get("price_change_percentage_7d") or 0
            price_change_30d = market_data.get("price_change_percentage_30d") or 0
            
            volume_24h = market_data.get("total_volume", {}).get("usd") or 0
            market_cap = market_data.get("market_cap", {}).get("usd") or 0
            market_cap_rank = market_data.get("market_cap_rank") or 999
            
            # Calculate momentum score
            momentum_score = 0
            confidence_factors = []
            
            # Price momentum analysis
            if price_change_24h > 5:
                momentum_score += 2
                confidence_factors.append("strong_24h_gain")
            elif price_change_24h > 2:
                momentum_score += 1
                confidence_factors.append("moderate_24h_gain")
            elif price_change_24h < -5:
                momentum_score -= 2
                confidence_factors.append("strong_24h_loss")
            elif price_change_24h < -2:
                momentum_score -= 1
                confidence_factors.append("moderate_24h_loss")
            
            # Weekly trend analysis
            if price_change_7d > 10:
                momentum_score += 1.5
                confidence_factors.append("strong_weekly_trend")
            elif price_change_7d > 5:
                momentum_score += 0.5
                confidence_factors.append("positive_weekly_trend")
            elif price_change_7d < -10:
                momentum_score -= 1.5
                confidence_factors.append("weak_weekly_trend")
            elif price_change_7d < -5:
                momentum_score -= 0.5
                confidence_factors.append("negative_weekly_trend")
            
            # Market cap rank consideration - FIXED: Added None check
            if market_cap_rank is not None and market_cap_rank <= 50:
                momentum_score += 0.5  # Top 50 coins get slight boost
                confidence_factors.append("top_50_coin")
            elif market_cap_rank is not None and market_cap_rank <= 100:
                momentum_score += 0.2  # Top 100 coins get small boost
                confidence_factors.append("top_100_coin")
            
            # Volume analysis (relative to market cap)
            if market_cap > 0:
                volume_to_mcap = volume_24h / market_cap
                if volume_to_mcap > 0.15:  # High volume relative to market cap
                    momentum_score += 1
                    confidence_factors.append("high_volume_activity")
                elif volume_to_mcap > 0.05:
                    momentum_score += 0.5
                    confidence_factors.append("moderate_volume_activity")
            
            # Determine direction and confidence
            if momentum_score >= 2:
                direction = "bullish"
                confidence = min(0.9, 0.6 + (momentum_score - 2) * 0.1)
            elif momentum_score <= -2:
                direction = "bearish" 
                confidence = min(0.9, 0.6 + abs(momentum_score + 2) * 0.1)
            else:
                direction = "neutral"
                confidence = 0.5
            
            analysis = f"24h: {price_change_24h:.1f}%, 7d: {price_change_7d:.1f}%, Score: {momentum_score:.1f}"
            
            return {
                "prediction": {
                    "direction": direction,
                    "confidence": confidence,
                    "analysis": analysis,
                    "factors": confidence_factors
                }
            }
            
        except Exception as e:
            logging.error(f"Error in CoinGecko market prediction: {str(e)}")
            return {"prediction": {"direction": "neutral", "confidence": 0.5}}
    
    async def get_whale_activity(self, token: str) -> Dict:
        """Analyze whale activity using CoinGecko volume and market data"""
        if not self.api_available:
            return {"accumulation_score": 0}
        
        try:
            coin_data = await self.get_coin_data(token)
            if not coin_data:
                return {"accumulation_score": 0}
            
            market_data = coin_data.get("market_data", {})
            
            # Get volume and market metrics with safe None handling
            volume_24h = market_data.get("total_volume", {}).get("usd") or 0
            market_cap = market_data.get("market_cap", {}).get("usd") or 0
            price_change_24h = market_data.get("price_change_percentage_24h") or 0
            
            # Calculate volume metrics
            if market_cap > 0:
                volume_to_mcap = volume_24h / market_cap
            else:
                volume_to_mcap = 0
            
            accumulation_score = 0
            activity_factors = []
            
            # High volume with price stability or gains suggests accumulation
            if volume_to_mcap > 0.3 and price_change_24h > 0:
                accumulation_score += 0.4
                activity_factors.append("high_volume_accumulation")
            elif volume_to_mcap > 0.2 and price_change_24h > -2:
                accumulation_score += 0.2
                activity_factors.append("moderate_volume_accumulation")
            elif volume_to_mcap > 0.15 and price_change_24h > 0:
                accumulation_score += 0.1
                activity_factors.append("steady_accumulation")
            
            # High volume with significant price drop suggests distribution
            elif volume_to_mcap > 0.3 and price_change_24h < -5:
                accumulation_score -= 0.4
                activity_factors.append("high_volume_distribution")
            elif volume_to_mcap > 0.2 and price_change_24h < -3:
                accumulation_score -= 0.2
                activity_factors.append("moderate_distribution")
            
            # Market cap rank influence (whales more active in top coins) - FIXED: Added None check
            market_cap_rank = market_data.get("market_cap_rank") or 999
            if market_cap_rank is not None and market_cap_rank <= 10 and volume_to_mcap > 0.1:
                accumulation_score += 0.1  # Top 10 coins with decent volume
                activity_factors.append("top_10_whale_interest")
            elif market_cap_rank is not None and market_cap_rank <= 50 and volume_to_mcap > 0.2:
                accumulation_score += 0.05
                activity_factors.append("established_coin_activity")
            
            # Price volatility analysis with safe None handling
            ath = market_data.get("ath", {}).get("usd") or 0
            current_price = market_data.get("current_price", {}).get("usd") or 0
            
            if ath > 0 and current_price > 0:
                price_from_ath = (current_price / ath - 1) * 100
                
                # Buying near lows could indicate smart money accumulation
                if price_from_ath < -50:  # More than 50% down from ATH
                    accumulation_score += 0.1
                    activity_factors.append("potential_bottom_accumulation")
                elif price_from_ath < -30:
                    accumulation_score += 0.05
                    activity_factors.append("value_accumulation")
            
            # Clamp to -1 to 1 range
            accumulation_score = max(-1, min(1, accumulation_score))
            
            analysis = f"Volume/MCap: {volume_to_mcap:.3f}, Price 24h: {price_change_24h:.1f}%"
            
            return {
                "accumulation_score": accumulation_score,
                "analysis": analysis,
                "factors": activity_factors
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
            
            # Get relevant metrics with safe None handling
            market_cap_rank = market_data.get("market_cap_rank") or 999
            price_change_7d = market_data.get("price_change_percentage_7d") or 0
            price_change_30d = market_data.get("price_change_percentage_30d") or 0
            volume_24h = market_data.get("total_volume", {}).get("usd") or 0
            market_cap = market_data.get("market_cap", {}).get("usd") or 0
            
            position_score = 0
            smart_money_factors = []
            
            # Smart money typically focuses on fundamentally strong projects - FIXED: Added None check
            if market_cap_rank is not None and market_cap_rank <= 20:
                position_score += 0.3
                smart_money_factors.append("top_20_project")
            elif market_cap_rank is not None and market_cap_rank <= 50:
                position_score += 0.1
                smart_money_factors.append("established_project")
            
            # Consistent positive performance suggests smart money confidence
            if price_change_7d > 5 and price_change_30d > 10:
                position_score += 0.3
                smart_money_factors.append("consistent_outperformance")
            elif price_change_7d > 0 and price_change_30d > 5:
                position_score += 0.1
                smart_money_factors.append("steady_growth")
            
            # Strong negative performance might indicate smart money exit
            elif price_change_7d < -10 and price_change_30d < -20:
                position_score -= 0.3
                smart_money_factors.append("significant_decline")
            elif price_change_7d < -5 and price_change_30d < -10:
                position_score -= 0.1
                smart_money_factors.append("negative_trend")
            
            # Volume analysis for smart money activity
            if market_cap > 0:
                volume_ratio = volume_24h / market_cap
                if volume_ratio > 0.15 and price_change_7d > 0:
                    position_score += 0.2
                    smart_money_factors.append("high_volume_buying")
                elif volume_ratio > 0.15 and price_change_7d < -5:
                    position_score -= 0.2
                    smart_money_factors.append("high_volume_selling")
            
            # ATH analysis - smart money often accumulates during corrections with safe None handling
            ath = market_data.get("ath", {}).get("usd") or 0
            current_price = market_data.get("current_price", {}).get("usd") or 0
            
            if ath > 0 and current_price > 0:
                price_from_ath = (current_price / ath - 1) * 100
                
                if -40 < price_from_ath < -20:  # 20-40% down from ATH
                    position_score += 0.1
                    smart_money_factors.append("potential_value_zone")
                elif price_from_ath > -10:  # Near ATH
                    if price_change_7d > 5:
                        position_score += 0.1
                        smart_money_factors.append("momentum_continuation")
                    else:
                        position_score -= 0.05
                        smart_money_factors.append("potential_resistance")
            
            # Determine position
            if position_score > 0.2:
                position = "bullish"
            elif position_score < -0.2:
                position = "bearish"
            else:
                position = "neutral"
            
            analysis = f"Score: {position_score:.2f}, Rank: {market_cap_rank}, Factors: {len(smart_money_factors)}"
            
            return {
                "position": position,
                "analysis": analysis,
                "factors": smart_money_factors,
                "score": position_score
            }
            
        except Exception as e:
            logging.error(f"Error in CoinGecko smart money analysis: {str(e)}")
            return {"position": "neutral"}
    
    async def get_trading_signal(self, token: str) -> Dict:
        """Get consolidated trading signal from CoinGecko data"""
        try:
            # Get all insights
            insights = await self.get_consolidated_insights(token)
            
            # Extract metrics
            metrics = insights.get("metrics", {})
            
            # Calculate signal strength
            signal_strength = self.get_signal_strength(insights)
            
            # Determine action based on metrics
            prediction_direction = metrics.get("prediction_direction", "neutral")
            sentiment = metrics.get("overall_sentiment", 0)
            smart_money = metrics.get("smart_money_direction", "neutral")
            
            # Weight the different signals
            action_score = 0
            
            # Prediction weight (40%)
            if prediction_direction == "bullish":
                action_score += 0.4 * metrics.get("prediction_confidence", 0.5)
            elif prediction_direction == "bearish":
                action_score -= 0.4 * metrics.get("prediction_confidence", 0.5)
            
            # Sentiment weight (30%)
            action_score += 0.3 * sentiment
            
            # Smart money weight (30%)
            if smart_money == "bullish":
                action_score += 0.3
            elif smart_money == "bearish":
                action_score -= 0.3
            
            # Determine action
            if action_score > 0.3 and signal_strength > 0.6:
                action = "buy"
            elif action_score < -0.3 and signal_strength > 0.6:
                action = "sell"
            else:
                action = "hold"
            
            return {
                "action": action,
                "strength": signal_strength,
                "insights": metrics,
                "action_score": action_score
            }
            
        except Exception as e:
            logging.error(f"Error getting trading signal: {str(e)}")
            return {"action": "hold", "strength": 0.5}

    async def get_sentiment_analysis(self, token: str) -> Dict:
        """Get sentiment analysis from CoinGecko community and market data"""
        if not self.api_available:
            return {"sentiment_score": 0}
        
        try:
            coin_data = await self.get_coin_data(token)
            if not coin_data:
                return {"sentiment_score": 0}
            
            # Extract sentiment indicators
            market_data = coin_data.get("market_data", {})
            community_data = coin_data.get("community_data", {})
            
            sentiment_score = 0
            sentiment_factors = []
            
            # Price momentum sentiment - FIXED: Better None handling
            price_change_24h = market_data.get("price_change_percentage_24h") or 0
            price_change_7d = market_data.get("price_change_percentage_7d") or 0
            
            # Convert price changes to sentiment (-1 to 1)
            price_sentiment = np.tanh(price_change_24h / 10)  # Normalize large moves
            weekly_sentiment = np.tanh(price_change_7d / 20)
            
            sentiment_score += price_sentiment * 0.4 + weekly_sentiment * 0.3
            
            # Community sentiment indicators - FIXED: Better None handling
            twitter_followers = community_data.get("twitter_followers") or 0
            reddit_subscribers = community_data.get("reddit_subscribers") or 0
            telegram_users = community_data.get("telegram_channel_user_count") or 0
            
            # Social media growth indicates positive sentiment
            if twitter_followers > 100000:
                sentiment_score += 0.1
                sentiment_factors.append("large_twitter_following")
            if reddit_subscribers > 50000:
                sentiment_score += 0.1
                sentiment_factors.append("active_reddit_community")
            if telegram_users > 10000:
                sentiment_score += 0.05
                sentiment_factors.append("active_telegram")
            
            # Market cap and volume sentiment - FIXED: Added proper None check
            market_cap_rank = market_data.get("market_cap_rank")
            if market_cap_rank is not None and market_cap_rank <= 20:
                sentiment_score += 0.1
                sentiment_factors.append("top_tier_coin")
            elif market_cap_rank is not None and market_cap_rank <= 50:
                sentiment_score += 0.05
                sentiment_factors.append("established_coin")
            
            # Volume trend sentiment - FIXED: Better None handling
            volume_24h = market_data.get("total_volume", {}).get("usd") or 0
            market_cap = market_data.get("market_cap", {}).get("usd") or 0
            
            if market_cap > 0:
                volume_ratio = volume_24h / market_cap
                if volume_ratio > 0.2:  # Very high volume
                    sentiment_score += 0.1
                    sentiment_factors.append("high_trading_interest")
                elif volume_ratio > 0.1:
                    sentiment_score += 0.05
                    sentiment_factors.append("moderate_trading_interest")
            
            # Clamp sentiment score to -1 to 1 range
            sentiment_score = max(-1, min(1, sentiment_score))
            
            analysis = f"Price sentiment: {price_sentiment:.2f}, Community factors: {len(sentiment_factors)}"
            
            return {
                "sentiment_score": sentiment_score,
                "analysis": analysis,
                "factors": sentiment_factors
            }
            
        except Exception as e:
            logging.error(f"Error in CoinGecko sentiment analysis: {str(e)}")
            return {"sentiment_score": 0}

    async def get_consolidated_insights(self, token: str, max_wait_time: int = 15) -> Dict:
        """Get consolidated insights from all CoinGecko data sources"""
        token = token.replace("USDT", "").upper()
        
        if not self.api_available:
            logging.warning(f"CoinGecko API unavailable for {token} analysis - using fallback data")
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
                    logging.error(f"Error getting CoinGecko data: {str(e)}")
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
            logging.info(f"CoinGecko insights for {token}: {metrics['prediction_direction']} "
                        f"({metrics['prediction_confidence']:.2f} confidence), "
                        f"sentiment: {metrics['overall_sentiment']:.2f}, "
                        f"whale: {metrics['whale_accumulation']:.2f}, "
                        f"smart money: {metrics['smart_money_direction']}")
            
            return consolidated
            
        except Exception as e:
            logging.error(f"Error in get_consolidated_insights: {str(e)}")
            return self._get_fallback_insights(token)
    
    def get_signal_strength(self, insights: Dict) -> float:
        """Calculate signal strength from CoinGecko insights"""
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
            
            # Get weights from config or use defaults
            weights = getattr(config, 'COINGECKO_WEIGHTS', {
                'PREDICTION': 0.35,
                'SENTIMENT': 0.30,
                'WHALE': 0.20,
                'SMART_MONEY': 0.15
            })
            
            # Calculate direction multipliers (-1 to 1)
            pred_mult = 1 if prediction_dir == 'bullish' else (-1 if prediction_dir == 'bearish' else 0)
            sm_mult = 1 if smart_money == 'bullish' else (-1 if smart_money == 'bearish' else 0)
            
            # Calculate weighted components
            prediction_comp = pred_mult * prediction_conf * weights['PREDICTION']
            sentiment_comp = sentiment * weights['SENTIMENT']
            whale_comp = whale_acc * weights['WHALE']
            sm_comp = sm_mult * weights['SMART_MONEY'] * 0.8  # Slightly reduced weight
            
            # Calculate final signal (-1 to 1 range)
            signal = prediction_comp + sentiment_comp + whale_comp + sm_comp
            
            # Convert to 0-1 range (strength)
            normalized_signal = abs(signal)  # Take absolute value for strength
            
            # Boost signal strength if multiple indicators agree
            agreement_count = 0
            if abs(prediction_comp) > 0.1:
                agreement_count += 1
            if abs(sentiment_comp) > 0.1:
                agreement_count += 1
            if abs(whale_comp) > 0.1:
                agreement_count += 1
            if abs(sm_comp) > 0.1:
                agreement_count += 1
            
            # Apply agreement bonus
            if agreement_count >= 3:
                normalized_signal *= 1.2
            elif agreement_count >= 2:
                normalized_signal *= 1.1
            
            # Ensure within bounds
            normalized_signal = min(1.0, max(0.0, normalized_signal))
            
            return normalized_signal
            
        except Exception as e:
            logging.error(f"Error calculating CoinGecko signal strength: {str(e)}")
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