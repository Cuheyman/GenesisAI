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
        """RETTET: Initialisering med hurtigere rate limits som før"""
        self.api_key = api_key or getattr(config, 'COINGECKO_API_KEY', '')
        self.base_url = "https://api.coingecko.com/api/v3"
        self.pro_url = "https://pro-api.coingecko.com/api/v3"
        self.cache = {}
        self.cache_expiry = {}
        
        # HURTIGERE rate limiting som før - IKKE ultra-konservativ
        if not self.api_key or not self.api_key.startswith('CG-'):
            self.min_request_interval = 2.0  # TILBAGE til 2 sekunder (fra 15)
            self.max_requests_per_minute = 20  # TILBAGE til 20 (fra 3)
            logging.info("CoinGecko initialiseret - Gratis tier")
        else:
            self.min_request_interval = 1.0   # Hurtig for pro
            self.max_requests_per_minute = 40  # Mere for pro
            logging.info("CoinGecko initialiseret - Pro tier")
        
        # Track requests
        self.request_history = []
        self.consecutive_rate_limits = 0
        self.backoff_multiplier = 1.0
        self.last_request_time = 0
        
        # Global cooldown UDEN spam logging
        self.global_cooldown_until = 0
        
        # Request timeout
        self.request_timeout = 15
        
        # Tjek om API er tilgængelig
        self.api_available = self._check_api_connection()
        
        # Opret coin ID mapping cache
        self.coin_id_cache = {}
        
        if self.api_available:
            if self.api_key and self.api_key.startswith('CG-'):
                logging.info("CoinGecko AI klient initialiseret med demo API nøgle")
            else:
                logging.info("CoinGecko AI klient initialiseret succesfuldt")
        else:
            logging.warning("CoinGecko AI klient initialiseret - forbindelse check fejlede men vil prøve igen")
        
    def _check_api_connection(self):
        """Check if the CoinGecko API is available"""
        try:
            test_url = "https://api.coingecko.com/api/v3/ping"
            headers = {"accept": "application/json"}
            
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
                    return True
            else:
                logging.warning(f"CoinGecko API returned status {response.status_code}")
                return True
                
        except Exception as e:
            logging.warning(f"CoinGecko connection check error: {str(e)} - will retry later")
            return True
    
    def _get_base_url(self):
        """Get the appropriate base URL (pro or free)"""
        return self.base_url
    
    def _get_headers(self):
        """Get headers for API requests"""
        headers = {"accept": "application/json"}
        if self.api_key:
            if self.api_key.startswith('CG-'):
                headers["x-cg-demo-api-key"] = self.api_key
            else:
                headers["x-cg-pro-api-key"] = self.api_key
        return headers
    
    def _check_rate_limit(self):
        """FIXED: More conservative rate limit checking"""
        current_time = time.time()
        
        # Check global cooldown first
        if current_time < self.global_cooldown_until:
            return self.global_cooldown_until - current_time
        
        # Remove requests older than 1 minute
        self.request_history = [req_time for req_time in self.request_history 
                               if current_time - req_time < 60]
        
        # Check if we're hitting the per-minute limit
        if len(self.request_history) >= self.max_requests_per_minute:
            oldest_request = min(self.request_history)
            wait_time = 60 - (current_time - oldest_request) + 5  # INCREASED buffer from 2 to 5 seconds
            return wait_time
        
        return 0
        
    async def _make_api_request(self, endpoint: str, params: Dict[str, Any] = None) -> Optional[Dict]:
     
        if not self.api_available:
            return None

        # Tjek global cooldown STILLE
        current_time = time.time()
        if current_time < self.global_cooldown_until:
            
            return None

        # Tjek rate limits STILLE
        rate_limit_wait = self._check_rate_limit()
        if rate_limit_wait > 0:
            # FJERNET: Ingen rate limit wait logging
            await asyncio.sleep(rate_limit_wait)

        # Apply backoff multiplier
        effective_interval = self.min_request_interval * self.backoff_multiplier
        
        # Sørg for minimum tid mellem requests STILLE
        time_since_last = current_time - self.last_request_time
        if time_since_last < effective_interval:
            wait_time = effective_interval - time_since_last
            # FJERNET: Ingen rate limiting logs
            await asyncio.sleep(wait_time)

        try:
            url = f"{self._get_base_url()}/{endpoint}"
            response = requests.get(
                url,
                params=params or {},
                headers=self._get_headers(),
                timeout=self.request_timeout
            )

            # Registrer request tid
            self.last_request_time = time.time()
            self.request_history.append(self.last_request_time)

            if response.status_code == 200:
                data = response.json()
                # FJERNET: Ingen debug success logs
                
                # Reset backoff ved success
                self.consecutive_rate_limits = 0
                self.backoff_multiplier = max(1.0, self.backoff_multiplier * 0.9)
                
                return data
                
            elif response.status_code == 429:
                # Håndter rate limit STILLE
                self.consecutive_rate_limits += 1
                self.backoff_multiplier = min(10.0, self.backoff_multiplier * 2.0)
                
                
            elif response.status_code == 403:
                if not hasattr(self, '_403_logged'):
                    logging.error("CoinGecko API forbudt - tjek API nøgle")
                    self._403_logged = True  # Log kun én gang
                self.api_available = False
                return None
                
            else:
                # FJERNET: Reduceret logging for andre fejl
                return None

        except requests.exceptions.Timeout:
            # FJERNET: Ingen timeout logs
            return None
        except requests.exceptions.ConnectionError:
            # FJERNET: Ingen connection error logs
            return None
        except Exception as e:
            # FJERNET: Ingen generelle error logs
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
        
        # Fallback to lowercase symbol if not in mappings
        coin_id = symbol.lower()
        self.coin_id_cache[symbol] = coin_id
        return coin_id
    
    async def get_coin_data(self, token: str) -> Optional[Dict]:
        """FIXED: Get comprehensive coin data with better error handling"""
        cache_key = f"coin_data_{token}"
        
        # Check cache
        cached_data = self._get_cached_data(cache_key, 300)
        if cached_data:
            return cached_data
        
        try:
            coin_id = await self._get_coin_id(token)
            
            params = {
                "localization": "false",
                "tickers": "false",
                "market_data": "true",
                "community_data": "true",
                "developer_data": "false",
                "sparkline": "false"
            }
            
            result = await self._make_api_request(f"coins/{coin_id}", params)
            
            # FIXED: Always validate result before caching
            if result and isinstance(result, dict):
                self._cache_data(cache_key, result, 300)
                return result
            else:
                logging.debug(f"Invalid coin data returned for {token}")
                return None
            
        except Exception as e:
            logging.error(f"Error getting coin data for {token}: {str(e)}")
            return None
    
    async def get_market_data(self, token: str) -> Optional[Dict]:
        """FIXED: Get market data with better validation"""
        cache_key = f"market_data_{token}"
        
        cached_data = self._get_cached_data(cache_key, 180)
        if cached_data:
            return cached_data
        
        try:
            coin_id = await self._get_coin_id(token)
            
            params = {
                "ids": coin_id,
                "vs_currencies": "usd",
                "include_market_cap": "true",
                "include_24hr_vol": "true",
                "include_24hr_change": "true",
                "include_last_updated_at": "true"
            }
            
            result = await self._make_api_request("simple/price", params)
            
            # FIXED: Better validation of nested response
            if result and isinstance(result, dict) and coin_id in result:
                market_data = result[coin_id]
                if isinstance(market_data, dict):  # Additional validation
                    self._cache_data(cache_key, market_data, 180)
                    return market_data
            
            logging.debug(f"No valid market data for {token}")
            return None
            
        except Exception as e:
            logging.error(f"Error getting market data for {token}: {str(e)}")
            return None
    
    async def get_market_prediction(self, token: str, timeframe: str = "4h") -> Dict:
        """FIXED: Get market prediction with safe fallbacks"""
        if not self.api_available:
            return self._get_default_prediction()
        
        try:
            coin_data = await self.get_coin_data(token)
            if not coin_data or not isinstance(coin_data, dict):
                return self._get_default_prediction()
            
            market_data = coin_data.get("market_data", {})
            if not isinstance(market_data, dict):
                return self._get_default_prediction()
            
            # FIXED: Safe extraction with defaults
            price_change_24h = self._safe_get_float(market_data, "price_change_percentage_24h", 0)
            price_change_7d = self._safe_get_float(market_data, "price_change_percentage_7d", 0)
            price_change_30d = self._safe_get_float(market_data, "price_change_percentage_30d", 0)
            
            volume_24h = self._safe_get_nested_float(market_data, ["total_volume", "usd"], 0)
            market_cap = self._safe_get_nested_float(market_data, ["market_cap", "usd"], 0)
            market_cap_rank = self._safe_get_int(market_data, "market_cap_rank", 999)
            
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
            
            # Market cap rank consideration
            if market_cap_rank <= 50:
                momentum_score += 0.5
                confidence_factors.append("top_50_coin")
            elif market_cap_rank <= 100:
                momentum_score += 0.2
                confidence_factors.append("top_100_coin")
            
            # Volume analysis
            if market_cap > 0:
                volume_to_mcap = volume_24h / market_cap
                if volume_to_mcap > 0.15:
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
            return self._get_default_prediction()
    
    async def get_whale_activity(self, token: str) -> Dict:
        """FIXED: Analyze whale activity with safe data handling"""
        if not self.api_available:
            return {"accumulation_score": 0}
        
        try:
            coin_data = await self.get_coin_data(token)
            if not coin_data or not isinstance(coin_data, dict):
                return {"accumulation_score": 0}
            
            market_data = coin_data.get("market_data", {})
            if not isinstance(market_data, dict):
                return {"accumulation_score": 0}
            
            # FIXED: Safe extraction
            volume_24h = self._safe_get_nested_float(market_data, ["total_volume", "usd"], 0)
            market_cap = self._safe_get_nested_float(market_data, ["market_cap", "usd"], 0)
            price_change_24h = self._safe_get_float(market_data, "price_change_percentage_24h", 0)
            
            # Calculate volume metrics
            volume_to_mcap = volume_24h / market_cap if market_cap > 0 else 0
            
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
            
            # Market cap rank influence
            market_cap_rank = self._safe_get_int(market_data, "market_cap_rank", 999)
            if market_cap_rank <= 10 and volume_to_mcap > 0.1:
                accumulation_score += 0.1
                activity_factors.append("top_10_whale_interest")
            elif market_cap_rank <= 50 and volume_to_mcap > 0.2:
                accumulation_score += 0.05
                activity_factors.append("established_coin_activity")
            
            # Price volatility analysis
            ath = self._safe_get_nested_float(market_data, ["ath", "usd"], 0)
            current_price = self._safe_get_nested_float(market_data, ["current_price", "usd"], 0)
            
            if ath > 0 and current_price > 0:
                price_from_ath = (current_price / ath - 1) * 100
                
                if price_from_ath < -50:
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
        """FIXED: Analyze smart money with safe data handling"""
        if not self.api_available:
            return {"position": "neutral"}
        
        try:
            coin_data = await self.get_coin_data(token)
            if not coin_data or not isinstance(coin_data, dict):
                return {"position": "neutral"}
            
            market_data = coin_data.get("market_data", {})
            if not isinstance(market_data, dict):
                return {"position": "neutral"}
            
            # FIXED: Safe extraction
            market_cap_rank = self._safe_get_int(market_data, "market_cap_rank", 999)
            price_change_7d = self._safe_get_float(market_data, "price_change_percentage_7d", 0)
            price_change_30d = self._safe_get_float(market_data, "price_change_percentage_30d", 0)
            volume_24h = self._safe_get_nested_float(market_data, ["total_volume", "usd"], 0)
            market_cap = self._safe_get_nested_float(market_data, ["market_cap", "usd"], 0)
            
            position_score = 0
            smart_money_factors = []
            
            # Smart money typically focuses on fundamentally strong projects
            if market_cap_rank <= 20:
                position_score += 0.3
                smart_money_factors.append("top_20_project")
            elif market_cap_rank <= 50:
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
            
            # ATH analysis
            ath = self._safe_get_nested_float(market_data, ["ath", "usd"], 0)
            current_price = self._safe_get_nested_float(market_data, ["current_price", "usd"], 0)
            
            if ath > 0 and current_price > 0:
                price_from_ath = (current_price / ath - 1) * 100
                
                if -40 < price_from_ath < -20:
                    position_score += 0.1
                    smart_money_factors.append("potential_value_zone")
                elif price_from_ath > -10:
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
        """FIXED: Get consolidated trading signal with safe handling"""
        try:
            insights = await self.get_consolidated_insights(token)
            
            # FIXED: Safe extraction
            if not insights or not isinstance(insights, dict):
                return {"action": "hold", "strength": 0.5}
            
            metrics = insights.get("metrics", {})
            if not isinstance(metrics, dict):
                return {"action": "hold", "strength": 0.5}
            
            signal_strength = self.get_signal_strength(insights)
            
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
        """FIXED: Get sentiment analysis with safe data handling"""
        if not self.api_available:
            return {"sentiment_score": 0}
        
        try:
            coin_data = await self.get_coin_data(token)
            if not coin_data or not isinstance(coin_data, dict):
                return {"sentiment_score": 0}
            
            market_data = coin_data.get("market_data", {})
            community_data = coin_data.get("community_data", {})
            
            if not isinstance(market_data, dict):
                market_data = {}
            if not isinstance(community_data, dict):
                community_data = {}
            
            sentiment_score = 0
            sentiment_factors = []
            
            # FIXED: Safe price momentum calculation
            price_change_24h = self._safe_get_float(market_data, "price_change_percentage_24h", 0)
            price_change_7d = self._safe_get_float(market_data, "price_change_percentage_7d", 0)
            
            # Convert price changes to sentiment
            price_sentiment = np.tanh(price_change_24h / 10)
            weekly_sentiment = np.tanh(price_change_7d / 20)
            
            sentiment_score += price_sentiment * 0.4 + weekly_sentiment * 0.3
            
            # FIXED: Safe community sentiment indicators
            twitter_followers = self._safe_get_int(community_data, "twitter_followers", 0)
            reddit_subscribers = self._safe_get_int(community_data, "reddit_subscribers", 0)
            telegram_users = self._safe_get_int(community_data, "telegram_channel_user_count", 0)
            
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
            
            # Market cap and volume sentiment
            market_cap_rank = self._safe_get_int(market_data, "market_cap_rank", 999)
            if market_cap_rank <= 20:
                sentiment_score += 0.1
                sentiment_factors.append("top_tier_coin")
            elif market_cap_rank <= 50:
                sentiment_score += 0.05
                sentiment_factors.append("established_coin")
            
            # Volume trend sentiment
            volume_24h = self._safe_get_nested_float(market_data, ["total_volume", "usd"], 0)
            market_cap_value = self._safe_get_nested_float(market_data, ["market_cap", "usd"], 0)
            
            if market_cap_value > 0:
                volume_ratio = volume_24h / market_cap_value
                if volume_ratio > 0.2:
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
        """FIXED: Get consolidated insights with better error handling"""
        token = token.replace("USDT", "").upper()
        
        if not self.api_available:
            logging.warning(f"CoinGecko API unavailable for {token} analysis - using fallback data")
            return self._get_fallback_insights(token)
        
        try:
            # FIXED: More conservative task handling
            tasks = [
                asyncio.create_task(self.get_market_prediction(token)),
                asyncio.create_task(self.get_sentiment_analysis(token)),
                asyncio.create_task(self.get_whale_activity(token)),
                asyncio.create_task(self.get_smart_money_positions(token))
            ]
            
            # FIXED: Shorter timeout and better error handling
            done, pending = await asyncio.wait(tasks, timeout=10)  # Reduced from 15 to 10 seconds
            
            # Cancel any pending tasks
            for task in pending:
                task.cancel()
            
            # Get results with defaults
            results = []
            for i, task in enumerate(tasks):
                try:
                    if task in done and not task.cancelled():
                        result = task.result()
                        # FIXED: Validate result before using
                        if result and isinstance(result, dict):
                            results.append(result)
                        else:
                            results.append(self._get_default_result(i))
                    else:
                        results.append(self._get_default_result(i))
                except Exception as e:
                    logging.error(f"Error getting CoinGecko data task {i}: {str(e)}")
                    results.append(self._get_default_result(i))
            
            # FIXED: Ensure we have exactly 4 results
            while len(results) < 4:
                results.append(self._get_default_result(len(results)))
            
            market_prediction, sentiment, whale_activity, smart_money = results[:4]
            
            consolidated = {
                "market_prediction": market_prediction,
                "sentiment": sentiment,
                "whale_activity": whale_activity,
                "smart_money": smart_money,
                "timestamp": time.time()
            }
            
            # FIXED: Safe metrics extraction
            prediction_data = market_prediction.get("prediction", {}) if isinstance(market_prediction, dict) else {}
            
            metrics = {
                "overall_sentiment": sentiment.get("sentiment_score", 0) if isinstance(sentiment, dict) else 0,
                "prediction_direction": prediction_data.get("direction", "neutral") if isinstance(prediction_data, dict) else "neutral",
                "prediction_confidence": prediction_data.get("confidence", 0.5) if isinstance(prediction_data, dict) else 0.5,
                "whale_accumulation": whale_activity.get("accumulation_score", 0) if isinstance(whale_activity, dict) else 0,
                "smart_money_direction": smart_money.get("position", "neutral") if isinstance(smart_money, dict) else "neutral"
            }
            
            consolidated["metrics"] = metrics
            
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
        """FIXED: Calculate signal strength with safe handling"""
        if not insights or not isinstance(insights, dict):
            return 0.5
        
        try:
            metrics = insights.get('metrics', {})
            if not isinstance(metrics, dict):
                return 0.5
            
            # Get individual metrics with safe defaults
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
            
            # Calculate direction multipliers
            pred_mult = 1 if prediction_dir == 'bullish' else (-1 if prediction_dir == 'bearish' else 0)
            sm_mult = 1 if smart_money == 'bullish' else (-1 if smart_money == 'bearish' else 0)
            
            # Calculate weighted components
            prediction_comp = pred_mult * prediction_conf * weights.get('PREDICTION', 0.35)
            sentiment_comp = sentiment * weights.get('SENTIMENT', 0.30)
            whale_comp = whale_acc * weights.get('WHALE', 0.20)
            sm_comp = sm_mult * weights.get('SMART_MONEY', 0.15) * 0.8
            
            # Calculate final signal
            signal = prediction_comp + sentiment_comp + whale_comp + sm_comp
            
            # Convert to 0-1 range (strength)
            normalized_signal = abs(signal)
            
            # Apply agreement bonus
            agreement_count = sum([
                1 for comp in [prediction_comp, sentiment_comp, whale_comp, sm_comp]
                if abs(comp) > 0.1
            ])
            
            if agreement_count >= 3:
                normalized_signal *= 1.2
            elif agreement_count >= 2:
                normalized_signal *= 1.1
            
            return min(1.0, max(0.0, normalized_signal))
            
        except Exception as e:
            logging.error(f"Error calculating CoinGecko signal strength: {str(e)}")
            return 0.5
    
    # FIXED: Add helper methods for safe data extraction
    def _safe_get_float(self, data: dict, key: str, default: float = 0.0) -> float:
        """Safely get a float value from dict"""
        try:
            value = data.get(key, default)
            return float(value) if value is not None else default
        except (ValueError, TypeError):
            return default
    
    def _safe_get_int(self, data: dict, key: str, default: int = 0) -> int:
        """Safely get an int value from dict"""
        try:
            value = data.get(key, default)
            return int(value) if value is not None else default
        except (ValueError, TypeError):
            return default
    
    def _safe_get_nested_float(self, data: dict, keys: list, default: float = 0.0) -> float:
        """Safely get a nested float value from dict"""
        try:
            current = data
            for key in keys:
                if isinstance(current, dict) and key in current:
                    current = current[key]
                else:
                    return default
            return float(current) if current is not None else default
        except (ValueError, TypeError):
            return default
    
    def _get_default_result(self, index: int) -> Dict:
        """Get default result for failed API calls"""
        defaults = [
            {"prediction": {"direction": "neutral", "confidence": 0.5}},  # market_prediction
            {"sentiment_score": 0},  # sentiment
            {"accumulation_score": 0},  # whale_activity
            {"position": "neutral"}  # smart_money
        ]
        return defaults[index] if index < len(defaults) else {}
    
    def _get_default_prediction(self) -> Dict:
        """Get default prediction when API fails"""
        return {"prediction": {"direction": "neutral", "confidence": 0.5}}
    
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
        
    async def get_consolidated_insights(self, token: str, max_wait_time: int = 8) -> Dict:
        """RETTET: Få consolidated insights med SIMPEL logging som før"""
        token = token.replace("USDT", "").upper()
        
        if not self.api_available:
            return self._get_fallback_insights(token)
        
        try:
            # MEGET kortere timeout for hurtigere responses
            tasks = [
                asyncio.create_task(self.get_market_prediction(token)),
                asyncio.create_task(self.get_sentiment_analysis(token)),
                asyncio.create_task(self.get_whale_activity(token)),
                asyncio.create_task(self.get_smart_money_positions(token))
            ]
            
            # Vent på alle tasks med reduceret timeout
            done, pending = await asyncio.wait(tasks, timeout=6)  # REDUCERET fra 15 til 6 sekunder
            
            # Cancel alle pending tasks
            for task in pending:
                task.cancel()
            
            # Få resultater med defaults
            results = []
            for i, task in enumerate(tasks):
                try:
                    if task in done and not task.cancelled():
                        result = task.result()
                        if result and isinstance(result, dict):
                            results.append(result)
                        else:
                            results.append(self._get_default_result(i))
                    else:
                        results.append(self._get_default_result(i))
                except Exception:
                    results.append(self._get_default_result(i))
            
            # Sørg for vi har præcis 4 resultater
            while len(results) < 4:
                results.append(self._get_default_result(len(results)))
            
            market_prediction, sentiment, whale_activity, smart_money = results[:4]
            
            consolidated = {
                "market_prediction": market_prediction,
                "sentiment": sentiment,
                "whale_activity": whale_activity,
                "smart_money": smart_money,
                "timestamp": time.time()
            }
            
            # Udtræk nøgle metrics
            prediction_data = market_prediction.get("prediction", {}) if isinstance(market_prediction, dict) else {}
            
            metrics = {
                "overall_sentiment": sentiment.get("sentiment_score", 0) if isinstance(sentiment, dict) else 0,
                "prediction_direction": prediction_data.get("direction", "neutral") if isinstance(prediction_data, dict) else "neutral",
                "prediction_confidence": prediction_data.get("confidence", 0.5) if isinstance(prediction_data, dict) else 0.5,
                "whale_accumulation": whale_activity.get("accumulation_score", 0) if isinstance(whale_activity, dict) else 0,
                "smart_money_direction": smart_money.get("position", "neutral") if isinstance(smart_money, dict) else "neutral"
            }
            
            consolidated["metrics"] = metrics
            
            # SIMPEL, REN logging som du havde før - KUN log insights
            logging.info(f"CoinGecko insights for {token}: {metrics['prediction_direction']} "
                        f"({metrics['prediction_confidence']:.2f} confidence), "
                        f"sentiment: {metrics['overall_sentiment']:.2f}, "
                        f"whale: {metrics['whale_accumulation']:.2f}, "
                        f"smart money: {metrics['smart_money_direction']}")
            
            return consolidated
            
        except Exception:
            # Returner fallback stille uden error logs
            return self._get_fallback_insights(token)
    def get_signal_strength(self, *args, **kwargs):
        return 0.5