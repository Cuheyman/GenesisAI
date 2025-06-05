# lunarcrush_client.py
import aiohttp
import asyncio
import logging
import time
from typing import Dict, List, Optional, Any
import json

class LunarCrushClient:
    """Client for LunarCrush API - Social sentiment and market intelligence"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key
        self.base_url = "https://lunarcrush.com/api3/public"
        self.cache = {}
        self.cache_duration = 300  # 5 minutes cache
        self.session = None
        self.rate_limit_remaining = 100
        self.rate_limit_reset = 0
        
    async def initialize(self):
        """Initialize the HTTP session"""
        if not self.session:
            self.session = aiohttp.ClientSession()
        logging.info("LunarCrush client initialized")
    
    async def close(self):
        """Close the HTTP session"""
        if self.session:
            await self.session.close()
    
    async def _make_request(self, endpoint: str, params: Dict = None) -> Optional[Dict]:
        """Make API request with caching and rate limiting"""
        # Check rate limit
        if self.rate_limit_remaining <= 0 and time.time() < self.rate_limit_reset:
            wait_time = self.rate_limit_reset - time.time()
            logging.warning(f"Rate limit reached, waiting {wait_time:.1f} seconds")
            await asyncio.sleep(wait_time)
        
        # Check cache
        cache_key = f"{endpoint}_{json.dumps(params or {}, sort_keys=True)}"
        if cache_key in self.cache:
            cached_data, cache_time = self.cache[cache_key]
            if time.time() - cache_time < self.cache_duration:
                return cached_data
        
        # Prepare headers
        headers = {"Accept": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        try:
            if not self.session:
                await self.initialize()
            
            url = f"{self.base_url}{endpoint}"
            async with self.session.get(url, params=params, headers=headers) as response:
                # Update rate limit info
                self.rate_limit_remaining = int(response.headers.get('X-RateLimit-Remaining', 100))
                self.rate_limit_reset = int(response.headers.get('X-RateLimit-Reset', 0))
                
                if response.status == 200:
                    data = await response.json()
                    # Cache the response
                    self.cache[cache_key] = (data, time.time())
                    return data
                else:
                    logging.error(f"LunarCrush API error: {response.status} - {await response.text()}")
                    return None
                    
        except Exception as e:
            logging.error(f"Error making LunarCrush request: {str(e)}")
            return None
    
    async def get_coin_social_data(self, symbol: str) -> Optional[Dict]:
        """Get social metrics for a cryptocurrency"""
        # LunarCrush uses lowercase symbols
        coin = symbol.lower().replace('usdt', '')
        
        data = await self._make_request(f"/coins/{coin}/v1")
        if data:
            return self._parse_coin_data(data)
        return None
    
    async def get_coins_list(self) -> Optional[List[Dict]]:
        """Get list of all tracked coins with social metrics"""
        data = await self._make_request("/coins/list/v2")
        if data and 'data' in data:
            return data['data']
        return []
    
    async def get_coin_time_series(self, symbol: str, interval: str = '1h', limit: int = 24) -> Optional[List[Dict]]:
        """Get historical time series data for a coin"""
        coin = symbol.lower().replace('usdt', '')
        
        params = {
            'interval': interval,
            'limit': limit
        }
        
        data = await self._make_request(f"/coins/{coin}/time-series/v2", params)
        if data and 'data' in data:
            return data['data']
        return []
    
    async def get_trending_coins(self, limit: int = 20) -> List[Dict]:
        """Get trending coins based on social activity"""
        coins = await self.get_coins_list()
        if not coins:
            return []
        
        # Sort by Galaxy Score (overall social strength)
        trending = sorted(coins, key=lambda x: x.get('gs', 0), reverse=True)[:limit]
        
        # Format for easy use
        formatted = []
        for coin in trending:
            formatted.append({
                'symbol': coin.get('s', '').upper() + 'USDT',
                'name': coin.get('n', ''),
                'galaxy_score': coin.get('gs', 0),
                'alt_rank': coin.get('ar', 999),
                'social_volume': coin.get('sv', 0),
                'social_contributors': coin.get('sc', 0),
                'price_change_24h': coin.get('pc', 0),
                'market_cap': coin.get('mc', 0)
            })
        
        return formatted
    
    async def get_social_sentiment(self, symbol: str) -> Dict:
        """Get social sentiment analysis for a coin"""
        coin_data = await self.get_coin_social_data(symbol)
        if not coin_data:
            return {'sentiment': 'neutral', 'score': 0}
        
        # Calculate sentiment based on various metrics
        sentiment_score = 0
        
        # Galaxy Score (1-100 scale)
        galaxy_score = coin_data.get('galaxy_score', 50)
        sentiment_score += (galaxy_score - 50) / 100  # Normalize to -0.5 to 0.5
        
        # Social volume change
        social_change = coin_data.get('social_volume_change_24h', 0)
        if social_change > 50:
            sentiment_score += 0.2
        elif social_change > 20:
            sentiment_score += 0.1
        elif social_change < -20:
            sentiment_score -= 0.1
        
        # Price correlation with social
        price_change = coin_data.get('price_change_24h', 0)
        if price_change > 0 and social_change > 0:
            sentiment_score += 0.2  # Positive correlation
        elif price_change < 0 and social_change < 0:
            sentiment_score -= 0.1  # Negative correlation
        
        # Determine sentiment category
        if sentiment_score > 0.3:
            sentiment = 'bullish'
        elif sentiment_score < -0.3:
            sentiment = 'bearish'
        else:
            sentiment = 'neutral'
        
        return {
            'sentiment': sentiment,
            'score': max(-1, min(1, sentiment_score)),
            'galaxy_score': galaxy_score,
            'social_volume': coin_data.get('social_volume', 0),
            'social_change_24h': social_change
        }
    
    async def get_influencer_activity(self, symbol: str) -> List[Dict]:
        """Get top influencers talking about a coin"""
        coin = symbol.lower().replace('usdt', '')
        
        data = await self._make_request(f"/topic/{coin}/creators/v1", {'limit': 10})
        if data and 'data' in data:
            return [{
                'username': creator.get('username', ''),
                'network': creator.get('network', ''),
                'followers': creator.get('followers', 0),
                'engagement': creator.get('engagement_rate', 0),
                'posts': creator.get('posts', 0)
            } for creator in data['data']]
        return []
    
    async def get_social_momentum(self, symbols: List[str]) -> Dict[str, float]:
        """Calculate social momentum scores for multiple symbols"""
        momentum_scores = {}
        
        # Get current snapshot
        coins = await self.get_coins_list()
        if not coins:
            return momentum_scores
        
        # Create lookup
        coin_data = {coin['s'].upper() + 'USDT': coin for coin in coins}
        
        for symbol in symbols:
            if symbol in coin_data:
                coin = coin_data[symbol]
                
                # Calculate momentum based on multiple factors
                momentum = 0
                
                # Galaxy Score momentum
                gs = coin.get('gs', 50)
                if gs > 70:
                    momentum += 0.3
                elif gs > 60:
                    momentum += 0.1
                
                # Social volume surge
                sv_change = coin.get('svt24', 0)  # 24h social volume change
                if sv_change > 100:
                    momentum += 0.4
                elif sv_change > 50:
                    momentum += 0.2
                
                # Alt Rank improvement (lower is better)
                alt_rank = coin.get('ar', 999)
                alt_rank_change = coin.get('arc24', 0)  # 24h change
                if alt_rank_change < -5:  # Improving rank
                    momentum += 0.2
                elif alt_rank_change < -2:
                    momentum += 0.1
                
                # Interaction rate
                if coin.get('social_interactions', 0) > coin.get('social_volume', 1) * 2:
                    momentum += 0.1
                
                momentum_scores[symbol] = min(1.0, momentum)
        
        return momentum_scores
    
    async def get_breaking_news(self, symbol: str = None, limit: int = 10) -> List[Dict]:
        """Get breaking news with social impact"""
        endpoint = f"/topic/{symbol.lower().replace('usdt', '')}/news/v1" if symbol else "/topics/list/v1"
        
        params = {'limit': limit}
        data = await self._make_request(endpoint, params)
        
        if data and 'data' in data:
            news_items = []
            for item in data['data']:
                news_items.append({
                    'title': item.get('title', ''),
                    'url': item.get('url', ''),
                    'source': item.get('source', ''),
                    'social_impact': item.get('social_score', 0),
                    'published': item.get('published_at', ''),
                    'sentiment': item.get('sentiment', 'neutral')
                })
            return news_items
        return []
    
    def _parse_coin_data(self, data: Dict) -> Dict:
        """Parse coin data from API response"""
        if 'data' in data and len(data['data']) > 0:
            coin = data['data'][0]
            return {
                'symbol': coin.get('s', '').upper() + 'USDT',
                'name': coin.get('n', ''),
                'galaxy_score': coin.get('gs', 0),
                'alt_rank': coin.get('ar', 999),
                'price': coin.get('p', 0),
                'price_change_24h': coin.get('pc', 0),
                'market_cap': coin.get('mc', 0),
                'volume_24h': coin.get('v', 0),
                'social_volume': coin.get('sv', 0),
                'social_volume_change_24h': coin.get('svt24', 0),
                'social_contributors': coin.get('sc', 0),
                'social_interactions': coin.get('si', 0),
                'social_dominance': coin.get('sd', 0),
                'correlation_rank': coin.get('cr', 0)
            }
        return {}
    
    async def get_market_metrics(self) -> Dict:
        """Get overall crypto market social metrics"""
        # Use top coins as market proxy
        coins = await self.get_coins_list()
        if not coins:
            return {}
        
        # Calculate aggregated metrics
        top_coins = coins[:50]  # Top 50 by Galaxy Score
        
        avg_galaxy_score = sum(c.get('gs', 0) for c in top_coins) / len(top_coins)
        total_social_volume = sum(c.get('sv', 0) for c in top_coins)
        bullish_count = sum(1 for c in top_coins if c.get('pc', 0) > 2)
        bearish_count = sum(1 for c in top_coins if c.get('pc', 0) < -2)
        
        market_sentiment = 'neutral'
        if bullish_count > bearish_count * 1.5:
            market_sentiment = 'bullish'
        elif bearish_count > bullish_count * 1.5:
            market_sentiment = 'bearish'
        
        return {
            'average_galaxy_score': avg_galaxy_score,
            'total_social_volume': total_social_volume,
            'market_sentiment': market_sentiment,
            'bullish_ratio': bullish_count / len(top_coins),
            'trending_count': sum(1 for c in top_coins if c.get('svt24', 0) > 50)
        }