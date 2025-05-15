import logging
import numpy as np
import pandas as pd
import time
import asyncio
from typing import List, Dict, Any, Tuple
import config

class AssetSelection:
    def __init__(self, binance_client, market_analysis):
        self.binance_client = binance_client
        self.market_analysis = market_analysis
        self.cache = {}
        self.cache_expiry = {}
        
    async def select_optimal_assets(self, limit=15):
        """Select optimal assets to trade based on opportunity scores"""
        try:
            # Get available pairs
            all_pairs = await self.get_available_pairs()
            
            # Get trending pairs for higher weighting
            trending_pairs = await self.get_trending_cryptos(limit=20)
            
            # Score each pair
            pair_scores = []
            
            for pair in all_pairs:
                try:
                    # Base score
                    base_score = 0.5
                    
                    # Bonus for trending pairs
                    if pair in trending_pairs:
                        trending_rank = trending_pairs.index(pair) + 1
                        # Higher bonus for higher ranked trending pairs
                        trending_bonus = max(0.5, (21 - trending_rank) / 20)
                        base_score += trending_bonus
                    
                    # Check market data for volatility and volume
                    # Get 4h data for more stable metrics
                    start_time = int(time.time() * 1000) - (5 * 24 * 60 * 60 * 1000)  # 5 days
                    klines = await self.market_analysis.get_klines(pair, start_time, '4h')
                    
                    if not klines or len(klines) < 10:
                        continue
                        
                    # Extract OHLCV data
                    closes = [float(k[4]) for k in klines]
                    highs = [float(k[2]) for k in klines]
                    lows = [float(k[3]) for k in klines]
                    volumes = [float(k[5]) for k in klines]
                    
                    # 1. Volatility score - we want moderate volatility
                    returns = [(closes[i]/closes[i-1]) - 1 for i in range(1, len(closes))]
                    volatility = np.std(returns) * 100  # Daily volatility as percentage
                    
                    # Optimal volatility between 1-4%
                    if 1.0 < volatility < 4.0:
                        vol_score = 0.5
                    elif 0.5 < volatility < 6.0:  # Still decent volatility
                        vol_score = 0.3
                    else:  # Too low or too high
                        vol_score = 0.1
                        
                    # 2. Volume and liquidity
                    avg_volume = sum(volumes) / len(volumes)
                    latest_volume = volumes[-1]
                    volume_trend = latest_volume / avg_volume
                    
                    # Higher score for increasing volume
                    if volume_trend > 1.5:
                        volume_score = 0.5  # Volume significantly increasing
                    elif volume_trend > 1.0:
                        volume_score = 0.3  # Volume moderately increasing
                    else:
                        volume_score = 0.1  # Volume flat or decreasing
                        
                    # 3. Trend strength and direction
                    ma20 = sum(closes[-20:]) / 20 if len(closes) >= 20 else sum(closes) / len(closes)
                    ma50 = sum(closes[-50:]) / 50 if len(closes) >= 50 else ma20
                    
                    # Recent momentum
                    short_momentum = closes[-1] / closes[-3] - 1 if len(closes) >= 3 else 0
                    
                    # Trend direction
                    if closes[-1] > ma20 > ma50:
                        trend_score = 0.5  # Strong uptrend
                    elif closes[-1] > ma20:
                        trend_score = 0.3  # Moderate uptrend or starting uptrend
                    elif closes[-1] < ma20 < ma50:
                        trend_score = 0.1  # Strong downtrend
                    else:
                        trend_score = 0.2  # Mixed trend
                        
                    # Adjust trend score based on recent momentum
                    if short_momentum > 0.03:  # Strong recent upward momentum
                        trend_score += 0.2
                    elif short_momentum < -0.03:  # Strong recent downward momentum
                        trend_score -= 0.1
                        
                    # 4. Technical pattern score
                    # Check for bullish pattern - price bouncing off support
                    if lows[-1] > lows[-2] > lows[-3] and closes[-1] > closes[-2]:
                        pattern_score = 0.4  # Bullish pattern
                    # Check for bearish pattern - price breaking below support
                    elif highs[-1] < highs[-2] < highs[-3] and closes[-1] < closes[-2]:
                        pattern_score = 0.1  # Bearish pattern
                    else:
                        pattern_score = 0.2  # No clear pattern
                        
                    # Combine scores with weighting
                    final_score = (
                        base_score * 0.3 +     # Base and trending score - 30%
                        vol_score * 0.2 +      # Volatility score - 20%
                        volume_score * 0.15 +  # Volume score - 15%
                        trend_score * 0.25 +   # Trend score - 25%
                        pattern_score * 0.1    # Pattern score - 10%
                    )
                    
                    # Normalize final score to 0-10 scale
                    normalized_score = min(10.0, final_score * 10)
                    
                    pair_scores.append((pair, normalized_score))
                    
                except Exception as e:
                    logging.debug(f"Error scoring {pair}: {str(e)}")
                    continue
                    
            # Sort pairs by score (highest first)
            sorted_pairs = sorted(pair_scores, key=lambda x: x[1], reverse=True)
            
            # Take top N pairs
            top_pairs = [pair for pair, score in sorted_pairs[:limit]]
            
            # Log selection
            logging.info(f"Selected optimal assets: {', '.join(top_pairs[:5])}... ({len(top_pairs)} total)")
            for pair, score in sorted_pairs[:5]:
                logging.info(f"  {pair}: score {score:.2f}")
                
            return top_pairs
            
        except Exception as e:
            logging.error(f"Error selecting optimal assets: {str(e)}")
            # Fallback to trending pairs on error
            return await self.get_trending_cryptos(limit=limit)
    
    async def get_available_pairs(self):
        """Get available trading pairs from Binance"""
        try:
            # Define a whitelist of top cryptocurrencies
            WHITELISTED_CRYPTOS = [
                'BTC', 'ETH', 'XRP', 'SOL', 'BNB', 'DOGE', 'ADA', 'TRX', 'LINK',
                'AVAX', 'SUI', 'XLM', 'TON', 'SHIB', 'HBAR', 'DOT', 'BGB',
                'LTC', 'BCH', 'OM', 'UNI', 'PEPE', 'NEAR', 'XMR', 'APT', 'ETC',
                'MNT', 'ICP', 'TRUMP', 'TAO', 'VET', 'CRO', 'OKB', 'POL', 'ALGO',
                'KAS', 'RENDER', 'FIL', 'ARB', 'FET', 'ATOM', 'THETA', 'BONK',
                'EOS', 'XTZ', 'IOTA', 'NEO', 'EGLD', 'ZEC', 'LAYER'
            ]
            
            # Create trading pairs by adding USDT suffix
            whitelisted_pairs = [f"{crypto}USDT" for crypto in WHITELISTED_CRYPTOS]
            
            return whitelisted_pairs
            
        except Exception as e:
            logging.error(f"Error getting available pairs: {str(e)}")
            # Return default pairs on error
            return ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'ADAUSDT']
    
    async def get_trending_cryptos(self, limit=10):
        """Get the top trending cryptocurrency pairs based on volume and price change"""
        try:
            cache_key = f"trending_cryptos_{limit}"
            current_time = time.time()
            
            if cache_key in self.cache and self.cache_expiry.get(cache_key, 0) > current_time:
                return self.cache[cache_key]
            
            logging.info(f"Fetching top {limit} trending cryptocurrency pairs")
            
            # Get 24hr ticker statistics for all pairs
            tickers = self.binance_client.get_ticker()
            
            # Filter for USDT pairs
            usdt_pairs = [ticker for ticker in tickers if ticker['symbol'].endswith('USDT')]
            
            # First sort by volume (to get high liquidity pairs)
            by_volume = sorted(usdt_pairs, key=lambda x: float(x['quoteVolume']), reverse=True)[:50]
            
            # Then from those high volume pairs, sort by price change to get trending coins
            trending = sorted(by_volume, key=lambda x: abs(float(x['priceChangePercent'])), reverse=True)
            
            # Extract just the symbols
            result = [ticker['symbol'] for ticker in trending[:limit]]
            
            # Cache the result for 1 hour
            self.cache[cache_key] = result
            self.cache_expiry[cache_key] = current_time + 3600
            
            logging.info(f"Found trending pairs: {', '.join(result[:5])}...")
            return result
            
        except Exception as e:
            logging.error(f"Error getting trending cryptocurrencies: {str(e)}")
            # Return default pairs on error
            default_pairs = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'ADAUSDT']
            return default_pairs[:limit]