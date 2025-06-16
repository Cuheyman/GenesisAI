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
        """Select optimal assets to trade based on opportunity scores with better rotation"""
        try:
            # Get ALL available pairs from market, not just whitelist
            tickers = self.binance_client.get_ticker()
            
            # Filter for USDT pairs with decent volume
            usdt_pairs = [
                t for t in tickers 
                if t['symbol'].endswith('USDT') and float(t['quoteVolume']) > 100000
            ]
            
            # Sort by different criteria to ensure variety
            volume_sorted = sorted(usdt_pairs, key=lambda x: float(x['quoteVolume']), reverse=True)[:50]
            change_sorted = sorted(usdt_pairs, key=lambda x: abs(float(x['priceChangePercent'])), reverse=True)[:50]
            
            # Get trending pairs
            trending_pairs = await self.get_trending_cryptos(limit=30)
            
            # Combine all sources for variety
            all_pairs = list(set(
                [t['symbol'] for t in volume_sorted] + 
                [t['symbol'] for t in change_sorted] + 
                trending_pairs
            ))
            
            # Add time-based rotation to prevent selecting same assets
            current_hour = datetime.now().hour
            rotation_offset = current_hour % len(all_pairs) if all_pairs else 0
            all_pairs = all_pairs[rotation_offset:] + all_pairs[:rotation_offset]
            
            # Score each pair
            pair_scores = []
            
            for pair in all_pairs[:100]:  # Analyze top 100 candidates
                try:
                    # Skip if recently traded (prevent buying same asset repeatedly)
                    if hasattr(self, 'recently_traded') and pair in self.recently_traded:
                        if time.time() - self.recently_traded[pair] < 3600:  # Skip if traded within 1 hour
                            continue
                    
                    # Base score with randomization for variety
                    base_score = 0.5 + (random.random() * 0.2)  # Add 0-0.2 random factor
                    
                    # Get recent price action (use shorter timeframe for momentum)
                    ticker_data = next((t for t in tickers if t['symbol'] == pair), None)
                    if not ticker_data:
                        continue
                    
                    price_change_24h = float(ticker_data['priceChangePercent'])
                    volume_24h = float(ticker_data['quoteVolume'])
                    
                    # MOMENTUM SCORING - favor recent movers
                    if 0.5 < price_change_24h < 8:  # Rising but not overextended
                        momentum_score = 0.8
                    elif -2 < price_change_24h < 0.5:  # Slight dip, potential bounce
                        momentum_score = 0.6
                    elif price_change_24h > 8:  # Overextended
                        momentum_score = 0.2
                    else:
                        momentum_score = 0.3
                    
                    # Volume scoring - higher volume = more opportunity
                    if volume_24h > 10000000:  # Over 10M
                        volume_score = 0.8
                    elif volume_24h > 5000000:  # Over 5M
                        volume_score = 0.6
                    elif volume_24h > 1000000:  # Over 1M
                        volume_score = 0.4
                    else:
                        volume_score = 0.2
                    
                    # Get 5m and 15m data for short-term momentum
                    start_time = int(time.time() * 1000) - (2 * 60 * 60 * 1000)  # 2 hours
                    klines_5m = await self.market_analysis.get_klines(pair, start_time, '5m')
                    
                    if klines_5m and len(klines_5m) >= 12:
                        recent_closes = [float(k[4]) for k in klines_5m[-12:]]
                        
                        # Very short-term momentum (last hour)
                        short_momentum = (recent_closes[-1] / recent_closes[0] - 1) * 100
                        
                        # Micro momentum (last 15 minutes)
                        micro_momentum = (recent_closes[-1] / recent_closes[-3] - 1) * 100
                        
                        # Score based on momentum
                        if 0.2 < short_momentum < 3:  # Positive momentum but not crazy
                            trend_score = 0.7
                        elif 0 < micro_momentum < 1:  # Just starting to move
                            trend_score = 0.6
                        else:
                            trend_score = 0.3
                    else:
                        trend_score = 0.3
                    
                    # Bonus for trending pairs
                    if pair in trending_pairs:
                        trending_rank = trending_pairs.index(pair) + 1
                        trending_bonus = max(0.3, (31 - trending_rank) / 30)
                    else:
                        trending_bonus = 0
                    
                    # Calculate final score with adjusted weights
                    final_score = (
                        base_score * 0.15 +        # Base (with randomization)
                        momentum_score * 0.35 +    # Recent price action - 35%
                        volume_score * 0.20 +      # Volume - 20%
                        trend_score * 0.20 +       # Short-term trend - 20%
                        trending_bonus * 0.10      # Trending bonus - 10%
                    )
                    
                    # Apply penalty for coins that moved too much already
                    if abs(price_change_24h) > 15:
                        final_score *= 0.5  # Halve score for overextended coins
                    
                    # Normalize final score to 0-10 scale
                    normalized_score = min(10.0, final_score * 10)
                    
                    pair_scores.append((pair, normalized_score))
                    
                except Exception as e:
                    logging.debug(f"Error scoring {pair}: {str(e)}")
                    continue
            
            # Sort pairs by score (highest first)
            sorted_pairs = sorted(pair_scores, key=lambda x: x[1], reverse=True)
            
            # Ensure diversity - don't return too many similar assets
            selected_pairs = []
            selected_bases = set()
            
            for pair, score in sorted_pairs:
                base = pair.replace('USDT', '')
                # Skip if we already have a similar asset (e.g., avoid both BTCUSDT and WBTCUSDT)
                if base not in selected_bases and not any(base.startswith(s) or s.startswith(base) for s in selected_bases):
                    selected_pairs.append(pair)
                    selected_bases.add(base)
                    if len(selected_pairs) >= limit:
                        break
            
            # Log selection
            logging.info(f"Selected optimal assets: {', '.join(selected_pairs[:5])}... ({len(selected_pairs)} total)")
            for i, pair in enumerate(selected_pairs[:5]):
                score = next((s for p, s in sorted_pairs if p == pair), 0)
                logging.info(f"  {pair}: score {score:.2f}")
            
            return selected_pairs
            
        except Exception as e:
            logging.error(f"Error selecting optimal assets: {str(e)}")
            # Fallback to trending pairs on error
            return await self.get_trending_cryptos(limit=limit)
    
    async def get_available_pairs(self):
        """Get available trading pairs from Binance"""
        try:
            # Define a whitelist of top cryptocurrencies
         # In asset_selection.py, update the WHITELISTED_CRYPTOS list
            WHITELISTED_CRYPTOS = [
                'BTC', 'ETH', 'XRP', 'SOL', 'BNB', 'DOGE', 'ADA', 'TRX', 'LINK',
                'AVAX', 'SUI', 'XLM', 'TON', 'SHIB', 'HBAR', 'DOT', 
                'LTC', 'BCH', 'OM', 'UNI', 'PEPE', 'NEAR', 'APT', 'ETC',
                'ICP', 'VET', 'POL', 'ALGO',
                'RENDER', 'FIL', 'ARB', 'FET', 'ATOM', 'THETA', 'BONK',
                'XTZ', 'IOTA', 'NEO', 'EGLD', 'ZEC', 'LAYER'
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