import logging
import numpy as np
import pandas as pd
import time
import asyncio
from typing import List, Dict, Any, Tuple
import config
from datetime import datetime

class AssetSelection:
    def __init__(self, binance_client, market_analysis):
        self.binance_client = binance_client
        self.market_analysis = market_analysis
        self.cache = {}
        self.cache_expiry = {}
        
    async def select_optimal_assets(self, limit=15):
        """Select optimal assets to trade based on opportunity scores"""
        try:
            # Get trending pairs first
            trending_pairs = await self.get_trending_cryptos(limit=30)
            
            # Get high volume pairs
            tickers = self.binance_client.get_ticker()
            usdt_pairs = [
                t for t in tickers 
                if t['symbol'].endswith('USDT') and float(t['quoteVolume']) > 100000
            ]
            
            # Sort by volume and price change
            volume_sorted = sorted(usdt_pairs, key=lambda x: float(x['quoteVolume']), reverse=True)[:50]
            
            # Combine trending and high volume
            all_pairs = list(set(trending_pairs + [t['symbol'] for t in volume_sorted[:30]]))
            
            # Score each pair with less restrictive criteria
            pair_scores = []
            
            for pair in all_pairs[:50]:  # Analyze top 50
                try:
                    ticker_data = next((t for t in tickers if t['symbol'] == pair), None)
                    if not ticker_data:
                        continue
                    
                    price_change_24h = float(ticker_data['priceChangePercent'])
                    volume_24h = float(ticker_data['quoteVolume'])
                    
                    # Base score
                    score = 3.0  # Start with decent base score
                    
                    # Volume bonus
                    if volume_24h > 10000000:
                        score += 2.0
                    elif volume_24h > 5000000:
                        score += 1.5
                    elif volume_24h > 1000000:
                        score += 1.0
                    
                    # Price change bonus (both up and down can be opportunities)
                    if 0 < price_change_24h < 5:  # Moderate upward movement
                        score += 1.5
                    elif -3 < price_change_24h < 0:  # Small dip (potential bounce)
                        score += 1.0
                    elif 5 < price_change_24h < 10:  # Strong but not overextended
                        score += 1.0
                    
                    # Trending bonus
                    if pair in trending_pairs[:10]:
                        score += 1.0
                    
                    pair_scores.append((pair, score))
                    
                except Exception as e:
                    logging.debug(f"Error scoring {pair}: {str(e)}")
                    continue
            
            # Sort by score
            sorted_pairs = sorted(pair_scores, key=lambda x: x[1], reverse=True)
            
            # Return top pairs
            selected_pairs = [pair for pair, score in sorted_pairs[:limit]]
            
            # Always ensure we have some pairs to analyze
            if len(selected_pairs) < 5:
                # Add some default high-volume pairs
                defaults = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT', 'XRPUSDT']
                for default in defaults:
                    if default not in selected_pairs:
                        selected_pairs.append(default)
                    if len(selected_pairs) >= limit:
                        break
            
            logging.info(f"Selected optimal assets: {', '.join(selected_pairs[:5])}... ({len(selected_pairs)} total)")
            
            return selected_pairs
            
        except Exception as e:
            logging.error(f"Error selecting optimal assets: {str(e)}")
            # Return default pairs on error
            return ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT', 'XRPUSDT']
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