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
        
    async def select_optimal_assets(self, max_pairs: int = 3) -> List[str]:
        """
        Select optimal trading pairs - LIMITED TO TOP 3 PAIRS
        This reduces API requests and focuses on quality trades
        """
        try:
            # Get valid symbols first
            valid_symbols = await self.get_valid_symbols()
            if not valid_symbols:
                logging.warning("No valid symbols available, using fallback list")
                return ["BTCUSDT", "ETHUSDT", "BNBUSDT"][:max_pairs]
            
            # Get trending cryptos (limited selection)
            trending = await self.get_trending_cryptos()
            if not trending:
                logging.warning("No trending cryptos available")
                return list(valid_symbols)[:max_pairs]
            
            # Filter to only valid symbols and limit to max_pairs
            valid_trending = []
            for symbol in trending:
                if symbol in valid_symbols:
                    valid_trending.append(symbol)
                    if len(valid_trending) >= max_pairs:
                        break
            
            # If we don't have enough trending, add some major pairs
            if len(valid_trending) < max_pairs:
                major_pairs = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "ADAUSDT", "SOLUSDT"]
                for pair in major_pairs:
                    if pair in valid_symbols and pair not in valid_trending:
                        valid_trending.append(pair)
                        if len(valid_trending) >= max_pairs:
                            break
            
            logging.info(f"Selected {len(valid_trending)} pairs for trading: {valid_trending}")
            return valid_trending[:max_pairs]
            
        except Exception as e:
            logging.error(f"Error in select_optimal_assets: {str(e)}")
            # Fallback to major pairs only
            return ["BTCUSDT", "ETHUSDT", "BNBUSDT"][:max_pairs]

    def mark_recently_traded(self, pair):
        """Mark a pair as recently traded to avoid immediate re-trading"""
        if not hasattr(self, 'recently_traded'):
            self.recently_traded = {}
        
        self.recently_traded[pair] = time.time()
        
        # Clean up old entries (older than 1 hour)
        current_time = time.time()
        self.recently_traded = {
            p: t for p, t in self.recently_traded.items() 
            if current_time - t < 3600  # Keep for 1 hour
        }

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
    
    async def get_valid_symbols(self):
        """Get valid USDT trading pairs from Binance (API symbols endpoint not available)"""
        try:
            # Get all USDT pairs from Binance
            tickers = self.binance_client.get_ticker()
            usdt_pairs = set([t['symbol'] for t in tickers if t['symbol'].endswith('USDT')])
            
            # Filter to only major pairs with good volume
            major_pairs = {
                'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'ADAUSDT', 
                'XRPUSDT', 'DOTUSDT', 'LINKUSDT', 'AVAXUSDT', 'UNIUSDT',
                'LTCUSDT', 'BCHUSDT', 'ATOMUSDT', 'ETCUSDT', 'XLMUSDT',
                'VETUSDT', 'FILUSDT', 'TRXUSDT', 'ICPUSDT', 'NEARUSDT'
            }
            
            # Return intersection of available pairs and major pairs
            valid_pairs = usdt_pairs.intersection(major_pairs)
            
            if not valid_pairs:
                logging.warning("No major pairs found, using default list")
                return major_pairs
            
            logging.info(f"Found {len(valid_pairs)} valid trading pairs")
            return valid_pairs
            
        except Exception as e:
            logging.error(f"Error getting valid symbols: {str(e)}")
            # Fallback to default major pairs
            return {
                'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'ADAUSDT',
                'XRPUSDT', 'DOTUSDT', 'LINKUSDT', 'AVAXUSDT', 'UNIUSDT'
            }

    async def get_trending_cryptos(self, limit=10):
        """Get the top trending cryptocurrency pairs based on volume and price change, filtered by valid symbols"""
        try:
            cache_key = f"trending_cryptos_{limit}"
            current_time = time.time()
            if cache_key in self.cache and self.cache_expiry.get(cache_key, 0) > current_time:
                return self.cache[cache_key]
            logging.info(f"Fetching top {limit} trending cryptocurrency pairs")
            tickers = self.binance_client.get_ticker()
            usdt_pairs = [ticker for ticker in tickers if ticker['symbol'].endswith('USDT')]
            by_volume = sorted(usdt_pairs, key=lambda x: float(x['quoteVolume']), reverse=True)[:50]
            trending = sorted(by_volume, key=lambda x: abs(float(x['priceChangePercent'])), reverse=True)
            result = [ticker['symbol'] for ticker in trending[:limit]]
            # Filter by valid symbols
            valid_symbols = await self.get_valid_symbols()
            filtered = [s for s in result if s in valid_symbols]
            self.cache[cache_key] = filtered
            self.cache_expiry[cache_key] = current_time + 3600
            logging.info(f"Found trending pairs: {', '.join(filtered[:5])}...")
            return filtered
        except Exception as e:
            logging.error(f"Error getting trending cryptocurrencies: {str(e)}")
            default_pairs = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'ADAUSDT']
            return default_pairs[:limit]