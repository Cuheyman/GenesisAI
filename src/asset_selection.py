import logging
import numpy as np
import pandas as pd
import time
import asyncio
from typing import List, Dict, Any, Tuple
import config
from datetime import datetime, timedelta
import statistics

class DynamicAssetSelection:
    def __init__(self, binance_client, market_analysis):
        self.binance_client = binance_client
        self.market_analysis = market_analysis
        self.cache = {}
        self.cache_expiry = {}
        self.excluded_pairs = set()  # Pairs to temporarily exclude
        self.performance_history = {}  # Track pair performance
        self.recently_traded = {}
        
    async def get_optimal_trading_pairs(self, max_pairs: int = 15, exclude_traded_pairs: set = None) -> List[str]:
        """
        FULLY DYNAMIC: Scan ALL available pairs and select best opportunities
        No hardcoded lists - pure market-driven selection
        """
        try:
            logging.info("Scanning ALL available pairs for optimal opportunities...")
            
            # Get ALL available USDT pairs from Binance
            all_pairs = await self._get_all_usdt_pairs()
            if not all_pairs:
                return self._get_emergency_fallback()
            
            logging.info(f"Found {len(all_pairs)} total USDT pairs available")
            
            # Apply realistic baseline filters
            filtered_pairs = await self._apply_baseline_filters(all_pairs)
            logging.info(f"After baseline filtering: {len(filtered_pairs)} pairs remain")
            
            # CRITICAL FIX: Filter out already traded pairs
            if exclude_traded_pairs:
                original_count = len(filtered_pairs)
                filtered_pairs = [pair for pair in filtered_pairs if pair not in exclude_traded_pairs]
                excluded_count = original_count - len(filtered_pairs)
                if excluded_count > 0:
                    logging.info(f"Excluded {excluded_count} already traded pairs: {list(exclude_traded_pairs)[:5]}...")
            
            # SAFETY: Validate symbols for live trading safety
            if hasattr(self, 'validate_symbols') and self.validate_symbols:
                safe_pairs = await self._validate_symbols_for_trading(filtered_pairs)
                logging.info(f"After symbol validation: {len(safe_pairs)} pairs remain")
                filtered_pairs = safe_pairs
            
            if len(filtered_pairs) < max_pairs:
                logging.warning(f"Only {len(filtered_pairs)} pairs passed filters, using all")
                return filtered_pairs
            
            # Score each pair based on multiple market factors
            scored_pairs = await self._score_market_opportunities(filtered_pairs)
            
            # Select diverse, high-scoring pairs
            selected_pairs = await self._select_diverse_opportunities(scored_pairs, max_pairs)
            
            # Cache the scored pairs data for later retrieval of scores
            self.cached_scores = {}
            self.cached_scores_time = time.time()
            for pair_data in scored_pairs:
                self.cached_scores[pair_data['symbol']] = pair_data['total_score']
            
            # Log selection results
            logging.info(f"Selected {len(selected_pairs)} optimal pairs:")
            for i, pair_data in enumerate(selected_pairs[:10]):  # Log top 10
                logging.info(f"  {i+1}. {pair_data['symbol']} (Score: {pair_data['total_score']:.2f}, "
                           f"Volume: ${pair_data['volume_usdt']:,.0f}, Change: {pair_data['price_change_pct']:+.2f}%)")
            
            # Return just the symbols
            return [pair_data['symbol'] for pair_data in selected_pairs]
            
        except Exception as e:
            logging.error(f"Error in dynamic asset selection: {str(e)}")
            return self._get_emergency_fallback()
    
    async def _get_all_usdt_pairs(self) -> List[str]:
        """Get ALL USDT trading pairs from Binance exchange"""
        try:
            # Get all tickers
            tickers = self.binance_client.get_ticker()
            
            # Filter to USDT pairs only
            usdt_pairs = []
            for ticker in tickers:
                symbol = ticker['symbol']
                if symbol.endswith('USDT'):
                    usdt_pairs.append(symbol)
            
            # Remove known problematic patterns (leveraged tokens, etc)
            excluded_patterns = [
                'BEAR', 'BULL', 'DOWN', 'UP', 'LONG', 'SHORT',  # Leveraged tokens
                '3L', '3S', '5L', '5S',  # Leveraged tokens
                'FDUSD', 'TUSD', 'BUSD',  # Other stablecoins as base
            ]
            
            filtered_pairs = []
            for pair in usdt_pairs:
                if not any(pattern in pair for pattern in excluded_patterns):
                    filtered_pairs.append(pair)
            
            return filtered_pairs
            
        except Exception as e:
            logging.error(f"Error getting all USDT pairs: {str(e)}")
            return []
    
    async def _apply_baseline_filters(self, pairs: List[str]) -> List[str]:
        """Apply realistic baseline filters for safety and liquidity"""
        try:
            # Get 24h ticker data for all pairs
            tickers = self.binance_client.get_ticker()
            ticker_dict = {t['symbol']: t for t in tickers}
            
            filtered_pairs = []
            
            for pair in pairs:
                ticker = ticker_dict.get(pair)
                if not ticker:
                    continue
                
                try:
                    # Extract key metrics
                    volume_usdt = float(ticker['quoteVolume'])
                    price = float(ticker['lastPrice'])
                    price_change_pct = float(ticker['priceChangePercent'])
                    trade_count = int(ticker['count']) if 'count' in ticker else 0
                    
                    # REALISTIC LIQUIDITY FILTERS:
                    
                    # 1. Minimum volume (realistic threshold)
                    if volume_usdt < 500000:  # $500K daily volume minimum
                        continue
                    
                    # 2. Minimum price (avoid dust tokens)
                    if price < 0.000001:  # Must be > 0.000001 USDT
                        continue
                    
                    # 3. Minimum trade activity 
                    if trade_count < 1000:  # At least 1000 trades in 24h
                        continue
                    
                    # 4. Exclude extreme movers (likely manipulation)
                    if abs(price_change_pct) > 50:  # No more than 50% daily move
                        continue
                    
                    # 5. Must have some volatility (not completely dead)
                    if abs(price_change_pct) < 0.1:  # At least 0.1% movement
                        continue
                    
                    filtered_pairs.append(pair)
                    
                except (ValueError, TypeError) as e:
                    logging.debug(f"Error processing {pair}: {str(e)}")
                    continue
            
            return filtered_pairs
            
        except Exception as e:
            logging.error(f"Error applying baseline filters: {str(e)}")
            return pairs  # Return unfiltered if error
    
    async def _validate_symbols_for_trading(self, pairs: List[str]) -> List[str]:
        """Validate multiple symbols for live trading safety"""
        try:
            # Skip validation if not configured
            if config.TEST_MODE and not getattr(config, 'VALIDATE_SYMBOLS_IN_TEST', False):
                return pairs
            
            api_key = getattr(config, 'API_KEY', '')
            if not api_key:
                logging.info("No API key configured, skipping symbol validation")
                return pairs
            
            # Validate symbols in batches for efficiency
            validated_pairs = []
            batch_size = 10  # Process 10 symbols at a time
            
            import aiohttp
            import asyncio
            
            api_base_url = getattr(config, 'API_BASE_URL', 'http://localhost:3001')
            
            async def validate_single_symbol(session, symbol):
                try:
                    url = f"{api_base_url}/api/v1/validate-symbol/{symbol}"
                    headers = {
                        'Authorization': f'Bearer {api_key}',
                        'Content-Type': 'application/json'
                    }
                    
                    async with session.get(url, headers=headers) as response:
                        if response.status == 200:
                            data = await response.json()
                            is_safe = data.get('safe_for_live_trading', False)
                            return symbol if is_safe else None
                        else:
                            logging.debug(f"Validation failed for {symbol}: HTTP {response.status}")
                            return None  # Exclude invalid symbols
                            
                except Exception as e:
                    logging.debug(f"Error validating {symbol}: {str(e)}")
                    return symbol  # Include on error (fail safe)
            
            # Process in batches to avoid overwhelming the API
            timeout = aiohttp.ClientTimeout(total=30)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                for i in range(0, len(pairs), batch_size):
                    batch = pairs[i:i + batch_size]
                    
                    # Create tasks for this batch
                    tasks = [validate_single_symbol(session, symbol) for symbol in batch]
                    
                    # Wait for batch to complete
                    results = await asyncio.gather(*tasks, return_exceptions=True)
                    
                    # Collect valid symbols
                    for result in results:
                        if isinstance(result, str):  # Valid symbol
                            validated_pairs.append(result)
                        elif result is None:  # Invalid symbol
                            continue
                        else:  # Exception occurred
                            logging.debug(f"Exception in symbol validation: {result}")
                    
                    # Small delay between batches
                    if i + batch_size < len(pairs):
                        await asyncio.sleep(0.5)
            
            logging.info(f"Symbol validation: {len(validated_pairs)}/{len(pairs)} symbols passed safety checks")
            return validated_pairs
            
        except Exception as e:
            logging.error(f"Error in bulk symbol validation: {str(e)}")
            return pairs  # Return all pairs on error
    
    async def _score_market_opportunities(self, pairs: List[str]) -> List[Dict]:
        """Score pairs based on multiple market factors"""
        try:
            # Get ticker data
            tickers = self.binance_client.get_ticker()
            ticker_dict = {t['symbol']: t for t in tickers}
            
            scored_pairs = []
            
            for pair in pairs:
                ticker = ticker_dict.get(pair)
                if not ticker:
                    continue
                
                try:
                    # Extract metrics
                    volume_usdt = float(ticker['quoteVolume'])
                    price_change_pct = float(ticker['priceChangePercent'])
                    price = float(ticker['lastPrice'])
                    high_24h = float(ticker['highPrice'])
                    low_24h = float(ticker['lowPrice'])
                    
                    # SCORING FACTORS:
                    
                    # 1. Volume Score (higher = better liquidity)
                    volume_score = min(volume_usdt / 10_000_000, 10)  # Cap at 10M
                    
                    # 2. Volatility Score (moderate volatility preferred)
                    volatility = abs(price_change_pct)
                    if volatility < 1:
                        volatility_score = volatility * 2  # Reward small moves
                    elif volatility < 5:
                        volatility_score = 5  # Optimal range
                    elif volatility < 15:
                        volatility_score = 15 - volatility  # Declining reward
                    else:
                        volatility_score = 0  # Too volatile
                    
                    # 3. Momentum Score (trending up gets bonus)
                    momentum_score = max(0, price_change_pct / 5)  # +1 for every 5%
                    momentum_score = min(momentum_score, 3)  # Cap at +3
                    
                    # 4. Range Position Score (where in 24h range)
                    if high_24h > low_24h:
                        range_position = (price - low_24h) / (high_24h - low_24h)
                        # Prefer pairs in middle of range or breaking out
                        if range_position > 0.8:
                            range_score = 3  # Near highs (breakout)
                        elif range_position < 0.2:
                            range_score = 2  # Near lows (reversal opportunity)
                        else:
                            range_score = 1  # Middle range
                    else:
                        range_score = 1
                    
                    # 5. Market Cap Category Bonus (estimate based on price)
                    if price > 100:  # High-price coins (likely large cap)
                        mcap_bonus = 2
                    elif price > 1:  # Mid-price coins
                        mcap_bonus = 1.5
                    elif price > 0.01:  # Lower-price coins
                        mcap_bonus = 1
                    else:  # Very low price (high risk)
                        mcap_bonus = 0.5
                    
                    # Calculate total score
                    total_score = (
                        volume_score * 0.3 +     # 30% weight to liquidity
                        volatility_score * 0.25 + # 25% weight to volatility
                        momentum_score * 0.2 +    # 20% weight to momentum
                        range_score * 0.15 +      # 15% weight to range position
                        mcap_bonus * 0.1          # 10% weight to market cap
                    )
                    
                    scored_pairs.append({
                        'symbol': pair,
                        'total_score': total_score,
                        'volume_usdt': volume_usdt,
                        'price_change_pct': price_change_pct,
                        'volatility': volatility,
                        'price': price,
                        'volume_score': volume_score,
                        'volatility_score': volatility_score,
                        'momentum_score': momentum_score,
                        'range_score': range_score,
                        'mcap_bonus': mcap_bonus
                    })
                    
                except (ValueError, TypeError) as e:
                    logging.debug(f"Error scoring {pair}: {str(e)}")
                    continue
            
            # Sort by total score
            scored_pairs.sort(key=lambda x: x['total_score'], reverse=True)
            
            return scored_pairs
            
        except Exception as e:
            logging.error(f"Error scoring opportunities: {str(e)}")
            return []
    
    async def _select_diverse_opportunities(self, scored_pairs: List[Dict], max_pairs: int) -> List[Dict]:
        """Select diverse opportunities avoiding over-concentration"""
        try:
            if len(scored_pairs) <= max_pairs:
                return scored_pairs
            
            selected = []
            used_base_assets = set()
            
            # Always include top 3 scoring pairs
            for pair_data in scored_pairs[:3]:
                selected.append(pair_data)
                base_asset = pair_data['symbol'].replace('USDT', '')
                used_base_assets.add(base_asset)
            
            # Fill remaining slots with diverse selections
            for pair_data in scored_pairs[3:]:
                if len(selected) >= max_pairs:
                    break
                
                symbol = pair_data['symbol']
                base_asset = symbol.replace('USDT', '')
                
                # Avoid similar assets (basic diversification)
                similar_assets = {
                    'BTC': {'WBTC', 'BTCB'},
                    'ETH': {'WETH', 'BETH', 'ETH2'},
                    'BNB': {'WBNB'},
                    'DOGE': {'SHIB', 'FLOKI', 'PEPE'},  # Meme coins
                    'USDC': {'BUSD', 'TUSD', 'USDD'},  # Stablecoins
                }
                
                # Check for conflicts
                conflict = False
                for used_asset in used_base_assets:
                    if (used_asset == base_asset or 
                        any(base_asset in group and used_asset in group for group in similar_assets.values())):
                        conflict = True
                        break
                
                if not conflict:
                    selected.append(pair_data)
                    used_base_assets.add(base_asset)
            
            # If we still need more pairs, add highest scoring regardless of diversity
            while len(selected) < max_pairs and len(selected) < len(scored_pairs):
                remaining = [p for p in scored_pairs if p not in selected]
                if remaining:
                    selected.append(remaining[0])
                else:
                    break
            
            return selected
            
        except Exception as e:
            logging.error(f"Error selecting diverse opportunities: {str(e)}")
            return scored_pairs[:max_pairs]
    
    def mark_recently_traded(self, pair: str):
        if not hasattr(self, 'recently_traded'):
            self.recently_traded = {}
        self.recently_traded[pair] = time.time()
        logging.info(f"Marked {pair} as recently traded")
    
    def _get_emergency_fallback(self) -> List[str]:
        """Emergency fallback to major pairs if all else fails"""
        return [
            'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'XRPUSDT',
            'ADAUSDT', 'DOGEUSDT', 'AVAXUSDT', 'LINKUSDT', 'UNIUSDT'
        ]
    
    # Legacy methods for backward compatibility
    async def get_available_pairs(self, exclude_traded_pairs: set = None):
        """Legacy method - now redirects to dynamic selection"""
        return await self.get_optimal_trading_pairs(max_pairs=20, exclude_traded_pairs=exclude_traded_pairs)
    
    async def get_trending_cryptos(self, limit=10, exclude_traded_pairs: set = None):
        """Legacy method - now redirects to dynamic selection"""
        pairs = await self.get_optimal_trading_pairs(max_pairs=limit, exclude_traded_pairs=exclude_traded_pairs)
        return pairs[:limit]
    
    async def get_valid_symbols(self, exclude_traded_pairs: set = None):
        """Legacy method - now returns all viable pairs"""
        return set(await self.get_optimal_trading_pairs(max_pairs=50, exclude_traded_pairs=exclude_traded_pairs))
    
    async def select_optimal_assets(self, max_pairs: int = 15, exclude_traded_pairs: set = None) -> List[str]:
        """Legacy method - redirects to new dynamic selection"""
        return await self.get_optimal_trading_pairs(max_pairs, exclude_traded_pairs)

# For backward compatibility
AssetSelection = DynamicAssetSelection