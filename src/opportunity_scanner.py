# opportunity_scanner.py
import asyncio
import aiohttp
import logging
import time
from typing import Dict, List, Any
import numpy as np
from datetime import datetime, timedelta
import requests
import json

class OpportunityScanner:
    def __init__(self, binance_client, coingecko_client=None):
        self.binance_client = binance_client
        self.coingecko_client = coingecko_client  # Optional - can be None
        self.opportunities = {}
        self.last_scan_time = 0
        self.scanned_pairs_cache = {}
        self.cache_duration = 60  # 1 minute cache for scanned pairs
        
    async def get_valid_symbols(self):
        """Get valid USDT trading pairs from Binance (cached for 1 hour)"""
        if hasattr(self, '_valid_symbols') and self._valid_symbols_time > time.time() - 3600:
            return self._valid_symbols
        
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
                valid_pairs = major_pairs
            
            # Cache the result
            self._valid_symbols = valid_pairs
            self._valid_symbols_time = time.time()
            
            logging.info(f"Cached {len(valid_pairs)} valid trading pairs")
            return valid_pairs
            
        except Exception as e:
            logging.error(f"Error getting valid symbols: {str(e)}")
            # Fallback to default major pairs
            fallback_pairs = {
                'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'ADAUSDT',
                'XRPUSDT', 'DOTUSDT', 'LINKUSDT', 'AVAXUSDT', 'UNIUSDT'
            }
            self._valid_symbols = fallback_pairs
            self._valid_symbols_time = time.time()
            return fallback_pairs

    async def scan_for_opportunities(self, max_opportunities: int = 3) -> List[Dict[str, Any]]:
        """
        Scan for trading opportunities - LIMITED TO TOP 3 OPPORTUNITIES
        This reduces API requests and focuses on quality trades
        """
        try:
            # Get valid symbols first
            valid_symbols = await self.get_valid_symbols()
            if not valid_symbols:
                logging.warning("No valid symbols available for opportunity scanning")
                return []
            
            opportunities = []
            
            # Scan for volume surges (limited)
            volume_opps = await self.detect_volume_surges()
            for opp in volume_opps:
                if opp['symbol'] in valid_symbols:
                    opportunities.append(opp)
                    if len(opportunities) >= max_opportunities:
                        break
            
            # If we need more opportunities, scan for momentum shifts
            if len(opportunities) < max_opportunities:
                momentum_opps = await self.detect_momentum_shift()
                for opp in momentum_opps:
                    if opp['symbol'] in valid_symbols and opp['symbol'] not in [o['symbol'] for o in opportunities]:
                        opportunities.append(opp)
                        if len(opportunities) >= max_opportunities:
                            break
            
            # If we still need more, scan for breakout patterns
            if len(opportunities) < max_opportunities:
                breakout_opps = await self.detect_breakout_patterns()
                for opp in breakout_opps:
                    if opp['symbol'] in valid_symbols and opp['symbol'] not in [o['symbol'] for o in opportunities]:
                        opportunities.append(opp)
                        if len(opportunities) >= max_opportunities:
                            break
            
            # Sort by score and return top opportunities
            opportunities.sort(key=lambda x: x.get('score', 0), reverse=True)
            top_opportunities = opportunities[:max_opportunities]
            
            logging.info(f"Found {len(top_opportunities)} opportunities: {[o['symbol'] for o in top_opportunities]}")
            return top_opportunities
            
        except Exception as e:
            logging.error(f"Error scanning for opportunities: {str(e)}")
            return []
    
    async def detect_volume_surges(self) -> List[Dict]:
        """Detect sudden volume increases that often precede price moves"""
        try:
            opportunities = []
            
            # Get 24hr ticker data
            tickers = self.binance_client.get_ticker()
            
            # Sort by volume to focus on liquid pairs
            sorted_tickers = sorted(tickers, key=lambda x: float(x['quoteVolume']), reverse=True)[:100]
            
            for ticker in sorted_tickers:
                symbol = ticker['symbol']
                if not symbol.endswith('USDT'):
                    continue
                
                # Skip if recently scanned
                if self._is_recently_scanned(symbol):
                    continue
                
                try:
                    volume_24h = float(ticker['volume'])
                    volume_quote = float(ticker['quoteVolume'])
                    count = float(ticker['count'])  # Number of trades
                    
                    # Skip if volume too low (less than $100k)
                    if volume_quote < 100000:
                        continue
                    
                    # Get 5-minute candles to detect recent surge
                    klines = self.binance_client.get_klines(
                        symbol=symbol,
                        interval='5m',
                        limit=24  # Last 2 hours
                    )
                    
                    if len(klines) < 24:
                        continue
                    
                    # Calculate volume surge metrics
                    volumes = [float(k[5]) for k in klines]
                    prices = [float(k[4]) for k in klines]
                    
                    # Recent vs average volume
                    recent_volume = np.mean(volumes[-3:])  # Last 15 min
                    older_volume = np.mean(volumes[:-3])   # Previous period
                    
                    if older_volume > 0:
                        volume_surge = recent_volume / older_volume
                        
                        # Price momentum check
                        price_change = (prices[-1] / prices[-6] - 1) * 100  # 30 min change
                        
                        # Look for 2x+ volume surge with positive price action
                        if volume_surge > 2.0 and price_change > -1:
                            # Additional check: not already pumped too much
                            price_change_24h = float(ticker['priceChangePercent'])
                            
                            if price_change_24h < 15:  # Not pumped over 15% yet
                                score = min(volume_surge / 2, 3.0)  # Cap at 3.0
                                
                                # Boost score for very fresh surges
                                if volume_surge > 5.0:
                                    score *= 1.5
                                
                                opportunities.append({
                                    'symbol': symbol,
                                    'type': 'volume_surge',
                                    'score': score,
                                    'data': {
                                        'volume_surge': volume_surge,
                                        'recent_volume': recent_volume,
                                        'price_change_30m': price_change,
                                        'price_change_24h': price_change_24h,
                                        'volume_24h_usd': volume_quote
                                    }
                                })
                                
                except Exception as e:
                    logging.debug(f"Error processing {symbol}: {str(e)}")
                    continue
            
            return opportunities
            
        except Exception as e:
            logging.error(f"Error detecting volume surges: {str(e)}")
            return []
    
    async def detect_momentum_shift(self) -> List[Dict]:
        """Detect quick momentum shifts that signal entry points"""
        try:
            opportunities = []
            
            # Get trending pairs from multiple timeframes
            tickers = self.binance_client.get_ticker()
            
            # Filter for decent volume
            active_pairs = [t for t in tickers if float(t['quoteVolume']) > 500000 and t['symbol'].endswith('USDT')]
            
            for ticker in active_pairs[:50]:  # Check top 50 by volume
                symbol = ticker['symbol']
                
                if self._is_recently_scanned(symbol):
                    continue
                
                try:
                    # Get 1-minute candles for micro momentum
                    klines_1m = self.binance_client.get_klines(
                        symbol=symbol,
                        interval='1m',
                        limit=30
                    )
                    
                    if len(klines_1m) < 30:
                        continue
                    
                    # Extract data
                    closes_1m = [float(k[4]) for k in klines_1m]
                    volumes_1m = [float(k[5]) for k in klines_1m]
                    
                    # Calculate momentum indicators
                    momentum_5m = (closes_1m[-1] / closes_1m[-5] - 1) * 100
                    momentum_15m = (closes_1m[-1] / closes_1m[-15] - 1) * 100
                    
                    # Volume acceleration
                    recent_vol = np.mean(volumes_1m[-5:])
                    older_vol = np.mean(volumes_1m[-30:-5])
                    vol_acceleration = recent_vol / older_vol if older_vol > 0 else 1
                    
                    # Detect bullish momentum shift
                    if (momentum_5m > 0.5 and momentum_15m > -0.5 and vol_acceleration > 1.5):
                        # Check it's not overextended
                        price_change_1h = float(ticker['priceChangePercent'])
                        
                        if -2 < price_change_1h < 5:  # Sweet spot
                            score = (momentum_5m * vol_acceleration) / 2
                            
                            opportunities.append({
                                'symbol': symbol,
                                'type': 'momentum_shift',
                                'score': min(score, 2.5),
                                'data': {
                                    'momentum_5m': momentum_5m,
                                    'momentum_15m': momentum_15m,
                                    'volume_acceleration': vol_acceleration,
                                    'price_change_1h': price_change_1h
                                }
                            })
                            
                except Exception as e:
                    continue
                    
            return opportunities
            
        except Exception as e:
            logging.error(f"Error detecting momentum shifts: {str(e)}")
            return []
    
    async def detect_breakout_patterns(self) -> List[Dict]:
        """Detect technical breakout patterns with improved accuracy"""
        opportunities = []
        
        try:
            # Get active pairs sorted by volume
            tickers = self.binance_client.get_ticker()
            active_pairs = sorted(
                [t for t in tickers if float(t['quoteVolume']) > 200000 and t['symbol'].endswith('USDT')],
                key=lambda x: float(x['quoteVolume']),
                reverse=True
            )[:75]  # Check more pairs
            
            for ticker in active_pairs:
                pair = ticker['symbol']
                
                if self._is_recently_scanned(pair):
                    continue
                    
                try:
                    # Get multiple timeframe data
                    klines_15m = self.binance_client.get_klines(
                        symbol=pair,
                        interval='15m',
                        limit=96  # 24 hours
                    )
                    
                    if len(klines_15m) < 96:
                        continue
                    
                    # Extract data
                    closes = np.array([float(k[4]) for k in klines_15m])
                    highs = np.array([float(k[2]) for k in klines_15m])
                    lows = np.array([float(k[3]) for k in klines_15m])
                    volumes = np.array([float(k[5]) for k in klines_15m])
                    
                    # Detect patterns
                    patterns = self._detect_patterns_advanced(closes, highs, lows, volumes)
                    
                    if patterns['breakout']:
                        # Verify with shorter timeframe
                        klines_5m = self.binance_client.get_klines(
                            symbol=pair,
                            interval='5m',
                            limit=12  # Last hour
                        )
                        
                        if len(klines_5m) >= 12:
                            recent_volumes = [float(k[5]) for k in klines_5m]
                            vol_surge = np.mean(recent_volumes[-3:]) / np.mean(recent_volumes[:-3])
                            
                            if vol_surge > 1.3:  # Volume confirmation
                                patterns['strength'] *= 1.2
                        
                        opportunities.append({
                            'symbol': pair,
                            'type': 'breakout_pattern',
                            'score': patterns['strength'],
                            'data': patterns
                        })
                        
                except Exception as e:
                    continue
                    
        except Exception as e:
            logging.error(f"Error detecting breakout patterns: {str(e)}")
            
        return opportunities
    
    def _detect_patterns_advanced(self, closes, highs, lows, volumes):
        """Advanced pattern detection with multiple confirmations"""
        patterns = {
            'breakout': False,
            'strength': 0,
            'pattern_type': None,
            'resistance_level': None,
            'support_level': None
        }
        
        try:
            # 1. Resistance breakout detection
            # Find recent resistance levels
            recent_highs = highs[-48:]  # Last 12 hours
            resistance_levels = []
            
            for i in range(len(recent_highs) - 4):
                window = recent_highs[i:i+5]
                if window[2] == max(window):  # Local high
                    resistance_levels.append(window[2])
            
            if resistance_levels:
                # Key resistance is the most touched level
                resistance = max(set(resistance_levels), key=resistance_levels.count)
                current_price = closes[-1]
                
                # Check if we just broke resistance
                if current_price > resistance * 1.002:  # 0.2% above resistance
                    # Verify it's a fresh breakout
                    prices_above = sum(1 for p in closes[-10:] if p > resistance)
                    
                    if prices_above <= 3:  # Just broke recently
                        # Volume confirmation
                        vol_avg = np.mean(volumes[-20:])
                        vol_recent = np.mean(volumes[-3:])
                        
                        if vol_recent > vol_avg * 1.5:
                            patterns['breakout'] = True
                            patterns['strength'] = 2.0
                            patterns['pattern_type'] = 'resistance_breakout'
                            patterns['resistance_level'] = resistance
            
            # 2. Consolidation breakout (flag pattern)
            if not patterns['breakout']:
                # Check for tight consolidation
                recent_range = (max(highs[-20:]) - min(lows[-20:])) / np.mean(closes[-20:])
                
                if recent_range < 0.03:  # Less than 3% range
                    # Check for directional move
                    price_change = (closes[-1] / closes[-20] - 1) * 100
                    vol_surge = volumes[-1] / np.mean(volumes[-20:])
                    
                    if abs(price_change) > 1 and vol_surge > 2:
                        patterns['breakout'] = True
                        patterns['strength'] = 1.8
                        patterns['pattern_type'] = 'consolidation_breakout'
            
            # 3. Moving average breakout
            if not patterns['breakout'] and len(closes) >= 50:
                ma20 = np.mean(closes[-20:])
                ma50 = np.mean(closes[-50:])
                
                # Price just crossed above both MAs
                if closes[-1] > ma20 > ma50 and closes[-5] < ma20:
                    patterns['breakout'] = True
                    patterns['strength'] = 1.5
                    patterns['pattern_type'] = 'ma_breakout'
            
            return patterns
            
        except Exception as e:
            logging.error(f"Error in pattern detection: {str(e)}")
            return patterns
    
    async def detect_new_listings(self) -> List[Dict]:
        """Detect newly listed or recently pumping small-cap coins"""
        opportunities = []
        
        try:
            tickers = self.binance_client.get_ticker()
            
            for ticker in tickers:
                symbol = ticker['symbol']
                if not symbol.endswith('USDT'):
                    continue
                
                try:
                    # Focus on smaller volume coins that might be new
                    volume = float(ticker['quoteVolume'])
                    if 50000 < volume < 5000000:  # Between 50k and 5M volume
                        
                        # Check if this might be newly listed
                        klines = self.binance_client.get_klines(
                            symbol=symbol,
                            interval='1d',
                            limit=30
                        )
                        
                        if len(klines) < 30:  # Less than 30 days of data
                            price_change = float(ticker['priceChangePercent'])
                            
                            # Look for strong momentum
                            if price_change > 5:
                                opportunities.append({
                                    'symbol': symbol,
                                    'type': 'new_listing',
                                    'score': 1.5,
                                    'data': {
                                        'days_listed': len(klines),
                                        'price_change_24h': price_change,
                                        'volume': volume
                                    }
                                })
                                
                except Exception:
                    continue
                    
        except Exception as e:
            logging.error(f"Error detecting new listings: {str(e)}")
            
        return opportunities
    
    async def detect_whale_movements(self) -> List[Dict]:
        """Detect whale accumulation or distribution"""
        opportunities = []
        
        try:
            # Get top volume pairs
            tickers = self.binance_client.get_ticker()
            active_pairs = sorted(
                [t for t in tickers if float(t['quoteVolume']) > 1000000 and t['symbol'].endswith('USDT')],
                key=lambda x: float(x['quoteVolume']),
                reverse=True
            )[:40]
            
            for ticker in active_pairs:
                pair = ticker['symbol']
                
                try:
                    # Get order book depth
                    depth = self.binance_client.get_order_book(symbol=pair, limit=20)
                    
                    # Calculate bid/ask volumes
                    bid_volume = sum(float(bid[0]) * float(bid[1]) for bid in depth['bids'])
                    ask_volume = sum(float(ask[0]) * float(ask[1]) for ask in depth['asks'])
                    
                    # Look for major imbalances
                    if bid_volume > ask_volume * 2.5:  # Strong buying pressure
                        # Check for large orders (whale walls)
                        large_bids = [float(bid[0]) * float(bid[1]) for bid in depth['bids'] if float(bid[0]) * float(bid[1]) > 50000]
                        
                        if large_bids:
                            opportunities.append({
                                'symbol': pair,
                                'type': 'whale_accumulation',
                                'score': min(bid_volume / ask_volume / 2, 2.5),
                                'data': {
                                    'bid_volume': bid_volume,
                                    'ask_volume': ask_volume,
                                    'large_bid_count': len(large_bids),
                                    'largest_bid': max(large_bids)
                                }
                            })
                            
                except Exception as e:
                    continue
                    
        except Exception as e:
            logging.error(f"Error detecting whale movements: {str(e)}")
            
        return opportunities
    
    async def detect_social_momentum(self) -> List[Dict]:
        """Enhanced social momentum detection using Binance data only"""
        opportunities = []
        
        try:
            # Use Binance ticker data for momentum detection
            tickers = self.binance_client.get_ticker()
            
            # Filter for active pairs with good volume
            active_pairs = [t for t in tickers if float(t['quoteVolume']) > 500000 and t['symbol'].endswith('USDT')]
            
            # Sort by price change to find trending pairs
            trending_pairs = sorted(active_pairs, key=lambda x: float(x['priceChangePercent']), reverse=True)[:20]
            
            for ticker in trending_pairs:
                symbol = ticker['symbol']
                
                # Skip if recently scanned
                if self._is_recently_scanned(symbol):
                    continue
                
                try:
                    price_change = float(ticker['priceChangePercent'])
                    volume = float(ticker['quoteVolume'])
                    trade_count = float(ticker['count'])
                    
                    # Look for positive momentum with good volume
                    if 0.5 < price_change < 8 and volume > 1000000:  # Sweet spot for momentum
                        # Calculate momentum score based on price change and volume
                        momentum_score = min(price_change * (volume / 1000000) / 10, 2.5)
                        
                        opportunities.append({
                            'symbol': symbol,
                            'type': 'social_momentum',
                            'score': momentum_score,
                            'data': {
                                'source': 'binance_momentum',
                                'price_change_24h': price_change,
                                'volume_24h': volume,
                                'trade_count': trade_count
                            }
                        })
                        
                except Exception as e:
                    logging.debug(f"Error processing {symbol} for social momentum: {str(e)}")
                    continue
                        
        except Exception as e:
            logging.error(f"Error detecting social momentum: {str(e)}")
            
        return opportunities
    
    async def detect_unusual_activity(self) -> List[Dict]:
        """Detect unusual trading patterns"""
        opportunities = []
        
        try:
            tickers = self.binance_client.get_ticker()
            
            # Sort by trade count to find unusual activity
            active_by_trades = sorted(
                [t for t in tickers if float(t['quoteVolume']) > 100000 and t['symbol'].endswith('USDT')],
                key=lambda x: float(x['count']),
                reverse=True
            )[:50]
            
            for ticker in active_by_trades:
                pair = ticker['symbol']
                
                try:
                    # Get recent trades
                    trades = self.binance_client.get_recent_trades(symbol=pair, limit=500)
                    
                    if len(trades) < 500:
                        continue
                    
                    # Analyze trade patterns
                    trade_sizes = [float(t['qty']) * float(t['price']) for t in trades]
                    buy_trades = [t for t in trades if not t['isBuyerMaker']]
                    
                    # Calculate metrics
                    avg_trade_size = np.mean(trade_sizes)
                    large_trades = [t for t in trade_sizes if t > avg_trade_size * 3]
                    buy_ratio = len(buy_trades) / len(trades)
                    
                    # Unusual buying pressure with large trades
                    if buy_ratio > 0.65 and len(large_trades) > 10 and avg_trade_size > 1000:
                        price_change = float(ticker['priceChangePercent'])
                        
                        if price_change < 8:  # Not pumped yet
                            opportunities.append({
                                'symbol': pair,
                                'type': 'unusual_buying',
                                'score': min(buy_ratio * 3, 2.0),
                                'data': {
                                    'buy_ratio': buy_ratio,
                                    'avg_trade_size': avg_trade_size,
                                    'large_trade_count': len(large_trades),
                                    'price_change_24h': price_change
                                }
                            })
                            
                except Exception as e:
                    continue
                    
        except Exception as e:
            logging.error(f"Error detecting unusual activity: {str(e)}")
            
        return opportunities
    
    def _combine_opportunities(self, opportunities: List[Dict]) -> List[Dict]:
        """Combine and score opportunities from different sources"""
        combined = {}
        
        for opp in opportunities:
            symbol = opp['symbol']
            
            if symbol not in combined:
                combined[symbol] = {
                    'symbol': symbol,
                    'sources': [],
                    'total_score': 0,
                    'data': {},
                    'timestamp': time.time()
                }
            
            # Add source data
            combined[symbol]['sources'].append(opp['type'])
            combined[symbol]['total_score'] += opp['score']
            combined[symbol]['data'][opp['type']] = opp['data']
        
        # Convert to list and add bonus for multiple sources
        opportunities = []
        for symbol, data in combined.items():
            # Bonus for multiple confirming sources
            source_bonus = (len(data['sources']) - 1) * 0.5
            data['score'] = data['total_score'] + source_bonus
            
            # Cap maximum score
            data['score'] = min(data['score'], 10.0)
            
            # Mark this pair as scanned
            self._mark_as_scanned(symbol)
            
            opportunities.append(data)
        
        return opportunities
    
    def _is_recently_scanned(self, symbol: str) -> bool:
        """Check if a pair was recently scanned"""
        if symbol in self.scanned_pairs_cache:
            if time.time() - self.scanned_pairs_cache[symbol] < self.cache_duration:
                return True
        return False
    
    def _mark_as_scanned(self, symbol: str):
        """Mark a pair as recently scanned"""
        self.scanned_pairs_cache[symbol] = time.time()
        
        # Clean old entries
        current_time = time.time()
        self.scanned_pairs_cache = {
            s: t for s, t in self.scanned_pairs_cache.items() 
            if current_time - t < self.cache_duration
        }