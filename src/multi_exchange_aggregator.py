# multi_exchange_aggregator.py
import asyncio
import aiohttp
import logging
from typing import Dict, List, Any, Optional
import ccxt.async_support as ccxt
import numpy as np
import time
import config
from datetime import datetime

class MultiExchangeAggregator:
    """
    Aggregate data from multiple exchanges to find arbitrage and momentum opportunities
    """
    
    def __init__(self, config, binance_client=None, coinbase_client=None, bybit_client=None):
        self.config = config
        self.binance_client = binance_client
        self.exchanges = {}
        self.initialized = False
        self.price_cache = {}
        self.cache_duration = 10  # 10 seconds cache
        self.coinbase_client = coinbase_client
        self.bybit_client = bybit_client
    async def initialize(self):
        """Initialize connections to multiple exchanges"""
        try:
            # Initialize Binance (use existing client)
            if self.binance_client:
                self.exchanges['binance'] = {
                    'client': self.binance_client,
                    'ccxt': None,  # We'll use the existing client
                    'weight': 0.7,
                    'enabled': True,
                    'maker_fee': 0.001,
                    'taker_fee': 0.001
                }
            
            # Initialize CCXT clients for other exchanges (when enabled)
            if self.config.ENABLE_COINBASE:
                self.exchanges['coinbase'] = {
                    'client': None,
                    'ccxt': ccxt.coinbase({
                        'enableRateLimit': True,
                        'options': {'defaultType': 'spot'}
                    }),
                    'weight': 0.15,
                    'enabled': True,
                    'maker_fee': 0.005,
                    'taker_fee': 0.005
                }
            
            if self.config.ENABLE_BYBIT:
                self.exchanges['bybit'] = {
                    'client': None,
                    'ccxt': ccxt.bybit({
                        'enableRateLimit': True,
                        'options': {'defaultType': 'spot'}
                    }),
                    'weight': 0.15,
                    'enabled': True,
                    'maker_fee': 0.001,
                    'taker_fee': 0.001
                }
            
            # Load markets for CCXT exchanges
            for exchange_name, exchange_data in self.exchanges.items():
                if exchange_data['ccxt']:
                    await exchange_data['ccxt'].load_markets()
            
            self.initialized = True
            logging.info(f"Multi-exchange aggregator initialized with exchanges: {list(self.exchanges.keys())}")
            
        except Exception as e:
            logging.error(f"Error initializing multi-exchange aggregator: {str(e)}")
            self.initialized = False
    
    async def get_aggregated_price(self, symbol: str) -> Dict:
        """Get aggregated price data across all exchanges"""
        cache_key = f"price_{symbol}"
        current_time = time.time()
        
        # Check cache
        if cache_key in self.price_cache:
            cached_data = self.price_cache[cache_key]
            if current_time - cached_data['timestamp'] < self.cache_duration:
                return cached_data['data']
        
        prices = {}
        volumes = {}
        
        for exchange_name, exchange_data in self.exchanges.items():
            if not exchange_data['enabled']:
                continue
                
            try:
                ticker = await self._get_ticker(exchange_name, symbol)
                if ticker:
                    prices[exchange_name] = {
                        'bid': ticker.get('bid', 0),
                        'ask': ticker.get('ask', 0),
                        'last': ticker.get('last', 0),
                        'volume': ticker.get('baseVolume', 0),
                        'timestamp': ticker.get('timestamp', current_time)
                    }
                    volumes[exchange_name] = ticker.get('quoteVolume', 0)
                    
            except Exception as e:
                logging.debug(f"Error getting price from {exchange_name}: {str(e)}")
                continue
        
        if not prices:
            return None
        
        # Calculate weighted average price
        total_volume = sum(volumes.values())
        if total_volume > 0:
            weighted_price = sum(
                prices[ex]['last'] * volumes[ex] / total_volume 
                for ex in prices if volumes.get(ex, 0) > 0
            )
        else:
            weighted_price = np.mean([p['last'] for p in prices.values()])
        
        # Find best bid/ask across exchanges
        best_bid = max((p['bid'] for p in prices.values() if p['bid'] > 0), default=0)
        best_ask = min((p['ask'] for p in prices.values() if p['ask'] > 0), default=float('inf'))
        
        # Calculate spread opportunity
        if best_bid > 0 and best_ask < float('inf'):
            spread_pct = ((best_ask - best_bid) / best_bid) * 100
        else:
            spread_pct = 0
        
        result = {
            'symbol': symbol,
            'weighted_price': weighted_price,
            'best_bid': best_bid,
            'best_ask': best_ask,
            'spread_pct': spread_pct,
            'prices_by_exchange': prices,
            'total_volume': total_volume,
            'timestamp': current_time
        }
        
        # Cache the result
        self.price_cache[cache_key] = {
            'data': result,
            'timestamp': current_time
        }
        
        return result
    
    async def detect_arbitrage_opportunities(self, min_profit_pct: float = 0.5) -> List[Dict]:
        """Detect arbitrage opportunities across exchanges"""
        opportunities = []
        
        # Get common trading pairs across exchanges
        common_pairs = await self._get_common_pairs()
        
        for symbol in common_pairs[:50]:  # Check top 50 pairs
            try:
                price_data = await self.get_aggregated_price(symbol)
                if not price_data:
                    continue
                
                prices = price_data['prices_by_exchange']
                if len(prices) < 2:
                    continue
                
                # Find arbitrage opportunities
                for buy_exchange in prices:
                    for sell_exchange in prices:
                        if buy_exchange == sell_exchange:
                            continue
                        
                        buy_price = prices[buy_exchange]['ask']
                        sell_price = prices[sell_exchange]['bid']
                        
                        if buy_price <= 0 or sell_price <= 0:
                            continue
                        
                        # Calculate profit considering fees
                        buy_fee = self.exchanges[buy_exchange]['taker_fee']
                        sell_fee = self.exchanges[sell_exchange]['taker_fee']
                        
                        gross_profit_pct = ((sell_price - buy_price) / buy_price) * 100
                        net_profit_pct = gross_profit_pct - (buy_fee + sell_fee) * 100
                        
                        if net_profit_pct >= min_profit_pct:
                            opportunities.append({
                                'symbol': symbol,
                                'buy_exchange': buy_exchange,
                                'sell_exchange': sell_exchange,
                                'buy_price': buy_price,
                                'sell_price': sell_price,
                                'gross_profit_pct': gross_profit_pct,
                                'net_profit_pct': net_profit_pct,
                                'volume': min(
                                    prices[buy_exchange].get('volume', 0),
                                    prices[sell_exchange].get('volume', 0)
                                ),
                                'timestamp': time.time()
                            })
                            
            except Exception as e:
                logging.debug(f"Error checking arbitrage for {symbol}: {str(e)}")
                continue
        
        # Sort by profit percentage
        opportunities.sort(key=lambda x: x['net_profit_pct'], reverse=True)
        
        return opportunities[:10]  # Return top 10 opportunities
    
    async def detect_cross_exchange_momentum(self) -> List[Dict]:
        """Detect momentum that's spreading across exchanges"""
        momentum_pairs = []
        
        try:
            # Get price movements on primary exchange (Binance)
            primary_movers = await self._get_top_movers('binance', limit=20)
            
            for symbol, primary_data in primary_movers.items():
                cross_exchange_data = {
                    'symbol': symbol,
                    'primary_exchange': 'binance',
                    'primary_change': primary_data['change_pct'],
                    'following_exchanges': [],
                    'momentum_score': 0
                }
                
                # Check if other exchanges are following
                for exchange_name in self.exchanges:
                    if exchange_name == 'binance' or not self.exchanges[exchange_name]['enabled']:
                        continue
                    
                    try:
                        ticker = await self._get_ticker(exchange_name, symbol)
                        if ticker and ticker.get('percentage'):
                            change_pct = ticker['percentage']
                            
                            # Check if movement is in same direction but lagging
                            if (primary_data['change_pct'] > 0 and change_pct > 0 and 
                                change_pct < primary_data['change_pct'] * 0.7):
                                cross_exchange_data['following_exchanges'].append({
                                    'exchange': exchange_name,
                                    'change_pct': change_pct,
                                    'lag_factor': primary_data['change_pct'] / change_pct
                                })
                                
                    except Exception as e:
                        continue
                
                # Calculate momentum score
                if cross_exchange_data['following_exchanges']:
                    avg_lag = np.mean([
                        ex['lag_factor'] 
                        for ex in cross_exchange_data['following_exchanges']
                    ])
                    cross_exchange_data['momentum_score'] = avg_lag * len(cross_exchange_data['following_exchanges'])
                    momentum_pairs.append(cross_exchange_data)
            
            # Sort by momentum score
            momentum_pairs.sort(key=lambda x: x['momentum_score'], reverse=True)
            
        except Exception as e:
            logging.error(f"Error detecting cross-exchange momentum: {str(e)}")
        
        return momentum_pairs[:10]
    
    async def get_best_exchange_for_trade(self, symbol: str, side: str = 'buy') -> Dict:
        """Find the best exchange to execute a trade"""
        best_exchange = None
        best_price = float('inf') if side == 'buy' else 0
        
        price_data = await self.get_aggregated_price(symbol)
        if not price_data:
            return None
        
        prices = price_data['prices_by_exchange']
        
        for exchange_name, price_info in prices.items():
            if side == 'buy':
                price = price_info['ask']
                if 0 < price < best_price:
                    best_price = price
                    best_exchange = exchange_name
            else:  # sell
                price = price_info['bid']
                if price > best_price:
                    best_price = price
                    best_exchange = exchange_name
        
        if best_exchange:
            return {
                'exchange': best_exchange,
                'price': best_price,
                'side': side,
                'symbol': symbol,
                'fee': self.exchanges[best_exchange]['taker_fee']
            }
        
        return None
    
    async def _get_ticker(self, exchange: str, symbol: str) -> Dict:
        """Get ticker data from specific exchange"""
        try:
            if exchange == 'binance' and self.binance_client:
                # Use existing Binance client
                ticker = self.binance_client.get_symbol_ticker(symbol=symbol)
                ticker_24hr = self.binance_client.get_ticker(symbol=symbol)
                
                return {
                    'symbol': symbol,
                    'bid': float(self.binance_client.get_order_book(symbol=symbol, limit=5)['bids'][0][0]),
                    'ask': float(self.binance_client.get_order_book(symbol=symbol, limit=5)['asks'][0][0]),
                    'last': float(ticker['price']),
                    'baseVolume': float(ticker_24hr['volume']),
                    'quoteVolume': float(ticker_24hr['quoteVolume']),
                    'percentage': float(ticker_24hr['priceChangePercent']),
                    'timestamp': int(time.time() * 1000)
                }
            
            elif self.exchanges.get(exchange, {}).get('ccxt'):
                # Use CCXT for other exchanges
                ccxt_client = self.exchanges[exchange]['ccxt']
                ticker = await ccxt_client.fetch_ticker(symbol)
                return ticker
                
        except Exception as e:
            logging.debug(f"Error fetching ticker from {exchange} for {symbol}: {str(e)}")
            
        return None
    
    async def _get_orderbook(self, exchange: str, symbol: str, limit: int = 10) -> Dict:
        """Get orderbook data from specific exchange"""
        try:
            if exchange == 'binance' and self.binance_client:
                return self.binance_client.get_order_book(symbol=symbol, limit=limit)
            
            elif self.exchanges.get(exchange, {}).get('ccxt'):
                ccxt_client = self.exchanges[exchange]['ccxt']
                orderbook = await ccxt_client.fetch_order_book(symbol, limit)
                return orderbook
                
        except Exception as e:
            logging.debug(f"Error fetching orderbook from {exchange}: {str(e)}")
            
        return None
    
    async def _get_common_pairs(self) -> List[str]:
        """Get trading pairs common across all active exchanges"""
        if len(self.exchanges) == 1:
            # Only Binance active, return top pairs
            tickers = self.binance_client.get_ticker()
            sorted_tickers = sorted(
                [t for t in tickers if t['symbol'].endswith('USDT')],
                key=lambda x: float(x['quoteVolume']),
                reverse=True
            )
            return [t['symbol'] for t in sorted_tickers[:100]]
        
        # TODO: When multiple exchanges are active, find intersection
        # For now, return common major pairs
        return [
            'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'XRPUSDT', 'SOLUSDT',
            'ADAUSDT', 'DOGEUSDT', 'AVAXUSDT', 'DOTUSDT', 'MATICUSDT'
        ]
    
    async def _get_top_movers(self, exchange: str, limit: int = 20) -> Dict:
        """Get top moving pairs from an exchange"""
        movers = {}
        
        try:
            if exchange == 'binance' and self.binance_client:
                tickers = self.binance_client.get_ticker()
                
                # Filter and sort by absolute price change
                usdt_tickers = [t for t in tickers if t['symbol'].endswith('USDT')]
                sorted_tickers = sorted(
                    usdt_tickers,
                    key=lambda x: abs(float(x['priceChangePercent'])),
                    reverse=True
                )[:limit]
                
                for ticker in sorted_tickers:
                    movers[ticker['symbol']] = {
                        'change_pct': float(ticker['priceChangePercent']),
                        'volume': float(ticker['quoteVolume'])
                    }
                    
        except Exception as e:
            logging.error(f"Error getting top movers from {exchange}: {str(e)}")
            
        return movers