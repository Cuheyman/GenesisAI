# websocket_manager.py
import asyncio
import json
import logging
import time
from typing import Dict, List, Callable, Any
from binance import AsyncClient, BinanceSocketManager
import websockets

class WebSocketManager:
    """Manage WebSocket connections for real-time market data"""
    
    def __init__(self, binance_client=None):
        self.binance_client = binance_client
        self.bm = None  # Binance Socket Manager
        self.active_streams = {}
        self.callbacks = {}
        self.price_cache = {}
        self.orderbook_cache = {}
        self.trade_cache = {}
        self.running = False
        
    async def initialize(self):
        """Initialize WebSocket connections"""
        try:
            if self.binance_client:
                # Create Binance Socket Manager
                self.bm = BinanceSocketManager(self.binance_client)
                logging.info("WebSocket manager initialized")
                return True
        except Exception as e:
            logging.error(f"Error initializing WebSocket manager: {str(e)}")
            return False
    
    async def start_price_streams(self, symbols: List[str], callback: Callable = None):
        """Start real-time price streams for multiple symbols"""
        try:
            # Create mini ticker streams for real-time prices
            streams = [f"{symbol.lower()}@miniTicker" for symbol in symbols]
            
            # Start multiplex socket
            ms = self.bm.multiplex_socket(streams)
            
            # Store the stream
            self.active_streams['price_multiplex'] = ms
            
            # Start listening
            async with ms as stream:
                while self.running:
                    msg = await stream.recv()
                    if msg:
                        await self._handle_price_update(msg, callback)
                        
        except Exception as e:
            logging.error(f"Error in price streams: {str(e)}")
            self.running = False
    
    async def start_orderbook_stream(self, symbol: str, callback: Callable = None):
        """Start real-time order book updates for a symbol"""
        try:
            # Depth stream with 5 levels
            depth = self.bm.depth_socket(symbol, depth=5)
            
            # Store the stream
            self.active_streams[f'orderbook_{symbol}'] = depth
            
            async with depth as stream:
                while self.running:
                    msg = await stream.recv()
                    if msg:
                        await self._handle_orderbook_update(symbol, msg, callback)
                        
        except Exception as e:
            logging.error(f"Error in orderbook stream for {symbol}: {str(e)}")
    
    async def start_trade_stream(self, symbol: str, callback: Callable = None):
        """Start real-time trade stream for a symbol"""
        try:
            # Aggregate trade stream
            trades = self.bm.aggtrade_socket(symbol)
            
            # Store the stream
            self.active_streams[f'trades_{symbol}'] = trades
            
            async with trades as stream:
                while self.running:
                    msg = await stream.recv()
                    if msg:
                        await self._handle_trade_update(symbol, msg, callback)
                        
        except Exception as e:
            logging.error(f"Error in trade stream for {symbol}: {str(e)}")
    
    async def start_kline_streams(self, symbols: List[str], interval: str = '1m', callback: Callable = None):
        """Start real-time kline/candlestick streams"""
        try:
            # Create kline streams
            streams = [f"{symbol.lower()}@kline_{interval}" for symbol in symbols]
            
            # Start multiplex socket
            ms = self.bm.multiplex_socket(streams)
            
            # Store the stream
            self.active_streams[f'klines_{interval}'] = ms
            
            async with ms as stream:
                while self.running:
                    msg = await stream.recv()
                    if msg:
                        await self._handle_kline_update(msg, callback)
                        
        except Exception as e:
            logging.error(f"Error in kline streams: {str(e)}")
    
    async def start_user_stream(self, callback: Callable = None):
        """Start user data stream for account updates"""
        try:
            # Get listen key
            listen_key = await self.binance_client.stream_get_listen_key()
            
            # Start user socket
            user_stream = self.bm.user_socket(listen_key)
            
            # Store the stream
            self.active_streams['user_data'] = user_stream
            
            # Keep alive task
            asyncio.create_task(self._keep_alive_user_stream(listen_key))
            
            async with user_stream as stream:
                while self.running:
                    msg = await stream.recv()
                    if msg:
                        await self._handle_user_update(msg, callback)
                        
        except Exception as e:
            logging.error(f"Error in user stream: {str(e)}")
    
    async def _handle_price_update(self, msg: Dict, callback: Callable = None):
        """Handle real-time price updates"""
        try:
            data = msg.get('data', {})
            symbol = data.get('s')
            
            if symbol:
                price_data = {
                    'symbol': symbol,
                    'price': float(data.get('c', 0)),
                    'open': float(data.get('o', 0)),
                    'high': float(data.get('h', 0)),
                    'low': float(data.get('l', 0)),
                    'volume': float(data.get('v', 0)),
                    'quote_volume': float(data.get('q', 0)),
                    'change_pct': float(data.get('P', 0)),
                    'timestamp': data.get('E', int(time.time() * 1000))
                }
                
                # Update cache
                self.price_cache[symbol] = price_data
                
                # Call callback if provided
                if callback:
                    await callback(symbol, price_data)
                    
        except Exception as e:
            logging.error(f"Error handling price update: {str(e)}")
    
    async def _handle_orderbook_update(self, symbol: str, msg: Dict, callback: Callable = None):
        """Handle real-time orderbook updates"""
        try:
            orderbook_data = {
                'symbol': symbol,
                'bids': msg.get('bids', []),
                'asks': msg.get('asks', []),
                'timestamp': msg.get('E', int(time.time() * 1000))
            }
            
            # Update cache
            self.orderbook_cache[symbol] = orderbook_data
            
            # Calculate metrics
            if orderbook_data['bids'] and orderbook_data['asks']:
                best_bid = float(orderbook_data['bids'][0][0])
                best_ask = float(orderbook_data['asks'][0][0])
                spread = best_ask - best_bid
                spread_pct = (spread / best_bid) * 100
                
                orderbook_data['best_bid'] = best_bid
                orderbook_data['best_ask'] = best_ask
                orderbook_data['spread'] = spread
                orderbook_data['spread_pct'] = spread_pct
            
            # Call callback if provided
            if callback:
                await callback(symbol, orderbook_data)
                
        except Exception as e:
            logging.error(f"Error handling orderbook update: {str(e)}")
    
    async def _handle_trade_update(self, symbol: str, msg: Dict, callback: Callable = None):
        """Handle real-time trade updates"""
        try:
            trade_data = {
                'symbol': symbol,
                'price': float(msg.get('p', 0)),
                'quantity': float(msg.get('q', 0)),
                'is_buyer_maker': msg.get('m', False),
                'trade_id': msg.get('a'),
                'timestamp': msg.get('E', int(time.time() * 1000))
            }
            
            # Update cache (keep last N trades)
            if symbol not in self.trade_cache:
                self.trade_cache[symbol] = []
            
            self.trade_cache[symbol].append(trade_data)
            
            # Keep only last 100 trades
            if len(self.trade_cache[symbol]) > 100:
                self.trade_cache[symbol] = self.trade_cache[symbol][-100:]
            
            # Call callback if provided
            if callback:
                await callback(symbol, trade_data)
                
        except Exception as e:
            logging.error(f"Error handling trade update: {str(e)}")
    
    async def _handle_kline_update(self, msg: Dict, callback: Callable = None):
        """Handle real-time kline updates"""
        try:
            data = msg.get('data', {})
            symbol = data.get('s')
            kline = data.get('k', {})
            
            if symbol and kline:
                kline_data = {
                    'symbol': symbol,
                    'interval': kline.get('i'),
                    'open': float(kline.get('o', 0)),
                    'high': float(kline.get('h', 0)),
                    'low': float(kline.get('l', 0)),
                    'close': float(kline.get('c', 0)),
                    'volume': float(kline.get('v', 0)),
                    'is_closed': kline.get('x', False),
                    'timestamp': kline.get('t', int(time.time() * 1000))
                }
                
                # Call callback if provided
                if callback:
                    await callback(symbol, kline_data)
                    
        except Exception as e:
            logging.error(f"Error handling kline update: {str(e)}")
    
    async def _handle_user_update(self, msg: Dict, callback: Callable = None):
        """Handle user account updates"""
        try:
            event_type = msg.get('e')
            
            if event_type == 'executionReport':
                # Order update
                order_data = {
                    'type': 'order',
                    'symbol': msg.get('s'),
                    'order_id': msg.get('i'),
                    'status': msg.get('X'),
                    'side': msg.get('S'),
                    'price': float(msg.get('p', 0)),
                    'quantity': float(msg.get('q', 0)),
                    'executed_qty': float(msg.get('z', 0)),
                    'timestamp': msg.get('E', int(time.time() * 1000))
                }
                
                if callback:
                    await callback('order', order_data)
                    
            elif event_type == 'outboundAccountPosition':
                # Account balance update
                balance_data = {
                    'type': 'balance',
                    'balances': msg.get('B', []),
                    'timestamp': msg.get('E', int(time.time() * 1000))
                }
                
                if callback:
                    await callback('balance', balance_data)
                    
        except Exception as e:
            logging.error(f"Error handling user update: {str(e)}")
    
    async def _keep_alive_user_stream(self, listen_key: str):
        """Keep user stream alive by extending listen key"""
        while self.running:
            try:
                await asyncio.sleep(1800)  # 30 minutes
                await self.binance_client.stream_keepalive(listen_key)
                logging.debug("Extended user stream listen key")
            except Exception as e:
                logging.error(f"Error extending listen key: {str(e)}")
                break
    
    def get_latest_price(self, symbol: str) -> Dict:
        """Get latest cached price for a symbol"""
        return self.price_cache.get(symbol)
    
    def get_latest_orderbook(self, symbol: str) -> Dict:
        """Get latest cached orderbook for a symbol"""
        return self.orderbook_cache.get(symbol)
    
    def get_recent_trades(self, symbol: str, limit: int = 10) -> List[Dict]:
        """Get recent cached trades for a symbol"""
        trades = self.trade_cache.get(symbol, [])
        return trades[-limit:] if trades else []
    
    async def start(self):
        """Start all WebSocket streams"""
        self.running = True
        logging.info("WebSocket manager started")
    
    async def stop(self):
        """Stop all WebSocket streams"""
        self.running = False
        
        # Close all active streams
        for stream_name, stream in self.active_streams.items():
            try:
                await stream.close()
                logging.info(f"Closed WebSocket stream: {stream_name}")
            except Exception as e:
                logging.error(f"Error closing stream {stream_name}: {str(e)}")
        
        self.active_streams.clear()
        logging.info("WebSocket manager stopped")