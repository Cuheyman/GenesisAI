import logging
import numpy as np
import asyncio
from typing import Dict, Any
import time

class OrderBookAnalysis:
    def __init__(self, binance_client):
        self.binance_client = binance_client
        self.cache = {}
        self.cache_expiry = {}
        
    async def get_order_book_data(self, pair: str) -> Dict[str, Any]:
        """Get and analyze order book data for a pair"""
        try:
            # Check cache first
            cache_key = f"orderbook_{pair}"
            current_time = time.time()
            
            if cache_key in self.cache and self.cache_expiry.get(cache_key, 0) > current_time:
                return self.cache[cache_key]
            
            # Get order book data from Binance
            depth = self.binance_client.get_order_book(symbol=pair, limit=20)
            
            # Extract bids and asks
            bids = depth['bids']
            asks = depth['asks']
            
            # Calculate volume on both sides
            bid_volume = sum(float(bid[0]) * float(bid[1]) for bid in bids)
            ask_volume = sum(float(ask[0]) * float(ask[1]) for ask in asks)
            
            # Calculate bid-ask spread
            top_bid = float(bids[0][0])
            top_ask = float(asks[0][0])
            bid_ask_spread = ((top_ask - top_bid) / top_bid) * 100  # as percentage
            
            # Calculate order book imbalance (OBI)
            total_volume = bid_volume + ask_volume
            obi = (bid_volume - ask_volume) / total_volume if total_volume > 0 else 0
            
            # Determine buy/sell pressure based on OBI
            if obi > 0.2:
                pressure = "strong_buy"
            elif obi > 0.05:
                pressure = "buy"
            elif obi < -0.2:
                pressure = "strong_sell"
            elif obi < -0.05:
                pressure = "sell"
            else:
                pressure = "neutral"
                
            # Calculate order book depth (total volume within 2% of mid price)
            mid_price = (top_bid + top_ask) / 2
            price_threshold = mid_price * 0.02
            
            # Bid depth (support)
            bid_depth = sum(float(bid[0]) * float(bid[1]) for bid in bids 
                          if mid_price - float(bid[0]) <= price_threshold)
            
            # Ask depth (resistance)
            ask_depth = sum(float(ask[0]) * float(ask[1]) for ask in asks
                          if float(ask[0]) - mid_price <= price_threshold)
            
            # Calculate support/resistance ratio
            sr_ratio = bid_depth / ask_depth if ask_depth > 0 else 1.0
            
            # Prepare result
            result = {
                'bid_volume': bid_volume,
                'ask_volume': ask_volume,
                'total_volume': total_volume,
                'bid_ask_spread': bid_ask_spread,
                'order_book_imbalance': obi,
                'pressure': pressure,
                'support_resistance_ratio': sr_ratio,
                'top_bid': top_bid,
                'top_ask': top_ask,
                'timestamp': current_time
            }
            
            # Cache the result for 30 seconds
            self.cache[cache_key] = result
            self.cache_expiry[cache_key] = current_time + 30
            
            return result
            
        except Exception as e:
            logging.error(f"Error getting order book data for {pair}: {str(e)}")
            return {
                'bid_volume': 0,
                'ask_volume': 0,
                'total_volume': 0,
                'bid_ask_spread': 0,
                'order_book_imbalance': 0,
                'pressure': "neutral",
                'support_resistance_ratio': 1.0,
                'top_bid': 0,
                'top_ask': 0
            }
    
    def analyze_liquidity(self, order_book_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze liquidity based on order book data"""
        try:
            pressure = order_book_data.get('pressure', 'neutral')
            obi = order_book_data.get('order_book_imbalance', 0)
            sr_ratio = order_book_data.get('support_resistance_ratio', 1.0)
            bid_ask_spread = order_book_data.get('bid_ask_spread', 0)
            
            # Calculate liquidity score (higher is more liquid)
            liquidity_score = 0
            
            # Tighter spreads mean more liquidity
            if bid_ask_spread < 0.05:  # Very tight spread
                liquidity_score += 3
            elif bid_ask_spread < 0.1:  # Normal spread
                liquidity_score += 2
            elif bid_ask_spread < 0.2:  # Wider spread
                liquidity_score += 1
                
            # Balanced order books indicate liquidity
            if abs(obi) < 0.05:  # Very balanced
                liquidity_score += 3
            elif abs(obi) < 0.1:  # Moderately balanced
                liquidity_score += 2
            elif abs(obi) < 0.2:  # Less balanced
                liquidity_score += 1
                
            # Higher volume indicates more liquidity
            total_volume = order_book_data.get('total_volume', 0)
            if total_volume > 10000:  # Very high volume
                liquidity_score += 3
            elif total_volume > 5000:  # High volume
                liquidity_score += 2
            elif total_volume > 1000:  # Moderate volume
                liquidity_score += 1
                
            # Normalize score to 0-10 range
            liquidity_score = min(10, liquidity_score)
            
            # Determine liquidity category
            if liquidity_score >= 7:
                liquidity_category = "high"
            elif liquidity_score >= 4:
                liquidity_category = "medium"
            else:
                liquidity_category = "low"
                
            return {
                'liquidity_score': liquidity_score,
                'liquidity_category': liquidity_category,
                'buy_sell_pressure': pressure,
                'support_resistance_ratio': sr_ratio
            }
            
        except Exception as e:
            logging.error(f"Error analyzing liquidity: {str(e)}")
            return {
                'liquidity_score': 5,  # Neutral default
                'liquidity_category': "medium",
                'buy_sell_pressure': "neutral",
                'support_resistance_ratio': 1.0
            }