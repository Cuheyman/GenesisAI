import numpy as np
import pandas as pd
import logging
import time
import asyncio
from typing import Dict, List, Any, Tuple
import config

class MarketAnalysis:
    def __init__(self, binance_client):
        self.binance_client = binance_client
        self.cache = {}
        self.cache_expiry = {}
        
    async def get_multi_timeframe_analysis(self, pair: str) -> Dict[str, Any]:
        """
        Analyzes a trading pair across multiple timeframes and returns consolidated metrics
        """
        try:
            # Define timeframes to analyze with weights (more weight to longer timeframes)
            timeframes = [
                {"interval": "5m", "weight": 0.1, "lookback_bars": 60},
                {"interval": "15m", "weight": 0.15, "lookback_bars": 40},
                {"interval": "1h", "weight": 0.25, "lookback_bars": 30},
                {"interval": "4h", "weight": 0.25, "lookback_bars": 15},
                {"interval": "1d", "weight": 0.25, "lookback_bars": 10}
            ]

            # Get current time
            now = int(time.time() * 1000)

            # Container for results
            timeframe_results = []

            # Analyze each timeframe
            for tf in timeframes:
                interval = tf["interval"]
                lookback_ms = 3600 * 1000 * 24 * 10  # 10 days for maximum timeframe

                # Get historical data
                klines = await self.get_klines(pair, now - lookback_ms, interval)

                if not klines or len(klines) < tf["lookback_bars"]:
                    logging.warning(f"Insufficient data for {pair} on {interval} timeframe")
                    continue

                # Calculate key metrics for this timeframe
                closes = np.array([float(k[4]) for k in klines])
                highs = np.array([float(k[2]) for k in klines])
                lows = np.array([float(k[3]) for k in klines])
                volumes = np.array([float(k[5]) for k in klines])

                # Calculate trend indicators
                ema20 = pd.Series(closes).ewm(span=20, adjust=False).mean().values
                ema50 = pd.Series(closes).ewm(span=50, adjust=False).mean().values

                # Calculate slopes of EMAs (last 3 periods)
                if len(ema20) > 3 and len(ema50) > 3:
                    ema20_slope = (ema20[-1] / ema20[-4] - 1) * 100
                    ema50_slope = (ema50[-1] / ema50[-4] - 1) * 100
                else:
                    ema20_slope = 0
                    ema50_slope = 0

                # Price location relative to EMAs
                curr_price = closes[-1]
                price_vs_ema20 = 1 if curr_price > ema20[-1] else -1
                price_vs_ema50 = 1 if curr_price > ema50[-1] else -1

                # Combine into trend score (-1 to 1)
                trend_score = (
                    (np.sign(ema20_slope) * min(abs(ema20_slope), 2) / 2) * 0.4 +
                    (np.sign(ema50_slope) * min(abs(ema50_slope), 2) / 2) * 0.3 +
                    (price_vs_ema20 * 0.15) +
                    (price_vs_ema50 * 0.15)
                )

                # Calculate RSI
                rsi = self._calculate_rsi(closes)
                rsi_normalized = (rsi - 50) / 50  # Convert 0-100 to -1 to 1

                # Momentum calculation
                roc_periods = 10
                if len(closes) > roc_periods:
                    roc = (closes[-1] / closes[-roc_periods-1] - 1)
                    # Normalize ROC to -1 to 1 scale (max ±10%)
                    roc_normalized = np.clip(roc * 10, -1, 1)
                else:
                    roc_normalized = 0

                # Combine into momentum score
                momentum_score = rsi_normalized * 0.6 + roc_normalized * 0.4

                # Calculate volatility
                atr = 0
                if len(highs) > 14 and len(lows) > 14 and len(closes) > 14:
                    tr_values = []
                    for i in range(1, len(closes)):
                        tr = max(
                            highs[i] - lows[i],
                            abs(highs[i] - closes[i-1]),
                            abs(lows[i] - closes[i-1])
                        )
                        tr_values.append(tr)

                    atr = sum(tr_values[-14:]) / 14 if tr_values else 0

                # Normalize ATR as percentage of price
                volatility_score = min(atr / curr_price * 100, 5) / 5 if curr_price > 0 else 0

                # Volume analysis
                if len(volumes) > 20:
                    avg_volume = np.mean(volumes[-20:])
                    recent_volume = np.mean(volumes[-3:])
                    volume_ratio = recent_volume / avg_volume if avg_volume > 0 else 1
                    volume_score = min(volume_ratio / 2, 1)  # Normalized to 0-1
                else:
                    volume_score = 0.5

                # Calculate Bollinger Bands
                if len(closes) >= 20:
                    bb_period = 20
                    sma = np.mean(closes[-bb_period:])
                    std = np.std(closes[-bb_period:])
                    upper_band = sma + (2 * std)
                    lower_band = sma - (2 * std)
                    bb_width = (upper_band - lower_band) / sma
                    bb_position = (curr_price - lower_band) / (upper_band - lower_band) if upper_band > lower_band else 0.5
                else:
                    bb_width = 0
                    bb_position = 0.5

                # Store results for this timeframe
                timeframe_result = {
                    "interval": interval,
                    "trend": trend_score,
                    "momentum": momentum_score,
                    "volatility": volatility_score,
                    "volume": volume_score,
                    "rsi": rsi,
                    "bb_width": bb_width,
                    "bb_position": bb_position,
                    "weight": tf["weight"]
                }

                timeframe_results.append(timeframe_result)

            # Consolidate results across timeframes using weighted average
            if not timeframe_results:
                return {
                    "mtf_trend": 0,
                    "mtf_momentum": 0,
                    "mtf_volatility": 0,
                    "mtf_volume": 0.5,
                    "overall_score": 0
                }

            # Calculate weighted averages
            weight_sum = sum(result["weight"] for result in timeframe_results)

            # Recalculate weights if some timeframes are missing
            if weight_sum < 0.99:  # Should be 1.0 but allow for float imprecision
                for result in timeframe_results:
                    result["weight"] = result["weight"] / weight_sum

            mtf_trend = sum(result["trend"] * result["weight"] for result in timeframe_results)
            mtf_momentum = sum(result["momentum"] * result["weight"] for result in timeframe_results)
            mtf_volatility = sum(result["volatility"] * result["weight"] for result in timeframe_results)
            mtf_volume = sum(result["volume"] * result["weight"] for result in timeframe_results)

            # Overall score combining trend and momentum (main factors)
            overall_score = mtf_trend * 0.6 + mtf_momentum * 0.4

            return {
                "mtf_trend": mtf_trend,
                "mtf_momentum": mtf_momentum,
                "mtf_volatility": mtf_volatility,
                "mtf_volume": mtf_volume,
                "overall_score": overall_score,
                "timeframe_data": timeframe_results
            }

        except Exception as e:
            logging.error(f"Error in multi-timeframe analysis for {pair}: {str(e)}")
            return {
                "mtf_trend": 0,
                "mtf_momentum": 0,
                "mtf_volatility": 0,
                "mtf_volume": 0.5,
                "overall_score": 0
            }
    
    async def get_klines(self, pair, start_time, interval):
        """Get kline data from Binance with caching"""
        cache_key = f"klines_{pair}_{interval}_{start_time}"
        current_time = time.time()
        
        # Check cache
        if cache_key in self.cache and self.cache_expiry.get(cache_key, 0) > current_time:
            return self.cache[cache_key]
            
        try:
            # Different cache expiry based on timeframe
            if interval == '1m' or interval == '5m':
                cache_duration = 300  # 5 minutes
            elif interval == '15m' or interval == '30m':
                cache_duration = 900  # 15 minutes
            elif interval == '1h':
                cache_duration = 1800  # 30 minutes
            else:
                cache_duration = 3600  # 1 hour
            
            # Get klines from Binance
            klines = self.binance_client.get_historical_klines(
                pair, interval, start_time, limit=1000
            )
            
            # Cache the results
            self.cache[cache_key] = klines
            self.cache_expiry[cache_key] = current_time + cache_duration
            
            return klines
        except Exception as e:
            logging.error(f"Error getting klines for {pair} {interval}: {str(e)}")
            return []
    
    def _calculate_rsi(self, prices, window=14):
        """Calculate RSI (Relative Strength Index)"""
        if len(prices) < window + 1:
            return 50  # Default to neutral if not enough data
            
        # Calculate price changes
        deltas = np.diff(prices)
        
        # Separate gains and losses
        seed = deltas[:window+1]
        up = seed[seed >= 0].sum() / window
        down = -seed[seed < 0].sum() / window
        
        if down == 0:
            return 100
            
        rs = up / down
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
        
    async def calculate_market_breadth(self, top_pairs=None):
        """Calculate percentage of top coins above their 200MA"""
        if not top_pairs:
            # Use top market cap coins if not specified
            top_pairs = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'ADAUSDT', 
                        'XRPUSDT', 'DOGEUSDT', 'AVAXUSDT', 'LINKUSDT', 'DOTUSDT']
        
        above_ma_count = 0
        valid_count = 0
        
        for pair in top_pairs:
            try:
                start_time = int(time.time() * 1000) - (86400 * 1000 * 100)  # 100 days
                klines = await self.get_klines(pair, start_time, '1d')
                
                if not klines or len(klines) < 50:
                    continue
                    
                closes = [float(k[4]) for k in klines]
                
                # Use shorter MA if we don't have full 200 days
                ma_length = min(len(closes), 50)
                sma = sum(closes[-ma_length:]) / ma_length
                current_price = closes[-1]
                
                valid_count += 1
                if current_price > sma:
                    above_ma_count += 1
            except Exception as e:
                logging.debug(f"Error calculating market breadth for {pair}: {str(e)}")
                
        if valid_count == 0:
            return 0.5  # Default neutral if no valid pairs
            
        breadth = above_ma_count / valid_count
        return breadth
    
    def calculate_bollinger_bands(self, prices, window=20, num_std=2.0):
        """Calculate Bollinger Bands for a price series"""
        if len(prices) < window:
            return {'upper': prices[-1], 'middle': prices[-1], 'lower': prices[-1]}

        prices_series = pd.Series(prices)
        middle_band = prices_series.rolling(window=window).mean().iloc[-1]
        std_dev = prices_series.rolling(window=window).std().iloc[-1]

        upper_band = middle_band + (std_dev * num_std)
        lower_band = middle_band - (std_dev * num_std)

        # Calculate %B (position within bands)
        current_price = prices[-1]
        percent_b = (current_price - lower_band) / (upper_band - lower_band) if (upper_band - lower_band) > 0 else 0.5

        return {
            'upper': upper_band,
            'middle': middle_band,
            'lower': lower_band,
            'percent_b': percent_b,
            'width': (upper_band - lower_band) / middle_band  # Normalized width
        }
    
    def calculate_macd(self, prices, fast_period=12, slow_period=26, signal_period=9):
        """Calculate MACD (Moving Average Convergence Divergence)"""
        if len(prices) < slow_period + signal_period:
            return {'macd': 0, 'signal': 0, 'histogram': 0, 'normalized': 0}

        prices_series = pd.Series(prices)

        # Calculate EMAs
        fast_ema = prices_series.ewm(span=fast_period, adjust=False).mean()
        slow_ema = prices_series.ewm(span=slow_period, adjust=False).mean()

        # Calculate MACD line and signal line
        macd_line = fast_ema - slow_ema
        signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()

        # Calculate histogram
        histogram = macd_line - signal_line

        # Get current values
        current_macd = macd_line.iloc[-1]
        current_signal = signal_line.iloc[-1]
        current_histogram = histogram.iloc[-1]

        # Calculate normalized MACD (as percentage of price)
        current_price = prices[-1]
        normalized_macd = current_macd / current_price if current_price > 0 else 0

        return {
            'macd': current_macd,
            'signal': current_signal,
            'histogram': current_histogram,
            'normalized': normalized_macd,
            'histogram_direction': np.sign(current_histogram),
            'crosses_above': current_macd > current_signal and macd_line.iloc[-2] <= signal_line.iloc[-2],
            'crosses_below': current_macd < current_signal and macd_line.iloc[-2] >= signal_line.iloc[-2]
        }