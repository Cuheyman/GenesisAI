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
            # Reduced requirements for better data availability
            timeframes = [
                {"interval": "5m", "weight": 0.1, "lookback_bars": 24},   # Reduced from 60
                {"interval": "15m", "weight": 0.15, "lookback_bars": 24}, # Reduced from 40
                {"interval": "1h", "weight": 0.25, "lookback_bars": 24},  # Reduced from 30
                {"interval": "4h", "weight": 0.25, "lookback_bars": 12},  # Reduced from 15
                {"interval": "1d", "weight": 0.25, "lookback_bars": 7}    # Reduced from 10
            ]

            # Get current time
            now = int(time.time() * 1000)

            # Container for results
            timeframe_results = []

            # Analyze each timeframe
            for tf in timeframes:
                interval = tf["interval"]
                
                # Reduced lookback time for better data availability
                if interval == "5m":
                    lookback_ms = 24 * 5 * 60 * 1000  # 2 hours
                elif interval == "15m":
                    lookback_ms = 24 * 15 * 60 * 1000  # 6 hours
                elif interval == "1h":
                    lookback_ms = 24 * 60 * 60 * 1000  # 24 hours
                elif interval == "4h":
                    lookback_ms = 12 * 4 * 60 * 60 * 1000  # 2 days
                else:  # 1d
                    lookback_ms = 7 * 24 * 60 * 60 * 1000  # 7 days

                # Get historical data
                klines = await self.get_klines(pair, now - lookback_ms, interval)

                if not klines:
                    logging.debug(f"No data for {pair} on {interval} timeframe")
                    continue
                    
                # More lenient data requirements
                min_required = max(3, tf["lookback_bars"] // 3)  # At least 3 candles, or 1/3 of desired
                if len(klines) < min_required:
                    logging.debug(f"Insufficient data for {pair} on {interval} timeframe: {len(klines)} < {min_required}")
                    continue

                # Calculate key metrics for this timeframe
                closes = np.array([float(k[4]) for k in klines])
                highs = np.array([float(k[2]) for k in klines])
                lows = np.array([float(k[3]) for k in klines])
                volumes = np.array([float(k[5]) for k in klines])

                # Skip if all prices are zero
                if np.all(closes == 0) or len(closes) < 2:
                    continue

                # Calculate trend indicators with adaptive periods
                ema_period_short = min(len(closes) // 2, 10)
                ema_period_long = min(len(closes) - 1, 20)
                
                if ema_period_short < 2:
                    ema_period_short = 2
                if ema_period_long <= ema_period_short:
                    ema_period_long = ema_period_short + 1

                # Calculate EMAs with adaptive periods
                if len(closes) >= ema_period_short:
                    ema_short = pd.Series(closes).ewm(span=ema_period_short, adjust=False).mean().values
                else:
                    ema_short = closes  # Use raw prices if insufficient data
                    
                if len(closes) >= ema_period_long:
                    ema_long = pd.Series(closes).ewm(span=ema_period_long, adjust=False).mean().values
                else:
                    ema_long = closes  # Use raw prices if insufficient data

                # Calculate slopes of EMAs (last few periods)
                slope_periods = min(3, len(closes) - 1)
                if len(ema_short) > slope_periods and len(ema_long) > slope_periods and slope_periods > 0:
                    ema_short_slope = (ema_short[-1] / ema_short[-slope_periods-1] - 1) * 100
                    ema_long_slope = (ema_long[-1] / ema_long[-slope_periods-1] - 1) * 100
                else:
                    ema_short_slope = 0
                    ema_long_slope = 0

                # Price location relative to EMAs
                curr_price = closes[-1]
                price_vs_ema_short = 1 if curr_price > ema_short[-1] else -1
                price_vs_ema_long = 1 if curr_price > ema_long[-1] else -1

                # Combine into trend score (-1 to 1)
                trend_score = (
                    (np.sign(ema_short_slope) * min(abs(ema_short_slope), 2) / 2) * 0.4 +
                    (np.sign(ema_long_slope) * min(abs(ema_long_slope), 2) / 2) * 0.3 +
                    (price_vs_ema_short * 0.15) +
                    (price_vs_ema_long * 0.15)
                )

                # Calculate RSI with adaptive period
                rsi_period = min(14, len(closes) - 1)
                rsi = self._calculate_rsi(closes, window=max(2, rsi_period))
                rsi_normalized = (rsi - 50) / 50  # Convert 0-100 to -1 to 1

                # Momentum calculation
                roc_periods = min(len(closes) - 1, 5)  # Adaptive ROC period
                if len(closes) > roc_periods and roc_periods > 0:
                    roc = (closes[-1] / closes[-roc_periods-1] - 1)
                    # Normalize ROC to -1 to 1 scale (max ±10%)
                    roc_normalized = np.clip(roc * 10, -1, 1)
                else:
                    roc_normalized = 0

                # Combine into momentum score
                momentum_score = rsi_normalized * 0.6 + roc_normalized * 0.4

                # Calculate volatility (simplified for limited data)
                if len(closes) > 1:
                    returns = np.diff(closes) / closes[:-1]
                    volatility_score = min(np.std(returns) * 100, 5) / 5  # Normalize to 0-1
                else:
                    volatility_score = 0.5  # Neutral if insufficient data

                # Volume analysis
                if len(volumes) > 1 and np.sum(volumes) > 0:
                    recent_volume = np.mean(volumes[-min(3, len(volumes)):])
                    older_volume = np.mean(volumes[:-min(3, len(volumes))] if len(volumes) > 3 else volumes)
                    volume_ratio = recent_volume / older_volume if older_volume > 0 else 1
                    volume_score = min(volume_ratio / 2, 1)  # Normalized to 0-1
                else:
                    volume_score = 0.5

                # Calculate Bollinger Bands (simplified)
                bb_period = min(len(closes), 10)
                if bb_period >= 2:
                    sma = np.mean(closes[-bb_period:])
                    std = np.std(closes[-bb_period:])
                    if std > 0:
                        upper_band = sma + (2 * std)
                        lower_band = sma - (2 * std)
                        bb_width = (upper_band - lower_band) / sma
                        bb_position = (curr_price - lower_band) / (upper_band - lower_band)
                    else:
                        bb_width = 0
                        bb_position = 0.5
                else:
                    bb_width = 0
                    bb_position = 0.5

                # Store results for this timeframe
                timeframe_result = {
                    "interval": interval,
                    "trend": np.clip(trend_score, -1, 1),  # Ensure bounds
                    "momentum": np.clip(momentum_score, -1, 1),  # Ensure bounds
                    "volatility": np.clip(volatility_score, 0, 1),  # Ensure bounds
                    "volume": np.clip(volume_score, 0, 1),  # Ensure bounds
                    "rsi": rsi,
                    "bb_width": bb_width,
                    "bb_position": bb_position,
                    "weight": tf["weight"],
                    "data_points": len(closes)
                }

                timeframe_results.append(timeframe_result)

            # Consolidate results across timeframes using weighted average
            if not timeframe_results:
                logging.warning(f"No timeframe data available for {pair}")
                return self._get_neutral_analysis()

            # Calculate weighted averages
            total_weight = sum(result["weight"] for result in timeframe_results)

            # Adjust weights if some timeframes are missing
            if total_weight > 0:
                for result in timeframe_results:
                    result["weight"] = result["weight"] / total_weight

                mtf_trend = sum(result["trend"] * result["weight"] for result in timeframe_results)
                mtf_momentum = sum(result["momentum"] * result["weight"] for result in timeframe_results)
                mtf_volatility = sum(result["volatility"] * result["weight"] for result in timeframe_results)
                mtf_volume = sum(result["volume"] * result["weight"] for result in timeframe_results)
            else:
                mtf_trend = 0
                mtf_momentum = 0
                mtf_volatility = 0.5
                mtf_volume = 0.5

            # Overall score combining trend and momentum (main factors)
            overall_score = mtf_trend * 0.6 + mtf_momentum * 0.4

            # Log successful analysis
            logging.debug(f"MTF analysis for {pair}: trend={mtf_trend:.3f}, momentum={mtf_momentum:.3f}, "
                         f"timeframes={len(timeframe_results)}")

            return {
                "mtf_trend": mtf_trend,
                "mtf_momentum": mtf_momentum,
                "mtf_volatility": mtf_volatility,
                "mtf_volume": mtf_volume,
                "overall_score": overall_score,
                "timeframe_data": timeframe_results,
                "timeframes_analyzed": len(timeframe_results)
            }

        except Exception as e:
            logging.error(f"Error in multi-timeframe analysis for {pair}: {str(e)}")
            return self._get_neutral_analysis()
    
    def _get_neutral_analysis(self):
        """Return neutral analysis when data is insufficient"""
        return {
            "mtf_trend": 0,
            "mtf_momentum": 0,
            "mtf_volatility": 0.5,
            "mtf_volume": 0.5,
            "overall_score": 0,
            "timeframe_data": [],
            "timeframes_analyzed": 0
        }
    
    async def get_klines(self, pair, start_time, interval):
        """Get kline data from Binance with caching and error handling"""
        cache_key = f"klines_{pair}_{interval}_{start_time}"
        current_time = time.time()
        
        # Check cache
        if cache_key in self.cache and self.cache_expiry.get(cache_key, 0) > current_time:
            return self.cache[cache_key]
            
        try:
            # Different cache expiry based on timeframe
            if interval in ['1m', '5m']:
                cache_duration = 60  # 1 minute for very short timeframes
            elif interval in ['15m', '30m']:
                cache_duration = 300  # 5 minutes
            elif interval == '1h':
                cache_duration = 900  # 15 minutes
            else:
                cache_duration = 1800  # 30 minutes for longer timeframes
            
            # Get klines from Binance with error handling
            try:
                klines = self.binance_client.get_historical_klines(
                    pair, interval, start_time, limit=1000
                )
            except Exception as api_error:
                logging.error(f"Binance API error for {pair} {interval}: {str(api_error)}")
                # Try with reduced limit
                try:
                    klines = self.binance_client.get_historical_klines(
                        pair, interval, start_time, limit=100
                    )
                    logging.info(f"Fallback successful for {pair} {interval} with reduced limit")
                except:
                    logging.error(f"Fallback also failed for {pair} {interval}")
                    return []
            
            # Validate klines data
            if not klines:
                logging.debug(f"No klines data returned for {pair} {interval}")
                return []
                
            # Filter out invalid klines
            valid_klines = []
            for kline in klines:
                try:
                    # Check if kline has required fields and valid data
                    if (len(kline) >= 6 and 
                        float(kline[4]) > 0 and  # Close price > 0
                        float(kline[5]) >= 0):   # Volume >= 0
                        valid_klines.append(kline)
                except (ValueError, TypeError):
                    continue
            
            if len(valid_klines) != len(klines):
                logging.debug(f"Filtered out {len(klines) - len(valid_klines)} invalid klines for {pair}")
            
            # Cache the results
            self.cache[cache_key] = valid_klines
            self.cache_expiry[cache_key] = current_time + cache_duration
            
            return valid_klines
            
        except Exception as e:
            logging.error(f"Error getting klines for {pair} {interval}: {str(e)}")
            return []
    
    def _calculate_rsi(self, prices, window=14):
        """Calculate RSI (Relative Strength Index) with improved error handling"""
        try:
            if len(prices) < 2:
                return 50  # Default to neutral if not enough data
                
            # Ensure we have enough data for the window
            effective_window = min(window, len(prices) - 1)
            if effective_window < 1:
                return 50
                
            # Calculate price changes
            deltas = np.diff(prices)
            
            if len(deltas) < effective_window:
                return 50
            
            # Separate gains and losses
            gains = np.where(deltas > 0, deltas, 0)
            losses = np.where(deltas < 0, -deltas, 0)
            
            # Calculate average gains and losses
            if len(gains) >= effective_window:
                avg_gain = np.mean(gains[-effective_window:])
                avg_loss = np.mean(losses[-effective_window:])
            else:
                avg_gain = np.mean(gains) if len(gains) > 0 else 0
                avg_loss = np.mean(losses) if len(losses) > 0 else 0
            
            if avg_loss == 0:
                return 100 if avg_gain > 0 else 50
                
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            
            # Ensure RSI is within bounds
            return np.clip(rsi, 0, 100)
            
        except Exception as e:
            logging.error(f"Error calculating RSI: {str(e)}")
            return 50  # Return neutral RSI on error
        
    async def calculate_market_breadth(self, top_pairs=None):
        """Calculate percentage of top coins above their 200MA"""
        try:
            if not top_pairs:
                # Use top market cap coins if not specified
                top_pairs = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'ADAUSDT', 
                            'XRPUSDT', 'DOGEUSDT', 'AVAXUSDT', 'LINKUSDT', 'DOTUSDT']
            
            above_ma_count = 0
            valid_count = 0
            
            for pair in top_pairs:
                try:
                    start_time = int(time.time() * 1000) - (50 * 24 * 60 * 60 * 1000)  # 50 days (reduced)
                    klines = await self.get_klines(pair, start_time, '1d')
                    
                    if not klines or len(klines) < 10:  # Reduced requirement
                        continue
                        
                    closes = [float(k[4]) for k in klines]
                    
                    # Use shorter MA if we don't have full data
                    ma_length = min(len(closes), 20)  # Much shorter than 200
                    if ma_length < 5:
                        continue
                        
                    sma = sum(closes[-ma_length:]) / ma_length
                    current_price = closes[-1]
                    
                    valid_count += 1
                    if current_price > sma:
                        above_ma_count += 1
                        
                except Exception as e:
                    logging.debug(f"Error calculating market breadth for {pair}: {str(e)}")
                    continue
                    
            if valid_count == 0:
                logging.warning("No valid pairs for market breadth calculation")
                return 0.5  # Default neutral if no valid pairs
                
            breadth = above_ma_count / valid_count
            logging.debug(f"Market breadth: {above_ma_count}/{valid_count} = {breadth:.3f}")
            return breadth
            
        except Exception as e:
            logging.error(f"Error calculating market breadth: {str(e)}")
            return 0.5  # Default to neutral
    
    def calculate_bollinger_bands(self, prices, window=20, num_std=2.0):
        """Calculate Bollinger Bands for a price series"""
        try:
            if len(prices) < 2:
                return {'upper': prices[-1], 'middle': prices[-1], 'lower': prices[-1]}

            # Adaptive window size
            effective_window = min(window, len(prices))
            
            prices_series = pd.Series(prices)
            middle_band = prices_series.rolling(window=effective_window, min_periods=1).mean().iloc[-1]
            std_dev = prices_series.rolling(window=effective_window, min_periods=1).std().iloc[-1]

            if pd.isna(std_dev) or std_dev == 0:
                std_dev = middle_band * 0.02  # Default to 2% of price

            upper_band = middle_band + (std_dev * num_std)
            lower_band = middle_band - (std_dev * num_std)

            # Calculate %B (position within bands)
            current_price = prices[-1]
            if (upper_band - lower_band) > 0:
                percent_b = (current_price - lower_band) / (upper_band - lower_band)
            else:
                percent_b = 0.5

            return {
                'upper': upper_band,
                'middle': middle_band,
                'lower': lower_band,
                'percent_b': percent_b,
                'width': (upper_band - lower_band) / middle_band if middle_band > 0 else 0
            }
            
        except Exception as e:
            logging.error(f"Error calculating Bollinger Bands: {str(e)}")
            current_price = prices[-1] if prices else 0
            return {
                'upper': current_price,
                'middle': current_price,
                'lower': current_price,
                'percent_b': 0.5,
                'width': 0
            }
    
    def calculate_macd(self, prices, fast_period=12, slow_period=26, signal_period=9):
        """Calculate MACD (Moving Average Convergence Divergence)"""
        try:
            # Adaptive periods based on available data
            data_length = len(prices)
            if data_length < slow_period + signal_period:
                # Use shorter periods if insufficient data
                fast_period = min(fast_period, data_length // 3)
                slow_period = min(slow_period, data_length // 2)
                signal_period = min(signal_period, data_length // 4)
                
                if fast_period < 2 or slow_period < 3 or signal_period < 2:
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

            # Check for crossovers
            crosses_above = False
            crosses_below = False
            
            if len(macd_line) > 1 and len(signal_line) > 1:
                crosses_above = current_macd > current_signal and macd_line.iloc[-2] <= signal_line.iloc[-2]
                crosses_below = current_macd < current_signal and macd_line.iloc[-2] >= signal_line.iloc[-2]

            return {
                'macd': current_macd,
                'signal': current_signal,
                'histogram': current_histogram,
                'normalized': normalized_macd,
                'histogram_direction': np.sign(current_histogram),
                'crosses_above': crosses_above,
                'crosses_below': crosses_below
            }
            
        except Exception as e:
            logging.error(f"Error calculating MACD: {str(e)}")
            return {
                'macd': 0,
                'signal': 0,
                'histogram': 0,
                'normalized': 0,
                'histogram_direction': 0,
                'crosses_above': False,
                'crosses_below': False
            }