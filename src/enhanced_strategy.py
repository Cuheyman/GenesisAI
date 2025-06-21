import logging
import numpy as np
import time
import asyncio
from typing import Dict, List, Any, Tuple
import config
from advanced_indicators import AdvancedIndicators

class EnhancedStrategy:
    def __init__(self, binance_client, ai_client, market_analysis, order_book):
        self.binance_client = binance_client
        self.ai_client = ai_client  # Now using CoinGecko AI client
        self.market_analysis = market_analysis
        self.order_book = order_book
        self.cache = {}
        self.cache_expiry = {}

        try:
            
            self.advanced_indicators = AdvancedIndicators()
            logging.info("✅ Advanced indicators initialized in strategy")
        except Exception as e:
            logging.warning(f"⚠️ Failed to initialize advanced indicators: {str(e)}")
            self.advanced_indicators = None

    async def analyze_pair(self, pair: str, mtf_analysis=None, order_book_data=None, 
                    correlation_data=None, market_state=None, nebula_signal=None) -> Dict[str, Any]:
        try:
            # Get technical analysis from multi-timeframe data
            if not mtf_analysis:
                mtf_analysis = await self.market_analysis.get_multi_timeframe_analysis(pair)
                
            # Validate MTF analysis
            if not mtf_analysis or mtf_analysis.get('timeframes_analyzed', 0) == 0:
                logging.warning(f"No timeframe data available for {pair}, using minimal analysis")
                mtf_analysis = await self._get_minimal_analysis(pair)
                
            # Get order book analysis
            if not order_book_data:
                order_book_data = await self.order_book.get_order_book_data(pair)
                
            # Extract token from pair (removing USDT)
            token = pair.replace("USDT", "")
            
            # Get AI insights with improved error handling
            ai_insights = None
            
            # Check if AI client is available before attempting to use it
            if hasattr(self.ai_client, 'api_available') and self.ai_client.api_available:
                try:
                    # Create a timeout for the AI call
                    ai_task = asyncio.create_task(self.ai_client.get_consolidated_insights(token))
                    ai_insights = await asyncio.wait_for(ai_task, timeout=10)  # Reduced timeout
                    
                    # Verify we got valid data
                    if not ai_insights or not isinstance(ai_insights, dict) or 'metrics' not in ai_insights:
                        logging.debug(f"Invalid AI insights for {pair}, using defaults")
                        ai_insights = self._get_default_ai_insights()
                        
                except asyncio.TimeoutError:
                    logging.debug(f"AI insights timed out for {pair}")
                    ai_insights = self._get_default_ai_insights()
                except Exception as e:
                    logging.debug(f"AI insights error for {pair}: {str(e)}")
                    ai_insights = self._get_default_ai_insights()
            else:
                logging.debug(f"AI client not available for {pair}")
                ai_insights = self._get_default_ai_insights()
            
            # 🔥 TILFØJ DETTE: Get advanced indicators signals (NYT!)
            advanced_signals = None
            if self.advanced_indicators:
                try:
                    advanced_task = asyncio.create_task(self.advanced_indicators.get_advanced_signals(pair))
                    advanced_signals = await asyncio.wait_for(advanced_task, timeout=20)
                    logging.debug(f"🔍 Advanced signals for {pair}: {len(advanced_signals) if advanced_signals else 0} indicators")
                except asyncio.TimeoutError:
                    logging.debug(f"⏰ Advanced indicators timed out for {pair}")
                    advanced_signals = None
                except Exception as e:
                    logging.debug(f"❌ Advanced indicators error for {pair}: {str(e)}")
                    advanced_signals = None
            else:
                logging.debug(f"Advanced indicators not available for {pair}")
                
            # Generate initial signals from technical analysis
            technical_signals = self._get_technical_signals(mtf_analysis)
            
            # Generate order book signals
            orderbook_signals = self._get_orderbook_signals(order_book_data)
            
            # Generate signals from AI insights
            ai_signals = self._get_ai_signals(ai_insights)
            
           
            combined_signals = self._combine_all_signals(
                technical_signals,
                orderbook_signals,
                ai_signals,
                correlation_data,
                market_state,
                nebula_signal,  
                advanced_signals  
            )
            
            return combined_signals
            
        except Exception as e:
            logging.error(f"Error analyzing {pair}: {str(e)}")
            return {
                "buy_signal": False,
                "sell_signal": False,
                "signal_strength": 0,
                "source": "error",
                "error": str(e)
            }

    
    async def _get_minimal_analysis(self, pair: str):
        """Get minimal analysis when MTF fails"""
        try:
            # Try to get at least some recent price data
            current_time = int(time.time() * 1000)
            klines = await self.market_analysis.get_klines(pair, current_time - (60 * 60 * 1000), '5m')  # 1 hour of 5m data
            
            if klines and len(klines) >= 3:
                closes = [float(k[4]) for k in klines]
                
                # Simple momentum calculation
                recent_change = (closes[-1] / closes[0] - 1) if closes[0] > 0 else 0
                momentum = np.clip(recent_change * 10, -1, 1)  # Scale and clip
                
                # Simple trend based on recent price direction
                if len(closes) >= 3:
                    trend = 1 if closes[-1] > closes[-3] else -1
                    trend_strength = abs(recent_change) * 5  # Scale
                    trend = trend * min(trend_strength, 1)
                else:
                    trend = 0
                
                return {
                    "mtf_trend": trend,
                    "mtf_momentum": momentum,
                    "mtf_volatility": 0.5,  # Neutral
                    "mtf_volume": 0.5,  # Neutral
                    "overall_score": (trend + momentum) / 2,
                    "timeframes_analyzed": 1,  # Minimal analysis
                    "data_source": "minimal"
                }
            else:
                logging.warning(f"No price data available for {pair}")
                return self._get_neutral_analysis()
                
        except Exception as e:
            logging.error(f"Error getting minimal analysis for {pair}: {str(e)}")
            return self._get_neutral_analysis()
    
    def _get_neutral_analysis(self):
        """Return completely neutral analysis"""
        return {
            "mtf_trend": 0,
            "mtf_momentum": 0,
            "mtf_volatility": 0.5,
            "mtf_volume": 0.5,
            "overall_score": 0,
            "timeframes_analyzed": 0,
            "data_source": "neutral"
        }
    
    def _get_technical_signals(self, mtf_analysis: Dict) -> Dict:
        """Generate trading signals from technical analysis with enhanced logic"""
        try:
            # Extract key metrics with defaults
            mtf_trend = mtf_analysis.get('mtf_trend', 0)
            mtf_momentum = mtf_analysis.get('mtf_momentum', 0)
            overall_score = mtf_analysis.get('overall_score', 0)
            mtf_volatility = mtf_analysis.get('mtf_volatility', 0.5)
            timeframes_analyzed = mtf_analysis.get('timeframes_analyzed', 0)
            
            # Adjust signal strength based on data quality
            data_quality_multiplier = min(1.0, timeframes_analyzed / 3)  # Scale based on timeframes
            if data_quality_multiplier < 0.3:
                data_quality_multiplier = 0.3  # Minimum multiplier
            
            buy_signal = False
            sell_signal = False
            signal_strength = 0
            
            # Strategy 1: Strong Trend Following (more lenient thresholds)
            if mtf_trend > 0.2 and mtf_momentum > 0.1:  # Reduced from 0.3 and 0.2
                buy_signal = True
                signal_strength = min(0.8, (mtf_trend + mtf_momentum) / 2 + 0.1) * data_quality_multiplier
                
            # Strategy 2: Momentum Breakout (reduced requirements)
            elif mtf_momentum > 0.3 and overall_score > 0.2:  # Reduced from 0.4 and 0.35
                buy_signal = True
                signal_strength = min(0.7, mtf_momentum + 0.1) * data_quality_multiplier
                
            # Strategy 3: Mean Reversion in Stable Markets
            elif mtf_volatility < 0.4 and mtf_momentum < -0.2 and mtf_trend > -0.15:  # More lenient
                buy_signal = True
                signal_strength = min(0.6, abs(mtf_momentum) + 0.1) * data_quality_multiplier
                
            # Strategy 4: Any positive momentum with decent trend (very lenient)
            elif mtf_trend > 0.1 and mtf_momentum > 0.05:
                buy_signal = True
                signal_strength = min(0.5, (mtf_trend + mtf_momentum) / 2) * data_quality_multiplier
                
            # Sell signals (mirror logic with negative values)
            elif mtf_trend < -0.2 and mtf_momentum < -0.1:
                sell_signal = True
                signal_strength = min(0.8, abs(mtf_trend + mtf_momentum) / 2 + 0.1) * data_quality_multiplier
                
            # Don't trade in extreme volatility without clear direction
            if mtf_volatility > 0.8 and abs(mtf_trend) < 0.2:
                buy_signal = False
                sell_signal = False
                signal_strength = 0
            
            # Lower minimum signal strength requirement
            if signal_strength < 0.25:  # Reduced from 0.35
                buy_signal = False
                sell_signal = False
                signal_strength = 0
                
            # Add bonus for multiple timeframe confirmation
            if timeframes_analyzed >= 3:
                signal_strength *= 1.1  # 10% bonus for good data
            elif timeframes_analyzed >= 2:
                signal_strength *= 1.05  # 5% bonus for decent data
            
            signal_strength = min(signal_strength, 1.0)  # Cap at 1.0
            
            return {
                "buy_signal": buy_signal,
                "sell_signal": sell_signal,
                "signal_strength": signal_strength,
                "source": "technical",
                "details": {
                    "trend": mtf_trend,
                    "momentum": mtf_momentum,
                    "overall_score": overall_score,
                    "volatility": mtf_volatility,
                    "timeframes": timeframes_analyzed,
                    "data_quality": data_quality_multiplier
                }
            }
            
        except Exception as e:
            logging.error(f"Error in technical signals: {str(e)}")
            return {
                "buy_signal": False,
                "sell_signal": False,
                "signal_strength": 0,
                "source": "technical",
                "error": str(e)
            }
        
    def _get_orderbook_signals(self, order_book_data: Dict) -> Dict:
        """Generate signals from order book data"""
        try:
            pressure = order_book_data.get('pressure', 'neutral')
            obi = order_book_data.get('order_book_imbalance', 0)
            
            buy_signal = False
            sell_signal = False
            signal_strength = 0
            
            # More lenient orderbook signal generation
            if pressure == 'strong_buy':
                buy_signal = True
                signal_strength = 0.7  # Reduced from 0.8
            elif pressure == 'buy':
                buy_signal = True
                signal_strength = 0.5  # Reduced from 0.6
            elif pressure == 'strong_sell':
                sell_signal = True
                signal_strength = 0.7  # Reduced from 0.8
            elif pressure == 'sell':
                sell_signal = True
                signal_strength = 0.5  # Reduced from 0.6
            elif abs(obi) > 0.1:  # Additional check for imbalance
                if obi > 0.1:
                    buy_signal = True
                    signal_strength = min(0.6, obi * 3)  # Scale OBI to signal strength
                elif obi < -0.1:
                    sell_signal = True
                    signal_strength = min(0.6, abs(obi) * 3)
            
            return {
                "buy_signal": buy_signal,
                "sell_signal": sell_signal,
                "signal_strength": signal_strength,
                "source": "orderbook",
                "details": {
                    "pressure": pressure,
                    "imbalance": obi
                }
            }
            
        except Exception as e:
            logging.error(f"Error in orderbook signals: {str(e)}")
            return {
                "buy_signal": False,
                "sell_signal": False,
                "signal_strength": 0,
                "source": "orderbook",
                "error": str(e)
            }
    
    def _get_default_ai_insights(self):
        """Get default AI insights when CoinGecko is unavailable"""
        return {
            "metrics": {
                "overall_sentiment": 0,
                "prediction_direction": "neutral",
                "prediction_confidence": 0.5,
                "whale_accumulation": 0,
                "smart_money_direction": "neutral"
            }
        }

    def _get_ai_signals(self, ai_insights: Dict) -> Dict:
        """Generate signals from CoinGecko AI insights with improved reliability"""
        try:
            # Extract metrics with defaults
            metrics = ai_insights.get('metrics', {})
            sentiment = metrics.get('overall_sentiment', 0)
            prediction = metrics.get('prediction_direction', 'neutral')
            confidence = metrics.get('prediction_confidence', 0.5)
            whale_activity = metrics.get('whale_accumulation', 0)
            smart_money = metrics.get('smart_money_direction', 'neutral')
            
            # Get signal strength from AI client if available
            signal_strength = 0.5  # Default
            if hasattr(self.ai_client, 'get_signal_strength'):
                try:
                    signal_strength = self.ai_client.get_signal_strength(ai_insights)
                except:
                    signal_strength = 0.5
            
            # Determine buy/sell signals with more lenient thresholds
            buy_signal = False
            sell_signal = False
            
            # Prediction-based signals (reduced confidence requirement)
            if prediction == 'bullish' and confidence > 0.4:  # Reduced from 0.5
                buy_signal = True
                pred_strength = min(0.8, confidence * 1.5)  # Scale up confidence
            elif prediction == 'bearish' and confidence > 0.4:
                sell_signal = True
                pred_strength = min(0.8, confidence * 1.5)
            else:
                pred_strength = 0
            
            # Sentiment-based signals (reduced threshold)
            if not buy_signal and not sell_signal:
                if sentiment > 0.2:  # Reduced from 0.3
                    buy_signal = True
                    sentiment_strength = min(0.6, sentiment + 0.3)
                elif sentiment < -0.2:  # Reduced from -0.3
                    sell_signal = True
                    sentiment_strength = min(0.6, abs(sentiment) + 0.3)
                else:
                    sentiment_strength = 0
            else:
                sentiment_strength = 0
            
            # Whale activity signals (reduced threshold)
            if not buy_signal and not sell_signal:
                if whale_activity > 0.2:  # Reduced from 0.3
                    buy_signal = True
                    whale_strength = min(0.7, whale_activity + 0.3)
                elif whale_activity < -0.2:
                    sell_signal = True
                    whale_strength = min(0.7, abs(whale_activity) + 0.3)
                else:
                    whale_strength = 0
            else:
                whale_strength = 0
                
            # Smart money signals (more responsive)
            if not buy_signal and not sell_signal:
                if smart_money == 'bullish':
                    buy_signal = True
                    sm_strength = 0.5  # Default moderate strength
                elif smart_money == 'bearish':
                    sell_signal = True
                    sm_strength = 0.5
                else:
                    sm_strength = 0
            else:
                sm_strength = 0
                
            # Determine final signal strength (use the strongest one)
            final_strength = max(pred_strength, sentiment_strength, whale_strength, sm_strength)
            
            # Apply minimum threshold but more lenient
            if final_strength >= config.MIN_SIGNAL_STRENGTH:  # Reduced from higher threshold
                buy_signal = False
                sell_signal = False
                final_strength = 0
            
            # Safety check for both buy and sell signals (shouldn't happen)
            if buy_signal and sell_signal:
                # In case of conflict, use prediction confidence direction
                if prediction == 'bullish':
                    sell_signal = False
                elif prediction == 'bearish':
                    buy_signal = False
                else:
                    # If still ambiguous, use sentiment
                    if sentiment > 0:
                        sell_signal = False
                    else:
                        buy_signal = False
            
            # Create source text for logging
            source_parts = []
            if pred_strength > 0:
                source_parts.append(f"prediction:{prediction}")
            if sentiment_strength > 0:
                source_parts.append(f"sentiment:{sentiment:.2f}")
            if whale_strength > 0:
                source_parts.append(f"whale:{whale_activity:.2f}")
            if sm_strength > 0:
                source_parts.append(f"smart_money:{smart_money}")
                
            source_text = ",".join(source_parts) if source_parts else "neutral"
            
            return {
                "buy_signal": buy_signal,
                "sell_signal": sell_signal,
                "signal_strength": final_strength,
                "source": f"coingecko:{source_text}",
                "details": {
                    "sentiment": sentiment,
                    "prediction": prediction,
                    "confidence": confidence,
                    "whale_activity": whale_activity,
                    "smart_money": smart_money,
                    "raw_strength": signal_strength
                }
            }
            
        except Exception as e:
            logging.error(f"Error in CoinGecko AI signals: {str(e)}")
            return {
                "buy_signal": False,
                "sell_signal": False,
                "signal_strength": 0,
                "source": "coingecko:error",
                "error": str(e)
            }
    
    def _combine_all_signals(self, technical_signals, orderbook_signals, 
                           ai_signals, correlation_data, market_state, nebula_signal=None, 
                           advanced_signals=None):  # 🔥 TILFØJ DENNE PARAMETER
        """Combine all signals from different sources with improved logic"""
        try:
            # Extract signals
            ta_buy = technical_signals.get('buy_signal', False)
            ta_sell = technical_signals.get('sell_signal', False)
            ta_strength = technical_signals.get('signal_strength', 0)
            
            ob_buy = orderbook_signals.get('buy_signal', False)
            ob_sell = orderbook_signals.get('sell_signal', False)
            ob_strength = orderbook_signals.get('signal_strength', 0)
            
            ai_buy = ai_signals.get('buy_signal', False)
            ai_sell = ai_signals.get('sell_signal', False)
            ai_strength = ai_signals.get('signal_strength', 0)
            
            # 🔥 TILFØJ DETTE: Extract advanced signals
            adv_buy = False
            adv_sell = False
            adv_strength = 0
            
            if advanced_signals and advanced_signals.get('overall_signal'):
                adv_signal = advanced_signals['overall_signal']
                adv_buy = adv_signal.get('buy_signal', False)
                adv_sell = adv_signal.get('sell_signal', False)
                adv_strength = adv_signal.get('signal_strength', 0)
                logging.debug(f"🔍 Advanced signal: buy={adv_buy}, sell={adv_sell}, strength={adv_strength:.3f}")
            
            # Add direct signal if available (from nebula_signal parameter for compatibility)
            direct_signal_action = None
            direct_signal_strength = 0
            
            if nebula_signal:
                direct_signal_action = nebula_signal.get('action', 'hold')
                direct_signal_strength = nebula_signal.get('strength', 0.5)
            
            # Convert action to buy/sell signal
            if direct_signal_action == 'buy' and direct_signal_strength > 0.5:
                ai_buy = True
                ai_strength = max(ai_strength, direct_signal_strength)
            elif direct_signal_action == 'sell' and direct_signal_strength > 0.5:
                ai_sell = True
                ai_strength = max(ai_strength, direct_signal_strength)

            # Initial values
            final_buy = False
            final_sell = False
            final_strength = 0
            signal_source = "none"
            
            # 🔥 OPDATER DETTE: Enhanced weight factors based on market state (TILFØJ advanced weight)
            ta_weight = getattr(config, 'TECHNICAL_WEIGHT', 0.4)      # Reduced to make room for advanced
            ob_weight = getattr(config, 'ORDERBOOK_WEIGHT', 0.15)    # Keep same
            ai_weight = getattr(config, 'ONCHAIN_WEIGHT', 0.2)       # Reduced  
            adv_weight = getattr(config, 'TAAPI_OVERALL_WEIGHT', 0.25)  # 🔥 TILFØJ DENNE
            
            # Adjust weights based on market regime
            regime = market_state.get('regime', 'NEUTRAL') if market_state else 'NEUTRAL'
            
            if regime == "BULL_TRENDING":
                ta_weight = 0.40
                ob_weight = 0.15
                ai_weight = 0.20
                adv_weight = 0.25  # 🔥 TILFØJ
            elif regime == "BEAR_TRENDING":
                ta_weight = 0.45
                ob_weight = 0.20
                ai_weight = 0.15
                adv_weight = 0.20  # 🔥 TILFØJ
            elif regime in ["BULL_VOLATILE", "BEAR_VOLATILE"]:
                ta_weight = 0.35
                ob_weight = 0.25
                ai_weight = 0.15
                adv_weight = 0.25  # 🔥 TILFØJ
            
            # More lenient signal combination logic
            
            # BUY SIGNAL LOGIC (🔥 TILFØJ ADVANCED)
            buy_sources = []
            buy_strength = 0
            
            if ta_buy and ta_strength > 0.2:  # Reduced threshold
                buy_sources.append(f"TA:{ta_strength:.2f}")
                buy_strength += ta_strength * ta_weight
            
            if ob_buy and ob_strength > 0.2:  # Reduced threshold
                buy_sources.append(f"OB:{ob_strength:.2f}")
                buy_strength += ob_strength * ob_weight
            
            if ai_buy and ai_strength > 0.2:  # Reduced threshold
                buy_sources.append(f"AI:{ai_strength:.2f}")
                buy_strength += ai_strength * ai_weight
            
            # 🔥 TILFØJ DETTE: Advanced indicators buy signal
            if adv_buy and adv_strength > 0.2:
                buy_sources.append(f"Taapi:{adv_strength:.2f}")
                buy_strength += adv_strength * adv_weight
            
            # SELL SIGNAL LOGIC (🔥 TILFØJ ADVANCED)
            sell_sources = []
            sell_strength = 0
            
            if ta_sell and ta_strength > 0.2:
                sell_sources.append(f"TA:{ta_strength:.2f}")
                sell_strength += ta_strength * ta_weight
            
            if ob_sell and ob_strength > 0.2:
                sell_sources.append(f"OB:{ob_strength:.2f}")
                sell_strength += ob_strength * ob_weight
            
            if ai_sell and ai_strength > 0.2:
                sell_sources.append(f"AI:{ai_strength:.2f}")
                sell_strength += ai_strength * ai_weight
            
            # 🔥 TILFØJ DETTE: Advanced indicators sell signal
            if adv_sell and adv_strength > 0.2:
                sell_sources.append(f"Taapi:{adv_strength:.2f}")
                sell_strength += adv_strength * adv_weight
            
            # DETERMINE FINAL SIGNAL
            
            # If we have both buy and sell signals, use the stronger one
            if buy_strength > 0 and sell_strength > 0:
                if buy_strength > sell_strength:
                    final_buy = True
                    final_sell = False
                    final_strength = buy_strength
                    signal_source = "combined_buy:" + ",".join(buy_sources)
                else:
                    final_buy = False
                    final_sell = True
                    final_strength = sell_strength
                    signal_source = "combined_sell:" + ",".join(sell_sources)
            
            # If we only have a buy signal
            elif buy_strength > 0:
                final_buy = True
                final_strength = buy_strength
                signal_source = "buy:" + ",".join(buy_sources)
            
            # If we only have a sell signal
            elif sell_strength > 0:
                final_sell = True
                final_strength = sell_strength
                signal_source = "sell:" + ",".join(sell_sources)
            
            # 🔥 TILFØJ DETTE: Enhanced logging for debugging
            if buy_sources or sell_sources:
                logging.info(f" Signal sources: {', '.join(buy_sources + sell_sources)} → Final: {final_strength:.3f}")
            
            # Adjust for correlation (if provided)
            if final_buy and correlation_data:
                portfolio_correlation = correlation_data.get('portfolio_correlation', 0)
                is_diversified = correlation_data.get('is_diversified', True)
                
                if not is_diversified:
                    # Reduce strength for highly correlated assets
                    final_strength *= 0.8  # Less penalty than before
                    signal_source += ",correlation_penalty"
                elif portfolio_correlation < 0.2:
                    # Boost strength for diversifying assets
                    final_strength = min(0.95, final_strength * 1.1)  # Smaller boost
                    signal_source += ",diversity_bonus"
            
            # Final check: ensure minimum signal threshold (more lenient)
            min_threshold = getattr(config, 'MIN_SIGNAL_STRENGTH', 0.25)  # More lenient default
            if final_strength < min_threshold:
                final_buy = False
                final_sell = False
                final_strength = 0
                signal_source = f"weak_signal_{final_strength:.2f}<{min_threshold}"
            
            return {
                "buy_signal": final_buy,
                "sell_signal": final_sell,
                "signal_strength": final_strength,
                "source": signal_source,
                "technical": technical_signals.get('details', {}),
                "orderbook": orderbook_signals.get('details', {}),
                "ai": ai_signals.get('details', {}),
                "advanced": advanced_signals.get('overall_signal', {}) if advanced_signals else {},  # 🔥 TILFØJ
                "timestamp": time.time()
            }
            
        except Exception as e:
            logging.error(f"Error combining signals: {str(e)}")
            return {
                "buy_signal": False,
                "sell_signal": False,
                "signal_strength": 0,
                "source": "error",
                "error": str(e)
            }