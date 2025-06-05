import logging
import numpy as np
import time
import asyncio
from typing import Dict, List, Any, Tuple
import config

class EnhancedStrategy:
    def __init__(self, binance_client, ai_client, market_analysis, order_book):
        self.binance_client = binance_client
        self.ai_client = ai_client  # Now using CoinGecko AI client
        self.market_analysis = market_analysis
        self.order_book = order_book
        self.cache = {}
        self.cache_expiry = {}
        
    async def analyze_pair(self, pair: str, mtf_analysis=None, order_book_data=None, 
                    correlation_data=None, market_state=None, nebula_signal=None) -> Dict[str, Any]:
        try:
            # Get technical analysis from multi-timeframe data
            if not mtf_analysis:
                mtf_analysis = await self.market_analysis.get_multi_timeframe_analysis(pair)
                
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
                    ai_insights = await asyncio.wait_for(ai_task, timeout=12)  # 12 second timeout
                    
                    # Verify we got valid data
                    if not ai_insights or not isinstance(ai_insights, dict) or 'metrics' not in ai_insights:
                        logging.warning(f"Invalid AI insights for {pair}")
                        ai_insights = self._get_default_ai_insights()
                        
                except asyncio.TimeoutError:
                    logging.warning(f"AI insights timed out for {pair}")
                    ai_insights = self._get_default_ai_insights()
                except Exception as e:
                    logging.warning(f"AI insights unavailable for {pair}: {str(e)}")
                    ai_insights = self._get_default_ai_insights()
            else:
                logging.debug(f"Skipping AI insights for {pair} - API unavailable")
                ai_insights = self._get_default_ai_insights()
                
            # 5. Generate initial signals from technical analysis
            technical_signals = self._get_technical_signals(mtf_analysis)
            
            # 6. Generate order book signals
            orderbook_signals = self._get_orderbook_signals(order_book_data)
            
            # 7. Generate signals from AI insights
            ai_signals = self._get_ai_signals(ai_insights)
            
            # 8. Combine all signals with appropriate weights
            combined_signals = self._combine_all_signals(
                technical_signals,
                orderbook_signals,
                ai_signals,
                correlation_data,
                market_state,
                nebula_signal  # Keep parameter name for compatibility
            )
            
            return combined_signals
            
        except Exception as e:
            logging.error(f"Error analyzing {pair}: {str(e)}")
            return {
                "buy_signal": False,
                "sell_signal": False,
                "signal_strength": 0,
                "error": str(e)
            }
    
    def _get_technical_signals(self, mtf_analysis: Dict) -> Dict:
        
        try:
            # Extract key metrics
            mtf_trend = mtf_analysis.get('mtf_trend', 0)
            mtf_momentum = mtf_analysis.get('mtf_momentum', 0)
            overall_score = mtf_analysis.get('overall_score', 0)
            mtf_volatility = mtf_analysis.get('mtf_volatility', 0)
            
            # Get timeframe data for confluence
            timeframe_data = mtf_analysis.get('timeframe_data', [])
            
            buy_signal = False
            sell_signal = False
            signal_strength = 0
            
            # Strategy 1: Trend Following with Momentum Confirmation
            # Buy when trend and momentum align positively
            if mtf_trend > 0.3 and mtf_momentum > 0.2:
                # Strong bullish alignment
                buy_signal = True
                signal_strength = min(0.9, (mtf_trend + mtf_momentum) / 2 + 0.2)
                
            # Strategy 2: Mean Reversion in Low Volatility
            # Buy oversold conditions in stable markets
            elif mtf_volatility < 0.3 and mtf_momentum < -0.3 and mtf_trend > -0.2:
                # Oversold in stable market
                buy_signal = True
                signal_strength = min(0.7, abs(mtf_momentum) + 0.2)
                
            # Strategy 3: Breakout Trading
            # Buy on strong momentum with increasing volume
            elif mtf_momentum > 0.4 and overall_score > 0.35:
                buy_signal = True
                signal_strength = min(0.8, mtf_momentum + 0.2)
                
            # Sell signals - Mirror of buy logic
            elif mtf_trend < -0.3 and mtf_momentum < -0.2:
                # Strong bearish alignment
                sell_signal = True
                signal_strength = min(0.9, abs(mtf_trend + mtf_momentum) / 2 + 0.2)
                
            # Avoid trading in high volatility without clear direction
            if mtf_volatility > 0.7 and abs(mtf_trend) < 0.3:
                buy_signal = False
                sell_signal = False
                signal_strength = 0
            
            # Ensure minimum signal strength for trades
            if signal_strength < 0.35:
                buy_signal = False
                sell_signal = False
                signal_strength = 0
            
            return {
                "buy_signal": buy_signal,
                "sell_signal": sell_signal,
                "signal_strength": signal_strength,
                "source": "technical",
                "details": {
                    "trend": mtf_trend,
                    "momentum": mtf_momentum,
                    "overall_score": overall_score,
                    "volatility": mtf_volatility
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
            
            # Generate signals based on pressure
            if pressure == 'strong_buy':
                buy_signal = True
                signal_strength = 0.8
            elif pressure == 'buy':
                buy_signal = True
                signal_strength = 0.6
            elif pressure == 'strong_sell':
                sell_signal = True
                signal_strength = 0.8
            elif pressure == 'sell':
                sell_signal = True
                signal_strength = 0.6
            
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
            # Extract metrics
            metrics = ai_insights.get('metrics', {})
            sentiment = metrics.get('overall_sentiment', 0)
            prediction = metrics.get('prediction_direction', 'neutral')
            confidence = metrics.get('prediction_confidence', 0.5)
            whale_activity = metrics.get('whale_accumulation', 0)
            smart_money = metrics.get('smart_money_direction', 'neutral')
            
            # Get signal strength from AI client
            signal_strength = self.ai_client.get_signal_strength(ai_insights)
            
            # Determine buy/sell signals more reliably with consolidated logic
            buy_signal = False
            sell_signal = False
            
            # Prediction-based signals
            if prediction == 'bullish' and confidence > 0.5:
                buy_signal = True
                pred_strength = min(0.9, confidence * 1.2)  # Scale up confidence as strength
            elif prediction == 'bearish' and confidence > 0.5:
                sell_signal = True
                pred_strength = min(0.9, confidence * 1.2)
            else:
                pred_strength = 0
            
            # Sentiment-based signals (if no strong prediction)
            if not buy_signal and not sell_signal:
                if sentiment > 0.3:  # Positive sentiment
                    buy_signal = True
                    sentiment_strength = min(0.7, sentiment + 0.2)  # Scale up sentiment as strength
                elif sentiment < -0.3:  # Negative sentiment
                    sell_signal = True 
                    sentiment_strength = min(0.7, abs(sentiment) + 0.2)
                else:
                    sentiment_strength = 0
            else:
                sentiment_strength = 0
            
            # Whale activity signals (if no strong prediction or sentiment)
            if not buy_signal and not sell_signal:
                if whale_activity > 0.3:  # Accumulation
                    buy_signal = True
                    whale_strength = min(0.8, whale_activity + 0.2)
                elif whale_activity < -0.3:  # Distribution
                    sell_signal = True
                    whale_strength = min(0.8, abs(whale_activity) + 0.2)
                else:
                    whale_strength = 0
            else:
                whale_strength = 0
                
            # Smart money signals (only if we have nothing else)
            if not buy_signal and not sell_signal:
                if smart_money == 'bullish':
                    buy_signal = True
                    sm_strength = 0.6  # Fixed moderate strength
                elif smart_money == 'bearish':
                    sell_signal = True
                    sm_strength = 0.6
                else:
                    sm_strength = 0
            else:
                sm_strength = 0
                
            # Determine final signal strength (use the strongest one)
            final_strength = max(pred_strength, sentiment_strength, whale_strength, sm_strength)
            
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
                "signal_strength": final_strength if (buy_signal or sell_signal) else 0,
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
                           ai_signals, correlation_data, market_state, nebula_signal=None):
        """Combine all signals from different sources"""
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
            
            # Add direct signal if available (from nebula_signal parameter for compatibility)
            direct_signal_action = None
            direct_signal_strength = 0
            
            if nebula_signal:
                direct_signal_action = nebula_signal.get('action', 'hold')
                direct_signal_strength = nebula_signal.get('strength', 0.5)
            
            # Convert action to buy/sell signal
            if direct_signal_action == 'buy' and direct_signal_strength > 0.6:
                ai_buy = True
                ai_strength = max(ai_strength, direct_signal_strength)
            elif direct_signal_action == 'sell' and direct_signal_strength > 0.6:
                ai_sell = True
                ai_strength = max(ai_strength, direct_signal_strength)

            # Initial values (will be modified based on signals)
            final_buy = False
            final_sell = False
            final_strength = 0
            signal_source = "none"
            
            # Weight factors based on market state
            ta_weight = config.TECHNICAL_WEIGHT  # Default technical weight
            ob_weight = 0.15  # Default orderbook weight
            ai_weight = config.ONCHAIN_WEIGHT  # Default AI weight
            
            # Adjust weights based on market regime
            regime = market_state.get('regime', 'NEUTRAL') if market_state else 'NEUTRAL'
            
            if regime == "BULL_TRENDING":
                # In bullish trend, emphasize technical and AI
                ta_weight = 0.55
                ob_weight = 0.10
                ai_weight = 0.35  # Increase AI weight
            elif regime == "BEAR_TRENDING":
                # In bearish trend, emphasize technical and orderbook
                ta_weight = 0.55
                ob_weight = 0.25  # Increase orderbook weight
                ai_weight = 0.20
            elif regime == "BULL_VOLATILE" or regime == "BEAR_VOLATILE":
                # In volatile markets, orderbook becomes more important
                ta_weight = 0.50
                ob_weight = 0.30  # Increase orderbook weight
                ai_weight = 0.20
            
            # SIGNAL COMBINATION LOGIC
            
            # Calculate weighted signal strengths
            ta_weighted = ta_strength * ta_weight
            ob_weighted = ob_strength * ob_weight
            ai_weighted = ai_strength * ai_weight
            
            # BUY SIGNAL LOGIC
            buy_sources = []
            buy_strength = 0
            
            if ta_buy:
                buy_sources.append(f"TA:{ta_weighted:.2f}")
                buy_strength += ta_weighted
            
            if ob_buy:
                buy_sources.append(f"OB:{ob_weighted:.2f}")
                buy_strength += ob_weighted
            
            if ai_buy:
                buy_sources.append(f"AI:{ai_weighted:.2f}")
                buy_strength += ai_weighted
            
            # SELL SIGNAL LOGIC
            sell_sources = []
            sell_strength = 0
            
            if ta_sell:
                sell_sources.append(f"TA:{ta_weighted:.2f}")
                sell_strength += ta_weighted
            
            if ob_sell:
                sell_sources.append(f"OB:{ob_weighted:.2f}")
                sell_strength += ob_weighted
            
            if ai_sell:
                sell_sources.append(f"AI:{ai_weighted:.2f}")
                sell_strength += ai_weighted
            
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
                signal_source = "combined_buy:" + ",".join(buy_sources)
            
            # If we only have a sell signal
            elif sell_strength > 0:
                final_sell = True
                final_strength = sell_strength
                signal_source = "combined_sell:" + ",".join(sell_sources)
            
            # Adjust for correlation
            if final_buy and correlation_data:
                portfolio_correlation = correlation_data.get('portfolio_correlation', 0)
                is_diversified = correlation_data.get('is_diversified', True)
                
                if not is_diversified:
                    # Reduce strength for highly correlated assets
                    final_strength *= 0.7
                    signal_source += ",correlation_penalty"
                elif portfolio_correlation < 0.2:
                    # Boost strength for diversifying assets
                    final_strength = min(0.95, final_strength * 1.2)
                    signal_source += ",diversity_bonus"
            
            # Final check: ensure minimum signal threshold
            if final_strength < 0.3:
                final_buy = False
                final_sell = False
                final_strength = 0
            
            return {
                "buy_signal": final_buy,
                "sell_signal": final_sell,
                "signal_strength": final_strength,
                "source": signal_source,
                "technical": technical_signals.get('details', {}),
                "orderbook": orderbook_signals.get('details', {}),
                "ai": ai_signals.get('details', {}),
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