import logging
import numpy as np
import time
from typing import Dict, List, Any, Tuple
import config

class EnhancedStrategy:
    def __init__(self, binance_client, nebula_client, market_analysis, order_book):
        self.binance_client = binance_client
        self.nebula = nebula_client
        self.market_analysis = market_analysis
        self.order_book = order_book
        self.cache = {}
        self.cache_expiry = {}
        
    # In enhanced_strategy.py, modify analyze_pair method
    async def analyze_pair(self, pair: str, mtf_analysis=None, order_book_data=None, 
                     correlation_data=None, market_state=None) -> Dict[str, Any]:
        try:
            # Get technical analysis from multi-timeframe data
            if not mtf_analysis:
                mtf_analysis = await self.market_analysis.get_multi_timeframe_analysis(pair)
                
            # Get order book analysis
            if not order_book_data:
                order_book_data = await self.order_book.get_order_book_data(pair)
                
            # Try to get Nebula AI insights but with timeout protection
            nebula_insights = None
            try:
                token = pair.replace("USDT", "")
                # Create a timeout for the nebula call
                nebula_task = asyncio.create_task(self.nebula.get_consolidated_insights(token))
                nebula_insights = await asyncio.wait_for(nebula_task, timeout=15)  # 15 second timeout
            except asyncio.TimeoutError:
                logging.warning(f"Nebula insights timed out for {pair}")
                nebula_insights = {
                    "metrics": {
                        "overall_sentiment": 0,
                        "prediction_direction": "neutral",
                        "prediction_confidence": 0.5,
                        "whale_accumulation": 0,
                        "smart_money_direction": "neutral"
                    }
                }
            except Exception as e:
                logging.warning(f"Nebula insights unavailable for {pair}: {str(e)}")
                nebula_insights = {
                    "metrics": {
                        "overall_sentiment": 0,
                        "prediction_direction": "neutral",
                        "prediction_confidence": 0.5,
                        "whale_accumulation": 0,
                        "smart_money_direction": "neutral"
                    }
                }
            
       
                
            # 5. Generate initial signals from technical analysis
            technical_signals = self._get_technical_signals(mtf_analysis)
            
            # 6. Generate order book signals
            orderbook_signals = self._get_orderbook_signals(order_book_data)
            
            # 7. Generate signals from Nebula insights
            nebula_signals = self._get_nebula_signals(nebula_insights)
            
            # 8. Combine all signals with appropriate weights
            combined_signals = self._combine_all_signals(
                technical_signals,
                orderbook_signals,
                nebula_signals,
                correlation_data,
                market_state
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
        """Generate signals from technical analysis"""
        try:
            mtf_trend = mtf_analysis.get('mtf_trend', 0)
            mtf_momentum = mtf_analysis.get('mtf_momentum', 0)
            mtf_overall = mtf_analysis.get('overall_score', 0)
            
            # Determine buy/sell signals
            buy_signal = False
            sell_signal = False
            
            # Strong buy if both trend and momentum are positive
            if mtf_trend > 0.3 and mtf_momentum > 0.3:
                buy_signal = True
                signal_strength = min(0.9, (mtf_trend + mtf_momentum) / 2)
            
            # Strong sell if both trend and momentum are negative
            elif mtf_trend < -0.3 and mtf_momentum < -0.3:
                sell_signal = True
                signal_strength = min(0.9, (abs(mtf_trend) + abs(mtf_momentum)) / 2)
                
            # Moderate buy if trend is positive with neutral momentum
            elif mtf_trend > 0.5 and mtf_momentum > -0.2:
                buy_signal = True
                signal_strength = min(0.7, mtf_trend * 0.8)
                
            # Moderate sell if trend is negative with neutral momentum
            elif mtf_trend < -0.5 and mtf_momentum < 0.2:
                sell_signal = True
                signal_strength = min(0.7, abs(mtf_trend) * 0.8)
                
            # Momentum-based signals when trend is neutral
            elif abs(mtf_trend) < 0.3 and abs(mtf_momentum) > 0.6:
                if mtf_momentum > 0:
                    buy_signal = True
                    signal_strength = min(0.6, mtf_momentum * 0.7)
                else:
                    sell_signal = True
                    signal_strength = min(0.6, abs(mtf_momentum) * 0.7)
            else:
                # No clear signal
                signal_strength = 0
                
            return {
                "buy_signal": buy_signal,
                "sell_signal": sell_signal,
                "signal_strength": signal_strength,
                "source": "technical",
                "details": {
                    "mtf_trend": mtf_trend,
                    "mtf_momentum": mtf_momentum,
                    "mtf_overall": mtf_overall
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
    
    def _get_nebula_signals(self, nebula_insights: Dict) -> Dict:
        """Generate signals from Nebula AI insights"""
        try:
            # Extract metrics
            metrics = nebula_insights.get('metrics', {})
            sentiment = metrics.get('overall_sentiment', 0)
            prediction = metrics.get('prediction_direction', 'neutral')
            confidence = metrics.get('prediction_confidence', 0.5)
            whale_activity = metrics.get('whale_accumulation', 0)
            smart_money = metrics.get('smart_money_direction', 'neutral')
            
            buy_signal = False
            sell_signal = False
            signal_strength = 0
            
            # Combined signal calculation
            if prediction == 'bullish' and sentiment > 0.2:
                buy_signal = True
                signal_strength = min(0.9, confidence * 0.8 + abs(sentiment) * 0.2)
            elif prediction == 'bearish' and sentiment < -0.2:
                sell_signal = True
                signal_strength = min(0.9, confidence * 0.8 + abs(sentiment) * 0.2)
            elif whale_activity > 0.5:  # Strong whale accumulation
                buy_signal = True
                signal_strength = min(0.8, whale_activity)
            elif whale_activity < -0.5:  # Strong whale distribution
                sell_signal = True
                signal_strength = min(0.8, abs(whale_activity))
            elif smart_money == 'bullish':
                buy_signal = True
                signal_strength = 0.7
            elif smart_money == 'bearish':
                sell_signal = True
                signal_strength = 0.7
            elif sentiment > 0.5:  # Very positive sentiment without prediction
                buy_signal = True
                signal_strength = min(0.6, sentiment)
            elif sentiment < -0.5:  # Very negative sentiment without prediction
                sell_signal = True
                signal_strength = min(0.6, abs(sentiment))
            
            return {
                "buy_signal": buy_signal,
                "sell_signal": sell_signal,
                "signal_strength": signal_strength,
                "source": "nebula",
                "details": {
                    "sentiment": sentiment,
                    "prediction": prediction,
                    "confidence": confidence,
                    "whale_activity": whale_activity,
                    "smart_money": smart_money
                }
            }
            
        except Exception as e:
            logging.error(f"Error in Nebula signals: {str(e)}")
            return {
                "buy_signal": False,
                "sell_signal": False,
                "signal_strength": 0,
                "source": "nebula",
                "error": str(e)
            }
    
    def _combine_all_signals(self, technical_signals, orderbook_signals, 
                           nebula_signals, correlation_data, market_state):
        """Combine all signals from different sources"""
        try:
            # Extract signals
            ta_buy = technical_signals.get('buy_signal', False)
            ta_sell = technical_signals.get('sell_signal', False)
            ta_strength = technical_signals.get('signal_strength', 0)
            
            ob_buy = orderbook_signals.get('buy_signal', False)
            ob_sell = orderbook_signals.get('sell_signal', False)
            ob_strength = orderbook_signals.get('signal_strength', 0)
            
            nb_buy = nebula_signals.get('buy_signal', False)
            nb_sell = nebula_signals.get('sell_signal', False)
            nb_strength = nebula_signals.get('signal_strength', 0)
            
            # Initial values (will be modified based on signals)
            final_buy = False
            final_sell = False
            final_strength = 0
            signal_source = "none"
            
            # Weight factors based on market state
            ta_weight = config.TECHNICAL_WEIGHT  # Default technical weight
            ob_weight = 0.15  # Default orderbook weight
            nb_weight = config.ONCHAIN_WEIGHT  # Default nebula weight
            
            # Adjust weights based on market regime
            regime = market_state.get('regime', 'NEUTRAL')
            
            if regime == "BULL_TRENDING":
                # In bullish trend, emphasize technical and nebula
                ta_weight = 0.55
                ob_weight = 0.10
                nb_weight = 0.35  # Increase nebula weight
            elif regime == "BEAR_TRENDING":
                # In bearish trend, emphasize technical and orderbook
                ta_weight = 0.55
                ob_weight = 0.25  # Increase orderbook weight
                nb_weight = 0.20
            elif regime == "BULL_VOLATILE" or regime == "BEAR_VOLATILE":
                # In volatile markets, orderbook becomes more important
                ta_weight = 0.50
                ob_weight = 0.30  # Increase orderbook weight
                nb_weight = 0.20
            
            # SIGNAL COMBINATION LOGIC
            
            # Calculate weighted signal strengths
            ta_weighted = ta_strength * ta_weight
            ob_weighted = ob_strength * ob_weight
            nb_weighted = nb_strength * nb_weight
            
            # BUY SIGNAL LOGIC
            buy_sources = []
            buy_strength = 0
            
            if ta_buy:
                buy_sources.append(f"TA:{ta_weighted:.2f}")
                buy_strength += ta_weighted
            
            if ob_buy:
                buy_sources.append(f"OB:{ob_weighted:.2f}")
                buy_strength += ob_weighted
            
            if nb_buy:
                buy_sources.append(f"NB:{nb_weighted:.2f}")
                buy_strength += nb_weighted
            
            # SELL SIGNAL LOGIC
            sell_sources = []
            sell_strength = 0
            
            if ta_sell:
                sell_sources.append(f"TA:{ta_weighted:.2f}")
                sell_strength += ta_weighted
            
            if ob_sell:
                sell_sources.append(f"OB:{ob_weighted:.2f}")
                sell_strength += ob_weighted
            
            if nb_sell:
                sell_sources.append(f"NB:{nb_weighted:.2f}")
                sell_strength += nb_weighted
            
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
                "nebula": nebula_signals.get('details', {}),
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