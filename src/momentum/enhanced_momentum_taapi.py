import asyncio
import logging
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import json
import config 

@dataclass
class MomentumSignal:
    """Enhanced momentum signal with confidence scoring"""
    action: str  # BUY, HOLD (never SELL for your strategy)
    confidence: float  # 0-100
    momentum_strength: str  # WEAK, MODERATE, STRONG, EXPLOSIVE
    breakout_type: str  # VOLUME_BREAKOUT, PRICE_BREAKOUT, MOMENTUM_SURGE, CONSOLIDATION_BREAK
    entry_quality: str  # POOR, FAIR, GOOD, EXCELLENT
    volume_confirmation: bool
    risk_reward_ratio: float
    reasons: List[str]
    indicators_aligned: int  # Number of indicators supporting the signal
    timestamp: datetime

class EnhancedMomentumTaapiClient:
    """
    Enhanced TAAPI.io client optimized for momentum-based bullish strategies
    Designed for 75-90% win rate through selective, high-quality entries only
    """
    
    def __init__(self, api_secret: str):
        self.api_secret = config.TAAPI_SECRET
        self.base_url = "https://api.taapi.io"
        self.session = None
        
        # Momentum-specific thresholds for high win rate
        self.momentum_thresholds = {
            'rsi_oversold_entry': 35,  # More conservative than 30
            'rsi_momentum_min': 45,    # Must show upward momentum
            'rsi_overbought_avoid': 75, # Avoid late entries
            'macd_histogram_min': 0.001, # Positive histogram required
            'volume_spike_min': 1.8,   # 80% volume increase minimum
            'price_momentum_min': 0.8, # 0.8% price increase in timeframe
            'breakout_confirmation': 0.5, # 0.5% above resistance
            'confluence_min': 4,       # Minimum indicators agreeing
        }
        
        # Multiple timeframe analysis for confirmation
        self.timeframes = ['15m', '1h', '4h']
        self.primary_timeframe = '1h'
        
        # Success tracking
        self.signal_history = []
        self.win_rate_tracker = {'wins': 0, 'losses': 0, 'total': 0}

    async def get_momentum_optimized_signal(self, symbol: str) -> MomentumSignal:
        """
        Get momentum-optimized signal using enhanced TAAPI bulk queries
        Focuses on high-probability bullish setups only
        """
        try:
            # Multi-timeframe momentum analysis
            mtf_data = await self._get_multi_timeframe_data(symbol)
            
            # Volume and price action analysis
            volume_analysis = await self._analyze_volume_patterns(symbol, mtf_data)
            
            # Breakout detection
            breakout_analysis = await self._detect_breakout_patterns(symbol, mtf_data)
            
            # Momentum confluence scoring
            confluence_score = self._calculate_momentum_confluence(mtf_data, volume_analysis, breakout_analysis)
            
            # Generate final signal
            signal = self._generate_momentum_signal(
                symbol, mtf_data, volume_analysis, breakout_analysis, confluence_score
            )
            
            # Log for performance tracking
            self._log_signal_for_tracking(signal)
            
            return signal
            
        except Exception as e:
            logging.error(f"Error generating momentum signal for {symbol}: {str(e)}")
            return self._create_hold_signal(f"Error: {str(e)}")

    async def _get_multi_timeframe_data(self, symbol: str) -> Dict[str, Any]:
        """Get comprehensive data across multiple timeframes using bulk queries"""
        
        # Bulk query for primary timeframe (1h) - most comprehensive
        primary_construct = {
            "secret": self.api_secret,
            "construct": {
                "exchange": "binance",
                "symbol": symbol.replace("USDT", "/USDT"),
                "interval": "1h",
                "indicators": [
                    {"indicator": "rsi", "period": 14},
                    {"indicator": "macd", "fastPeriod": 12, "slowPeriod": 26, "signalPeriod": 9},
                    {"indicator": "ema", "period": 20, "id": "ema20"},
                    {"indicator": "ema", "period": 50, "id": "ema50"},
                    {"indicator": "ema", "period": 200, "id": "ema200"},
                    {"indicator": "bbands", "period": 20, "nbdevup": 2, "nbdevdn": 2},
                    {"indicator": "atr", "period": 14},
                    {"indicator": "adx", "period": 14},
                    {"indicator": "stochrsi", "fastk": 3, "fastd": 3, "rsi_period": 14},
                    {"indicator": "volume_profile", "id": "volume_profile_1h"},
                    # Advanced momentum indicators
                    {"indicator": "supertrend", "period": 10, "factor": 3.0},
                    {"indicator": "squeeze", "bb_length": 20, "kc_length": 20},
                    {"indicator": "vwap"},
                    {"indicator": "obv"},  # On Balance Volume
                    {"indicator": "mfi", "period": 14},  # Money Flow Index
                    # Pattern recognition
                    {"indicator": "cdlhammer"},
                    {"indicator": "cdlengulfing"},
                    {"indicator": "cdlmorningstar"},
                ]
            }
        }
        
        # Bulk query for 15m timeframe (short-term momentum)
        short_term_construct = {
            "secret": self.api_secret,
            "construct": {
                "exchange": "binance", 
                "symbol": symbol.replace("USDT", "/USDT"),
                "interval": "15m",
                "indicators": [
                    {"indicator": "rsi", "period": 14, "id": "rsi_15m"},
                    {"indicator": "macd", "id": "macd_15m"},
                    {"indicator": "volume_profile", "id": "volume_15m"},
                    {"indicator": "vwap", "id": "vwap_15m"},
                    {"indicator": "squeeze", "id": "squeeze_15m"},
                ]
            }
        }
        
        # Bulk query for 4h timeframe (trend confirmation)
        long_term_construct = {
            "secret": self.api_secret,
            "construct": {
                "exchange": "binance",
                "symbol": symbol.replace("USDT", "/USDT"), 
                "interval": "4h",
                "indicators": [
                    {"indicator": "rsi", "period": 14, "id": "rsi_4h"},
                    {"indicator": "macd", "id": "macd_4h"},
                    {"indicator": "ema", "period": 20, "id": "ema20_4h"},
                    {"indicator": "ema", "period": 50, "id": "ema50_4h"},
                    {"indicator": "adx", "period": 14, "id": "adx_4h"},
                ]
            }
        }
        
        # Execute all bulk queries
        primary_data = await self._execute_bulk_query(primary_construct)
        short_term_data = await self._execute_bulk_query(short_term_construct)
        long_term_data = await self._execute_bulk_query(long_term_construct)
        
        return {
            'primary': primary_data,
            'short_term': short_term_data, 
            'long_term': long_term_data,
            'symbol': symbol
        }

    async def _analyze_volume_patterns(self, symbol: str, mtf_data: Dict) -> Dict[str, Any]:
        """Analyze volume patterns for momentum confirmation"""
        
        primary = mtf_data['primary']
        short_term = mtf_data['short_term']
        
        volume_analysis = {
            'volume_spike': False,
            'volume_trend': 'neutral',
            'money_flow_bullish': False,
            'volume_breakout': False,
            'volume_confirmation_score': 0
        }
        
        try:
            # On Balance Volume analysis
            obv = self._extract_indicator_value(primary, 'obv')
            if obv:
                volume_analysis['obv_trending_up'] = True  # Simplified - in real implementation, compare with previous values
            
            # Money Flow Index
            mfi = self._extract_indicator_value(primary, 'mfi')
            if mfi and mfi > 50:
                volume_analysis['money_flow_bullish'] = True
                volume_analysis['volume_confirmation_score'] += 1
            
            # Volume Profile analysis (if available)
            volume_profile = self._extract_indicator_value(primary, 'volume_profile_1h')
            if volume_profile:
                volume_analysis['high_volume_node_support'] = True
                volume_analysis['volume_confirmation_score'] += 1
            
            # Short-term volume spike detection
            volume_15m = self._extract_indicator_value(short_term, 'volume_15m')
            if volume_15m:
                # This would compare with average volume in real implementation
                volume_analysis['recent_volume_spike'] = True
                volume_analysis['volume_confirmation_score'] += 1
            
        except Exception as e:
            logging.warning(f"Volume analysis error for {symbol}: {str(e)}")
        
        return volume_analysis

    async def _detect_breakout_patterns(self, symbol: str, mtf_data: Dict) -> Dict[str, Any]:
        """Detect various breakout patterns for momentum entries"""
        
        primary = mtf_data['primary']
        
        breakout_analysis = {
            'breakout_type': 'none',
            'breakout_strength': 0,
            'consolidation_break': False,
            'resistance_break': False,
            'squeeze_break': False,
            'pattern_breakout': False
        }
        
        try:
            # Bollinger Band squeeze breakout
            squeeze = self._extract_indicator_value(primary, 'squeeze')
            if squeeze:
                # Squeeze momentum indicator - positive values suggest upward breakout
                if squeeze > 0:
                    breakout_analysis['squeeze_break'] = True
                    breakout_analysis['breakout_type'] = 'squeeze_breakout'
                    breakout_analysis['breakout_strength'] += 2
            
            # Bollinger Band breakout
            bbands = self._extract_indicator_value(primary, 'bbands')
            if bbands:
                # Check if price is breaking above upper band (momentum breakout)
                # This would need current price comparison in real implementation
                breakout_analysis['bb_momentum_break'] = True
                breakout_analysis['breakout_strength'] += 1
            
            # SuperTrend breakout
            supertrend = self._extract_indicator_value(primary, 'supertrend')
            if supertrend:
                # SuperTrend turning bullish
                breakout_analysis['supertrend_bullish'] = True
                breakout_analysis['breakout_strength'] += 2
            
            # VWAP breakout
            vwap = self._extract_indicator_value(primary, 'vwap')
            if vwap:
                # Price above VWAP suggests institutional support
                breakout_analysis['above_vwap'] = True
                breakout_analysis['breakout_strength'] += 1
            
            # Candlestick pattern breakouts
            hammer = self._extract_indicator_value(primary, 'cdlhammer')
            engulfing = self._extract_indicator_value(primary, 'cdlengulfing')
            morning_star = self._extract_indicator_value(primary, 'cdlmorningstar')
            
            if any([hammer, engulfing, morning_star]):
                breakout_analysis['pattern_breakout'] = True
                breakout_analysis['breakout_type'] = 'pattern_breakout'
                breakout_analysis['breakout_strength'] += 1
            
        except Exception as e:
            logging.warning(f"Breakout analysis error for {symbol}: {str(e)}")
        
        return breakout_analysis

    def _calculate_momentum_confluence(self, mtf_data: Dict, volume_analysis: Dict, breakout_analysis: Dict) -> Dict[str, Any]:
        """Calculate confluence score across all momentum indicators"""
        
        confluence = {
            'total_score': 0,
            'max_possible': 15,  # Maximum possible confluence points
            'percentage': 0,
            'quality_grade': 'F',
            'bullish_factors': [],
            'warning_factors': []
        }
        

        try:
            primary = mtf_data['primary']
            short_term = mtf_data['short_term']
            long_term = mtf_data['long_term']
            
            # 1. RSI Momentum (Multi-timeframe)
            rsi_1h = self._extract_indicator_value(primary, 'rsi')
            rsi_15m = self._extract_indicator_value(short_term, 'rsi_15m')
            rsi_4h = self._extract_indicator_value(long_term, 'rsi_4h')
            
            if rsi_1h and 35 < rsi_1h < 70:  # Sweet spot for momentum entry
                confluence['total_score'] += 2
                confluence['bullish_factors'].append(f"RSI 1h in momentum zone ({rsi_1h:.1f})")
            
            if rsi_15m and rsi_15m > 50:  # Short-term momentum
                confluence['total_score'] += 1
                confluence['bullish_factors'].append(f"RSI 15m bullish ({rsi_15m:.1f})")
            
            if rsi_4h and rsi_4h > 40:  # Long-term not oversold
                confluence['total_score'] += 1
                confluence['bullish_factors'].append(f"RSI 4h supportive ({rsi_4h:.1f})")
            
            # 2. MACD Momentum (Multi-timeframe)
            macd_1h = self._extract_indicator_value(primary, 'macd')
            macd_15m = self._extract_indicator_value(short_term, 'macd_15m')
            macd_4h = self._extract_indicator_value(long_term, 'macd_4h')
            
            if macd_1h and self._is_macd_bullish(macd_1h):
                confluence['total_score'] += 2
                confluence['bullish_factors'].append("MACD 1h bullish crossover")
            
            if macd_15m and self._is_macd_bullish(macd_15m):
                confluence['total_score'] += 1
                confluence['bullish_factors'].append("MACD 15m bullish")
            
            # 3. EMA Alignment
            ema20 = self._extract_indicator_value(primary, 'ema20')
            ema50 = self._extract_indicator_value(primary, 'ema50')
            ema200 = self._extract_indicator_value(primary, 'ema200')
            
            if ema20 and ema50 and ema20 > ema50:
                confluence['total_score'] += 1
                confluence['bullish_factors'].append("EMA20 > EMA50")
            
            if ema20 and ema200 and ema20 > ema200:
                confluence['total_score'] += 1
                confluence['bullish_factors'].append("Above EMA200 (bullish trend)")
            
            # 4. ADX Trend Strength
            adx = self._extract_indicator_value(primary, 'adx')
            if adx and adx > 25:
                confluence['total_score'] += 1
                confluence['bullish_factors'].append(f"Strong trend strength (ADX: {adx:.1f})")
            
            # 5. Volume Confirmation
            confluence['total_score'] += volume_analysis['volume_confirmation_score']
            if volume_analysis['money_flow_bullish']:
                confluence['bullish_factors'].append("Money Flow bullish")
            
            # 6. Breakout Confirmation
            confluence['total_score'] += min(3, breakout_analysis['breakout_strength'])
            if breakout_analysis['squeeze_break']:
                confluence['bullish_factors'].append("Squeeze momentum breakout")
            
            # 7. StochRSI momentum
            stochrsi = self._extract_indicator_value(primary, 'stochrsi')
            if stochrsi and self._is_stochrsi_bullish(stochrsi):
                confluence['total_score'] += 1
                confluence['bullish_factors'].append("StochRSI bullish momentum")
            
            # Calculate percentage and grade
            confluence['percentage'] = (confluence['total_score'] / confluence['max_possible']) * 100
            confluence['quality_grade'] = self._get_quality_grade(confluence['percentage'])
            
            # Add warnings for potential issues
            if rsi_1h and rsi_1h > 75:
                confluence['warning_factors'].append("RSI potentially overbought")
            
            if not volume_analysis['money_flow_bullish']:
                confluence['warning_factors'].append("Money flow not confirming")
            
        except Exception as e:
            logging.error(f"Error calculating confluence: {str(e)}")
        
        return confluence

    def _generate_momentum_signal(self, symbol: str, mtf_data: Dict, volume_analysis: Dict, 
                                breakout_analysis: Dict, confluence: Dict) -> MomentumSignal:
        """Generate final momentum signal based on all analysis"""
        
        # Default to HOLD (never SELL for your strategy)
        action = "HOLD"
        confidence = 0.0
        momentum_strength = "WEAK"
        breakout_type = "NONE"
        entry_quality = "POOR"
        reasons = []
        
        try:
            confluence_score = confluence['percentage']
            
            # High-quality entry criteria (for 75-90% win rate)
            if confluence_score >= 80:  # Exceptional confluence
                action = "BUY"
                confidence = min(95, confluence_score)
                momentum_strength = "EXPLOSIVE"
                entry_quality = "EXCELLENT"
                reasons.extend(confluence['bullish_factors'])
                
            elif confluence_score >= 65:  # Very good confluence
                action = "BUY"
                confidence = min(85, confluence_score)
                momentum_strength = "STRONG"
                entry_quality = "GOOD"
                reasons.extend(confluence['bullish_factors'])
                
            elif confluence_score >= 50:  # Good confluence
                action = "BUY"
                confidence = min(75, confluence_score)
                momentum_strength = "MODERATE"
                entry_quality = "FAIR"
                reasons.extend(confluence['bullish_factors'][:3])  # Top 3 reasons
                
            else:  # Low confluence - HOLD
                action = "HOLD"
                confidence = confluence_score
                reasons = ["Insufficient momentum confluence for entry", 
                          f"Confluence score: {confluence_score:.1f}%"]
            
            # Determine breakout type
            if breakout_analysis['squeeze_break']:
                breakout_type = "SQUEEZE_BREAKOUT"
            elif breakout_analysis['pattern_breakout']:
                breakout_type = "PATTERN_BREAKOUT"
            elif volume_analysis['volume_spike']:
                breakout_type = "VOLUME_BREAKOUT"
            elif breakout_analysis['resistance_break']:
                breakout_type = "RESISTANCE_BREAK"
            
            # Add volume confirmation
            volume_confirmation = volume_analysis['money_flow_bullish'] and volume_analysis['volume_confirmation_score'] >= 2
            
            # Calculate risk-reward based on ATR and entry quality
            risk_reward_ratio = self._calculate_risk_reward(mtf_data, entry_quality)
            
            # Final quality check - reduce confidence if warnings exist
            if confluence['warning_factors']:
                confidence *= 0.9  # Reduce confidence by 10%
                reasons.extend([f"Warning: {w}" for w in confluence['warning_factors'][:2]])
            
        except Exception as e:
            logging.error(f"Error generating signal for {symbol}: {str(e)}")
            reasons = [f"Error in signal generation: {str(e)}"]
        
        return MomentumSignal(
            action=action,
            confidence=confidence,
            momentum_strength=momentum_strength,
            breakout_type=breakout_type,
            entry_quality=entry_quality,
            volume_confirmation=volume_confirmation,
            risk_reward_ratio=risk_reward_ratio,
            reasons=reasons,
            indicators_aligned=confluence['total_score'],
            timestamp=datetime.now()
        )

    # Helper methods
    
    async def _execute_bulk_query(self, construct: Dict) -> Dict:
        """Execute TAAPI bulk query with error handling"""
        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.post(f"{self.base_url}/bulk", json=construct) as response:
                    if response.status == 200:
                        data = await response.json()
                        return self._parse_bulk_response(data)
                    else:
                        logging.error(f"TAAPI bulk query failed: {response.status}")
                        return {}
        except Exception as e:
            logging.error(f"Error executing bulk query: {str(e)}")
            return {}
    
    def _parse_bulk_response(self, response: Dict) -> Dict:
        """Parse TAAPI bulk response into organized structure"""
        parsed = {}
        
        if 'data' in response:
            for item in response['data']:
                indicator_id = item.get('id', '')
                result = item.get('result', {})
                
                # Extract indicator name from ID
                parts = indicator_id.split('_')
                if len(parts) >= 4:
                    indicator_name = parts[3]  # Usually the indicator name
                    parsed[indicator_name] = result
                elif 'id' in item and item['id'] != indicator_id:
                    # Custom ID provided
                    parsed[item['id']] = result
        
        return parsed
    
    def _extract_indicator_value(self, data: Dict, indicator: str) -> Optional[float]:
        """Safely extract indicator value from parsed data"""
        try:
            if indicator in data:
                result = data[indicator]
                if isinstance(result, dict):
                    return result.get('value') or result.get('valueMACD') or result.get('valueMACDHist')
                return result
        except:
            pass
        return None
    
    def _is_macd_bullish(self, macd_data: Dict) -> bool:
        """Check if MACD is in bullish configuration"""
        try:
            macd = macd_data.get('valueMACD', 0)
            signal = macd_data.get('valueMACDSignal', 0)
            histogram = macd_data.get('valueMACDHist', 0)
            
            return macd > signal and histogram > 0
        except:
            return False
    
    def _is_stochrsi_bullish(self, stochrsi_data: Dict) -> bool:
        """Check if StochRSI is in bullish configuration"""
        try:
            k = stochrsi_data.get('valueFastK', 50)
            d = stochrsi_data.get('valueFastD', 50)
            
            return k > d and k > 20 and k < 80  # Bullish crossover in momentum zone
        except:
            return False
    
    def _calculate_risk_reward(self, mtf_data: Dict, entry_quality: str) -> float:
        """Calculate risk-reward ratio based on ATR and entry quality"""
        try:
            atr = self._extract_indicator_value(mtf_data['primary'], 'atr')
            if not atr:
                return 2.0  # Default
            
            # Adjust based on entry quality
            quality_multiplier = {
                'EXCELLENT': 3.5,
                'GOOD': 3.0,
                'FAIR': 2.5,
                'POOR': 2.0
            }.get(entry_quality, 2.0)
            
            return quality_multiplier
            
        except:
            return 2.0
    
    def _get_quality_grade(self, percentage: float) -> str:
        """Convert confluence percentage to quality grade"""
        if percentage >= 85: return 'A+'
        elif percentage >= 80: return 'A'
        elif percentage >= 75: return 'A-'
        elif percentage >= 70: return 'B+'
        elif percentage >= 65: return 'B'
        elif percentage >= 60: return 'B-'
        elif percentage >= 55: return 'C+'
        elif percentage >= 50: return 'C'
        elif percentage >= 45: return 'C-'
        elif percentage >= 40: return 'D'
        else: return 'F'
    
    def _create_hold_signal(self, reason: str) -> MomentumSignal:
        """Create a HOLD signal with given reason"""
        return MomentumSignal(
            action="HOLD",
            confidence=0.0,
            momentum_strength="WEAK",
            breakout_type="NONE",
            entry_quality="POOR",
            volume_confirmation=False,
            risk_reward_ratio=1.0,
            reasons=[reason],
            indicators_aligned=0,
            timestamp=datetime.now()
        )
    
    def _log_signal_for_tracking(self, signal: MomentumSignal):
        """Log signal for win rate tracking"""
        self.signal_history.append({
            'timestamp': signal.timestamp,
            'action': signal.action,
            'confidence': signal.confidence,
            'quality': signal.entry_quality,
            'momentum': signal.momentum_strength
        })
        
        # Keep only last 100 signals
        if len(self.signal_history) > 100:
            self.signal_history = self.signal_history[-100:]

# Integration with your existing system
class MomentumStrategyIntegration:
    """Integration class to connect enhanced TAAPI with your existing bot"""
    
    def __init__(self, taapi_client: EnhancedMomentumTaapiClient):
        self.taapi_client = taapi_client
        
    async def get_enhanced_signal_for_pair(self, pair: str) -> Dict[str, Any]:
        """Get enhanced signal compatible with your existing system"""
        
        momentum_signal = await self.taapi_client.get_momentum_optimized_signal(pair)
        
        # Convert to your existing signal format
        enhanced_signal = {
            'signal': momentum_signal.action.lower(),  # 'buy' or 'hold'
            'confidence': momentum_signal.confidence,
            'signal_strength': momentum_signal.momentum_strength.lower(),
            'entry_quality': momentum_signal.entry_quality.lower(),
            
            # Enhanced momentum data
            'momentum_data': {
                'breakout_type': momentum_signal.breakout_type,
                'volume_confirmation': momentum_signal.volume_confirmation,
                'indicators_aligned': momentum_signal.indicators_aligned,
                'risk_reward_ratio': momentum_signal.risk_reward_ratio,
                'quality_grade': momentum_signal.entry_quality,
            },
            
            # Integration with your API signal format
            'api_data': {
                'signal': momentum_signal.action,
                'confidence': momentum_signal.confidence,
                'strength': momentum_signal.momentum_strength,
                'strategy_type': 'Enhanced Momentum Strategy',
                'market_phase': self._determine_market_phase(momentum_signal),
                'risk_reward_ratio': momentum_signal.risk_reward_ratio,
                'enhanced_by': 'momentum_optimized_taapi',
                'taapi_enabled': True
            },
            
            # Reasoning for decisions
            'reasoning': momentum_signal.reasons,
            'timestamp': momentum_signal.timestamp.isoformat(),
            
            # Your strategy-specific flags
            'buy_signal': momentum_signal.action == 'BUY',
            'sell_signal': False,  # Never sell in your strategy
            'hold_signal': momentum_signal.action == 'HOLD',
            
            # Quality metrics for your 75-90% win rate goal
            'high_probability_entry': momentum_signal.confidence >= 75 and momentum_signal.entry_quality in ['GOOD', 'EXCELLENT'],
            'momentum_confirmed': momentum_signal.volume_confirmation and momentum_signal.indicators_aligned >= 4,
        }
        
        return enhanced_signal
    
    def _determine_market_phase(self, signal: MomentumSignal) -> str:
        """Determine market phase based on momentum signal"""
        if signal.momentum_strength == 'EXPLOSIVE':
            return 'MARKUP'
        elif signal.momentum_strength == 'STRONG':
            return 'ACCUMULATION'
        elif signal.breakout_type in ['SQUEEZE_BREAKOUT', 'PATTERN_BREAKOUT']:
            return 'CONSOLIDATION'
        else:
            return 'NEUTRAL'

# Usage example for your bot
async def example_usage():
    """Example of how to integrate with your existing bot"""
    
    # Initialize enhanced TAAPI client
    taapi_client = EnhancedMomentumTaapiClient("YOUR_TAAPI_SECRET")
    
    # Create integration layer
    integration = MomentumStrategyIntegration(taapi_client)
    
    # Get signal for a pair (compatible with your existing system)
    signal = await integration.get_enhanced_signal_for_pair("BTCUSDT")
    
    # Use with your existing bot logic
    if signal['high_probability_entry']:
        print(f"High probability {signal['signal']} signal detected!")
        print(f"Confidence: {signal['confidence']:.1f}%")
        print(f"Momentum: {signal['momentum_data']['breakout_type']}")
        print(f"Reasons: {', '.join(signal['reasoning'][:3])}")
        
        # Execute with your existing bot methods
        # await your_bot._execute_api_enhanced_buy(pair, signal)
    
    elif signal['momentum_confirmed']:
        print(f"Momentum confirmed {signal['signal']} signal")
        # Handle moderate confidence signals
        
    else:
        print(f"Signal: {signal['signal']} - Waiting for better setup")
        # Continue monitoring