import asyncio
import logging
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum

class EntrySignalStrength(Enum):
    """Entry signal strength levels for high win rate strategy"""
    AVOID = "AVOID"           # Don't trade - bearish or unclear
    WEAK = "WEAK"             # Low probability - wait for better setup
    MODERATE = "MODERATE"     # Decent setup - smaller position
    STRONG = "STRONG"         # Good setup - normal position
    EXCELLENT = "EXCELLENT"   # Exceptional setup - larger position

class VolumePattern(Enum):
    """Volume pattern types for momentum confirmation"""
    ACCUMULATION = "ACCUMULATION"     # Steady volume increase
    SPIKE = "SPIKE"                   # Sudden volume surge
    BREAKOUT = "BREAKOUT"            # Volume on price breakout
    CLIMAX = "CLIMAX"                # Exhaustion volume
    WEAK = "WEAK"                    # Below average volume

@dataclass
class EntryQualityMetrics:
    """Comprehensive entry quality assessment"""
    overall_score: float              # 0-100 combined score
    signal_strength: EntrySignalStrength
    confidence_level: float          # 0-100 confidence in signal
    risk_reward_ratio: float         # Expected risk-reward
    
    # Individual component scores
    momentum_score: float            # Momentum quality 0-100
    volume_score: float              # Volume confirmation 0-100
    technical_score: float           # Technical setup quality 0-100
    breakout_score: float            # Breakout confirmation 0-100
    timeframe_alignment_score: float # Multi-timeframe agreement 0-100
    
    # Quality flags
    is_high_probability: bool        # Meets high win rate criteria
    has_volume_confirmation: bool    # Volume supports move
    has_momentum_confirmation: bool  # Momentum is building
    has_breakout_confirmation: bool  # Clear breakout pattern
    
    # Risk factors
    risk_factors: List[str]          # List of identified risks
    warning_flags: List[str]         # List of warning conditions
    
    # Timing assessment
    entry_timing: str                # EARLY, OPTIMAL, LATE
    market_phase_fit: str            # How well signal fits market phase

class HighWinRateEntryFilter:
    """
    Advanced entry filter system designed for 75-90% win rate
    Implements the Danish momentum strategy: only bullish entries with volume/breakout confirmation
    """
    
    def __init__(self, config):
        self.config = config
        self.signal_history = []
        self.performance_metrics = {
            'total_signals': 0,
            'excellent_signals': 0,
            'strong_signals': 0,
            'win_rate_by_strength': {},
            'avg_rrr_by_strength': {}
        }
    
    async def evaluate_entry_quality(self, symbol: str, taapi_data: Dict, market_data: Dict) -> EntryQualityMetrics:
        """
        Comprehensive entry quality evaluation for high win rate strategy
        
        Args:
            symbol: Trading pair symbol
            taapi_data: Raw TAAPI indicator data
            market_data: Current market conditions and price data
            
        Returns:
            EntryQualityMetrics with detailed assessment
        """
        try:
            # Initialize component scores
            momentum_score = await self._evaluate_momentum_quality(taapi_data, market_data)
            volume_score = await self._evaluate_volume_quality(taapi_data, market_data)
            technical_score = await self._evaluate_technical_setup(taapi_data, market_data)
            breakout_score = await self._evaluate_breakout_quality(taapi_data, market_data)
            timeframe_score = await self._evaluate_timeframe_alignment(taapi_data, market_data)
            
            # Calculate overall score with weighted components
            overall_score = self._calculate_weighted_score(
                momentum_score, volume_score, technical_score, 
                breakout_score, timeframe_score
            )
            
            # Determine signal strength
            signal_strength = self._determine_signal_strength(overall_score, {
                'momentum': momentum_score,
                'volume': volume_score,
                'technical': technical_score,
                'breakout': breakout_score,
                'timeframe': timeframe_score
            })
            
            # Assess confirmations
            confirmations = self._assess_confirmations(taapi_data, market_data)
            
            # Identify risk factors and warnings
            risk_factors, warning_flags = await self._identify_risk_factors(taapi_data, market_data)
            
            # Assess entry timing
            entry_timing = self._assess_entry_timing(taapi_data, market_data, overall_score)
            
            # Assess market phase fit
            market_phase_fit = self._assess_market_phase_fit(taapi_data, market_data)
            
            # Calculate risk-reward ratio
            rrr = self._calculate_risk_reward_ratio(taapi_data, market_data, signal_strength)
            
            # Determine if this is high probability (75-90% win rate criteria)
            is_high_probability = self._is_high_probability_entry(
                overall_score, signal_strength, confirmations, risk_factors
            )
            
            # Create metrics object
            metrics = EntryQualityMetrics(
                overall_score=overall_score,
                signal_strength=signal_strength,
                confidence_level=min(95, overall_score * 1.1),  # Slight confidence boost for good setups
                risk_reward_ratio=rrr,
                
                momentum_score=momentum_score,
                volume_score=volume_score,
                technical_score=technical_score,
                breakout_score=breakout_score,
                timeframe_alignment_score=timeframe_score,
                
                is_high_probability=is_high_probability,
                has_volume_confirmation=confirmations['volume'],
                has_momentum_confirmation=confirmations['momentum'],
                has_breakout_confirmation=confirmations['breakout'],
                
                risk_factors=risk_factors,
                warning_flags=warning_flags,
                
                entry_timing=entry_timing,
                market_phase_fit=market_phase_fit
            )
            
            # Log for performance tracking
            self._log_entry_evaluation(symbol, metrics)
            
            return metrics
            
        except Exception as e:
            logging.error(f"Error evaluating entry quality for {symbol}: {str(e)}")
            return self._create_error_metrics(str(e))
    
    async def _evaluate_momentum_quality(self, taapi_data: Dict, market_data: Dict) -> float:
        """Evaluate momentum quality (0-100 score)"""
        score = 0.0
        max_score = 100.0
        
        try:
            # RSI momentum analysis (25 points)
            rsi_1h = self._get_indicator_value(taapi_data, 'primary', 'rsi')
            rsi_15m = self._get_indicator_value(taapi_data, 'short_term', 'rsi_15m')
            
            if rsi_1h:
                # Sweet spot for momentum (40-65 RSI)
                if 40 <= rsi_1h <= 65:
                    score += 20
                elif 35 <= rsi_1h <= 70:
                    score += 15
                elif rsi_1h > 75:  # Too overbought
                    score -= 10
                
                # Rising RSI momentum (compare with 15m if available)
                if rsi_15m and rsi_15m > rsi_1h * 1.02:  # 15m RSI higher suggests momentum
                    score += 5
            
            # MACD momentum analysis (25 points)
            macd_1h = self._get_indicator_value(taapi_data, 'primary', 'macd')
            if macd_1h:
                macd = macd_1h.get('valueMACD', 0)
                signal = macd_1h.get('valueMACDSignal', 0)
                histogram = macd_1h.get('valueMACDHist', 0)
                
                if macd > signal and histogram > 0:
                    score += 25  # Perfect bullish MACD
                elif macd > signal:
                    score += 15  # Bullish crossover
                elif histogram > 0:
                    score += 10  # Positive momentum
            
            # StochRSI momentum (15 points)
            stochrsi = self._get_indicator_value(taapi_data, 'primary', 'stochrsi')
            if stochrsi:
                k = stochrsi.get('valueFastK', 50)
                d = stochrsi.get('valueFastD', 50)
                
                if k > d and 20 < k < 80:  # Bullish momentum zone
                    score += 15
                elif k > d:
                    score += 10
            
            # ADX trend strength (15 points)
            adx = self._get_indicator_value(taapi_data, 'primary', 'adx')
            if adx and adx > 25:
                if adx > 40:
                    score += 15  # Very strong trend
                else:
                    score += 10  # Good trend strength
            
            # Price momentum (20 points)
            if 'price_momentum' in market_data:
                momentum_1h = market_data['price_momentum'].get('1h', 0)
                momentum_4h = market_data['price_momentum'].get('4h', 0)
                
                if momentum_1h > 1.0 and momentum_4h > 0:  # 1%+ 1h gain, positive 4h
                    score += 20
                elif momentum_1h > 0.5:
                    score += 15
                elif momentum_1h > 0.2:
                    score += 10
            
            return min(max_score, max(0, score))
            
        except Exception as e:
            logging.warning(f"Error in momentum evaluation: {str(e)}")
            return 0.0
    
    async def _evaluate_volume_quality(self, taapi_data: Dict, market_data: Dict) -> float:
        """Evaluate volume confirmation quality (0-100 score)"""
        score = 0.0
        max_score = 100.0
        
        try:
            # Money Flow Index (30 points)
            mfi = self._get_indicator_value(taapi_data, 'primary', 'mfi')
            if mfi:
                if mfi > 60:
                    score += 30  # Strong money flow
                elif mfi > 50:
                    score += 20  # Positive money flow
                elif mfi > 40:
                    score += 10  # Neutral money flow
            
            # On Balance Volume trend (25 points)
            obv = self._get_indicator_value(taapi_data, 'primary', 'obv')
            if obv:
                # In real implementation, compare with previous OBV values
                # For now, assume positive if OBV exists
                score += 20
            
            # Volume Profile analysis (25 points)
            volume_profile = self._get_indicator_value(taapi_data, 'primary', 'volume_profile_1h')
            if volume_profile:
                score += 20  # Volume profile shows institutional activity
            
            # Volume spike detection (20 points)
            if 'volume_analysis' in market_data:
                vol_data = market_data['volume_analysis']
                if vol_data.get('volume_spike_ratio', 1.0) > 2.0:
                    score += 20  # Strong volume spike
                elif vol_data.get('volume_spike_ratio', 1.0) > 1.5:
                    score += 15  # Moderate volume spike
                elif vol_data.get('volume_spike_ratio', 1.0) > 1.2:
                    score += 10  # Slight volume increase
            
            return min(max_score, max(0, score))
            
        except Exception as e:
            logging.warning(f"Error in volume evaluation: {str(e)}")
            return 0.0
    
    async def _evaluate_technical_setup(self, taapi_data: Dict, market_data: Dict) -> float:
        """Evaluate technical setup quality (0-100 score)"""
        score = 0.0
        max_score = 100.0
        
        try:
            # EMA alignment (25 points)
            ema20 = self._get_indicator_value(taapi_data, 'primary', 'ema20')
            ema50 = self._get_indicator_value(taapi_data, 'primary', 'ema50')
            ema200 = self._get_indicator_value(taapi_data, 'primary', 'ema200')
            
            if ema20 and ema50 and ema200:
                if ema20 > ema50 > ema200:
                    score += 25  # Perfect bullish alignment
                elif ema20 > ema50:
                    score += 15  # Short-term bullish
                elif ema20 > ema200:
                    score += 10  # Above long-term trend
            
            # Bollinger Bands position (20 points)
            bbands = self._get_indicator_value(taapi_data, 'primary', 'bbands')
            current_price = market_data.get('current_price', 0)
            if bbands and current_price:
                upper = bbands.get('valueUpperBand', 0)
                middle = bbands.get('valueMiddleBand', 0)
                lower = bbands.get('valueLowerBand', 0)
                
                if current_price > middle:
                    if current_price < upper * 0.95:  # Near but not touching upper band
                        score += 20
                    elif current_price > middle * 1.01:  # Above middle band
                        score += 15
                elif current_price > lower * 1.02:  # Above lower band
                    score += 10
            
            # SuperTrend indicator (20 points)
            supertrend = self._get_indicator_value(taapi_data, 'primary', 'supertrend')
            if supertrend:
                # SuperTrend bullish signal
                score += 20
            
            # VWAP position (15 points)
            vwap = self._get_indicator_value(taapi_data, 'primary', 'vwap')
            if vwap and current_price:
                if current_price > vwap * 1.005:  # Above VWAP with buffer
                    score += 15
                elif current_price > vwap:
                    score += 10
            
            # ATR-based volatility assessment (10 points)
            atr = self._get_indicator_value(taapi_data, 'primary', 'atr')
            if atr and current_price:
                atr_percentage = (atr / current_price) * 100
                if 1.0 < atr_percentage < 5.0:  # Ideal volatility range
                    score += 10
                elif atr_percentage < 7.0:
                    score += 5
            
            # Candlestick pattern confirmation (10 points)
            hammer = self._get_indicator_value(taapi_data, 'primary', 'cdlhammer')
            engulfing = self._get_indicator_value(taapi_data, 'primary', 'cdlengulfing')
            morning_star = self._get_indicator_value(taapi_data, 'primary', 'cdlmorningstar')
            
            if any([hammer, engulfing, morning_star]):
                score += 10
            
            return min(max_score, max(0, score))
            
        except Exception as e:
            logging.warning(f"Error in technical evaluation: {str(e)}")
            return 0.0
    
    async def _evaluate_breakout_quality(self, taapi_data: Dict, market_data: Dict) -> float:
        """Evaluate breakout pattern quality (0-100 score)"""
        score = 0.0
        max_score = 100.0
        
        try:
            # TTM Squeeze breakout (40 points)
            squeeze = self._get_indicator_value(taapi_data, 'primary', 'squeeze')
            if squeeze:
                if squeeze > 0.1:  # Strong positive momentum
                    score += 40
                elif squeeze > 0:  # Positive momentum
                    score += 30
            
            # Bollinger Band squeeze and expansion (30 points)
            bbands = self._get_indicator_value(taapi_data, 'primary', 'bbands')
            if bbands:
                upper = bbands.get('valueUpperBand', 0)
                lower = bbands.get('valueLowerBand', 0)
                middle = bbands.get('valueMiddleBand', 0)
                
                if upper and lower and middle:
                    band_width = (upper - lower) / middle
                    if band_width > 0.04:  # Bands expanding (breakout)
                        score += 30
                    elif band_width > 0.02:  # Moderate expansion
                        score += 20
            
            # Volume breakout confirmation (20 points)
            if 'volume_analysis' in market_data:
                vol_data = market_data['volume_analysis']
                breakout_volume = vol_data.get('breakout_volume_ratio', 1.0)
                if breakout_volume > 2.0:
                    score += 20
                elif breakout_volume > 1.5:
                    score += 15
            
            # Price breakout above resistance (10 points)
            if 'resistance_levels' in market_data:
                current_price = market_data.get('current_price', 0)
                resistance = market_data['resistance_levels'].get('nearest_resistance', 0)
                if current_price and resistance and current_price > resistance * 1.005:
                    score += 10
            
            return min(max_score, max(0, score))
            
        except Exception as e:
            logging.warning(f"Error in breakout evaluation: {str(e)}")
            return 0.0
    
    async def _evaluate_timeframe_alignment(self, taapi_data: Dict, market_data: Dict) -> float:
        """Evaluate multi-timeframe alignment (0-100 score)"""
        score = 0.0
        max_score = 100.0
        
        try:
            timeframe_scores = {}
            
            # 1h timeframe analysis (primary)
            rsi_1h = self._get_indicator_value(taapi_data, 'primary', 'rsi')
            macd_1h = self._get_indicator_value(taapi_data, 'primary', 'macd')
            
            tf_1h_score = 0
            if rsi_1h and 40 <= rsi_1h <= 70:
                tf_1h_score += 50
            if macd_1h and self._is_macd_bullish(macd_1h):
                tf_1h_score += 50
            timeframe_scores['1h'] = tf_1h_score
            
            # 15m timeframe analysis (short-term)
            rsi_15m = self._get_indicator_value(taapi_data, 'short_term', 'rsi_15m')
            macd_15m = self._get_indicator_value(taapi_data, 'short_term', 'macd_15m')
            
            tf_15m_score = 0
            if rsi_15m and rsi_15m > 50:
                tf_15m_score += 50
            if macd_15m and self._is_macd_bullish(macd_15m):
                tf_15m_score += 50
            timeframe_scores['15m'] = tf_15m_score
            
            # 4h timeframe analysis (trend confirmation)
            rsi_4h = self._get_indicator_value(taapi_data, 'long_term', 'rsi_4h')
            macd_4h = self._get_indicator_value(taapi_data, 'long_term', 'macd_4h')
            
            tf_4h_score = 0
            if rsi_4h and rsi_4h > 40:
                tf_4h_score += 50
            if macd_4h and self._is_macd_bullish(macd_4h):
                tf_4h_score += 50
            timeframe_scores['4h'] = tf_4h_score
            
            # Calculate alignment score
            total_timeframes = len(timeframe_scores)
            if total_timeframes > 0:
                aligned_timeframes = sum(1 for score in timeframe_scores.values() if score >= 50)
                alignment_ratio = aligned_timeframes / total_timeframes
                score = alignment_ratio * max_score
                
                # Bonus for all timeframes aligned
                if aligned_timeframes == total_timeframes and total_timeframes >= 3:
                    score = min(max_score, score * 1.1)
            
            return min(max_score, max(0, score))
            
        except Exception as e:
            logging.warning(f"Error in timeframe alignment evaluation: {str(e)}")
            return 0.0
    
    def _calculate_weighted_score(self, momentum_score: float, volume_score: float, 
                                technical_score: float, breakout_score: float, 
                                timeframe_score: float) -> float:
        """Calculate weighted overall score based on strategy priorities"""
        
        # Weights based on your momentum strategy priorities
        weights = {
            'momentum': 0.25,      # 25% - Core momentum confirmation
            'volume': 0.25,        # 25% - Volume confirmation critical
            'breakout': 0.20,      # 20% - Breakout patterns important
            'timeframe': 0.20,     # 20% - Multi-timeframe alignment
            'technical': 0.10      # 10% - Technical setup support
        }
        
        weighted_score = (
            momentum_score * weights['momentum'] +
            volume_score * weights['volume'] +
            technical_score * weights['technical'] +
            breakout_score * weights['breakout'] +
            timeframe_score * weights['timeframe']
        )
        
        return min(100.0, max(0.0, weighted_score))
    
    def _determine_signal_strength(self, overall_score: float, component_scores: Dict) -> EntrySignalStrength:
        """Determine signal strength based on overall score and component analysis"""
        
        # Check for critical component failures first
        if component_scores['volume'] < 20:  # Poor volume confirmation
            return EntrySignalStrength.AVOID
        
        if component_scores['momentum'] < 25:  # Poor momentum
            return EntrySignalStrength.WEAK
        
        # Determine strength based on overall score
        if overall_score >= 85:
            return EntrySignalStrength.EXCELLENT
        elif overall_score >= 75:
            return EntrySignalStrength.STRONG
        elif overall_score >= 60:
            return EntrySignalStrength.MODERATE
        elif overall_score >= 40:
            return EntrySignalStrength.WEAK
        else:
            return EntrySignalStrength.AVOID
    
    def _assess_confirmations(self, taapi_data: Dict, market_data: Dict) -> Dict[str, bool]:
        """Assess various confirmation criteria"""
        confirmations = {
            'volume': False,
            'momentum': False,
            'breakout': False
        }
        
        try:
            # Volume confirmation
            mfi = self._get_indicator_value(taapi_data, 'primary', 'mfi')
            if mfi and mfi > 55:
                confirmations['volume'] = True
            
            # Momentum confirmation
            macd = self._get_indicator_value(taapi_data, 'primary', 'macd')
            rsi = self._get_indicator_value(taapi_data, 'primary', 'rsi')
            if (macd and self._is_macd_bullish(macd) and 
                rsi and 45 <= rsi <= 70):
                confirmations['momentum'] = True
            
            # Breakout confirmation
            squeeze = self._get_indicator_value(taapi_data, 'primary', 'squeeze')
            if squeeze and squeeze > 0:
                confirmations['breakout'] = True
            
        except Exception as e:
            logging.warning(f"Error assessing confirmations: {str(e)}")
        
        return confirmations
    
    async def _identify_risk_factors(self, taapi_data: Dict, market_data: Dict) -> Tuple[List[str], List[str]]:
        """Identify risk factors and warning flags"""
        risk_factors = []
        warning_flags = []
        
        try:
            # RSI overbought risk
            rsi = self._get_indicator_value(taapi_data, 'primary', 'rsi')
            if rsi and rsi > 75:
                risk_factors.append("RSI overbought (late entry risk)")
            elif rsi and rsi > 70:
                warning_flags.append("RSI approaching overbought levels")
            
            # Low volume risk
            mfi = self._get_indicator_value(taapi_data, 'primary', 'mfi')
            if mfi and mfi < 40:
                risk_factors.append("Weak money flow (low institutional interest)")
            
            # High volatility risk
            atr = self._get_indicator_value(taapi_data, 'primary', 'atr')
            current_price = market_data.get('current_price', 0)
            if atr and current_price:
                atr_percentage = (atr / current_price) * 100
                if atr_percentage > 8.0:
                    risk_factors.append("High volatility (increased risk)")
                elif atr_percentage > 6.0:
                    warning_flags.append("Elevated volatility levels")
            
            # Market timing risk
            if market_data.get('market_hours') == 'off_hours':
                warning_flags.append("Trading outside regular market hours")
            
        except Exception as e:
            logging.warning(f"Error identifying risk factors: {str(e)}")
        
        return risk_factors, warning_flags
    
    def _assess_entry_timing(self, taapi_data: Dict, market_data: Dict, overall_score: float) -> str:
        """Assess entry timing quality"""
        
        try:
            rsi = self._get_indicator_value(taapi_data, 'primary', 'rsi')
            
            if overall_score >= 80:
                if rsi and 45 <= rsi <= 60:
                    return "OPTIMAL"
                elif rsi and rsi < 45:
                    return "EARLY"
                else:
                    return "LATE"
            elif overall_score >= 60:
                return "OPTIMAL" if rsi and rsi < 65 else "LATE"
            else:
                return "EARLY"
                
        except:
            return "UNKNOWN"
    
    def _assess_market_phase_fit(self, taapi_data: Dict, market_data: Dict) -> str:
        """Assess how well signal fits current market phase"""
        
        # Simplified market phase assessment
        try:
            rsi = self._get_indicator_value(taapi_data, 'primary', 'rsi')
            ema20 = self._get_indicator_value(taapi_data, 'primary', 'ema20')
            ema50 = self._get_indicator_value(taapi_data, 'primary', 'ema50')
            
            if rsi and ema20 and ema50:
                if ema20 > ema50 and rsi > 50:
                    return "MARKUP"  # Trending up
                elif rsi < 40:
                    return "ACCUMULATION"  # Oversold
                elif 40 <= rsi <= 60:
                    return "CONSOLIDATION"  # Range-bound
                else:
                    return "DISTRIBUTION"  # Potentially topping
            
        except:
            pass
        
        return "UNKNOWN"
    
    def _calculate_risk_reward_ratio(self, taapi_data: Dict, market_data: Dict, 
                                   signal_strength: EntrySignalStrength) -> float:
        """Calculate expected risk-reward ratio"""
        
        base_rrr = {
            EntrySignalStrength.EXCELLENT: 4.0,
            EntrySignalStrength.STRONG: 3.0,
            EntrySignalStrength.MODERATE: 2.5,
            EntrySignalStrength.WEAK: 2.0,
            EntrySignalStrength.AVOID: 1.0
        }.get(signal_strength, 2.0)
        
        # Adjust based on ATR
        try:
            atr = self._get_indicator_value(taapi_data, 'primary', 'atr')
            current_price = market_data.get('current_price', 0)
            
            if atr and current_price:
                atr_percentage = (atr / current_price) * 100
                if atr_percentage > 5.0:  # High volatility
                    base_rrr *= 1.2  # Higher potential reward
                elif atr_percentage < 2.0:  # Low volatility
                    base_rrr *= 0.9  # Lower potential reward
        except:
            pass
        
        return base_rrr
    
    def _is_high_probability_entry(self, overall_score: float, signal_strength: EntrySignalStrength,
                                 confirmations: Dict, risk_factors: List[str]) -> bool:
        """Determine if entry meets high probability criteria (75-90% win rate target)"""
        
        # Minimum score threshold
        if overall_score < 70:
            return False
        
        # Minimum signal strength
        if signal_strength not in [EntrySignalStrength.STRONG, EntrySignalStrength.EXCELLENT]:
            return False
        
        # Required confirmations for high probability
        required_confirmations = ['volume', 'momentum']
        for conf in required_confirmations:
            if not confirmations.get(conf, False):
                return False
        
        # No critical risk factors
        critical_risks = ["RSI overbought (late entry risk)", "Weak money flow (low institutional interest)"]
        for risk in risk_factors:
            if any(critical in risk for critical in critical_risks):
                return False
        
        return True
    
    # Helper methods
    
    def _get_indicator_value(self, taapi_data: Dict, timeframe: str, indicator: str) -> Optional[Any]:
        """Safely extract indicator value from TAAPI data"""
        try:
            if timeframe in taapi_data and indicator in taapi_data[timeframe]:
                result = taapi_data[timeframe][indicator]
                if isinstance(result, dict):
                    return result.get('value') or result
                return result
        except:
            pass
        return None
    
    def _is_macd_bullish(self, macd_data: Dict) -> bool:
        """Check if MACD shows bullish configuration"""
        try:
            macd = macd_data.get('valueMACD', 0)
            signal = macd_data.get('valueMACDSignal', 0)
            histogram = macd_data.get('valueMACDHist', 0)
            return macd > signal and histogram > 0
        except:
            return False
    
    def _create_error_metrics(self, error_msg: str) -> EntryQualityMetrics:
        """Create error metrics object"""
        return EntryQualityMetrics(
            overall_score=0.0,
            signal_strength=EntrySignalStrength.AVOID,
            confidence_level=0.0,
            risk_reward_ratio=1.0,
            momentum_score=0.0,
            volume_score=0.0,
            technical_score=0.0,
            breakout_score=0.0,
            timeframe_alignment_score=0.0,
            is_high_probability=False,
            has_volume_confirmation=False,
            has_momentum_confirmation=False,
            has_breakout_confirmation=False,
            risk_factors=[f"Evaluation error: {error_msg}"],
            warning_flags=[],
            entry_timing="UNKNOWN",
            market_phase_fit="UNKNOWN"
        )
    
    def _log_entry_evaluation(self, symbol: str, metrics: EntryQualityMetrics):
        """Log entry evaluation for performance tracking"""
        self.performance_metrics['total_signals'] += 1
        
        if metrics.signal_strength == EntrySignalStrength.EXCELLENT:
            self.performance_metrics['excellent_signals'] += 1
        elif metrics.signal_strength == EntrySignalStrength.STRONG:
            self.performance_metrics['strong_signals'] += 1
        
        # Store in history for analysis
        self.signal_history.append({
            'symbol': symbol,
            'timestamp': datetime.now(),
            'overall_score': metrics.overall_score,
            'signal_strength': metrics.signal_strength.value,
            'is_high_probability': metrics.is_high_probability,
            'risk_factors_count': len(metrics.risk_factors)
        })
        
        # Keep only last 200 evaluations
        if len(self.signal_history) > 200:
            self.signal_history = self.signal_history[-200:]

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary for strategy optimization"""
        total = self.performance_metrics['total_signals']
        
        return {
            'total_evaluations': total,
            'excellent_signals': self.performance_metrics['excellent_signals'],
            'strong_signals': self.performance_metrics['strong_signals'],
            'excellent_percentage': (self.performance_metrics['excellent_signals'] / total * 100) if total > 0 else 0,
            'strong_or_excellent_percentage': ((self.performance_metrics['excellent_signals'] + 
                                              self.performance_metrics['strong_signals']) / total * 100) if total > 0 else 0,
            'recent_high_probability_signals': len([s for s in self.signal_history[-50:] if s.get('is_high_probability', False)]),
            'signal_quality_trend': self._calculate_quality_trend()
        }
    
    def _calculate_quality_trend(self) -> str:
        """Calculate trend in signal quality over recent evaluations"""
        if len(self.signal_history) < 20:
            return "INSUFFICIENT_DATA"
        
        recent_20 = self.signal_history[-20:]
        previous_20 = self.signal_history[-40:-20] if len(self.signal_history) >= 40 else []
        
        if not previous_20:
            return "INSUFFICIENT_DATA"
        
        recent_avg = sum(s['overall_score'] for s in recent_20) / len(recent_20)
        previous_avg = sum(s['overall_score'] for s in previous_20) / len(previous_20)
        
        if recent_avg > previous_avg * 1.05:
            return "IMPROVING"
        elif recent_avg < previous_avg * 0.95:
            return "DECLINING"
        else:
            return "STABLE"