# nebula_integration.py
import asyncio
import aiohttp
import logging
import json
import time
import uuid
from typing import Dict, Any, Optional, List, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import config


class NebulaActionType(Enum):
    TRADING_ADVICE = "trading_advice"
    POSITION_UPDATE = "position_update"
    RISK_ASSESSMENT = "risk_assessment"
    ORDER_LADDER = "order_ladder"
    STOP_LOSS_UPDATE = "stop_loss_update"
    MARKET_ANALYSIS = "market_analysis"

@dataclass
class NebulaResponse:
    message: str
    actions: List[Dict[str, Any]]
    session_id: str
    request_id: str
    timestamp: float
    success: bool = True
    error: Optional[str] = None

@dataclass
class TradingAdvice:
    action: str  # "buy", "sell", "hold", "close"
    pair: str
    confidence: float
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    position_size: Optional[float] = None
    rationale: str = ""
    timeframe: str = "immediate"
    risk_level: str = "medium"

class NebulaAIIntegration:
    
    
    def __init__(self, secret_key: str, trading_bot_instance=None):
        self.secret_key = secret_key
        self.base_url = "https://nebula-api.thirdweb.com"
        self.trading_bot = trading_bot_instance
        
        # Session management
        self.session_id = str(uuid.uuid4())
        self.request_history = []
        self.response_cache = {}
        
        # Rate limiting and reliability
        self.last_request_time = 0
        self.min_request_interval = 1.0  # 1 second between requests
        self.max_retries = 3
        self.timeout = 5.0
        
        # Performance tracking
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.average_response_time = 0
        
        # Fallback configuration
        self.fallback_enabled = True
        self.consecutive_failures = 0
        self.max_consecutive_failures = 3
        self.cooldown_period = 300  # 5 minutes
        self.last_failure_time = 0
        
        logging.info(f"Nebula AI Integration initialized with session: {self.session_id}")

    async def _make_api_request(self, message: str, context: Dict[str, Any] = None) -> NebulaResponse:
        """Make a request to Nebula API with proper error handling and rate limiting"""
        
        # Rate limiting
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.min_request_interval:
            await asyncio.sleep(self.min_request_interval - time_since_last)
        
        # Check if we're in cooldown period
        if self.consecutive_failures >= self.max_consecutive_failures:
            if current_time - self.last_failure_time < self.cooldown_period:
                raise Exception(f"Nebula API in cooldown period ({self.cooldown_period}s)")
        
        request_start = time.time()
        self.total_requests += 1
        
        # Prepare request payload
        payload = {
            "message": message,
            "stream": False,
            "session_id": self.session_id
        }
        
        # Add context if provided
        if context:
            payload["context"] = json.dumps(context)
        
        headers = {
            "Content-Type": "application/json",
            "x-secret-key": self.secret_key
        }
        
        # Make the request with retries
        for attempt in range(self.max_retries):
            try:
                async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout)) as session:
                    async with session.post(f"{self.base_url}/chat", 
                                          headers=headers, 
                                          json=payload) as response:
                        
                        response_time = time.time() - request_start
                        self.last_request_time = time.time()
                        
                        if response.status == 200:
                            data = await response.json()
                            
                            # Update performance metrics
                            self.successful_requests += 1
                            self.consecutive_failures = 0
                            self._update_average_response_time(response_time)
                            
                            nebula_response = NebulaResponse(
                                message=data.get("message", ""),
                                actions=data.get("actions", []),
                                session_id=data.get("session_id", self.session_id),
                                request_id=data.get("request_id", str(uuid.uuid4())),
                                timestamp=current_time,
                                success=True
                            )
                            
                            # Log successful request
                            logging.debug(f"Nebula API success: {response_time:.2f}s, {len(data.get('message', ''))[:100]}...")
                            
                            return nebula_response
                            
                        elif response.status == 401:
                            error_msg = "Nebula API authentication failed - check secret key"
                            logging.error(error_msg)
                            raise Exception(error_msg)
                            
                        elif response.status == 422:
                            error_data = await response.json()
                            error_msg = f"Nebula API validation error: {error_data}"
                            logging.error(error_msg)
                            raise Exception(error_msg)
                            
                        else:
                            error_msg = f"Nebula API HTTP error: {response.status}"
                            logging.warning(f"{error_msg}, attempt {attempt + 1}/{self.max_retries}")
                            if attempt == self.max_retries - 1:
                                raise Exception(error_msg)
                            
            except asyncio.TimeoutError:
                error_msg = f"Nebula API timeout after {self.timeout}s"
                logging.warning(f"{error_msg}, attempt {attempt + 1}/{self.max_retries}")
                if attempt == self.max_retries - 1:
                    raise Exception(error_msg)
                    
            except Exception as e:
                if attempt == self.max_retries - 1:
                    self.failed_requests += 1
                    self.consecutive_failures += 1
                    self.last_failure_time = time.time()
                    
                    return NebulaResponse(
                        message="",
                        actions=[],
                        session_id=self.session_id,
                        request_id=str(uuid.uuid4()),
                        timestamp=current_time,
                        success=False,
                        error=str(e)
                    )
                    
                await asyncio.sleep(1 * (attempt + 1))  # Exponential backoff
    
    def _update_average_response_time(self, response_time: float):
        """Update rolling average response time"""
        if self.successful_requests == 1:
            self.average_response_time = response_time
        else:
            # Exponentially weighted moving average
            alpha = 0.1
            self.average_response_time = alpha * response_time + (1 - alpha) * self.average_response_time

    async def get_trading_decision(self, pair: str, current_position: Optional[Dict] = None, 
                                 market_context: Optional[Dict] = None) -> TradingAdvice:
        """Get trading advice for a specific pair"""
        
        # Prepare context
        context = {
            "pair": pair,
            "current_time": datetime.now().isoformat(),
            "bot_type": "crypto_trading",
            "request_type": "trading_decision"
        }
        
        if current_position:
            context["current_position"] = current_position
            
        if market_context:
            context["market_context"] = market_context
            
        # Get current market data for context
        try:
            current_price = await self._get_current_price(pair)
            if current_price:
                context["current_price"] = current_price
        except:
            pass
        
        # Construct the message
        if current_position:
            message = f"I have an open {current_position.get('side', 'long')} position in {pair}. Should I hold, close, or modify it? Current P/L: {current_position.get('pnl', 'unknown')}%. Provide specific trading advice with entry/exit levels."
        else:
            message = f"Should I enter a position in {pair} right now? If yes, provide entry price, stop loss, take profit, and position size recommendation. If no, explain why and suggest monitoring levels."
        
        # Make the API call
        response = await self._make_api_request(message, context)
        
        if not response.success:
            logging.error(f"Nebula API failed for {pair}: {response.error}")
            # Return neutral advice as fallback
            return TradingAdvice(
                action="hold",
                pair=pair,
                confidence=0.0,
                rationale=f"Nebula API unavailable: {response.error}"
            )
        
        # Parse the response into trading advice
        return self._parse_trading_advice(response, pair)
    
    async def get_portfolio_assessment(self, positions: Dict[str, Any]) -> Dict[str, Any]:
        """Get overall portfolio risk and performance assessment"""
        
        context = {
            "request_type": "portfolio_assessment",
            "positions": positions,
            "total_positions": len(positions),
            "timestamp": datetime.now().isoformat()
        }
        
        # Calculate portfolio metrics
        total_value = sum(pos.get('value', 0) for pos in positions.values())
        total_pnl = sum(pos.get('pnl', 0) for pos in positions.values())
        
        message = f"Analyze my current crypto trading portfolio. I have {len(positions)} open positions worth ${total_value:.2f} total with {total_pnl:.2f}% overall P/L. Assess risk levels, suggest adjustments, and identify any positions that should be closed or modified."
        
        response = await self._make_api_request(message, context)
        
        if not response.success:
            return {"error": response.error, "success": False}
        
        return self._parse_portfolio_assessment(response)
    
    async def get_market_timing_advice(self, pairs: List[str], timeframe: str = "1h") -> Dict[str, Any]:
        """Get market timing advice for multiple pairs"""
        
        context = {
            "request_type": "market_timing",
            "pairs": pairs,
            "timeframe": timeframe,
            "market_analysis": await self._get_market_overview()
        }
        
        message = f"Analyze current market conditions for these crypto pairs: {', '.join(pairs)}. Provide market timing advice - which pairs are in good entry zones, which should be avoided, and overall market sentiment for the {timeframe} timeframe."
        
        response = await self._make_api_request(message, context)
        
        if not response.success:
            return {"error": response.error, "success": False}
        
        return self._parse_market_timing(response, pairs)
    
    async def update_stop_losses(self, positions: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
        """Get updated stop loss recommendations for current positions"""
        
        context = {
            "request_type": "stop_loss_update",
            "positions": positions
        }
        
        message = f"Review my current stop losses for {len(positions)} crypto positions. Suggest optimal stop loss levels based on current market conditions, volatility, and position performance. Consider both trailing stops and fixed stops."
        
        response = await self._make_api_request(message, context)
        
        if not response.success:
            return {"error": response.error}
        
        return self._parse_stop_loss_updates(response, positions)
    
    def _parse_trading_advice(self, response: NebulaResponse, pair: str) -> TradingAdvice:
        """Parse Nebula response into structured trading advice"""
        
        message = response.message.lower()
        
        # Extract action
        action = "hold"  # default
        if any(word in message for word in ["buy", "long", "enter", "purchase"]):
            action = "buy"
        elif any(word in message for word in ["sell", "short", "exit", "close"]):
            action = "sell"
        elif "hold" in message or "wait" in message:
            action = "hold"
        
        # Extract confidence (look for percentages or confidence words)
        confidence = 0.5  # default
        try:
            import re
            confidence_matches = re.findall(r'(\d+)%?\s*(?:confidence|certain|sure)', message)
            if confidence_matches:
                confidence = float(confidence_matches[0]) / 100
            elif "high confidence" in message or "strongly" in message:
                confidence = 0.8
            elif "low confidence" in message or "uncertain" in message:
                confidence = 0.3
            elif "moderate" in message or "medium" in message:
                confidence = 0.6
        except:
            pass
        
        # Extract price levels
        entry_price = self._extract_price(message, ["entry", "enter at", "buy at"])
        stop_loss = self._extract_price(message, ["stop loss", "stop at", "stop"])
        take_profit = self._extract_price(message, ["take profit", "target", "tp"])
        
        # Extract position size
        position_size = None
        try:
            size_matches = re.findall(r'(\d+(?:\.\d+)?)%?\s*(?:position|size|allocation)', message)
            if size_matches:
                position_size = float(size_matches[0])
        except:
            pass
        
        # Risk level assessment
        risk_level = "medium"
        if any(word in message for word in ["high risk", "risky", "volatile"]):
            risk_level = "high"
        elif any(word in message for word in ["low risk", "safe", "conservative"]):
            risk_level = "low"
        
        return TradingAdvice(
            action=action,
            pair=pair,
            confidence=confidence,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            position_size=position_size,
            rationale=response.message,
            risk_level=risk_level
        )
    
    def _extract_price(self, text: str, keywords: List[str]) -> Optional[float]:
        """Extract price from text based on keywords"""
        try:
            import re
            for keyword in keywords:
                pattern = f"{keyword}[:\s]*\\$?([0-9]+(?:\\.[0-9]+)?)"
                matches = re.findall(pattern, text, re.IGNORECASE)
                if matches:
                    return float(matches[0])
        except:
            pass
        return None
    
    def _parse_portfolio_assessment(self, response: NebulaResponse) -> Dict[str, Any]:
        """Parse portfolio assessment response"""
        message = response.message.lower()
        
        risk_level = "medium"
        if "high risk" in message or "overexposed" in message:
            risk_level = "high"
        elif "low risk" in message or "conservative" in message:
            risk_level = "low"
        
        recommendations = []
        if "close" in message:
            recommendations.append("consider_closing_positions")
        if "reduce" in message:
            recommendations.append("reduce_position_sizes")
        if "diversify" in message:
            recommendations.append("increase_diversification")
        
        return {
            "risk_level": risk_level,
            "recommendations": recommendations,
            "analysis": response.message,
            "timestamp": response.timestamp,
            "success": True
        }
    
    def _parse_market_timing(self, response: NebulaResponse, pairs: List[str]) -> Dict[str, Any]:
        """Parse market timing advice"""
        return {
            "market_sentiment": self._extract_sentiment(response.message),
            "pair_analysis": self._extract_pair_recommendations(response.message, pairs),
            "timing_advice": response.message,
            "timestamp": response.timestamp,
            "success": True
        }
    
    def _parse_stop_loss_updates(self, response: NebulaResponse, positions: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
        """Parse stop loss update recommendations"""
        # Extract stop loss levels from the response
        # This would need more sophisticated parsing based on Nebula's response format
        return {
            "updated_stops": {},
            "rationale": response.message,
            "timestamp": response.timestamp
        }
    
    def _extract_sentiment(self, text: str) -> str:
        """Extract market sentiment from text"""
        text_lower = text.lower()
        if any(word in text_lower for word in ["bullish", "positive", "optimistic", "uptrend"]):
            return "bullish"
        elif any(word in text_lower for word in ["bearish", "negative", "pessimistic", "downtrend"]):
            return "bearish"
        else:
            return "neutral"
    
    def _extract_pair_recommendations(self, text: str, pairs: List[str]) -> Dict[str, str]:
        """Extract recommendations for specific pairs"""
        recommendations = {}
        text_lower = text.lower()
        
        for pair in pairs:
            pair_lower = pair.lower()
            if pair_lower in text_lower:
                # Look for action words near the pair name
                if any(word in text_lower for word in [f"{pair_lower} buy", f"{pair_lower} long"]):
                    recommendations[pair] = "buy"
                elif any(word in text_lower for word in [f"{pair_lower} sell", f"{pair_lower} short"]):
                    recommendations[pair] = "sell"
                else:
                    recommendations[pair] = "monitor"
            else:
                recommendations[pair] = "no_signal"
        
        return recommendations
    
    async def _get_current_price(self, pair: str) -> Optional[float]:
        """Get current price for a pair (integrate with your existing price feed)"""
        try:
            if self.trading_bot:
                return await self.trading_bot.get_current_price(pair)
        except:
            pass
        return None
    
    async def _get_market_overview(self) -> Dict[str, Any]:
        """Get current market overview (integrate with your existing market analysis)"""
        try:
            if self.trading_bot:
                return {
                    "market_state": getattr(self.trading_bot, 'market_state', {}),
                    "top_movers": "BTC, ETH trending",  # Simplified
                    "volatility": "medium"
                }
        except:
            pass
        return {"status": "unknown"}
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for the Nebula integration"""
        uptime_pct = (self.successful_requests / self.total_requests * 100) if self.total_requests > 0 else 0
        
        return {
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "uptime_percentage": uptime_pct,
            "average_response_time": self.average_response_time,
            "consecutive_failures": self.consecutive_failures,
            "session_id": self.session_id,
            "last_request": self.last_request_time
        }
    
    async def health_check(self) -> bool:
        """Check if Nebula API is healthy"""
        try:
            response = await self._make_api_request("Health check - respond with OK", {"test": True})
            return response.success
        except:
            return False


# Integration with your existing trading bot
class NebulaEnhancedTradingBot:
    """
    Enhanced trading bot that integrates Nebula AI for decision making
    while maintaining fallback to internal logic
    """
    
    def __init__(self, existing_bot, nebula_secret_key: str):
        self.bot = existing_bot
        self.nebula = NebulaAIIntegration(nebula_secret_key, existing_bot)
        
        # Integration settings
        self.use_nebula_for_entries = True
        self.use_nebula_for_exits = True
        self.use_nebula_for_risk_management = True
        self.minimum_confidence_threshold = 0.6
        self.nebula_position_size_multiplier = 1.0
        
        # Performance tracking
        self.nebula_trades = []
        self.internal_trades = []
        
    async def enhanced_analyze_and_trade(self, pair: str, priority: str = 'normal'):
        """Enhanced trading analysis that incorporates Nebula AI advice"""
        
        try:
            # Get internal analysis first (for fallback and comparison)
            internal_analysis = await self.bot.strategy.analyze_pair(pair)
            
            # Get current position if any
            current_position = self.bot.active_positions.get(pair)
            
            # Get Nebula advice
            nebula_advice = await self.nebula.get_trading_decision(
                pair=pair,
                current_position=current_position,
                market_context=self.bot.market_state
            )
            
            # Combine internal analysis with Nebula advice
            final_decision = self._combine_internal_and_nebula_signals(
                internal_analysis, nebula_advice, pair
            )
            
            # Execute the decision
            if final_decision['action'] == 'buy' and not current_position:
                await self._execute_enhanced_buy(pair, final_decision)
            elif final_decision['action'] == 'sell' and current_position:
                await self._execute_enhanced_sell(pair, final_decision)
            
            # Log the decision process
            self._log_decision_process(pair, internal_analysis, nebula_advice, final_decision)
            
        except Exception as e:
            logging.error(f"Error in enhanced analysis for {pair}: {str(e)}")
            # Fallback to original analysis
            await self.bot.analyze_and_trade(pair, priority)
    
    def _combine_internal_and_nebula_signals(self, internal: Dict, nebula: TradingAdvice, pair: str) -> Dict:
        """Combine internal bot signals with Nebula advice"""
        
        # If Nebula has high confidence, prioritize its advice
        if nebula.confidence >= 0.8:
            logging.info(f"Using Nebula high-confidence advice for {pair}: {nebula.action} (conf: {nebula.confidence:.2f})")
            return {
                'action': nebula.action,
                'confidence': nebula.confidence,
                'source': 'nebula_primary',
                'entry_price': nebula.entry_price,
                'stop_loss': nebula.stop_loss,
                'take_profit': nebula.take_profit,
                'position_size': nebula.position_size,
                'rationale': nebula.rationale
            }
        
        # If internal signals are strong and Nebula confidence is low, use internal
        elif internal.get('signal_strength', 0) >= 0.7 and nebula.confidence < 0.5:
            logging.info(f"Using internal high-confidence signal for {pair}: {internal.get('buy_signal', False)}")
            return {
                'action': 'buy' if internal.get('buy_signal', False) else ('sell' if internal.get('sell_signal', False) else 'hold'),
                'confidence': internal.get('signal_strength', 0),
                'source': 'internal_primary',
                'rationale': f"Internal signal strength: {internal.get('signal_strength', 0):.2f}"
            }
        
        # If both agree, increase confidence
        elif ((internal.get('buy_signal', False) and nebula.action == 'buy') or 
              (internal.get('sell_signal', False) and nebula.action == 'sell')):
            combined_confidence = min(0.95, (internal.get('signal_strength', 0) + nebula.confidence) / 2 * 1.2)
            logging.info(f"Internal and Nebula signals agree for {pair}: {nebula.action} (combined conf: {combined_confidence:.2f})")
            return {
                'action': nebula.action,
                'confidence': combined_confidence,
                'source': 'combined_agreement',
                'entry_price': nebula.entry_price,
                'stop_loss': nebula.stop_loss,
                'take_profit': nebula.take_profit,
                'rationale': f"Both signals agree. Internal: {internal.get('signal_strength', 0):.2f}, Nebula: {nebula.confidence:.2f}"
            }
        
        # If they disagree or both are weak, be conservative
        else:
            logging.info(f"Signals disagree or weak for {pair}, holding position")
            return {
                'action': 'hold',
                'confidence': 0.3,
                'source': 'conservative_hold',
                'rationale': f"Signal disagreement or insufficient confidence. Internal: {internal.get('signal_strength', 0):.2f}, Nebula: {nebula.confidence:.2f}"
            }
    
    async def _execute_enhanced_buy(self, pair: str, decision: Dict):
        """Execute buy order with Nebula-enhanced parameters"""
        
        # Use Nebula's position size if provided, otherwise use bot's calculation
        if decision.get('position_size'):
            position_size = decision['position_size'] * self.nebula_position_size_multiplier
        else:
            position_size = self.bot.risk_manager._calculate_position_size(decision['confidence'])
        
        # Execute the buy order
        success = await self.bot.execute_buy(pair, position_size, {
            'buy_signal': True,
            'signal_strength': decision['confidence'],
            'source': decision['source']
        })
        
        if success:
            # Set enhanced stop loss and take profit if Nebula provided them
            if decision.get('stop_loss') or decision.get('take_profit'):
                await self._set_enhanced_exit_levels(pair, decision)
            
            # Track as Nebula-influenced trade
            self.nebula_trades.append({
                'pair': pair,
                'action': 'buy',
                'timestamp': time.time(),
                'confidence': decision['confidence'],
                'source': decision['source'],
                'rationale': decision.get('rationale', '')
            })
    
    async def _execute_enhanced_sell(self, pair: str, decision: Dict):
        """Execute sell order with Nebula rationale"""
        
        success = await self.bot.execute_sell(pair, {
            'sell_signal': True,
            'signal_strength': decision['confidence'],
            'source': decision['source']
        })
        
        if success:
            self.nebula_trades.append({
                'pair': pair,
                'action': 'sell',
                'timestamp': time.time(),
                'confidence': decision['confidence'],
                'source': decision['source'],
                'rationale': decision.get('rationale', '')
            })
    
    async def _set_enhanced_exit_levels(self, pair: str, decision: Dict):
        """Set stop loss and take profit based on Nebula advice"""
        if pair in self.bot.active_positions:
            position = self.bot.active_positions[pair]
            
            # Update stop loss if provided
            if decision.get('stop_loss'):
                # Implement your stop loss logic here
                logging.info(f"Setting Nebula stop loss for {pair}: {decision['stop_loss']}")
            
            # Update take profit if provided
            if decision.get('take_profit'):
                # Implement your take profit logic here
                logging.info(f"Setting Nebula take profit for {pair}: {decision['take_profit']}")
    
    def _log_decision_process(self, pair: str, internal: Dict, nebula: TradingAdvice, final: Dict):
        """Log the decision-making process for analysis"""
        
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'pair': pair,
            'internal_signal': {
                'buy': internal.get('buy_signal', False),
                'sell': internal.get('sell_signal', False),
                'strength': internal.get('signal_strength', 0),
                'source': internal.get('source', 'unknown')
            },
            'nebula_advice': {
                'action': nebula.action,
                'confidence': nebula.confidence,
                'rationale': nebula.rationale[:100] + '...' if len(nebula.rationale) > 100 else nebula.rationale
            },
            'final_decision': final
        }
        
        # Log to your trade logger
        self.bot.trade_logger.info(f"DECISION_PROCESS: {json.dumps(log_entry)}")
    
    async def get_nebula_portfolio_review(self):
        """Get comprehensive portfolio review from Nebula"""
        
        try:
            positions = {}
            for pair, pos in self.bot.active_positions.items():
                current_price = await self.bot.get_current_price(pair)
                profit_pct = self.bot.calculate_profit_percent(pair)
                
                positions[pair] = {
                    'entry_price': pos.get('entry_price', 0),
                    'current_price': current_price,
                    'quantity': pos.get('quantity', 0),
                    'value': pos.get('position_size', 0),
                    'pnl': profit_pct,
                    'age_hours': (time.time() - pos.get('entry_time', time.time())) / 3600
                }
            
            assessment = await self.nebula.get_portfolio_assessment(positions)
            
            if assessment.get('success'):
                logging.info(f"Nebula portfolio assessment: {assessment.get('risk_level', 'unknown')} risk")
                return assessment
            else:
                logging.warning(f"Nebula portfolio assessment failed: {assessment.get('error', 'unknown')}")
                return None
                
        except Exception as e:
            logging.error(f"Error getting Nebula portfolio review: {str(e)}")
            return None
    
    def get_integration_stats(self) -> Dict[str, Any]:
        """Get comprehensive stats on the Nebula integration performance"""
        
        nebula_stats = self.nebula.get_performance_stats()
        
        total_trades = len(self.nebula_trades) + len(self.internal_trades)
        nebula_trade_pct = (len(self.nebula_trades) / total_trades * 100) if total_trades > 0 else 0
        
        return {
            'nebula_api_stats': nebula_stats,
            'trading_stats': {
                'total_trades': total_trades,
                'nebula_influenced_trades': len(self.nebula_trades),
                'internal_only_trades': len(self.internal_trades),
                'nebula_trade_percentage': nebula_trade_pct
            },
            'recent_decisions': self.nebula_trades[-5:] if self.nebula_trades else []
        }