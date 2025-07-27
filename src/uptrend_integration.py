# uptrend_integration.py
import asyncio
import aiohttp
import logging
from typing import Dict, List, Optional
from datetime import datetime

class UptrendDetectionIntegration:
    """Integration for uptrend detection in trading bot"""
    
    def __init__(self, api_base_url: str = "http://localhost:3001", api_key: str = "1234"):
        self.api_base_url = api_base_url
        self.api_key = api_key
        self.headers = {"Authorization": f"Bearer {api_key}"}
        self.active_uptrends = {}
        self.last_scan_time = 0
        self.scan_interval = 30  # seconds
        
    async def get_uptrend_alerts(self, 
                                stage: str = 'early_uptrend',
                                min_strength: int = 60,
                                limit: int = 5) -> List[Dict]:
        """Get real-time uptrend alerts for immediate action"""
        try:
            url = f"{self.api_base_url}/api/v1/uptrend-alerts"
            params = {
                'stage': stage,
                'min_strength': min_strength,
                'limit': limit
            }
            
            logging.info(f"🔍 Calling uptrend alerts: {url} with params: {params}")
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, headers=self.headers) as response:
                    logging.info(f"📡 Uptrend alerts response status: {response.status}")
                    
                    if response.status == 200:
                        data = await response.json()
                        logging.info(f"📊 Uptrend alerts response: {data}")
                        
                        if data.get('success'):
                            alerts = data.get('alerts', [])
                            logging.info(f"✅ Found {len(alerts)} uptrend alerts")
                            return alerts
                        else:
                            logging.warning(f"❌ Uptrend alerts API returned success=false: {data.get('error', 'Unknown error')}")
                    else:
                        logging.error(f"❌ Uptrend alerts API returned status {response.status}")
                        
            return []
            
        except Exception as e:
            logging.error(f"❌ Error getting uptrend alerts: {e}")
            return []
    
    async def scan_for_uptrends(self,
                               symbols: Optional[List[str]] = None,
                               min_strength: int = 40,
                               stages: Optional[List[str]] = None) -> List[Dict]:
        """Comprehensive uptrend scanning"""
        try:
            url = f"{self.api_base_url}/api/v1/uptrend-scanner"
            payload = {
                'min_strength': min_strength,
                'max_results': 20
            }
            
            if symbols:
                payload['symbols'] = symbols
            if stages:
                payload['stages'] = stages
                
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, headers=self.headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data['success']:
                            return data['data']['uptrends']
            return []
            
        except Exception as e:
            logging.error(f"Error scanning uptrends: {e}")
            return []
    
    async def analyze_symbol_uptrend(self, symbol: str) -> Optional[Dict]:
        """Get detailed uptrend analysis for a specific symbol"""
        try:
            url = f"{self.api_base_url}/api/v1/uptrend-analysis/{symbol}"
            
            logging.info(f"🔍 Analyzing uptrend for {symbol}: {url}")
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=self.headers) as response:
                    logging.info(f"📡 Uptrend analysis response status for {symbol}: {response.status}")
                    
                    if response.status == 200:
                        data = await response.json()
                        logging.info(f"📊 Uptrend analysis response for {symbol}: {data}")
                        
                        if data.get('success'):
                            analysis = data.get('analysis')
                            logging.info(f"✅ Got uptrend analysis for {symbol}: {analysis}")
                            return analysis
                        else:
                            logging.warning(f"❌ Uptrend analysis API returned success=false for {symbol}: {data.get('error', 'Unknown error')}")
                    else:
                        logging.error(f"❌ Uptrend analysis API returned status {response.status} for {symbol}")
                        
            return None
            
        except Exception as e:
            logging.error(f"❌ Error analyzing {symbol}: {e}")
            return None
    
    async def batch_screen_uptrends(self,
                                   volume_threshold: float = 1000000,
                                   momentum_timeframes: List[str] = ['micro', 'short'],
                                   technical_filters: Dict = None) -> List[Dict]:
        """Advanced batch screening with custom filters"""
        try:
            url = f"{self.api_base_url}/api/v1/uptrend-batch-screen"
            payload = {
                'volume_threshold': volume_threshold,
                'momentum_timeframes': momentum_timeframes
            }
            
            if technical_filters:
                payload['technical_filters'] = technical_filters
                
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, headers=self.headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data['success']:
                            return data['results']['opportunities']
            return []
            
        except Exception as e:
            logging.error(f"Error in batch screening: {e}")
            return []

class UptrendTradingStrategy:
    """Enhanced trading strategy using uptrend detection"""
    
    def __init__(self, uptrend_detector: UptrendDetectionIntegration):
        self.detector = uptrend_detector
        self.active_positions = {}
        self.pending_entries = {}
        
    async def execute_uptrend_strategy(self):
        """Main strategy execution loop"""
        while True:
            try:
                # 1. Get immediate buy alerts
                alerts = await self.detector.get_uptrend_alerts(
                    stage='early_uptrend',
                    min_strength=65,
                    limit=3
                )
                
                for alert in alerts:
                    if alert['urgency'] == 'high' and alert['symbol'] not in self.active_positions:
                        await self.process_buy_alert(alert)
                
                # 2. Scan for developing uptrends
                uptrends = await self.detector.scan_for_uptrends(
                    min_strength=50,
                    stages=['accumulation', 'early_uptrend']
                )
                
                for uptrend in uptrends:
                    await self.evaluate_uptrend_opportunity(uptrend)
                
                # 3. Monitor existing positions
                await self.monitor_positions()
                
                await asyncio.sleep(30)  # 30 second cycle
                
            except Exception as e:
                logging.error(f"Strategy execution error: {e}")
                await asyncio.sleep(60)
    
    async def process_buy_alert(self, alert: Dict):
        """Process immediate buy alert"""
        logging.info(f"🚀 UPTREND ALERT: {alert['symbol']} - {alert['reason']}")
        logging.info(f"   Strength: {alert['strength']}%, Confidence: {alert['confidence']}%")
        logging.info(f"   Action: {alert['action']} ({alert['urgency']} urgency)")
        logging.info(f"   Position Size: {alert['recommended_position']}")
        logging.info(f"   Stop Loss: {alert['stop_loss']}%, Targets: {alert['targets']}")
        
        # Execute trade based on alert
        trade_params = {
            'symbol': alert['symbol'],
            'side': 'BUY',
            'position_size': self.calculate_position_size(alert['recommended_position']),
            'stop_loss': alert['stop_loss'],
            'take_profits': alert['targets'],
            'reason': f"Uptrend Alert: {alert['reason']}"
        }
        
        # Add to pending entries for execution
        self.pending_entries[alert['symbol']] = trade_params
    
    async def evaluate_uptrend_opportunity(self, uptrend: Dict):
        """Evaluate uptrend for potential entry"""
        symbol = uptrend['symbol']
        
        # Skip if already in position
        if symbol in self.active_positions:
            return
            
        # Different strategies based on stage
        if uptrend['stage'] == 'accumulation':
            # Scale in during accumulation
            if uptrend['strength'] > 45 and uptrend['volume']['trend'] == 'increasing':
                logging.info(f"📊 Accumulation detected in {symbol}")
                await self.plan_scale_in_entry(uptrend)
                
        elif uptrend['stage'] == 'early_uptrend':
            # Enter on confirmation
            if uptrend['momentum']['trending'] and uptrend['strength'] > 60:
                logging.info(f"📈 Early uptrend confirmed in {symbol}")
                await self.plan_momentum_entry(uptrend)
    
    async def plan_scale_in_entry(self, uptrend: Dict):
        """Plan gradual entry during accumulation"""
        symbol = uptrend['symbol']
        
        # Get detailed analysis
        analysis = await self.detector.analyze_symbol_uptrend(symbol)
        if not analysis:
            return
            
        if analysis['entry_recommendation']['action'] in ['SCALE_IN', 'MONITOR']:
            entry_plan = {
                'symbol': symbol,
                'strategy': 'scale_in',
                'initial_size': 0.25,  # 25% of full position
                'add_levels': 3,       # 3 additional entries
                'conditions': {
                    'min_rsi': 35,
                    'max_rsi': 50,
                    'volume_increase': True
                }
            }
            logging.info(f"Planning scale-in entry for {symbol}")
            self.pending_entries[symbol] = entry_plan
    
    async def plan_momentum_entry(self, uptrend: Dict):
        """Plan momentum-based entry"""
        symbol = uptrend['symbol']
        
        entry_plan = {
            'symbol': symbol,
            'strategy': 'momentum',
            'position_size': self.calculate_position_size(uptrend['recommendation']['positionSize']),
            'stop_loss': uptrend['recommendation']['stopLoss'],
            'targets': uptrend['recommendation']['targets'],
            'entry_conditions': {
                'volume_spike': uptrend['volume']['spike'],
                'patterns': uptrend['patterns']
            }
        }
        logging.info(f"Planning momentum entry for {symbol}")
        self.pending_entries[symbol] = entry_plan
    
    async def monitor_positions(self):
        """Monitor existing positions for uptrend continuation"""
        for symbol, position in self.active_positions.items():
            # Get current uptrend status
            analysis = await self.detector.analyze_symbol_uptrend(symbol)
            if not analysis:
                continue
                
            # Check if uptrend is weakening
            if not analysis['is_uptrend'] or analysis['strength'] < 30:
                logging.warning(f"⚠️ Uptrend weakening for {symbol}")
                await self.plan_exit(symbol, "Uptrend reversal detected")
                
            # Check if entering parabolic phase (take profits)
            elif analysis['stage'] == 'parabolic':
                logging.warning(f"🎯 Parabolic phase for {symbol} - taking profits")
                await self.plan_partial_exit(symbol, 0.5, "Parabolic move detected")
    
    def calculate_position_size(self, recommendation: str) -> float:
        """Calculate position size based on recommendation"""
        size_map = {
            'minimal': 0.05,   # 5% of capital
            'small': 0.10,     # 10% of capital
            'medium': 0.15,    # 15% of capital
            'large': 0.20      # 20% of capital
        }
        return size_map.get(recommendation, 0.10)
    
    async def plan_exit(self, symbol: str, reason: str):
        """Plan position exit"""
        logging.info(f"Planning exit for {symbol}: {reason}")
        # Implementation for exit planning
        
    async def plan_partial_exit(self, symbol: str, percentage: float, reason: str):
        """Plan partial position exit"""
        logging.info(f"Planning {percentage*100}% exit for {symbol}: {reason}")
        # Implementation for partial exit

# Integration with existing bot
class HybridBotWithUptrend:
    """Enhanced hybrid bot with uptrend detection"""
    
    def __init__(self, existing_bot):
        self.bot = existing_bot
        self.uptrend_detector = UptrendDetectionIntegration()
        self.uptrend_strategy = UptrendTradingStrategy(self.uptrend_detector)
        
    async def enhanced_asset_selection(self) -> List[str]:
        """Enhanced asset selection using uptrend detection"""
        # Get regular candidates from existing system
        regular_candidates = await self.bot.get_tier_candidate_assets()
        
        # Get uptrend candidates
        uptrend_alerts = await self.uptrend_detector.get_uptrend_alerts(
            stage='early_uptrend',
            min_strength=60,
            limit=10
        )
        
        # Combine and prioritize
        priority_symbols = [alert['symbol'] for alert in uptrend_alerts]
        
        # Add regular candidates that aren't already in priority
        for tier in ['tier1_candidates', 'tier2_candidates']:
            if tier in regular_candidates:
                for candidate in regular_candidates[tier]:
                    if candidate['symbol'] not in priority_symbols:
                        priority_symbols.append(candidate['symbol'])
        
        return priority_symbols[:15]  # Return top 15
    
    async def should_enter_position(self, symbol: str, regular_signal: Dict) -> bool:
        """Enhanced entry decision using uptrend analysis"""
        # Get uptrend analysis
        uptrend = await self.uptrend_detector.analyze_symbol_uptrend(symbol)
        
        if not uptrend:
            return regular_signal.get('action') == 'BUY'
        
        # Combine signals
        regular_confidence = regular_signal.get('confidence', 0)
        uptrend_strength = uptrend.get('strength', 0)
        
        # Enhanced decision logic
        if uptrend['is_uptrend'] and uptrend['entry_recommendation']['action'] == 'BUY_NOW':
            return True
        
        if regular_confidence > 60 and uptrend_strength > 50:
            return True
            
        if uptrend['stage'] == 'early_uptrend' and regular_confidence > 55:
            return True
            
        return False

# Example usage
async def main():
    # Initialize uptrend detection
    detector = UptrendDetectionIntegration()
    
    # Example 1: Get immediate alerts
    print("🚨 Checking for immediate uptrend alerts...")
    alerts = await detector.get_uptrend_alerts()
    for alert in alerts:
        print(f"  {alert['symbol']}: {alert['action']} - {alert['reason']}")
    
    # Example 2: Scan specific symbols
    print("\n📊 Scanning specific symbols...")
    symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'ADAUSDT']
    uptrends = await detector.scan_for_uptrends(symbols=symbols, min_strength=30)
    for uptrend in uptrends:
        print(f"  {uptrend['symbol']}: Stage={uptrend['stage']}, Strength={uptrend['strength']}%")
    
    # Example 3: Detailed analysis
    print("\n🔍 Detailed analysis for BTCUSDT...")
    analysis = await detector.analyze_symbol_uptrend('BTCUSDT')
    if analysis:
        print(f"  Uptrend: {analysis['is_uptrend']}")
        print(f"  Stage: {analysis['stage']}")
        print(f"  Entry: {analysis['entry_recommendation']['action']}")
        print(f"  Confidence: {analysis['confidence']}%")

if __name__ == "__main__":
    asyncio.run(main()) 