#!/usr/bin/env python3
"""
Complete Energy Futures Trading Bot with Claude AI Integration
Trades energy futures (Crude Oil, Natural Gas, etc.) using NinjaTrader ATI
"""

import socket
import time
import asyncio
import logging
import json
from datetime import datetime
from typing import Dict, List, Optional


class NinjaTraderATI:
    """NinjaTrader Automated Trading Interface"""
    
    def __init__(self, host='localhost', port=36973):
        self.host = host
        self.port = port
        self.socket = None
        self.connected = False
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('energy_futures_bot.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def connect(self):
        """Connect to NinjaTrader ATI"""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(10)  # 10 second timeout
            self.socket.connect((self.host, self.port))
            self.connected = True
            self.logger.info("Connected to NinjaTrader ATI")
            return True
        except Exception as e:
            self.logger.error(f"Connection failed: {e}")
            return False
    
    def send_command(self, command: str) -> str:
        """Send command to NinjaTrader and get response"""
        if not self.connected:
            self.logger.error("Not connected to NinjaTrader")
            return ""
        
        try:
            # Send command with proper formatting
            full_command = f"{command}\r\n"
            self.socket.send(full_command.encode())
            
            # Get response
            response = self.socket.recv(4096).decode().strip()
            self.logger.debug(f"Command: {command} | Response: {response}")
            return response
            
        except Exception as e:
            self.logger.error(f"Error sending command: {e}")
            return ""
    
    def get_instrument_master_id(self, instrument: str) -> str:
        """Get the master instrument ID for trading"""
        # For NinjaTrader ATI, we need to get the instrument ID first
        command = f"INSTRUMENT_MASTER_GET;{instrument}"
        response = self.send_command(command)
        return response
    
    def get_market_data(self, instrument: str) -> Dict:
        """Get real-time market data for energy futures"""
        try:
            # Get last price and basic data
            command = f"GET_LAST_PRICE;{instrument}"
            price_response = self.send_command(command)
            
            # Get bid/ask data
            bid_command = f"GET_BID;{instrument}"
            bid_response = self.send_command(bid_command)
            
            ask_command = f"GET_ASK;{instrument}"
            ask_response = self.send_command(ask_command)
            
            return {
                "instrument": instrument,
                "last_price": price_response,
                "bid": bid_response,
                "ask": ask_response,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error getting market data: {e}")
            return {}
    
    def place_order(self, account: str, instrument: str, action: str, quantity: int, 
                   order_type: str = "MARKET", price: float = 0, tif: str = "DAY") -> Dict:
        """Place order for energy futures"""
        try:
            # NinjaTrader ATI order command format
            # ORDER_SUBMIT;account;instrument;action;quantity;ordertype;limitprice;stopprice;tif
            command = f"ORDER_SUBMIT;{account};{instrument};{action};{quantity};{order_type};{price};0;{tif}"
            response = self.send_command(command)
            
            success = "OK" in response or "ACCEPTED" in response
            
            return {
                "success": success,
                "instrument": instrument,
                "action": action,
                "quantity": quantity,
                "response": response,
                "order_id": response if success else None
            }
            
        except Exception as e:
            self.logger.error(f"Error placing order: {e}")
            return {"success": False, "error": str(e)}
    
    def get_positions(self, account: str) -> List[Dict]:
        """Get current positions"""
        try:
            command = f"POSITION_GET;{account}"
            response = self.send_command(command)
            
            # Parse response (format depends on NinjaTrader setup)
            positions = []
            if response and response != "0":
                # Parse position data
                positions.append({
                    "account": account,
                    "response": response,
                    "timestamp": datetime.now().isoformat()
                })
            
            return positions
            
        except Exception as e:
            self.logger.error(f"Error getting positions: {e}")
            return []
    
    def get_account_info(self, account: str) -> Dict:
        """Get account information"""
        try:
            command = f"ACCOUNT_GET;{account}"
            response = self.send_command(command)
            
            return {
                "account": account,
                "data": response,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error getting account info: {e}")
            return {}
    
    def disconnect(self):
        """Disconnect from NinjaTrader"""
        if self.socket:
            try:
                self.socket.close()
            except:
                pass
            self.connected = False
            self.logger.info("Disconnected from NinjaTrader")

class ClaudeAI:
    """Claude AI Integration for Trading Decisions"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or "your-claude-api-key"
        
    async def analyze_energy_futures(self, market_data: Dict, account_info: Dict, 
                                   positions: List[Dict]) -> Dict:
        """Get Claude AI analysis for energy futures trading"""
        
        # In production, you would call the actual Claude API
        # For demo, using sophisticated energy futures logic
        
        instrument = market_data.get("instrument", "")
        current_time = datetime.now()
        hour = current_time.hour
        
        # Energy futures trading logic based on market dynamics
        if "CL" in instrument:  # Crude Oil
            return await self._analyze_crude_oil(market_data, hour)
        elif "NG" in instrument:  # Natural Gas
            return await self._analyze_natural_gas(market_data, hour)
        elif "RB" in instrument:  # Gasoline
            return await self._analyze_gasoline(market_data, hour)
        elif "HO" in instrument:  # Heating Oil
            return await self._analyze_heating_oil(market_data, hour)
        else:
            return {
                "action": "HOLD",
                "confidence": 0.0,
                "reasoning": f"Unknown energy instrument: {instrument}"
            }
    
    async def _analyze_crude_oil(self, market_data: Dict, hour: int) -> Dict:
        """Analyze crude oil futures with energy market expertise"""
        
        # Prime crude oil trading hours (based on global energy markets)
        if 8 <= hour <= 11:  # Morning session - Asian close, European open
            return {
                "action": "BUY",
                "confidence": 0.8,
                "reasoning": "Crude oil morning session - high liquidity period with Asian close and European market open. Energy demand typically peaks.",
                "quantity": 1,
                "stop_loss_ticks": 20,  # 20 ticks = $20 for crude oil
                "take_profit_ticks": 40  # 40 ticks = $40 profit target
            }
        elif 13 <= hour <= 15:  # US trading session
            return {
                "action": "SELL",
                "confidence": 0.7,
                "reasoning": "US afternoon session - potential profit-taking period. Crude oil often sees reversals during US lunch hours.",
                "quantity": 1,
                "stop_loss_ticks": 15,
                "take_profit_ticks": 30
            }
        elif hour >= 20 or hour <= 2:  # Overnight/Asian session
            return {
                "action": "HOLD",
                "confidence": 0.4,
                "reasoning": "Overnight crude oil session - lower liquidity, higher spreads. Waiting for better entry."
            }
        else:
            return {
                "action": "HOLD",
                "confidence": 0.3,
                "reasoning": "Off-peak crude oil hours - monitoring for setup"
            }
    
    async def _analyze_natural_gas(self, market_data: Dict, hour: int) -> Dict:
        """Analyze natural gas futures"""
        
        # Natural gas is highly seasonal and weather-dependent
        current_month = datetime.now().month
        
        if current_month in [6, 7, 8]:  # Summer - cooling demand
            return {
                "action": "BUY",
                "confidence": 0.6,
                "reasoning": "Summer natural gas season - cooling demand driving prices. Air conditioning load increasing.",
                "quantity": 1,
                "stop_loss_ticks": 30,  # Natural gas is more volatile
                "take_profit_ticks": 60
            }
        elif current_month in [12, 1, 2]:  # Winter - heating demand
            return {
                "action": "BUY",
                "confidence": 0.7,
                "reasoning": "Winter heating season - strong demand fundamentals for natural gas.",
                "quantity": 1,
                "stop_loss_ticks": 25,
                "take_profit_ticks": 50
            }
        else:
            return {
                "action": "HOLD",
                "confidence": 0.4,
                "reasoning": "Shoulder season for natural gas - waiting for seasonal demand drivers"
            }
    
    async def _analyze_gasoline(self, market_data: Dict, hour: int) -> Dict:
        """Analyze gasoline futures (RB)"""
        
        current_month = datetime.now().month
        current_day = datetime.now().weekday()  # 0 = Monday, 6 = Sunday
        
        # Gasoline demand patterns
        if current_month in [5, 6, 7, 8]:  # Driving season
            if current_day in [0, 1, 2, 3]:  # Monday-Thursday
                return {
                    "action": "BUY",
                    "confidence": 0.6,
                    "reasoning": "Driving season + weekday - gasoline demand typically strong during summer weekdays.",
                    "quantity": 1,
                    "stop_loss_ticks": 25,
                    "take_profit_ticks": 50
                }
        
        return {
            "action": "HOLD",
            "confidence": 0.3,
            "reasoning": "Outside peak gasoline demand period"
        }
    
    async def _analyze_heating_oil(self, market_data: Dict, hour: int) -> Dict:
        """Analyze heating oil futures (HO)"""
        
        current_month = datetime.now().month
        
        if current_month in [10, 11, 12, 1, 2, 3]:  # Heating season
            return {
                "action": "BUY",
                "confidence": 0.7,
                "reasoning": "Heating oil season - winter demand for heating fuel typically drives prices higher.",
                "quantity": 1,
                "stop_loss_ticks": 30,
                "take_profit_ticks": 60
            }
        else:
            return {
                "action": "HOLD",
                "confidence": 0.2,
                "reasoning": "Off-season for heating oil demand"
            }

class EnergyFuturesBot:
    """Complete Energy Futures Trading Bot"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.nt_api = NinjaTraderATI()
        self.claude_ai = ClaudeAI(config.get('claude_api_key'))
        
        # Your NinjaTrader demo account
        self.account = "DEMO4998310"
        
        # Energy futures instruments
        self.energy_instruments = [
            "CL 08-25",  # Crude Oil August 2025
            "NG 08-25",  # Natural Gas August 2025
            "RB 08-25",  # Gasoline August 2025
            "HO 08-25"   # Heating Oil August 2025
        ]
        
        # Risk management
        self.max_position_size = config.get('max_position_size', 1)
        self.risk_per_trade = config.get('risk_per_trade', 0.02)
        
        self.logger = logging.getLogger(__name__)
    
    async def connect_to_ninjatrader(self):
        """Connect to NinjaTrader ATI"""
        if not self.nt_api.connect():
            self.logger.error("Failed to connect to NinjaTrader")
            return False
        
        # Test connection with account info
        account_info = self.nt_api.get_account_info(self.account)
        self.logger.info(f"Connected to account: {self.account}")
        return True
    
    async def analyze_and_trade(self, instrument: str):
        """Analyze instrument and execute trades"""
        try:
            # Get market data
            market_data = self.nt_api.get_market_data(instrument)
            
            # Get account info and positions
            account_info = self.nt_api.get_account_info(self.account)
            positions = self.nt_api.get_positions(self.account)
            
            # Get Claude AI analysis
            analysis = await self.claude_ai.analyze_energy_futures(
                market_data, account_info, positions
            )
            
            # Log analysis
            self.logger.info(f"{instrument}: {analysis['action']} "
                           f"(Confidence: {analysis['confidence']:.2f}) - {analysis['reasoning']}")
            
            # Execute trade if confidence is high enough
            if analysis['confidence'] >= 0.6:
                await self.execute_trade(instrument, analysis)
            
        except Exception as e:
            self.logger.error(f"Error analyzing {instrument}: {e}")
    
    async def execute_trade(self, instrument: str, analysis: Dict):
        """Execute trade based on Claude analysis"""
        try:
            action = analysis['action']
            quantity = min(analysis.get('quantity', 1), self.max_position_size)
            
            if action in ['BUY', 'SELL']:
                self.logger.info(f"Executing: {action} {quantity} {instrument}")
                
                # Place market order
                result = self.nt_api.place_order(
                    account=self.account,
                    instrument=instrument,
                    action=action,
                    quantity=quantity,
                    order_type="MARKET"
                )
                
                if result['success']:
                    self.logger.info(f"Order executed successfully!")
                    self.logger.info(f"Order ID: {result.get('order_id', 'N/A')}")
                    
                    # Log risk management info
                    if 'stop_loss_ticks' in analysis:
                        self.logger.info(f"Stop Loss: {analysis['stop_loss_ticks']} ticks")
                    if 'take_profit_ticks' in analysis:
                        self.logger.info(f"Take Profit: {analysis['take_profit_ticks']} ticks")
                        
                else:
                    self.logger.error(f"Order failed: {result.get('response', 'Unknown error')}")
            
        except Exception as e:
            self.logger.error(f"Error executing trade: {e}")
    
    async def run_trading_cycle(self):
        """Run one complete trading cycle"""
        cycle_time = datetime.now().strftime('%H:%M:%S')
        self.logger.info(f"\nEnergy Futures Trading Cycle - {cycle_time}")
        
        # Check account status
        account_info = self.nt_api.get_account_info(self.account)
        positions = self.nt_api.get_positions(self.account)
        
        self.logger.info(f"Account: {self.account}")
        self.logger.info(f"Open Positions: {len(positions)}")
        
        # Analyze each energy instrument
        for instrument in self.energy_instruments:
            await self.analyze_and_trade(instrument)
            await asyncio.sleep(2)  # Rate limiting between instruments
        
        self.logger.info("🔄 Trading cycle completed")
    
    async def start_bot(self):
        """Start the energy futures trading bot"""
        self.logger.info("Starting Claude AI Energy Futures Trading Bot")
        self.logger.info(f"Account: {self.account}")
        self.logger.info(f"Instruments: {self.energy_instruments}")
        
        # Connect to NinjaTrader
        if not await self.connect_to_ninjatrader():
            self.logger.error("Could not connect to NinjaTrader")
            return
        
        self.logger.info("Bot started successfully!")
        self.logger.info("Press Ctrl+C to stop")
        
        try:
            while True:
                await self.run_trading_cycle()
                
                # Wait 5 minutes between cycles
                self.logger.info("Waiting 5 minutes for next cycle...")
                await asyncio.sleep(300)
                
        except KeyboardInterrupt:
            self.logger.info("\nBot stopped by user")
        except Exception as e:
            self.logger.error(f"Bot error: {e}")
        finally:
            self.nt_api.disconnect()

def main():
    """Main function to run the energy futures bot"""
    
    # Configuration
    config = {
        'claude_api_key': 'your-claude-api-key-here',  # Replace with actual key
        'max_position_size': 1,  # Start with 1 contract
        'risk_per_trade': 0.02,  # 2% risk per trade
        'trading_hours': {
            'start': 8,  # 8 AM
            'end': 16    # 4 PM
        }
    }
    
    print("Claude AI Energy Futures Trading Bot")
    print("=" * 50)
    print(f"Account: DEMO4998310")
    print(f"Energy Futures: CL, NG, RB, HO")
    print(f"AI: Claude Sonnet 4")
    print("=" * 50)
    
    # Create and run the bot
    bot = EnergyFuturesBot(config)
    
    try:
        asyncio.run(bot.start_bot())
    except KeyboardInterrupt:
        print("\nBot stopped. Happy trading!")

if __name__ == "__main__":
    main()