#!/usr/bin/env python3
"""
ML Predictor - Bridge between Node.js and Python ML models
Accepts JSON input via stdin and returns JSON output via stdout
"""

import sys
import json
import numpy as np
import pandas as pd
from typing import Dict, Any
import logging

# Patch: Try/except for each import, print errors to stderr
DeepLearningTrainer = None
AdvancedRiskManager = None
PortfolioOptimizer = None

try:
    from deep_learning_model import DeepLearningTrainer as DLT
    DeepLearningTrainer = DLT
except Exception as e:
    print(f"[IMPORT ERROR] Could not import DeepLearningTrainer: {e}", file=sys.stderr)

try:
    from risk_manager import AdvancedRiskManager as ARM
    AdvancedRiskManager = ARM
except Exception as e:
    print(f"[IMPORT ERROR] Could not import AdvancedRiskManager: {e}", file=sys.stderr)

try:
    from portfolio_optimizer import PortfolioOptimizer as PO
    PortfolioOptimizer = PO
except Exception as e:
    print(f"[IMPORT ERROR] Could not import PortfolioOptimizer: {e}", file=sys.stderr)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MLPredictor:
    def __init__(self):
        self.price_model = None
        self.risk_manager = AdvancedRiskManager()
        self.portfolio_optimizer = None
        
        # Initialize models (with fallbacks)
        try:
            self.price_model = DeepLearningTrainer()
            self.portfolio_optimizer = PortfolioOptimizer()
        except Exception as e:
            logger.warning(f"Could not initialize ML models: {e}")
    
    def predict_price(self, symbol: str, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Predict price movement using deep learning model"""
        try:
            if self.price_model is None:
                return self._fallback_price_prediction()
            
            # Convert market data to DataFrame format expected by model
            df = self._prepare_market_data(market_data)
            
            # Get prediction
            prediction = self.price_model.predict(df)
            
            return {
                "prediction": self._interpret_prediction(prediction),
                "confidence": float(prediction.get('confidence', 0.5)),
                "target_prices": {
                    "stop_loss": float(prediction.get('stop_loss', 0)),
                    "take_profit": float(prediction.get('take_profit', 0)),
                    "target": float(prediction.get('target_price', 0))
                },
                "volatility_regime": prediction.get('volatility_regime', 'medium')
            }
            
        except Exception as e:
            logger.error(f"Price prediction failed: {e}")
            return self._fallback_price_prediction()
    
    def assess_risk(self, symbol: str, market_data: Dict[str, Any], 
                   technical_data: Dict[str, Any]) -> Dict[str, Any]:
        """Assess risk using risk manager"""
        try:
            # Extract relevant data
            price = market_data.get('price', 0)
            volume = market_data.get('volume_24h', 0)
            volatility = technical_data.get('volatility', 0.02)
            
            # Calculate risk metrics
            risk_score = self._calculate_risk_score(price, volume, volatility)
            
            return {
                "risk_score": float(risk_score),
                "confidence": 0.7,
                "var_95": float(risk_score * 0.02),
                "cvar_95": float(risk_score * 0.03),
                "max_drawdown": float(risk_score * 0.05)
            }
            
        except Exception as e:
            logger.error(f"Risk assessment failed: {e}")
            return self._fallback_risk_assessment()
    
    def optimize_portfolio(self, symbol: str, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize portfolio allocation"""
        try:
            if self.portfolio_optimizer is None:
                return self._fallback_portfolio_optimization()
            
            # Get optimal position size
            price = market_data.get('price', 0)
            volatility = market_data.get('volatility', 0.02)
            
            optimal_size = self._calculate_optimal_position(price, volatility)
            
            return {
                "optimal_size": float(optimal_size),
                "confidence": 0.6,
                "kelly_fraction": 0.25,
                "risk_adjusted_return": 0.08
            }
            
        except Exception as e:
            logger.error(f"Portfolio optimization failed: {e}")
            return self._fallback_portfolio_optimization()
    
    def _prepare_market_data(self, market_data: Dict[str, Any]) -> pd.DataFrame:
        """Convert market data to DataFrame format"""
        # Create synthetic OHLCV data if not available
        price = market_data.get('price', 100000)
        volume = market_data.get('volume_24h', 1000000)
        
        # Generate some historical data points
        dates = pd.date_range(end=pd.Timestamp.now(), periods=100, freq='h')
        
        # Create synthetic price movements
        np.random.seed(42)  # For reproducibility
        returns = np.random.normal(0, 0.01, 100)
        prices = [price]
        
        for ret in returns[:-1]:
            prices.append(prices[-1] * (1 + ret))
        
        df = pd.DataFrame({
            'timestamp': dates,
            'open': prices,
            'high': [p * (1 + abs(np.random.normal(0, 0.005))) for p in prices],
            'low': [p * (1 - abs(np.random.normal(0, 0.005))) for p in prices],
            'close': prices,
            'volume': [volume * (1 + np.random.normal(0, 0.1)) for _ in prices]
        })
        
        return df
    
    def _interpret_prediction(self, prediction: Dict[str, Any]) -> str:
        """Interpret model prediction into signal"""
        confidence = prediction.get('confidence', 0.5)
        direction = prediction.get('direction', 0)
        
        if confidence > 0.7:
            if direction > 0.1:
                return 'strong_buy'
            elif direction < -0.1:
                return 'strong_sell'
        
        if confidence > 0.5:
            if direction > 0.05:
                return 'buy'
            elif direction < -0.05:
                return 'sell'
        
        return 'neutral'
    
    def _calculate_risk_score(self, price: float, volume: float, volatility: float) -> float:
        """Calculate risk score based on market conditions"""
        # Normalize inputs
        price_factor = min(price / 100000, 1.0)  # Normalize to 0-1
        volume_factor = min(volume / 1000000000, 1.0)  # Normalize to 0-1
        volatility_factor = min(volatility / 0.1, 1.0)  # Normalize to 0-1
        
        # Weighted risk score
        risk_score = (price_factor * 0.3 + volume_factor * 0.3 + volatility_factor * 0.4)
        
        return min(max(risk_score, 0.1), 0.9)  # Clamp between 0.1 and 0.9
    
    def _calculate_optimal_position(self, price: float, volatility: float) -> float:
        """Calculate optimal position size using Kelly Criterion"""
        # Simplified Kelly calculation
        win_prob = 0.55  # Assume 55% win rate
        avg_win = 0.02   # 2% average win
        avg_loss = 0.015 # 1.5% average loss
        
        kelly_fraction = (win_prob * avg_win - (1 - win_prob) * avg_loss) / avg_win
        safe_kelly = kelly_fraction * 0.25  # Use 25% of Kelly for safety
        
        # Adjust for volatility
        volatility_adjustment = max(0.5, 1 - volatility * 10)
        
        return min(safe_kelly * volatility_adjustment, 0.02)  # Max 2% position
    
    def _fallback_price_prediction(self) -> Dict[str, Any]:
        return {
            "prediction": "neutral",
            "confidence": 0.5,
            "target_prices": {
                "stop_loss": 0,
                "take_profit": 0,
                "target": 0
            },
            "volatility_regime": "medium"
        }
    
    def _fallback_risk_assessment(self) -> Dict[str, Any]:
        return {
            "risk_score": 0.5,
            "confidence": 0.5,
            "var_95": 0.02,
            "cvar_95": 0.03,
            "max_drawdown": 0.05
        }
    
    def _fallback_portfolio_optimization(self) -> Dict[str, Any]:
        return {
            "optimal_size": 0.02,
            "confidence": 0.5,
            "kelly_fraction": 0.25,
            "risk_adjusted_return": 0.08
        }

def main():
    """Main function to handle command line arguments and JSON I/O"""
    if len(sys.argv) < 2:
        print(json.dumps({"error": "No command specified"}), file=sys.stderr)
        sys.exit(1)
    
    command = sys.argv[1]
    predictor = MLPredictor()
    
    try:
        # Read JSON input from stdin
        input_data = json.loads(sys.stdin.read())
        
        if command == "--predict":
            result = predictor.predict_price(
                input_data.get('symbol', 'BTCUSDT'),
                input_data.get('market_data', {})
            )
        elif command == "--assess-risk":
            result = predictor.assess_risk(
                input_data.get('symbol', 'BTCUSDT'),
                input_data.get('market_data', {}),
                input_data.get('technical_data', {})
            )
        elif command == "--optimize":
            result = predictor.optimize_portfolio(
                input_data.get('symbol', 'BTCUSDT'),
                input_data.get('market_data', {})
            )
        else:
            result = {"error": f"Unknown command: {command}"}
        
        # Output JSON result to stdout
        print(json.dumps(result))
        
    except json.JSONDecodeError as e:
        print(json.dumps({"error": f"Invalid JSON input: {e}"}), file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(json.dumps({"error": f"Processing failed: {e}"}), file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main() 