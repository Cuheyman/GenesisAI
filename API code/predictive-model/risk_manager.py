# Add to predictive-model/risk_manager.py
class AdvancedRiskManager:
    def __init__(self):
        self.trade_history = []
        self.kelly_fraction = 0.25  # Use 25% of Kelly for safety
        
    def calculate_kelly_position_size(self, win_probability, avg_win, avg_loss, bankroll):
        """Calculate optimal position size using Kelly Criterion"""
        if avg_loss == 0:
            return 0
            
        loss_probability = 1 - win_probability
        kelly_percentage = (win_probability * avg_win - loss_probability * abs(avg_loss)) / avg_win
        
        # Apply safety factor
        safe_kelly = kelly_percentage * self.kelly_fraction
        
        # Never risk more than 2% of bankroll
        max_risk = 0.02
        position_size = min(safe_kelly, max_risk) * bankroll
        
        return max(0, position_size)
        
    def calculate_var(self, returns, confidence_level=0.95):
        """Calculate Value at Risk"""
        return np.percentile(returns, (1 - confidence_level) * 100)
        
    def calculate_cvar(self, returns, confidence_level=0.95):
        """Calculate Conditional Value at Risk (Expected Shortfall)"""
        var = self.calculate_var(returns, confidence_level)
        return returns[returns <= var].mean()
        
    def get_risk_adjusted_position(self, base_position, market_conditions):
        """Adjust position size based on market conditions"""
        volatility_multiplier = 1.0
        
        if market_conditions['volatility'] == 'high':
            volatility_multiplier = 0.5
        elif market_conditions['volatility'] == 'extreme':
            volatility_multiplier = 0.25
            
        correlation_multiplier = 1.0
        if market_conditions['correlation'] > 0.8:
            correlation_multiplier = 0.7  # Reduce when everything moves together
            
        regime_multiplier = 1.0
        if market_conditions['regime'] == 'bear':
            regime_multiplier = 0.6
        elif market_conditions['regime'] == 'uncertain':
            regime_multiplier = 0.8
            
        adjusted_position = base_position * volatility_multiplier * correlation_multiplier * regime_multiplier
        
        return adjusted_position

__all__ = ['AdvancedRiskManager']