# Add to predictive-model/portfolio_optimizer.py
from scipy.optimize import minimize
import cvxpy as cp

class PortfolioOptimizer:
    def __init__(self):
        self.returns_history = {}
        self.risk_free_rate = 0.02  # 2% annual
        
    def calculate_sharpe_ratio(self, returns, risk_free_rate=None):
        if risk_free_rate is None:
            risk_free_rate = self.risk_free_rate / 252  # Daily rate
            
        excess_returns = returns - risk_free_rate
        return np.sqrt(252) * excess_returns.mean() / returns.std()
        
    def optimize_portfolio_weights(self, expected_returns, cov_matrix, constraints=None):
        """Optimize portfolio weights using Modern Portfolio Theory"""
        n_assets = len(expected_returns)
        
        # Variables
        weights = cp.Variable(n_assets)
        
        # Objective: Maximize Sharpe Ratio (approximated)
        portfolio_return = expected_returns @ weights
        portfolio_risk = cp.quad_form(weights, cov_matrix)
        
        objective = cp.Maximize(portfolio_return - 0.5 * portfolio_risk)
        
        # Constraints
        constraints_list = [
            weights >= 0,  # No short selling
            cp.sum(weights) == 1,  # Fully invested
        ]
        
        # Add custom constraints
        if constraints:
            if 'max_weight' in constraints:
                constraints_list.append(weights <= constraints['max_weight'])
            if 'min_weight' in constraints:
                constraints_list.append(weights >= constraints['min_weight'])
                
        # Solve
        problem = cp.Problem(objective, constraints_list)
        problem.solve()
        
        return weights.value
        
    def calculate_efficient_frontier(self, returns_df, n_portfolios=100):
        """Calculate the efficient frontier"""
        mean_returns = returns_df.mean()
        cov_matrix = returns_df.cov()
        
        # Generate target returns
        min_ret = mean_returns.min()
        max_ret = mean_returns.max()
        target_returns = np.linspace(min_ret, max_ret, n_portfolios)
        
        efficient_frontier = []
        
        for target_return in target_returns:
            weights = self._minimize_risk_for_target_return(
                mean_returns, cov_matrix, target_return
            )
            
            if weights is not None:
                portfolio_return = np.sum(mean_returns * weights)
                portfolio_risk = np.sqrt(weights.T @ cov_matrix @ weights)
                
                efficient_frontier.append({
                    'return': portfolio_return,
                    'risk': portfolio_risk,
                    'weights': weights
                })
                
        return efficient_frontier

__all__ = ['PortfolioOptimizer']