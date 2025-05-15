import numpy as np
import pandas as pd
import logging
import time
import asyncio
from typing import Dict, List, Any, Tuple

class CorrelationAnalysis:
    def __init__(self, market_analysis):
        self.market_analysis = market_analysis
        self.correlation_matrix = {}
        self.last_update_time = 0
        self.update_interval = 3600  # Update correlation matrix hourly
        
    async def get_correlation_matrix(self, pairs: List[str]) -> Dict[str, Dict[str, float]]:
        """Get correlation matrix for list of trading pairs"""
        current_time = time.time()
        
        # Update correlation matrix if it's stale or doesn't exist
        if not self.correlation_matrix or current_time - self.last_update_time > self.update_interval:
            await self.update_correlation_matrix(pairs)
            
        return self.correlation_matrix
    
    async def update_correlation_matrix(self, pairs: List[str]):
        """Update the correlation matrix for all trading pairs"""
        try:
            logging.info("Updating correlation matrix...")
            
            # Get price data for all pairs
            price_data = {}
            
            # Gather price data in parallel
            async def get_pair_prices(pair):
                # Get recent klines
                klines = await self.market_analysis.get_klines(
                    pair, 
                    int(time.time() * 1000) - (86400 * 1000 * 5),  # 5 days
                    '1h'
                )
                
                if klines and len(klines) > 24:  # Need at least 24 hours of data
                    # Extract closing prices
                    closes = [float(k[4]) for k in klines]
                    return pair, closes
                return pair, []
            
            # Get price data for all pairs in parallel
            tasks = [get_pair_prices(pair) for pair in pairs]
            results = await asyncio.gather(*tasks)
            
            # Filter results to pairs with sufficient data
            for pair, closes in results:
                if closes:
                    price_data[pair] = closes
            
            # Create correlation matrix
            matrix = {}
            for pair1 in price_data:
                matrix[pair1] = {}
                for pair2 in price_data:
                    if pair1 == pair2:
                        matrix[pair1][pair2] = 1.0  # Self-correlation is always 1.0
                    else:
                        # Calculate correlation
                        prices1 = price_data[pair1]
                        prices2 = price_data[pair2]
                        
                        # Ensure both price series are the same length
                        min_length = min(len(prices1), len(prices2))
                        if min_length >= 24:  # Need at least 24 data points
                            prices1 = prices1[-min_length:]
                            prices2 = prices2[-min_length:]
                            
                            try:
                                correlation = np.corrcoef(prices1, prices2)[0, 1]
                                matrix[pair1][pair2] = correlation
                            except:
                                matrix[pair1][pair2] = 0.0
                        else:
                            matrix[pair1][pair2] = 0.0
            
            self.correlation_matrix = matrix
            self.last_update_time = time.time()
            
            logging.info(f"Correlation matrix updated for {len(matrix)} pairs")
            
        except Exception as e:
            logging.error(f"Error updating correlation matrix: {str(e)}")
    
    def get_pair_correlation(self, pair1: str, pair2: str) -> float:
        """Get correlation between two specific pairs"""
        if pair1 in self.correlation_matrix and pair2 in self.correlation_matrix[pair1]:
            return self.correlation_matrix[pair1][pair2]
        return 0.0  # Default to no correlation if data not available
    
    def get_portfolio_correlation(self, pair: str, active_positions: List[str]) -> float:
        """Get average correlation of a pair with current portfolio"""
        if not active_positions:
            return 0.0
            
        correlations = []
        for position_pair in active_positions:
            if position_pair != pair:  # Skip self-correlation
                correlation = self.get_pair_correlation(pair, position_pair)
                correlations.append(abs(correlation))  # Use absolute correlation
                
        # Return average correlation
        if correlations:
            return sum(correlations) / len(correlations)
        return 0.0
    
    def get_correlation_adjusted_weight(self, pair: str, active_positions: List[str], base_weight: float) -> float:
        """Adjust position weight based on correlation with existing portfolio"""
        # Get average correlation
        avg_correlation = self.get_portfolio_correlation(pair, active_positions)
        
        # High correlation should reduce weight
        if avg_correlation > 0.7:  # Very high correlation
            return base_weight * 0.6  # Reduce by 40%
        elif avg_correlation > 0.5:  # Moderately high correlation
            return base_weight * 0.8  # Reduce by 20%
        elif avg_correlation < 0.2:  # Very low correlation
            return base_weight * 1.2  # Increase by 20%
        
        # Default to unchanged weight
        return base_weight
    
    def are_pairs_diversified(self, pair: str, active_positions: List[str], threshold: float = 0.7) -> bool:
        """Check if adding a pair would maintain portfolio diversification"""
        # If no positions yet, it's always diversified
        if not active_positions:
            return True
            
        # Get average correlation with existing positions
        avg_correlation = self.get_portfolio_correlation(pair, active_positions)
        
        # Return true if correlation is below threshold
        return avg_correlation < threshold