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
                try:
                    # Get recent klines
                    klines = await self.market_analysis.get_klines(
                        pair, 
                        int(time.time() * 1000) - (86400 * 1000 * 5),  # 5 days
                        '1h'
                    )
                    
                    # FIXED: Proper null checking and validation
                    if klines is not None and isinstance(klines, list) and len(klines) > 24:
                        # Extract closing prices with additional validation
                        closes = []
                        for k in klines:
                            try:
                                # Ensure k is not None and has enough elements
                                if k is not None and len(k) > 4:
                                    close_price = float(k[4])
                                    if close_price > 0:  # Ensure valid price
                                        closes.append(close_price)
                            except (ValueError, TypeError, IndexError) as e:
                                logging.debug(f"Invalid kline data for {pair}: {e}")
                                continue
                        
                        if len(closes) >= 24:  # Still need at least 24 valid data points
                            return pair, closes
                        else:
                            logging.debug(f"Insufficient valid price data for {pair}: {len(closes)} points")
                    else:
                        logging.debug(f"No klines data for {pair} or insufficient length")
                        
                except Exception as e:
                    logging.error(f"Error getting price data for {pair}: {str(e)}")
                
                return pair, []
            
            # Get price data for all pairs in parallel
            tasks = [get_pair_prices(pair) for pair in pairs]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Filter results to pairs with sufficient data
            valid_pairs_count = 0
            for result in results:
                try:
                    if isinstance(result, Exception):
                        logging.error(f"Exception in correlation data gathering: {result}")
                        continue
                    
                    pair, closes = result
                    if closes and len(closes) >= 24:
                        price_data[pair] = closes
                        valid_pairs_count += 1
                except Exception as e:
                    logging.error(f"Error processing correlation result: {e}")
                    continue
            
            logging.info(f"Successfully gathered price data for {valid_pairs_count}/{len(pairs)} pairs")
            
            # Create correlation matrix only if we have valid data
            if len(price_data) < 2:
                logging.warning("Insufficient pairs for correlation matrix calculation")
                # Create empty matrix to avoid downstream errors
                self.correlation_matrix = {}
                self.last_update_time = time.time()
                return
            
            matrix = {}
            for pair1 in price_data:
                matrix[pair1] = {}
                for pair2 in price_data:
                    if pair1 == pair2:
                        matrix[pair1][pair2] = 1.0  # Self-correlation is always 1.0
                    else:
                        try:
                            # Calculate correlation
                            prices1 = price_data[pair1]
                            prices2 = price_data[pair2]
                            
                            # Ensure both price series are the same length
                            min_length = min(len(prices1), len(prices2))
                            if min_length >= 24:  # Need at least 24 data points
                                prices1 = prices1[-min_length:]
                                prices2 = prices2[-min_length:]
                                
                                # Additional validation before correlation calculation
                                if len(prices1) == len(prices2) and len(prices1) > 1:
                                    correlation = np.corrcoef(prices1, prices2)[0, 1]
                                    
                                    # Handle NaN correlations (can happen with constant prices)
                                    if np.isnan(correlation) or np.isinf(correlation):
                                        correlation = 0.0
                                    
                                    matrix[pair1][pair2] = float(correlation)
                                else:
                                    matrix[pair1][pair2] = 0.0
                            else:
                                matrix[pair1][pair2] = 0.0
                        except Exception as e:
                            logging.debug(f"Error calculating correlation between {pair1} and {pair2}: {e}")
                            matrix[pair1][pair2] = 0.0
            
            self.correlation_matrix = matrix
            self.last_update_time = time.time()
            
            logging.info(f"Correlation matrix updated for {len(matrix)} pairs")
            
        except Exception as e:
            logging.error(f"Error updating correlation matrix: {str(e)}")
            # Ensure we have a valid (even if empty) correlation matrix
            if not hasattr(self, 'correlation_matrix') or self.correlation_matrix is None:
                self.correlation_matrix = {}
    
    def get_pair_correlation(self, pair1: str, pair2: str) -> float:
        """Get correlation between two specific pairs"""
        try:
            if (self.correlation_matrix and 
                pair1 in self.correlation_matrix and 
                pair2 in self.correlation_matrix[pair1]):
                return self.correlation_matrix[pair1][pair2]
        except Exception as e:
            logging.debug(f"Error getting correlation between {pair1} and {pair2}: {e}")
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