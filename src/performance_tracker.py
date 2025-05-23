import sqlite3
import logging
import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import time
import config

class PerformanceTracker:
    def __init__(self, db_path=None):
        self.db_path = db_path or config.DB_PATH
        self.day_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        self.daily_profits = []
        self.trades_today = 0
        self.winning_trades_today = 0
        self.last_equity = 0
        self.equity_high_water_mark = 0
        
    async def record_equity(self, equity, daily_roi=0, drawdown=0):
        """Record current equity and performance metrics"""
        try:
            # Update equity high water mark
            if equity > self.equity_high_water_mark:
                self.equity_high_water_mark = equity
                
            # Store last equity value
            self.last_equity = equity
            
            # Record in database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Insert into equity_history
            cursor.execute('''
                INSERT INTO equity_history (timestamp, equity)
                VALUES (?, ?)
            ''', (int(time.time()), equity))
            
            # Update bot_stats with last equity
            cursor.execute('''
                INSERT OR REPLACE INTO bot_stats (key, value, last_updated)
                VALUES (?, ?, ?)
            ''', ('last_equity', str(equity), datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
            
            # Insert into performance table
            cursor.execute('''
                INSERT INTO performance 
                (timestamp, equity, total_equity, daily_pl, drawdown)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                equity,
                equity,
                daily_roi,
                drawdown
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logging.error(f"Error recording equity: {str(e)}")
        
    async def record_trade(self, trade_data):
        """Record completed trade metrics"""
        try:
            # Update counters
            self.trades_today += 1
            if trade_data.get('profit_loss', 0) > 0:
                self.winning_trades_today += 1
                
            self.daily_profits.append(trade_data.get('profit_loss', 0))
            
            # Record metrics in database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Update trade statistics
            win_rate = self.winning_trades_today / self.trades_today if self.trades_today > 0 else 0
            
            # Calculate profit factor
            gains = sum([p for p in self.daily_profits if p > 0])
            losses = sum([abs(p) for p in self.daily_profits if p < 0])
            profit_factor = gains / losses if losses > 0 else (1.0 if gains > 0 else 0.0)
            
            # Update bot_stats
            cursor.execute('''
                INSERT OR REPLACE INTO bot_stats (key, value, last_updated)
                VALUES (?, ?, ?)
            ''', ('win_rate', str(win_rate), datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
            
            cursor.execute('''
                INSERT OR REPLACE INTO bot_stats (key, value, last_updated)
                VALUES (?, ?, ?)
            ''', ('profit_factor', str(profit_factor), datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
            
            cursor.execute('''
                INSERT OR REPLACE INTO bot_stats (key, value, last_updated)
                VALUES (?, ?, ?)
            ''', ('trades_today', str(self.trades_today), datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logging.error(f"Error recording trade: {str(e)}")
    
    async def generate_performance_report(self):
        """Generate comprehensive performance report"""
        try:
            # Get data from database
            conn = sqlite3.connect(self.db_path)
            
            # Get equity history
            equity_df = pd.read_sql_query('''
                SELECT timestamp, equity FROM equity_history
                ORDER BY timestamp
            ''', conn)
            
            # Get trade history
            trades_df = pd.read_sql_query('''
                SELECT * FROM trades
                WHERE type IN ('sell', 'partial_sell')
                ORDER BY timestamp
            ''', conn)
            
            # Calculate key metrics
            if len(equity_df) > 0:
                # Convert timestamp to datetime
                equity_df['datetime'] = pd.to_datetime(equity_df['timestamp'], unit='s')
                
                # Get start and current equity
                start_equity = equity_df['equity'].iloc[0] if len(equity_df) > 0 else 0
                current_equity = equity_df['equity'].iloc[-1] if len(equity_df) > 0 else 0
                
                # Calculate total return and drawdown
                if start_equity > 0:
                    total_return_pct = ((current_equity / start_equity) - 1) * 100
                else:
                    total_return_pct = 0
                    
                # Calculate drawdown
                if len(equity_df) > 1:
                    equity_df['high_water_mark'] = equity_df['equity'].cummax()
                    equity_df['drawdown'] = (equity_df['high_water_mark'] - equity_df['equity']) / equity_df['high_water_mark'] * 100
                    max_drawdown = equity_df['drawdown'].max()
                else:
                    max_drawdown = 0
                    
                # Calculate daily and weekly returns
                if len(equity_df) > 1:
                    equity_df = equity_df.set_index('datetime')
                    daily_returns = equity_df['equity'].resample('D').last().pct_change() * 100
                    weekly_returns = equity_df['equity'].resample('W').last().pct_change() * 100
                    
                    avg_daily_return = daily_returns.mean() if len(daily_returns) > 0 else 0
                    avg_weekly_return = weekly_returns.mean() if len(weekly_returns) > 0 else 0
                else:
                    avg_daily_return = 0
                    avg_weekly_return = 0
            else:
                # No equity data
                total_return_pct = 0
                max_drawdown = 0
                avg_daily_return = 0
                avg_weekly_return = 0
                current_equity = 0
                start_equity = 0
                
            # Trade metrics
            if len(trades_df) > 0:
                # Count wins and losses
                winning_trades = trades_df[trades_df['profit_loss'] > 0]
                losing_trades = trades_df[trades_df['profit_loss'] <= 0]
                
                win_count = len(winning_trades)
                loss_count = len(losing_trades)
                total_trades = len(trades_df)
                
                # Calculate win rate
                win_rate = (win_count / total_trades) * 100 if total_trades > 0 else 0
                
                # Calculate profit factor
                total_profits = winning_trades['profit_loss'].sum() if len(winning_trades) > 0 else 0
                total_losses = abs(losing_trades['profit_loss'].sum()) if len(losing_trades) > 0 else 0
                
                profit_factor = total_profits / total_losses if total_losses > 0 else float('inf')
                
                # Calculate average profit and loss
                avg_profit = winning_trades['profit_loss'].mean() if len(winning_trades) > 0 else 0
                avg_loss = losing_trades['profit_loss'].mean() if len(losing_trades) > 0 else 0
                
                # Calculate expected value
                expected_value = (win_rate/100 * avg_profit) + ((1-win_rate/100) * avg_loss)
                
                # Get recent performance (last week)
                one_week_ago = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d %H:%M:%S')
                recent_trades = trades_df[trades_df['timestamp'] > one_week_ago]
                
                recent_win_count = len(recent_trades[recent_trades['profit_loss'] > 0])
                recent_trade_count = len(recent_trades)
                
                recent_win_rate = (recent_win_count / recent_trade_count) * 100 if recent_trade_count > 0 else 0
                
                # Performance by pair
                pair_performance = trades_df.groupby('pair')['profit_loss'].agg(['sum', 'count', 'mean'])
                pair_performance = pair_performance.sort_values('sum', ascending=False)
                
                best_pair = pair_performance.index[0] if len(pair_performance) > 0 else "None"
                worst_pair = pair_performance.index[-1] if len(pair_performance) > 1 else "None"
                
                best_pair_profit = pair_performance['sum'].iloc[0] if len(pair_performance) > 0 else 0
                worst_pair_profit = pair_performance['sum'].iloc[-1] if len(pair_performance) > 1 else 0
            else:
                # No trade data
                win_rate = 0
                profit_factor = 0
                total_trades = 0
                win_count = 0
                loss_count = 0
                avg_profit = 0
                avg_loss = 0
                expected_value = 0
                recent_win_rate = 0
                best_pair = "None"
                worst_pair = "None"
                best_pair_profit = 0
                worst_pair_profit = 0
            
            # Create report
            report = {
                "equity": {
                    "current": round(current_equity, 2),
                    "start": round(start_equity, 2),
                    "total_return_pct": round(total_return_pct, 2),
                    "max_drawdown_pct": round(max_drawdown, 2),
                    "avg_daily_return_pct": round(avg_daily_return, 2),
                    "avg_weekly_return_pct": round(avg_weekly_return, 2)
                },
                "trades": {
                    "total": total_trades,
                    "win_count": win_count,
                    "loss_count": loss_count,
                    "win_rate_pct": round(win_rate, 2),
                    "profit_factor": round(profit_factor, 2),
                    "avg_profit": round(avg_profit, 2),
                    "avg_loss": round(avg_loss, 2),
                    "expected_value": round(expected_value, 2),
                    "recent_win_rate_pct": round(recent_win_rate, 2)
                },
                "pairs": {
                    "best_pair": best_pair,
                    "best_pair_profit": round(best_pair_profit, 2),
                    "worst_pair": worst_pair,
                    "worst_pair_profit": round(worst_pair_profit, 2)
                },
                "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            conn.close()
            return report
            
        except Exception as e:
            logging.error(f"Error generating performance report: {str(e)}")
            return {
                "error": str(e),
                "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
    
    def check_daily_roi_target(self, current_equity, initial_equity):
        """Check if daily ROI target has been met"""
        try:
            # Calculate today's ROI
            today_date = datetime.now().strftime('%Y-%m-%d')
            
            # Get day start equity from database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT value FROM bot_stats
                WHERE key = ?
            ''', (f"equity_{today_date}_start",))
            
            result = cursor.fetchone()
            
            if result:
                day_start_equity = float(result[0])
            else:
                # First equity record of the day
                day_start_equity = current_equity
                cursor.execute('''
                    INSERT OR REPLACE INTO bot_stats (key, value, last_updated)
                    VALUES (?, ?, ?)
                ''', (f"equity_{today_date}_start", str(current_equity), datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
                
            conn.commit()
            conn.close()
            
            # Calculate daily ROI
            daily_roi = ((current_equity / day_start_equity) - 1) * 100
            
            # Check against targets
            min_target = config.TARGET_DAILY_ROI_MIN * 100
            max_target = config.TARGET_DAILY_ROI_MAX * 100
            optimal_target = config.TARGET_DAILY_ROI_OPTIMAL * 100
            
            # Check if we've met targets
            if daily_roi >= optimal_target:
                return True, f"Daily ROI target reached: {daily_roi:.2f}% (optimal target: {optimal_target:.1f}%)"
            elif daily_roi >= min_target:
                return True, f"Daily ROI minimum target reached: {daily_roi:.2f}% (min target: {min_target:.1f}%)"
            else:
                return False, f"Current daily ROI: {daily_roi:.2f}% (target: {min_target:.1f}% - {max_target:.1f}%)"
                
        except Exception as e:
            logging.error(f"Error checking daily ROI target: {str(e)}")
            return False, "Error checking daily ROI target"