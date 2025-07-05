# predictive-model/reinforcement_learning_agent.py
import gym
from gym import spaces
import numpy as np
import pandas as pd
from stable_baselines3 import PPO, A2C, SAC, TD3
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.noise import NormalActionNoise
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
import logging
import json
import redis
import psycopg2
from datetime import datetime

logger = logging.getLogger(__name__)

class AdvancedTradingEnvironment(gym.Env):
    """
    Advanced trading environment with realistic market dynamics,
    transaction costs, and risk management
    """
    
    def __init__(self, 
                 data: pd.DataFrame,
                 initial_balance: float = 10000,
                 transaction_cost: float = 0.001,
                 max_position_size: float = 0.5,
                 stop_loss: float = 0.05,
                 take_profit: float = 0.10,
                 leverage: float = 1.0,
                 render_mode: str = None):
        
        super(AdvancedTradingEnvironment, self).__init__()
        
        self.data = data
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.max_position_size = max_position_size
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.leverage = leverage
        self.render_mode = render_mode
        
        # State tracking
        self.current_step = 0
        self.current_balance = initial_balance
        self.position = 0  # Current position in base currency
        self.entry_price = 0
        self.trades = []
        self.equity_curve = [initial_balance]
        self.max_drawdown = 0
        
        # Action space: [action_type, position_size]
        # action_type: 0=hold, 1=buy, 2=sell, 3=close
        # position_size: 0-1 (percentage of available balance)
        self.action_space = spaces.Box(
            low=np.array([0, 0]),
            high=np.array([3, 1]),
            dtype=np.float32
        )
        
        # Observation space: comprehensive market state
        # Includes: OHLCV, technical indicators, portfolio state, market microstructure
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(50,),  # Adjust based on your features
            dtype=np.float32
        )
        
        # Calculate features once
        self._prepare_features()
        
    def _prepare_features(self):
        """Prepare all features for the environment"""
        # Technical indicators
        self.data['returns'] = self.data['close'].pct_change()
        self.data['log_returns'] = np.log(self.data['close'] / self.data['close'].shift(1))
        
        # Price features
        self.data['hl_ratio'] = self.data['high'] / self.data['low']
        self.data['co_ratio'] = self.data['close'] / self.data['open']
        
        # Volume features
        self.data['volume_sma'] = self.data['volume'].rolling(20).mean()
        self.data['volume_ratio'] = self.data['volume'] / self.data['volume_sma']
        
        # Volatility
        self.data['volatility'] = self.data['returns'].rolling(20).std()
        self.data['volatility_ratio'] = (
            self.data['volatility'] / self.data['volatility'].rolling(100).mean()
        )
        
        # Market microstructure
        self.data['spread'] = self.data['high'] - self.data['low']
        self.data['spread_pct'] = self.data['spread'] / self.data['close']
        
        # Fill NaN values
        self.data.fillna(method='ffill', inplace=True)
        self.data.fillna(0, inplace=True)
        
    def reset(self, seed: Optional[int] = None):
        """Reset the environment to initial state"""
        super().reset(seed=seed)
        
        # Reset state variables
        self.current_step = 50  # Start with some history
        self.current_balance = self.initial_balance
        self.position = 0
        self.entry_price = 0
        self.trades = []
        self.equity_curve = [self.initial_balance]
        self.max_drawdown = 0
        
        return self._get_observation(), {}
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one step in the environment"""
        
        # Parse action
        action_type = int(action[0])
        position_size = float(action[1])
        
        # Get current price
        current_price = self.data['close'].iloc[self.current_step]
        
        # Execute action
        reward = self._execute_action(action_type, position_size, current_price)
        
        # Update portfolio value
        portfolio_value = self._calculate_portfolio_value(current_price)
        self.equity_curve.append(portfolio_value)
        
        # Calculate drawdown
        peak = max(self.equity_curve)
        drawdown = (peak - portfolio_value) / peak
        self.max_drawdown = max(self.max_drawdown, drawdown)
        
        # Check for stop conditions
        done = False
        truncated = False
        
        # Episode ends if we run out of data
        if self.current_step >= len(self.data) - 1:
            done = True
            
        # Episode ends if we lose too much money
        if portfolio_value < self.initial_balance * 0.5:  # 50% loss
            done = True
            reward -= 1.0  # Additional penalty for large loss
            
        # Move to next step
        self.current_step += 1
        
        # Additional info for logging
        info = {
            'portfolio_value': portfolio_value,
            'position': self.position,
            'balance': self.current_balance,
            'drawdown': drawdown,
            'num_trades': len(self.trades),
            'current_price': current_price
        }
        
        return self._get_observation(), reward, done, truncated, info
    
    def _execute_action(self, action_type: int, position_size: float, 
                       current_price: float) -> float:
        """Execute trading action and calculate reward"""
        
        reward = 0
        position_value = self.position * current_price if self.position > 0 else 0
        
        # Hold
        if action_type == 0:
            # Small penalty for holding to encourage decisiveness
            reward = -0.0001
            
        # Buy
        elif action_type == 1 and self.position == 0:
            # Calculate position size
            max_position_value = self.current_balance * self.max_position_size
            position_value = max_position_value * position_size
            
            # Apply leverage
            position_value *= self.leverage
            
            # Calculate shares to buy
            shares = position_value / current_price
            
            # Transaction cost
            cost = position_value * self.transaction_cost
            
            if self.current_balance >= cost:
                self.position = shares
                self.entry_price = current_price
                self.current_balance -= cost
                
                self.trades.append({
                    'type': 'buy',
                    'price': current_price,
                    'shares': shares,
                    'value': position_value,
                    'cost': cost,
                    'timestamp': self.current_step
                })
                
                # Small reward for taking action
                reward = 0.001
            else:
                # Penalty for invalid action
                reward = -0.01
                
        # Sell (short selling)
        elif action_type == 2 and self.position == 0:
            # Similar logic for short selling
            max_position_value = self.current_balance * self.max_position_size
            position_value = max_position_value * position_size
            
            # Apply leverage
            position_value *= self.leverage
            
            # Calculate shares to short
            shares = -position_value / current_price
            
            # Transaction cost
            cost = abs(position_value) * self.transaction_cost
            
            if self.current_balance >= cost:
                self.position = shares
                self.entry_price = current_price
                self.current_balance -= cost
                
                self.trades.append({
                    'type': 'sell',
                    'price': current_price,
                    'shares': shares,
                    'value': position_value,
                    'cost': cost,
                    'timestamp': self.current_step
                })
                
                reward = 0.001
            else:
                reward = -0.01
                
        # Close position
        elif action_type == 3 and self.position != 0:
            # Calculate P&L
            if self.position > 0:  # Long position
                pnl = (current_price - self.entry_price) * self.position
            else:  # Short position
                pnl = (self.entry_price - current_price) * abs(self.position)
                
            # Transaction cost
            cost = abs(self.position * current_price) * self.transaction_cost
            
            # Update balance
            self.current_balance += pnl - cost
            
            # Calculate return
            position_return = pnl / (abs(self.position) * self.entry_price)
            
            # Reward based on return
            reward = position_return * 10  # Scale reward
            
            # Additional rewards/penalties
            if position_return > 0:
                # Bonus for profitable trade
                reward += 0.01
                
                # Extra bonus for very profitable trades
                if position_return > 0.05:
                    reward += 0.05
            else:
                # Penalty for losing trade
                reward -= 0.01
                
            self.trades.append({
                'type': 'close',
                'price': current_price,
                'pnl': pnl,
                'return': position_return,
                'cost': cost,
                'timestamp': self.current_step
            })
            
            self.position = 0
            self.entry_price = 0
            
        # Check stop loss and take profit
        if self.position != 0:
            if self.position > 0:  # Long position
                current_return = (current_price - self.entry_price) / self.entry_price
            else:  # Short position
                current_return = (self.entry_price - current_price) / self.entry_price
                
            # Stop loss
            if current_return <= -self.stop_loss:
                self._force_close_position(current_price, 'stop_loss')
                reward = -0.5  # Large penalty for hitting stop loss
                
            # Take profit
            elif current_return >= self.take_profit:
                self._force_close_position(current_price, 'take_profit')
                reward = 0.5  # Large reward for hitting take profit
                
        return reward
    
    def _force_close_position(self, current_price: float, reason: str):
        """Force close position (stop loss or take profit)"""
        if self.position > 0:  # Long position
            pnl = (current_price - self.entry_price) * self.position
        else:  # Short position
            pnl = (self.entry_price - current_price) * abs(self.position)
            
        # Transaction cost
        cost = abs(self.position * current_price) * self.transaction_cost
        
        # Update balance
        self.current_balance += pnl - cost
        
        self.trades.append({
            'type': f'close_{reason}',
            'price': current_price,
            'pnl': pnl,
            'cost': cost,
            'timestamp': self.current_step
        })
        
        self.position = 0
        self.entry_price = 0
    
    def _calculate_portfolio_value(self, current_price: float) -> float:
        """Calculate total portfolio value"""
        position_value = 0
        
        if self.position > 0:  # Long position
            position_value = self.position * current_price
        elif self.position < 0:  # Short position
            # For short position, calculate unrealized P&L
            position_value = abs(self.position) * self.entry_price
            unrealized_pnl = (self.entry_price - current_price) * abs(self.position)
            position_value += unrealized_pnl
            
        return self.current_balance + position_value
    
    def _get_observation(self) -> np.ndarray:
        """Get current state observation"""
        idx = self.current_step
        
        # Market features
        market_features = [
            self.data['close'].iloc[idx],
            self.data['open'].iloc[idx],
            self.data['high'].iloc[idx],
            self.data['low'].iloc[idx],
            self.data['volume'].iloc[idx],
            self.data['returns'].iloc[idx],
            self.data['volatility'].iloc[idx],
            self.data['rsi'].iloc[idx] if 'rsi' in self.data else 50,
            self.data['macd'].iloc[idx] if 'macd' in self.data else 0,
            self.data['bb_upper'].iloc[idx] if 'bb_upper' in self.data else self.data['close'].iloc[idx],
            self.data['bb_lower'].iloc[idx] if 'bb_lower' in self.data else self.data['close'].iloc[idx],
        ]
        
        # Portfolio features
        portfolio_value = self._calculate_portfolio_value(self.data['close'].iloc[idx])
        
        portfolio_features = [
            self.current_balance / self.initial_balance,
            portfolio_value / self.initial_balance,
            1 if self.position > 0 else (-1 if self.position < 0 else 0),
            abs(self.position) * self.data['close'].iloc[idx] / portfolio_value if portfolio_value > 0 else 0,
            len(self.trades),
            self.max_drawdown
        ]
        
        # Recent price action (last 5 periods)
        recent_returns = []
        for i in range(1, 6):
            if idx - i >= 0:
                recent_returns.append(self.data['returns'].iloc[idx - i])
            else:
                recent_returns.append(0)
                
        # Time features
        time_features = [
            idx / len(self.data),  # Progress through dataset
            np.sin(2 * np.pi * idx / 24),  # Hour of day (if hourly data)
            np.cos(2 * np.pi * idx / 24),
        ]
        
        # Combine all features
        observation = np.array(
            market_features + portfolio_features + recent_returns + time_features,
            dtype=np.float32
        )
        
        # Pad to match observation space if needed
        if len(observation) < self.observation_space.shape[0]:
            observation = np.pad(
                observation, 
                (0, self.observation_space.shape[0] - len(observation)),
                mode='constant'
            )
            
        return observation
    
    def render(self):
        """Render the environment (optional)"""
        if self.render_mode == "human":
            print(f"Step: {self.current_step}")
            print(f"Price: {self.data['close'].iloc[self.current_step]:.2f}")
            print(f"Portfolio Value: {self._calculate_portfolio_value(self.data['close'].iloc[self.current_step]):.2f}")
            print(f"Position: {self.position:.4f}")
            print(f"Balance: {self.current_balance:.2f}")
            print(f"Trades: {len(self.trades)}")
            print("-" * 50)


class ReinforcementLearningTrader:
    """Complete RL trading system with multiple algorithms"""
    
    def __init__(self, 
                 algorithm: str = 'PPO',
                 model_save_path: str = './models/rl',
                 redis_host: str = 'localhost',
                 redis_port: int = 6379):
        
        self.algorithm = algorithm
        self.model_save_path = model_save_path
        self.model = None
        self.env = None
        
        # Redis for metrics
        self.redis_client = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
        
        # Supported algorithms
        self.algorithms = {
            'PPO': PPO,
            'A2C': A2C,
            'SAC': SAC,
            'TD3': TD3
        }
        
    def create_env(self, data: pd.DataFrame, **env_kwargs) -> gym.Env:
        """Create trading environment"""
        return AdvancedTradingEnvironment(data, **env_kwargs)
    
    def train(self, 
              train_data: pd.DataFrame,
              val_data: pd.DataFrame,
              total_timesteps: int = 1000000,
              n_envs: int = 4,
              **kwargs):
        """Train the RL agent"""
        
        logger.info(f"Starting {self.algorithm} training...")
        
        # Create vectorized environments for parallel training
        train_envs = DummyVecEnv([
            lambda: self.create_env(train_data, **kwargs) for _ in range(n_envs)
        ])
        
        # Create evaluation environment
        eval_env = self.create_env(val_data, **kwargs)
        
        # Select algorithm
        algorithm_class = self.algorithms[self.algorithm]
        
        # Model configuration based on algorithm
        if self.algorithm == 'PPO':
            policy_kwargs = dict(
                net_arch=[dict(pi=[256, 256], vf=[256, 256])],
                activation_fn=nn.ReLU
            )
            
            self.model = algorithm_class(
                "MlpPolicy",
                train_envs,
                learning_rate=3e-4,
                n_steps=2048,
                batch_size=64,
                n_epochs=10,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                clip_range_vf=None,
                ent_coef=0.01,
                vf_coef=0.5,
                max_grad_norm=0.5,
                use_sde=False,
                sde_sample_freq=-1,
                policy_kwargs=policy_kwargs,
                verbose=1,
                tensorboard_log=f"./tensorboard/{self.algorithm}"
            )
            
        elif self.algorithm == 'SAC':
            policy_kwargs = dict(
                net_arch=[256, 256],
                activation_fn=nn.ReLU
            )
            
            # Add noise for exploration
            n_actions = train_envs.action_space.shape[-1]
            action_noise = NormalActionNoise(
                mean=np.zeros(n_actions),
                sigma=0.1 * np.ones(n_actions)
            )
            
            self.model = algorithm_class(
                "MlpPolicy",
                train_envs,
                learning_rate=3e-4,
                buffer_size=100000,
                learning_starts=1000,
                batch_size=256,
                tau=0.005,
                gamma=0.99,
                train_freq=1,
                gradient_steps=1,
                action_noise=action_noise,
                policy_kwargs=policy_kwargs,
                verbose=1,
                tensorboard_log=f"./tensorboard/{self.algorithm}"
            )
            
        # Callbacks
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=self.model_save_path,
            log_path=f"{self.model_save_path}/logs",
            eval_freq=10000,
            deterministic=True,
            render=False,
            n_eval_episodes=10
        )
        
        # Custom callback for tracking metrics
        class MetricsCallback(BaseCallback):
            def __init__(self, redis_client):
                super().__init__()
                self.redis_client = redis_client
                self.episode_rewards = []
                self.episode_lengths = []
                
            def _on_rollout_end(self):
                # Log metrics to Redis
                metrics = {
                    'timesteps': self.num_timesteps,
                    'episodes': len(self.episode_rewards),
                    'mean_reward': np.mean(self.episode_rewards[-100:]) if self.episode_rewards else 0,
                    'mean_length': np.mean(self.episode_lengths[-100:]) if self.episode_lengths else 0,
                    'timestamp': datetime.now().isoformat()
                }
                
                self.redis_client.hset(
                    'rl_training_metrics',
                    f'step_{self.num_timesteps}',
                    json.dumps(metrics)
                )
                
                return True
                
            def _on_step(self):
                if self.locals.get('dones')[0]:
                    self.episode_rewards.append(self.locals['rewards'][0])
                    self.episode_lengths.append(self.locals['infos'][0].get('episode', {}).get('l', 0))
                    
                return True
        
        metrics_callback = MetricsCallback(self.redis_client)
        
        # Train the model
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=[eval_callback, metrics_callback],
            log_interval=100,
            tb_log_name=f"{self.algorithm}_run",
            reset_num_timesteps=True,
            progress_bar=True
        )
        
        # Save final model
        self.model.save(f"{self.model_save_path}/{self.algorithm}_final")
        
        return self.model
    
    def predict(self, observation: np.ndarray, deterministic: bool = True):
        """Make trading decision"""
        if self.model is None:
            raise ValueError("Model not trained or loaded")
            
        action, _states = self.model.predict(observation, deterministic=deterministic)
        return action
    
    def backtest(self, test_data: pd.DataFrame, **env_kwargs) -> Dict:
        """Backtest the trained model"""
        
        logger.info("Starting backtest...")
        
        # Create test environment
        test_env = self.create_env(test_data, **env_kwargs)
        
        # Run backtest
        obs, _ = test_env.reset()
        done = False
        
        while not done:
            action = self.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = test_env.step(action)
            
        # Get results
        results = {
            'total_return': (info['portfolio_value'] - test_env.initial_balance) / test_env.initial_balance,
            'sharpe_ratio': self._calculate_sharpe_ratio(test_env.equity_curve),
            'max_drawdown': test_env.max_drawdown,
            'num_trades': len(test_env.trades),
            'win_rate': self._calculate_win_rate(test_env.trades),
            'profit_factor': self._calculate_profit_factor(test_env.trades),
            'equity_curve': test_env.equity_curve,
            'trades': test_env.trades
        }
        
        # Save results
        self._save_backtest_results(results)
        
        return results
    
    def _calculate_sharpe_ratio(self, equity_curve: List[float]) -> float:
        """Calculate Sharpe ratio"""
        returns = np.diff(equity_curve) / equity_curve[:-1]
        
        if len(returns) < 2:
            return 0
            
        return np.sqrt(252) * np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
    
    def _calculate_win_rate(self, trades: List[Dict]) -> float:
        """Calculate win rate"""
        if not trades:
            return 0
            
        wins = sum(1 for trade in trades if trade.get('pnl', 0) > 0)
        return wins / len(trades)
    
    def _calculate_profit_factor(self, trades: List[Dict]) -> float:
        """Calculate profit factor"""
        if not trades:
            return 0
            
        gross_profit = sum(trade.get('pnl', 0) for trade in trades if trade.get('pnl', 0) > 0)
        gross_loss = abs(sum(trade.get('pnl', 0) for trade in trades if trade.get('pnl', 0) < 0))
        
        return gross_profit / gross_loss if gross_loss > 0 else float('inf')
    
    def _save_backtest_results(self, results: Dict):
        """Save backtest results to Redis"""
        
        # Remove non-serializable data
        results_to_save = {
            'total_return': results['total_return'],
            'sharpe_ratio': results['sharpe_ratio'],
            'max_drawdown': results['max_drawdown'],
            'num_trades': results['num_trades'],
            'win_rate': results['win_rate'],
            'profit_factor': results['profit_factor'],
            'timestamp': datetime.now().isoformat()
        }
        
        self.redis_client.set(
            f'backtest_results_{self.algorithm}',
            json.dumps(results_to_save)
        )
        
        logger.info(f"Backtest results saved: {results_to_save}")