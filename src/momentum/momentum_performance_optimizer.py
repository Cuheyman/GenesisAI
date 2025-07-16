#!/usr/bin/env python3
"""
Performance Optimizer & Analytics System
Continuously monitors and optimizes the momentum strategy for 75-90% win rate

Features:
- Real-time performance tracking
- Automatic threshold optimization
- Signal quality analytics
- Win rate prediction
- Strategy refinement suggestions
"""

import asyncio
import logging
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import sqlite3
from pathlib import Path

@dataclass
class TradeOutcome:
    """Individual trade outcome tracking"""
    pair: str
    entry_time: datetime
    exit_time: Optional[datetime]
    entry_price: float
    exit_price: Optional[float]
    signal_confidence: float
    entry_quality_score: float
    signal_strength: str
    breakout_type: str
    volume_confirmed: bool
    momentum_confirmed: bool
    is_high_probability: bool
    risk_reward_ratio: float
    
    # Outcome metrics
    pnl_percentage: Optional[float] = None
    is_winner: Optional[bool] = None
    hold_duration_hours: Optional[float] = None
    max_drawdown: Optional[float] = None
    max_profit: Optional[float] = None
    
    # Context at entry
    market_phase: str = "UNKNOWN"
    rsi_at_entry: Optional[float] = None
    volume_spike_ratio: Optional[float] = None
    
    def calculate_outcome(self):
        """Calculate trade outcome metrics"""
        if self.exit_price and self.exit_time:
            self.pnl_percentage = ((self.exit_price - self.entry_price) / self.entry_price) * 100
            self.is_winner = self.pnl_percentage > 0
            self.hold_duration_hours = (self.exit_time - self.entry_time).total_seconds() / 3600

@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics"""
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    max_consecutive_wins: int
    max_consecutive_losses: int
    avg_hold_time_hours: float
    best_trade: float
    worst_trade: float
    total_pnl: float
    
    # Quality-based metrics
    high_prob_win_rate: float
    excellent_signal_win_rate: float
    strong_signal_win_rate: float
    
    # Strategy-specific metrics
    volume_confirmed_win_rate: float
    breakout_confirmed_win_rate: float
    momentum_confirmed_win_rate: float

class PerformanceOptimizer:
    """
    Advanced performance optimization system for momentum strategy
    Continuously analyzes performance and suggests improvements
    """
    
    def __init__(self, db_path: str = "momentum_performance.db"):
        self.db_path = Path(db_path)
        self.logger = logging.getLogger(__name__)
        
        # Initialize database
        self._init_database()
        
        # Performance tracking
        self.active_trades = {}  # pair -> TradeOutcome
        self.completed_trades = deque(maxlen=1000)  # Last 1000 trades
        self.performance_history = deque(maxlen=100)  # Last 100 performance snapshots
        
        # Optimization parameters
        self.optimization_targets = {
            'min_win_rate': 0.75,
            'target_win_rate': 0.85,
            'min_profit_factor': 1.5,
            'target_profit_factor': 2.5,
            'max_consecutive_losses': 5
        }
        
        # Adaptive thresholds
        self.adaptive_thresholds = {
            'confluence_score': 70.0,
            'confidence_score': 75.0,
            'volume_spike_min': 1.8,
            'rsi_entry_max': 65.0,
            'momentum_strength_min': 'MODERATE'
        }
        
        # Analytics cache
        self.analytics_cache = {}
        self.last_optimization = datetime.now()
        
    def _init_database(self):
        """Initialize SQLite database for persistent storage"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS trades (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        pair TEXT,
                        entry_time TEXT,
                        exit_time TEXT,
                        entry_price REAL,
                        exit_price REAL,
                        signal_confidence REAL,
                        entry_quality_score REAL,
                        signal_strength TEXT,
                        breakout_type TEXT,
                        volume_confirmed INTEGER,
                        momentum_confirmed INTEGER,
                        is_high_probability INTEGER,
                        risk_reward_ratio REAL,
                        pnl_percentage REAL,
                        is_winner INTEGER,
                        hold_duration_hours REAL,
                        market_phase TEXT,
                        rsi_at_entry REAL,
                        volume_spike_ratio REAL,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS performance_snapshots (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT,
                        win_rate REAL,
                        total_trades INTEGER,
                        profit_factor REAL,
                        avg_hold_time REAL,
                        confluence_threshold REAL,
                        confidence_threshold REAL,
                        high_prob_win_rate REAL,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                conn.commit()
                self.logger.info("Database initialized successfully")
                
        except Exception as e:
            self.logger.error(f"Database initialization error: {str(e)}")
    
    async def track_trade_entry(self, pair: str, entry_data: Dict[str, Any]) -> str:
        """Track a new trade entry"""
        
        trade_id = f"{pair}_{int(datetime.now().timestamp())}"
        
        trade_outcome = TradeOutcome(
            pair=pair,
            entry_time=datetime.now(),
            exit_time=None,
            entry_price=entry_data.get('entry_price', 0.0),
            exit_price=None,
            signal_confidence=entry_data.get('confidence', 0.0),
            entry_quality_score=entry_data.get('entry_quality_score', 0.0),
            signal_strength=entry_data.get('signal_strength', 'UNKNOWN'),
            breakout_type=entry_data.get('breakout_type', 'NONE'),
            volume_confirmed=entry_data.get('volume_confirmed', False),
            momentum_confirmed=entry_data.get('momentum_confirmed', False),
            is_high_probability=entry_data.get('is_high_probability', False),
            risk_reward_ratio=entry_data.get('risk_reward_ratio', 1.0),
            market_phase=entry_data.get('market_phase', 'UNKNOWN'),
            rsi_at_entry=entry_data.get('rsi_at_entry'),
            volume_spike_ratio=entry_data.get('volume_spike_ratio')
        )
        
        self.active_trades[trade_id] = trade_outcome
        
        self.logger.info(f"""
        TRADE ENTRY TRACKED: {trade_id}
        Pair: {pair}
        Entry Price: {trade_outcome.entry_price}
        Confidence: {trade_outcome.signal_confidence:.1f}%
        Quality Score: {trade_outcome.entry_quality_score:.1f}
        High Probability: {trade_outcome.is_high_probability}
        """)
        
        return trade_id
    
    async def track_trade_exit(self, trade_id: str, exit_price: float, exit_reason: str = "MANUAL"):
        """Track trade exit and calculate performance"""
        
        if trade_id not in self.active_trades:
            self.logger.warning(f"Trade ID {trade_id} not found in active trades")
            return
        
        trade = self.active_trades[trade_id]
        trade.exit_time = datetime.now()
        trade.exit_price = exit_price
        trade.calculate_outcome()
        
        # Move to completed trades
        self.completed_trades.append(trade)
        del self.active_trades[trade_id]
        
        # Save to database
        await self._save_trade_to_db(trade)
        
        # Log outcome
        self.logger.info(f"""
        TRADE EXIT TRACKED: {trade_id}
        Pair: {trade.pair}
        PnL: {trade.pnl_percentage:.2f}%
        Winner: {trade.is_winner}
        Hold Time: {trade.hold_duration_hours:.1f}h
        Exit Reason: {exit_reason}
        """)
        
        # Trigger optimization check if needed
        if len(self.completed_trades) % 10 == 0:  # Every 10 trades
            await self._check_optimization_triggers()
    
    async def _save_trade_to_db(self, trade: TradeOutcome):
        """Save completed trade to database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT INTO trades (
                        pair, entry_time, exit_time, entry_price, exit_price,
                        signal_confidence, entry_quality_score, signal_strength,
                        breakout_type, volume_confirmed, momentum_confirmed,
                        is_high_probability, risk_reward_ratio, pnl_percentage,
                        is_winner, hold_duration_hours, market_phase,
                        rsi_at_entry, volume_spike_ratio
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    trade.pair, trade.entry_time.isoformat(), 
                    trade.exit_time.isoformat() if trade.exit_time else None,
                    trade.entry_price, trade.exit_price, trade.signal_confidence,
                    trade.entry_quality_score, trade.signal_strength,
                    trade.breakout_type, int(trade.volume_confirmed),
                    int(trade.momentum_confirmed), int(trade.is_high_probability),
                    trade.risk_reward_ratio, trade.pnl_percentage,
                    int(trade.is_winner) if trade.is_winner is not None else None,
                    trade.hold_duration_hours, trade.market_phase,
                    trade.rsi_at_entry, trade.volume_spike_ratio
                ))
                conn.commit()
        except Exception as e:
            self.logger.error(f"Error saving trade to database: {str(e)}")
    
    async def get_current_performance(self) -> PerformanceMetrics:
        """Calculate current performance metrics"""
        
        if len(self.completed_trades) < 10:
            return self._get_default_metrics()
        
        trades = list(self.completed_trades)
        winning_trades = [t for t in trades if t.is_winner]
        losing_trades = [t for t in trades if not t.is_winner]
        
        # Basic metrics
        total_trades = len(trades)
        win_count = len(winning_trades)
        loss_count = len(losing_trades)
        win_rate = win_count / total_trades if total_trades > 0 else 0
        
        # PnL metrics
        avg_win = np.mean([t.pnl_percentage for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t.pnl_percentage for t in losing_trades]) if losing_trades else 0
        profit_factor = abs(avg_win * win_count / (avg_loss * loss_count)) if loss_count > 0 and avg_loss != 0 else float('inf')
        
        # Consecutive metrics
        max_consecutive_wins = self._calculate_max_consecutive(trades, True)
        max_consecutive_losses = self._calculate_max_consecutive(trades, False)
        
        # Time metrics
        avg_hold_time = np.mean([t.hold_duration_hours for t in trades if t.hold_duration_hours])
        
        # Best/worst trades
        best_trade = max([t.pnl_percentage for t in trades if t.pnl_percentage])
        worst_trade = min([t.pnl_percentage for t in trades if t.pnl_percentage])
        total_pnl = sum([t.pnl_percentage for t in trades if t.pnl_percentage])
        
        # Quality-based metrics
        high_prob_trades = [t for t in trades if t.is_high_probability]
        high_prob_win_rate = len([t for t in high_prob_trades if t.is_winner]) / len(high_prob_trades) if high_prob_trades else 0
        
        excellent_trades = [t for t in trades if t.signal_strength == 'EXCELLENT']
        excellent_win_rate = len([t for t in excellent_trades if t.is_winner]) / len(excellent_trades) if excellent_trades else 0
        
        strong_trades = [t for t in trades if t.signal_strength == 'STRONG']
        strong_win_rate = len([t for t in strong_trades if t.is_winner]) / len(strong_trades) if strong_trades else 0
        
        # Strategy-specific metrics
        volume_trades = [t for t in trades if t.volume_confirmed]
        volume_win_rate = len([t for t in volume_trades if t.is_winner]) / len(volume_trades) if volume_trades else 0
        
        breakout_trades = [t for t in trades if t.breakout_type != 'NONE']
        breakout_win_rate = len([t for t in breakout_trades if t.is_winner]) / len(breakout_trades) if breakout_trades else 0
        
        momentum_trades = [t for t in trades if t.momentum_confirmed]
        momentum_win_rate = len([t for t in momentum_trades if t.is_winner]) / len(momentum_trades) if momentum_trades else 0
        
        return PerformanceMetrics(
            total_trades=total_trades,
            winning_trades=win_count,
            losing_trades=loss_count,
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            max_consecutive_wins=max_consecutive_wins,
            max_consecutive_losses=max_consecutive_losses,
            avg_hold_time_hours=avg_hold_time,
            best_trade=best_trade,
            worst_trade=worst_trade,
            total_pnl=total_pnl,
            high_prob_win_rate=high_prob_win_rate,
            excellent_signal_win_rate=excellent_win_rate,
            strong_signal_win_rate=strong_win_rate,
            volume_confirmed_win_rate=volume_win_rate,
            breakout_confirmed_win_rate=breakout_win_rate,
            momentum_confirmed_win_rate=momentum_win_rate
        )
    
    async def _check_optimization_triggers(self):
        """Check if optimization is needed and trigger if necessary"""
        
        current_metrics = await self.get_current_performance()
        
        # Check if we need to optimize
        needs_optimization = (
            current_metrics.win_rate < self.optimization_targets['min_win_rate'] or
            current_metrics.profit_factor < self.optimization_targets['min_profit_factor'] or
            current_metrics.max_consecutive_losses > self.optimization_targets['max_consecutive_losses']
        )
        
        if needs_optimization:
            self.logger.warning(f"""
            OPTIMIZATION TRIGGER ACTIVATED
            Current Win Rate: {current_metrics.win_rate*100:.1f}%
            Target Win Rate: {self.optimization_targets['target_win_rate']*100:.1f}%
            Current Profit Factor: {current_metrics.profit_factor:.2f}
            Max Consecutive Losses: {current_metrics.max_consecutive_losses}
            """)
            
            await self._optimize_thresholds(current_metrics)
    
    async def _optimize_thresholds(self, current_metrics: PerformanceMetrics):
        """Optimize strategy thresholds based on performance analysis"""
        
        optimization_suggestions = []
        
        # Analyze which signal types perform best
        signal_analysis = await self._analyze_signal_performance()
        
        # If win rate is too low, increase selectivity
        if current_metrics.win_rate < self.optimization_targets['min_win_rate']:
            
            # Increase confluence score threshold
            if current_metrics.high_prob_win_rate > current_metrics.win_rate * 1.1:
                new_confluence = min(85, self.adaptive_thresholds['confluence_score'] + 5)
                optimization_suggestions.append(f"Increase confluence threshold: {self.adaptive_thresholds['confluence_score']} -> {new_confluence}")
                self.adaptive_thresholds['confluence_score'] = new_confluence
            
            # Increase confidence threshold
            new_confidence = min(90, self.adaptive_thresholds['confidence_score'] + 5)
            optimization_suggestions.append(f"Increase confidence threshold: {self.adaptive_thresholds['confidence_score']} -> {new_confidence}")
            self.adaptive_thresholds['confidence_score'] = new_confidence
            
            # Require stronger volume confirmation
            if current_metrics.volume_confirmed_win_rate > current_metrics.win_rate * 1.05:
                new_volume = min(3.0, self.adaptive_thresholds['volume_spike_min'] + 0.2)
                optimization_suggestions.append(f"Increase volume spike requirement: {self.adaptive_thresholds['volume_spike_min']} -> {new_volume}")
                self.adaptive_thresholds['volume_spike_min'] = new_volume
        
        # If consecutive losses are too high, be more conservative with RSI
        if current_metrics.max_consecutive_losses > self.optimization_targets['max_consecutive_losses']:
            new_rsi_max = max(60, self.adaptive_thresholds['rsi_entry_max'] - 3)
            optimization_suggestions.append(f"Lower RSI entry maximum: {self.adaptive_thresholds['rsi_entry_max']} -> {new_rsi_max}")
            self.adaptive_thresholds['rsi_entry_max'] = new_rsi_max
        
        # Log optimization actions
        if optimization_suggestions:
            self.logger.info("THRESHOLD OPTIMIZATION APPLIED:")
            for suggestion in optimization_suggestions:
                self.logger.info(f"  - {suggestion}")
            
            # Save optimization to database
            await self._save_performance_snapshot(current_metrics)
        
        self.last_optimization = datetime.now()
    
    async def _analyze_signal_performance(self) -> Dict[str, Any]:
        """Analyze performance by signal characteristics"""
        
        if len(self.completed_trades) < 20:
            return {}
        
        trades = list(self.completed_trades)
        analysis = {}
        
        # Performance by signal strength
        by_strength = defaultdict(list)
        for trade in trades:
            by_strength[trade.signal_strength].append(trade.is_winner)
        
        analysis['by_signal_strength'] = {
            strength: {
                'win_rate': sum(wins)/len(wins) if wins else 0,
                'trade_count': len(wins)
            }
            for strength, wins in by_strength.items()
        }
        
        # Performance by breakout type
        by_breakout = defaultdict(list)
        for trade in trades:
            by_breakout[trade.breakout_type].append(trade.is_winner)
        
        analysis['by_breakout_type'] = {
            breakout_type: {
                'win_rate': sum(wins)/len(wins) if wins else 0,
                'trade_count': len(wins)
            }
            for breakout_type, wins in by_breakout.items()
        }
        
        # Performance by market phase
        by_phase = defaultdict(list)
        for trade in trades:
            by_phase[trade.market_phase].append(trade.is_winner)
        
        analysis['by_market_phase'] = {
            phase: {
                'win_rate': sum(wins)/len(wins) if wins else 0,
                'trade_count': len(wins)
            }
            for phase, wins in by_phase.items()
        }
        
        return analysis
    
    async def _save_performance_snapshot(self, metrics: PerformanceMetrics):
        """Save performance snapshot to database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT INTO performance_snapshots (
                        timestamp, win_rate, total_trades, profit_factor,
                        avg_hold_time, confluence_threshold, confidence_threshold,
                        high_prob_win_rate
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    datetime.now().isoformat(), metrics.win_rate, metrics.total_trades,
                    metrics.profit_factor, metrics.avg_hold_time_hours,
                    self.adaptive_thresholds['confluence_score'],
                    self.adaptive_thresholds['confidence_score'],
                    metrics.high_prob_win_rate
                ))
                conn.commit()
        except Exception as e:
            self.logger.error(f"Error saving performance snapshot: {str(e)}")
    
    def get_optimization_recommendations(self) -> List[str]:
        """Get current optimization recommendations"""
        
        recommendations = []
        
        if len(self.completed_trades) < 20:
            recommendations.append("Collect more trade data (minimum 20 trades) for meaningful optimization")
            return recommendations
        
        recent_trades = list(self.completed_trades)[-20:]
        recent_win_rate = sum([1 for t in recent_trades if t.is_winner]) / len(recent_trades)
        
        if recent_win_rate < 0.75:
            recommendations.append("Recent win rate below target (75%) - consider increasing selectivity")
            recommendations.append(f"Current thresholds: Confluence {self.adaptive_thresholds['confluence_score']:.0f}%, Confidence {self.adaptive_thresholds['confidence_score']:.0f}%")
        
        # Analyze best performing characteristics
        high_prob_trades = [t for t in recent_trades if t.is_high_probability]
        if high_prob_trades:
            hp_win_rate = sum([1 for t in high_prob_trades if t.is_winner]) / len(high_prob_trades)
            if hp_win_rate > recent_win_rate * 1.1:
                recommendations.append(f"High-probability trades performing well ({hp_win_rate*100:.1f}% win rate) - focus on these setups")
        
        volume_confirmed_trades = [t for t in recent_trades if t.volume_confirmed]
        if volume_confirmed_trades:
            vol_win_rate = sum([1 for t in volume_confirmed_trades if t.is_winner]) / len(volume_confirmed_trades)
            if vol_win_rate > recent_win_rate * 1.05:
                recommendations.append(f"Volume-confirmed trades outperforming ({vol_win_rate*100:.1f}% vs {recent_win_rate*100:.1f}%) - prioritize volume confirmation")
        
        return recommendations
    
    def get_current_thresholds(self) -> Dict[str, Any]:
        """Get current adaptive thresholds"""
        return self.adaptive_thresholds.copy()
    
    async def generate_performance_report(self) -> str:
        """Generate comprehensive performance report"""
        
        metrics = await self.get_current_performance()
        recommendations = self.get_optimization_recommendations()
        signal_analysis = await self._analyze_signal_performance()
        
        report = f"""
=== MOMENTUM STRATEGY PERFORMANCE REPORT ===
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

OVERALL PERFORMANCE:
- Total Trades: {metrics.total_trades}
- Win Rate: {metrics.win_rate*100:.1f}% (Target: {self.optimization_targets['target_win_rate']*100:.1f}%)
- Profit Factor: {metrics.profit_factor:.2f} (Target: {self.optimization_targets['target_profit_factor']:.1f}+)
- Average Win: {metrics.avg_win:.2f}%
- Average Loss: {metrics.avg_loss:.2f}%
- Best Trade: {metrics.best_trade:.2f}%
- Worst Trade: {metrics.worst_trade:.2f}%
- Total PnL: {metrics.total_pnl:.2f}%

QUALITY METRICS:
- High Probability Win Rate: {metrics.high_prob_win_rate*100:.1f}%
- Excellent Signal Win Rate: {metrics.excellent_signal_win_rate*100:.1f}%
- Strong Signal Win Rate: {metrics.strong_signal_win_rate*100:.1f}%

STRATEGY CONFIRMATION METRICS:
- Volume Confirmed Win Rate: {metrics.volume_confirmed_win_rate*100:.1f}%
- Breakout Confirmed Win Rate: {metrics.breakout_confirmed_win_rate*100:.1f}%
- Momentum Confirmed Win Rate: {metrics.momentum_confirmed_win_rate*100:.1f}%

RISK METRICS:
- Max Consecutive Wins: {metrics.max_consecutive_wins}
- Max Consecutive Losses: {metrics.max_consecutive_losses}
- Average Hold Time: {metrics.avg_hold_time_hours:.1f} hours

CURRENT ADAPTIVE THRESHOLDS:
- Confluence Score: {self.adaptive_thresholds['confluence_score']:.0f}%
- Confidence Score: {self.adaptive_thresholds['confidence_score']:.0f}%
- Volume Spike Minimum: {self.adaptive_thresholds['volume_spike_min']:.1f}x
- RSI Entry Maximum: {self.adaptive_thresholds['rsi_entry_max']:.0f}

OPTIMIZATION RECOMMENDATIONS:
"""
        
        for i, rec in enumerate(recommendations, 1):
            report += f"{i}. {rec}\n"
        
        if signal_analysis:
            report += "\nSIGNAL PERFORMANCE ANALYSIS:\n"
            
            if 'by_signal_strength' in signal_analysis:
                report += "By Signal Strength:\n"
                for strength, data in signal_analysis['by_signal_strength'].items():
                    report += f"  - {strength}: {data['win_rate']*100:.1f}% win rate ({data['trade_count']} trades)\n"
            
            if 'by_breakout_type' in signal_analysis:
                report += "By Breakout Type:\n"
                for breakout, data in signal_analysis['by_breakout_type'].items():
                    report += f"  - {breakout}: {data['win_rate']*100:.1f}% win rate ({data['trade_count']} trades)\n"
        
        return report
    
    # Helper methods
    
    def _calculate_max_consecutive(self, trades: List[TradeOutcome], target_outcome: bool) -> int:
        """Calculate maximum consecutive wins or losses"""
        max_consecutive = 0
        current_consecutive = 0
        
        for trade in trades:
            if trade.is_winner == target_outcome:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0
        
        return max_consecutive
    
    def _get_default_metrics(self) -> PerformanceMetrics:
        """Get default metrics when insufficient data"""
        return PerformanceMetrics(
            total_trades=0, winning_trades=0, losing_trades=0,
            win_rate=0.0, avg_win=0.0, avg_loss=0.0, profit_factor=0.0,
            max_consecutive_wins=0, max_consecutive_losses=0,
            avg_hold_time_hours=0.0, best_trade=0.0, worst_trade=0.0,
            total_pnl=0.0, high_prob_win_rate=0.0,
            excellent_signal_win_rate=0.0, strong_signal_win_rate=0.0,
            volume_confirmed_win_rate=0.0, breakout_confirmed_win_rate=0.0,
            momentum_confirmed_win_rate=0.0
        )

# Real-time Performance Monitor
class RealTimePerformanceMonitor:
    """Real-time performance monitoring with alerts"""
    
    def __init__(self, optimizer: PerformanceOptimizer):
        self.optimizer = optimizer
        self.logger = logging.getLogger(__name__)
        self.alert_thresholds = {
            'consecutive_losses': 3,
            'win_rate_drop': 0.10,  # Alert if win rate drops 10% below target
            'drawdown_limit': 0.15   # Alert if drawdown exceeds 15%
        }
        
    async def monitor_performance(self):
        """Continuous performance monitoring"""
        
        while True:
            try:
                # Check every 5 minutes
                await asyncio.sleep(300)
                
                current_metrics = await self.optimizer.get_current_performance()
                
                # Check alert conditions
                await self._check_performance_alerts(current_metrics)
                
            except Exception as e:
                self.logger.error(f"Error in performance monitoring: {str(e)}")
                await asyncio.sleep(60)
    
    async def _check_performance_alerts(self, metrics: PerformanceMetrics):
        """Check for performance alert conditions"""
        
        # Consecutive losses alert
        if metrics.max_consecutive_losses >= self.alert_thresholds['consecutive_losses']:
            self.logger.warning(f"ALERT: {metrics.max_consecutive_losses} consecutive losses detected!")
        
        # Win rate drop alert
        target_win_rate = self.optimizer.optimization_targets['target_win_rate']
        if metrics.win_rate < target_win_rate - self.alert_thresholds['win_rate_drop']:
            self.logger.warning(f"ALERT: Win rate ({metrics.win_rate*100:.1f}%) significantly below target ({target_win_rate*100:.1f}%)")
        
        # Recent performance check
        if len(self.optimizer.completed_trades) >= 10:
            recent_trades = list(self.optimizer.completed_trades)[-10:]
            recent_win_rate = sum([1 for t in recent_trades if t.is_winner]) / len(recent_trades)
            
            if recent_win_rate < 0.6:  # Less than 60% in last 10 trades
                self.logger.warning(f"ALERT: Recent performance declining ({recent_win_rate*100:.1f}% in last 10 trades)")

# Usage Example
async def example_performance_optimization():
    """Example of how to use the performance optimizer"""
    
    # Initialize optimizer
    optimizer = PerformanceOptimizer("momentum_performance.db")
    
    # Start real-time monitoring
    monitor = RealTimePerformanceMonitor(optimizer)
    
    # Example trade tracking
    # When a trade is entered:
    trade_entry_data = {
        'entry_price': 50000.0,
        'confidence': 85.0,
        'entry_quality_score': 78.5,
        'signal_strength': 'STRONG',
        'breakout_type': 'VOLUME_BREAKOUT',
        'volume_confirmed': True,
        'momentum_confirmed': True,
        'is_high_probability': True,
        'risk_reward_ratio': 3.0,
        'market_phase': 'MARKUP',
        'rsi_at_entry': 58.0,
        'volume_spike_ratio': 2.3
    }
    
    trade_id = await optimizer.track_trade_entry('BTCUSDT', trade_entry_data)
    
    # Simulate trade exit after some time
    await asyncio.sleep(1)  # In reality, this would be hours/days
    await optimizer.track_trade_exit(trade_id, 51500.0, "TAKE_PROFIT")
    
    # Generate performance report
    report = await optimizer.generate_performance_report()
    print(report)
    
    # Get current thresholds
    thresholds = optimizer.get_current_thresholds()
    print(f"Current adaptive thresholds: {thresholds}")

if __name__ == "__main__":
    asyncio.run(example_performance_optimization())