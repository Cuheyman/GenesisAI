# Enhanced Momentum Strategy Configuration
# Optimized for 75-90% win rate through selective, high-quality entries

import os
from typing import Dict, List, Any
from dataclasses import dataclass

@dataclass
class MomentumStrategyConfig:
    """Configuration for momentum-based bullish strategy"""
    
    # TAAPI Configuration
    TAAPI_API_SECRET = os.getenv('eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJjbHVlIjoiNjg1NDFjNDI4MDZmZjE2NTFlNTY4ZGNhIiwiaWF0IjoxNzUyNDIyMzg4LCJleHAiOjMzMjU2ODg2Mzg4fQ.Q4GOQ6s32PcS3S8zBNTGxJXHtoAt6bveeav8aIegmTU', 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJjbHVlIjoiNjg1NDFjNDI4MDZmZjE2NTFlNTY4ZGNhIiwiaWF0IjoxNzUyNDIyMzg4LCJleHAiOjMzMjU2ODg2Mzg4fQ.Q4GOQ6s32PcS3S8zBNTGxJXHtoAt6bveeav8aIegmTU')
    TAAPI_RATE_LIMIT_DELAY = 1.2  # Seconds between requests for pro plan
    TAAPI_MAX_RETRIES = 3
    TAAPI_TIMEOUT = 15  # Seconds
    
    # Strategy Core Settings (Your Danish Strategy Requirements)
    IGNORE_BEARISH_SIGNALS = True  # Fokuserer på at gå kun ind når momentum og indikatorer viser styrke
    ONLY_BULLISH_ENTRIES = True    # Går kun med trends, ignorerer bearish signaler
    REQUIRE_VOLUME_CONFIRMATION = True  # Reagerer først når der er bekræftet volumen + prisbevægelse
    REQUIRE_BREAKOUT_CONFIRMATION = True  # Vælger udelukkende at handle på udvælgte, stærke bullish setups
    
    # High Win Rate Thresholds (75-90% target)
    MIN_CONFLUENCE_SCORE = 65  # Minimum confluence percentage for entry
    MIN_CONFIDENCE_SCORE = 70  # Minimum confidence for high-probability entries  
    EXCELLENT_ENTRY_THRESHOLD = 80  # Threshold for "excellent" quality entries
    
    # Momentum Detection Parameters
    MOMENTUM_THRESHOLDS = {
        # RSI Configuration (conservative for high win rate)
        'rsi_oversold_entry': 38,      # More conservative than standard 30
        'rsi_momentum_sweet_spot': (40, 65),  # Ideal RSI range for momentum entries
        'rsi_overbought_avoid': 72,    # Avoid late entries
        'rsi_divergence_weight': 2.0,  # Extra weight for bullish divergence
        
        # MACD Configuration
        'macd_histogram_min': 0.001,   # Must be positive for bullish momentum
        'macd_crossover_weight': 2.5,  # High weight for fresh crossovers
        'macd_acceleration_bonus': 1.0, # Bonus for accelerating histogram
        
        # Volume Analysis (critical for your strategy)
        'volume_spike_min': 1.8,       # 80% volume increase minimum
        'volume_trend_periods': 20,     # Periods to analyze volume trend
        'money_flow_threshold': 55,     # MFI threshold for bullish money flow
        'obv_trend_weight': 1.5,       # On Balance Volume trend importance
        
        # Price Action & Breakouts
        'price_momentum_min': 0.8,     # 0.8% price increase in timeframe
        'breakout_confirmation': 0.5,  # 0.5% above resistance for confirmation
        'consolidation_break_min': 1.2, # 1.2% for consolidation breakouts
        'squeeze_momentum_threshold': 0.1, # TTM Squeeze momentum threshold
        
        # Multi-Timeframe Alignment
        'mtf_alignment_weight': 2.0,   # Weight for multi-timeframe agreement
        'trend_alignment_bonus': 1.5,  # Bonus when all timeframes align
        
        # Technical Confluence Requirements
        'min_indicators_aligned': 4,   # Minimum indicators supporting signal
        'max_indicators_for_100': 8,   # Max indicators for 100% confluence
        'ema_alignment_bonus': 1.0,    # Bonus for EMA alignment (20>50>200)
        'vwap_respect_weight': 1.2,    # Weight for respecting VWAP levels
    }
    
    # Market Phase Strategy Settings
    MARKET_PHASE_CONFIG = {
        'accumulation': {
            'preferred_timeframes': ['1h', '4h'],
            'min_confluence': 60,
            'volume_importance': 2.0,
            'patience_multiplier': 1.5,  # Wait for better setups in accumulation
        },
        'markup': {
            'preferred_timeframes': ['15m', '1h'],
            'min_confluence': 70,
            'momentum_importance': 2.5,
            'trend_following': True,
        },
        'distribution': {
            'preferred_timeframes': ['4h', '1d'],
            'min_confluence': 80,  # Very selective in distribution
            'avoid_late_entries': True,
            'defensive_mode': True,
        },
        'consolidation': {
            'preferred_timeframes': ['1h', '4h'],
            'min_confluence': 65,
            'breakout_focus': True,
            'patience_multiplier': 2.0,  # Very patient in consolidation
        }
    }
    
    # Entry Quality Filters (for 75-90% win rate)
    ENTRY_QUALITY_FILTERS = {
        'excellent_entry': {
            'min_confluence': 80,
            'min_indicators': 6,
            'required_volume_spike': True,
            'required_momentum_alignment': True,
            'max_risk_reward': 4.0,
            'avoid_late_cycle': True,
        },
        'good_entry': {
            'min_confluence': 70,
            'min_indicators': 5,
            'required_volume_confirmation': True,
            'required_trend_alignment': True,
            'max_risk_reward': 3.0,
        },
        'fair_entry': {
            'min_confluence': 60,
            'min_indicators': 4,
            'required_basic_momentum': True,
            'max_risk_reward': 2.5,
        },
        'avoid_entry': {
            'max_confluence': 50,
            'bearish_divergence': True,
            'late_cycle_entry': True,
            'low_volume': True,
            'overbought_rsi': True,
        }
    }
    
    # Risk Management for High Win Rate Strategy
    RISK_MANAGEMENT = {
        'position_sizing': {
            'excellent_entry': 0.08,  # 8% position size for best setups
            'good_entry': 0.06,       # 6% for good setups
            'fair_entry': 0.04,       # 4% for fair setups
            'max_total_exposure': 0.25, # Max 25% total crypto exposure
        },
        'stop_loss': {
            'atr_multiplier': 1.5,    # ATR-based stop loss
            'min_stop_distance': 0.02, # Minimum 2% stop loss
            'max_stop_distance': 0.08, # Maximum 8% stop loss
            'trailing_activation': 0.03, # Start trailing at 3% profit
        },
        'take_profit': {
            'partial_profit_1': 0.05,  # Take 30% at 5% profit
            'partial_profit_2': 0.10,  # Take 40% at 10% profit
            'final_target': 0.15,      # Final 30% at 15% profit
            'trend_continuation_hold': True, # Hold partial if trend continues
        }
    }
    
    # Timeframe Configuration
    TIMEFRAME_CONFIG = {
        'primary': '1h',           # Main analysis timeframe
        'confirmation': '4h',      # Trend confirmation timeframe  
        'entry': '15m',           # Entry timing timeframe
        'trend': '1d',            # Long-term trend analysis
        
        'analysis_order': ['1d', '4h', '1h', '15m'],  # Top-down analysis
        'min_timeframes_aligned': 3,  # Minimum timeframes that must agree
    }
    
    # Symbol Filtering (focus on high-quality pairs)
    SYMBOL_FILTERS = {
        'min_volume_24h': 10000000,   # Minimum $10M 24h volume
        'min_market_cap': 100000000,  # Minimum $100M market cap
        'avoid_new_listings': 30,     # Avoid coins listed less than 30 days
        'preferred_pairs': [
            'BTCUSDT', 'ETHUSDT', 'ADAUSDT', 'SOLUSDT', 'DOTUSDT',
            'LINKUSDT', 'MATICUSDT', 'AVAXUSDT', 'ATOMUSDT', 'ALGOUSDT'
        ],
        'blacklisted_pairs': [
            # Add any pairs you want to avoid
        ]
    }
    
    # Performance Tracking for 75-90% Win Rate Goal
    PERFORMANCE_TRACKING = {
        'target_win_rate': 0.80,      # 80% target win rate
        'min_sample_size': 20,        # Minimum trades for win rate calculation
        'performance_review_frequency': 10,  # Review every 10 trades
        'adjustment_threshold': 0.75,  # Adjust strategy if win rate below 75%
        
        'metrics_to_track': [
            'win_rate',
            'average_rrr',         # Risk-reward ratio
            'average_hold_time',
            'confluence_accuracy', # How well confluence predicts success
            'timeframe_accuracy',  # Which timeframes are most predictive
            'indicator_performance', # Which indicators perform best
        ]
    }
    
    # Alert and Notification Settings
    ALERT_CONFIG = {
        'excellent_entry_alert': True,    # Alert for excellent entries
        'confluence_threshold_alert': 85, # Alert when confluence > 85%
        'volume_spike_alert': True,       # Alert for significant volume spikes
        'breakout_confirmation_alert': True, # Alert for confirmed breakouts
        
        'notification_methods': ['console', 'file'],  # Add 'webhook', 'email' as needed
        'alert_cooldown': 300,  # 5 minutes between alerts for same pair
    }
    
    # Danish Strategy Specific Settings (En strategi der helt undgår at shorte eller handle på bearish signaler)
    DANISH_STRATEGY_SPECIFIC = {
        'momentum_only': True,           # Går kun ind når momentum og indikatorer viser styrke
        'volume_spike_requirement': True, # Reagerer først når der er volumen + prisbevægelse  
        'breakout_focus': True,          # Fokuserer på at gå ind når prisen bryder opad fra en modstand eller konsolidering
        'ignore_bearish_completely': True, # Vælger udelukkende at handle på stærke bullish setups
        'patience_for_quality': True,    # Går kun lang (køber), når momentum og indikatorer viser styrke
        
        # Translation of your strategy requirements:
        'wait_for_confirmation': True,   # Fokuserer på at gå ind ved bekræftede opadgående bevægelser
        'volume_confirmation_required': True, # fx gennem breakout eller volume spikes
        'momentum_confirmation_required': True, # gennem breakout eller volume spikes
    }

# Global configuration instance
momentum_config = MomentumStrategyConfig()

# Validation function
def validate_config() -> Dict[str, Any]:
    """Validate configuration settings and return status"""
    validation_results = {
        'valid': True,
        'warnings': [],
        'errors': [],
        'recommendations': []
    }
    
    # Check TAAPI configuration
    if not momentum_config.TAAPI_API_SECRET or momentum_config.TAAPI_API_SECRET == 'your_secret_here':
        validation_results['errors'].append("TAAPI_API_SECRET not configured")
        validation_results['valid'] = False
    
    # Check threshold consistency
    if momentum_config.MIN_CONFLUENCE_SCORE > momentum_config.EXCELLENT_ENTRY_THRESHOLD:
        validation_results['warnings'].append("MIN_CONFLUENCE_SCORE higher than EXCELLENT_ENTRY_THRESHOLD")
    
    # Check risk management consistency
    total_partial_profits = (
        momentum_config.RISK_MANAGEMENT['take_profit']['partial_profit_1'] +
        momentum_config.RISK_MANAGEMENT['take_profit']['partial_profit_2']
    )
    if total_partial_profits >= momentum_config.RISK_MANAGEMENT['take_profit']['final_target']:
        validation_results['warnings'].append("Partial profit levels may be too close to final target")
    
    # Recommendations for high win rate
    if momentum_config.MIN_CONFLUENCE_SCORE < 70:
        validation_results['recommendations'].append("Consider increasing MIN_CONFLUENCE_SCORE to 70+ for higher win rate")
    
    if momentum_config.MOMENTUM_THRESHOLDS['volume_spike_min'] < 1.5:
        validation_results['recommendations'].append("Consider increasing volume_spike_min for better signal quality")
    
    return validation_results

# Configuration helper functions
def get_timeframe_config(market_phase: str = 'neutral') -> Dict[str, Any]:
    """Get timeframe configuration based on market phase"""
    if market_phase.lower() in momentum_config.MARKET_PHASE_CONFIG:
        phase_config = momentum_config.MARKET_PHASE_CONFIG[market_phase.lower()]
        return {
            'primary': momentum_config.TIMEFRAME_CONFIG['primary'],
            'preferred': phase_config['preferred_timeframes'],
            'min_confluence': phase_config['min_confluence'],
            'analysis_order': momentum_config.TIMEFRAME_CONFIG['analysis_order']
        }
    
    return momentum_config.TIMEFRAME_CONFIG

def get_entry_quality_requirements(quality_level: str) -> Dict[str, Any]:
    """Get entry requirements for specified quality level"""
    return momentum_config.ENTRY_QUALITY_FILTERS.get(quality_level, {})

def get_position_size(entry_quality: str, account_balance: float) -> float:
    """Calculate position size based on entry quality and account balance"""
    size_pct = momentum_config.RISK_MANAGEMENT['position_sizing'].get(entry_quality, 0.04)
    return account_balance * size_pct

def should_take_trade(confluence_score: float, entry_quality: str, market_phase: str = 'neutral') -> bool:
    """Determine if trade should be taken based on configuration"""
    
    # Check minimum confluence
    if confluence_score < momentum_config.MIN_CONFLUENCE_SCORE:
        return False
    
    # Check quality-specific requirements
    quality_reqs = get_entry_quality_requirements(entry_quality)
    if quality_reqs and confluence_score < quality_reqs.get('min_confluence', 0):
        return False
    
    # Check market phase specific requirements
    if market_phase.lower() in momentum_config.MARKET_PHASE_CONFIG:
        phase_config = momentum_config.MARKET_PHASE_CONFIG[market_phase.lower()]
        if confluence_score < phase_config['min_confluence']:
            return False
    
    # All checks passed
    return True

# Export key configurations for easy import
__all__ = [
    'momentum_config',
    'MomentumStrategyConfig', 
    'validate_config',
    'get_timeframe_config',
    'get_entry_quality_requirements',
    'get_position_size',
    'should_take_trade'
]