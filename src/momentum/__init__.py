"""
Momentum Trading System Package
===============================
High-probability momentum trading components for the Genesis AI Bot
"""

# Package information
__version__ = "1.0.0"
__author__ = "Genesis AI Team"
__description__ = "Momentum trading system with TAAPI integration"

# Try to import core components
try:
    from momentum.enhanced_momentum_taapi import EnhancedMomentumTaapiClient, MomentumSignal
    from momentum.momentum_bot_integration import MomentumTradingOrchestrator
    from momentum.momentum_strategy_config import momentum_config
    from momentum.momentum_performance_optimizer import PerformanceOptimizer
    
    # Try to import optional components
    try:
        from momentum.high_winrate_entry_filter import HighWinRateEntryFilter, EntrySignalStrength
        HIGH_WINRATE_FILTER_AVAILABLE = True
    except ImportError:
        HIGH_WINRATE_FILTER_AVAILABLE = False
        print("High win-rate entry filter not available")
    
    MOMENTUM_SYSTEM_AVAILABLE = True
    
except ImportError as e:
    print(f"Warning: Some momentum components not available: {e}")
    MOMENTUM_SYSTEM_AVAILABLE = False
    HIGH_WINRATE_FILTER_AVAILABLE = False

# Export main components
__all__ = [
    'EnhancedMomentumTaapiClient',
    'MomentumTradingOrchestrator', 
    'momentum_config',
    'PerformanceOptimizer',
    'MomentumSignal',
    'MOMENTUM_SYSTEM_AVAILABLE',
    'HIGH_WINRATE_FILTER_AVAILABLE'
]

# Add conditional exports if available
if HIGH_WINRATE_FILTER_AVAILABLE:
    __all__.extend(['HighWinRateEntryFilter', 'EntrySignalStrength'])