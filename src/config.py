import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

"""
Configuration file for Nebula AI integration - FIXED PRO PLAN VERSION
"""

# =============================================================================
# 🔥 CRITICAL FIXES: PRO PLAN TAAPI CONFIGURATION 
# =============================================================================

TAAPI_SECRET = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJjbHVlIjoiNjg1NDFjNDI4MDZmZjE2NTFlNTY4ZGNhIiwiaWF0IjoxNzUyNDIyMzg4LCJleHAiOjMzMjU2ODg2Mzg4fQ.Q4GOQ6s32PcS3S8zBNTGxJXHtoAt6bveeav8aIegmTU"
TAAPI_API_SECRET = TAAPI_SECRET

# 🚨 CRITICAL FIX: PRO PLAN SETTINGS (NOT FREE PLAN)
TAAPI_FREE_PLAN_MODE = False  # 🔥 FIXED: Changed from True
TAAPI_SINGLE_REQUEST_MODE = False  # 🔥 Enable bulk queries
TAAPI_RATE_LIMIT_DELAY = 1200  # 🔥 FIXED: Pro plan timing (was 65000!)
TAAPI_MAX_REQUESTS_PER_HOUR = 7200  # Pro plan limit
TAAPI_MAX_CONSECUTIVE_ERRORS = 5
TAAPI_CIRCUIT_BREAKER_TIMEOUT = 300000  # 5 minutes
TAAPI_CACHE_EXPIRY = 300000  # 5 minutes

# Enhanced Pro Plan Features
TAAPI_BULK_QUERY_ENABLED = True
TAAPI_DYNAMIC_SYMBOL_DETECTION = True

# Service Health Settings
TAAPI_ENABLE_CIRCUIT_BREAKER = True
TAAPI_ENABLE_CACHING = True
TAAPI_ENABLE_SYMBOL_VALIDATION = True

# Debug Settings
TAAPI_DEBUG_LOGGING = True

# 🇩🇰 Danish Strategy Fallback Settings
TAAPI_USE_FALLBACK_ON_ERROR = True
TAAPI_FALLBACK_CONFIDENCE_PENALTY = 10  # Reduced for pro plan
TAAPI_SMART_FALLBACK_MODE = True

# =============================================================================
# 🇩🇰 DANISH STRATEGY CONFIGURATION - HIGH CONFIDENCE
# =============================================================================

# 🔥 CRITICAL FIX: Danish Strategy Confidence Thresholds
API_MIN_CONFIDENCE = 55.0  # 🔥 DUAL-TIER SYSTEM: Lowered to 55% for Tier 3 (was 60%)
API_OVERRIDE_CONFIDENCE = 80.0  # 🔥 FIXED: Higher override (was 50%)
API_EXCELLENT_THRESHOLD = 85.0  # Excellent entry threshold

# Danish Strategy Core Requirements
DANISH_IGNORE_BEARISH_SIGNALS = True
DANISH_ONLY_BULLISH_ENTRIES = True
DANISH_REQUIRE_VOLUME_CONFIRMATION = True
DANISH_REQUIRE_BREAKOUT_CONFIRMATION = True

# Danish Strategy Thresholds
DANISH_MIN_CONFLUENCE_SCORE = 75
DANISH_MIN_CONFIDENCE_SCORE = 80
DANISH_EXCELLENT_ENTRY_THRESHOLD = 85

# Volume and Momentum Requirements
DANISH_MIN_VOLUME_SPIKE = 1.5
DANISH_MIN_RSI_MOMENTUM = 40
DANISH_MAX_RSI_OVERBOUGHT = 72

# =============================================================================
# 🎯 FIXED POSITION SIZING - CONSISTENT 20% PER POSITION
# =============================================================================

# 🔥 CRITICAL FIX: Consistent Position Sizing
# 🔥 AGGRESSIVE DUAL-TIER POSITION SIZING - 100% EQUITY USAGE
# =============================================================================

# 🚀 AGGRESSIVE DUAL-TIER SYSTEM (Updated from single 20% system)
# Tier 1: 20% per trade (excellent signals) - up to 5 trades = 100% max
# Tier 2: 10% per trade (good signals) - up to 8 trades = 80% max  
# Tier 3: 8% per trade (fair signals) - up to 10 trades = 80% max

# AGGRESSIVE DUAL-TIER POSITION SIZING
DUAL_TIER_ENABLED = True  # Enable aggressive dual-tier system

# TIER 1 - ULTRA-SELECTIVE (Excellent signals)
TIER1_POSITION_SIZE = 0.20      # 20% per trade (AGGRESSIVE)
TIER1_MAX_POSITIONS = 5         # Max 5 positions 
TIER1_MAX_EXPOSURE = 1.00       # 5 × 20% = 100% max exposure

# TIER 2 - MODERATE (Good signals)  
TIER2_POSITION_SIZE = 0.15      # 15% per trade (AGGRESSIVE)
TIER2_MAX_POSITIONS = 8         # Max 8 positions
TIER2_MAX_EXPOSURE = 1.20       # 8 × 15% = 120% max exposure

# TIER 3 - FAIR (Fair signals)
TIER3_POSITION_SIZE = 0.10      # 10% per trade
TIER3_MAX_POSITIONS = 10        # Max 10 positions  
TIER3_MAX_EXPOSURE = 1.00       # 10 × 10% = 100% max exposure

# AGGRESSIVE SYSTEM LIMITS
MAX_TOTAL_EXPOSURE = 1.00       # 100% total exposure (FULL EQUITY)
TOTAL_ALLOCATION_LIMIT = 1.00   # Never exceed 100% of cash

# Legacy compatibility (highest tier for fallback)
MAX_POSITIONS = TIER1_MAX_POSITIONS
POSITION_SIZE_FIXED = TIER1_POSITION_SIZE  # 20% for excellent signals
MIN_POSITION_SIZE_PCT = TIER3_POSITION_SIZE  # 8% minimum  
MAX_POSITION_SIZE_PCT = TIER1_POSITION_SIZE  # 20% maximum

# Quick mode settings - Use Tier 1 settings for quick mode
QUICK_MODE_POSITION_SIZE = TIER1_POSITION_SIZE  # 20% per position
QUICK_MODE_MAX_POSITIONS = TIER1_MAX_POSITIONS  # Max 5 positions
NORMAL_MODE_MAX_POSITIONS = TIER1_MAX_POSITIONS + TIER2_MAX_POSITIONS  # 13 total positions

# Minimum position sizes in USD
MIN_POSITION_SIZE_USD = 20.0  # $20 minimum
QUICK_MODE_MIN_POSITION = 20.0  # Consistent minimums
NORMAL_MODE_MIN_POSITION = 20.0

# =============================================================================
# ENHANCED SIGNAL API CONFIGURATION - PRO PLAN OPTIMIZED
# =============================================================================

SIGNAL_API_URL = os.getenv('SIGNAL_API_URL', 'http://localhost:3001/api')
SIGNAL_API_KEY = os.getenv('SIGNAL_API_KEY', '1234')
ENABLE_ENHANCED_API = True

# Pro plan optimized intervals
API_REQUEST_TIMEOUT = 15  # Reduced timeout for pro plan
API_MIN_INTERVAL = 0.1   # 0.1 seconds for Taapi Pro plan - no rate limiting needed
API_CACHE_DURATION = 10  # 10 seconds cache - TEMPORARY for fresh API calls

# API feature usage
USE_API_POSITION_SIZING = True
USE_API_STOP_LOSS = True
USE_API_TAKE_PROFIT = True
API_FALLBACK_ENABLED = True

# Signal weights
API_SIGNAL_WEIGHT = 0.70  # 70% weight to API
INTERNAL_SIGNAL_WEIGHT = 0.30  # 30% internal

# =============================================================================
# RISK MANAGEMENT - DANISH STRATEGY OPTIMIZED
# =============================================================================

# 🇩🇰 AGGRESSIVE Danish Strategy Risk Settings (Updated for 60% momentum gains)
DANISH_MAX_POSITION_SIZE = 0.20  # 20% maximum per position (Tier 1)
DANISH_STOP_LOSS_PERCENTAGE = 0.03  # 3% stop loss
DANISH_TAKE_PROFIT_1 = 0.15  # 15% first target (AGGRESSIVE - was 5%)
DANISH_TAKE_PROFIT_2 = 0.30  # 30% second target (AGGRESSIVE - was 8%)
DANISH_TAKE_PROFIT_3 = 0.60  # 60% final target (AGGRESSIVE - was 12%)

# Stop loss settings
MOMENTUM_STOP_LOSS = -2.0   # 2% stop for momentum
QUICK_STOP_LOSS = -2.5      # 2.5% for regular trades
MAX_STOP_LOSS = -3.0        # Never exceed 3% loss

# Daily targets - Conservative
TARGET_DAILY_ROI_MIN = 0.02     # 2% minimum daily target
TARGET_DAILY_ROI_OPTIMAL = 0.03  # 3% optimal daily target
TARGET_DAILY_ROI_MAX = 0.05      # 5% maximum daily target

# Drawdown protection
DRAWDOWN_THRESHOLDS = {
    'warning': 0.05,      # 5%
    'reduce_risk': 0.08,  # 8%
    'high_alert': 0.10,   # 10%
    'emergency': 0.15     # 15%
}

# Conservative thresholds - DISABLED for Danish strategy
CONSERVATIVE_THRESHOLDS_ENABLED = False

# =============================================================================
# TRADING PARAMETERS - DANISH STRATEGY FOCUSED
# =============================================================================

# Trading modes
TEST_MODE = True  # Set to False for live trading
MOMENTUM_TRADE_ENABLED = True
QUICK_PROFIT_MODE = True
ENABLE_OPPORTUNITY_SCANNER = True

# Pair Selection - Focus on high-quality pairs
PRIORITY_CRYPTO_LIST = [
    'BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT', 'XRPUSDT',
    'ADAUSDT', 'DOGEUSDT', 'MATICUSDT', 'DOTUSDT', 'AVAXUSDT',
    'LINKUSDT', 'UNIUSDT', 'ATOMUSDT', 'LTCUSDT', 'NEARUSDT',
    'INJUSDT', 'SUIUSDT', 'SEIUSDT', 'ARBUSDT', 'OPUSDT',
    'ORDIUSDT', 'MKRUSDT', 'BERAUSDT', 'KAITOUSDT', 'NXPCUSDT',
    'CRVUSDT', 'ZKUSDT', 'WBETHUSDT'
]

# Volume and quality filters for Danish strategy
MIN_VOLUME_USD = 1000000    # $1M daily volume minimum
MIN_VOLATILITY_PCT = 0.5    # 0.5% daily volatility minimum
MAX_SPREAD_PCT = 0.25       # Max 0.25% spread

# =============================================================================
# PROFIT TARGETS - DANISH STRATEGY OPTIMIZED
# =============================================================================

# Danish momentum targets
MOMENTUM_TAKE_PROFIT = [
    {"minutes": 5, "profit_pct": 0.5},   # 0.5% in 5 min
    {"minutes": 15, "profit_pct": 0.4},  # 0.4% in 15 min
    {"minutes": 30, "profit_pct": 0.3},  # 0.3% in 30 min
    {"minutes": 60, "profit_pct": 0.25}, # 0.25% in 1 hour
]

# Regular position targets
REGULAR_TAKE_PROFIT = [
    {"minutes": 30, "profit_pct": 0.8},  # 0.8% in 30 min
    {"minutes": 60, "profit_pct": 0.6},  # 0.6% in 1 hour
    {"minutes": 120, "profit_pct": 0.5}, # 0.5% in 2 hours
    {"minutes": 180, "profit_pct": 0.4}, # 0.4% in 3 hours
]

# Partial profit taking
ENABLE_PARTIAL_PROFITS = True
PARTIAL_PROFIT_LEVELS = [
    {"profit_pct": 0.2, "sell_pct": 0.33},   # Sell 33% at 0.2%
    {"profit_pct": 0.3, "sell_pct": 0.33},   # Sell 33% at 0.3%
    {"profit_pct": 0.5, "sell_pct": 0.34},   # Sell 34% at 0.5%
]

# =============================================================================
# TRADING MODE - SPOT ONLY (NO LEVERAGE)
# =============================================================================

TRADING_MODE = "SPOT_ONLY"  # Only spot trading
ENABLE_MARGIN_TRADING = False
ENABLE_FUTURES_TRADING = False
ENABLE_LEVERAGE = False
MAX_LEVERAGE = 1.0  # No leverage

# =============================================================================
# MARKET ANALYSIS SETTINGS
# =============================================================================

# Scanning frequency - Optimized for Danish strategy
OPPORTUNITY_SCAN_INTERVAL = 30  # Every 30 seconds
MARKET_UPDATE_INTERVAL = 60     # Market state every minute
POSITION_MONITOR_INTERVAL = 30  # Monitor positions every 30 seconds

# Timeframes for analysis
ANALYSIS_TIMEFRAMES = ['5m', '15m', '1h']
DEFAULT_API_TIMEFRAME = '15m'

# Signal quality filters for Danish strategy
MIN_API_TECHNICAL_SCORE = 60   # Higher threshold for Danish strategy
MIN_API_MOMENTUM_SCORE = 65     # Higher momentum requirement
MIN_API_ONCHAIN_SCORE = 55      # Higher on-chain requirement



# Time horizons
RESPECT_API_TIME_HORIZONS = True
API_MAX_HOLD_TIME_HOURS = 24    # Max 24 hours
API_MIN_HOLD_TIME_MINUTES = 3   # Min 3 minutes
# =============================================================================
# ADVANCED FEATURES - DANISH STRATEGY
# =============================================================================

# Trailing stops
ENABLE_TRAILING_STOPS = True
TRAILING_ACTIVATION_THRESHOLD = 0.02  # 2% activation
TRAILING_STOP_PERCENTAGE = 0.015      # 1.5% trail

# Breakeven stops
ENABLE_BREAKEVEN_STOP = True
BREAKEVEN_TRIGGER_PCT = 0.5  # Move to breakeven at 0.5%

# Dynamic sizing based on signal strength
ENABLE_DYNAMIC_SIZING = True
SIGNAL_STRENGTH_THRESHOLDS = {
    'very_strong': 0.85,   # > 85% confidence
    'strong': 0.75,        # > 75% confidence  
    'moderate': 0.70,      # > 70% confidence (Danish minimum)
}

POSITION_SIZE_MULTIPLIERS = {
    'very_strong': 1.0,   # Full 20% position
    'strong': 1.0,        # Full 20% position
    'moderate': 0.8,      # 16% position for moderate signals
}

# =============================================================================
# API KEYS AND CREDENTIALS
# =============================================================================

BINANCE_API_KEY = os.getenv('BINANCE_API_KEY', '5kKZ7ENLfCdW3q5NKR9ZREysG8iY6Cx6SHd0qNNMf5BYUAUkFYR6KCBSyqZlKl9O')
BINANCE_API_SECRET = os.getenv('BINANCE_API_SECRET', '9j9ArHrr6f2Pn2Slwdmk6qiZgWdMFa4ELxtDeQpG02jZTw8eYQwqyr2taUuGpPdA')

# Disable unused services for clean operation
ENABLE_NEBULA = False
ENABLE_COINGECKO = False
NEBULA_SECRET_KEY = None
COINGECKO_API_KEY = None

# =============================================================================
# SIGNAL ANALYSIS WEIGHTS
# =============================================================================

TECHNICAL_WEIGHT = 0.40
ONCHAIN_WEIGHT = 0.35
ORDERBOOK_WEIGHT = 0.25

# =============================================================================
# LOGGING AND MONITORING
# =============================================================================

LOG_LEVEL = 'INFO'
ENABLE_TRADE_LOGGING = True
ENABLE_PERFORMANCE_MONITORING = True
DEBUG_MODE = False

# Database and file paths
DB_PATH = os.getenv('DB_PATH', 'trading_bot.db')
LOG_DIR = 'logs'
PERFORMANCE_DB = 'performance.db'
TRADE_LOG_FILE = 'trades.log'

# =============================================================================
# RATE LIMITING - PRO PLAN OPTIMIZED
# =============================================================================

BINANCE_API_CALLS_PER_MINUTE = 100  # Pro plan can handle more
API_CALL_DELAY = 2.0  # 2 seconds between calls (increased from 1.0)
TRADING_CYCLE_INTERVAL = 120  # 2 minutes between cycles
MAX_CONCURRENT_ANALYSIS = 5  # Increased back to 5 for better API response handling
API_REQUEST_TIMEOUT = 20.0  # 20 second timeout (increased from 10.0 to give API more time)

# =============================================================================
# VALIDATION FUNCTIONS
# =============================================================================

def validate_config():
    """Validate configuration settings"""
    errors = []
    
    # Check TAAPI configuration
    if TAAPI_FREE_PLAN_MODE:
        errors.append("CRITICAL: TAAPI_FREE_PLAN_MODE is True - should be False for pro plan")
    
    if TAAPI_RATE_LIMIT_DELAY > 60000:
        errors.append(f"CRITICAL: TAAPI_RATE_LIMIT_DELAY is {TAAPI_RATE_LIMIT_DELAY}ms - should be 1200ms for pro plan")
    
    # Check Danish strategy configuration
    if API_MIN_CONFIDENCE < 55:
        errors.append(f"CRITICAL: API_MIN_CONFIDENCE is {API_MIN_CONFIDENCE}% - Dual-Tier System requires 55%+")
    
    # Check dual-tier position sizing (updated validation)
    if DUAL_TIER_ENABLED:
        # Validate dual-tier system
        if TIER1_POSITION_SIZE > MAX_TOTAL_EXPOSURE:
            errors.append(f"CRITICAL: Tier 1 position size ({TIER1_POSITION_SIZE*100}%) exceeds max exposure ({MAX_TOTAL_EXPOSURE*100}%)")
        
        if TIER2_POSITION_SIZE > MAX_TOTAL_EXPOSURE:
            errors.append(f"CRITICAL: Tier 2 position size ({TIER2_POSITION_SIZE*100}%) exceeds max exposure ({MAX_TOTAL_EXPOSURE*100}%)")
        
        if TIER3_POSITION_SIZE > MAX_TOTAL_EXPOSURE:
            errors.append(f"CRITICAL: Tier 3 position size ({TIER3_POSITION_SIZE*100}%) exceeds max exposure ({MAX_TOTAL_EXPOSURE*100}%)")
        
        # Check that tier sizes are in descending order
        if not (TIER1_POSITION_SIZE >= TIER2_POSITION_SIZE >= TIER3_POSITION_SIZE):
            errors.append(f"CRITICAL: Tier position sizes should be descending: T1={TIER1_POSITION_SIZE*100}%, T2={TIER2_POSITION_SIZE*100}%, T3={TIER3_POSITION_SIZE*100}%")
        
        # Validate max exposure scenarios
        max_tier1_exposure = TIER1_MAX_POSITIONS * TIER1_POSITION_SIZE
        if max_tier1_exposure > MAX_TOTAL_EXPOSURE:
            errors.append(f"CRITICAL: Max Tier 1 exposure ({max_tier1_exposure*100}%) exceeds total limit ({MAX_TOTAL_EXPOSURE*100}%)")
        
        print(f"✅ DUAL-TIER VALIDATION: T1={TIER1_POSITION_SIZE*100}%, T2={TIER2_POSITION_SIZE*100}%, T3={TIER3_POSITION_SIZE*100}% - Max Exposure: {MAX_TOTAL_EXPOSURE*100}%")
    else:
        # Legacy single-tier validation
        if MIN_POSITION_SIZE_PCT != MAX_POSITION_SIZE_PCT:
            errors.append(f"Position sizing inconsistent: MIN={MIN_POSITION_SIZE_PCT}, MAX={MAX_POSITION_SIZE_PCT}")
        
        if MAX_POSITIONS * MAX_POSITION_SIZE_PCT > 1.0:
            errors.append(f"CRITICAL: {MAX_POSITIONS} positions × {MAX_POSITION_SIZE_PCT*100}% = {MAX_POSITIONS * MAX_POSITION_SIZE_PCT * 100}% > 100%")
    
    # Check API credentials
    if not BINANCE_API_KEY or len(BINANCE_API_KEY) < 10:
        errors.append("CRITICAL: BINANCE_API_KEY missing or invalid")
    
    if not BINANCE_API_SECRET or len(BINANCE_API_SECRET) < 10:
        errors.append("CRITICAL: BINANCE_API_SECRET missing or invalid")
    
    if errors:
        print("\nCONFIGURATION ERRORS:")
        for error in errors:
            print(f"   {error}")
        return False
    else:
        print("\n CONFIGURATION VALIDATION PASSED")
        return True

def print_config_summary():
    """Print configuration summary"""
    print("\n" + "="*80)
    if DUAL_TIER_ENABLED:
        print("🚀 AGGRESSIVE DUAL-TIER DANISH MOMENTUM STRATEGY CONFIGURATION")
    else:
        print("🇩🇰 DANISH MOMENTUM STRATEGY CONFIGURATION SUMMARY")
    print("="*80)
    print(f"TAAPI Free Plan Mode: {TAAPI_FREE_PLAN_MODE} ({'CORRECT' if not TAAPI_FREE_PLAN_MODE else ' WRONG'})")
    print(f"TAAPI Rate Limit: {TAAPI_RATE_LIMIT_DELAY}ms ({'PRO PLAN' if TAAPI_RATE_LIMIT_DELAY < 10000 else 'FREE PLAN'})")
    print(f"API Min Confidence: {API_MIN_CONFIDENCE}% ({'DUAL-TIER SYSTEM' if API_MIN_CONFIDENCE >= 55 else 'TOO LOW'})")
    
    if DUAL_TIER_ENABLED:
        print(f"🎯 AGGRESSIVE DUAL-TIER POSITION SIZING:")
        print(f"  • Tier 1 (Excellent): {TIER1_POSITION_SIZE*100:.0f}% per trade (max {TIER1_MAX_POSITIONS} positions = {TIER1_MAX_EXPOSURE*100:.0f}% exposure)")
        print(f"  • Tier 2 (Good): {TIER2_POSITION_SIZE*100:.0f}% per trade (max {TIER2_MAX_POSITIONS} positions = {TIER2_MAX_EXPOSURE*100:.0f}% exposure)")
        print(f"  • Tier 3 (Fair): {TIER3_POSITION_SIZE*100:.0f}% per trade (max {TIER3_MAX_POSITIONS} positions = {TIER3_MAX_EXPOSURE*100:.0f}% exposure)")
        print(f"  • Max Total Exposure: {MAX_TOTAL_EXPOSURE*100:.0f}% (AGGRESSIVE 100% EQUITY USAGE)")
        print(f"🎯 AGGRESSIVE TAKE PROFITS:")
        print(f"  • TP1: {DANISH_TAKE_PROFIT_1*100:.0f}% | TP2: {DANISH_TAKE_PROFIT_2*100:.0f}% | TP3: {DANISH_TAKE_PROFIT_3*100:.0f}%")
    else:
        print(f"Position Size: {MAX_POSITION_SIZE_PCT*100}% per position ({'CORRECT' if MAX_POSITION_SIZE_PCT == 0.20 else 'INCONSISTENT'})")
        print(f"Max Positions: {MAX_POSITIONS}")
        print(f"Total Allocation: {MAX_POSITIONS * MAX_POSITION_SIZE_PCT * 100}%")
    
    print(f"Danish Strategy Features:")
    print(f"  • Only Bullish Entries: {DANISH_ONLY_BULLISH_ENTRIES}")
    print(f"  • Volume Confirmation: {DANISH_REQUIRE_VOLUME_CONFIRMATION}")
    print(f"  • Breakout Confirmation: {DANISH_REQUIRE_BREAKOUT_CONFIRMATION}")
    print(f"  • Ignore Bearish Signals: {DANISH_IGNORE_BEARISH_SIGNALS}")
    print(f"Trading Mode: {TRADING_MODE}")
    print(f"Test Mode: {TEST_MODE}")
    print("="*80)

# Validate on import
if __name__ == "__main__":
    validate_config()
    print_config_summary()
else:
    # Auto-validate in production
    if not os.getenv('SKIP_CONFIG_VALIDATION'):
        if not validate_config():
            raise ValueError("Configuration validation failed - check errors above")

# =============================================================================
# FINAL CONSISTENCY CHECKS - DUAL-TIER SYSTEM
# =============================================================================

if DUAL_TIER_ENABLED:
    # Dual-tier system validation
    assert TIER1_POSITION_SIZE == 0.20, f"Tier 1 position size must be exactly 20%, got {TIER1_POSITION_SIZE*100}%"
    assert TIER2_POSITION_SIZE == 0.15, f"Tier 2 position size must be exactly 15%, got {TIER2_POSITION_SIZE*100}%"
    assert TIER3_POSITION_SIZE == 0.10, f"Tier 3 position size must be exactly 10%, got {TIER3_POSITION_SIZE*100}%"
    assert MAX_TOTAL_EXPOSURE == 1.00, f"Max total exposure must be 100% for aggressive strategy, got {MAX_TOTAL_EXPOSURE*100}%"
    assert TIER1_MAX_POSITIONS == 5, f"Tier 1 must allow 5 positions for 100% exposure, got {TIER1_MAX_POSITIONS}"
    
    print("CONFIGURATION LOADED: AGGRESSIVE DUAL-TIER Danish Strategy + 100% Equity Usage + Pro Plan")
else:
    # Legacy single-tier validation
    assert MIN_POSITION_SIZE_PCT == 0.20, "Position size must be exactly 20%"
    assert MAX_POSITION_SIZE_PCT == 0.20, "Position size must be exactly 20%"
    assert POSITION_SIZE_FIXED == 0.20, "Position size must be exactly 20%"
    assert MAX_POSITIONS == 5, "Must have exactly 5 positions for 20% each"
    
    print("CONFIGURATION LOADED: Danish Strategy + 20% Position Sizing + Pro Plan")

# Ensure pro plan configuration (applies to both systems)
assert not TAAPI_FREE_PLAN_MODE, "Must use pro plan mode"
assert TAAPI_RATE_LIMIT_DELAY == 1200, "Must use pro plan rate limiting"
assert API_MIN_CONFIDENCE >= 55, "Must use Dual-Tier System confidence thresholds"