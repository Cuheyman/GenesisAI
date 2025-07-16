import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


"""
Configuration file for Nebula AI integration
Add these settings to your existing config.py
"""



TAAPI_SECRET="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJjbHVlIjoiNjg1NDFjNDI4MDZmZjE2NTFlNTY4ZGNhIiwiaWF0IjoxNzUyNDIyMzg4LCJleHAiOjMzMjU2ODg2Mzg4fQ.Q4GOQ6s32PcS3S8zBNTGxJXHtoAt6bveeav8aIegmTU"
TAAPI_API_SECRET = TAAPI_SECRET

# 🚨 FREE PLAN OPTIMIZED SETTINGS
TAAPI_RATE_LIMIT_DELAY=65000
TAAPI_MAX_CONSECUTIVE_ERRORS=8
TAAPI_CIRCUIT_BREAKER_TIMEOUT=600000
TAAPI_CACHE_EXPIRY=600000

# Free Plan Specific Settings
TAAPI_FREE_PLAN_MODE=True
TAAPI_SINGLE_REQUEST_MODE=True
TAAPI_MAX_REQUESTS_PER_HOUR=60

# Service Health Settings
TAAPI_ENABLE_CIRCUIT_BREAKER=True
TAAPI_ENABLE_CACHING=True
TAAPI_ENABLE_SYMBOL_VALIDATION=True

# Debug Settings
TAAPI_DEBUG_LOGGING=False

# Fallback Settings
TAAPI_USE_FALLBACK_ON_ERROR=True
TAAPI_FALLBACK_CONFIDENCE_PENALTY=20
TAAPI_SMART_FALLBACK_MODE=True




# Optimized configuration for 70-80% profitability in crypto markets

# =============================================================================
# ENHANCED SIGNAL API CONFIGURATION
# =============================================================================

# API Settings - Increased frequency for volatile crypto markets
SIGNAL_API_URL = os.getenv('SIGNAL_API_URL', 'http://localhost:3001/api')
SIGNAL_API_KEY = os.getenv('SIGNAL_API_KEY', '1234')
ENABLE_ENHANCED_API = True  # Re-enable API with correct endpoint

# Reduced intervals for more opportunities
API_REQUEST_TIMEOUT = 20  # 20 seconds max (increased to reduce timeouts)
API_MIN_INTERVAL = 1.0   # 1 second = 3600 requests/hour (ULTRA AGGRESSIVE like successful bot)
API_CACHE_DURATION = 10  # 10 seconds cache (reduced from 30 for maximum freshness)

# Lower confidence for more opportunities like successful bot
API_MIN_CONFIDENCE = 15.0  # Aggressive threshold (was 20%) - matches successful bots
API_OVERRIDE_CONFIDENCE = 50.0  # Lowered for more trades (was 70%)

# API feature usage (optimized)
USE_API_POSITION_SIZING = True
USE_API_STOP_LOSS = True
USE_API_TAKE_PROFIT = True
API_FALLBACK_ENABLED = True

# =============================================================================
# TRADING PARAMETERS - AGGRESSIVE FOR CRYPTO
# =============================================================================

# Enable all trading modes
TEST_MODE = True  # Set to True for testing first
MOMENTUM_TRADE_ENABLED = True
QUICK_PROFIT_MODE = True
ENABLE_OPPORTUNITY_SCANNER = True

# Position Management (AGGRESSIVE)
MAX_POSITIONS = int(os.getenv('MAX_POSITIONS', '5'))  # Reduced from 10 to 5 positions max
MIN_POSITION_SIZE_USD = 20.0  # $20 minimum position size  
# Position sizing per trade - CONSERVATIVE APPROACH
MIN_POSITION_SIZE_PCT = 0.18  # Min 18% of account per position (slightly below max)
MAX_POSITION_SIZE_PCT = 0.20  # Max 20% of account per position (5 × 20% = 100%)

# Pair Selection - Focus on volatile, liquid pairs
PRIORITY_CRYPTO_LIST = [
    'BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT', 'XRPUSDT',
    'ADAUSDT', 'DOGEUSDT', 'MATICUSDT', 'DOTUSDT', 'AVAXUSDT',
    'LINKUSDT', 'UNIUSDT', 'ATOMUSDT', 'LTCUSDT', 'NEARUSDT',
    'INJUSDT', 'SUIUSDT', 'SEIUSDT', 'ARBUSDT', 'OPUSDT'
]

# Volume and Liquidity Filters
MIN_VOLUME_USD = 1000000    # $1M daily volume (lowered from $50M!)
MIN_VOLATILITY_PCT = 0.5    # 0.5% daily volatility minimum (lowered from 1.5%)
MAX_SPREAD_PCT = 0.25       # Max 0.25% spread (increased tolerance)

# =============================================================================
# PROFIT TARGETS - REALISTIC FOR CRYPTO
# =============================================================================

# Quick scalping targets (REALISTIC)
MOMENTUM_TAKE_PROFIT = [
    {"minutes": 5, "profit_pct": 0.5},   # 0.5% in 5 min (was 2%)
    {"minutes": 15, "profit_pct": 0.4},  # 0.4% in 15 min (was 1.5%)
    {"minutes": 30, "profit_pct": 0.3},  # 0.3% in 30 min (was 1%)
    {"minutes": 60, "profit_pct": 0.2},  # 0.2% in 1 hour (was 0.8%)
]

# Regular position targets (REALISTIC)
REGULAR_TAKE_PROFIT = [
    {"minutes": 10, "profit_pct": 0.4},  # 0.4% in 10 min (new quick target)
    {"minutes": 30, "profit_pct": 0.3},  # 0.3% in 30 min (was 3%)
    {"minutes": 60, "profit_pct": 0.25}, # 0.25% in 1 hour (was 2.5%)
    {"minutes": 120, "profit_pct": 0.2}, # 0.2% in 2 hours (was 2%)
    {"minutes": 180, "profit_pct": 0.15}, # 0.15% in 3 hours (was 1.5%)
]

# Partial profit taking (MUCH MORE AGGRESSIVE)
ENABLE_PARTIAL_PROFITS = True
PARTIAL_PROFIT_LEVELS = [
    {"profit_pct": 0.15, "sell_pct": 0.33},  # Sell 33% at 0.15% (was 0.3%)
    {"profit_pct": 0.25, "sell_pct": 0.33},  # Sell 33% at 0.25% (was 0.6%)
    {"profit_pct": 0.4, "sell_pct": 0.34},   # Sell 34% at 0.4% (was 1.0%)
]

# =============================================================================
# RISK MANAGEMENT - ADAPTED FOR CRYPTO
# =============================================================================

# Stop losses - TIGHT for quick exits
MOMENTUM_STOP_LOSS = -1.0   # 1% stop for momentum (tightened from 2%)
QUICK_STOP_LOSS = -1.5      # 1.5% for regular trades (tightened from 2.5%)
MAX_STOP_LOSS = -2.0        # Never exceed 2% loss (tightened from 3%)

# Dynamic stop adjustment
ENABLE_TRAILING_STOPS = True
TRAILING_ACTIVATION_THRESHOLD = 0.01  # Activate at 1% profit
TRAILING_STOP_PERCENTAGE = 0.01       # Trail by 1%

# Breakeven stops
ENABLE_BREAKEVEN_STOP = True
BREAKEVEN_TRIGGER_PCT = 0.5  # Move to breakeven at 0.5%

# Daily targets (ULTRA-AGGRESSIVE for 70-80% daily)
TARGET_DAILY_ROI_MIN = 0.50     # 50% minimum daily target
TARGET_DAILY_ROI_OPTIMAL = 0.70  # 70% optimal daily target
TARGET_DAILY_ROI_MAX = 0.80      # 80% maximum daily target

# Drawdown protection - MUCH LESS AGGRESSIVE (like successful bots)
CONSERVATIVE_THRESHOLDS_ENABLED = False  # Keep disabled
DRAWDOWN_THRESHOLDS = {
    'warning': 0.15,      # 15% (was 5% - much more lenient)
    'reduce_risk': 0.25,  # 25% (was 8% - much more lenient) 
    'high_alert': 0.35,   # 35% (was 10% - much more lenient)
    'emergency': 0.50     # 50% (was 15% - much more lenient)
}

# =============================================================================
# MARKET ANALYSIS SETTINGS
# =============================================================================

# Scanning frequency
OPPORTUNITY_SCAN_INTERVAL = 30  # Every 30 seconds
MARKET_UPDATE_INTERVAL = 60     # Market state every minute

# Timeframes for analysis
ANALYSIS_TIMEFRAMES = ['5m', '15m', '1h']  # Multiple timeframes
DEFAULT_API_TIMEFRAME = '15m'  # Best for crypto scalping

# Signal combination weights
API_SIGNAL_WEIGHT = 0.70       # 70% weight to API
INTERNAL_SIGNAL_WEIGHT = 0.30  # 30% internal

# =============================================================================
# ADVANCED FEATURES
# =============================================================================

# Correlation limits
MAX_CORRELATION = 0.6  # Don't trade highly correlated pairs
MIN_PAIRS_FOR_CORRELATION = 3

# Market regime adjustments
BULL_MARKET_POSITION_MULTIPLIER = 1.2
BEAR_MARKET_POSITION_MULTIPLIER = 0.8
HIGH_VOLATILITY_POSITION_REDUCTION = 0.7

# Time-based adjustments
WEEKEND_POSITION_REDUCTION = 0.8
LOW_VOLUME_HOUR_REDUCTION = 0.9

# Anti-manipulation
MAX_SINGLE_CANDLE_MOVEMENT = 0.10  # Skip if >10% in one candle
MIN_TRADE_COUNT_FILTER = 1000      # Minimum trades per period

# =============================================================================
# API STRATEGY PARAMETERS
# =============================================================================

# Signal quality filters
MIN_API_TECHNICAL_SCORE = 40   # Lowered from 45
MIN_API_MOMENTUM_SCORE = 35     # Lowered from 40
MIN_API_ONCHAIN_SCORE = 30      # Lowered from 35

# Time horizons
RESPECT_API_TIME_HORIZONS = True
API_MAX_HOLD_TIME_HOURS = 24    # Max 24 hours
API_MIN_HOLD_TIME_MINUTES = 3   # Min 3 minutes

# Dynamic adjustments
ENABLE_SIGNAL_STRENGTH_SCALING = True
SIGNAL_STRENGTH_POSITION_MULTIPLIER = {
    'STRONG': 1.5,
    'MODERATE': 1.0,
    'WEAK': 0.7
}

# =============================================================================
# PERFORMANCE TRACKING
# =============================================================================

# Metrics to monitor
TRACK_SLIPPAGE = True
TRACK_API_LATENCY = True
TRACK_WIN_RATE_BY_HOUR = True

# Alert thresholds
MIN_HOURLY_WIN_RATE = 0.45  # Alert if <45%
MAX_CONSECUTIVE_LOSSES = 5   # Alert after 5 losses
MAX_DAILY_API_ERRORS = 10    # Alert on API issues







# =============================================================================
# API-ONLY CONFIGURATION (SIMPLIFIED)
# =============================================================================

# Disable all external APIs - using only Enhanced Signal API
ENABLE_NEBULA = False
ENABLE_COINGECKO = False


# Remove API keys for unused services
NEBULA_SECRET_KEY = None
COINGECKO_API_KEY = None


# =============================================================================
# ULTRA-CONSERVATIVE COINGECKO RATE LIMITING (FIXED)
# =============================================================================

# Ultra-conservative CoinGecko rate limiting for free tier
COINGECKO_FREE_TIER_INTERVAL = float(os.getenv('COINGECKO_FREE_TIER_INTERVAL', '30.0'))  # INCREASED from 15.0 to 30.0
COINGECKO_FREE_TIER_MAX_PER_MINUTE = int(os.getenv('COINGECKO_FREE_TIER_MAX_PER_MINUTE', '2'))  # REDUCED from 3 to 2

# Demo/Pro tier settings (much more conservative)
COINGECKO_DEMO_TIER_INTERVAL = float(os.getenv('COINGECKO_DEMO_TIER_INTERVAL', '15.0'))  # INCREASED from 10.0 to 15.0
COINGECKO_DEMO_TIER_MAX_PER_MINUTE = int(os.getenv('COINGECKO_DEMO_TIER_MAX_PER_MINUTE', '3'))  # REDUCED from 5 to 3

# Rate limit backoff settings (much more aggressive)
COINGECKO_INITIAL_BACKOFF = int(os.getenv('COINGECKO_INITIAL_BACKOFF', '120'))  # INCREASED from 60 to 120 seconds
COINGECKO_MAX_BACKOFF_MULTIPLIER = float(os.getenv('COINGECKO_MAX_BACKOFF_MULTIPLIER', '15.0'))  # INCREASED from 10.0 to 15.0

# Batch processing settings for CoinGecko (much smaller and slower)
COINGECKO_BATCH_SIZE = int(os.getenv('COINGECKO_BATCH_SIZE', '2'))  # REDUCED from 3 to 2
COINGECKO_BATCH_DELAY = float(os.getenv('COINGECKO_BATCH_DELAY', '45.0'))  # INCREASED from 20.0 to 45.0 seconds

# Auto-disable CoinGecko if too many rate limits
DISABLE_COINGECKO_ON_LIMITS = os.getenv('DISABLE_COINGECKO_ON_LIMITS', 'true').lower() == 'true'  # CHANGED to true

# COINGECKO FALLBACK SETTINGS (NEW)
COINGECKO_MAX_CONSECUTIVE_LIMITS = int(os.getenv('COINGECKO_MAX_CONSECUTIVE_LIMITS', '3'))  # NEW
COINGECKO_COOLDOWN_MINUTES = int(os.getenv('COINGECKO_COOLDOWN_MINUTES', '30'))  # NEW

# Reduce CoinGecko weight when having issues
COINGECKO_REDUCED_WEIGHT = float(os.getenv('COINGECKO_REDUCED_WEIGHT', '0.15'))  # NEW - reduced from 0.35

# =============================================================================
# REDUCED TRADING FREQUENCY TO LOWER API PRESSURE (UPDATED)
# =============================================================================

# Increase trading cycle intervals significantly to reduce API pressure
TRADING_CYCLE_INTERVAL = int(os.getenv('TRADING_CYCLE_INTERVAL', '300'))  # INCREASED from 180 to 300 seconds (5 minutes)

# Reduce the number of pairs analyzed simultaneously even more
MAX_CONCURRENT_ANALYSIS = int(os.getenv('MAX_CONCURRENT_ANALYSIS', '3'))  # REDUCED from 5 to 3

# Increase correlation matrix update interval
CORRELATION_UPDATE_INTERVAL = int(os.getenv('CORRELATION_UPDATE_INTERVAL', '14400'))  # INCREASED to 4 hours

# Add longer delays between different API calls
API_CALL_DELAY = float(os.getenv('API_CALL_DELAY', '3.0'))  # INCREASED from 1.0 to 3.0 seconds

# MUCH less aggressive opportunity scanning
OPPORTUNITY_SCAN_INTERVAL = 30  # Every 30 seconds (optimized)

# =============================================================================
# CONSERVATIVE SIGNAL THRESHOLDS (UPDATED)
# =============================================================================

# Higher minimum signal strength to reduce unnecessary trades
MIN_SIGNAL_STRENGTH = float(os.getenv('MIN_SIGNAL_STRENGTH', '0.3'))  # INCREASED from 0.35 to 0.4

# Reduce API pressure by caching more aggressively
CACHE_KLINES_SECONDS = int(os.getenv('CACHE_KLINES_SECONDS', '600'))  # INCREASED from 300 to 600 (10 minutes)
CACHE_ORDERBOOK_SECONDS = int(os.getenv('CACHE_ORDERBOOK_SECONDS', '60'))  # INCREASED from 30 to 60 seconds
# CoinGecko cache removed - API-only mode
CACHE_PRICE_SECONDS = int(os.getenv('CACHE_PRICE_SECONDS', '30'))  # INCREASED from 10 to 30 seconds

# =============================================================================
# DYNAMIC POSITION SIZING CONFIGURATION
# =============================================================================

# Enable dynamic position sizing
ENABLE_DYNAMIC_SIZING = True

# Minimum position size (must be above exchange minimums)
DYNAMIC_MIN_POSITION = float(os.getenv('DYNAMIC_MIN_POSITION', '15'))  # $15 minimum
MIN_NOTIONAL_BUFFER = float(os.getenv('MIN_NOTIONAL_BUFFER', '1.2'))  # 20% buffer

# Signal strength thresholds for dynamic sizing
SIGNAL_STRENGTH_THRESHOLDS = {
    'very_strong': 0.8,   # > 80% confidence
    'strong': 0.6,        # > 60% confidence  
    'moderate': 0.4,      # > 40% confidence
    'weak': 0.3          # > 30% confidence
}

# Position size multipliers based on signal strength
POSITION_SIZE_MULTIPLIERS = {
    'very_strong': 1.5,   # 150% of base size
    'strong': 1.2,        # 120% of base size
    'moderate': 1.0,      # 100% of base size
    'weak': 0.8          # 80% of base size
}

# Override the old MIN_POSITION_SIZE
MIN_POSITION_SIZE = float(os.getenv('MIN_POSITION_SIZE', '15'))  # Changed from 100 to 15

# Pair-specific minimums for problematic pairs
PAIR_SPECIFIC_MINIMUMS = {
    'BCHUSDT': 20,    # BCH might need slightly higher minimum
    'SYRUPUSDT': 15,  # Standard minimum
    'XRPUSDT': 15,    # Standard minimum
}

# Debug mode for position sizing
DEBUG_POSITION_SIZING = True  # Set to False in production


# =============================================================================
# API KEYS AND CREDENTIALS
# =============================================================================

# Binance API credentials
BINANCE_API_KEY = os.getenv('BINANCE_API_KEY', '5kKZ7ENLfCdW3q5NKR9ZREysG8iY6Cx6SHd0qNNMf5BYUAUkFYR6KCBSyqZlKl9O')
BINANCE_API_SECRET = os.getenv('BINANCE_API_SECRET', '9j9ArHrr6f2Pn2Slwdmk6qiZgWdMFa4ELxtDeQpG02jZTw8eYQwqyr2taUuGpPdA')

COINBASE_API_KEY = os.getenv('COINBASE_API_KEY', 'e42ec218-6413-45f9-a99c-6b1e97295885')
COINBASE_API_SECRET = os.getenv('COINBASE_API_SECRET', 'jNGhqLSyOyT/omCAuG1wJYadF090v2chmoPPnDdroiZoJSHQwEM6RaSF5kjRVjXXnI91P0MJuKiowdkQb8peJg==')

# CoinGecko disabled - using API-only mode
ENABLE_COINGECKO = False

# =============================================================================
# TRADING CONFIGURATION
# =============================================================================

# Trading mode (REMOVED - now defined at top of file)
# TEST_MODE = False  # Set to False for live trading (optimized)

# Database path
DB_PATH = os.getenv('DB_PATH', 'trading_bot.db')

# Maximum number of concurrent positions - defined above in POSITION MANAGEMENT section

# =============================================================================
# ENHANCED TRADING PARAMETERS
# =============================================================================

# Opportunity Scanner Settings
ENABLE_OPPORTUNITY_SCANNER = os.getenv('ENABLE_OPPORTUNITY_SCANNER', 'true').lower() == 'true'
MOMENTUM_TRADE_ENABLED = os.getenv('MOMENTUM_TRADE_ENABLED', 'true').lower() == 'true'
QUICK_PROFIT_MODE = os.getenv('QUICK_PROFIT_MODE', 'true').lower() == 'true'

# Volume and momentum thresholds
MIN_VOLUME_USD = float(os.getenv('MIN_VOLUME_USD', '100000'))  # Minimum 24h volume in USD
MAX_PRICE_CHANGE_24H = float(os.getenv('MAX_PRICE_CHANGE_24H', '15'))  # Don't chase if already up more than 15%
VOLUME_SURGE_MULTIPLIER = float(os.getenv('VOLUME_SURGE_MULTIPLIER', '2.0'))  # Look for 2x volume surges
MOMENTUM_TIMEFRAME = os.getenv('MOMENTUM_TIMEFRAME', '5m')  # Quick momentum detection

# =============================================================================
# RISK MANAGEMENT
# =============================================================================

# Daily ROI targets (optimized)
TARGET_DAILY_ROI_MIN = 0.01     # 1% minimum (optimized)
TARGET_DAILY_ROI_OPTIMAL = 0.02  # 2% target (optimized)
TARGET_DAILY_ROI_MAX = 0.03      # 3% maximum (optimized)

# Drawdown protection thresholds (optimized)
DRAWDOWN_THRESHOLDS = {
    'warning': 0.05,      # 5%
    'reduce_risk': 0.08,  # 8%
    'high_alert': 0.10,   # 10%
    'emergency': 0.15     # 15%
}

# Conservative thresholds for new users - DISABLED FOR OPTIMIZATION
CONSERVATIVE_THRESHOLDS_ENABLED = False  # Critical change! (optimized)
CONSERVATIVE_THRESHOLDS = {
    'warning': 0.02,      # 2%
    'reduce_risk': 0.03,  # 3%
    'high_alert': 0.05,   # 5%
    'emergency': 0.08     # 8%
}

# Recovery settings
RECOVERY_PROFIT_TARGET = float(os.getenv('RECOVERY_PROFIT_TARGET', '0.03'))  # 3% recovery needed
RECOVERY_TIMEOUT_DAYS = int(os.getenv('RECOVERY_TIMEOUT_DAYS', '14'))

# Position sizing (CONSERVATIVE for safe trading)
MIN_POSITION_SIZE_PCT = 0.18  # Min 18% of account per position (slightly below max)
MAX_POSITION_SIZE_PCT = 0.20  # Max 20% of account per position (5 × 20% = 100%)

# =============================================================================
# ENHANCED POSITION MANAGEMENT
# =============================================================================

# Dynamic profit targets for momentum trades (optimized)
MOMENTUM_TAKE_PROFIT = [
    {"minutes": 5, "profit_pct": 0.4},   # 0.4% in 5 min
    {"minutes": 15, "profit_pct": 0.3},  # 0.3% in 15 min
    {"minutes": 30, "profit_pct": 0.25}, # 0.25% in 30 min
    {"minutes": 60, "profit_pct": 0.2},  # 0.2% in 1 hour
]

REGULAR_TAKE_PROFIT = [
    {"minutes": 30, "profit_pct": 0.8},  # 0.8% in 30 min
    {"minutes": 60, "profit_pct": 0.6},  # 0.6% in 1 hour
    {"minutes": 120, "profit_pct": 0.5}, # 0.5% in 2 hours
    {"minutes": 180, "profit_pct": 0.4}, # 0.4% in 3 hours
]

# Quick profit targets
QUICK_PROFIT_TARGETS = [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]  # Multiple targets in %

# Stop loss settings (optimized)
MOMENTUM_STOP_LOSS = -2.0   # 2% stop for momentum (optimized)
QUICK_STOP_LOSS = -2.5      # 2.5% for regular trades (optimized)

# Trailing stop loss settings
ENABLE_TRAILING_STOPS = os.getenv('ENABLE_TRAILING_STOPS', 'true').lower() == 'true'
TRAILING_ACTIVATION_THRESHOLD = float(os.getenv('TRAILING_ACTIVATION_THRESHOLD', '0.02'))  # 2%
TRAILING_STOP_PERCENTAGE = float(os.getenv('TRAILING_STOP_PERCENTAGE', '0.015'))  # 1.5%

# Trailing take profit settings
ENABLE_TRAILING_TP = os.getenv('ENABLE_TRAILING_TP', 'true').lower() == 'true'
TRAILING_TP_PERCENTAGE = float(os.getenv('TRAILING_TP_PERCENTAGE', '0.03'))  # 3%

# Position scaling settings
ENABLE_POSITION_SCALING = os.getenv('ENABLE_POSITION_SCALING', 'true').lower() == 'true'
POSITION_SCALE_THRESHOLD = float(os.getenv('POSITION_SCALE_THRESHOLD', '0.02'))  # 2%
SCALE_FACTOR = float(os.getenv('SCALE_FACTOR', '0.5'))  # 50% of original position
MAX_SCALE_COUNT = int(os.getenv('MAX_SCALE_COUNT', '2'))  # Maximum 2 scale-ups per position
MIN_MARKET_TREND_SCORE = float(os.getenv('MIN_MARKET_TREND_SCORE', '0.3'))

# Quick trade settings
MAX_POSITION_AGE_MINUTES = int(os.getenv('MAX_POSITION_AGE_MINUTES', '360'))  # Maximum 6 hours per position
ENABLE_PARTIAL_PROFITS = os.getenv('ENABLE_PARTIAL_PROFITS', 'true').lower() == 'true'
PARTIAL_PROFIT_PERCENTAGE = float(os.getenv('PARTIAL_PROFIT_PERCENTAGE', '0.5'))  # Sell 50% at first target

# =============================================================================
# SIGNAL ANALYSIS WEIGHTS
# =============================================================================

# Technical analysis weight in signal combination
TECHNICAL_WEIGHT = float(os.getenv('TECHNICAL_WEIGHT', '0.40'))

# CoinGecko analysis weight in signal combination  
ONCHAIN_WEIGHT = float(os.getenv('ONCHAIN_WEIGHT', '0.35'))

# Order book analysis weight
ORDERBOOK_WEIGHT = float(os.getenv('ORDERBOOK_WEIGHT', '0.25'))

# CoinGecko signal component weights
COINGECKO_WEIGHTS = {
    'PREDICTION': float(os.getenv('CG_PREDICTION_WEIGHT', '0.35')),
    'SENTIMENT': float(os.getenv('CG_SENTIMENT_WEIGHT', '0.30')),
    'WHALE': float(os.getenv('CG_WHALE_WEIGHT', '0.20')),
    'SMART_MONEY': float(os.getenv('CG_SMART_MONEY_WEIGHT', '0.15'))
}

# Opportunity weights
OPPORTUNITY_WEIGHTS = {
    'volume_surge': 0.25,
    'momentum_shift': 0.20,
    'breakout_pattern': 0.20,
    'whale_accumulation': 0.15,
    'social_momentum': 0.10,
    'unusual_buying': 0.10
}

# =============================================================================
# AGGRESSIVE QUICK MODE CONFIGURATION
# =============================================================================

# Quick mode position sizing - 20% of equity per position
QUICK_MODE_POSITION_SIZE = float(os.getenv('QUICK_MODE_POSITION_SIZE', '0.20'))  # 20% of equity

# Quick mode max positions - only 2 positions at once
QUICK_MODE_MAX_POSITIONS = int(os.getenv('QUICK_MODE_MAX_POSITIONS', '5'))  # Max 5 positions in quick mode

# Normal mode settings - MATCHED TO GLOBAL LIMITS
NORMAL_MODE_MAX_POSITIONS = int(os.getenv('NORMAL_MODE_MAX_POSITIONS', '5'))  # Reduced from 10 to 5

# Minimum position sizes for different modes
QUICK_MODE_MIN_POSITION = float(os.getenv('QUICK_MODE_MIN_POSITION', '300'))   # $300 minimum in quick mode
NORMAL_MODE_MIN_POSITION = float(os.getenv('NORMAL_MODE_MIN_POSITION', '50'))  # $50 minimum in normal mode

# Signal strength requirements for big positions
QUICK_MODE_MIN_SIGNAL = float(os.getenv('QUICK_MODE_MIN_SIGNAL', '0.3'))  

# Note: MAX_POSITIONS is defined above in POSITION MANAGEMENT section
# Dynamic max positions are handled by the risk manager based on mode

# =============================================================================
# ASSET SELECTION (ENHANCED)
# =============================================================================

# Enhanced cryptocurrency list - includes all major and trending coins
EXPANDED_CRYPTO_LIST = [
    # Major coins
    'BTC', 'ETH', 'XRP', 'SOL', 'BNB', 'DOGE', 'ADA', 'TRX', 'LINK', 'AVAX',
    'SUI', 'XLM', 'TON', 'SHIB', 'HBAR', 'DOT', 'LTC', 'BCH', 'OM', 'UNI',
    'PEPE', 'NEAR', 'APT', 'ETC', 'ICP', 'VET', 'POL', 'ALGO', 'RENDER',
    'FIL', 'ARB', 'FET', 'ATOM', 'THETA', 'BONK', 'XTZ', 'IOTA',
    'NEO', 'EGLD', 'ZEC', 'LAYER',
    # Trending/Momentum coins
    'MASK', 'INJ', 'JUP', 'MEW', 'ACH', 'MANA', 'MOVE', 'OP', 'CHZ', 'ENS',
    'API3', 'NEIRO', 'TUT', 'VANA', 'CHILLGUY', 'AUCTION', 'JTO', 'NOT', 'ORDI',
    'PIPPIN', 'WIF', 'BOME', 'FLOKI', 'PEOPLE', 'TURBO',
    # DeFi and Gaming
    'FTM', 'SAND', 'AXS', 'GALA', 'MATIC', 'CRV', 'LDO', 'IMX', 'GRT',
    'AAVE', 'SNX', 'COMP', 'YFI', 'SUSHI', 'ZRX',
    # Additional high-volume coins
    'JASMY', 'FTT', 'GMT', 'APE', 'ROSE', 'MAGIC', 'HIGH', 'RDNT'
]

# Asset selection settings
MAX_ASSETS_TO_ANALYZE = int(os.getenv('MAX_ASSETS_TO_ANALYZE', '50'))  # REDUCED from 50 to 30 for better performance
TRENDING_PAIRS_LIMIT = int(os.getenv('TRENDING_PAIRS_LIMIT', '15'))  # REDUCED from 20 to 15
MAX_CONCURRENT_SCANS = int(os.getenv('MAX_CONCURRENT_SCANS', '50'))  # REDUCED from 100 to 50

# =============================================================================
# MARKET ANALYSIS
# =============================================================================

# Correlation threshold for diversification
MAX_CORRELATION_THRESHOLD = float(os.getenv('MAX_CORRELATION_THRESHOLD', '0.7'))

# Market regime detection settings
MARKET_BREADTH_BULLISH = float(os.getenv('MARKET_BREADTH_BULLISH', '0.6'))  # 60% of coins above MA
MARKET_BREADTH_BEARISH = float(os.getenv('MARKET_BREADTH_BEARISH', '0.4'))  # 40% of coins above MA

# Volatility thresholds (normalized 0-1)
HIGH_VOLATILITY_THRESHOLD = float(os.getenv('HIGH_VOLATILITY_THRESHOLD', '0.7'))
LOW_VOLATILITY_THRESHOLD = float(os.getenv('LOW_VOLATILITY_THRESHOLD', '0.3'))

# =============================================================================
# SCANNER FREQUENCIES AND INTERVALS
# =============================================================================

# Scanner intervals (in seconds) - MORE CONSERVATIVE
POSITION_MONITOR_INTERVAL = int(os.getenv('POSITION_MONITOR_INTERVAL', '30'))  # INCREASED from 20 to 30 seconds
MARKET_ANALYSIS_INTERVAL = int(os.getenv('MARKET_ANALYSIS_INTERVAL', '600'))  # INCREASED from 300 to 600 (10 minutes)
QUICK_SCAN_INTERVAL = int(os.getenv('QUICK_SCAN_INTERVAL', '30'))  # INCREASED from 10 to 30 seconds

# =============================================================================
# PERFORMANCE TARGETS
# =============================================================================

# Daily targets
DAILY_TRADE_TARGET = int(os.getenv('DAILY_TRADE_TARGET', '15'))  # REDUCED from 20 to 15 trades per day
HOURLY_PROFIT_TARGET = float(os.getenv('HOURLY_PROFIT_TARGET', '0.003'))  # REDUCED from 0.005 to 0.003 (0.3% per hour)

# Risk settings for momentum trading
MOMENTUM_POSITION_SIZE_MULTIPLIER = float(os.getenv('MOMENTUM_POSITION_SIZE_MULTIPLIER', '1.1'))  # REDUCED from 1.2 to 1.1
MAX_MOMENTUM_POSITIONS = int(os.getenv('MAX_MOMENTUM_POSITIONS', '3'))  # REDUCED from 5 to 3

# =============================================================================
# MULTIPLE EXCHANGE SUPPORT (FUTURE)
# =============================================================================

EXCHANGES = {
    'binance': {
        'enabled': True,
        'weight': 0.7  # Primary exchange
    },
    'coinbase': {
        'enabled': False,  # Future implementation
        'weight': 0.15
    },
    'bybit': {
        'enabled': False,  # Future implementation
        'weight': 0.15
    }
}

# =============================================================================
# ADDITIONAL DATA SOURCES
# =============================================================================

# Enable additional data sources - MORE CONSERVATIVE
ENABLE_ONCHAIN_METRICS = os.getenv('ENABLE_ONCHAIN_METRICS', 'false').lower() == 'true'  # DISABLED by default
ENABLE_SOCIAL_SENTIMENT = os.getenv('ENABLE_SOCIAL_SENTIMENT', 'false').lower() == 'true'  # DISABLED by default
ENABLE_DEX_DATA = os.getenv('ENABLE_DEX_DATA', 'false').lower() == 'true'  # Future implementation

# =============================================================================
# LOGGING AND MONITORING
# =============================================================================

# Log level
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO').upper()

# Performance reporting frequency (minutes)
PERFORMANCE_REPORT_INTERVAL = int(os.getenv('PERFORMANCE_REPORT_INTERVAL', '60'))

# Trade logging
ENABLE_TRADE_LOGGING = os.getenv('ENABLE_TRADE_LOGGING', 'true').lower() == 'true'

# =============================================================================
# API RATE LIMITING AND CACHING
# =============================================================================

# API rate limiting - MORE CONSERVATIVE
BINANCE_API_CALLS_PER_MINUTE = int(os.getenv('BINANCE_API_CALLS_PER_MINUTE', '60'))  # REDUCED from 100 to 60
COINGECKO_API_CALLS_PER_MINUTE = int(os.getenv('COINGECKO_API_CALLS_PER_MINUTE', '3'))  # HEAVILY REDUCED from 50 to 3

# =============================================================================
# DEVELOPMENT AND TESTING
# =============================================================================

# Enable debug mode
DEBUG_MODE = os.getenv('DEBUG_MODE', 'false').lower() == 'true'

# Paper trading simulation settings
PAPER_TRADING_INITIAL_BALANCE = float(os.getenv('PAPER_TRADING_INITIAL_BALANCE', '10000'))

# Backtesting settings
BACKTEST_START_DATE = os.getenv('BACKTEST_START_DATE', '2024-01-01')
BACKTEST_END_DATE = os.getenv('BACKTEST_END_DATE', '2024-12-31')

# =============================================================================
# VALIDATION
# =============================================================================

def validate_api_config():
    """Validate Enhanced Signal API configuration"""
    errors = []
    
    # Check required settings
    if ENABLE_ENHANCED_API:
        if not SIGNAL_API_URL or SIGNAL_API_URL == 'your-api-key-here':
            errors.append("SIGNAL_API_URL must be configured when ENABLE_ENHANCED_API is True")
        
        if not SIGNAL_API_KEY or SIGNAL_API_KEY == 'your-api-key-here':
            errors.append("SIGNAL_API_KEY must be configured when ENABLE_ENHANCED_API is True")
    
    # Validate thresholds
    if not 0 <= API_MIN_CONFIDENCE <= 100:
        errors.append("API_MIN_CONFIDENCE must be between 0 and 100")
    
    if not 0 <= API_OVERRIDE_CONFIDENCE <= 100:
        errors.append("API_OVERRIDE_CONFIDENCE must be between 0 and 100")
    
    if API_MIN_CONFIDENCE >= API_OVERRIDE_CONFIDENCE:
        errors.append("API_OVERRIDE_CONFIDENCE must be higher than API_MIN_CONFIDENCE")
    
    # Validate weights
    total_weight = API_SIGNAL_WEIGHT + INTERNAL_SIGNAL_WEIGHT
    if abs(total_weight - 1.0) > 0.01:
        errors.append(f"API signal weights must sum to 1.0, got {total_weight}")
    
    # Validate position sizing (updated for optimized config)
    if MAX_POSITION_SIZE_PCT <= MIN_POSITION_SIZE_PCT:
        errors.append("MAX_POSITION_SIZE_PCT must be greater than MIN_POSITION_SIZE_PCT")
    
    if MAX_POSITION_SIZE_PCT > 0.50:
        errors.append("MAX_POSITION_SIZE_PCT should not exceed 50% for safety (AGGRESSIVE MODE)")
    
    # Validate timeframes (updated for optimized config)
    valid_timeframes = ['1m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '12h', '1d']
    if DEFAULT_API_TIMEFRAME not in valid_timeframes:
        errors.append(f"DEFAULT_API_TIMEFRAME must be one of {valid_timeframes}")
    
    # Validate API intervals (Updated for ultra-aggressive trading)
    if API_MIN_INTERVAL < 1:
        errors.append("API_MIN_INTERVAL should be at least 1 second for maximum performance")
    
    if API_REQUEST_TIMEOUT < 5:
        errors.append("API_REQUEST_TIMEOUT should be at least 5 seconds")
    
    if errors:
        raise ValueError("Enhanced Signal API configuration validation failed:\n" + 
                        "\n".join(f"- {error}" for error in errors))
    
    return True




def validate_config():
    """Validate configuration settings"""
    errors = []
    
    # Check required API keys - Fixed validation
    binance_key_valid = BINANCE_API_KEY and BINANCE_API_KEY.strip() != '' and len(BINANCE_API_KEY.strip()) > 10
    binance_secret_valid = BINANCE_API_SECRET and BINANCE_API_SECRET.strip() != '' and len(BINANCE_API_SECRET.strip()) > 10
    
    if not binance_key_valid or not binance_secret_valid:
        errors.append("Binance API key and secret are required and must be valid")

    if ENABLE_ENHANCED_API:
        validate_api_config()
    
    # Validate drawdown thresholds
    thresholds_to_check = CONSERVATIVE_THRESHOLDS if CONSERVATIVE_THRESHOLDS_ENABLED else DRAWDOWN_THRESHOLDS
    thresholds = list(thresholds_to_check.values())
    if not all(thresholds[i] < thresholds[i+1] for i in range(len(thresholds)-1)):
        errors.append("Drawdown thresholds must be in ascending order")
    
    # Validate weights sum to reasonable values
    signal_weights = TECHNICAL_WEIGHT + ONCHAIN_WEIGHT + ORDERBOOK_WEIGHT
    if abs(signal_weights - 1.0) > 0.1:
        errors.append(f"Signal analysis weights sum to {signal_weights:.2f}, should be close to 1.0")
    
    coingecko_weights = sum(COINGECKO_WEIGHTS.values())
    if abs(coingecko_weights - 1.0) > 0.1:
        errors.append(f"CoinGecko weights sum to {coingecko_weights:.2f}, should be close to 1.0")
    
    # Validate percentage values
    percentage_configs = [
        ('TARGET_DAILY_ROI_MIN', TARGET_DAILY_ROI_MIN),
        ('TARGET_DAILY_ROI_OPTIMAL', TARGET_DAILY_ROI_OPTIMAL),
        ('TARGET_DAILY_ROI_MAX', TARGET_DAILY_ROI_MAX),
        ('TRAILING_ACTIVATION_THRESHOLD', TRAILING_ACTIVATION_THRESHOLD),
        ('TRAILING_STOP_PERCENTAGE', TRAILING_STOP_PERCENTAGE),
    ]
    
    for name, value in percentage_configs:
        if not 0 <= value <= 1:
            errors.append(f"{name} should be between 0 and 1 (as decimal), got {value}")
    
    # Validate scanner intervals (updated for optimized config)
    if OPPORTUNITY_SCAN_INTERVAL < 15:
        errors.append("OPPORTUNITY_SCAN_INTERVAL should be at least 15 seconds to avoid overwhelming the system")
    
    # Validate crypto list (updated for optimized config)
    if len(PRIORITY_CRYPTO_LIST) < 10:
        errors.append("PRIORITY_CRYPTO_LIST should contain at least 10 cryptocurrencies")
    
    if errors:
        raise ValueError("Configuration validation failed:\n" + "\n".join(f"- {error}" for error in errors))
    
    return True


def print_api_config_summary():
        """Print summary of API configuration"""
        print("\n" + "="*60)
        print("ENHANCED SIGNAL API CONFIGURATION SUMMARY")
        print("="*60)
        print(f"API Enabled: {ENABLE_ENHANCED_API}")
        print(f"API URL: {SIGNAL_API_URL}")
        print(f"API Key Configured: {'Yes' if SIGNAL_API_KEY != 'your-api-key-here' else 'No'}")
        print(f"Fallback Enabled: {API_FALLBACK_ENABLED}")
        print(f"Min Confidence: {API_MIN_CONFIDENCE}%")
        print(f"Override Confidence: {API_OVERRIDE_CONFIDENCE}%")
        print(f"API Weight: {API_SIGNAL_WEIGHT*100:.1f}%")
        print(f"Position Sizing: {'API' if USE_API_POSITION_SIZING else 'Internal'}")
        print(f"Stop Loss: {'API' if USE_API_STOP_LOSS else 'Internal'}")
        print(f"Take Profit: {'API' if USE_API_TAKE_PROFIT else 'Internal'}")
        print("="*60)

# Validate configuration on import
if __name__ == "__main__":
    try:
        validate_config()
        print_api_config_summary()
        print("✅ Configuration validation passed!")
        print("Configuration validation passed!")
        print(f"- Opportunity Scanner: {'ENABLED' if ENABLE_OPPORTUNITY_SCANNER else 'DISABLED'}")
        print(f"- Momentum Trading: {'ENABLED' if MOMENTUM_TRADE_ENABLED else 'DISABLED'}")
        print(f"- Quick Profit Mode: {'ENABLED' if QUICK_PROFIT_MODE else 'DISABLED'}")
        print(f"- Trading Mode: {'TEST' if TEST_MODE else 'LIVE'}")
        print(f"- Max Positions: {MAX_POSITIONS}")
        print(f"- Crypto Pairs Available: {len(PRIORITY_CRYPTO_LIST)}")
        print(f"- Conservative Mode: {'ENABLED' if CONSERVATIVE_THRESHOLDS_ENABLED else 'DISABLED'}")
        print(f"- CoinGecko Rate Limit: {COINGECKO_FREE_TIER_INTERVAL}s interval, {COINGECKO_FREE_TIER_MAX_PER_MINUTE} requests/min")
    except ValueError as e:
        print(f"Configuration validation failed: {e}")
else:
    # Only validate in non-test environments
    if not os.getenv('SKIP_CONFIG_VALIDATION'):
        validate_config()
        validate_api_config()

# Signal confidence filtering - LOWERED THRESHOLD
MIN_SIGNAL_CONFIDENCE = float(os.getenv('MIN_SIGNAL_CONFIDENCE', '15'))  # Lowered from 20% to 15%
CONFIDENCE_BOOST_THRESHOLD = 30  # Boost position size above this confidence

# Confidence blending for improved signal quality
USE_REGIME_CONFIDENCE_BOOST = True  # Boost confidence when market regime is strong
REGIME_CONFIDENCE_WEIGHT = 0.3  # 30% weight for market regime confidence
ML_CONFIDENCE_WEIGHT = 0.7  # 70% weight for ML confidence

# Symbol validation for live trading safety
VALIDATE_SYMBOLS_IN_TEST = bool(os.getenv('VALIDATE_SYMBOLS_IN_TEST', 'False').lower() == 'true')  # Whether to validate symbols in test mode
SYMBOL_VALIDATION_ENABLED = bool(os.getenv('SYMBOL_VALIDATION_ENABLED', 'True').lower() == 'true')  # Enable symbol validation
SYMBOL_VALIDATION_CACHE_HOURS = int(os.getenv('SYMBOL_VALIDATION_CACHE_HOURS', '24'))  # How long to cache validation results

# =============================================================================
# TRADING MODE - SPOT ONLY (NO LEVERAGE)
# =============================================================================

# CRITICAL: Only use spot trading with available cash
TRADING_MODE = "SPOT_ONLY"  # Never use margin or futures
ENABLE_MARGIN_TRADING = False
ENABLE_FUTURES_TRADING = False
ENABLE_LEVERAGE = False
MAX_LEVERAGE = 1.0  # No leverage allowed

# Position allocation - STRICT LIMITS
POSITION_COUNT_LIMIT = 5  # Maximum 5 positions
POSITION_SIZE_FIXED = 0.20  # Fixed 20% per position
TOTAL_ALLOCATION_LIMIT = 1.00  # Never exceed 100% of cash

# =============================================================================
# POSITION MANAGEMENT - CONSERVATIVE APPROACH  
# =============================================================================