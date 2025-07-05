import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


"""
Configuration file for Nebula AI integration
Add these settings to your existing config.py
"""

# =============================================================================
# ENHANCED SIGNAL API CONFIGURATION (ADD THIS SECTION)
# =============================================================================

# Enhanced Signal API settings - CRITICAL FOR BOT OPERATION
SIGNAL_API_URL = os.getenv('SIGNAL_API_URL', 'http://localhost:3001/api')
SIGNAL_API_KEY = os.getenv('SIGNAL_API_KEY', '1234')

# Enable/disable Enhanced Signal API
ENABLE_ENHANCED_API = os.getenv('ENABLE_ENHANCED_API', 'false').lower() == 'true'

# API client settings - 24 REQUESTS PER HOUR (1 request every 2.5 minutes)
API_REQUEST_TIMEOUT = int(os.getenv('API_REQUEST_TIMEOUT', '30'))  # 30 seconds timeout
API_MIN_INTERVAL = float(os.getenv('API_MIN_INTERVAL', '150.0'))     # 150 seconds = 2.5 minutes = 24 requests per hour
API_CACHE_DURATION = int(os.getenv('API_CACHE_DURATION', '600'))    # 10 minutes cache to reduce API calls

# API reliability settings
API_MAX_RETRIES = int(os.getenv('API_MAX_RETRIES', '2'))  # REDUCED from 5 to 2
API_FALLBACK_ENABLED = os.getenv('API_FALLBACK_ENABLED', 'true').lower() == 'true'
API_HEALTH_CHECK_INTERVAL = int(os.getenv('API_HEALTH_CHECK_INTERVAL', '600'))  # INCREASED from 60 to 600 seconds (10 minutes)

# Signal processing settings
API_MIN_CONFIDENCE = float(os.getenv('API_MIN_CONFIDENCE', '55'))      # Minimum confidence threshold (55%)
API_OVERRIDE_CONFIDENCE = float(os.getenv('API_OVERRIDE_CONFIDENCE', '85'))  # High confidence override (85%)

# API feature usage
USE_API_POSITION_SIZING = os.getenv('USE_API_POSITION_SIZING', 'true').lower() == 'true'
USE_API_STOP_LOSS = os.getenv('USE_API_STOP_LOSS', 'true').lower() == 'true'
USE_API_TAKE_PROFIT = os.getenv('USE_API_TAKE_PROFIT', 'true').lower() == 'true'
USE_API_TIME_HORIZONS = os.getenv('USE_API_TIME_HORIZONS', 'true').lower() == 'true'

# Signal weighting (when combining API with internal signals)
API_SIGNAL_WEIGHT = float(os.getenv('API_SIGNAL_WEIGHT', '0.85'))      # 85% weight to API signals
INTERNAL_SIGNAL_WEIGHT = float(os.getenv('INTERNAL_SIGNAL_WEIGHT', '0.15'))  # 15% weight to internal

# =============================================================================
# API ANALYSIS PARAMETERS (CUSTOMIZE THESE)
# =============================================================================

# Default analysis parameters for API calls
DEFAULT_API_TIMEFRAME = os.getenv('DEFAULT_API_TIMEFRAME', '1h')
DEFAULT_API_ANALYSIS_DEPTH = os.getenv('DEFAULT_API_ANALYSIS_DEPTH', 'comprehensive')
DEFAULT_API_RISK_LEVEL = os.getenv('DEFAULT_API_RISK_LEVEL', 'moderate')

# Market condition based parameters
API_VOLATILE_TIMEFRAME = os.getenv('API_VOLATILE_TIMEFRAME', '15m')
API_STABLE_TIMEFRAME = os.getenv('API_STABLE_TIMEFRAME', '4h')
API_TRENDING_ANALYSIS = os.getenv('API_TRENDING_ANALYSIS', 'comprehensive')
API_RANGING_ANALYSIS = os.getenv('API_RANGING_ANALYSIS', 'advanced')

# =============================================================================
# ENHANCED POSITION MANAGEMENT WITH API
# =============================================================================

# API-based position sizing
API_MAX_POSITION_SIZE_PCT = float(os.getenv('API_MAX_POSITION_SIZE_PCT', '12'))   # Max 12% per position with API
API_MIN_POSITION_SIZE_PCT = float(os.getenv('API_MIN_POSITION_SIZE_PCT', '2'))    # Min 2% per position

# API confidence based sizing multipliers
API_CONFIDENCE_MULTIPLIERS = {
    'very_high': float(os.getenv('API_VERY_HIGH_CONFIDENCE_MULT', '1.3')),  # 85%+ confidence
    'high': float(os.getenv('API_HIGH_CONFIDENCE_MULT', '1.1')),            # 70-84% confidence  
    'medium': float(os.getenv('API_MEDIUM_CONFIDENCE_MULT', '1.0')),        # 55-69% confidence
    'low': float(os.getenv('API_LOW_CONFIDENCE_MULT', '0.7'))               # 55-60% confidence
}

# On-chain score based adjustments
API_ONCHAIN_SCORE_MULTIPLIERS = {
    'excellent': float(os.getenv('API_ONCHAIN_EXCELLENT_MULT', '1.2')),     # 80+ on-chain score
    'good': float(os.getenv('API_ONCHAIN_GOOD_MULT', '1.1')),              # 60-79 on-chain score
    'fair': float(os.getenv('API_ONCHAIN_FAIR_MULT', '1.0')),              # 40-59 on-chain score
    'poor': float(os.getenv('API_ONCHAIN_POOR_MULT', '0.8'))               # <40 on-chain score
}

# Whale influence adjustments
API_WHALE_INFLUENCE_MULTIPLIERS = {
    'POSITIVE': float(os.getenv('API_WHALE_POSITIVE_MULT', '1.15')),        # Whale accumulation
    'NEUTRAL': float(os.getenv('API_WHALE_NEUTRAL_MULT', '1.0')),          # No whale activity
    'NEGATIVE': float(os.getenv('API_WHALE_NEGATIVE_MULT', '0.85'))         # Whale distribution
}

# =============================================================================
# API RISK MANAGEMENT OVERRIDES
# =============================================================================

# Override risk management with API recommendations
USE_API_RISK_MANAGEMENT = os.getenv('USE_API_RISK_MANAGEMENT', 'true').lower() == 'true'

# API-based stop loss settings
API_DYNAMIC_STOP_LOSS = os.getenv('API_DYNAMIC_STOP_LOSS', 'true').lower() == 'true'
API_MAX_STOP_LOSS_PCT = float(os.getenv('API_MAX_STOP_LOSS_PCT', '3.0'))    # Max 3% stop loss
API_MIN_STOP_LOSS_PCT = float(os.getenv('API_MIN_STOP_LOSS_PCT', '1.0'))    # Min 1% stop loss

# API-based take profit settings  
API_DYNAMIC_TAKE_PROFIT = os.getenv('API_DYNAMIC_TAKE_PROFIT', 'true').lower() == 'true'
API_PARTIAL_PROFIT_ENABLED = os.getenv('API_PARTIAL_PROFIT_ENABLED', 'true').lower() == 'true'
API_MAX_TAKE_PROFIT_PCT = float(os.getenv('API_MAX_TAKE_PROFIT_PCT', '15.0'))  # Max 15% take profit

# API time horizon respect
RESPECT_API_TIME_HORIZONS = os.getenv('RESPECT_API_TIME_HORIZONS', 'true').lower() == 'true'
API_MAX_HOLD_TIME_HOURS = int(os.getenv('API_MAX_HOLD_TIME_HOURS', '48'))     # Max 48 hours
API_MIN_HOLD_TIME_MINUTES = int(os.getenv('API_MIN_HOLD_TIME_MINUTES', '5'))  # Min 5 minutes

# =============================================================================
# API MONITORING AND ALERTING
# =============================================================================

# API performance monitoring
ENABLE_API_PERFORMANCE_MONITORING = os.getenv('ENABLE_API_PERFORMANCE_MONITORING', 'true').lower() == 'true'
API_STATS_LOG_INTERVAL = int(os.getenv('API_STATS_LOG_INTERVAL', '1800'))  # Log every 30 minutes
API_HEALTH_LOG_INTERVAL = int(os.getenv('API_HEALTH_LOG_INTERVAL', '300'))  # Check every 5 minutes

# API failure handling
API_MAX_CONSECUTIVE_FAILURES = int(os.getenv('API_MAX_CONSECUTIVE_FAILURES', '5'))
API_FAILURE_COOLDOWN_MINUTES = int(os.getenv('API_FAILURE_COOLDOWN_MINUTES', '15'))
API_AUTO_FALLBACK_THRESHOLD = float(os.getenv('API_AUTO_FALLBACK_THRESHOLD', '0.6'))  # <60% success rate

# =============================================================================
# SIGNAL VALIDATION AND FILTERING  
# =============================================================================

# API signal validation
VALIDATE_API_SIGNALS = os.getenv('VALIDATE_API_SIGNALS', 'true').lower() == 'true'
API_SIGNAL_CONSISTENCY_CHECK = os.getenv('API_SIGNAL_CONSISTENCY_CHECK', 'true').lower() == 'true'

# Signal filtering based on market conditions
FILTER_API_SIGNALS_BY_REGIME = os.getenv('FILTER_API_SIGNALS_BY_REGIME', 'true').lower() == 'true'
API_AVOID_VOLATILE_MARKETS = os.getenv('API_AVOID_VOLATILE_MARKETS', 'false').lower() == 'true'

# Minimum signal requirements
API_MIN_TECHNICAL_SCORE = float(os.getenv('API_MIN_TECHNICAL_SCORE', '45'))   # Min technical score
API_MIN_MOMENTUM_SCORE = float(os.getenv('API_MIN_MOMENTUM_SCORE', '40'))     # Min momentum score  
API_MIN_ONCHAIN_SCORE = float(os.getenv('API_MIN_ONCHAIN_SCORE', '35'))      # Min on-chain score





# =============================================================================
# API-ONLY CONFIGURATION (SIMPLIFIED)
# =============================================================================

# Disable all external APIs - using only Enhanced Signal API
ENABLE_NEBULA = False
ENABLE_COINGECKO = False
ENABLE_TAAPI = False

# Remove API keys for unused services
NEBULA_SECRET_KEY = None
COINGECKO_API_KEY = None
TAAPI_API_SECRET = None

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
OPPORTUNITY_SCAN_INTERVAL = int(os.getenv('OPPORTUNITY_SCAN_INTERVAL', '120'))  # INCREASED from 60 to 120 seconds

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
# TAAPI.IO CONFIGURATION
# =============================================================================

# Disable Taapi.io completely to avoid unnecessary API calls
ENABLE_TAAPI = False
TAAPI_API_SECRET = None

# Taapi.io rate limiting (Free tier: 1 request per 15 seconds)
TAAPI_REQUESTS_PER_WINDOW = int(os.getenv('TAAPI_REQUESTS_PER_WINDOW', '1'))
TAAPI_TIME_WINDOW = int(os.getenv('TAAPI_TIME_WINDOW', '15'))

# Advanced indicators configuration
TAAPI_PREFERRED_EXCHANGE = os.getenv('TAAPI_PREFERRED_EXCHANGE', 'binance')

# Taapi.io indicator weights in signal combination
TAAPI_ICHIMOKU_WEIGHT = float(os.getenv('TAAPI_ICHIMOKU_WEIGHT', '0.15'))
TAAPI_SUPERTREND_WEIGHT = float(os.getenv('TAAPI_SUPERTREND_WEIGHT', '0.20'))
TAAPI_TDSEQUENTIAL_WEIGHT = float(os.getenv('TAAPI_TDSEQUENTIAL_WEIGHT', '0.10'))
TAAPI_FISHER_WEIGHT = float(os.getenv('TAAPI_FISHER_WEIGHT', '0.10'))
TAAPI_CHOPPINESS_WEIGHT = float(os.getenv('TAAPI_CHOPPINESS_WEIGHT', '0.05'))
TAAPI_CANDLESTICK_WEIGHT = float(os.getenv('TAAPI_CANDLESTICK_WEIGHT', '0.15'))

# Overall Taapi.io indicators weight in final signal
TAAPI_OVERALL_WEIGHT = float(os.getenv('TAAPI_OVERALL_WEIGHT', '0.25'))

# Timeframes for different Taapi.io indicators
TAAPI_ICHIMOKU_TIMEFRAME = os.getenv('TAAPI_ICHIMOKU_TIMEFRAME', '4h')
TAAPI_TDSEQUENTIAL_TIMEFRAME = os.getenv('TAAPI_TDSEQUENTIAL_TIMEFRAME', '1d')
TAAPI_SUPERTREND_TIMEFRAME = os.getenv('TAAPI_SUPERTREND_TIMEFRAME', '1h')
TAAPI_FISHER_TIMEFRAME = os.getenv('TAAPI_FISHER_TIMEFRAME', '1h')
TAAPI_CHOPPINESS_TIMEFRAME = os.getenv('TAAPI_CHOPPINESS_TIMEFRAME', '4h')
TAAPI_CANDLESTICK_TIMEFRAME = os.getenv('TAAPI_CANDLESTICK_TIMEFRAME', '1h')

# Cache settings for Taapi.io
TAAPI_CACHE_ENABLED = os.getenv('TAAPI_CACHE_ENABLED', 'true').lower() == 'true'

# Error handling
TAAPI_ERROR_BACKOFF_SECONDS = int(os.getenv('TAAPI_ERROR_BACKOFF_SECONDS', '300'))  # 5 minutes
TAAPI_MAX_RETRIES = int(os.getenv('TAAPI_MAX_RETRIES', '2'))

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

# Trading mode
TEST_MODE = os.getenv('TEST_MODE', 'true').lower() == 'true'

# Database path
DB_PATH = os.getenv('DB_PATH', 'trading_bot.db')

# Maximum number of concurrent positions
MAX_POSITIONS = int(os.getenv('MAX_POSITIONS', '10'))

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

# Daily ROI targets
TARGET_DAILY_ROI_MIN = float(os.getenv('TARGET_DAILY_ROI_MIN', '0.005'))  # 0.5%
TARGET_DAILY_ROI_OPTIMAL = float(os.getenv('TARGET_DAILY_ROI_OPTIMAL', '0.015'))  # 1.5%
TARGET_DAILY_ROI_MAX = float(os.getenv('TARGET_DAILY_ROI_MAX', '0.025'))  # 2.5%

# Drawdown protection thresholds
DRAWDOWN_THRESHOLDS = {
    'warning': float(os.getenv('DRAWDOWN_WARNING', '0.03')),      # 3%
    'reduce_risk': float(os.getenv('DRAWDOWN_REDUCE_RISK', '0.05')),  # 5%
    'high_alert': float(os.getenv('DRAWDOWN_HIGH_ALERT', '0.08')),    # 8%
    'emergency': float(os.getenv('DRAWDOWN_EMERGENCY', '0.12'))       # 12%
}

# Conservative thresholds for new users - ENABLED BY DEFAULT NOW
CONSERVATIVE_THRESHOLDS_ENABLED = os.getenv('CONSERVATIVE_MODE', 'true').lower() == 'true'  # CHANGED to true
CONSERVATIVE_THRESHOLDS = {
    'warning': 0.02,      # 2%
    'reduce_risk': 0.03,  # 3%
    'high_alert': 0.05,   # 5%
    'emergency': 0.08     # 8%
}

# Recovery settings
RECOVERY_PROFIT_TARGET = float(os.getenv('RECOVERY_PROFIT_TARGET', '0.03'))  # 3% recovery needed
RECOVERY_TIMEOUT_DAYS = int(os.getenv('RECOVERY_TIMEOUT_DAYS', '14'))

# Position sizing
MAX_POSITION_SIZE_PCT = float(os.getenv('MAX_POSITION_SIZE_PCT', '0.15'))  # Max 15% of account per position

# =============================================================================
# ENHANCED POSITION MANAGEMENT
# =============================================================================

# Dynamic profit targets for momentum trades
MOMENTUM_TAKE_PROFIT = [
    {"minutes": 5, "profit_pct": 1.0},    # 1% in 5 minutes
    {"minutes": 15, "profit_pct": 0.8},   # 0.8% in 15 minutes
    {"minutes": 30, "profit_pct": 0.6},   # 0.6% in 30 minutes
    {"minutes": 60, "profit_pct": 0.4},   # 0.4% in 1 hour
]

REGULAR_TAKE_PROFIT = [
    {"minutes": 30, "profit_pct": 1.5},   # 1.5% in 30 min
    {"minutes": 60, "profit_pct": 1.0},   # 1% in 1 hour
    {"minutes": 120, "profit_pct": 0.8},  # 0.8% in 2 hours
    {"minutes": 180, "profit_pct": 0.5},  # 0.5% in 3 hours
]

# Quick profit targets
QUICK_PROFIT_TARGETS = [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]  # Multiple targets in %

# Stop loss settings
MOMENTUM_STOP_LOSS = float(os.getenv('MOMENTUM_STOP_LOSS', '-1.0'))  # 1% stop loss for momentum trades
QUICK_STOP_LOSS = float(os.getenv('QUICK_STOP_LOSS', '-1.5'))     # 1.5% for regular quick trades

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

# Quick mode position sizing - 40% of equity per position
QUICK_MODE_POSITION_SIZE = float(os.getenv('QUICK_MODE_POSITION_SIZE', '0.40'))  # 40% of equity

# Quick mode max positions - only 2 positions at once
QUICK_MODE_MAX_POSITIONS = int(os.getenv('QUICK_MODE_MAX_POSITIONS', '5'))  # Max 2 positions

# Normal mode settings (existing behavior)
NORMAL_MODE_MAX_POSITIONS = int(os.getenv('NORMAL_MODE_MAX_POSITIONS', '10'))  # Normal max

# Minimum position sizes for different modes
QUICK_MODE_MIN_POSITION = float(os.getenv('QUICK_MODE_MIN_POSITION', '300'))   # $300 minimum in quick mode
NORMAL_MODE_MIN_POSITION = float(os.getenv('NORMAL_MODE_MIN_POSITION', '50'))  # $50 minimum in normal mode

# Signal strength requirements for big positions
QUICK_MODE_MIN_SIGNAL = float(os.getenv('QUICK_MODE_MIN_SIGNAL', '0.3'))  

# Dynamic MAX_POSITIONS based on mode
def get_max_positions():
    """Get max positions based on current trading mode"""
    if QUICK_PROFIT_MODE == True:
        return QUICK_MODE_MAX_POSITIONS  # 5 positions
    else:
        return NORMAL_MODE_MAX_POSITIONS  # 10 positions

# Update MAX_POSITIONS dynamically
MAX_POSITIONS = get_max_positions()

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
    
    # Validate position sizing
    if API_MAX_POSITION_SIZE_PCT <= API_MIN_POSITION_SIZE_PCT:
        errors.append("API_MAX_POSITION_SIZE_PCT must be greater than API_MIN_POSITION_SIZE_PCT")
    
    if API_MAX_POSITION_SIZE_PCT > 25:
        errors.append("API_MAX_POSITION_SIZE_PCT should not exceed 25% for safety")
    
    # Validate timeframes
    valid_timeframes = ['1m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '12h', '1d']
    if DEFAULT_API_TIMEFRAME not in valid_timeframes:
        errors.append(f"DEFAULT_API_TIMEFRAME must be one of {valid_timeframes}")
    
    # Validate analysis depths
    valid_depths = ['basic', 'advanced', 'comprehensive']
    if DEFAULT_API_ANALYSIS_DEPTH not in valid_depths:
        errors.append(f"DEFAULT_API_ANALYSIS_DEPTH must be one of {valid_depths}")
    
    # Validate risk levels
    valid_risk_levels = ['conservative', 'moderate', 'aggressive']
    if DEFAULT_API_RISK_LEVEL not in valid_risk_levels:
        errors.append(f"DEFAULT_API_RISK_LEVEL must be one of {valid_risk_levels}")
    
    if errors:
        raise ValueError("Enhanced Signal API configuration validation failed:\n" + 
                        "\n".join(f"- {error}" for error in errors))
    
    return True






def validate_taapi_config():
    """Validate Taapi.io configuration settings"""
    errors = []
    
    if ENABLE_TAAPI and not TAAPI_API_SECRET:
        errors.append("TAAPI_API_SECRET is required when ENABLE_TAAPI is True")
    
    if TAAPI_REQUESTS_PER_WINDOW < 1:
        errors.append("TAAPI_REQUESTS_PER_WINDOW must be at least 1")
    
    if TAAPI_TIME_WINDOW < 1:
        errors.append("TAAPI_TIME_WINDOW must be at least 1 second")
    
    # Validate timeframes
    valid_timeframes = ['1m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '12h', '1d', '1w']
    timeframe_configs = [
        ('TAAPI_ICHIMOKU_TIMEFRAME', TAAPI_ICHIMOKU_TIMEFRAME),
        ('TAAPI_TDSEQUENTIAL_TIMEFRAME', TAAPI_TDSEQUENTIAL_TIMEFRAME),
        ('TAAPI_SUPERTREND_TIMEFRAME', TAAPI_SUPERTREND_TIMEFRAME),
        ('TAAPI_FISHER_TIMEFRAME', TAAPI_FISHER_TIMEFRAME),
        ('TAAPI_CHOPPINESS_TIMEFRAME', TAAPI_CHOPPINESS_TIMEFRAME),
        ('TAAPI_CANDLESTICK_TIMEFRAME', TAAPI_CANDLESTICK_TIMEFRAME)
    ]
    
    for config_name, timeframe in timeframe_configs:
        if timeframe not in valid_timeframes:
            errors.append(f"{config_name} must be one of {valid_timeframes}")
    
    # Validate weights (should be between 0 and 1)
    weight_configs = [
        ('TAAPI_ICHIMOKU_WEIGHT', TAAPI_ICHIMOKU_WEIGHT),
        ('TAAPI_SUPERTREND_WEIGHT', TAAPI_SUPERTREND_WEIGHT),
        ('TAAPI_TDSEQUENTIAL_WEIGHT', TAAPI_TDSEQUENTIAL_WEIGHT),
        ('TAAPI_FISHER_WEIGHT', TAAPI_FISHER_WEIGHT),
        ('TAAPI_CHOPPINESS_WEIGHT', TAAPI_CHOPPINESS_WEIGHT),
        ('TAAPI_CANDLESTICK_WEIGHT', TAAPI_CANDLESTICK_WEIGHT),
        ('TAAPI_OVERALL_WEIGHT', TAAPI_OVERALL_WEIGHT)
    ]
    
    for config_name, weight in weight_configs:
        if not 0 <= weight <= 1:
            errors.append(f"{config_name} must be between 0 and 1")
    
    return errors

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
    
    # Validate scanner intervals
    if OPPORTUNITY_SCAN_INTERVAL < 60:
        errors.append("OPPORTUNITY_SCAN_INTERVAL should be at least 60 seconds to avoid rate limits")
    
    # Validate crypto list
    if len(EXPANDED_CRYPTO_LIST) < 20:
        errors.append("EXPANDED_CRYPTO_LIST should contain at least 20 cryptocurrencies")
    
    # Validate Taapi.io config
    taapi_errors = validate_taapi_config()
    errors.extend(taapi_errors)
    
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
        print(f"- Crypto Pairs Available: {len(EXPANDED_CRYPTO_LIST)}")
        print(f"- Conservative Mode: {'ENABLED' if CONSERVATIVE_THRESHOLDS_ENABLED else 'DISABLED'}")
        print(f"- CoinGecko Rate Limit: {COINGECKO_FREE_TIER_INTERVAL}s interval, {COINGECKO_FREE_TIER_MAX_PER_MINUTE} requests/min")
    except ValueError as e:
        print(f"Configuration validation failed: {e}")
else:
    # Only validate in non-test environments
    if not os.getenv('SKIP_CONFIG_VALIDATION'):
        validate_config()
        validate_api_config()