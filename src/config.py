import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# nebula_config.py
"""
Configuration file for Nebula AI integration
Add these settings to your existing config.py
"""

# Nebula AI API Configuration
ENABLE_NEBULA = True  # Set to False to disable Nebula integration
NEBULA_SECRET_KEY = "S4I2YbrgLJSH6E3OpuFZx156vjC_MiD1EcQvgvrkx40f3bNZNwjtT_3ZIxhb046nZBg2hGKtY9p6JLOGPhSAsA"  # Get from thirdweb dashboard

# Nebula Integration Settings
NEBULA_MIN_CONFIDENCE_THRESHOLD = 0.6  # Minimum confidence to act on Nebula advice
NEBULA_POSITION_SIZE_MULTIPLIER = 1.0  # Adjust Nebula position sizes (0.5 = half size, 2.0 = double)
NEBULA_TIMEOUT_SECONDS = 5.0  # API timeout
NEBULA_MAX_RETRIES = 3  # Maximum retry attempts
NEBULA_RATE_LIMIT_SECONDS = 1.0  # Minimum time between requests

# Integration Strategy
USE_NEBULA_FOR_ENTRIES = True  # Use Nebula for buy decisions
USE_NEBULA_FOR_EXITS = True   # Use Nebula for sell decisions
USE_NEBULA_FOR_RISK_MANAGEMENT = True  # Use Nebula for stop loss/take profit
NEBULA_OVERRIDE_CONFIDENCE = 0.8  # If Nebula confidence > this, override internal signals

# Fallback Behavior
NEBULA_FALLBACK_ENABLED = True  # Fall back to internal logic if Nebula fails
NEBULA_MAX_CONSECUTIVE_FAILURES = 3  # After this many failures, enter cooldown
NEBULA_COOLDOWN_PERIOD_SECONDS = 300  # 5 minutes cooldown after failures

# Performance Tracking
ENABLE_NEBULA_PERFORMANCE_LOGGING = True  # Log detailed Nebula performance
NEBULA_DECISION_LOG_FILE = "logs/nebula_decisions.log"


# =============================================================================
# ULTRA-CONSERVATIVE COINGECKO RATE LIMITING (FIXED)
# =============================================================================

# Ultra-conservative CoinGecko rate limiting for free tier
COINGECKO_FREE_TIER_INTERVAL = float(os.getenv('COINGECKO_FREE_TIER_INTERVAL', '15.0'))  # INCREASED from 5.0 to 15.0
COINGECKO_FREE_TIER_MAX_PER_MINUTE = int(os.getenv('COINGECKO_FREE_TIER_MAX_PER_MINUTE', '3'))  # REDUCED from 10 to 3

# Demo/Pro tier settings (much more conservative)
COINGECKO_DEMO_TIER_INTERVAL = float(os.getenv('COINGECKO_DEMO_TIER_INTERVAL', '10.0'))  # INCREASED from 2.0 to 10.0
COINGECKO_DEMO_TIER_MAX_PER_MINUTE = int(os.getenv('COINGECKO_DEMO_TIER_MAX_PER_MINUTE', '5'))  # REDUCED from 25 to 5

# Rate limit backoff settings (much more aggressive)
COINGECKO_INITIAL_BACKOFF = int(os.getenv('COINGECKO_INITIAL_BACKOFF', '60'))  # INCREASED from 30 to 60 seconds
COINGECKO_MAX_BACKOFF_MULTIPLIER = float(os.getenv('COINGECKO_MAX_BACKOFF_MULTIPLIER', '10.0'))  # INCREASED from 5.0 to 10.0

# Batch processing settings for CoinGecko (much smaller and slower)
COINGECKO_BATCH_SIZE = int(os.getenv('COINGECKO_BATCH_SIZE', '3'))  # REDUCED from 10 to 3
COINGECKO_BATCH_DELAY = float(os.getenv('COINGECKO_BATCH_DELAY', '20.0'))  # INCREASED from 8.0 to 20.0 seconds

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
CACHE_COINGECKO_SECONDS = int(os.getenv('CACHE_COINGECKO_SECONDS', '900'))  # INCREASED from 180 to 900 (15 minutes)
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

# Taapi.io API configuration
TAAPI_API_SECRET = os.getenv('TAAPI_API_SECRET', 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJjbHVlIjoiNjg1NDFjNDI4MDZmZjE2NTFlNTY4ZGNhIiwiaWF0IjoxNzUwNTg1Njg5LCJleHAiOjMzMjU1MDQ5Njg5fQ.U_dTn5P_jpvqJTpTK639n6XIsxHkGdqUDogS6NDrVKU')

# Enable/Disable Taapi.io advanced indicators
ENABLE_TAAPI = os.getenv('ENABLE_TAAPI', 'true').lower() == 'true'

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

# CoinGecko API key (optional - free tier available without key)
COINGECKO_API_KEY = os.getenv('COINGECKO_API_KEY', '')

# Enable/Disable CoinGecko AI analysis
ENABLE_COINGECKO = os.getenv('ENABLE_COINGECKO', 'true').lower() == 'true'

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

# Validate configuration on import
if __name__ == "__main__":
    try:
        validate_config()
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