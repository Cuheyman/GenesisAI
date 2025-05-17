import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Keys
BINANCE_API_KEY = os.getenv('BINANCE_API_KEY', '5kKZ7ENLfCdW3q5NKR9ZREysG8iY6Cx6SHd0qNNMf5BYUAUkFYR6KCBSyqZlKl9O')
BINANCE_API_SECRET = os.getenv('BINANCE_API_SECRET', '9j9ArHrr6f2Pn2Slwdmk6qiZgWdMFa4ELxtDeQpG02jZTw8eYQwqyr2taUuGpPdA')
THIRDWEB_API_KEY = os.getenv('THIRDWEB_API_KEY', 'JxXegUHylyRpgMFRw3rVP2Fq-ki4Is79rEyOAB0pPJsULKArQSCfV6U70iiZoVyGcfutg316RnVrHrGssiTcnA')
WEB3_PROVIDER_URL = os.getenv('WEB3_PROVIDER_URL', 'https://1.rpc.thirdweb.com/')


NEBULA_PROXY_URL = "http://localhost:3000"

USE_NEBULA_PROXY = True
THIRDWEB_API_KEY = "b6fce8fb1b94674981826f06e7a4f399"  # Already in your proxy server
NEBULA_WEIGHTS = {
    "PREDICTION": 0.3,   # Weight for price predictions
    "SENTIMENT": 0.2,    # Weight for sentiment analysis
    "WHALE": 0.25,       # Weight for whale tracking
    "SMART_MONEY": 0.25  # Weight for smart money positions
}
ONCHAIN_WEIGHT = 0.25    # Weight for all on-chain signals (in enhanced_strategy.py)
TECHNICAL_WEIGHT = 0.60  # Weight for technical analysis signals




# Performance targets
TARGET_DAILY_ROI_MIN = float(os.getenv('TARGET_DAILY_ROI_MIN', '0.005'))  # 0.5%
TARGET_DAILY_ROI_MAX = float(os.getenv('TARGET_DAILY_ROI_MAX', '0.02'))   # 2%
TARGET_DAILY_ROI_OPTIMAL = float(os.getenv('TARGET_DAILY_ROI_OPTIMAL', '0.01'))  # 1% optimal target

# Drawdown protection thresholds
DRAWDOWN_THRESHOLDS = {
    'warning': 0.05,        # 5% drawdown - warning
    'reduce_risk': 0.10,    # 10% drawdown - reduce risk
    'high_alert': 0.15,     # 15% drawdown - high alert
    'emergency': 0.20       # 20% drawdown - emergency measures
}

# Use more conservative default thresholds for new users
CONSERVATIVE_THRESHOLDS = {
    'warning': 0.03,        # 3% drawdown - warning (more sensitive)
    'reduce_risk': 0.07,    # 7% drawdown - reduce risk
    'high_alert': 0.12,     # 12% drawdown - high alert
    'emergency': 0.15       # 15% drawdown - emergency measures
}
ENABLE_NEBULA = False  # Set to False to disable Nebula integration completely

# Enable conservative thresholds by default for safety
CONSERVATIVE_THRESHOLDS_ENABLED = True

# Recovery settings
RECOVERY_PROFIT_TARGET = 0.05  # 5% recovery before exiting recovery mode
RECOVERY_TIMEOUT_DAYS = 14     # Maximum days in recovery mode before easing
AGGRESSIVE_RECOVERY = False    # If True, take more aggressive recovery actions

# Risk management
MAX_DRAWDOWN_PERCENT = float(os.getenv('MAX_DRAWDOWN_PERCENT', '5.0'))   # Maximum allowed drawdown before reducing exposure
POSITION_SIZE_BASE_PERCENT = float(os.getenv('POSITION_SIZE_BASE_PERCENT', '3.0'))  # Base position size as percentage of portfolio
MAX_POSITIONS = int(os.getenv('MAX_POSITIONS', '15'))  # Maximum concurrent positions
PROFIT_TAKING_LEVELS = [0.5, 1.0, 2.0]  # Take partial profits at these percentages

# Strategy weights
TECHNICAL_WEIGHT = 0.6       # Weight for technical analysis signals
ONCHAIN_WEIGHT = 0.4         # Weight for on-chain signals

# Timeframes to analyze
TIMEFRAMES = ["5m", "15m", "1h", "4h"]
PRIMARY_TIMEFRAME = "1h"

# Asset preferences
PREFERRED_ASSETS = ["BTC", "ETH", "BNB", "SOL", "MATIC", "LINK", "AVAX", "DOT", "UNI", "AAVE"]

# High-volatility safeguards
VOLATILITY_THRESHOLD = 2.5  # Reduce exposure when volatility exceeds this multiple of average
MAX_SLIPPAGE_PERCENT = 0.5  # Maximum allowed slippage on orders


NEBULA_REQUEST_TIMEOUT = 10  # Timeout in seconds for Nebula API requests
NEBULA_MAX_RETRIES = 2      # Maximum number of retries for Nebula API

# Nebula AI settings
NEBULA_MODELS = {
    "market_prediction": "nebula-market-prediction",
    "sentiment": "nebula-sentiment",
    "whale_tracking": "nebula-whale-tracking",
    "smart_money": "nebula-smart-money"
}

# Database settings
DB_PATH = "crypto_bot.db"


# Testing mode
TEST_MODE = os.getenv('TEST_MODE', 'true').lower() == 'true'

# Position scaling settings
ENABLE_POSITION_SCALING = True
MAX_SCALE_COUNT = 3
SCALE_FACTOR = 0.5
POSITION_SCALE_THRESHOLD = 2.0
MIN_MARKET_TREND_SCORE = 0.6

# Trailing take-profit configuration
ENABLE_TRAILING_TP = True
TRAILING_TP_PERCENTAGE = 0.03  # Default trailing distance (3%)
TRAILING_ACTIVATION_THRESHOLD = 0.02  # Activate at 2% profit

# Correlation settings
MAX_CORRELATION = 0.7  # Maximum correlation between positions
CORRELATION_THRESHOLD = 0.7  # Threshold for diversification check
