// ===============================================
// SERVER.JS - Enhanced Main Application
// ===============================================


const express = require('express');
const cors = require('cors');
const helmet = require('helmet');
const rateLimit = require('express-rate-limit');
const Anthropic = require('@anthropic-ai/sdk');
const axios = require('axios');
const fetch = require('node-fetch'); // Add this if not already installed
require('dotenv').config();

const app = express();
const PORT = process.env.PORT || 3000;

// Initialize Claude AI
const anthropic = new Anthropic({
  apiKey: process.env.CLAUDE_API_KEY,
});

// ===============================================
// STARTUP LOGGING SYSTEM
// ===============================================

const startupLogger = {
  log: (message, data = {}) => {
    const timestamp = new Date().toISOString();
    console.log(`[STARTUP] ${timestamp} - ${message}`, data);
  },
  
  error: (message, error = null) => {
    const timestamp = new Date().toISOString();
    console.error(`[STARTUP ERROR] ${timestamp} - ${message}`, error);
  },
  
  success: (message, data = {}) => {
    const timestamp = new Date().toISOString();
    console.log(`[STARTUP SUCCESS] ${timestamp} - ${message}`, data);
  }
};

// Startup logging function
const logStartupStatus = async () => {
  startupLogger.log('=== GENESIS AI TRADING BOT STARTUP ===');
  
  // Environment check
  startupLogger.log('Checking environment variables...');
  const envVars = {
    NODE_ENV: process.env.NODE_ENV || 'development',
    PORT: process.env.PORT || 3000,
    API_KEY_SECRET: process.env.API_KEY_SECRET ? 'SET' : 'MISSING',
    CLAUDE_API_KEY: process.env.CLAUDE_API_KEY ? 'SET' : 'MISSING',
    ANTHROPIC_API_KEY: process.env.ANTHROPIC_API_KEY ? 'SET' : 'MISSING',
    NEBULA_API_KEY: process.env.NEBULA_API_KEY ? 'SET' : 'MISSING',
    PYTHON_PATH: process.env.PYTHON_PATH || 'python',
    BINANCE_API_KEY: process.env.BINANCE_API_KEY ? 'SET' : 'MISSING',
    BINANCE_API_SECRET: process.env.BINANCE_API_SECRET ? 'SET' : 'MISSING'
  };
  startupLogger.log('Environment variables status:', envVars);
  
  // API Key validation
  if (!process.env.API_KEY_SECRET) {
    startupLogger.error('API_KEY_SECRET is missing - API authentication will fail');
  } else {
    startupLogger.success('API_KEY_SECRET is configured');
  }
  
  if (!process.env.CLAUDE_API_KEY && !process.env.ANTHROPIC_API_KEY) {
    startupLogger.error('CLAUDE_API_KEY/ANTHROPIC_API_KEY is missing - Claude AI will not work');
  } else {
    startupLogger.success('Claude API key is configured');
  }
  
  if (!process.env.NEBULA_API_KEY) {
    startupLogger.error('NEBULA_API_KEY is missing - Nebula AI will not work');
  } else {
    startupLogger.success('NEBULA_API_KEY is configured');
  }
  
  if (!process.env.BINANCE_API_KEY || !process.env.BINANCE_API_SECRET) {
    startupLogger.error('BINANCE_API_KEY/SECRET is missing - Symbol validation may be limited');
  } else {
    startupLogger.success('Binance API credentials are configured');
  }
  
  // Python environment check
  startupLogger.log('Checking Python environment...');
  try {
    const { spawn } = require('child_process');
    const pythonProcess = spawn(process.env.PYTHON_PATH || 'python', ['--version']);
    
    pythonProcess.stdout.on('data', (data) => {
      startupLogger.success(`Python version: ${data.toString().trim()}`);
    });
    
    pythonProcess.stderr.on('data', (data) => {
      startupLogger.error('Python version check failed:', data.toString());
    });
    
    pythonProcess.on('close', (code) => {
      if (code === 0) {
        startupLogger.success('Python environment is available');
      } else {
        startupLogger.error('Python environment check failed');
      }
    });
  } catch (error) {
    startupLogger.error('Failed to check Python environment:', error);
  }
  
  // Dependencies check
  startupLogger.log('Checking Node.js dependencies...');
  try {
    const requiredModules = [
      'axios', 'winston', '@anthropic-ai/sdk', 
      'express', 'cors', 'helmet', 'express-rate-limit',
      'node-fetch', 'binance-api-node'
    ];
    
    for (const module of requiredModules) {
      try {
        require(module);
        startupLogger.success(`✓ ${module} is available`);
      } catch (error) {
        startupLogger.error(`✗ ${module} is missing or failed to load`);
      }
    }
  } catch (error) {
    startupLogger.error('Dependency check failed:', error);
  }
  
  // File system check
  startupLogger.log('Checking file system structure...');
  const fs = require('fs');
  const path = require('path');
  
  const requiredPaths = [
    './logs',
    './predictive-model',
    './services',
    './middleware'
  ];
  
  for (const dirPath of requiredPaths) {
    if (fs.existsSync(dirPath)) {
      startupLogger.success(`✓ Directory exists: ${dirPath}`);
    } else {
      startupLogger.error(`✗ Directory missing: ${dirPath}`);
    }
  }
  
  // Check if log files are writable
  try {
    fs.accessSync('./logs', fs.constants.W_OK);
    startupLogger.success('Log directory is writable');
  } catch (error) {
    startupLogger.error('Log directory is not writable');
  }
  
  // ML model files check
  startupLogger.log('Checking ML model files...');
  const mlFiles = [
    './predictive-model/ml_predictor.py',
    './predictive-model/__init__.py',
    './predictive-model/risk_manager.py',
    './predictive-model/portfolio_optimizer.py',
    './predictive-model/rl_agent.py',
    './predictive-model/deep_learning_model.py',
    './predictive-model/order_flow_analyzer.py'
  ];
  
  for (const file of mlFiles) {
    if (fs.existsSync(file)) {
      startupLogger.success(`✓ ML file exists: ${file}`);
    } else {
      startupLogger.error(`✗ ML file missing: ${file}`);
    }
  }
  
  // Service files check
  startupLogger.log('Checking service files...');
  const serviceFiles = [
    './services/lunarCrushService.js',
    './services/mlcService.js',
    './services/performanceMonitor.js',
    './services/realtimeService.js',
    './services/sentimentAnalyzer.js',
    './services/defiIntegration.js'
  ];
  
  for (const file of serviceFiles) {
    if (fs.existsSync(file)) {
      startupLogger.success(`✓ Service file exists: ${file}`);
    } else {
      startupLogger.error(`✗ Service file missing: ${file}`);
    }
  }
  
  // Network connectivity test
  startupLogger.log('Testing network connectivity...');
  try {
    // Test CoinGecko API
    const coinGeckoTest = axios.get('https://api.coingecko.com/api/v3/ping', { timeout: 5000 })
      .then(() => {
        startupLogger.success('✓ CoinGecko API is reachable');
      })
      .catch((error) => {
        startupLogger.error('✗ CoinGecko API is not reachable:', error.message);
      });
    
    // Test Binance API
    const binanceTest = axios.get('https://api.binance.com/api/v3/ping', { timeout: 5000 })
      .then(() => {
        startupLogger.success('✓ Binance API is reachable');
      })
      .catch((error) => {
        startupLogger.error('✗ Binance API is not reachable:', error.message);
      });
    
  } catch (error) {
    startupLogger.error('Network connectivity test failed:', error);
  }
  
  startupLogger.log('=== STARTUP LOGGING COMPLETE ===');
};

// ===============================================
// MIDDLEWARE SETUP
// ===============================================

// Security middleware
app.use(helmet());
app.use(cors({
  origin: process.env.NODE_ENV === 'production' ? ['your-domain.com'] : '*',
  methods: ['GET', 'POST'],
  allowedHeaders: ['Content-Type', 'Authorization']
}));

// Body parsing
app.use(express.json({ limit: '10mb' }));
app.use(express.urlencoded({ extended: true }));

// Rate limiting
const limiter = rateLimit({
  windowMs: parseInt(process.env.RATE_LIMIT_WINDOW) || 3600000, // 1 hour
  max: parseInt(process.env.RATE_LIMIT_MAX) || 100, // requests per window
  message: {
    error: 'Too many requests',
    retryAfter: 3600
  },
  standardHeaders: true,
  legacyHeaders: false
});

app.use('/api/', limiter);

// ===============================================
// LOGGING SETUP
// ===============================================

const winston = require('winston');

const logger = winston.createLogger({
  level: process.env.LOG_LEVEL || 'info',
  format: winston.format.combine(
    winston.format.timestamp(),
    winston.format.errors({ stack: true }),
    winston.format.json()
  ),
  transports: [
    new winston.transports.Console(), // Ensure Console transport is always present
    new winston.transports.File({ filename: 'logs/error.log', level: 'error' }),
    new winston.transports.File({ filename: 'logs/combined.log' })
  ]
});

// ===============================================
// DYNAMIC SYMBOL VALIDATION SYSTEM
// ===============================================

// Cache for valid symbols
let validSymbolsCache = {
  symbols: [],
  lastUpdated: 0,
  updateInterval: 3600000, // 1 hour
  isUpdating: false,
  symbolMetadata: {}, // Store additional info about symbols
  symbolsByVolume: [], // Symbols sorted by 24h volume
  stablecoins: [] // List of stablecoin pairs
};

// Fallback symbols in case Binance API is unavailable
const FALLBACK_SYMBOLS = [
  'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'SOLUSDT', 
  'DOTUSDT', 'LINKUSDT', 'LTCUSDT', 'XRPUSDT', 'MATICUSDT'
];

// Symbol validation statistics
const symbolStats = {
  validationAttempts: 0,
  validationFailures: 0,
  lastFailedSymbols: [],
  apiRefreshAttempts: 0,
  apiRefreshFailures: 0,
  lastError: null
};

const Binance = require('binance-api-node').default;

// Initialize Binance client (add this at the top of your file if not already there)
const binanceClient = Binance({
  apiKey: process.env.BINANCE_API_KEY,
  apiSecret: process.env.BINANCE_API_SECRET,
  test: false // Set to true for testnet
});

async function fetchValidSymbolsFromBinance() {
  try {
    logger.info('Fetching valid symbols from Binance API...');
    symbolStats.apiRefreshAttempts++;

    // Use Binance SDK instead of fetch
    const data = await binanceClient.exchangeInfo();

    // Enhanced filtering for USDT spot trading pairs
    const validSymbols = [];
    const symbolMetadata = {};
    const stablecoins = [];

    data.symbols.forEach(symbol => {
      // Basic filtering
      if (symbol.status !== 'TRADING' || 
          !symbol.symbol.endsWith('USDT') ||
          !symbol.isSpotTradingAllowed) {
        return;
      }

      // Exclude leveraged tokens and other derivatives
      const excludePatterns = ['BEAR', 'BULL', 'DOWN', 'UP', 'LONG', 'SHORT'];
      if (excludePatterns.some(pattern => symbol.symbol.includes(pattern))) {
        return;
      }

      validSymbols.push(symbol.symbol);

      // Store metadata for enhanced filtering
      symbolMetadata[symbol.symbol] = {
        baseAsset: symbol.baseAsset,
        quoteAsset: symbol.quoteAsset,
        filters: symbol.filters,
        permissions: symbol.permissions,
        orderTypes: symbol.orderTypes,
        minNotional: getMinNotionalValue(symbol.filters),
        tickSize: getTickSize(symbol.filters),
        stepSize: getStepSize(symbol.filters)
      };

      // Identify stablecoins
      const stablecoinBases = ['USDC', 'BUSD', 'TUSD', 'USDP', 'DAI', 'FRAX', 'GUSD'];
      if (stablecoinBases.includes(symbol.baseAsset)) {
        stablecoins.push(symbol.symbol);
      }
    });

    logger.info(`Successfully fetched ${validSymbols.length} valid USDT trading pairs`);
    logger.info(`Identified ${stablecoins.length} stablecoin pairs`);

    return {
      symbols: validSymbols.sort(),
      metadata: symbolMetadata,
      stablecoins: stablecoins
    };
  } catch (error) {
    symbolStats.apiRefreshFailures++;
    symbolStats.lastError = error.message;

    if (error.name === 'AbortError') {
      logger.error('Binance API request timed out');
    } else {
      logger.error('Failed to fetch symbols from Binance:', error);
    }
    throw error;
  }
}

// Helper functions (keep these if you don't already have them)
function getMinNotionalValue(filters) {
  const minNotionalFilter = filters.find(f => f.filterType === 'MIN_NOTIONAL');
  return minNotionalFilter ? parseFloat(minNotionalFilter.minNotional) : null;
}

function getTickSize(filters) {
  const priceFilter = filters.find(f => f.filterType === 'PRICE_FILTER');
  return priceFilter ? parseFloat(priceFilter.tickSize) : null;
}

function getStepSize(filters) {
  const lotSizeFilter = filters.find(f => f.filterType === 'LOT_SIZE');
  return lotSizeFilter ? parseFloat(lotSizeFilter.stepSize) : null;
}

/**
 * Get minimum notional value from symbol filters
 */
function getMinNotionalValue(filters) {
  const minNotionalFilter = filters.find(f => f.filterType === 'MIN_NOTIONAL');
  return minNotionalFilter ? parseFloat(minNotionalFilter.minNotional) : 10;
}

/**
 * Get tick size from symbol filters
 */
function getTickSize(filters) {
  const priceFilter = filters.find(f => f.filterType === 'PRICE_FILTER');
  return priceFilter ? parseFloat(priceFilter.tickSize) : 0.00000001;
}

/**
 * Get step size from symbol filters
 */
function getStepSize(filters) {
  const lotSizeFilter = filters.find(f => f.filterType === 'LOT_SIZE');
  return lotSizeFilter ? parseFloat(lotSizeFilter.stepSize) : 0.00000001;
}

/**
 * Fetch 24hr ticker data to sort symbols by volume
 */
async function fetchSymbolVolumes() {
  try {
    // Use Binance SDK to get 24hr ticker statistics
    const tickers = await binanceClient.dailyStats();
    const volumeMap = {};

    tickers.forEach(ticker => {
      if (ticker.symbol.endsWith('USDT')) {
        volumeMap[ticker.symbol] = parseFloat(ticker.quoteVolume);
      }
    });

    return volumeMap;
  } catch (error) {
    logger.error('Failed to fetch symbol volumes:', error);
    return {};
  }
}

/**
 * Update the valid symbols cache with enhanced data
 * @param {boolean} force - Force update even if cache is fresh
 */
async function updateValidSymbols(force = false) {
  const now = Date.now();
  
  // Check if update is needed
  if (!force && 
      validSymbolsCache.symbols.length > 0 && 
      (now - validSymbolsCache.lastUpdated) < validSymbolsCache.updateInterval) {
    return validSymbolsCache.symbols;
  }
  
  // Prevent multiple simultaneous updates
  if (validSymbolsCache.isUpdating) {
    return validSymbolsCache.symbols;
  }
  
  validSymbolsCache.isUpdating = true;
  
  try {
    // Fetch symbols and metadata
    const symbolData = await fetchValidSymbolsFromBinance();
    
    // Sanity check - Binance should have at least 200 USDT pairs
    if (symbolData.symbols.length < 100) {
      throw new Error(`Suspiciously low symbol count: ${symbolData.symbols.length}`);
    }
    
    // Fetch volume data for sorting
    const volumeData = await fetchSymbolVolumes();
    
    // Sort symbols by volume (high to low)
    const symbolsByVolume = symbolData.symbols.sort((a, b) => {
      const volA = volumeData[a] || 0;
      const volB = volumeData[b] || 0;
      return volB - volA;
    });
    
    // Update cache
    validSymbolsCache.symbols = symbolData.symbols;
    validSymbolsCache.symbolMetadata = symbolData.metadata;
    validSymbolsCache.stablecoins = symbolData.stablecoins;
    validSymbolsCache.symbolsByVolume = symbolsByVolume;
    validSymbolsCache.lastUpdated = now;
    
    logger.info(`Symbol cache updated with ${symbolData.symbols.length} symbols`);
    logger.info(`Top 5 by volume: ${symbolsByVolume.slice(0, 5).join(', ')}`);
    
    return symbolData.symbols;
  } catch (error) {
    logger.error('Failed to update symbol cache, using fallback or existing cache');
    
    // If we have no symbols at all, use fallback
    if (validSymbolsCache.symbols.length === 0) {
      validSymbolsCache.symbols = FALLBACK_SYMBOLS;
      logger.warn(`Using ${FALLBACK_SYMBOLS.length} fallback symbols`);
    }
    
    return validSymbolsCache.symbols;
  } finally {
    validSymbolsCache.isUpdating = false;
  }
}

/**
 * Get valid symbols (from cache or fetch if needed)
 * @returns {Promise<string[]>} Array of valid symbols
 */
async function getValidSymbols() {
  if (validSymbolsCache.symbols.length === 0) {
    // Initial load
    return await updateValidSymbols(true);
  }
  
  // Check if cache needs refresh (non-blocking)
  const now = Date.now();
  if ((now - validSymbolsCache.lastUpdated) > validSymbolsCache.updateInterval) {
    // Update in background, don't wait
    updateValidSymbols().catch(err => 
      logger.error('Background symbol update failed:', err)
    );
  }
  
  return validSymbolsCache.symbols;
}

/**
 * Check if a symbol is valid
 * @param {string} symbol - Symbol to check
 * @returns {Promise<boolean>} Whether the symbol is valid
 */
async function isValidSymbol(symbol) {
  const validSymbols = await getValidSymbols();
  return validSymbols.includes(symbol);
}

/**
 * Get symbol metadata
 * @param {string} symbol - Symbol to get metadata for
 * @returns {Object|null} Symbol metadata or null if not found
 */
function getSymbolMetadata(symbol) {
  return validSymbolsCache.symbolMetadata[symbol] || null;
}

/**
 * Get top symbols by volume
 * @param {number} count - Number of symbols to return
 * @returns {string[]} Top symbols by 24h volume
 */
function getTopSymbolsByVolume(count = 10) {
  return validSymbolsCache.symbolsByVolume.slice(0, count);
}

// ===============================================
// LUNARCRUSH SERVICE
// ===============================================

const CoinGeckoService = require('./services/lunarCrushService');
const MLCService = require('./services/mlcService');
const coinGeckoService = new CoinGeckoService();
const mlcService = new MLCService();

class EnhancedCoinGeckoService {
  constructor() {
    this.coinGecko = coinGeckoService;
  }

  async getOnChainAnalysis(symbol, walletAddress = null) {
    try {
      const analysis = await this.coinGecko.getComprehensiveAnalysis(symbol, walletAddress);
      
      return {
        whale_activity: {
          large_transfers_24h: Math.floor(analysis.whale_activity.whale_activity_score * 100),
          whale_accumulation: analysis.whale_activity.large_transactions ? 'buying' : 'neutral',
          top_holder_changes: analysis.coin_data.dominance || 15
        },
        network_metrics: {
          active_addresses: Math.floor((analysis.coin_data.volume_24h / 1000000) * 100), // Estimate based on volume
          transaction_volume_24h: analysis.coin_data.volume_24h,
          gas_usage_trend: analysis.market_sentiment > 0.5 ? 'increasing' : 'stable'
        },
        defi_metrics: {
          total_locked_value: analysis.coin_data.market_cap * 0.1,
          yield_farming_apy: analysis.market_metrics.sharpe_ratio * 10,
          protocol_inflows: analysis.market_sentiment * 1000000
        },
        sentiment_indicators: {
          on_chain_sentiment: analysis.sentiment_score > 0.3 ? 'bullish' : analysis.sentiment_score < -0.3 ? 'bearish' : 'neutral',
          smart_money_flow: 'neutral', // Not available from CoinGecko
          derivative_metrics: {
            funding_rates: analysis.market_sentiment * 0.1,
            open_interest_change: analysis.market_sentiment * 20
          }
        },
        cross_chain_analysis: {
          arbitrage_opportunities: false, // Not available from CoinGecko
          bridge_volumes: analysis.coin_data.volume_24h * 0.05,
          chain_dominance: 'ethereum'
        },
        risk_assessment: {
          liquidity_score: analysis.confidence_score * 100,
          volatility_prediction: analysis.market_metrics.volatility,
          market_manipulation_risk: analysis.risk_indicators.overall_risk > 0.7 ? 'high' : analysis.risk_indicators.overall_risk > 0.4 ? 'medium' : 'low'
        },
        timestamp: Date.now(),
        source: 'coingecko',
        confidence: analysis.confidence_score,
        market_metrics: analysis.market_metrics
      };
    } catch (error) {
      logger.error('CoinGecko analysis failed:', error);
      return this.getFallbackOnChainData(symbol);
    }
  }

  getFallbackOnChainData(symbol) {
    return {
      whale_activity: {
        large_transfers_24h: 25,
        whale_accumulation: 'neutral',
        top_holder_changes: 0
      },
      network_metrics: {
        active_addresses: 75000,
        transaction_volume_24h: 750000000,
        gas_usage_trend: 'stable'
      },
      defi_metrics: {
        total_locked_value: 5000000000,
        yield_farming_apy: 8.5,
        protocol_inflows: 0
      },
      sentiment_indicators: {
        on_chain_sentiment: 'neutral',
        smart_money_flow: 'neutral',
        derivative_metrics: {
          funding_rates: 0.01,
          open_interest_change: 0
        }
      },
      cross_chain_analysis: {
        arbitrage_opportunities: false,
        bridge_volumes: 50000000,
        chain_dominance: 'ethereum'
      },
      risk_assessment: {
        liquidity_score: 75,
        volatility_prediction: 25,
        market_manipulation_risk: 'low'
      },
      timestamp: Date.now(),
      source: 'coingecko_fallback'
    };
  }
}

// ===============================================
// TECHNICAL ANALYSIS (Enhanced)
// ===============================================

class TechnicalAnalysis {
  static calculateSMA(prices, period) {
    if (prices.length < period) return null;
    const sum = prices.slice(-period).reduce((a, b) => a + b, 0);
    return sum / period;
  }

  static calculateEMA(prices, period) {
    if (prices.length < period) return null;
    const multiplier = 2 / (period + 1);
    let ema = prices.slice(0, period).reduce((a, b) => a + b, 0) / period;
    
    for (let i = period; i < prices.length; i++) {
      ema = (prices[i] * multiplier) + (ema * (1 - multiplier));
    }
    return ema;
  }

  static calculateRSI(prices, period = 14) {
    if (prices.length < period + 1) return null;
    
    let gains = 0, losses = 0;
    for (let i = 1; i <= period; i++) {
      const change = prices[prices.length - i] - prices[prices.length - i - 1];
      if (change > 0) gains += change;
      else losses -= change;
    }
    
    const avgGain = gains / period;
    const avgLoss = losses / period;
    if (avgLoss === 0) return 100;
    
    const rs = avgGain / avgLoss;
    return 100 - (100 / (1 + rs));
  }

  static calculateMACD(prices) {
    const ema12 = this.calculateEMA(prices, 12);
    const ema26 = this.calculateEMA(prices, 26);
    if (!ema12 || !ema26) return null;
    
    const macd = ema12 - ema26;
    const signal = macd * 0.9; // Simplified signal line
    const histogram = macd - signal;
    
    return { macd, signal, histogram };
  }

  static calculateBollingerBands(prices, period = 20, stdDev = 2) {
    const sma = this.calculateSMA(prices, period);
    if (!sma) return null;
    
    const squaredDiffs = prices.slice(-period).map(price => Math.pow(price - sma, 2));
    const variance = squaredDiffs.reduce((a, b) => a + b, 0) / period;
    const standardDeviation = Math.sqrt(variance);
    
    return {
      upper: sma + (standardDeviation * stdDev),
      middle: sma,
      lower: sma - (standardDeviation * stdDev)
    };
  }

  static calculateStochastic(prices, period = 14) {
    if (prices.length < period) return null;
    const recentPrices = prices.slice(-period);
    const high = Math.max(...recentPrices);
    const low = Math.min(...recentPrices);
    const current = prices[prices.length - 1];
    
    const k = ((current - low) / (high - low)) * 100;
    return { k, d: k * 0.9 }; // Simplified %D
  }

  static calculateATR(prices, period = 14) {
    if (prices.length < period) return null;
    const ranges = [];
    
    for (let i = 1; i < period; i++) {
      const high = Math.max(...prices.slice(-i-1, -i+1));
      const low = Math.min(...prices.slice(-i-1, -i+1));
      ranges.push(high - low);
    }
    
    return ranges.reduce((a, b) => a + b, 0) / ranges.length;
  }

  static calculateVolatility(prices, period = 20) {
    if (prices.length < period) return null;
    const returns = [];
    
    for (let i = 1; i < period; i++) {
      const idx = prices.length - i;
      returns.push(Math.log(prices[idx] / prices[idx - 1]));
    }
    
    const avgReturn = returns.reduce((a, b) => a + b, 0) / returns.length;
    const variance = returns.reduce((sum, ret) => sum + Math.pow(ret - avgReturn, 2), 0) / returns.length;
    return Math.sqrt(variance * 252); // Annualized volatility
  }

  static calculateAdvancedMetrics(prices, volumes) {
    const current = prices[prices.length - 1];
    const sma20 = this.calculateSMA(prices, 20);
    const sma50 = this.calculateSMA(prices, 50);
    const ema12 = this.calculateEMA(prices, 12);
    const ema26 = this.calculateEMA(prices, 26);

    // Volume-based indicators
    const avgVolume = volumes ? this.calculateSMA(volumes, 20) : null;
    const currentVolume = volumes ? volumes[volumes.length - 1] : null;
    const volumeRatio = avgVolume && currentVolume ? currentVolume / avgVolume : 1;

    // Advanced momentum indicators
    const roc = prices.length >= 10 ? ((current - prices[prices.length - 10]) / prices[prices.length - 10]) * 100 : 0;
    const momentum = prices.length >= 5 ? current - prices[prices.length - 5] : 0;

    return {
      price: current,
      sma_20: sma20,
      sma_50: sma50,
      ema_12: ema12,
      ema_26: ema26,
      rsi: this.calculateRSI(prices),
      macd: this.calculateMACD(prices),
      bollinger_bands: this.calculateBollingerBands(prices),
      stochastic: this.calculateStochastic(prices),
      atr: this.calculateATR(prices),
      volatility: this.calculateVolatility(prices),
      volume_ratio: volumeRatio,
      rate_of_change: roc,
      momentum: momentum,
      trend_strength: sma20 ? Math.abs((current - sma20) / sma20) * 100 : 0,
      market_regime: this.determineMarketRegime(prices, volumes)
    };
  }

  static determineMarketRegime(prices, volumes) {
    const volatility = this.calculateVolatility(prices) || 0.02;
    const sma20 = this.calculateSMA(prices, 20);
    const sma50 = this.calculateSMA(prices, 50);
    const current = prices[prices.length - 1];
    
    if (volatility > 0.05) return 'high_volatility';
    if (sma20 && sma50) {
      if (current > sma20 && sma20 > sma50) return 'uptrend';
      if (current < sma20 && sma20 < sma50) return 'downtrend';
    }
    return 'sideways';
  }
}

// ===============================================
// MARKET DATA SERVICE (Enhanced)
// ===============================================

// ===============================================
// MARKET DATA SERVICE (Enhanced) - FIXED VERSION
// ===============================================

class MarketDataService {
  static generateEnhancedData(symbol, timeframe = '1h', bars = 100) {
    // Expanded base prices for more symbols
    const basePrices = {
      // Major pairs
      'BTCUSDT': 43500,
      'ETHUSDT': 2650,
      'BNBUSDT': 315,
      'ADAUSDT': 0.52,
      'SOLUSDT': 98,
      'DOTUSDT': 7.8,
      'LINKUSDT': 15.2,
      'LTCUSDT': 72,
      'XRPUSDT': 0.63,
      'MATICUSDT': 0.89,
      
      // Additional pairs from your logs
      'PENGUUSDT': 0.015,
      'PROMUSDT': 6.39,
      'HBARUSDT': 0.28,
      'LPTUSDT': 18.5,
      'ONDOUSDT': 1.25,
      'WBTCUSDT': 43500,
      'AAVEUSDT': 340,
      
      // Common altcoins
      'HFTUSDT': 0.45,
      'MAVUSDT': 0.32,
      'JTOUSDT': 3.2,
      'SUPERUSDT': 1.8,
      'SYRUPUSDT': 0.95,
      'SEIUSDT': 0.48,
      'ENAUSDT': 1.15,
      'NEIROUSDT': 0.002,
      'JUPUSDT': 1.05,
      'GPSUSDT': 0.18,
      'INITUSDT': 0.22,
      'ORDIUSDT': 42,
      '1000SATSUSDT': 0.0003,
      'TRBUSDT': 68,
      'ZKUSDT': 0.185,
      'KAITOUSDT': 0.0015,
      'ARBUSDT': 0.82,
      
      // Missing symbols from your errors
      'SAHARAUSDT': 0.0025,
      'RSRUSDT': 0.85,
      'ARKUSDT': 0.72,
      'AWEUSDT': 0.35,
      'VIRTUALUSDT': 2.8,
      'TONUSDT': 5.2,
      'PIXELUSDT': 0.25,  // Added this missing symbol
      'UNIUSDT': 8.5,     // Added this missing symbol
      'APTUSDT': 12,      // Added this missing symbol
      
      // DeFi tokens
      'AVAXUSDT': 36,
      'ATOMUSDT': 7.2,
      'ALGOUSDT': 0.28,
      'FTMUSDT': 0.72,
      'SANDUSDT': 0.68,
      'MANAUSDT': 0.58,
      'AXSUSDT': 8.5,
      'GALAUSDT': 0.045,
      'CRVUSDT': 0.85,
      'LDOUSDT': 2.1,
      'IMXUSDT': 1.6,
      'GRTUSDT': 0.28,
      'COMPUSDT': 78,
      'YFIUSDT': 8500,
      'SUSHIUSDT': 1.2,
      'ZRXUSDT': 0.52,
      'JASMYUSDT': 0.035,
      'FTTUSDT': 2.8,
      'GMTUSDT': 0.18,
      'APEUSDT': 1.45,
      'ROSEUSDT': 0.085,
      'MAGICUSDT': 0.68,
      'HIGHUSDT': 2.3,
      'RDNTUSDT': 0.095,
      'INJUSDT': 24,
      'OPUSDT': 2.1,
      'CHZUSDT': 0.095,
      'ENSUSDT': 32,
      'API3USDT': 2.8,
      'MASKUSDT': 3.2,
      'MEWUSDT': 0.012,
      'ACHUSDT': 0.032,
      'MOVEUSDT': 0.88,
      'NOTUSDT': 0.0085,
      'WIFUSDT': 3.2,
      'BOMEUSDT': 0.014,
      'FLOKIUSDT': 0.00025,
      'PEOPLEUSDT': 0.065,
      'TURBOUSDT': 0.008,
      'NEOUSDT': 18,
      'EGLDUSDT': 28,
      'ZECUSDT': 45,
      'LAYERUSDT': 0.18,
      'NEARUSDT': 5.8,
      'ETCUSDT': 28,
      'ICPUSDT': 12.5,
      'VETUSDT': 0.045,
      'POLUSDT': 0.68,
      'RENDERUSDT': 7.2,
      'FILUSDT': 5.8,
      'FETUSDT': 1.6,
      'THETAUSDT': 2.1,
      'BONKUSDT': 0.000035,
      'XTZUSDT': 1.2,
      'IOTAUSDT': 0.28
    };

    const basePrice = basePrices[symbol] || 1.0; // Default fallback
    // FIXED: Pass basePrices as parameter to getVolatilityForSymbol
    const volatility = this.getVolatilityForSymbol(symbol, basePrices);
    
    // Generate realistic price and volume history
    const prices = [];
    const volumes = [];
    let currentPrice = basePrice * (0.95 + Math.random() * 0.1);
    let trend = (Math.random() - 0.5) * 0.001;
    
    for (let i = 0; i < bars; i++) {
      // Price generation with more realistic movement
      if (Math.random() < 0.1) {
        trend = (Math.random() - 0.5) * 0.001;
      }
      
      const randomChange = (Math.random() - 0.5) * volatility * currentPrice;
      const trendChange = trend * currentPrice;
      currentPrice += randomChange + trendChange;
      
      if (currentPrice < 0.000001) currentPrice = 0.000001; // Prevent negative prices
      prices.push(currentPrice);
      
      // Volume generation (correlated with price movements)
      const baseVolume = this.getBaseVolumeForSymbol(symbol);
      const volatilityMultiplier = 1 + Math.abs(randomChange / currentPrice) * 5;
      const volume = baseVolume * volatilityMultiplier * (0.5 + Math.random());
      volumes.push(volume);
    }

    const priceChange24h = ((prices[prices.length - 1] - prices[prices.length - 24]) / prices[prices.length - 24]) * 100;

    return {
      symbol,
      current_price: prices[prices.length - 1],
      price_history: prices,
      volume_history: volumes,
      volume_24h: volumes.slice(-24).reduce((a, b) => a + b, 0),
      price_change_24h: priceChange24h,
      market_cap: currentPrice * this.getCirculatingSupply(symbol),
      timestamp: Date.now(),
      timeframe,
      bars_count: bars
    };
  }

  // FIXED: Added basePrices parameter to resolve the scope issue
  static getVolatilityForSymbol(symbol, basePrices = null) {
    // Expanded volatility mapping
    const volatilities = {
      // Major pairs - lower volatility
      'BTCUSDT': 0.02,
      'ETHUSDT': 0.025,
      'BNBUSDT': 0.03,
      
      // Mid-cap altcoins
      'ADAUSDT': 0.04,
      'SOLUSDT': 0.035,
      'DOTUSDT': 0.035,
      'LINKUSDT': 0.04,
      'LTCUSDT': 0.03,
      'XRPUSDT': 0.045,
      'MATICUSDT': 0.05,
      
      // Small-cap and newer tokens - higher volatility
      'PENGUUSDT': 0.08,
      'PROMUSDT': 0.06,
      'HBARUSDT': 0.05,
      'LPTUSDT': 0.055,
      'ONDOUSDT': 0.065,
      'WBTCUSDT': 0.02,
      'AAVEUSDT': 0.05,
      'PIXELUSDT': 0.08,  // Added missing symbols
      'UNIUSDT': 0.045,
      'APTUSDT': 0.055,
      
      // Very small caps - highest volatility
      'NEIROUSDT': 0.12,
      '1000SATSUSDT': 0.15,
      'BONKUSDT': 0.18,
      'FLOKIUSDT': 0.15,
      'PEOPLEUSDT': 0.10,
      'SAHARAUSDT': 0.20,
      'RSRUSDT': 0.08,
      'ARKUSDT': 0.07,
      'AWEUSDT': 0.12,
      'VIRTUALUSDT': 0.09
    };
    
    // Default volatility based on symbol characteristics
    if (symbol.includes('1000') || symbol.includes('PEPE') || symbol.includes('SHIB')) {
      return 0.15; // Meme coins
    } else if (basePrices && basePrices[symbol] && basePrices[symbol] < 0.01) {
      return 0.12; // Very low price coins
    } else if (basePrices && basePrices[symbol] && basePrices[symbol] < 1) {
      return 0.08; // Low price coins
    }
    
    return volatilities[symbol] || 0.06; // Default medium volatility
  }

  static getBaseVolumeForSymbol(symbol) {
    // Expanded volume mapping based on market cap tiers
    const baseVolumes = {
      // Tier 1 - Highest volume
      'BTCUSDT': 80000000,
      'ETHUSDT': 60000000,
      'BNBUSDT': 20000000,
      
      // Tier 2 - High volume
      'ADAUSDT': 12000000,
      'SOLUSDT': 16000000,
      'XRPUSDT': 30000000,
      'DOTUSDT': 8000000,
      'LINKUSDT': 6000000,
      'LTCUSDT': 12000000,
      'MATICUSDT': 8000000,
      
      // Tier 3 - Medium volume
      'AAVEUSDT': 4000000,
      'AVAXUSDT': 6000000,
      'ATOMUSDT': 3000000,
      'INJUSDT': 5000000,
      'NEARUSDT': 4000000,
      'APTUSDT': 3500000,
      'PIXELUSDT': 2500000,  // Added missing symbols
      'UNIUSDT': 8000000,
      
      // Tier 4 - Lower volume
      'PENGUUSDT': 800000,
      'PROMUSDT': 600000,
      'HBARUSDT': 1500000,
      'LPTUSDT': 1200000,
      'ONDOUSDT': 800000,
      
      // Tier 5 - Lowest volume
      'NEIROUSDT': 200000,
      '1000SATSUSDT': 300000,
      'BONKUSDT': 400000,
      'FLOKIUSDT': 350000,
      'SAHARAUSDT': 150000,
      'RSRUSDT': 250000,
      'ARKUSDT': 300000,
      'AWEUSDT': 180000,
      'VIRTUALUSDT': 500000
    };
    
    return baseVolumes[symbol] || 1000000; // Default 1M volume
  }

  static getCirculatingSupply(symbol) {
    // Expanded supply data
    const supplies = {
      'BTCUSDT': 19.7e6,
      'ETHUSDT': 120e6,
      'BNBUSDT': 166e6,
      'ADAUSDT': 35e9,
      'SOLUSDT': 400e6,
      'DOTUSDT': 1.2e9,
      'LINKUSDT': 500e6,
      'LTCUSDT': 73e6,
      'XRPUSDT': 53e9,
      'MATICUSDT': 9e9,
      
      // New additions
      'PENGUUSDT': 88e12,
      'PROMUSDT': 2e6,
      'HBARUSDT': 50e9,
      'LPTUSDT': 27e6,
      'ONDOUSDT': 1e9,
      'WBTCUSDT': 160e3,
      'AAVEUSDT': 16e6,
      'PIXELUSDT': 5e9,    // Added missing symbols
      'UNIUSDT': 1e9,
      'APTUSDT': 1e9,
      
      // Default estimates for missing symbols
      'NEIROUSDT': 420e12,
      '1000SATSUSDT': 21e15,
      'BONKUSDT': 90e12,
      'FLOKIUSDT': 9e12,
      'SAHARAUSDT': 1e12,
      'RSRUSDT': 1e9,
      'ARKUSDT': 100e6,
      'AWEUSDT': 1e9,
      'VIRTUALUSDT': 1e9
    };
    
    return supplies[symbol] || 1e9; // Default 1B supply
  }
}
// ===============================================
// ENHANCED AI SIGNAL GENERATOR
// ===============================================

class EnhancedAISignalGenerator {
  constructor() {
    this.coinGeckoService = new EnhancedCoinGeckoService();
    this.mlcService = mlcService;
  }

  async generateAdvancedSignal(marketData, technicalData, onChainData, requestParams) {
    try {
      // Generate signal using both Claude and CoinGecko insights
      const claudeAnalysis = await this.generateClaudeSignal(marketData, technicalData, onChainData, requestParams);
      
      // Get ML predictions
      const mlResults = await this.mlcService.getMLPredictions(
        requestParams.symbol || 'BTCUSDT', 
        marketData, 
        technicalData
      );
      
      // Log data source information
      const dataSource = onChainData.source || 'unknown';
      const mlSource = mlResults.timestamp ? 'ml_models' : 'fallback';
      const isRealData = dataSource === 'coingecko' && onChainData.source !== 'coingecko_fallback';
      logger.info(`Signal generation data sources: onchain=${dataSource}, ml=${mlSource}`, {
        symbol: requestParams.symbol,
        onchain_score: isRealData ? 'real_data' : 'fallback_data',
        ml_confidence: mlResults.ml_confidence
      });
      
      // Enhance with on-chain specific insights
      const enhancedSignal = this.enhanceWithOnChainData(claudeAnalysis, onChainData);
      
      // Enhance with ML insights
      const mlEnhancedSignal = this.mlcService.enhanceSignalWithML(enhancedSignal, mlResults);
      
      return mlEnhancedSignal;
      
    } catch (error) {
      logger.error('Enhanced signal generation failed:', error);
      logger.info('Using fallback signal generation due to error');
      return this.generateFallbackSignal(marketData, technicalData, requestParams);
    }
  }

  async generateClaudeSignal(marketData, technicalData, onChainData, requestParams) {
    const prompt = this.buildAdvancedPrompt(marketData, technicalData, onChainData, requestParams);
    
    try {
      const message = await anthropic.messages.create({
        model: 'claude-sonnet-4-20250514',
        max_tokens: 4000,
        temperature: 0.3,
        messages: [
          {
            role: 'user',
            content: prompt
          }
        ]
      });
      let responseText = message.content[0].text;
      // Strip Markdown code block if present
      if (responseText.startsWith('```json')) {
        responseText = responseText.replace(/^```json\s*/, '').replace(/```\s*$/, '');
      } else if (responseText.startsWith('```')) {
        responseText = responseText.replace(/^```\s*/, '').replace(/```\s*$/, '');
      }
      return JSON.parse(responseText);
    } catch (error) {
      logger.error('Claude API call failed:', error);
      throw error;
    }
  }

  buildAdvancedPrompt(marketData, technicalData, onChainData, requestParams) {
    return `You are an institutional-grade cryptocurrency trading AI with access to both traditional technical analysis and advanced on-chain data. Generate a comprehensive trading signal.

MARKET DATA:
- Symbol: ${marketData.symbol}
- Current Price: $${marketData.current_price.toFixed(8)}
- 24h Change: ${marketData.price_change_24h.toFixed(2)}%
- Volume 24h: $${(marketData.volume_24h / 1e6).toFixed(2)}M
- Market Cap: $${(marketData.market_cap / 1e9).toFixed(2)}B
- Volume Ratio: ${technicalData.volume_ratio?.toFixed(2)}x
- Market Regime: ${technicalData.market_regime}

TECHNICAL INDICATORS:
- RSI: ${technicalData.rsi?.toFixed(2)}
- MACD: ${technicalData.macd?.macd?.toFixed(4)}
- MACD Signal: ${technicalData.macd?.signal?.toFixed(4)}
- MACD Histogram: ${technicalData.macd?.histogram?.toFixed(4)}
- SMA 20: $${technicalData.sma_20?.toFixed(8)}
- SMA 50: $${technicalData.sma_50?.toFixed(8)}
- EMA 12: $${technicalData.ema_12?.toFixed(8)}
- EMA 26: $${technicalData.ema_26?.toFixed(8)}
- Bollinger Upper: $${technicalData.bollinger_bands?.upper?.toFixed(8)}
- Bollinger Lower: $${technicalData.bollinger_bands?.lower?.toFixed(8)}
- Stochastic %K: ${technicalData.stochastic?.k?.toFixed(2)}
- ATR: ${technicalData.atr?.toFixed(8)}
- Volatility: ${(technicalData.volatility * 100)?.toFixed(2)}%
- Rate of Change: ${technicalData.rate_of_change?.toFixed(2)}%
- Momentum: ${technicalData.momentum?.toFixed(8)}

ON-CHAIN DATA (LunarCrush):
- Whale Activity: ${onChainData.whale_activity?.large_transfers_24h} large transfers, ${onChainData.whale_activity?.whale_accumulation} trend
- Network Health: ${onChainData.network_metrics?.active_addresses} active addresses, ${onChainData.network_metrics?.gas_usage_trend} gas trend
- DeFi Metrics: $${(onChainData.defi_metrics?.total_locked_value / 1e9)?.toFixed(2)}B TVL, ${onChainData.defi_metrics?.yield_farming_apy?.toFixed(2)}% APY
- Smart Money: ${onChainData.sentiment_indicators?.smart_money_flow} flow, ${onChainData.sentiment_indicators?.on_chain_sentiment} sentiment
- Funding Rates: ${(onChainData.sentiment_indicators?.derivative_metrics?.funding_rates * 100)?.toFixed(3)}%
- Cross-chain: ${onChainData.cross_chain_analysis?.arbitrage_opportunities ? 'Arbitrage available' : 'No arbitrage'}
- Risk Level: ${onChainData.risk_assessment?.market_manipulation_risk}, Liquidity: ${onChainData.risk_assessment?.liquidity_score?.toFixed(0)}/100

REQUEST PARAMETERS:
- Analysis Depth: ${requestParams.analysis_depth}
- Risk Level: ${requestParams.risk_level}
- Timeframe: ${requestParams.timeframe}

Generate a comprehensive trading signal with this EXACT JSON structure:

{
  "signal": "BUY" | "SELL" | "HOLD",
  "confidence": 0-100,
  "strength": "WEAK" | "MODERATE" | "STRONG" | "VERY_STRONG",
  "timeframe": "SCALP" | "INTRADAY" | "SWING" | "POSITION",
  "entry_price": number,
  "stop_loss": number,
  "take_profit_1": number,
  "take_profit_2": number,
  "take_profit_3": number,
  "risk_reward_ratio": number,
  "position_size_percent": 1-25,
  "market_sentiment": "BULLISH" | "BEARISH" | "NEUTRAL",
  "volatility_rating": "LOW" | "MEDIUM" | "HIGH" | "EXTREME",
  "key_levels": {
    "support": [number, number],
    "resistance": [number, number]
  },
  "technical_score": 0-100,
  "momentum_score": 0-100,
  "trend_score": 0-100,
  "onchain_score": 0-100,
  "volume_confirmation": true | false,
  "whale_influence": "POSITIVE" | "NEGATIVE" | "NEUTRAL",
  "defi_impact": "SUPPORTIVE" | "BEARISH" | "NEUTRAL",
  "cross_chain_factor": "BULLISH" | "BEARISH" | "NEUTRAL",
  "risk_factors": ["factor1", "factor2", "factor3"],
  "catalysts": ["catalyst1", "catalyst2"],
  "time_horizon_hours": number,
  "max_drawdown_percent": number,
  "probability_success": 0-100,
  "reasoning": "comprehensive analysis combining technical and on-chain factors",
  "next_review_timestamp": timestamp_number,
  "confluence_factors": number,
  "market_structure": "TRENDING" | "RANGING" | "BREAKOUT" | "REVERSAL",
  "institutional_flow": "BUYING" | "SELLING" | "NEUTRAL",
  "liquidity_analysis": {
    "depth_score": 0-100,
    "slippage_risk": "LOW" | "MEDIUM" | "HIGH",
    "execution_feasibility": "EXCELLENT" | "GOOD" | "FAIR" | "POOR"
  }
}

IMPORTANT: Weight on-chain data heavily in your analysis. Whale accumulation, smart money flows, and DeFi metrics often predict price movements before technical indicators. Consider the full market context including cross-chain dynamics.

Respond with ONLY the JSON object. No additional text.`;
  }

  enhanceWithOnChainData(claudeSignal, onChainData) {
    // Apply on-chain data adjustments to Claude's signal
    let adjustedConfidence = claudeSignal.confidence;
    
    // Whale activity adjustments
    if (onChainData.whale_activity?.whale_accumulation === 'buying' && claudeSignal.signal === 'BUY') {
      adjustedConfidence = Math.min(100, adjustedConfidence + 10);
    } else if (onChainData.whale_activity?.whale_accumulation === 'selling' && claudeSignal.signal === 'SELL') {
      adjustedConfidence = Math.min(100, adjustedConfidence + 10);
    } else if (onChainData.whale_activity?.whale_accumulation !== 'neutral' && 
               ((onChainData.whale_activity?.whale_accumulation === 'buying' && claudeSignal.signal === 'SELL') ||
                (onChainData.whale_activity?.whale_accumulation === 'selling' && claudeSignal.signal === 'BUY'))) {
      adjustedConfidence = Math.max(10, adjustedConfidence - 15);
    }

    // Smart money flow adjustments
    if (onChainData.sentiment_indicators?.smart_money_flow === 'inflow' && claudeSignal.signal === 'BUY') {
      adjustedConfidence = Math.min(100, adjustedConfidence + 8);
    } else if (onChainData.sentiment_indicators?.smart_money_flow === 'outflow' && claudeSignal.signal === 'SELL') {
      adjustedConfidence = Math.min(100, adjustedConfidence + 8);
    }

    // Risk assessment adjustments
    if (onChainData.risk_assessment?.market_manipulation_risk === 'high') {
      adjustedConfidence = Math.max(20, adjustedConfidence - 20);
    } else if (onChainData.risk_assessment?.liquidity_score < 30) {
      adjustedConfidence = Math.max(15, adjustedConfidence - 15);
    }

    return {
      ...claudeSignal,
      confidence: Math.round(adjustedConfidence),
      onchain_score: this.calculateOnChainScore(onChainData),
      whale_influence: this.determineWhaleInfluence(onChainData),
      defi_impact: this.determineDeFiImpact(onChainData),
      cross_chain_factor: this.determineCrossChainFactor(onChainData),
      enhanced_by: 'lunarcrush_integration'
    };
  }

  calculateOnChainScore(onChainData) {
    let score = 50; // Base neutral score
    
    // Whale activity scoring
    if (onChainData.whale_activity?.whale_accumulation === 'buying') score += 15;
    else if (onChainData.whale_activity?.whale_accumulation === 'selling') score -= 15;
    
    // Smart money flow scoring
    if (onChainData.sentiment_indicators?.smart_money_flow === 'inflow') score += 10;
    else if (onChainData.sentiment_indicators?.smart_money_flow === 'outflow') score -= 10;
    
    // Network health scoring
    if (onChainData.network_metrics?.gas_usage_trend === 'increasing') score += 5;
    else if (onChainData.network_metrics?.gas_usage_trend === 'decreasing') score -= 5;
    
    // Risk adjustments
    if (onChainData.risk_assessment?.market_manipulation_risk === 'high') score -= 20;
    else if (onChainData.risk_assessment?.market_manipulation_risk === 'low') score += 10;
    
    // Liquidity adjustments
    if (onChainData.risk_assessment?.liquidity_score > 80) score += 10;
    else if (onChainData.risk_assessment?.liquidity_score < 30) score -= 15;
    
    return Math.max(0, Math.min(100, score));
  }

  determineWhaleInfluence(onChainData) {
    const accumulation = onChainData.whale_activity?.whale_accumulation;
    const transfers = onChainData.whale_activity?.large_transfers_24h || 0;
    
    if (accumulation === 'buying' && transfers > 20) return 'POSITIVE';
    if (accumulation === 'selling' && transfers > 20) return 'NEGATIVE';
    return 'NEUTRAL';
  }

  determineDeFiImpact(onChainData) {
    const inflows = onChainData.defi_metrics?.protocol_inflows || 0;
    const apy = onChainData.defi_metrics?.yield_farming_apy || 0;
    
    if (inflows > 0 && apy > 10) return 'SUPPORTIVE';
    if (inflows < -50000000) return 'BEARISH';
    return 'NEUTRAL';
  }

  determineCrossChainFactor(onChainData) {
    const arbitrage = onChainData.cross_chain_analysis?.arbitrage_opportunities;
    const bridgeVolumes = onChainData.cross_chain_analysis?.bridge_volumes || 0;
    
    if (arbitrage && bridgeVolumes > 100000000) return 'BULLISH';
    if (bridgeVolumes < 10000000) return 'BEARISH';
    return 'NEUTRAL';
  }

  generateFallbackSignal(marketData, technicalData, requestParams) {
    // Fallback when both AI services fail
    const price = marketData.current_price;
    const rsi = technicalData.rsi || 50;
    const macd = technicalData.macd?.macd || 0;
    const volatility = technicalData.volatility || 0.02;
    
    let signal = 'HOLD';
    let confidence = 50;
    let reasoning = 'Fallback analysis - limited data available';
    
    // Add randomization to make fallback signals more varied
    const confidenceVariation = (Math.random() - 0.5) * 20; // ±10 points variation
    
    if (rsi < 30 && macd > 0) {
      signal = 'BUY';
      confidence = 65 + confidenceVariation;
      reasoning = 'Oversold RSI with positive MACD momentum';
    } else if (rsi > 70 && macd < 0) {
      signal = 'SELL';
      confidence = 65 + confidenceVariation;
      reasoning = 'Overbought RSI with negative MACD momentum';
    } else {
      // Add some randomization to HOLD signals too
      confidence = 50 + confidenceVariation;
    }
    
    // Ensure confidence stays within reasonable bounds
    confidence = Math.max(20, Math.min(85, confidence));

    const stopLossDistance = price * (volatility * 1.5);
    const takeProfitDistance = stopLossDistance * (2 + Math.random());

    // Calculate proper stop loss and take profit based on signal
    let stopLoss, takeProfit1, takeProfit2, takeProfit3;
    
    if (signal === 'BUY') {
      stopLoss = price - stopLossDistance;
      takeProfit1 = price + takeProfitDistance * 0.6;
      takeProfit2 = price + takeProfitDistance;
      takeProfit3 = price + takeProfitDistance * 1.5;
    } else if (signal === 'SELL') {
      stopLoss = price + stopLossDistance;
      takeProfit1 = price - takeProfitDistance * 0.6;
      takeProfit2 = price - takeProfitDistance;
      takeProfit3 = price - takeProfitDistance * 1.5;
    } else {
      // HOLD signal - set reasonable levels for potential breakout
      stopLoss = price - stopLossDistance * 0.8;
      takeProfit1 = price + takeProfitDistance * 0.4;
      takeProfit2 = price + takeProfitDistance * 0.8;
      takeProfit3 = price + takeProfitDistance * 1.2;
    }

    return {
      signal,
      confidence: Math.round(confidence),
      strength: confidence > 70 ? 'STRONG' : confidence > 55 ? 'MODERATE' : 'WEAK',
      timeframe: this.mapTimeframe(requestParams.timeframe),
      entry_price: price,
      stop_loss: stopLoss,
      take_profit_1: takeProfit1,
      take_profit_2: takeProfit2,
      take_profit_3: takeProfit3,
      risk_reward_ratio: takeProfitDistance / stopLossDistance,
      position_size_percent: this.calculatePositionSize(confidence, volatility, requestParams.risk_level),
      market_sentiment: signal === 'BUY' ? 'BULLISH' : signal === 'SELL' ? 'BEARISH' : 'NEUTRAL',
      volatility_rating: volatility < 0.02 ? 'LOW' : volatility < 0.04 ? 'MEDIUM' : volatility < 0.06 ? 'HIGH' : 'EXTREME',
      technical_score: Math.round(50 + (confidence - 50) * 0.8),
      momentum_score: Math.round(50 + (macd * 1000)),
      trend_score: Math.round(50 + ((price - technicalData.sma_20) / technicalData.sma_20) * 200),
      onchain_score: 50 + Math.round((Math.random() - 0.5) * 20), // Add variation to onchain score
      reasoning,
      source: 'fallback_analysis',
      timestamp: Date.now()
    };
  }

  mapTimeframe(timeframe) {
    const mapping = {
      '1m': 'SCALP',
      '5m': 'SCALP',
      '15m': 'INTRADAY',
      '1h': 'INTRADAY',
      '4h': 'SWING',
      '1d': 'POSITION'
    };
    return mapping[timeframe] || 'SWING';
  }

  calculatePositionSize(confidence, volatility, riskLevel) {
    const baseSize = {
      'conservative': 2,
      'moderate': 5,
      'aggressive': 10
    };
    
    const base = baseSize[riskLevel] || 5;
    const confidenceMultiplier = confidence / 100;
    const volatilityAdjustment = 1 / (1 + volatility * 10);
    
    return Math.max(1, Math.min(25, Math.round(base * confidenceMultiplier * volatilityAdjustment)));
  }
}

// ===============================================
// AUTHENTICATION MIDDLEWARE
// ===============================================

const authenticateAPI = (req, res, next) => {
  const authHeader = req.headers.authorization;
  
  if (!authHeader || !authHeader.startsWith('Bearer ')) {
    return res.status(401).json({
      success: false,
      error: 'Missing or invalid authorization header',
      code: 'AUTH_REQUIRED'
    });
  }
  
  const token = authHeader.substring(7);
  
  if (token !== process.env.API_KEY_SECRET) {
    return res.status(401).json({
      success: false,
      error: 'Invalid API key',
      code: 'INVALID_TOKEN'
    });
  }
  
  next();
};

// ===============================================
// VALIDATION MIDDLEWARE
// ===============================================

const validateSignalRequest = async (req, res, next) => {
  const { symbol, timeframe, analysis_depth, risk_level } = req.body;
  
  const validTimeframes = ['1m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '12h', '1d'];
  const validAnalysisDepths = ['basic', 'advanced', 'comprehensive'];
  const validRiskLevels = ['conservative', 'moderate', 'aggressive'];
  
  const errors = [];
  
  // Validate symbol dynamically
  if (!symbol) {
    errors.push('Symbol is required');
  } else {
    const isValid = await isValidSymbol(symbol);
    if (!isValid) {
      errors.push(`Invalid symbol: ${symbol}. Must be an active USDT trading pair on Binance.`);
    }
  }
  
  if (!timeframe || !validTimeframes.includes(timeframe)) {
    errors.push(`Invalid timeframe. Must be one of: ${validTimeframes.join(', ')}`);
  }
  
  if (analysis_depth && !validAnalysisDepths.includes(analysis_depth)) {
    errors.push(`Invalid analysis_depth. Must be one of: ${validAnalysisDepths.join(', ')}`);
  }
  
  if (risk_level && !validRiskLevels.includes(risk_level)) {
    errors.push(`Invalid risk_level. Must be one of: ${validRiskLevels.join(', ')}`);
  }
  
  if (errors.length > 0) {
    const validSymbols = await getValidSymbols();
    symbolStats.validationAttempts++;
    symbolStats.validationFailures++;
    
    // Track failed symbols
    if (symbol && !(await isValidSymbol(symbol))) {
      symbolStats.lastFailedSymbols.push({
        symbol: symbol,
        timestamp: new Date().toISOString()
      });
      
      // Keep only last 10 failed symbols
      if (symbolStats.lastFailedSymbols.length > 10) {
        symbolStats.lastFailedSymbols.shift();
      }
    }
    
    return res.status(400).json({
      success: false,
      error: 'Validation failed',
      details: errors,
      supported_symbols_count: validSymbols.length,
      note: 'Symbols are dynamically fetched from Binance. Any active USDT spot trading pair is supported.',
      hint: 'Use GET /api/v1/symbols to get the list of valid symbols'
    });
  }
  
  symbolStats.validationAttempts++;
  next();
};

// Batch validation middleware
const validateBatchSignalRequest = async (req, res, next) => {
  const { symbols, timeframe, analysis_depth, risk_level } = req.body;
  
  const validTimeframes = ['1m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '12h', '1d'];
  const validAnalysisDepths = ['basic', 'advanced', 'comprehensive'];
  const validRiskLevels = ['conservative', 'moderate', 'aggressive'];
  
  const errors = [];
  
  // Validate symbols array
  if (!symbols || !Array.isArray(symbols) || symbols.length === 0) {
    errors.push('Symbols array is required');
  } else if (symbols.length > 10) {
    errors.push('Maximum 10 symbols allowed per batch request');
  } else {
    // Validate each symbol
    const validSymbolsList = await getValidSymbols();
    const invalidSymbols = symbols.filter(symbol => !validSymbolsList.includes(symbol));
    
    if (invalidSymbols.length > 0) {
      errors.push(`Invalid symbols: ${invalidSymbols.join(', ')}`);
    }
  }
  
  if (!timeframe || !validTimeframes.includes(timeframe)) {
    errors.push(`Invalid timeframe. Must be one of: ${validTimeframes.join(', ')}`);
  }
  
  if (analysis_depth && !validAnalysisDepths.includes(analysis_depth)) {
    errors.push(`Invalid analysis_depth. Must be one of: ${validAnalysisDepths.join(', ')}`);
  }
  
  if (risk_level && !validRiskLevels.includes(risk_level)) {
    errors.push(`Invalid risk_level. Must be one of: ${validRiskLevels.join(', ')}`);
  }
  
  if (errors.length > 0) {
    return res.status(400).json({
      success: false,
      error: 'Validation failed',
      details: errors
    });
  }
  
  next();
};

// ===============================================
// API ROUTES
// ===============================================

// Health check endpoint
app.get('/api/health', (req, res) => {
  res.json({
    success: true,
    status: 'healthy',
    timestamp: Date.now(),
    version: '2.0.0',
    ai_services: {
      claude: process.env.CLAUDE_API_KEY ? 'configured' : 'missing',
      lunarcrush: process.env.LUNARCRUSH_API_KEY ? 'configured' : 'missing'
    },
    symbol_validation: {
      mode: 'dynamic',
      source: 'binance_api',
      cache_status: validSymbolsCache.symbols.length > 0 ? 'populated' : 'empty',
      last_updated: validSymbolsCache.lastUpdated || 'never'
    },
    uptime: process.uptime()
  });
});

// Symbol list endpoint
app.get('/api/v1/symbols', authenticateAPI, async (req, res) => {
  try {
    const symbols = await getValidSymbols();
    
    res.json({
      success: true,
      data: {
        symbols: symbols,
        count: symbols.length,
        top_by_volume: getTopSymbolsByVolume(20),
        stablecoins: validSymbolsCache.stablecoins,
        last_updated: validSymbolsCache.lastUpdated,
        update_interval: validSymbolsCache.updateInterval,
        cache_expires_in: Math.max(0, validSymbolsCache.updateInterval - (Date.now() - validSymbolsCache.lastUpdated))
      }
    });
  } catch (error) {
    logger.error('Failed to get symbols:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to retrieve symbol list'
    });
  }
});

// Symbol search endpoint
app.get('/api/v1/symbols/search', authenticateAPI, async (req, res) => {
  try {
    const { query } = req.query;
    
    if (!query || query.length < 1) {
      return res.status(400).json({
        success: false,
        error: 'Query must be at least 1 character'
      });
    }
    
    const symbols = await getValidSymbols();
    
    // Filter symbols that contain the query (case insensitive)
    const matches = symbols.filter(symbol => 
      symbol.toLowerCase().includes(query.toLowerCase())
    );
    
    // Sort by relevance (exact match first, then by length)
    matches.sort((a, b) => {
      const aExact = a.toLowerCase() === query.toLowerCase();
      const bExact = b.toLowerCase() === query.toLowerCase();
      if (aExact && !bExact) return -1;
      if (!aExact && bExact) return 1;
      return a.length - b.length;
    });
    
    res.json({
      success: true,
      data: {
        query: query,
        matches: matches.slice(0, 50), // Limit to 50 results
        count: matches.length,
        metadata: matches.slice(0, 10).reduce((acc, symbol) => {
          acc[symbol] = getSymbolMetadata(symbol);
          return acc;
        }, {})
      }
    });
  } catch (error) {
    logger.error('Symbol search failed:', error);
    res.status(500).json({
      success: false,
      error: 'Search failed'
    });
  }
});

// Force symbol refresh endpoint (admin)
app.post('/api/v1/admin/refresh-symbols', authenticateAPI, async (req, res) => {
  try {
    logger.info('Manual symbol refresh requested');
    const symbols = await updateValidSymbols(true);
    
    res.json({
      success: true,
      message: 'Symbols refreshed successfully',
      count: symbols.length,
      new_symbols: symbols.filter(s => !validSymbolsCache.symbols.includes(s)),
      removed_symbols: validSymbolsCache.symbols.filter(s => !symbols.includes(s))
    });
  } catch (error) {
    logger.error('Manual symbol refresh failed:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to refresh symbols',
      message: error.message
    });
  }
});

// Symbol statistics endpoint (admin)
app.get('/api/v1/admin/symbol-stats', authenticateAPI, async (req, res) => {
  const validSymbols = await getValidSymbols();
  
  res.json({
    success: true,
    data: {
      total_valid_symbols: validSymbols.length,
      cache_last_updated: new Date(validSymbolsCache.lastUpdated).toISOString(),
      cache_age_seconds: Math.floor((Date.now() - validSymbolsCache.lastUpdated) / 1000),
      validation_stats: symbolStats,
      top_symbols_by_volume: getTopSymbolsByVolume(10),
      stablecoin_count: validSymbolsCache.stablecoins.length,
      sample_symbols: validSymbols.slice(0, 20),
      cache_status: {
        is_updating: validSymbolsCache.isUpdating,
        update_interval_ms: validSymbolsCache.updateInterval,
        next_update_in_seconds: Math.max(0, 
          Math.floor((validSymbolsCache.updateInterval - (Date.now() - validSymbolsCache.lastUpdated)) / 1000)
        )
      }
    }
  });
});

// Enhanced signal generation endpoint
app.post('/api/v1/signals/generate', authenticateAPI, validateSignalRequest, async (req, res) => {
  const startTime = Date.now();
  const requestId = `req_${startTime}_${Math.random().toString(36).substr(2, 9)}`;
  
  try {
    const {
      symbol,
      timeframe = '1h',
      analysis_depth = 'comprehensive',
      risk_level = 'moderate',
      bars = 100,
      wallet_address = null
    } = req.body;

    logger.info(`Enhanced signal generation request: ${requestId}`, { 
      symbol, timeframe, analysis_depth, risk_level 
    });

    // Generate enhanced market data with volumes
    const marketData = MarketDataService.generateEnhancedData(symbol, timeframe, bars);
    
    // Calculate comprehensive technical indicators
    const technicalData = TechnicalAnalysis.calculateAdvancedMetrics(
      marketData.price_history, 
      marketData.volume_history
    );
    
    // Initialize enhanced AI signal generator
    const aiGenerator = new EnhancedAISignalGenerator();
    
    // Get on-chain analysis from LunarCrush
    logger.info(`Fetching on-chain data for ${symbol}`, { requestId });
    const onChainData = await aiGenerator.coinGeckoService.getOnChainAnalysis(symbol, wallet_address);
    
    // Generate advanced AI signal combining Claude + LunarCrush
    logger.info(`Generating enhanced AI signal`, { requestId });
    const aiSignal = await aiGenerator.generateAdvancedSignal(marketData, technicalData, onChainData, {
      timeframe,
      analysis_depth,
      risk_level
    });
    
    const processingTime = Date.now() - startTime;
    
    // Build comprehensive response
    const response = {
      success: true,
      request_id: requestId,
      timestamp: Date.now(),
      data: {
        ...aiSignal,
        market_data: {
          symbol: marketData.symbol,
          price: marketData.current_price,
          volume_24h: marketData.volume_24h,
          price_change_24h: marketData.price_change_24h,
          market_cap: marketData.market_cap,
          timeframe: marketData.timeframe
        },
        technical_indicators: {
          rsi: technicalData.rsi,
          macd: technicalData.macd,
          sma_20: technicalData.sma_20,
          sma_50: technicalData.sma_50,
          ema_12: technicalData.ema_12,
          ema_26: technicalData.ema_26,
          bollinger_bands: technicalData.bollinger_bands,
          stochastic: technicalData.stochastic,
          atr: technicalData.atr,
          volatility: technicalData.volatility,
          volume_ratio: technicalData.volume_ratio,
          market_regime: technicalData.market_regime
        },
        onchain_analysis: {
          ...onChainData,
          data_source: onChainData.source
        },
        metadata: {
          processing_time_ms: processingTime,
          ai_models: ['claude-4-sonnet', 'lunarcrush-api'],
          data_sources: ['technical_analysis', 'onchain_data', 'market_data'],
          analysis_depth,
          risk_level,
          confidence_factors: aiSignal.confluence_factors || 8,
          symbol_metadata: getSymbolMetadata(symbol)
        }
      }
    };
    
    logger.info(`Enhanced signal generated successfully: ${requestId}`, {
      signal: aiSignal.signal,
      confidence: aiSignal.confidence,
      onchain_score: aiSignal.onchain_score,
      processing_time: processingTime
    });
    
    res.json(response);
    
  } catch (error) {
    logger.error(`Enhanced signal generation failed: ${requestId}`, {
      error: error.message,
      stack: error.stack,
      symbol: req.body.symbol
    });
    
    res.status(500).json({
      success: false,
      request_id: requestId,
      error: 'Internal server error',
      message: 'Signal generation failed',
      debug: process.env.NODE_ENV === 'development' ? {
        error: error.message,
        stack: error.stack
      } : undefined,
      timestamp: Date.now()
    });
  }
});

// Batch signal generation with AI enhancement
app.post('/api/v1/signals/batch', authenticateAPI, validateBatchSignalRequest, async (req, res) => {
  const startTime = Date.now();
  const requestId = `batch_${startTime}_${Math.random().toString(36).substr(2, 9)}`;
  
  const { symbols, timeframe = '1h', analysis_depth = 'advanced', risk_level = 'moderate' } = req.body;
  
  logger.info(`Batch signal request: ${requestId}`, {
    symbols_count: symbols.length,
    timeframe,
    analysis_depth,
    risk_level
  });
  
  const results = [];
  const aiGenerator = new EnhancedAISignalGenerator();
  
  // Process symbols in parallel with concurrency limit
  const concurrencyLimit = 3;
  const symbolChunks = [];
  
  for (let i = 0; i < symbols.length; i += concurrencyLimit) {
    symbolChunks.push(symbols.slice(i, i + concurrencyLimit));
  }
  
  for (const chunk of symbolChunks) {
    const chunkPromises = chunk.map(async (symbol) => {
      try {
        const marketData = MarketDataService.generateEnhancedData(symbol, timeframe);
        const technicalData = TechnicalAnalysis.calculateAdvancedMetrics(
          marketData.price_history,
          marketData.volume_history
        );
        const onChainData = await aiGenerator.coinGeckoService.getOnChainAnalysis(symbol);
        const aiSignal = await aiGenerator.generateAdvancedSignal(marketData, technicalData, onChainData, {
          timeframe, analysis_depth, risk_level
        });
        
        return {
          symbol,
          success: true,
          signal: aiSignal.signal,
          confidence: aiSignal.confidence,
          onchain_score: aiSignal.onchain_score,
          entry_price: aiSignal.entry_price,
          whale_influence: aiSignal.whale_influence,
          stop_loss: aiSignal.stop_loss,
          take_profit_1: aiSignal.take_profit_1
        };
      } catch (error) {
        logger.error(`Batch signal failed for ${symbol}:`, error);
        return {
          symbol,
          success: false,
          error: error.message
        };
      }
    });
    
    const chunkResults = await Promise.all(chunkPromises);
    results.push(...chunkResults);
  }
  
  const processingTime = Date.now() - startTime;
  const successCount = results.filter(r => r.success).length;
  
  logger.info(`Batch signal completed: ${requestId}`, {
    total_symbols: symbols.length,
    successful: successCount,
    failed: symbols.length - successCount,
    processing_time_ms: processingTime
  });
  
  res.json({
    success: true,
    request_id: requestId,
    timestamp: Date.now(),
    ai_models: ['claude-4-sonnet', 'lunarcrush-api'],
    processing_time_ms: processingTime,
    summary: {
      total: symbols.length,
      successful: successCount,
      failed: symbols.length - successCount
    },
    results
  });
});

// API documentation endpoint (enhanced)
app.get('/api/docs', (req, res) => {
  res.json({
    name: 'Enhanced Crypto Signal API',
    version: '2.0.0',
    description: 'AI-powered cryptocurrency trading signal generation with Claude + LunarCrush integration',
    symbol_validation: {
      mode: 'dynamic',
      source: 'binance_api',
      update_frequency: 'hourly',
      supported_count: validSymbolsCache.symbols.length || 'Loading...'
    },
    ai_models: {
      claude: 'Claude 4 Sonnet for general market analysis and pattern recognition',
      lunarcrush: 'LunarCrush API for social sentiment, market metrics, and on-chain data analysis'
    },
    endpoints: {
      'GET /api/v1/symbols': {
        description: 'Get list of all valid trading symbols',
        auth_required: true,
        response: {
          symbols: 'Array of valid USDT trading pairs',
          count: 'Total number of symbols',
          top_by_volume: 'Top symbols sorted by 24h volume',
          stablecoins: 'List of stablecoin pairs'
        }
      },
      'GET /api/v1/symbols/search': {
        description: 'Search for symbols by query',
        auth_required: true,
        parameters: {
          query: 'Search query (min 1 character)'
        }
      },
      'POST /api/v1/signals/generate': {
        description: 'Generate enhanced trading signal with AI analysis',
        auth_required: true,
        parameters: {
          symbol: 'String - Trading pair (dynamically validated)',
          timeframe: 'String - Time period (1m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 12h, 1d)',
          analysis_depth: 'String - Analysis level (basic, advanced, comprehensive)',
          risk_level: 'String - Risk tolerance (conservative, moderate, aggressive)',
          wallet_address: 'String - Optional wallet address for personalized analysis'
        },
        features: [
          'Real-time on-chain whale tracking',
          'Multi-chain DeFi analysis',
          'Smart money flow detection',
          'Cross-chain arbitrage opportunities',
          'Advanced technical analysis',
          'AI-powered pattern recognition',
          'Dynamic symbol validation'
        ],
        rate_limit: '100 requests/hour'
      },
      'POST /api/v1/signals/batch': {
        description: 'Generate signals for multiple symbols simultaneously',
        max_symbols: 10,
        concurrency: 3
      },
      'POST /api/v1/admin/refresh-symbols': {
        description: 'Force refresh the symbol cache (admin only)',
        auth_required: true
      },
      'GET /api/v1/admin/symbol-stats': {
        description: 'Get symbol validation statistics (admin only)',
        auth_required: true
      }
    },
    data_sources: [
      'Real-time blockchain data (2,500+ EVM chains)',
      'Whale wallet movements',
      'DeFi protocol metrics',
      'Cross-chain bridge volumes',
      'Technical indicators',
      'Market microstructure',
      'Dynamic Binance symbol list'
    ],
    support: 'api@yourcompany.com'
  });
});

// 404 handler
app.use('*', (req, res) => {
  res.status(404).json({
    success: false,
    error: 'Endpoint not found',
    available_endpoints: [
      'GET /api/health',
      'GET /api/docs',
      'GET /api/v1/symbols',
      'GET /api/v1/symbols/search',
      'POST /api/v1/signals/generate',
      'POST /api/v1/signals/batch',
      'POST /api/v1/admin/refresh-symbols',
      'GET /api/v1/admin/symbol-stats'
    ]
  });
});

// Error handling middleware
app.use((error, req, res, next) => {
  logger.error('Unhandled error:', error);
  
  res.status(500).json({
    success: false,
    error: 'Internal server error',
    message: process.env.NODE_ENV === 'production' ? 'Something went wrong' : error.message,
    request_id: `error_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`
  });
});

// ===============================================
// SERVER STARTUP
// ===============================================

// Initialize symbols on startup
async function initializeSymbols() {
  try {
    logger.info('Initializing symbol validation...');
    const symbols = await updateValidSymbols(true);
    logger.info(`Symbol validation initialized with ${symbols.length} valid pairs`);
  } catch (error) {
    logger.error('Failed to initialize symbols:', error);
    logger.warn('API will start with fallback symbols and retry in background');
  }
}

// Main startup function
async function startServer() {
  try {
    // Run startup diagnostics
    await logStartupStatus();
    
    // Start the server
    const server = app.listen(PORT, () => {
      logger.info(`Enhanced Crypto Signal API server running on port ${PORT}`);
      logger.info(`AI Models: Claude 4 Sonnet + LunarCrush API`);
      logger.info(`Blockchain Coverage: 2,500+ EVM Networks`);
      logger.info(`Symbol Validation: Dynamic (Binance API)`);
      logger.info(`Environment: ${process.env.NODE_ENV || 'development'}`);
      logger.info(`Health check: http://localhost:${PORT}/api/health`);
      logger.info(`Documentation: http://localhost:${PORT}/api/docs`);
      
      // Initialize symbols after server starts
      initializeSymbols();
      
      // Set up periodic symbol updates
      setInterval(async () => {
        try {
          await updateValidSymbols();
          logger.info('Periodic symbol update completed');
        } catch (error) {
          logger.error('Periodic symbol update failed:', error);
        }
      }, 3600000); // Update every hour
    });
    
    // Graceful shutdown
    process.on('SIGTERM', () => {
      logger.info('SIGTERM received, shutting down gracefully...');
      server.close(() => {
        logger.info('Server closed');
        process.exit(0);
      });
    });
    
    process.on('SIGINT', () => {
      logger.info('SIGINT received, shutting down gracefully...');
      server.close(() => {
        logger.info('Server closed');
        process.exit(0);
      });
    });
    
  } catch (error) {
    logger.error('Failed to start server:', error);
    process.exit(1);
  }
}

// Start the server
startServer();

module.exports = app;