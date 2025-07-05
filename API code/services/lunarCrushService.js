const axios = require('axios');

// Use the configured logger from the main application
let logger;

// Initialize logger when the service is loaded
try {
  const winston = require('winston');
  logger = winston.createLogger({
    level: 'info',
    format: winston.format.combine(
      winston.format.timestamp(),
      winston.format.json()
    ),
    transports: [
      new winston.transports.Console(),
      new winston.transports.File({ filename: 'logs/lunarCrush.log' })
    ]
  });
} catch (error) {
  // Fallback to console if winston is not available
  logger = {
    info: (msg, data) => console.log(`[INFO] ${msg}`, data),
    warn: (msg, data) => console.warn(`[WARN] ${msg}`, data),
    error: (msg, data) => console.error(`[ERROR] ${msg}`, data)
  };
}

class CoinGeckoService {
  constructor() {
    this.baseURL = 'https://api.coingecko.com/api/v3';
    this.cache = new Map();
    this.cacheTimeout = 15 * 60 * 1000; // 15 minutes cache (longer to reduce API calls)
    this.requestQueue = [];
    this.isProcessing = false;
    this.lastRequestTime = 0;
    this.minRequestInterval = 5 * 60 * 1000; // 5 minutes between requests (for 2 per 10 minutes)
    this.fallbackOnly = process.env.COINGECKO_FALLBACK_ONLY === 'true';
    this.rateLimitHit = false;
    this.dailyRequestCount = 0;
    this.lastResetTime = Date.now();
    this.maxDailyRequests = 288; // 288 requests per day (12 per hour × 24 hours)
    this.hourlyRequestCount = 0;
    this.lastHourReset = Date.now();
    this.maxHourlyRequests = 12; // 12 requests per hour (2 per 10 minutes)
  }

  // Queue system for rate limiting
  async queueRequest(requestFn) {
    return new Promise((resolve, reject) => {
      this.requestQueue.push({ requestFn, resolve, reject });
      logger.info(`Request queued. Queue length: ${this.requestQueue.length}`);
      this.processQueue();
    });
  }

  // Check if we should use fallback based on rate limits
  shouldUseFallback() {
    const now = Date.now();
    
    // Reset daily count if 24 hours have passed
    if (now - this.lastResetTime > 24 * 60 * 60 * 1000) {
      this.dailyRequestCount = 0;
      this.lastResetTime = now;
    }
    
    // Reset hourly count if 1 hour has passed
    if (now - this.lastHourReset > 60 * 60 * 1000) {
      this.hourlyRequestCount = 0;
      this.lastHourReset = now;
    }

    // Calculate time-based rate limiting (2 requests per 10 minutes)
    const timeSinceHourStart = now - this.lastHourReset;
    const tenMinuteWindows = Math.floor(timeSinceHourStart / (10 * 60 * 1000));
    const maxRequestsForCurrentTime = (tenMinuteWindows + 1) * 2; // 2 requests per 10-minute window, starting with 2
    
    // Use fallback if we're approaching any limits
    if (this.dailyRequestCount >= this.maxDailyRequests) {
      logger.info(`Daily request limit reached (${this.maxDailyRequests}), using fallback`);
      return true;
    }
    
    if (this.hourlyRequestCount >= this.maxHourlyRequests) {
      logger.info(`Hourly request limit reached (${this.maxHourlyRequests}), using fallback`);
      return true;
    }
    
    // Time-based rate limiting: only allow requests based on elapsed time
    if (this.hourlyRequestCount >= maxRequestsForCurrentTime) {
      logger.info(`Time-based rate limit: ${this.hourlyRequestCount} requests used, only ${maxRequestsForCurrentTime} allowed for current time period, using fallback`);
      return true;
    }

    return false;
  }

  async processQueue() {
    if (this.isProcessing || this.requestQueue.length === 0) {
      return;
    }

    this.isProcessing = true;
    logger.info(`Processing queue. Items in queue: ${this.requestQueue.length}`);

    while (this.requestQueue.length > 0) {
      const { requestFn, resolve, reject } = this.requestQueue.shift();
      
      try {
        // Check if we should use fallback
        if (this.shouldUseFallback()) {
          logger.info('Using fallback due to daily limits');
          resolve(await this.getFallbackData());
          continue;
        }

        // Ensure minimum interval between requests (5 minutes for 2 requests per 10 minutes)
        const now = Date.now();
        const timeSinceLastRequest = now - this.lastRequestTime;
        const minInterval = 5 * 60 * 1000; // 5 minutes between requests (2 per 10 minutes)
        
        if (timeSinceLastRequest < minInterval) {
          const delayTime = minInterval - timeSinceLastRequest;
          logger.info(`Rate limiting: waiting ${delayTime}ms before next request (5-minute interval)`);
          await this.delay(delayTime);
        }
        
        logger.info(`Making CoinGecko request. Queue remaining: ${this.requestQueue.length}`);
        const result = await requestFn();
        this.lastRequestTime = Date.now();
        this.dailyRequestCount++;
        this.hourlyRequestCount++;
        logger.info(`CoinGecko request successful. Queue remaining: ${this.requestQueue.length}, Daily: ${this.dailyRequestCount}/${this.maxDailyRequests}, Hourly: ${this.hourlyRequestCount}/${this.maxHourlyRequests}`);
        resolve(result);
      } catch (error) {
        logger.error(`CoinGecko request failed:`, error.message);
        reject(error);
      }
    }

    this.isProcessing = false;
    logger.info('Queue processing complete');
  }

  // Helper method to get fallback data
  async getFallbackData() {
    return {
      source: 'fallback',
      timestamp: new Date().toISOString(),
      data: this.getFallbackMarketData(),
      reason: 'daily_limit_reached'
    };
  }

  async getCoinData(symbol) {
    // If fallback-only mode is enabled or we've hit rate limits, use fallback immediately
    if (this.fallbackOnly || this.rateLimitHit) {
      logger.info(`Using fallback data for ${symbol} (fallback-only: ${this.fallbackOnly}, rate-limit-hit: ${this.rateLimitHit})`);
      return this.getFallbackCoinData(symbol);
    }

    return this.queueRequest(async () => {
      const cacheKey = `coin_${symbol}`;
      if (this.cache.has(cacheKey)) {
        const cached = this.cache.get(cacheKey);
        if (Date.now() - cached.timestamp < this.cacheTimeout) {
          return cached.data;
        }
      }
      try {
        // CoinGecko uses ids like 'bitcoin', 'ethereum', etc.
        const id = this.symbolToId(symbol);
        if (!id) {
          logger.warn(`No CoinGecko ID found for symbol: ${symbol}`);
          return this.getFallbackCoinData(symbol);
        }
        
        const url = `${this.baseURL}/coins/${id}`;
        const response = await axios.get(url, { 
          timeout: 10000,
          headers: {
            'Accept': 'application/json',
            'User-Agent': 'GenesisAI-TradingBot/1.0'
          }
        });
        
        if (response.status === 429) {
          logger.warn('CoinGecko rate limit hit, switching to fallback mode');
          this.rateLimitHit = true;
          return this.getFallbackCoinData(symbol);
        }
        
        console.log('CoinGecko getCoinData raw response:', response.data ? 'Data received' : 'Empty response');
        const data = this.parseCoinData(response.data);
        this.cache.set(cacheKey, { data, timestamp: Date.now() });
        return data;
      } catch (error) {
        if (error.response?.status === 429) {
          logger.warn('CoinGecko rate limit exceeded, switching to fallback mode');
          this.rateLimitHit = true;
        } else {
          logger.error('CoinGecko coin data request failed:', error?.response?.data || error?.message || error);
        }
        return this.getFallbackCoinData(symbol);
      }
    });
  }

  async getMarketMetrics(symbol) {
    // If fallback-only mode is enabled or we've hit rate limits, use fallback immediately
    if (this.fallbackOnly || this.rateLimitHit) {
      logger.info(`Using fallback market data for ${symbol} (fallback-only: ${this.fallbackOnly}, rate-limit-hit: ${this.rateLimitHit})`);
      return this.getFallbackMarketData();
    }

    return this.queueRequest(async () => {
      try {
        const coinId = this.symbolToId(symbol);
        if (!coinId) {
          logger.warn(`No CoinGecko ID found for symbol: ${symbol}`);
          return this.getFallbackMarketData();
        }

        const url = `https://api.coingecko.com/api/v3/coins/${coinId}`;
        
        const response = await axios.get(url, {
          timeout: 10000,
          headers: {
            'Accept': 'application/json',
            'User-Agent': 'GenesisAI-TradingBot/1.0'
          }
        });

        if (response.status === 429) {
          logger.warn('CoinGecko rate limit hit, switching to fallback mode');
          this.rateLimitHit = true;
          return this.getFallbackMarketData();
        }

        const data = response.data;
        
        // Log the raw response for debugging
        console.log(`CoinGecko getMarketMetrics raw response:`, data ? 'Data received' : 'Empty response');
        
        if (!data || !data.market_data) {
          logger.warn(`Invalid CoinGecko response for ${symbol}`);
          return this.getFallbackMarketData();
        }

        return {
          current_price: data.market_data.current_price?.usd || 0,
          market_cap: data.market_data.market_cap?.usd || 0,
          volume_24h: data.market_data.total_volume?.usd || 0,
          price_change_24h: data.market_data.price_change_24h || 0,
          price_change_percentage_24h: data.market_data.price_change_percentage_24h || 0,
          circulating_supply: data.market_data.circulating_supply || 0,
          total_supply: data.market_data.total_supply || 0,
          max_supply: data.market_data.max_supply || 0,
          ath: data.market_data.ath?.usd || 0,
          ath_change_percentage: data.market_data.ath_change_percentage?.usd || 0,
          atl: data.market_data.atl?.usd || 0,
          atl_change_percentage: data.market_data.atl_change_percentage?.usd || 0,
          last_updated: data.last_updated,
          source: 'coingecko'
        };
      } catch (error) {
        if (error.response?.status === 429) {
          logger.warn('CoinGecko rate limit exceeded, switching to fallback mode');
          this.rateLimitHit = true;
        } else {
          logger.error('CoinGecko market metrics request failed:', error.response?.data || error.message);
        }
        return this.getFallbackMarketData();
      }
    });
  }

  async getComprehensiveAnalysis(symbol, walletAddress = null) {
    try {
      // Use sequential requests instead of Promise.all to respect rate limiting
      const coinData = await this.getCoinData(symbol);
      const marketData = await this.getMarketMetrics(symbol);
      
      return {
        source: 'coingecko',
        timestamp: new Date().toISOString(),
        symbol: symbol,
        coin_data: coinData,
        market_metrics: marketData,
        sentiment_score: 0, // CoinGecko does not provide social sentiment
        market_sentiment: this.calculateMarketSentiment(marketData),
        whale_activity: {}, // Not available from CoinGecko
        social_velocity: {}, // Not available from CoinGecko
        risk_indicators: this.calculateRiskIndicators(coinData, {}, marketData),
        confidence_score: this.calculateConfidenceScore(coinData, {}, marketData)
      };
    } catch (error) {
      logger.error('Comprehensive CoinGecko analysis failed:', error?.response?.data || error?.message || error);
      return this.getFallbackComprehensiveData(symbol);
    }
  }

  symbolToId(symbol) {
    // Map common symbols to CoinGecko ids
    const map = {
      'BTCUSDT': 'bitcoin',
      'ETHUSDT': 'ethereum',
      'ADAUSDT': 'cardano',
      'SOLUSDT': 'solana',
      'XRPUSDT': 'ripple',
      'BNBUSDT': 'binancecoin',
      'DOGEUSDT': 'dogecoin',
      'MATICUSDT': 'matic-network',
      'DOTUSDT': 'polkadot',
      'LTCUSDT': 'litecoin',
      'USDCUSDT': 'usd-coin',
      'BCHUSDT': 'bitcoin-cash',
      'LINKUSDT': 'chainlink',
      'TRXUSDT': 'tron',
      'AVAXUSDT': 'avalanche-2',
      'SHIBUSDT': 'shiba-inu',
      'WBTCUSDT': 'wrapped-bitcoin',
      'UNIUSDT': 'uniswap',
      'XLMUSDT': 'stellar',
      'ATOMUSDT': 'cosmos',
      'ETCUSDT': 'ethereum-classic',
      'FILUSDT': 'filecoin',
      'APTUSDT': 'aptos',
      'ARBUSDT': 'arbitrum',
      'OPUSDT': 'optimism',
      'SUIUSDT': 'sui',
      'PEPEUSDT': 'pepe',
      'TUSDUSDT': 'true-usd',
      'DAIUSDT': 'dai',
      'FDUSDUSDT': 'first-digital-usd',
      'RNDRUSDT': 'render-token',
      'INJUSDT': 'injective-protocol',
      'BUSDUSDT': 'binance-usd',
      'AAVEUSDT': 'aave',
      'STETHUSDT': 'staked-ether',
      'LDOUSDT': 'lido-dao',
      'MKRUSDT': 'maker',
      'QNTUSDT': 'quant-network',
      'NEARUSDT': 'near',
      'GRTUSDT': 'the-graph',
      'ALGOUSDT': 'algorand',
      'EGLDUSDT': 'elrond-erd-2',
      'CRVUSDT': 'curve-dao-token',
      'SANDUSDT': 'the-sandbox',
      'AXSUSDT': 'axie-infinity',
      'IMXUSDT': 'immutable-x',
      'MANAUSDT': 'decentraland',
      'FTMUSDT': 'fantom',
      'XMRUSDT': 'monero',
      'HBARUSDT': 'hedera-hashgraph',
      'KAVAUSDT': 'kava',
      'RUNEUSDT': 'thorchain',
      'CROUSDT': 'crypto-com-chain',
      'MINAUSDT': 'mina-protocol',
      'GMXUSDT': 'gmx',
      'DYDXUSDT': 'dydx',
      'LUNCUSDT': 'terra-luna',
      'LUNAUSDT': 'terra-luna-2',
      'ONEUSDT': 'harmony',
      'ZECUSDT': 'zcash',
      'XEMUSDT': 'nem',
      'ONTUSDT': 'ontology',
      'ICXUSDT': 'icon',
      'QTUMUSDT': 'qtum',
      'ZENUSDT': 'horizen',
      'DASHUSDT': 'dash',
      'ENJUSDT': 'enjincoin',
      'YFIUSDT': 'yearn-finance',
      'COMPUSDT': 'compound-governance-token',
      'BATUSDT': 'basic-attention-token',
      'ZRXUSDT': '0x',
      'OMGUSDT': 'omisego',
      'BNTUSDT': 'bancor',
      'BALUSDT': 'balancer',
      'SRMUSDT': 'serum',
      'SUSHIUSDT': 'sushi',
      'RENUSDT': 'ren',
      'CVCUSDT': 'civic',
      'ANKRUSDT': 'ankr',
      'OCEANUSDT': 'ocean-protocol',
      'STMXUSDT': 'stormx',
      'CHRUSDT': 'chromia',
      'BANDUSDT': 'band-protocol',
      'ALICEUSDT': 'my-neighbor-alice',
      'CTSIUSDT': 'cartesi',
      'DGBUSDT': 'digibyte',
      'NKNUSDT': 'nkn',
      'DOCKUSDT': 'dock',
      'TWTUSDT': 'trust-wallet-token',
      'API3USDT': 'api3',
      'FETUSDT': 'fetch-ai',
      'AGIXUSDT': 'singularitynet',
      'GALAUSDT': 'gala',
      'SXPUSDT': 'solar',
      'BICOUSDT': 'biconomy',
      'IDUSDT': 'space-id',
      'JOEUSDT': 'joe',
      'LITUSDT': 'litentry',
      'MOVRUSDT': 'moonriver',
      'GLMRUSDT': 'moonbeam',
      'ASTRUSDT': 'astar',
      'ACAUSDT': 'acala',
      'KSMUSDT': 'kusama',
      'PHAUSDT': 'pha',
      'CFGUSDT': 'centrifuge',
      'BONDUSDT': 'barnbridge',
      'RAYUSDT': 'raydium',
      'PORTOUSDT': 'porto',
      'CITYUSDT': 'manchester-city-fan-token',
      'PSGUSDT': 'paris-saint-germain-fan-token',
      'JUVUSDT': 'juventus-fan-token',
      'ATMUSDT': 'atletico-madrid-fan-token',
      'ASRUSDT': 'as-roma-fan-token',
      'BARUSDT': 'fc-barcelona-fan-token',
      'OGUSDT': 'og-fan-token',
      'NMRUSDT': 'numeraire',
      'FORTHUSDT': 'ampleforth-governance-token',
      'MLNUSDT': 'melon',
      'RLCUSDT': 'iexec-rlc',
      'PAXGUSDT': 'pax-gold',
      'USDTUSDT': 'tether',
    };
    return map[symbol] || symbol.replace('USDT', '').toLowerCase();
  }

  parseCoinData(data) {
    if (!data || !data.market_data) return this.getFallbackCoinData();
    return {
      price: data.market_data.current_price.usd || 0,
      price_change_24h: data.market_data.price_change_percentage_24h || 0,
      volume_24h: data.market_data.total_volume.usd || 0,
      market_cap: data.market_data.market_cap.usd || 0,
      circulating_supply: data.market_data.circulating_supply || 0,
      max_supply: data.market_data.max_supply || 0,
      dominance: data.market_cap_rank || 0,
      rank: data.market_cap_rank || 0,
      volatility: 0 // Not available from CoinGecko
    };
  }

  parseMarketData(data) {
    if (!data) return this.getFallbackMarketData();
    return {
      price: data.current_price || 0,
      volume_24h: data.total_volume || 0,
      market_cap: data.market_cap || 0,
      liquidity: 0, // Not available
      bid_ask_spread: 0, // Not available
      volatility: 0, // Not available
      sharpe_ratio: 0, // Not available
      sortino_ratio: 0, // Not available
      max_drawdown: 0, // Not available
      value_at_risk: 0 // Not available
    };
  }

  calculateMarketSentiment(marketData) {
    const {
      price_change_24h = 0,
      volume_24h = 0,
      volatility = 0
    } = marketData;

    // Positive sentiment for price increases and high volume
    let sentiment = 0;
    
    if (price_change_24h > 0) {
      sentiment += Math.min(price_change_24h / 10, 0.5);
    } else {
      sentiment -= Math.min(Math.abs(price_change_24h) / 10, 0.5);
    }

    // Volume impact
    if (volume_24h > 1000000) {
      sentiment += 0.2;
    }

    // Volatility penalty
    if (volatility > 50) {
      sentiment -= 0.1;
    }

    return Math.max(-1, Math.min(1, sentiment));
  }

  calculateRiskIndicators(coinData, socialData, marketData) {
    const {
      volatility = 0,
      max_drawdown = 0,
      value_at_risk = 0
    } = marketData;

    const {
      social_score = 0
    } = socialData;

    return {
      volatility_risk: Math.min(volatility / 100, 1),
      drawdown_risk: Math.min(max_drawdown / 50, 1),
      var_risk: Math.min(value_at_risk / 20, 1),
      social_risk: social_score < -50 ? 0.5 : 0,
      overall_risk: Math.min((volatility + max_drawdown + value_at_risk) / 150, 1)
    };
  }

  calculateConfidenceScore(coinData, socialData, marketData) {
    const {
      volume_24h = 0,
      market_cap = 0
    } = coinData;

    const {
      social_volume = 0,
      social_contributors = 0
    } = socialData;

    const {
      liquidity = 0
    } = marketData;

    // Higher confidence for higher volume, market cap, and social activity
    let confidence = 0.5; // Base confidence

    if (volume_24h > 1000000) confidence += 0.2;
    if (market_cap > 100000000) confidence += 0.1;
    if (social_volume > 1000) confidence += 0.1;
    if (social_contributors > 100) confidence += 0.1;
    if (liquidity > 100000) confidence += 0.1;

    return Math.min(confidence, 1);
  }

  // Fallback data methods
  getFallbackCoinData(symbol = 'BTC') {
    return {
      price: 45000,
      price_change_24h: 2.5,
      volume_24h: 2500000000,
      market_cap: 850000000000,
      circulating_supply: 19500000,
      max_supply: 21000000,
      dominance: 45.2,
      rank: 1,
      volatility: 35.5
    };
  }

  getFallbackMarketData(symbol = 'BTC') {
    return {
      price: 45000,
      volume_24h: 2500000000,
      market_cap: 850000000000,
      liquidity: 500000000,
      bid_ask_spread: 0.1,
      volatility: 35.5,
      sharpe_ratio: 1.2,
      sortino_ratio: 1.8,
      max_drawdown: 15.5,
      value_at_risk: 8.5
    };
  }

  getFallbackComprehensiveData(symbol) {
    return {
      source: 'coingecko_fallback',
      timestamp: new Date().toISOString(),
      symbol: symbol,
      coin_data: this.getFallbackCoinData(symbol),
      market_metrics: this.getFallbackMarketData(symbol),
      sentiment_score: 0.3,
      market_sentiment: 0.4,
      risk_indicators: {
        volatility_risk: 0.3,
        drawdown_risk: 0.2,
        var_risk: 0.4,
        social_risk: 0,
        overall_risk: 0.3
      },
      confidence_score: 0.6
    };
  }

  // Add delay utility function
  delay(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
  }
}

module.exports = CoinGeckoService; 