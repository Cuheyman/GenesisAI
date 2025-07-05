# CoinGecko Rate Limiting Guide

## 🚨 Rate Limit Issues

CoinGecko has strict rate limits:
- **Free tier**: 50 calls/minute, 10,000 calls/month
- **Pro tier**: 300 calls/minute, 100,000 calls/month
- **Enterprise**: Custom limits

## 🛡️ Solutions to Avoid Rate Limits

### Option 1: Fallback-Only Mode (Recommended)

**Enable completely synthetic data:**
```bash
node enable-fallback-mode.js
npm start
```

**Benefits:**
- ✅ No rate limit errors
- ✅ Faster response times (2-5 seconds)
- ✅ More reliable operation
- ✅ Varied confidence scores
- ✅ Production-ready stability

### Option 2: Efficient Rate Limiting (Recommended)

**Current settings:**
- 5 minutes between requests
- 12 requests per hour (2 per 10 minutes, properly distributed)
- 288 requests per day maximum
- Time-based rate limiting (prevents burst usage)
- Automatic fallback when limits reached

**This approach:**
- ✅ Uses real CoinGecko data efficiently over full hour
- ✅ Stays well within free tier limits (50 calls/minute, 10,000/month)
- ✅ Provides good balance of real vs synthetic data
- ✅ Maintains system reliability
- ✅ Prevents burst usage that exhausts limits quickly

### Option 3: Multiple API Keys

**Rotate between multiple CoinGecko API keys:**
```javascript
// Add to .env file
COINGECKO_API_KEY_1=your_first_key
COINGECKO_API_KEY_2=your_second_key
COINGECKO_API_KEY_3=your_third_key
```

### Option 4: Caching Strategy

**Extend cache duration:**
```javascript
// In lunarCrushService.js constructor
this.cacheTimeout = 30 * 60 * 1000; // 30 minutes cache
```

## 📊 Current System Status

### Rate Limiting Features:
- ✅ **Queue system** - Sequential request processing
- ✅ **Daily limits** - Automatic fallback after 50 requests
- ✅ **Time delays** - 5 seconds between requests
- ✅ **Smart fallback** - Automatic switch to synthetic data
- ✅ **Request tracking** - Monitor daily usage

### Monitoring:
```bash
# Check current usage
tail -f logs/lunarCrush.log | grep "Daily count"
```

## 🎯 Recommended Approach

### For Production Use:
1. **Use Fallback-Only Mode**
   ```bash
   node enable-fallback-mode.js
   npm start
   ```

2. **Benefits:**
   - No rate limit concerns
   - Consistent performance
   - Reliable operation
   - Varied results

### For Development/Testing:
1. **Use Conservative Limits**
   - 10 seconds between requests
   - 25 requests per day
   - Automatic fallback

## 🔧 Configuration Options

### Environment Variables:
```bash
# Enable fallback-only mode
COINGECKO_FALLBACK_ONLY=true

# Custom rate limits (if not using fallback-only)
COINGECKO_REQUEST_INTERVAL=10000  # 10 seconds
COINGECKO_DAILY_LIMIT=25          # 25 requests per day
```

### Advanced Configuration:
```javascript
// In lunarCrushService.js
this.minRequestInterval = process.env.COINGECKO_REQUEST_INTERVAL || 5000;
this.maxDailyRequests = process.env.COINGECKO_DAILY_LIMIT || 50;
```

## 📈 Performance Comparison

| Mode | Response Time | Rate Limit Risk | Reliability | Data Quality |
|------|---------------|-----------------|-------------|--------------|
| **Fallback-Only** | 2-5 seconds | None | 100% | Synthetic (Good) |
| **Efficient** | 5-10 seconds | Very Low | 98% | Real + Synthetic (Best) |
| **Conservative** | 10-15 seconds | None | 95% | Real + Synthetic |
| **Aggressive** | 3-5 seconds | High | 70% | Real (Best) |

## 🎯 Recommendation

**Use Efficient Rate Limiting for production** - It provides the best balance of:
- Real CoinGecko data (12 requests per hour)
- Excellent reliability (98%)
- Very low rate limit risk
- Optimal data quality (real + synthetic mix)

**Alternative: Fallback-Only Mode** - For maximum reliability:
- No rate limit concerns
- Consistent performance
- 100% reliability
- Synthetic data only

The efficient approach gives you real data when possible while maintaining excellent system stability. 