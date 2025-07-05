# GenesisAI Rate Limit Optimization

## Overview
This document outlines the optimizations made to respect API rate limits and remove unnecessary API dependencies from the GenesisAI trading bot.

## Changes Made

### 1. CoinGecko Rate Limiting (Ultra-Conservative)

**Before:**
- Free tier: 2-second intervals, 20 requests per minute
- Demo tier: 1-second intervals, 40 requests per minute

**After:**
- Free tier: **30-second intervals, 2 requests per minute**
- Demo tier: **15-second intervals, 3 requests per minute**
- Backoff multiplier: Increased from 10x to 15x
- Initial backoff: Increased from 60s to 120s
- Auto-disable after 5 consecutive rate limit violations

**Configuration:**
```python
COINGECKO_FREE_TIER_INTERVAL = 30.0  # 30 seconds between requests
COINGECKO_FREE_TIER_MAX_PER_MINUTE = 2  # Max 2 requests per minute
COINGECKO_DEMO_TIER_INTERVAL = 15.0  # 15 seconds between requests  
COINGECKO_DEMO_TIER_MAX_PER_MINUTE = 3  # Max 3 requests per minute
COINGECKO_INITIAL_BACKOFF = 120  # 2 minutes initial backoff
COINGECKO_MAX_BACKOFF_MULTIPLIER = 15.0  # 15x backoff multiplier
```

### 2. Taapi.io Completely Removed

**Before:**
- Taapi.io was enabled and making API calls
- Causing unnecessary rate limit issues
- Adding complexity without significant benefit

**After:**
- **Taapi.io completely disabled**
- `ENABLE_TAAPI = False`
- `TAAPI_API_SECRET = None`
- Advanced indicators module disabled
- Removes all Taapi.io API calls

### 3. Enhanced Signal API Optimization

**Before:**
- API enabled by default
- 3-second intervals between requests
- 60-second cache duration
- 5 retries on failure

**After:**
- **API disabled by default** (`ENABLE_ENHANCED_API = False`)
- 10-second intervals between requests (if enabled)
- 300-second cache duration (5 minutes)
- 2 retries on failure
- 600-second health check interval (10 minutes)

**Configuration:**
```python
ENABLE_ENHANCED_API = False  # Disabled by default
API_MIN_INTERVAL = 10.0  # 10 seconds between requests
API_CACHE_DURATION = 300  # 5 minutes cache
API_MAX_RETRIES = 2  # Only 2 retries
API_HEALTH_CHECK_INTERVAL = 600  # 10 minutes
```

### 4. Improved Error Handling

**Rate Limit Handling:**
- Global cooldown after 3 consecutive rate limit violations
- Auto-disable CoinGecko after 5 consecutive violations
- Aggressive backoff with 15x multiplier
- Better logging without spam

**Connection Error Handling:**
- Reduced timeout from 60s to 30s
- Fewer retry attempts
- Graceful fallback to internal analysis

## Benefits

### 1. Rate Limit Compliance
- **CoinGecko free tier**: Now respects 2 requests/minute limit
- **No more 429 errors**: Bot will wait instead of hitting rate limits
- **Automatic recovery**: Backoff and cooldown mechanisms

### 2. Reduced API Dependencies
- **Removed Taapi.io**: No more unnecessary API calls
- **Disabled Enhanced API by default**: Only enable when needed
- **Simplified architecture**: Fewer external dependencies

### 3. Better Performance
- **Longer cache durations**: Fewer API calls needed
- **Conservative intervals**: Prevents rate limit issues
- **Graceful degradation**: Falls back to internal analysis

### 4. Cost Optimization
- **Fewer API calls**: Reduced costs for paid APIs
- **Better resource usage**: Less bandwidth and processing
- **Longer uptime**: Fewer rate limit suspensions

## Usage

### Running the Bot
The bot will now run with ultra-conservative rate limiting:

```bash
cd src
python main.py
```

### Monitoring Rate Limits
Check the logs for rate limit information:
- `CoinGecko initialiseret - Gratis tier (30.0s interval, 2/min)`
- Rate limit warnings when approaching limits
- Global cooldown messages when limits exceeded

### Testing Configuration
Run the test script to verify settings:

```bash
python test_rate_limits.py
```

## Configuration Options

### Environment Variables
You can override the default settings with environment variables:

```bash
# CoinGecko settings
export COINGECKO_FREE_TIER_INTERVAL=30.0
export COINGECKO_FREE_TIER_MAX_PER_MINUTE=2

# API settings  
export ENABLE_ENHANCED_API=false
export API_MIN_INTERVAL=10.0
export API_CACHE_DURATION=300
```

### Re-enabling APIs (if needed)
If you want to re-enable certain APIs:

```python
# In config.py
ENABLE_ENHANCED_API = True  # Re-enable Enhanced Signal API
ENABLE_TAAPI = True  # Re-enable Taapi.io (not recommended)
```

## Troubleshooting

### Rate Limit Issues
If you still see rate limit errors:
1. Check the current rate limit settings
2. Increase intervals further if needed
3. Monitor the logs for backoff messages

### API Connection Issues
If APIs are not responding:
1. Check network connectivity
2. Verify API keys (if using paid tiers)
3. Check if APIs are in maintenance mode

### Performance Issues
If the bot seems slow:
1. This is expected with conservative rate limiting
2. Consider upgrading to paid API tiers
3. Adjust cache durations if needed

## Conclusion

These optimizations ensure that the GenesisAI bot:
- **Respects all API rate limits**
- **Reduces unnecessary API calls**
- **Provides better stability and reliability**
- **Maintains trading functionality with internal analysis**

The bot will now run much more reliably without hitting rate limits, while still providing effective trading signals through internal analysis and the remaining API integrations. 