# Enhanced Crypto Signal API

AI-powered cryptocurrency trading signal generation with Claude + LunarCrush integration.

## 🚀 Features

- **AI-Powered Analysis**: Claude 4 Sonnet for market analysis and pattern recognition
- **Social Sentiment**: LunarCrush API for real-time social sentiment and market metrics
- **Technical Analysis**: Advanced technical indicators and market regime detection
- **Dynamic Symbol Validation**: Real-time Binance API integration
- **Risk Management**: Comprehensive risk assessment and position sizing
- **Batch Processing**: Generate signals for multiple symbols simultaneously
- **Rate Limiting**: Built-in rate limiting and security middleware

## 📋 Prerequisites

- Node.js >= 16.0.0
- npm or yarn
- API keys for:
  - Claude AI (Anthropic)
  - LunarCrush
  - Binance (optional, for enhanced symbol validation)

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd API-ENDPOINT-GENESISAI/src
   ```

2. **Install dependencies**
   ```bash
   npm install
   ```

3. **Set up environment variables**
   Create a `.env` file in the `src` directory:
   ```env
   # Server Configuration
   PORT=3000
   NODE_ENV=development

   # API Authentication
   API_KEY=your_api_key_here
   JWT_SECRET=your_jwt_secret_here

   # AI Services
   CLAUDE_API_KEY=your_claude_api_key_here
   LUNARCRUSH_API_KEY=your_lunarcrush_api_key_here

   # Binance API (optional)
   BINANCE_API_KEY=your_binance_api_key_here
   BINANCE_API_SECRET=your_binance_api_secret_here

   # Rate Limiting
   RATE_LIMIT_WINDOW=3600000
   RATE_LIMIT_MAX=100

   # Logging
   LOG_LEVEL=info
   ```

4. **Start the server**
   ```bash
   npm start
   ```

## 🔧 API Endpoints

### Health Check
```http
GET /api/health
```

### Get Valid Symbols
```http
GET /api/v1/symbols
Authorization: Bearer YOUR_API_KEY
```

### Search Symbols
```http
GET /api/v1/symbols/search?query=BTC
Authorization: Bearer YOUR_API_KEY
```

### Generate Signal
```http
POST /api/v1/signals/generate
Authorization: Bearer YOUR_API_KEY
Content-Type: application/json

{
  "symbol": "BTCUSDT",
  "timeframe": "1h",
  "analysis_depth": "comprehensive",
  "risk_level": "moderate",
  "wallet_address": "optional_wallet_address"
}
```

### Batch Signal Generation
```http
POST /api/v1/signals/batch
Authorization: Bearer YOUR_API_KEY
Content-Type: application/json

{
  "symbols": ["BTCUSDT", "ETHUSDT", "ADAUSDT"],
  "timeframe": "1h",
  "analysis_depth": "advanced",
  "risk_level": "moderate"
}
```

### API Documentation
```http
GET /api/docs
```

## 📊 Response Format

### Signal Generation Response
```json
{
  "success": true,
  "request_id": "req_1234567890_abc123",
  "timestamp": 1234567890,
  "data": {
    "signal": "BUY",
    "confidence": 0.85,
    "entry_price": 45000,
    "stop_loss": 43000,
    "take_profit_1": 47000,
    "take_profit_2": 49000,
    "position_size": 0.1,
    "market_data": {
      "symbol": "BTCUSDT",
      "price": 45000,
      "volume_24h": 2500000000,
      "price_change_24h": 2.5
    },
    "technical_indicators": {
      "rsi": 65.5,
      "macd": "bullish",
      "bollinger_bands": "upper_band_touch"
    },
    "onchain_analysis": {
      "whale_activity": {
        "large_transfers_24h": 25,
        "whale_accumulation": "buying"
      },
      "sentiment_score": 0.75,
      "social_velocity": "high"
    }
  }
}
```

## 🔍 Troubleshooting

### Common Issues

1. **LunarCrush API Errors**
   - Verify your LunarCrush API key is valid
   - Check API rate limits
   - Ensure proper symbol format (remove USDT suffix)

2. **Claude AI Errors**
   - Verify your Claude API key is valid
   - Check API quota and rate limits
   - Ensure proper request formatting

3. **Symbol Validation Errors**
   - Check Binance API connectivity
   - Verify symbol format (must end with USDT)
   - Check if symbol is actively trading

4. **Rate Limiting**
   - Reduce request frequency
   - Implement proper caching
   - Consider upgrading API tier

### Error Response Format
```json
{
  "success": false,
  "error": "Error description",
  "request_id": "req_1234567890_abc123",
  "timestamp": 1234567890
}
```

## 🔒 Security

- **API Key Authentication**: All endpoints require valid API key
- **Rate Limiting**: Built-in rate limiting to prevent abuse
- **Input Validation**: Comprehensive request validation
- **CORS Protection**: Configurable CORS settings
- **Helmet Security**: Security headers middleware

## 📈 Performance

- **Caching**: 5-minute cache for LunarCrush data
- **Concurrent Processing**: Batch requests processed in parallel
- **Memory Management**: Efficient memory usage with cleanup
- **Error Handling**: Graceful fallbacks for service failures

## 🚀 Deployment

### Docker
```bash
docker build -t crypto-signal-api .
docker run -p 3000:3000 --env-file .env crypto-signal-api
```

### Kubernetes
```bash
kubectl apply -f kubernetes/deployment.yaml
```

### Environment Variables for Production
```env
NODE_ENV=production
PORT=3000
ALLOWED_ORIGINS=https://yourdomain.com
LOG_LEVEL=warn
```

## 📝 Changelog

### v2.0.0
- ✅ Replaced Nebula AI with LunarCrush integration
- ✅ Enhanced social sentiment analysis
- ✅ Improved error handling and fallbacks
- ✅ Added comprehensive caching
- ✅ Updated documentation and examples

### v1.0.0
- ✅ Initial release with Nebula AI integration
- ✅ Basic signal generation
- ✅ Technical analysis indicators

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For support and questions:
- Create an issue in the repository
- Check the troubleshooting section
- Review the API documentation at `/api/docs`

## 🔗 Links

- [LunarCrush API Documentation](https://lunarcrush.com/developers)
- [Claude AI Documentation](https://docs.anthropic.com/)
- [Binance API Documentation](https://binance-docs.github.io/apidocs/) 