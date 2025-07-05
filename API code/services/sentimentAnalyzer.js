// Add to api-server/services/sentimentAnalyzer.js
const axios = require('axios');
const Sentiment = require('sentiment');
const natural = require('natural');

class SentimentAnalyzer {
    constructor() {
        this.sentiment = new Sentiment();
        this.tokenizer = new natural.WordTokenizer();
        this.sources = {
            twitter: { weight: 0.3, api: process.env.TWITTER_API },
            reddit: { weight: 0.2, api: process.env.REDDIT_API },
            news: { weight: 0.4, api: process.env.NEWS_API },
            telegram: { weight: 0.1, api: process.env.TELEGRAM_API }
        };
    }

    async analyzeSentiment(symbol) {
        const sentimentScores = {};
        
        // Gather sentiment from all sources
        const promises = Object.entries(this.sources).map(async ([source, config]) => {
            try {
                const data = await this.fetchSourceData(source, symbol);
                const score = this.calculateSentimentScore(data);
                sentimentScores[source] = {
                    score,
                    weight: config.weight,
                    sampleSize: data.length
                };
            } catch (error) {
                console.error(`Error fetching ${source} sentiment:`, error);
                sentimentScores[source] = { score: 0, weight: 0, sampleSize: 0 };
            }
        });

        await Promise.all(promises);

        // Calculate weighted sentiment
        const weightedSentiment = this.calculateWeightedSentiment(sentimentScores);
        
        // Analyze sentiment velocity (rate of change)
        const sentimentVelocity = await this.calculateSentimentVelocity(symbol, weightedSentiment);
        
        // Detect sentiment anomalies
        const anomalies = this.detectSentimentAnomalies(sentimentScores);

        return {
            overall: weightedSentiment,
            sources: sentimentScores,
            velocity: sentimentVelocity,
            anomalies,
            confidence: this.calculateConfidence(sentimentScores),
            timestamp: new Date().toISOString()
        };
    }

    calculateSentimentScore(texts) {
        if (!texts || texts.length === 0) return 0;

        const scores = texts.map(text => {
            const result = this.sentiment.analyze(text);
            
            // Advanced scoring with context
            let contextMultiplier = 1;
            
            // Check for strong positive indicators
            if (text.match(/bull|moon|pump|breakout|accumulation/gi)) {
                contextMultiplier += 0.2;
            }
            
            // Check for strong negative indicators
            if (text.match(/bear|dump|crash|scam|rug/gi)) {
                contextMultiplier -= 0.2;
            }
            
            // Normalize score to -1 to 1 range
            const normalizedScore = Math.tanh(result.score / 10) * contextMultiplier;
            
            return normalizedScore;
        });

        return scores.reduce((a, b) => a + b, 0) / scores.length;
    }

    detectSentimentAnomalies(sentimentScores) {
        const scores = Object.values(sentimentScores).map(s => s.score);
        const mean = scores.reduce((a, b) => a + b, 0) / scores.length;
        const stdDev = Math.sqrt(scores.reduce((sq, n) => sq + Math.pow(n - mean, 2), 0) / scores.length);

        const anomalies = [];
        Object.entries(sentimentScores).forEach(([source, data]) => {
            const zScore = (data.score - mean) / stdDev;
            if (Math.abs(zScore) > 2) {
                anomalies.push({
                    source,
                    zScore,
                    type: zScore > 0 ? 'extremely_positive' : 'extremely_negative'
                });
            }
        });

        return anomalies;
    }
}