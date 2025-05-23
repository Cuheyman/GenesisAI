// nebula-proxy.js
const express = require('express');
const { ThirdwebSDK } = require('@thirdweb-dev/sdk');
const cors = require('cors');
require('dotenv').config();

const app = express();
app.use(express.json());
app.use(cors());

// Initialize ThirdWeb SDK with your API key
const sdk = ThirdwebSDK.fromPrivateKey(
  process.env.PRIVATE_KEY,
  "ethereum", // or your preferred chain
  {
    secretKey: process.env.THIRDWEB_API_KEY,
  }
);

// Get the Nebula AI instance
const nebula = sdk.nebula;

// Endpoint for market prediction
app.post('/nebula/predict', async (req, res) => {
  try {
    const { token, timeframe } = req.body.inputs;
    
    const prediction = await nebula.chat({
      messages: [
        {
          role: "user",
          content: `Predict price movement for ${token} in the ${timeframe} timeframe, including direction, confidence level, and key indicators.`
        }
      ]
    });
    
    res.json({
      prediction: {
        direction: extractDirection(prediction.content), // You'll need to implement this function
        confidence: extractConfidence(prediction.content),
        analysis: prediction.content
      }
    });
  } catch (error) {
    console.error('Error in prediction:', error);
    res.status(500).json({ error: error.message });
  }
});

// Endpoint for sentiment analysis
app.post('/nebula/analyze', async (req, res) => {
  try {
    const { token, timeframe } = req.body.inputs;
    
    const sentiment = await nebula.chat({
      messages: [
        {
          role: "user",
          content: `Analyze on-chain sentiment for ${token} over the past ${timeframe}, including social metrics, trading volume trends, and overall sentiment score.`
        }
      ]
    });
    
    res.json({
      sentiment_score: extractSentimentScore(sentiment.content),
      analysis: sentiment.content
    });
  } catch (error) {
    console.error('Error in sentiment analysis:', error);
    res.status(500).json({ error: error.message });
  }
});

// Endpoint for whale tracking
app.post('/nebula/whale-tracking', async (req, res) => {
  try {
    const { token, lookback_hours } = req.body.inputs;
    
    const whaleActivity = await nebula.chat({
      messages: [
        {
          role: "user",
          content: `Track whale wallet activity for ${token} over the past ${lookback_hours} hours. Identify accumulation or distribution patterns.`
        }
      ]
    });
    
    res.json({
      accumulation_score: extractAccumulationScore(whaleActivity.content),
      analysis: whaleActivity.content
    });
  } catch (error) {
    console.error('Error in whale tracking:', error);
    res.status(500).json({ error: error.message });
  }
});

// Helper functions to parse AI responses
function extractDirection(text) {
  if (text.toLowerCase().includes('bullish')) return 'bullish';
  if (text.toLowerCase().includes('bearish')) return 'bearish';
  return 'neutral';
}

function extractConfidence(text) {
  // Simple regex to find confidence percentage
  const match = text.match(/confidence(\s+of)?\s+(\d+)%/i);
  if (match) return parseInt(match[2]) / 100;
  
  // Fallback to sentiment strength words
  if (text.toLowerCase().includes('strongly')) return 0.8;
  if (text.toLowerCase().includes('moderately')) return 0.6;
  if (text.toLowerCase().includes('slightly')) return 0.4;
  return 0.5; // Default moderate confidence
}

function extractSentimentScore(text) {
  // Parse sentiment from -1 to 1
  if (text.toLowerCase().includes('very positive') || text.toLowerCase().includes('extremely bullish')) return 0.8;
  if (text.toLowerCase().includes('positive') || text.toLowerCase().includes('bullish')) return 0.5;
  if (text.toLowerCase().includes('somewhat positive') || text.toLowerCase().includes('slightly bullish')) return 0.2;
  if (text.toLowerCase().includes('neutral')) return 0;
  if (text.toLowerCase().includes('somewhat negative') || text.toLowerCase().includes('slightly bearish')) return -0.2;
  if (text.toLowerCase().includes('negative') || text.toLowerCase().includes('bearish')) return -0.5;
  if (text.toLowerCase().includes('very negative') || text.toLowerCase().includes('extremely bearish')) return -0.8;
  return 0; // Default neutral
}

function extractAccumulationScore(text) {
  // Parse accumulation from -1 to 1
  if (text.toLowerCase().includes('strong accumulation')) return 0.8;
  if (text.toLowerCase().includes('accumulation')) return 0.5;
  if (text.toLowerCase().includes('slight accumulation')) return 0.2;
  if (text.toLowerCase().includes('neutral') || text.toLowerCase().includes('mixed')) return 0;
  if (text.toLowerCase().includes('slight distribution')) return -0.2;
  if (text.toLowerCase().includes('distribution')) return -0.5;
  if (text.toLowerCase().includes('strong distribution')) return -0.8;
  return 0; // Default neutral
}

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`Nebula Proxy server running on port ${PORT}`);
});