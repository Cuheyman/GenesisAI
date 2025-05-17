const express = require('express');
const axios = require('axios');
const cors = require('cors');
require('dotenv').config();

const app = express();
app.use(express.json());
app.use(cors());

// Add clear console output to confirm server is running
console.log('Starting Nebula AI proxy server...');

process.on('unhandledRejection', (reason, promise) => {
    console.error('Unhandled Rejection at:', promise, 'reason:', reason);
  });

// Health check endpoint
app.get('/health', (req, res) => {
  console.log('Health check requested');
  res.status(200).json({ status: 'ok', timestamp: new Date().toISOString() });
});

// Market prediction endpoint
app.post('/nebula/predict', async (req, res) => {
  try {
    console.log('Prediction request received:', req.body.inputs);
    const { token, timeframe } = req.body.inputs;
    
    // Simulate a response since we can't directly access ThirdWeb's SDK
    const prediction = {
      prediction: {
        direction: ['bullish', 'bearish', 'neutral'][Math.floor(Math.random() * 3)],
        confidence: (Math.random() * 0.5 + 0.3).toFixed(2), // Random confidence between 0.3 and 0.8
        analysis: `Simulated price prediction for ${token} in the ${timeframe} timeframe.`
      }
    };
    
    console.log('Sending prediction response');
    res.json(prediction);
  } catch (error) {
    console.error('Error in prediction endpoint:', error.message);
    res.status(500).json({ error: error.message });
  }
});

// Sentiment analysis endpoint
app.post('/nebula/analyze', async (req, res) => {
  try {
    console.log('Sentiment analysis request received:', req.body.inputs);
    const { token } = req.body.inputs;
    
    // Simulate a response
    const sentiment = {
      sentiment_score: (Math.random() * 2 - 1).toFixed(2), // Random score between -1 and 1
      analysis: `Simulated sentiment analysis for ${token}.`
    };
    
    console.log('Sending sentiment response');
    res.json(sentiment);
  } catch (error) {
    console.error('Error in sentiment analysis endpoint:', error.message);
    res.status(500).json({ error: error.message });
  }
});

// Whale tracking endpoint
app.post('/nebula/whale-tracking', async (req, res) => {
  try {
    console.log('Whale tracking request received:', req.body.inputs);
    const { token, lookback_hours } = req.body.inputs;
    
    // Simulate a response
    const whaleActivity = {
      accumulation_score: (Math.random() * 2 - 1).toFixed(2), // Random score between -1 and 1
      analysis: `Simulated whale activity analysis for ${token} over ${lookback_hours} hours.`
    };
    
    console.log('Sending whale activity response');
    res.json(whaleActivity);
  } catch (error) {
    console.error('Error in whale tracking endpoint:', error.message);
    res.status(500).json({ error: error.message });
  }
});

// Smart money endpoint
app.post('/nebula/smart-money', async (req, res) => {
  try {
    console.log('Smart money request received:', req.body.inputs);
    const { token } = req.body.inputs;
    
    // Simulate a response
    const positions = ['bullish', 'neutral', 'bearish'];
    const position = positions[Math.floor(Math.random() * positions.length)];
    
    const smartMoney = {
      position: position,
      analysis: `Simulated smart money analysis for ${token} shows a ${position} position.`
    };
    
    console.log('Sending smart money response');
    res.json(smartMoney);
  } catch (error) {
    console.error('Error in smart money endpoint:', error.message);
    res.status(500).json({ error: error.message });
  }
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`\n===== Nebula AI Proxy Server =====`);
  console.log(`Server is running on http://localhost:${PORT}`);
  console.log(`Health check: http://localhost:${PORT}/health`);
  console.log(`Available endpoints:`);
  console.log(`  - POST /nebula/predict`);
  console.log(`  - POST /nebula/analyze`);
  console.log(`  - POST /nebula/whale-tracking`);
  console.log(`  - POST /nebula/smart-money`);
  console.log(`=====================================\n`);
});

// Catch and log any uncaught exceptions
process.on('uncaughtException', (error) => {
  console.error('UNCAUGHT EXCEPTION:', error);
});