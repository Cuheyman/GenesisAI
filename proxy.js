// nebula-direct-proxy.js
const express = require('express');
const fetch = require('node-fetch');  // You may need to install this: npm install node-fetch
const cors = require('cors');
const env = require("dotenv");


const app = express();
app.use(express.json());
app.use(cors());
const SECRET_KEY1 = env.configDotenv()

console.log('Starting Nebula Direct API proxy server...');

// Nebula API configuration
const NEBULA_API_URL = "";
const SECRET_KEY = "";  // Your API key

// Session management
let sessionId = null;

// Health check endpoint
app.get('/health', async (req, res) => {
  console.log('Health check requested');
  
  // Check if we can access the Nebula API
  try {
    // Try to create a session if we don't have one
    if (!sessionId) {
      sessionId = await createSession();
      console.log('Created new session:', sessionId);
    }
    
    res.status(200).json({ 
      status: 'ok', 
      timestamp: new Date().toISOString(),
      nebula_session: sessionId ? true : false
    });
  } catch (error) {
    console.error('Health check error:', error);
    res.status(200).json({
      status: 'warning',
      timestamp: new Date().toISOString(),
      nebula_session: false,
      error: error.message
    });
  }
});

// Utility function to make API requests
async function apiRequest(endpoint, method, body = {}) {
  try {
    console.log(`Making ${method} request to ${endpoint}`);
    
    const response = await fetch(`${NEBULA_API_URL}${endpoint}`, {
      method,
      headers: {
        "Content-Type": "application/json",
        "x-secret-key": SECRET_KEY,
      },
      body: Object.keys(body).length ? JSON.stringify(body) : undefined,
    });

    if (!response.ok) {
      const errorText = await response.text();
      console.error("API Response Error:", errorText);
      throw new Error(`API Error: ${response.statusText}`);
    }

    return await response.json();
  } catch (error) {
    console.error(`API Request Error (${endpoint}):`, error);
    throw error;
  }
}

// Create a new Session
async function createSession(title = "Nebula Trading Bot Session") {
  try {
    const response = await apiRequest("/session", "POST", { title });
    return response.result.id;
  } catch (error) {
    console.error('Error creating session:', error);
    return null;
  }
}

// Ensure we have a session
async function ensureSession() {
  if (!sessionId) {
    sessionId = await createSession();
  }
  return sessionId;
}

// Endpoint for market prediction
app.post('/nebula/predict', async (req, res) => {
  try {
    console.log('Prediction request received:', req.body.inputs);
    const { token, timeframe } = req.body.inputs;
    
    // Make sure we have a session
    await ensureSession();
    
    // Create the query for Nebula
    const message = `Predict price movement for ${token} in the ${timeframe} timeframe, including direction, confidence level, and key indicators.`;
    
    // Send request to Nebula
    const response = await apiRequest("/chat", "POST", {
      message,
      session_id: sessionId
    });
    
    // Extract the prediction using the helper functions from your original proxy
    const content = response.message.content;
    
    res.json({
      prediction: {
        direction: extractDirection(content),
        confidence: extractConfidence(content),
        analysis: content
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
    console.log('Sentiment analysis request received:', req.body.inputs);
    const { token, timeframe } = req.body.inputs;
    
    // Make sure we have a session
    await ensureSession();
    
    // Create the query for Nebula
    const message = `Analyze on-chain sentiment for ${token} over the past ${timeframe || '24h'}, including social metrics, trading volume trends, and overall sentiment score.`;
    
    // Send request to Nebula
    const response = await apiRequest("/chat", "POST", {
      message,
      session_id: sessionId
    });
    
    // Extract sentiment
    const content = response.message.content;
    
    res.json({
      sentiment_score: extractSentimentScore(content),
      analysis: content
    });
  } catch (error) {
    console.error('Error in sentiment analysis:', error);
    res.status(500).json({ error: error.message });
  }
});

// Endpoint for whale tracking
app.post('/nebula/whale-tracking', async (req, res) => {
  try {
    console.log('Whale tracking request received:', req.body.inputs);
    const { token, lookback_hours } = req.body.inputs;
    
    // Make sure we have a session
    await ensureSession();
    
    // Create the query for Nebula
    const message = `Track whale wallet activity for ${token} over the past ${lookback_hours || 24} hours. Identify accumulation or distribution patterns.`;
    
    // Send request to Nebula
    const response = await apiRequest("/chat", "POST", {
      message,
      session_id: sessionId
    });
    
    // Extract accumulation score
    const content = response.message.content;
    
    res.json({
      accumulation_score: extractAccumulationScore(content),
      analysis: content
    });
  } catch (error) {
    console.error('Error in whale tracking:', error);
    res.status(500).json({ error: error.message });
  }
});

// Endpoint for smart money positions
app.post('/nebula/smart-money', async (req, res) => {
  try {
    console.log('Smart money request received:', req.body.inputs);
    const { token } = req.body.inputs;
    
    // Make sure we have a session
    await ensureSession();
    
    // Create the query for Nebula
    const message = `Analyze smart money positions for ${token}. Determine if sophisticated investors are bullish, bearish, or neutral.`;
    
    // Send request to Nebula
    const response = await apiRequest("/chat", "POST", {
      message,
      session_id: sessionId
    });
    
    // Extract smart money position
    const content = response.message.content;
    
    res.json({
      position: extractSmartMoneyPosition(content),
      analysis: content
    });
  } catch (error) {
    console.error('Error in smart money analysis:', error);
    res.status(500).json({ error: error.message });
  }
});

// Helper functions to parse AI responses - reusing from your original proxy
function extractDirection(text) {
  if (!text) return 'neutral';
  const lowerText = text.toLowerCase();
  if (lowerText.includes('strongly bullish') || lowerText.includes('very bullish')) return 'bullish';
  if (lowerText.includes('bullish')) return 'bullish';
  if (lowerText.includes('strongly bearish') || lowerText.includes('very bearish')) return 'bearish';
  if (lowerText.includes('bearish')) return 'bearish';
  return 'neutral';
}

function extractConfidence(text) {
  if (!text) return 0.5;
  // Simple regex to find confidence percentage
  const match = text.match(/confidence(\s+of)?\s+(\d+)%/i);
  if (match) return parseInt(match[2]) / 100;
  
  // Fallback to sentiment strength words
  const lowerText = text.toLowerCase();
  if (lowerText.includes('strongly') || lowerText.includes('very') || lowerText.includes('high confidence')) return 0.8;
  if (lowerText.includes('moderately') || lowerText.includes('reasonable confidence')) return 0.6;
  if (lowerText.includes('slightly') || lowerText.includes('some confidence')) return 0.4;
  return 0.5; // Default moderate confidence
}

function extractSentimentScore(text) {
  if (!text) return 0;
  const lowerText = text.toLowerCase();
  // Parse sentiment from -1 to 1
  if (lowerText.includes('very positive') || lowerText.includes('extremely bullish')) return 0.8;
  if (lowerText.includes('positive') || lowerText.includes('bullish')) return 0.5;
  if (lowerText.includes('somewhat positive') || lowerText.includes('slightly bullish')) return 0.2;
  if (lowerText.includes('neutral')) return 0;
  if (lowerText.includes('somewhat negative') || lowerText.includes('slightly bearish')) return -0.2;
  if (lowerText.includes('negative') || lowerText.includes('bearish')) return -0.5;
  if (lowerText.includes('very negative') || lowerText.includes('extremely bearish')) return -0.8;
  return 0; // Default neutral
}

function extractAccumulationScore(text) {
  if (!text) return 0;
  const lowerText = text.toLowerCase();
  // Parse accumulation from -1 to 1
  if (lowerText.includes('strong accumulation') || lowerText.includes('significant accumulation')) return 0.8;
  if (lowerText.includes('accumulation')) return 0.5;
  if (lowerText.includes('slight accumulation') || lowerText.includes('some accumulation')) return 0.2;
  if (lowerText.includes('neutral') || lowerText.includes('mixed')) return 0;
  if (lowerText.includes('slight distribution') || lowerText.includes('some distribution')) return -0.2;
  if (lowerText.includes('distribution')) return -0.5;
  if (lowerText.includes('strong distribution') || lowerText.includes('significant distribution')) return -0.8;
  return 0; // Default neutral
}

function extractSmartMoneyPosition(text) {
  if (!text) return 'neutral';
  const lowerText = text.toLowerCase();
  if (lowerText.includes('bullish')) return 'bullish';
  if (lowerText.includes('bearish')) return 'bearish';
  return 'neutral';
}

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`\n===== Nebula Direct API Proxy Server =====`);
  console.log(`Server is running on http://localhost:${PORT}`);
  console.log(`Health check: http://localhost:${PORT}/health`);
  console.log(`Available endpoints:`);
  console.log(`  - POST /nebula/predict`);
  console.log(`  - POST /nebula/analyze`);
  console.log(`  - POST /nebula/whale-tracking`);
  console.log(`  - POST /nebula/smart-money`);
  console.log(`=====================================\n`);
  
  // Create initial session
  createSession().then(id => {
    if (id) {
      sessionId = id;
      console.log(`Initial session created: ${sessionId}`);
    } else {
      console.error('Failed to create initial session');
    }
  });
});

// Catch and log any uncaught exceptions
process.on('uncaughtException', (error) => {
  console.error('UNCAUGHT EXCEPTION:', error);
});

process.on('unhandledRejection', (reason, promise) => {
  console.error('Unhandled Rejection at:', promise, 'reason:', reason);
});