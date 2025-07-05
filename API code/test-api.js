const axios = require('axios');

// Test configuration
const BASE_URL = 'http://localhost:3001';
const API_KEY = process.env.API_KEY || '1234';

// Test data
const testSymbol = 'BTCUSDT';
const testRequest = {
  symbol: testSymbol,
  timeframe: '1h',
  analysis_depth: 'comprehensive',
  risk_level: 'moderate'
};

// Helper function to make authenticated requests
const makeRequest = async (method, endpoint, data = null) => {
  try {
    const config = {
      method,
      url: `${BASE_URL}${endpoint}`,
      headers: {
        'Authorization': `Bearer ${API_KEY}`,
        'Content-Type': 'application/json'
      }
    };

    if (data) {
      config.data = data;
    }

    const response = await axios(config);
    return { success: true, data: response.data };
  } catch (error) {
    return { 
      success: false, 
      error: error.response?.data || error.message,
      status: error.response?.status
    };
  }
};

// Test functions
const testHealthCheck = async () => {
  console.log('🔍 Testing health check...');
  const result = await makeRequest('GET', '/api/health');
  
  if (result.success) {
    console.log('✅ Health check passed');
    console.log('   Status:', result.data.status);
    console.log('   AI Services:', result.data.ai_services);
  } else {
    console.log('❌ Health check failed:', result.error);
  }
  return result.success;
};

const testSymbols = async () => {
  console.log('\n🔍 Testing symbols endpoint...');
  const result = await makeRequest('GET', '/api/v1/symbols');
  
  if (result.success) {
    console.log('✅ Symbols endpoint passed');
    console.log(`   Found ${result.data.data.count} symbols`);
    console.log(`   Top symbols: ${result.data.data.top_by_volume.slice(0, 5).join(', ')}`);
  } else {
    console.log('❌ Symbols endpoint failed:', result.error);
  }
  return result.success;
};

const testSymbolSearch = async () => {
  console.log('\n🔍 Testing symbol search...');
  const result = await makeRequest('GET', '/api/v1/symbols/search?query=BTC');
  
  if (result.success) {
    console.log('✅ Symbol search passed');
    console.log(`   Found ${result.data.data.count} matches for "BTC"`);
    console.log(`   First 3 matches: ${result.data.data.matches.slice(0, 3).join(', ')}`);
  } else {
    console.log('❌ Symbol search failed:', result.error);
  }
  return result.success;
};

const testSignalGeneration = async () => {
  console.log('\n🔍 Testing signal generation...');
  const result = await makeRequest('POST', '/api/v1/signals/generate', testRequest);
  
  if (result.success) {
    console.log('✅ Signal generation passed');
    console.log(`   Signal: ${result.data.data.signal}`);
    console.log(`   Confidence: ${result.data.data.confidence}`);
    console.log(`   Processing time: ${result.data.data.metadata.processing_time_ms}ms`);
    console.log(`   AI Models: ${result.data.data.metadata.ai_models.join(', ')}`);
  } else {
    console.log('❌ Signal generation failed:', result.error);
  }
  return result.success;
};

const testBatchSignals = async () => {
  console.log('\n🔍 Testing batch signal generation...');
  const batchRequest = {
    symbols: ['BTCUSDT', 'ETHUSDT'],
    timeframe: '1h',
    analysis_depth: 'advanced',
    risk_level: 'moderate'
  };
  
  const result = await makeRequest('POST', '/api/v1/signals/batch', batchRequest);
  
  if (result.success) {
    console.log('✅ Batch signal generation passed');
    console.log(`   Processed ${result.data.summary.total} symbols`);
    console.log(`   Successful: ${result.data.summary.successful}`);
    console.log(`   Failed: ${result.data.summary.failed}`);
    console.log(`   Processing time: ${result.data.processing_time_ms}ms`);
  } else {
    console.log('❌ Batch signal generation failed:', result.error);
  }
  return result.success;
};

const testDocumentation = async () => {
  console.log('\n🔍 Testing documentation endpoint...');
  const result = await makeRequest('GET', '/api/docs');
  
  if (result.success) {
    console.log('✅ Documentation endpoint passed');
    console.log(`   API Version: ${result.data.version}`);
    console.log(`   AI Models: ${Object.keys(result.data.ai_models).join(', ')}`);
    console.log(`   Endpoints: ${Object.keys(result.data.endpoints).length} available`);
  } else {
    console.log('❌ Documentation endpoint failed:', result.error);
  }
  return result.success;
};

// Main test runner
const runTests = async () => {
  console.log('🚀 Starting API Tests...\n');
  
  const tests = [
    { name: 'Health Check', fn: testHealthCheck },
    { name: 'Symbols', fn: testSymbols },
    { name: 'Symbol Search', fn: testSymbolSearch },
    { name: 'Signal Generation', fn: testSignalGeneration },
    { name: 'Batch Signals', fn: testBatchSignals },
    { name: 'Documentation', fn: testDocumentation }
  ];
  
  const results = [];
  
  for (const test of tests) {
    const success = await test.fn();
    results.push({ name: test.name, success });
  }
  
  // Summary
  console.log('\n📊 Test Summary:');
  console.log('================');
  
  const passed = results.filter(r => r.success).length;
  const total = results.length;
  
  results.forEach(result => {
    const status = result.success ? '✅' : '❌';
    console.log(`${status} ${result.name}`);
  });
  
  console.log(`\n🎯 Results: ${passed}/${total} tests passed`);
  
  if (passed === total) {
    console.log('🎉 All tests passed! API is working correctly.');
  } else {
    console.log('⚠️  Some tests failed. Check the errors above.');
  }
  
  return passed === total;
};

// Run tests if this file is executed directly
if (require.main === module) {
  runTests()
    .then(success => {
      process.exit(success ? 0 : 1);
    })
    .catch(error => {
      console.error('💥 Test runner error:', error);
      process.exit(1);
    });
}

module.exports = { runTests, makeRequest }; 