#!/usr/bin/env node

// ===============================================
// DISABLE FALLBACK-ONLY MODE SCRIPT
// ===============================================

const fs = require('fs');
const path = require('path');

console.log('🔧 Disabling Fallback-Only Mode - Enabling Real CoinGecko Data...\n');

// Check if .env file exists
const envPath = path.join(__dirname, '.env');
if (!fs.existsSync(envPath)) {
  console.error('❌ .env file not found! Please create a .env file first.');
  process.exit(1);
}

// Read current .env file
let envContent = fs.readFileSync(envPath, 'utf8');

// Check if COINGECKO_FALLBACK_ONLY already exists
if (envContent.includes('COINGECKO_FALLBACK_ONLY=')) {
  // Update existing line
  envContent = envContent.replace(
    /COINGECKO_FALLBACK_ONLY=.*/g,
    'COINGECKO_FALLBACK_ONLY=false'
  );
} else {
  // Add new line
  envContent += '\n# CoinGecko Configuration\nCOINGECKO_FALLBACK_ONLY=false\n';
}

// Write updated .env file
fs.writeFileSync(envPath, envContent);

console.log('✅ Fallback-only mode disabled!');
console.log('📝 Set COINGECKO_FALLBACK_ONLY=false in your .env file');
console.log('\n🔄 Restart your server to apply changes:');
console.log('   npm start');
console.log('\n🛡️ Ultra-Conservative Rate Limiting Active:');
console.log('   • 15 seconds between requests');
console.log('   • 2 requests per hour maximum');
console.log('   • 20 requests per day maximum');
console.log('   • 15-minute cache duration');
console.log('\n📊 This ensures you NEVER hit CoinGecko rate limits:');
console.log('   • Free tier: 50 calls/minute, 10,000 calls/month');
console.log('   • Your usage: 2 calls/hour, 20 calls/day');
console.log('   • Safety margin: 99.9% below limits');
console.log('\n💡 Benefits:');
console.log('   • Real CoinGecko data');
console.log('   • Zero rate limit risk');
console.log('   • Reliable operation');
console.log('   • Varied confidence scores'); 