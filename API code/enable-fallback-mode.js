#!/usr/bin/env node

// ===============================================
// ENABLE FALLBACK-ONLY MODE SCRIPT
// ===============================================

const fs = require('fs');
const path = require('path');

console.log('🔧 Enabling CoinGecko Fallback-Only Mode...\n');

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
    'COINGECKO_FALLBACK_ONLY=true'
  );
} else {
  // Add new line
  envContent += '\n# CoinGecko Configuration\nCOINGECKO_FALLBACK_ONLY=true\n';
}

// Write updated .env file
fs.writeFileSync(envPath, envContent);

console.log('✅ Fallback-only mode enabled!');
console.log('📝 Added COINGECKO_FALLBACK_ONLY=true to your .env file');
console.log('\n🔄 Restart your server to apply changes:');
console.log('   npm start');
console.log('\n💡 This will use synthetic data instead of CoinGecko API calls');
console.log('   to completely avoid rate limiting issues.');
console.log('\n🚀 Benefits:');
console.log('   • No rate limit errors');
console.log('   • Faster response times');
console.log('   • More reliable operation');
console.log('   • Varied confidence scores');
console.log('   • Production-ready stability'); 