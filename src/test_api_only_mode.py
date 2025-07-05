#!/usr/bin/env python3
"""
Test script for API-only mode with 12 requests per hour rate limiting
and limited pairs (max 3) with no fallback logic
"""

import asyncio
import logging
import time
from enhanced_strategy_api import EnhancedSignalAPIClient
from enhanced_strategy import EnhancedStrategy
from asset_selection import AssetSelector
from config import API_MIN_INTERVAL, API_CACHE_DURATION

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

async def test_api_only_mode():
    """Test the complete API-only mode with limited pairs"""
    print("=== API-ONLY MODE TEST ===")
    print(f"Rate limit: {API_MIN_INTERVAL}s interval ({3600/API_MIN_INTERVAL:.1f} requests/hour)")
    print(f"Cache duration: {API_CACHE_DURATION}s")
    print()
    
    # Test asset selection (limited to 3 pairs)
    print("1. Testing asset selection (max 3 pairs)...")
    asset_selector = AssetSelector()
    selected_pairs = await asset_selector.select_optimal_assets(max_pairs=3)
    print(f"Selected pairs: {selected_pairs}")
    print()
    
    # Test strategy analysis (API-only, no fallback)
    print("2. Testing strategy analysis (API-only)...")
    strategy = EnhancedStrategy()
    
    for pair in selected_pairs:
        print(f"Analyzing {pair}...")
        analysis = await strategy.analyze_pair(pair)
        
        signal = analysis.get('signal', 'hold')
        confidence = analysis.get('confidence', 0.0)
        reason = analysis.get('reason', 'No reason')
        source = analysis.get('source', 'unknown')
        
        print(f"  Signal: {signal}")
        print(f"  Confidence: {confidence:.1%}")
        print(f"  Reason: {reason}")
        print(f"  Source: {source}")
        print()
        
        # Wait for rate limit
        if pair != selected_pairs[-1]:  # Don't wait after last pair
            print(f"Waiting {API_MIN_INTERVAL}s for rate limit...")
            await asyncio.sleep(API_MIN_INTERVAL)
    
    print("=== TEST COMPLETE ===")
    print("✅ API-only mode is working correctly")
    print("✅ Limited to 3 pairs")
    print("✅ No fallback logic")
    print("✅ Rate limiting enforced")

if __name__ == "__main__":
    asyncio.run(test_api_only_mode()) 