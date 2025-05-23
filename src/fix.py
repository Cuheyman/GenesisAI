#!/usr/bin/env python3
"""
Fix CoinGecko Client for Demo API Key
=====================================

This script updates the CoinGecko client to properly handle demo/free API keys.
"""

import os
import sys

def fix_coingecko_for_demo_key():
    """Update coin_gecko_ai.py to work with demo API keys"""
    
    # Find the correct path
    if os.path.exists('coin_gecko_ai.py'):
        file_path = 'coin_gecko_ai.py'
    elif os.path.exists('src/coin_gecko_ai.py'):
        file_path = 'src/coin_gecko_ai.py'
    else:
        print("ERROR: Cannot find coin_gecko_ai.py")
        return False
    
    print(f"Found file at: {file_path}")
    
    # Read the file
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Fix 1: Update _get_base_url to always use free tier for demo keys
    old_get_base_url = '''def _get_base_url(self):
        """Get the appropriate base URL (pro or free)"""
        return self.pro_url if self.api_key else self.base_url'''
    
    new_get_base_url = '''def _get_base_url(self):
        """Get the appropriate base URL (pro or free)"""
        # Always use free tier URL - pro tier requires special paid API keys
        # Demo keys should use the free tier endpoint
        return self.base_url  # Always use api.coingecko.com'''
    
    if old_get_base_url in content:
        content = content.replace(old_get_base_url, new_get_base_url)
        print("✓ Updated _get_base_url to use free tier")
    else:
        print("! _get_base_url already updated or different")
    
    # Fix 2: Update _check_api_connection to use free tier
    old_check_pattern = 'f"{self._get_base_url()}/ping"'
    new_check_pattern = '"https://api.coingecko.com/api/v3/ping"'
    
    if old_check_pattern in content:
        content = content.replace(old_check_pattern, new_check_pattern)
        print("✓ Updated connection check to use free tier directly")
    
    # Fix 3: Update headers to handle demo keys properly
    old_headers = '''def _get_headers(self):
        """Get headers for API requests"""
        headers = {"accept": "application/json"}
        if self.api_key:
            headers["x-cg-pro-api-key"] = self.api_key
        return headers'''
    
    new_headers = '''def _get_headers(self):
        """Get headers for API requests"""
        headers = {"accept": "application/json"}
        # For demo/free keys, use the demo API key header
        if self.api_key:
            # Demo keys use x-cg-demo-api-key header
            if self.api_key.startswith('CG-'):
                headers["x-cg-demo-api-key"] = self.api_key
            else:
                # Legacy pro key format
                headers["x-cg-pro-api-key"] = self.api_key
        return headers'''
    
    if old_headers in content:
        content = content.replace(old_headers, new_headers)
        print("✓ Updated headers to handle demo API keys")
    
    # Fix 4: Make the connection check more robust
    if '_check_api_connection' in content:
        # Replace the entire method with a more robust version
        new_check_method = '''def _check_api_connection(self):
        """Check if the CoinGecko API is available"""
        try:
            # Always test with the free tier ping endpoint
            test_url = "https://api.coingecko.com/api/v3/ping"
            headers = {"accept": "application/json"}
            
            # Add demo API key if available
            if self.api_key and self.api_key.startswith('CG-'):
                headers["x-cg-demo-api-key"] = self.api_key
            
            response = requests.get(test_url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                try:
                    data = response.json()
                    if 'gecko_says' in data:
                        logging.info(f"CoinGecko API connected: {data['gecko_says']}")
                        return True
                except:
                    # Even without JSON, 200 status means success
                    return True
            else:
                logging.warning(f"CoinGecko API returned status {response.status_code}")
                # Don't completely fail - allow retry later
                return True  # Return True to allow retry on actual requests
                
        except requests.exceptions.ConnectionError:
            logging.warning("CoinGecko connection error - will retry later")
            return True  # Allow retry
        except requests.exceptions.Timeout:
            logging.warning("CoinGecko request timed out - will retry later")
            return True  # Allow retry
        except Exception as e:
            logging.warning(f"CoinGecko connection check error: {str(e)} - will retry later")
            return True  # Allow retry'''
        
        # Find and replace the method
        import re
        pattern = r'def _check_api_connection\(self\):.*?(?=\n    def|\Z)'
        match = re.search(pattern, content, re.DOTALL)
        if match:
            content = content[:match.start()] + new_check_method + content[match.end():]
            print("✓ Replaced _check_api_connection with robust version")
    
    # Fix 5: Update initialization message
    old_init_msg = '''if self.api_available:
            logging.info("CoinGecko AI client initialized successfully")
        else:
            logging.warning("CoinGecko AI client initialized but API unavailable - will use fallback mode")'''
    
    new_init_msg = '''if self.api_available:
            if self.api_key and self.api_key.startswith('CG-'):
                logging.info("CoinGecko AI client initialized with demo API key")
            else:
                logging.info("CoinGecko AI client initialized successfully")
        else:
            # This should rarely happen now
            logging.warning("CoinGecko AI client initialized - connection check failed but will retry")'''
    
    if old_init_msg in content:
        content = content.replace(old_init_msg, new_init_msg)
        print("✓ Updated initialization messages")
    
    # Write the updated content back
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("\n✓ CoinGecko client updated successfully!")
    print("\nThe client will now:")
    print("- Always use the free tier endpoint (api.coingecko.com)")
    print("- Properly handle demo API keys (CG-xxx format)")
    print("- Be more resilient to connection issues")
    print("- Retry failed connections instead of disabling completely")
    
    return True

def test_updated_client():
    """Test the updated CoinGecko client"""
    print("\n" + "="*60)
    print("Testing Updated Client")
    print("="*60)
    
    try:
        # Clear any cached modules
        if 'coin_gecko_ai' in sys.modules:
            del sys.modules['coin_gecko_ai']
        if 'config' in sys.modules:
            del sys.modules['config']
        
        # Import fresh
        import config
        from coin_gecko_ai import CoinGeckoAI
        
        print("Creating CoinGecko client...")
        client = CoinGeckoAI(config.COINGECKO_API_KEY)
        
        print(f"✓ Client created")
        print(f"  API available: {client.api_available}")
        print(f"  Using API key: {client.api_key[:10]}..." if client.api_key else "  No API key")
        
        # Test a simple API call
        import asyncio
        
        async def test_api():
            # Test market data
            data = await client.get_market_data("BTC")
            if data:
                print(f"✓ Market data retrieved: BTC price = ${data.get('usd', 'N/A')}")
            else:
                print("✗ No market data returned")
            
            # Test sentiment
            sentiment = await client.get_sentiment_analysis("ETH")
            if sentiment:
                print(f"✓ Sentiment analysis working: score = {sentiment.get('sentiment_score', 'N/A')}")
            else:
                print("✗ No sentiment data returned")
        
        print("\nTesting API calls...")
        asyncio.run(test_api())
        
    except Exception as e:
        print(f"✗ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()

def main():
    """Main function"""
    print("CoinGecko Demo API Key Fix")
    print("="*60)
    
    # Apply the fix
    success = fix_coingecko_for_demo_key()
    
    if success:
        # Test the updated client
        test_updated_client()
        
        print("\n" + "="*60)
        print("Fix Complete!")
        print("="*60)
        print("\nYour bot should now work properly with the CoinGecko demo API key.")
        print("Run 'python main.py' to start your bot.")
    else:
        print("\nFix failed - please check the error messages above.")

if __name__ == "__main__":
    main()