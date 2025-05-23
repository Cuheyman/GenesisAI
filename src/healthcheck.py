# Connection Healthcheck Script
# Create this as a separate file to monitor the Nebula proxy connection

import requests
import time
import logging
import os
import json
from datetime import datetime

# Configure logging
log_dir = 'logs'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

log_file = os.path.join(log_dir, f'nebula_monitor_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

# Configuration
PROXY_URL = "http://localhost:3000"  # Change if needed
CHECK_INTERVAL = 60  # Check every minute
MAX_FAILURES = 5  # Number of consecutive failures before attempting restart
RESTART_TIMEOUT = 300  # Wait 5 minutes between restart attempts

def check_proxy_health():
    """Check if the Nebula proxy is operational"""
    try:
        response = requests.get(f"{PROXY_URL}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            return True, data
        else:
            return False, f"Status code: {response.status_code}"
    except requests.RequestException as e:
        return False, str(e)
    except Exception as e:
        return False, f"Unknown error: {str(e)}"

def test_nebula_endpoints():
    """Test each Nebula endpoint with a simple request"""
    endpoints = [
        {"name": "predict", "data": {"inputs": {"token": "btc", "timeframe": "1h"}}},
        {"name": "analyze", "data": {"inputs": {"token": "eth", "timeframe": "24h"}}},
        {"name": "whale-tracking", "data": {"inputs": {"token": "sol", "lookback_hours": 24}}},
        {"name": "smart-money", "data": {"inputs": {"token": "bnb"}}}
    ]
    
    results = {}
    
    for endpoint in endpoints:
        try:
            response = requests.post(
                f"{PROXY_URL}/nebula/{endpoint['name']}", 
                json=endpoint['data'],
                headers={"Content-Type": "application/json"},
                timeout=10
            )
            
            if response.status_code == 200:
                results[endpoint['name']] = "OK"
            else:
                results[endpoint['name']] = f"Error: {response.status_code}"
        except Exception as e:
            results[endpoint['name']] = f"Exception: {str(e)}"
    
    return results

def restart_proxy():
    """Attempt to restart the Nebula proxy"""
    logging.warning("Attempting to restart Nebula proxy...")
    
    try:
        # This assumes the proxy is running in the current directory
        # Adjust paths as needed for your environment
        import subprocess
        import sys
        import os
        
        # Kill existing node processes running the proxy
        if sys.platform == 'win32':
            # Windows
            subprocess.run("taskkill /f /im node.exe", shell=True)
        else:
            # Linux/Mac - more targeted approach
            subprocess.run("pkill -f 'node.*nebula-proxy.js'", shell=True)
        
        # Wait a moment
        time.sleep(2)
        
        # Start the proxy in a new process
        proxy_path = os.path.join(os.getcwd(), "nebula-proxy.js")
        
        if sys.platform == 'win32':
            # Windows - use start to run in background
            subprocess.Popen(f"start cmd /c node {proxy_path}", shell=True)
        else:
            # Linux/Mac
            subprocess.Popen(f"node {proxy_path} > logs/nebula-proxy.log 2>&1 &", shell=True)
        
        logging.info("Proxy restart command issued")
        return True
    except Exception as e:
        logging.error(f"Failed to restart proxy: {str(e)}")
        return False

def main():
    failures = 0
    last_restart = 0
    
    logging.info("Nebula proxy monitor started")
    
    while True:
        health_ok, health_data = check_proxy_health()
        
        if health_ok:
            # If proxy is healthy, reset failure counter
            if failures > 0:
                logging.info(f"Proxy recovered after {failures} failed checks")
            failures = 0
            
            # Log health status
            logging.info(f"Proxy health: OK - {health_data}")
            
            # Every 5 minutes, test all endpoints
            if int(time.time()) % 300 < CHECK_INTERVAL:
                endpoint_results = test_nebula_endpoints()
                logging.info(f"Endpoint tests: {json.dumps(endpoint_results)}")
        else:
            # Increment failure counter
            failures += 1
            logging.warning(f"Proxy health check failed ({failures}/{MAX_FAILURES}): {health_data}")
            
            # If we've reached max failures and enough time has passed since last restart
            current_time = time.time()
            if failures >= MAX_FAILURES and (current_time - last_restart) > RESTART_TIMEOUT:
                restart_success = restart_proxy()
                if restart_success:
                    last_restart = current_time
                    failures = 0  # Reset counter after restart attempt
                
        # Wait before next check
        time.sleep(CHECK_INTERVAL)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logging.info("Monitor stopped by user")
    except Exception as e:
        logging.error(f"Critical error: {str(e)}")
        import traceback
        traceback.print_exc()