import asyncio
import logging
import os
import sys
from datetime import datetime
from hybrid_bot import HybridTradingBot

# Setup logging directory
log_dir = 'logs'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# Setup logging
log_file = os.path.join(log_dir, f'hybrid_bot_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)

if __name__ == "__main__":
    try:
        # Initialize the bot
        bot = HybridTradingBot()
        
        
        # Run the bot
        asyncio.run(bot.run())
    except KeyboardInterrupt:
        logging.info("Bot stopped by user")
    except Exception as e:
        logging.error(f"Critical error: {str(e)}")
        import traceback
        traceback.print_exc()