import asyncio
import logging
import os
import sys
from datetime import datetime
from hybrid_bot import HybridTradingBot
import config

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
    bot = None
    try:
        # Initialize the bot
        bot = HybridTradingBot()
        # Run the bot with mode selection
        if getattr(config, "USE_TIERED_GUIDE_MODE", False):
            asyncio.run(bot.tiered_trading_task())
        else:
            asyncio.run(bot.run())
    except KeyboardInterrupt:
        logging.info("Bot stopped by user")
    except Exception as e:
        logging.error(f"Critical error: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        # Ensure cleanup happens
        if bot:
            try:
                asyncio.run(bot.cleanup())
                logging.info("Bot cleanup completed")
            except Exception as e:
                logging.error(f"Error during cleanup: {str(e)}")