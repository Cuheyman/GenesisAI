# symbol_validator.py
import logging
from typing import List, Set

class SymbolValidator:
    """Validates trading symbols before making API requests"""
    
    def __init__(self):
        # Core valid symbols - add more as needed
        self.valid_symbols = {
            # Major cryptocurrencies
            'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'XRPUSDT', 'ADAUSDT',
            'SOLUSDT', 'DOTUSDT', 'DOGEUSDT', 'MATICUSDT', 'LTCUSDT',
            'LINKUSDT', 'AVAXUSDT', 'UNIUSDT', 'ATOMUSDT', 'ETCUSDT',
            
            # Popular altcoins
            'SUIUSDT', 'HBARUSDT', 'APTUSDT', 'ARBUSDT', 'OPUSDT',
            'INJUSDT', 'SEIUSDT', 'TIAUSDT', 'NEARUSDT', 'FTMUSDT',
            
            # Meme coins
            'SHIBUSDT', 'PEPEUSDT', 'FLOKIUSDT', 'BONKUSDT', 'WIFUSDT',
            
            # DeFi tokens
            'AAVEUSDT', 'COMPUSDT', 'MKRUSDT', 'SNXUSDT', 'CRVUSDT',
            'SUSHIUSDT', 'YFIUSDT', '1INCHUSDT', 'BALUSDT', 'LDOUSDT',
            
            # Layer 2 & Scaling
            'STXUSDT', 'IMXUSDT', 'MANAUSDT', 'SANDUSDT', 'AXSUSDT',
            'ENJUSDT', 'GALAUSDT', 'ALICEUSDT', 'CHZUSDT', 'FLOWUSDT',
            
            # Exchange tokens
            'FTTUSDT', 'OKBUSDT', 'LEOUSDT', 'HTUSDT', 'KCSUSDT',
            
            # Others
            'ALGOUSDT', 'VETUSDT', 'XLMUSDT', 'XTZUSDT', 'EOSUSDT',
            'ZILUSDT', 'DASHUSDT', 'ZECUSDT', 'XMRUSDT', 'WAVESUSDT',
            'QNTUSDT', 'GRTUSDT', 'FILUSDT', 'ARUSDT', 'KLAYUSDT',
            
            # New trending tokens (verify these exist on Binance)
            'KAITOUSDT', 'PIXELUSDT', 'PENGUUSDT', 'ONDOUSDT', 'PROMUSDT',
            'XTZUSDT', 'SUPERUSDT', 'CATUSDT', 'CHEEMSUSDT', 'SATSUSDT'
        }
        
        # Blacklisted symbols that cause issues
        self.blacklisted_symbols = {
            'SYRUPUSDT',  # This doesn't exist
            'TESTUSDT',   # Test symbols
            'DUMMYUSDT'   # Dummy symbols
        }
        
        logging.info(f"Symbol validator initialized with {len(self.valid_symbols)} valid symbols")
    
    def is_valid_symbol(self, symbol: str) -> bool:
        """Check if a symbol is valid for trading"""
        # Convert to uppercase for consistency
        symbol = symbol.upper()
        
        # Check blacklist first
        if symbol in self.blacklisted_symbols:
            logging.warning(f"Symbol {symbol} is blacklisted")
            return False
        
        # Check if in valid symbols
        if symbol not in self.valid_symbols:
            logging.warning(f"Symbol {symbol} not in valid symbols list")
            return False
        
        return True
    
    def filter_valid_symbols(self, symbols: List[str]) -> List[str]:
        """Filter a list of symbols to only valid ones"""
        valid = []
        for symbol in symbols:
            if self.is_valid_symbol(symbol):
                valid.append(symbol.upper())
            else:
                logging.debug(f"Filtered out invalid symbol: {symbol}")
        
        return valid
    
    def add_symbol(self, symbol: str):
        """Add a new valid symbol"""
        self.valid_symbols.add(symbol.upper())
        logging.info(f"Added {symbol} to valid symbols")
    
    def remove_symbol(self, symbol: str):
        """Remove a symbol from valid list"""
        self.valid_symbols.discard(symbol.upper())
        logging.info(f"Removed {symbol} from valid symbols")
    
    def get_all_valid_symbols(self) -> List[str]:
        """Get all valid symbols as a sorted list"""
        return sorted(list(self.valid_symbols))

# Create a global instance
symbol_validator = SymbolValidator()