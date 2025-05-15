import sqlite3
import logging
import asyncio
import time
from typing import List, Dict, Any, Tuple
import os

class DatabaseManager:
    def __init__(self, db_path):
        self.db_path = db_path
        self.db_lock = asyncio.Lock()
        self.conn = None
        self.init_db()
        
    def init_db(self):
        """Initialize database with basic schema"""
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create trades table if not exists
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trade_id TEXT,
                    pair TEXT NOT NULL,
                    type TEXT NOT NULL,
                    price REAL NOT NULL,
                    quantity REAL NOT NULL,
                    value REAL NOT NULL,
                    profit_loss REAL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create balances table if not exists
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS balances (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    pair TEXT NOT NULL,
                    amount REAL NOT NULL,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create performance table if not exists
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    equity REAL,
                    total_equity REAL,
                    daily_pl REAL,
                    weekly_pl REAL,
                    monthly_pl REAL,
                    win_rate REAL,
                    win_count INTEGER,
                    loss_count INTEGER,
                    profit_factor REAL
                )
            ''')
            
            # Create positions table if not exists
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS positions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    pair TEXT NOT NULL,
                    entry_price REAL NOT NULL,
                    quantity REAL NOT NULL,
                    entry_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    stop_loss REAL,
                    take_profit REAL,
                    status TEXT DEFAULT 'open'
                )
            ''')
            
            # Create strategies table if not exists
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS strategies (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    pair TEXT NOT NULL,
                    profit_loss REAL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create bot_stats table if not exists
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS bot_stats (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    key TEXT NOT NULL,
                    value TEXT NOT NULL,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create equity_history table if not exists
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS equity_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp INTEGER,
                    equity REAL
                )
            ''')
            
            conn.commit()
            logging.info("Database initialization complete")
            
        except sqlite3.Error as e:
            logging.error(f"Database initialization error: {e}")
        finally:
            if conn:
                conn.close()
    
    def update_schema(self):
        """Update database schema with new columns"""
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get current columns in positions table
            cursor.execute("PRAGMA table_info(positions)")
            existing_columns = [col[1] for col in cursor.fetchall()]
            
            # Add new columns to positions table if they don't exist
            schema_updates = []
            
            if 'take_profit_pct' not in existing_columns:
                schema_updates.append("ALTER TABLE positions ADD COLUMN take_profit_pct REAL")
                
            if 'stop_loss_pct' not in existing_columns:
                schema_updates.append("ALTER TABLE positions ADD COLUMN stop_loss_pct REAL")
                
            if 'is_new_asset' not in existing_columns:
                schema_updates.append("ALTER TABLE positions ADD COLUMN is_new_asset INTEGER DEFAULT 0")
                
            if 'exit_price' not in existing_columns:
                schema_updates.append("ALTER TABLE positions ADD COLUMN exit_price REAL")
                
            if 'exit_time' not in existing_columns:
                schema_updates.append("ALTER TABLE positions ADD COLUMN exit_time TEXT")
                
            if 'profit_loss' not in existing_columns:
                schema_updates.append("ALTER TABLE positions ADD COLUMN profit_loss REAL")
                
            if 'signal_source' not in existing_columns:
                schema_updates.append("ALTER TABLE positions ADD COLUMN signal_source TEXT")
                
            if 'signal_strength' not in existing_columns:
                schema_updates.append("ALTER TABLE positions ADD COLUMN signal_strength REAL")
                
            # Check for trades table updates
            cursor.execute("PRAGMA table_info(trades)")
            existing_columns = [col[1] for col in cursor.fetchall()]
            
            if 'signal_source' not in existing_columns:
                schema_updates.append("ALTER TABLE trades ADD COLUMN signal_source TEXT")
                
            if 'signal_strength' not in existing_columns:
                schema_updates.append("ALTER TABLE trades ADD COLUMN signal_strength REAL")
                
            # Execute all schema updates
            for update in schema_updates:
                try:
                    cursor.execute(update)
                    logging.info(f"Schema update applied: {update}")
                except sqlite3.Error as e:
                    logging.error(f"Error updating schema: {e}")
                    
            conn.commit()
            
            if schema_updates:
                logging.info(f"Database schema updated successfully with {len(schema_updates)} changes")
                
        except sqlite3.Error as e:
            logging.error(f"Schema update error: {e}")
        finally:
            if conn:
                conn.close()
    
    async def execute_query(self, query, params=(), commit=False, fetch_all=False, fetch_one=False):
        """Execute a database query with proper error handling"""
        async with self.db_lock:
            conn = None
            result = None
            max_retries = 3
            retry_delay = 0.5  # Start with 0.5 second delay
            
            for attempt in range(max_retries):
                try:
                    # Create a new connection for this specific query
                    conn = sqlite3.connect(self.db_path, timeout=10)
                    cursor = conn.cursor()
                    cursor.execute(query, params)
                    
                    if fetch_all:
                        result = cursor.fetchall()
                    elif fetch_one:
                        result = cursor.fetchone()
                        
                    if commit:
                        conn.commit()
                        
                    return result
                    
                except sqlite3.OperationalError as e:
                    if "database is locked" in str(e) and attempt < max_retries - 1:
                        # Retry with exponential backoff
                        logging.debug(f"Database locked, retrying in {retry_delay}s (attempt {attempt+1}/{max_retries})")
                        await asyncio.sleep(retry_delay)
                        retry_delay *= 2  # Exponential backoff
                    else:
                        logging.error(f"Database error: {str(e)}")
                        raise
                except Exception as e:
                    logging.error(f"Error executing query: {str(e)}")
                    raise
                finally:
                    if conn:
                        conn.close()
            
            return result
    
    def execute_query_sync(self, query, params=(), commit=False, fetch_all=False, fetch_one=False):
        """Synchronous version of execute_query for use in non-async contexts"""
        conn = None
        result = None
        max_retries = 3
        retry_delay = 0.5  # Start with 0.5 second delay
        
        for attempt in range(max_retries):
            try:
                # Create a new connection for this specific query
                conn = sqlite3.connect(self.db_path, timeout=10)
                cursor = conn.cursor()
                cursor.execute(query, params)
                
                if fetch_all:
                    result = cursor.fetchall()
                elif fetch_one:
                    result = cursor.fetchone()
                    
                if commit:
                    conn.commit()
                    
                return result
                
            except sqlite3.OperationalError as e:
                if "database is locked" in str(e) and attempt < max_retries - 1:
                    # Retry with exponential backoff
                    logging.debug(f"Database locked, retrying in {retry_delay}s (attempt {attempt+1}/{max_retries})")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    logging.error(f"Database error: {str(e)}")
                    raise
            except Exception as e:
                logging.error(f"Error executing query: {str(e)}")
                raise
            finally:
                if conn:
                    conn.close()
        
        return result