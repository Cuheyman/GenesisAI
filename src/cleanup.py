#!/usr/bin/env python3
"""
Trading Bot Cleanup Script
==========================
This script removes all logs, databases, and cache files for a fresh start.
"""

import os
import shutil
import glob
from datetime import datetime

def cleanup_bot_data():
    """Remove all bot-generated data files"""
    
    print("Trading Bot Data Cleanup")
    print("=" * 50)
    
    # Counter for deleted items
    deleted_count = 0
    
    # 1. Delete logs directory
    if os.path.exists('logs'):
        try:
            file_count = len(os.listdir('logs'))
            shutil.rmtree('logs')
            print(f"✓ Deleted logs directory ({file_count} files)")
            deleted_count += file_count
        except Exception as e:
            print(f"✗ Error deleting logs: {e}")
    else:
        print("- No logs directory found")
    
    # 2. Delete database files
    db_files = glob.glob('*.db') + glob.glob('*.db-journal') + glob.glob('*.db-wal')
    for db_file in db_files:
        try:
            os.remove(db_file)
            print(f"✓ Deleted database: {db_file}")
            deleted_count += 1
        except Exception as e:
            print(f"✗ Error deleting {db_file}: {e}")
    
    # 3. Delete specific known database files
    known_db_files = [
        'trading_bot.db',
        'bot_data.db',
        'performance.db',
        'trades.db'
    ]
    
    for db_name in known_db_files:
        if os.path.exists(db_name) and db_name not in db_files:
            try:
                os.remove(db_name)
                print(f"✓ Deleted database: {db_name}")
                deleted_count += 1
            except Exception as e:
                print(f"✗ Error deleting {db_name}: {e}")
    
    # 4. Delete cache directories
    cache_dirs = ['__pycache__', '.cache', 'cache']
    for cache_dir in cache_dirs:
        if os.path.exists(cache_dir):
            try:
                shutil.rmtree(cache_dir)
                print(f"✓ Deleted cache directory: {cache_dir}")
                deleted_count += 1
            except Exception as e:
                print(f"✗ Error deleting {cache_dir}: {e}")
    
    # 5. Delete any CSV export files
    csv_files = glob.glob('*.csv')
    for csv_file in csv_files:
        if 'trade' in csv_file.lower() or 'performance' in csv_file.lower():
            try:
                os.remove(csv_file)
                print(f"✓ Deleted CSV file: {csv_file}")
                deleted_count += 1
            except Exception as e:
                print(f"✗ Error deleting {csv_file}: {e}")
    
    # 6. Clean up Python cache files
    for root, dirs, files in os.walk('.'):
        # Skip virtual environments
        if 'venv' in root or 'env' in root:
            continue
            
        for file in files:
            if file.endswith('.pyc') or file.endswith('.pyo'):
                try:
                    os.remove(os.path.join(root, file))
                    deleted_count += 1
                except:
                    pass
    
    print("\n" + "=" * 50)
    print(f"Cleanup complete! Deleted {deleted_count} items.")
    print("Your bot is ready for a fresh start.")
    
    # Create new logs directory
    os.makedirs('logs', exist_ok=True)
    print("\n✓ Created fresh logs directory")

def backup_before_cleanup():
    """Create a backup before cleaning up"""
    backup_dir = f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    print(f"\nCreating backup in {backup_dir}...")
    os.makedirs(backup_dir, exist_ok=True)
    
    # Backup logs
    if os.path.exists('logs'):
        shutil.copytree('logs', os.path.join(backup_dir, 'logs'))
        print("✓ Backed up logs")
    
    # Backup databases
    for db_file in glob.glob('*.db'):
        shutil.copy2(db_file, backup_dir)
        print(f"✓ Backed up {db_file}")
    
    print(f"✓ Backup complete in {backup_dir}")
    
    return backup_dir

def main():
    """Main cleanup function"""
    print("Trading Bot Cleanup Utility")
    print("===========================\n")
    
    # Show what will be deleted
    print("This will delete:")
    print("- All log files in the logs directory")
    print("- All database files (*.db)")
    print("- All cache files and directories")
    print("- Any trading-related CSV exports")
    print()
    
    # Ask for confirmation
    response = input("Do you want to create a backup first? (y/n): ").lower()
    
    if response == 'y':
        backup_dir = backup_before_cleanup()
        print(f"\nBackup created in: {backup_dir}")
    
    response = input("\nProceed with cleanup? (yes/no): ").lower()
    
    if response == 'yes':
        cleanup_bot_data()
    else:
        print("Cleanup cancelled.")

if __name__ == "__main__":
    main()