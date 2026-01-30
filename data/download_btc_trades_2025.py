#!/usr/bin/env python3
"""
Download all BTCUSDT spot trades for 2025 from Binance public data archive.
Run this on your local machine - it bypasses Claude's network restrictions.

Usage:
    python download_btc_trades_2025.py

Output: Creates ./btc_trades_2025/ with daily zip files
"""

import os
import sys
from datetime import datetime, timedelta
import urllib.request
from pathlib import Path

BASE_URL = "https://data.binance.vision/"
SYMBOL = "BTCUSDT"
YEAR = 2025
OUTPUT_DIR = "./btc_trades_2025"

def download_file(url, save_path):
    """Download file with progress bar"""
    try:
        dl_file = urllib.request.urlopen(url)
        length = dl_file.getheader('content-length')
        
        if length:
            length = int(length)
            blocksize = max(4096, length // 100)
        else:
            blocksize = 4096
        
        with open(save_path, 'wb') as out_file:
            dl_progress = 0
            while True:
                buf = dl_file.read(blocksize)
                if not buf:
                    break
                dl_progress += len(buf)
                out_file.write(buf)
                
                if length:
                    done = int(50 * dl_progress / length)
                    sys.stdout.write(f"\r[{'#' * done}{'.' * (50-done)}] {dl_progress}/{length} bytes")
                    sys.stdout.flush()
            print()  # newline after progress bar
        return True
        
    except urllib.error.HTTPError as e:
        if e.code == 404:
            print(f" - File not found (future date or missing data)")
        else:
            print(f" - HTTP Error {e.code}")
        return False
    except Exception as e:
        print(f" - Error: {e}")
        return False

def main():
    # Create output directory
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    
    # Generate all dates from Jan 1, 2025 to today
    start_date = datetime(YEAR, 1, 1)
    end_date = datetime.now()
    
    total_days = (end_date - start_date).days + 1
    dates = [start_date + timedelta(days=i) for i in range(total_days)]
    
    print(f"Downloading {total_days} days of {SYMBOL} trades from 2025...")
    print(f"Output directory: {OUTPUT_DIR}\n")
    
    successful = 0
    failed = 0
    
    for idx, date in enumerate(dates, 1):
        date_str = date.strftime("%Y-%m-%d")
        filename = f"{SYMBOL}-trades-{date_str}.zip"
        url = f"{BASE_URL}data/spot/daily/trades/{SYMBOL}/{filename}"
        save_path = os.path.join(OUTPUT_DIR, filename)
        
        # Skip if already downloaded
        if os.path.exists(save_path):
            print(f"[{idx}/{total_days}] {date_str} - Already exists, skipping")
            successful += 1
            continue
        
        print(f"[{idx}/{total_days}] {date_str}", end=" ")
        
        if download_file(url, save_path):
            successful += 1
        else:
            failed += 1
    
    print(f"\n{'='*60}")
    print(f"Download complete!")
    print(f"Successful: {successful}/{total_days}")
    print(f"Failed: {failed}/{total_days}")
    print(f"{'='*60}")
    
    if successful > 0:
        print(f"\nFiles saved to: {os.path.abspath(OUTPUT_DIR)}")
        print(f"\nNext step: Extract and process the CSV files inside the zips")

if __name__ == "__main__":
    main()
