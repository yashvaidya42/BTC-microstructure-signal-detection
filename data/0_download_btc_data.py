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
import json
BASE_URL = "https://data.binance.vision/"
SYMBOL = "BTCUSDT"
YEAR = 2026
OUTPUT_DIR = "./btc_trades_2021"

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
    # --- Configuration ---
    BASE_OUTPUT_DIR = "."
    START_YEAR = 2026  
    END_YEAR = 2026    
    
    # Initialize dictionary to track the layout
    data_layout = {
        "dataset": "BTCUSDT Spot Trades",
        "range": f"{START_YEAR}-{END_YEAR}",
        "structure": {}
    }

    print(f"Starting batch download for {START_YEAR}-{END_YEAR}...")
    print(f"Root Directory: {BASE_OUTPUT_DIR}\n")

    for year in range(START_YEAR, END_YEAR + 1):
        # 1. Create a specific folder for this year
        year_dir = os.path.join(BASE_OUTPUT_DIR, str(year))
        Path(year_dir).mkdir(parents=True, exist_ok=True)
        
        # 2. Add to JSON manifest
        data_layout["structure"][year] = {
            "path": year_dir,
            "files": []
        }

        # 3. Calculate dates for this specific year
        start_date = datetime(year, 1, 1)
        
        # If it's the current year, stop at yesterday; otherwise go to Dec 31
        if year == datetime.now().year:
            # Subtract 1 day to stop at yesterday
            end_date = datetime.now() - timedelta(days=1) 
        else:
            end_date = datetime(year, 12, 31)

        total_days = (end_date - start_date).days + 1
        print(f"--- Processing {year} ({total_days} days) ---")

        for i in range(total_days):
            current_date = start_date + timedelta(days=i)
            date_str = current_date.strftime("%Y-%m-%d")
            
            # Construct filename and URL
            filename = f"{SYMBOL}-trades-{date_str}.zip"
            url = f"{BASE_URL}data/spot/daily/trades/{SYMBOL}/{filename}"
            save_path = os.path.join(year_dir, filename)
            
            # Skip if exists
            if os.path.exists(save_path):
                # Optional: Uncomment to see skipped files
                # print(f"[{year}] {date_str} - Exists")
                data_layout["structure"][year]["files"].append(filename)
                continue
            
            print(f"[{year}] Downloading {date_str}...", end=" ")
            
            if download_file(url, save_path):
                data_layout["structure"][year]["files"].append(filename)
            else:
                # 404s are expected for today's data (not uploaded yet) or very old data moved to monthly
                pass

    # 4. Output the JSON Manifest
    json_path = os.path.join(BASE_OUTPUT_DIR, "data_layout.json")
    with open(json_path, 'w') as f:
        json.dump(data_layout, f, indent=2)

    print(f"\n{'='*60}")
    print(f"All downloads complete.")
    print(f"Data layout map saved to: {os.path.abspath(json_path)}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()