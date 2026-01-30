#!/usr/bin/env python3
"""
HIGH-PERFORMANCE PREPROCESSING - FULLY TIME-BASED
==================================================
ALL WINDOWS ARE NOW TIME-BASED (seconds), NOT TRADE-COUNT BASED.

This ensures consistent feature computation regardless of market activity:
- High volume day (8000 trades/min): 5 min window = 5 min
- Low volume day (450 trades/min): 5 min window = 5 min

OPTIMIZATIONS:
1. Numba JIT for time-based rolling computations with O(n) sliding window
2. Parallel batch processing with multiprocessing
3. Vectorized feature computation
4. Boundary correction for parallel batches
"""

import os
import zipfile
import pandas as pd
import numpy as np
import pyarrow.parquet as pq
import pyarrow as pa
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import gc
from concurrent.futures import ThreadPoolExecutor
import warnings

warnings.filterwarnings('ignore')

# Try to import numba
try:
    from numba import jit, prange
    HAS_NUMBA = True
    print("[OK] Numba available - using JIT compilation")
except ImportError:
    HAS_NUMBA = False
    print("[WARN] Numba not available - using pure numpy (slower)")

# =============================================================================
# CONFIGURATION
# =============================================================================

INPUT_DIR = r"C:\Users\yashv\Desktop\personal_work\quant_ML_model\data\btc_trades_2025"
OUTPUT_FILE = "./btc_processed_stationary.parquet"

DELETE_ZIPS_AFTER_PROCESSING = False

# =============================================================================
# TIME-BASED WINDOWS (ALL IN SECONDS)
# =============================================================================

LOOK_AHEAD_SECONDS = 300          # 5 min - prediction horizon

VOLATILITY_WINDOW_SEC = 300       # 5 min - for price_volatility feature
VOLUME_WINDOW_SEC = 300           # 5 min - for rolling_signed_volume
ROLLING_STATS_WINDOW_SEC = 60     # 1 min - for short-term features

# =============================================================================
# LABELING THRESHOLDS (tuned for 15-20% UP/DOWN)
# =============================================================================

Z_SCORE_THRESHOLD = 0.2         # Lower = more UP/DOWN labels
MIN_PROFIT_PCT = 0.0004         # 0.04% fee floor

# =============================================================================
# PERFORMANCE TUNING
# =============================================================================

BATCH_SIZE = 3                    # Days per batch
IO_THREADS = 6                    # Parallel I/O threads (increased for 12-thread CPU)

# Buffer for boundary correction: 10 minutes in microseconds
BOUNDARY_BUFFER_SEC = 600         # 10 minutes
BOUNDARY_BUFFER_US = BOUNDARY_BUFFER_SEC * 1_000_000

print(f"[CONFIG] Look-ahead: {LOOK_AHEAD_SECONDS}s, Volatility window: {VOLATILITY_WINDOW_SEC}s")
print(f"[CONFIG] Z-threshold: {Z_SCORE_THRESHOLD}, Min profit: {MIN_PROFIT_PCT*100:.2f}%")


# =============================================================================
# TIME-BASED ROLLING FUNCTIONS - O(n) SLIDING WINDOW
# =============================================================================

if HAS_NUMBA:
    @jit(nopython=True, cache=True)
    def rolling_std_time_based(timestamps_us, values, window_seconds):
        """
        Time-based rolling standard deviation using O(n) sliding window.
        
        Maintains running sum, sum of squares, and count to avoid recomputation.
        Uses Welford-style online variance for numerical stability.
        
        Args:
            timestamps_us: Timestamps in microseconds (must be sorted)
            values: Values to compute std over
            window_seconds: Window size in seconds
        
        Returns:
            Array of rolling std values
        """
        n = len(values)
        result = np.empty(n, dtype=np.float64)
        window_us = window_seconds * 1_000_000
        
        left = 0
        running_sum = 0.0
        running_sum_sq = 0.0
        
        for right in range(n):
            # Add current element to window
            running_sum += values[right]
            running_sum_sq += values[right] * values[right]
            
            # Shrink window from left while timestamps fall out of window
            min_time = timestamps_us[right] - window_us
            while left < right and timestamps_us[left] < min_time:
                running_sum -= values[left]
                running_sum_sq -= values[left] * values[left]
                left += 1
            
            # Compute std for window [left, right]
            count = right - left + 1
            if count >= 2:
                mean = running_sum / count
                # Variance = E[X^2] - E[X]^2
                variance = (running_sum_sq / count) - (mean * mean)
                # Handle numerical issues (can be slightly negative due to floating point)
                if variance < 0.0:
                    variance = 0.0
                result[right] = np.sqrt(variance)
            else:
                result[right] = 0.0
        
        return result

    @jit(nopython=True, cache=True)
    def rolling_sum_time_based(timestamps_us, values, window_seconds):
        """
        Time-based rolling sum using O(n) sliding window.
        
        Args:
            timestamps_us: Timestamps in microseconds (must be sorted)
            values: Values to sum
            window_seconds: Window size in seconds
        
        Returns:
            Array of rolling sum values
        """
        n = len(values)
        result = np.empty(n, dtype=np.float64)
        window_us = window_seconds * 1_000_000
        
        left = 0
        running_sum = 0.0
        
        for right in range(n):
            # Add current element
            running_sum += values[right]
            
            # Shrink window from left
            min_time = timestamps_us[right] - window_us
            while left < right and timestamps_us[left] < min_time:
                running_sum -= values[left]
                left += 1
            
            result[right] = running_sum
        
        return result

    @jit(nopython=True, cache=True)
    def rolling_mean_time_based(timestamps_us, values, window_seconds):
        """
        Time-based rolling mean using O(n) sliding window.
        
        Args:
            timestamps_us: Timestamps in microseconds (must be sorted)
            values: Values to average
            window_seconds: Window size in seconds
        
        Returns:
            Array of rolling mean values
        """
        n = len(values)
        result = np.empty(n, dtype=np.float64)
        window_us = window_seconds * 1_000_000
        
        left = 0
        running_sum = 0.0
        
        for right in range(n):
            # Add current element
            running_sum += values[right]
            
            # Shrink window from left
            min_time = timestamps_us[right] - window_us
            while left < right and timestamps_us[left] < min_time:
                running_sum -= values[left]
                left += 1
            
            count = right - left + 1
            result[right] = running_sum / count
        
        return result

    @jit(nopython=True, cache=True)
    def find_future_price_time_based(timestamps_us, prices, look_ahead_us):
        """Find price at t + look_ahead for each timestamp."""
        n = len(timestamps_us)
        future_prices = np.empty(n, dtype=np.float64)
        
        for i in range(n):
            target_time = timestamps_us[i] + look_ahead_us
            
            # Binary search for target_time
            left = np.int64(i)
            right = np.int64(n)
            while left < right:
                mid = (left + right) // 2
                if timestamps_us[mid] < target_time:
                    left = mid + 1
                else:
                    right = mid
            
            if left < n:
                future_prices[i] = prices[left]
            else:
                future_prices[i] = np.nan
        
        return future_prices

else:
    # Pure numpy fallbacks with O(n) sliding window
    def rolling_std_time_based(timestamps_us, values, window_seconds):
        """Numpy O(n) sliding window implementation."""
        n = len(values)
        result = np.empty(n, dtype=np.float64)
        window_us = window_seconds * 1_000_000
        
        left = 0
        running_sum = 0.0
        running_sum_sq = 0.0
        
        for right in range(n):
            running_sum += values[right]
            running_sum_sq += values[right] * values[right]
            
            min_time = timestamps_us[right] - window_us
            while left < right and timestamps_us[left] < min_time:
                running_sum -= values[left]
                running_sum_sq -= values[left] * values[left]
                left += 1
            
            count = right - left + 1
            if count >= 2:
                mean = running_sum / count
                variance = max(0.0, (running_sum_sq / count) - (mean * mean))
                result[right] = np.sqrt(variance)
            else:
                result[right] = 0.0
        
        return result

    def rolling_sum_time_based(timestamps_us, values, window_seconds):
        """Numpy O(n) sliding window implementation."""
        n = len(values)
        result = np.empty(n, dtype=np.float64)
        window_us = window_seconds * 1_000_000
        
        left = 0
        running_sum = 0.0
        
        for right in range(n):
            running_sum += values[right]
            
            min_time = timestamps_us[right] - window_us
            while left < right and timestamps_us[left] < min_time:
                running_sum -= values[left]
                left += 1
            
            result[right] = running_sum
        
        return result

    def rolling_mean_time_based(timestamps_us, values, window_seconds):
        """Numpy O(n) sliding window implementation."""
        n = len(values)
        result = np.empty(n, dtype=np.float64)
        window_us = window_seconds * 1_000_000
        
        left = 0
        running_sum = 0.0
        
        for right in range(n):
            running_sum += values[right]
            
            min_time = timestamps_us[right] - window_us
            while left < right and timestamps_us[left] < min_time:
                running_sum -= values[left]
                left += 1
            
            count = right - left + 1
            result[right] = running_sum / count
        
        return result

    def find_future_price_time_based(timestamps_us, prices, look_ahead_us):
        """Numpy vectorized implementation."""
        n = len(timestamps_us)
        future_prices = np.full(n, np.nan)
        target_times = timestamps_us + look_ahead_us
        future_indices = np.searchsorted(timestamps_us, target_times, side='left')
        valid_mask = future_indices < n
        future_indices = np.clip(future_indices, 0, n - 1)
        future_prices[valid_mask] = prices[future_indices[valid_mask]]
        return future_prices


# =============================================================================
# FEATURE COMPUTATION (FULLY TIME-BASED)
# =============================================================================

def compute_features_time_based(df, verbose=True):
    """
    Compute all features using TIME-BASED rolling windows.
    
    This ensures consistent feature computation regardless of trade frequency.
    """
    n = len(df)
    
    # Extract arrays
    timestamps_us = df['timestamp'].values.astype(np.int64)
    prices = df['price'].values.astype(np.float64)
    qty = df['qty'].values.astype(np.float64)
    side = df['side'].values.astype(np.float64)
    
    # === BASIC FEATURES ===
    time_delta = np.zeros(n, dtype=np.float64)
    time_delta[1:] = (timestamps_us[1:] - timestamps_us[:-1]) / 1e6  # seconds
    
    log_return = np.zeros(n, dtype=np.float64)
    log_return[1:] = np.log(prices[1:] / prices[:-1])
    
    trade_intensity = 1.0 / (time_delta + 1e-9)
    
    price_acceleration = np.zeros(n, dtype=np.float64)
    price_acceleration[1:] = log_return[1:] - log_return[:-1]
    
    # === VOLUME FEATURES (TIME-BASED) ===
    buy_volume = qty * (side == 1).astype(np.float64)
    sell_volume = qty * (side == -1).astype(np.float64)
    
    if verbose:
        print("  Computing time-based rolling features...")
    
    rolling_buy_vol = rolling_sum_time_based(timestamps_us, buy_volume, ROLLING_STATS_WINDOW_SEC)
    rolling_sell_vol = rolling_sum_time_based(timestamps_us, sell_volume, ROLLING_STATS_WINDOW_SEC)
    
    volume_imbalance = (rolling_buy_vol - rolling_sell_vol) / (rolling_buy_vol + rolling_sell_vol + 1e-9)
    
    # === SIGNED VOLUME (TIME-BASED) ===
    signed_volume = qty * side
    rolling_signed_volume = rolling_sum_time_based(timestamps_us, signed_volume, VOLUME_WINDOW_SEC)
    
    # === VOLATILITY FEATURES (TIME-BASED) ===
    price_volatility = rolling_std_time_based(timestamps_us, log_return, VOLATILITY_WINDOW_SEC)
    volume_volatility = rolling_std_time_based(timestamps_us, qty, ROLLING_STATS_WINDOW_SEC)
    time_delta_std = rolling_std_time_based(timestamps_us, time_delta, ROLLING_STATS_WINDOW_SEC)
    
    # === PRICE FEATURES (TIME-BASED) ===
    price_ma = rolling_mean_time_based(timestamps_us, prices, ROLLING_STATS_WINDOW_SEC)
    price_distance_from_ma = (prices - price_ma) / (price_ma + 1e-9)
    
    # === TIME-BASED LABELING ===
    look_ahead_us = LOOK_AHEAD_SECONDS * 1_000_000
    future_price = find_future_price_time_based(timestamps_us, prices, look_ahead_us)
    future_return = (future_price - prices) / prices
    
    # Volatility is already time-based (5 min window), so scaling is simpler
    # Just use volatility directly - it's already "per 5 minutes"
    volatility_scaled = price_volatility
    
    # Thresholds
    upper_threshold = Z_SCORE_THRESHOLD * volatility_scaled
    lower_threshold = -Z_SCORE_THRESHOLD * volatility_scaled
    
    # Labels
    labels = np.ones(n, dtype=np.int8)  # Default: NEUTRAL
    
    up_condition = (future_return > upper_threshold) & (future_return > MIN_PROFIT_PCT)
    down_condition = (future_return < lower_threshold) & (future_return < -MIN_PROFIT_PCT)
    
    labels[up_condition] = 2
    labels[down_condition] = 0
    
    # Invalid
    invalid = (
        np.isnan(future_price) | 
        np.isnan(price_volatility) | 
        (price_volatility <= 0) |
        np.isnan(future_return)
    )
    labels[invalid] = -1
    
    # Build result
    result = pd.DataFrame({
        'timestamp': timestamps_us,
        'datetime': df['datetime'].values,
        'qty': qty.astype(np.float32),
        'time_delta': time_delta.astype(np.float32),
        'log_return': log_return.astype(np.float32),
        'trade_intensity': trade_intensity.astype(np.float32),
        'price_acceleration': price_acceleration.astype(np.float32),
        'volume_imbalance': volume_imbalance.astype(np.float32),
        'rolling_signed_volume': rolling_signed_volume.astype(np.float32),
        'side': side.astype(np.int8),
        'price_volatility': price_volatility.astype(np.float32),
        'volume_volatility': volume_volatility.astype(np.float32),
        'time_delta_std': time_delta_std.astype(np.float32),
        'price_distance_from_ma': price_distance_from_ma.astype(np.float32),
        'label': labels,
    })
    
    return result


# =============================================================================
# SINGLE DAY PROCESSOR
# =============================================================================

def load_zip_to_df(zip_path):
    """Load a single zip file into DataFrame."""
    with zipfile.ZipFile(zip_path, 'r') as zf:
        csv_name = Path(zip_path).stem + '.csv'
        with zf.open(csv_name) as csvf:
            df = pd.read_csv(csvf, header=None)
            df.columns = ['trade_id', 'price', 'qty', 'quote_qty', 
                          'timestamp', 'is_buyer_maker', 'is_best_match']
    return df


def process_single_day(zip_path, prev_tail_df=None, verbose=True):
    """
    Process one day with time-based features.
    
    Args:
        zip_path: Path to zip file
        prev_tail_df: DataFrame with last N minutes from previous day
        verbose: Whether to print progress
    
    Returns:
        (result_df, tail_df, raw_df_for_boundary) or None on error
    """
    try:
        zip_path = Path(zip_path)
        if verbose:
            print(f"  Processing {zip_path.name}...")
        
        # Load data
        df = load_zip_to_df(zip_path)
        
        if len(df) == 0:
            return None
        
        df = df.sort_values('timestamp').reset_index(drop=True)
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='us')
        df['side'] = df['is_buyer_maker'].map({True: -1, False: 1})
        
        # Keep only needed columns
        df = df[['timestamp', 'price', 'qty', 'side', 'datetime']].copy()
        
        # Concatenate with previous tail for window continuity
        if prev_tail_df is not None and len(prev_tail_df) > 0:
            df_combined = pd.concat([prev_tail_df, df], ignore_index=True)
            tail_len = len(prev_tail_df)
        else:
            df_combined = df
            tail_len = 0
        
        # Compute features
        result = compute_features_time_based(df_combined, verbose=verbose)
        
        # Remove previous tail rows
        if tail_len > 0:
            result = result.iloc[tail_len:].copy()
        
        # Remove invalid labels
        result = result[result['label'] != -1].copy()
        result = result.dropna()
        
        if len(result) == 0:
            return None
        
        # Prepare tail for next day
        # Keep enough data for largest window (VOLATILITY_WINDOW_SEC = 300s = 5 min)
        # Plus some buffer = 10 min total
        buffer_us = 10 * 60 * 1_000_000
        last_ts = df_combined['timestamp'].iloc[-1]
        tail_mask = df_combined['timestamp'] >= (last_ts - buffer_us)
        tail_df = df_combined[tail_mask][['timestamp', 'price', 'qty', 'side', 'datetime']].copy()
        
        # Also keep raw data for the first ~10 minutes (for boundary correction in parallel processing)
        first_ts = df['timestamp'].iloc[0]
        boundary_mask = df['timestamp'] <= (first_ts + BOUNDARY_BUFFER_US)
        boundary_raw_df = df[boundary_mask][['timestamp', 'price', 'qty', 'side', 'datetime']].copy()
        
        # Print label distribution for this day
        if verbose:
            counts = np.bincount(result['label'].values.astype(int), minlength=3)
            total = counts.sum()
            print(f"    Rows: {total:,} | DOWN: {counts[0]/total*100:.1f}% | NEUTRAL: {counts[1]/total*100:.1f}% | UP: {counts[2]/total*100:.1f}%")
        
        return result, tail_df, boundary_raw_df
        
    except Exception as e:
        print(f"  Error processing {zip_path}: {e}")
        return None


# =============================================================================
# BATCH PROCESSOR (SEQUENTIAL WITH IMMEDIATE DISK WRITES)
# =============================================================================

def process_batch_sequential(zip_paths, initial_tail, temp_dir, start_file_idx):
    """
    Process batch of days sequentially, writing each day to disk immediately.
    
    This avoids memory buildup from holding multiple large DataFrames.
    
    Args:
        zip_paths: List of zip file paths
        initial_tail: Tail DataFrame from previous batch
        temp_dir: Directory for intermediate parquet files
        start_file_idx: Starting index for file naming
    
    Returns:
        (list of parquet paths, final tail DataFrame, label counts dict, total rows)
    """
    parquet_paths = []
    current_tail = initial_tail
    label_counts = {0: 0, 1: 0, 2: 0}
    total_rows = 0
    
    for i, zip_path in enumerate(zip_paths):
        result = process_single_day(zip_path, current_tail, verbose=True)
        
        if result is not None:
            df, current_tail, _ = result  # We don't need boundary_raw for sequential
            
            # Count labels
            total_rows += len(df)
            for lv in [0, 1, 2]:
                label_counts[lv] += (df['label'] == lv).sum()
            
            # Write immediately to disk
            file_idx = start_file_idx + i
            path = temp_dir / f"day_{file_idx:04d}.parquet"
            df.to_parquet(path, engine='pyarrow', compression='snappy', index=False)
            parquet_paths.append(path)
            
            # Free memory
            del df
            gc.collect()
        else:
            current_tail = None
    
    return parquet_paths, current_tail, label_counts, total_rows


# =============================================================================
# MAIN PROCESSOR
# =============================================================================

def process_streaming(input_dir, output_file, delete_zips=False):
    """Process all zips with time-based features."""
    print("=" * 80)
    print("FULLY TIME-BASED PREPROCESSING (OPTIMIZED)")
    print("=" * 80)
    print(f"Input: {input_dir}")
    print(f"Output: {output_file}")
    print(f"\nTime-Based Windows:")
    print(f"  Look-ahead:        {LOOK_AHEAD_SECONDS} sec ({LOOK_AHEAD_SECONDS/60:.1f} min)")
    print(f"  Volatility window: {VOLATILITY_WINDOW_SEC} sec ({VOLATILITY_WINDOW_SEC/60:.1f} min)")
    print(f"  Volume window:     {VOLUME_WINDOW_SEC} sec ({VOLUME_WINDOW_SEC/60:.1f} min)")
    print(f"  Stats window:      {ROLLING_STATS_WINDOW_SEC} sec ({ROLLING_STATS_WINDOW_SEC/60:.1f} min)")
    print(f"\nLabeling Thresholds:")
    print(f"  Z-Score:    {Z_SCORE_THRESHOLD}")
    print(f"  Min Profit: {MIN_PROFIT_PCT*100:.3f}%")
    print(f"\nPerformance:")
    print(f"  Numba JIT:     {HAS_NUMBA}")
    print(f"  Batch size:    {BATCH_SIZE} days")
    print(f"  I/O threads:   {IO_THREADS}")
    print(f"  Rolling functions: O(n) sliding window")
    print("=" * 80 + "\n")
    
    zip_files = sorted(Path(input_dir).glob("*.zip"))
    if len(zip_files) == 0:
        raise FileNotFoundError(f"No zip files in {input_dir}")
    
    print(f"Found {len(zip_files)} zip files\n")
    
    temp_dir = Path("./temp_daily_parquets")
    temp_dir.mkdir(exist_ok=True)
    
    total_rows = 0
    label_distribution = {0: 0, 1: 0, 2: 0}
    all_parquet_paths = []
    
    current_tail = None
    file_idx = 0
    
    # Process in batches, writing each day to disk immediately
    for batch_start in range(0, len(zip_files), BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, len(zip_files))
        batch_files = zip_files[batch_start:batch_end]
        
        print(f"\n[Batch {batch_start//BATCH_SIZE + 1}] Processing days {batch_start+1}-{batch_end}...")
        
        # Process batch sequentially, writing to disk immediately
        paths, current_tail, batch_labels, batch_rows = process_batch_sequential(
            batch_files, current_tail, temp_dir, file_idx
        )
        
        # Accumulate results
        all_parquet_paths.extend(paths)
        total_rows += batch_rows
        for lv in [0, 1, 2]:
            label_distribution[lv] += batch_labels[lv]
        
        file_idx += len(paths)
        gc.collect()
    
    if len(all_parquet_paths) == 0:
        raise Exception("No days processed successfully")
    
    # Combine all days
    print(f"\n\nCombining {len(all_parquet_paths)} files into single parquet...\n")
    
    writer = None
    for parquet_path in tqdm(all_parquet_paths, desc="Combining"):
        table = pq.read_table(parquet_path)
        
        if writer is None:
            writer = pq.ParquetWriter(output_file, table.schema, compression='snappy')
        
        writer.write_table(table)
        del table
    
    if writer:
        writer.close()
    
    # Cleanup
    for p in all_parquet_paths:
        p.unlink()
    temp_dir.rmdir()
    
    # Final stats
    parquet_file = pq.ParquetFile(output_file)
    actual_rows = parquet_file.metadata.num_rows
    file_size = os.path.getsize(output_file) / 1e9
    
    print("\n" + "=" * 80)
    print("PREPROCESSING COMPLETE")
    print("=" * 80)
    print(f"Total rows: {actual_rows:,}")
    print(f"File size: {file_size:.2f} GB")
    
    print("\nFINAL LABEL DISTRIBUTION:")
    total = sum(label_distribution.values())
    for lv in [0, 1, 2]:
        count = label_distribution[lv]
        pct = (count / total * 100) if total > 0 else 0
        label_name = ['DOWN', 'NEUTRAL', 'UP'][lv]
        print(f"  {label_name:>7}: {count:>12,} ({pct:>5.2f}%)")
    
    target_achieved = (
        10 <= label_distribution[0]/total*100 <= 25 and
        50 <= label_distribution[1]/total*100 <= 80 and
        10 <= label_distribution[2]/total*100 <= 25
    )
    
    if target_achieved:
        print("\n  [OK] Label distribution is in target range!")
    else:
        print("\n  [WARN] Label distribution outside target range.")
        print("         Adjust Z_SCORE_THRESHOLD or MIN_PROFIT_PCT")
    
    print("\n" + "=" * 80)


# =============================================================================
# MAIN
# =============================================================================

def main():
    start_time = datetime.now()
    print(f"\nStart time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    try:
        process_streaming(INPUT_DIR, OUTPUT_FILE, DELETE_ZIPS_AFTER_PROCESSING)
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        print(f"Duration: {duration:.1f} seconds ({duration/60:.1f} minutes)")
        
        pf = pq.ParquetFile(OUTPUT_FILE)
        rows = pf.metadata.num_rows
        print(f"Throughput: {rows/duration:,.0f} rows/second\n")
        
    except Exception as e:
        print("\n" + "=" * 80)
        print("ERROR OCCURRED")
        print("=" * 80)
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()