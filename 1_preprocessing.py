#!/usr/bin/env python3
"""
HIGH-PERFORMANCE PREPROCESSING - HYBRID MEMORY-AWARE
=====================================================
Optimized for: Intel i5-11400H (6 cores, 12 threads), 8GB RAM

KEY FEATURES:
1. O(n) sliding window rolling functions (Numba JIT)
2. Hybrid parallel/sequential processing with RAM monitoring
3. NO COMBINE STEP - outputs individual daily parquets directly
4. Automatic metadata.json generation for downstream consumption

OUTPUT: ./processed_daily_parquets/day_XXXX.parquet + metadata.json
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
import multiprocessing as mp
import time
import json
import warnings

warnings.filterwarnings('ignore')

# Try to import psutil for memory monitoring
try:
    import psutil
    HAS_PSUTIL = True
    print("[OK] psutil available - memory monitoring enabled")
except ImportError:
    HAS_PSUTIL = False
    print("[WARN] psutil not available - using sequential mode for safety")

# Try to import numba
try:
    from numba import jit
    HAS_NUMBA = True
    print("[OK] Numba available - using JIT compilation")
except ImportError:
    HAS_NUMBA = False
    print("[WARN] Numba not available - using pure numpy (slower)")

# =============================================================================
# CONFIGURATION
# =============================================================================

INPUT_DIR = r"C:\Users\yashv\Desktop\personal_work\quant_ML_model\data\btc_trades_2025"
OUTPUT_DIR = "./processed_daily_parquets"  # Final output directory (no temp)

DELETE_ZIPS_AFTER_PROCESSING = False

# =============================================================================
# TIME-BASED WINDOWS (ALL IN SECONDS)
# =============================================================================

LOOK_AHEAD_SECONDS = 300          # 5 min - prediction horizon
VOLATILITY_WINDOW_SEC = 300       # 5 min - for price_volatility feature
VOLUME_WINDOW_SEC = 300           # 5 min - for rolling_signed_volume
ROLLING_STATS_WINDOW_SEC = 60     # 1 min - for short-term features

# =============================================================================
# LABELING THRESHOLDS
# =============================================================================

Z_SCORE_THRESHOLD = 1           # Adaptive threshold multiplier
MIN_PROFIT_PCT = 0.001           # 0.07% minimum move (fee floor)

# =============================================================================
# PERFORMANCE TUNING (Optimized for i5-11400H + 8GB RAM)
# =============================================================================

BATCH_SIZE = 2                    # Days per batch (more work per IPC call)
NUM_WORKERS = 6                   # Conservative: 2 workers to avoid OOM

# =============================================================================
# MEMORY MANAGEMENT
# =============================================================================

RAM_THRESHOLD_PERCENT = 85        # Switch to sequential above this
RAM_SAFE_PERCENT = 70             # Resume parallel below this

# Buffer for boundary correction: 10 minutes in microseconds
BOUNDARY_BUFFER_SEC = 600
BOUNDARY_BUFFER_US = BOUNDARY_BUFFER_SEC * 1_000_000

print(f"[CONFIG] Look-ahead: {LOOK_AHEAD_SECONDS}s, Volatility window: {VOLATILITY_WINDOW_SEC}s")
print(f"[CONFIG] Z-threshold: {Z_SCORE_THRESHOLD}, Min profit: {MIN_PROFIT_PCT*100:.2f}%")
print(f"[CONFIG] Batch size: {BATCH_SIZE}, Workers: {NUM_WORKERS}")


# =============================================================================
# MEMORY MONITORING UTILITIES
# =============================================================================

def get_ram_usage_percent():
    """Get current RAM usage as percentage (0-100)."""
    if HAS_PSUTIL:
        return psutil.virtual_memory().percent
    return 90.0  # Conservative fallback


def log_ram_status(prefix=""):
    """Log current RAM status."""
    if HAS_PSUTIL:
        mem = psutil.virtual_memory()
        print(f"{prefix}[RAM] {mem.percent:.1f}% ({mem.used/1024**3:.1f}/{mem.total/1024**3:.1f} GB)")


def is_ram_critical():
    """Check if RAM usage is above critical threshold."""
    return get_ram_usage_percent() >= RAM_THRESHOLD_PERCENT


def is_ram_safe():
    """Check if RAM has recovered to safe levels."""
    return get_ram_usage_percent() < RAM_SAFE_PERCENT


# =============================================================================
# TIME-BASED ROLLING FUNCTIONS - O(n) SLIDING WINDOW
# =============================================================================

if HAS_NUMBA:
    @jit(nopython=True, cache=True)
    def rolling_std_time_based(timestamps_us, values, window_seconds):
        """Time-based rolling standard deviation using O(n) sliding window."""
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
                variance = (running_sum_sq / count) - (mean * mean)
                if variance < 0.0:
                    variance = 0.0
                result[right] = np.sqrt(variance)
            else:
                result[right] = 0.0
        
        return result

    @jit(nopython=True, cache=True)
    def rolling_sum_time_based(timestamps_us, values, window_seconds):
        """Time-based rolling sum using O(n) sliding window."""
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

    @jit(nopython=True, cache=True)
    def rolling_mean_time_based(timestamps_us, values, window_seconds):
        """Time-based rolling mean using O(n) sliding window."""
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

    @jit(nopython=True, cache=True)
    def find_future_price_time_based(timestamps_us, prices, look_ahead_us):
        """Find price at t + look_ahead for each timestamp."""
        n = len(timestamps_us)
        future_prices = np.empty(n, dtype=np.float64)
        
        for i in range(n):
            target_time = timestamps_us[i] + look_ahead_us
            
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
    # Pure numpy fallbacks
    def rolling_std_time_based(timestamps_us, values, window_seconds):
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
        n = len(timestamps_us)
        future_prices = np.full(n, np.nan)
        target_times = timestamps_us + look_ahead_us
        future_indices = np.searchsorted(timestamps_us, target_times, side='left')
        valid_mask = future_indices < n
        future_indices = np.clip(future_indices, 0, n - 1)
        future_prices[valid_mask] = prices[future_indices[valid_mask]]
        return future_prices


# =============================================================================
# FEATURE COMPUTATION
# =============================================================================

def compute_features_time_based(df, verbose=True):
    """Compute all features using TIME-BASED rolling windows."""
    n = len(df)
    
    timestamps_us = df['timestamp'].values.astype(np.int64)
    prices = df['price'].values.astype(np.float64)
    qty = df['qty'].values.astype(np.float64)
    side = df['side'].values.astype(np.float64)
    
    # Basic features
    time_delta = np.zeros(n, dtype=np.float64)
    time_delta[1:] = (timestamps_us[1:] - timestamps_us[:-1]) / 1e6
    
    log_return = np.zeros(n, dtype=np.float64)
    log_return[1:] = np.log(prices[1:] / prices[:-1])
    
    trade_intensity = 1.0 / (time_delta + 1e-9)
    
    price_acceleration = np.zeros(n, dtype=np.float64)
    price_acceleration[1:] = log_return[1:] - log_return[:-1]
    
    # Volume features
    buy_volume = qty * (side == 1).astype(np.float64)
    sell_volume = qty * (side == -1).astype(np.float64)
    
    rolling_buy_vol = rolling_sum_time_based(timestamps_us, buy_volume, ROLLING_STATS_WINDOW_SEC)
    rolling_sell_vol = rolling_sum_time_based(timestamps_us, sell_volume, ROLLING_STATS_WINDOW_SEC)
    volume_imbalance = (rolling_buy_vol - rolling_sell_vol) / (rolling_buy_vol + rolling_sell_vol + 1e-9)
    
    signed_volume = qty * side
    rolling_signed_volume = rolling_sum_time_based(timestamps_us, signed_volume, VOLUME_WINDOW_SEC)
    
    # Volatility features
    price_volatility = rolling_std_time_based(timestamps_us, log_return, VOLATILITY_WINDOW_SEC)
    volume_volatility = rolling_std_time_based(timestamps_us, qty, ROLLING_STATS_WINDOW_SEC)
    time_delta_std = rolling_std_time_based(timestamps_us, time_delta, ROLLING_STATS_WINDOW_SEC)
    
    # Price features
    price_ma = rolling_mean_time_based(timestamps_us, prices, ROLLING_STATS_WINDOW_SEC)
    price_distance_from_ma = (prices - price_ma) / (price_ma + 1e-9)
    
    # Labeling
    look_ahead_us = LOOK_AHEAD_SECONDS * 1_000_000
    future_price = find_future_price_time_based(timestamps_us, prices, look_ahead_us)
    future_return = (future_price - prices) / prices
    
    upper_threshold = Z_SCORE_THRESHOLD * price_volatility
    lower_threshold = -Z_SCORE_THRESHOLD * price_volatility
    
    labels = np.ones(n, dtype=np.int8)
    up_condition = (future_return > upper_threshold) & (future_return > MIN_PROFIT_PCT)
    down_condition = (future_return < lower_threshold) & (future_return < -MIN_PROFIT_PCT)
    labels[up_condition] = 2
    labels[down_condition] = 0
    
    invalid = (
        np.isnan(future_price) | 
        np.isnan(price_volatility) | 
        (price_volatility <= 0) |
        np.isnan(future_return)
    )
    labels[invalid] = -1
    
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
    """Process one day with time-based features."""
    try:
        zip_path = Path(zip_path)
        if verbose:
            print(f"  Processing {zip_path.name}...")
        
        df = load_zip_to_df(zip_path)
        if len(df) == 0:
            return None
        
        df = df.sort_values('timestamp').reset_index(drop=True)
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='us')
        df['side'] = df['is_buyer_maker'].map({True: -1, False: 1})
        df = df[['timestamp', 'price', 'qty', 'side', 'datetime']].copy()
        
        if prev_tail_df is not None and len(prev_tail_df) > 0:
            df_combined = pd.concat([prev_tail_df, df], ignore_index=True)
            tail_len = len(prev_tail_df)
        else:
            df_combined = df
            tail_len = 0
        
        result = compute_features_time_based(df_combined, verbose=verbose)
        
        if tail_len > 0:
            result = result.iloc[tail_len:].copy()
        
        result = result[result['label'] != -1].copy()
        result = result.dropna()
        
        if len(result) == 0:
            return None
        
        # Tail for next day
        buffer_us = 10 * 60 * 1_000_000
        last_ts = df_combined['timestamp'].iloc[-1]
        tail_mask = df_combined['timestamp'] >= (last_ts - buffer_us)
        tail_df = df_combined[tail_mask][['timestamp', 'price', 'qty', 'side', 'datetime']].copy()
        
        # Boundary data for parallel correction
        first_ts = df['timestamp'].iloc[0]
        boundary_mask = df['timestamp'] <= (first_ts + BOUNDARY_BUFFER_US)
        boundary_raw_df = df[boundary_mask][['timestamp', 'price', 'qty', 'side', 'datetime']].copy()
        
        if verbose:
            counts = np.bincount(result['label'].values.astype(int), minlength=3)
            total = counts.sum()
            print(f"    Rows: {total:,} | DOWN: {counts[0]/total*100:.1f}% | NEUTRAL: {counts[1]/total*100:.1f}% | UP: {counts[2]/total*100:.1f}%")
        
        return result, tail_df, boundary_raw_df
        
    except Exception as e:
        print(f"  Error processing {zip_path}: {e}")
        return None


# =============================================================================
# BATCH PROCESSORS
# =============================================================================

def process_batch_sequential_with_disk_write(zip_paths, initial_tail, output_dir, start_file_idx):
    """Process batch sequentially, writing each day to disk immediately."""
    parquet_paths = []
    current_tail = initial_tail
    label_counts = {0: 0, 1: 0, 2: 0}
    total_rows = 0
    
    for i, zip_path in enumerate(zip_paths):
        result = process_single_day(zip_path, current_tail, verbose=True)
        
        if result is not None:
            df, current_tail, _ = result
            
            total_rows += len(df)
            for lv in [0, 1, 2]:
                label_counts[lv] += (df['label'] == lv).sum()
            
            file_idx = start_file_idx + i
            path = output_dir / f"day_{file_idx:04d}.parquet"
            df.to_parquet(path, engine='pyarrow', compression='snappy', index=False)
            parquet_paths.append(path)
            
            del df
            gc.collect()
        else:
            current_tail = None
    
    return parquet_paths, current_tail, label_counts, total_rows


def process_batch_for_parallel(args):
    """Worker function for parallel batch processing."""
    batch_idx, zip_paths = args
    
    results = []
    boundary_raws = []
    current_tail = None
    
    for zip_path in zip_paths:
        result = process_single_day(zip_path, current_tail, verbose=True)
        if result is not None:
            df, current_tail, boundary_raw = result
            results.append(df)
            boundary_raws.append(boundary_raw)
        else:
            current_tail = None
            boundary_raws.append(None)
    
    return batch_idx, results, current_tail, boundary_raws


def recompute_initial_features(prev_tail_df, boundary_raw_df, current_result_df):
    """Recompute features for boundary correction between parallel batches."""
    if prev_tail_df is None or len(prev_tail_df) == 0:
        return current_result_df
    if boundary_raw_df is None or len(boundary_raw_df) == 0:
        return current_result_df
    
    combined_raw = pd.concat([prev_tail_df, boundary_raw_df], ignore_index=True)
    tail_len = len(prev_tail_df)
    
    recomputed = compute_features_time_based(combined_raw, verbose=False)
    recomputed_boundary = recomputed.iloc[tail_len:].copy()
    recomputed_boundary = recomputed_boundary[recomputed_boundary['label'] != -1].copy()
    recomputed_boundary = recomputed_boundary.dropna()
    
    if len(recomputed_boundary) == 0:
        return current_result_df
    
    recomputed_timestamps = set(recomputed_boundary['timestamp'].values)
    mask = ~current_result_df['timestamp'].isin(recomputed_timestamps)
    result_without_boundary = current_result_df[mask]
    
    corrected = pd.concat([recomputed_boundary, result_without_boundary], ignore_index=True)
    corrected = corrected.sort_values('timestamp').reset_index(drop=True)
    
    return corrected


# =============================================================================
# HYBRID MEMORY-AWARE BATCH PROCESSING
# =============================================================================

def process_batches_hybrid(zip_files, output_dir, num_workers):
    """
    Hybrid batch processor with RAM monitoring.
    
    - RAM < 85%: parallel mode
    - RAM >= 85%: sequential mode with immediate disk writes
    
    Writes directly to output_dir (NO COMBINE STEP).
    """
    # Split into batches
    batches = []
    for batch_start in range(0, len(zip_files), BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, len(zip_files))
        batch_files = zip_files[batch_start:batch_end]
        batch_idx = batch_start // BATCH_SIZE
        batches.append((batch_idx, list(batch_files)))
    
    total_batches = len(batches)
    print(f"\n[HYBRID] Processing {total_batches} batches with up to {num_workers} workers")
    log_ram_status("[HYBRID] Initial ")
    
    # Tracking
    all_parquet_paths = []
    label_distribution = {0: 0, 1: 0, 2: 0}
    total_rows = 0
    file_idx = 0
    
    # Parallel tracking
    pending_async_results = []
    parallel_results = {}
    
    # Tail continuity
    sequential_tail = None
    last_processed_batch_idx = -1
    
    # Stats
    parallel_batches = 0
    sequential_batches = 0
    
    # Create pool
    pool = None
    if HAS_PSUTIL and num_workers > 1:
        pool = mp.Pool(processes=num_workers)
    
    try:
        pbar = tqdm(total=total_batches, desc="Processing batches")
        batch_queue_idx = 0
        
        while batch_queue_idx < total_batches or pending_async_results:
            
            # Collect completed parallel results
            still_pending = []
            for batch_idx, async_result in pending_async_results:
                if async_result.ready():
                    try:
                        result = async_result.get(timeout=1)
                        parallel_results[result[0]] = (result[1], result[2], result[3])
                    except Exception as e:
                        print(f"\n[ERROR] Batch {batch_idx} failed: {e}")
                else:
                    still_pending.append((batch_idx, async_result))
            pending_async_results = still_pending
            
            # Process completed results in order -> write to disk
            while last_processed_batch_idx + 1 in parallel_results:
                next_idx = last_processed_batch_idx + 1
                results, final_tail, boundary_raws = parallel_results.pop(next_idx)
                
                for day_idx, df in enumerate(results):
                    if day_idx == 0 and sequential_tail is not None:
                        boundary_raw = boundary_raws[day_idx] if boundary_raws else None
                        df = recompute_initial_features(sequential_tail, boundary_raw, df)
                    
                    total_rows += len(df)
                    for lv in [0, 1, 2]:
                        label_distribution[lv] += (df['label'] == lv).sum()
                    
                    # Write directly to output directory
                    path = output_dir / f"day_{file_idx:04d}.parquet"
                    df.to_parquet(path, engine='pyarrow', compression='snappy', index=False)
                    all_parquet_paths.append(path)
                    file_idx += 1
                
                sequential_tail = final_tail
                last_processed_batch_idx = next_idx
                pbar.update(1)
                
                del results
                gc.collect()
            
            # Submit new batches
            if batch_queue_idx < total_batches:
                ram_critical = is_ram_critical()
                
                if ram_critical or pool is None:
                    # SEQUENTIAL MODE
                    batch_idx, batch_files = batches[batch_queue_idx]
                    print(f"\n[SEQUENTIAL] Batch {batch_idx + 1}/{total_batches} | RAM: {get_ram_usage_percent():.1f}%")
                    
                    # Wait for pending parallel tasks
                    if pending_async_results:
                        print(f"  Waiting for {len(pending_async_results)} pending tasks...")
                        for pidx, async_result in pending_async_results:
                            try:
                                result = async_result.get(timeout=300)
                                parallel_results[result[0]] = (result[1], result[2], result[3])
                            except Exception as e:
                                print(f"  [ERROR] Batch {pidx} failed: {e}")
                        pending_async_results = []
                        
                        # Process all completed
                        while last_processed_batch_idx + 1 in parallel_results:
                            next_idx = last_processed_batch_idx + 1
                            results, final_tail, boundary_raws = parallel_results.pop(next_idx)
                            
                            for day_idx, df in enumerate(results):
                                if day_idx == 0 and sequential_tail is not None:
                                    boundary_raw = boundary_raws[day_idx] if boundary_raws else None
                                    df = recompute_initial_features(sequential_tail, boundary_raw, df)
                                
                                total_rows += len(df)
                                for lv in [0, 1, 2]:
                                    label_distribution[lv] += (df['label'] == lv).sum()
                                
                                path = output_dir / f"day_{file_idx:04d}.parquet"
                                df.to_parquet(path, engine='pyarrow', compression='snappy', index=False)
                                all_parquet_paths.append(path)
                                file_idx += 1
                            
                            sequential_tail = final_tail
                            last_processed_batch_idx = next_idx
                            pbar.update(1)
                            
                            del results
                            gc.collect()
                    
                    gc.collect()
                    
                    # Process sequentially
                    paths, sequential_tail, batch_labels, batch_rows = \
                        process_batch_sequential_with_disk_write(
                            batch_files, sequential_tail, output_dir, file_idx
                        )
                    
                    all_parquet_paths.extend(paths)
                    file_idx += len(paths)
                    total_rows += batch_rows
                    for lv in [0, 1, 2]:
                        label_distribution[lv] += batch_labels[lv]
                    
                    last_processed_batch_idx = batch_idx
                    sequential_batches += 1
                    batch_queue_idx += 1
                    pbar.update(1)
                    
                    gc.collect()
                    
                elif len(pending_async_results) < num_workers:
                    # PARALLEL MODE - submit if slots available
                    batch_idx, batch_files = batches[batch_queue_idx]
                    print(f"\n[PARALLEL] Batch {batch_idx + 1}/{total_batches} | RAM: {get_ram_usage_percent():.1f}%")
                    
                    async_result = pool.apply_async(
                        process_batch_for_parallel,
                        args=((batch_idx, batch_files),)
                    )
                    pending_async_results.append((batch_idx, async_result))
                    parallel_batches += 1
                    batch_queue_idx += 1
                else:
                    time.sleep(0.1)
            else:
                time.sleep(0.1)
        
        pbar.close()
        
    finally:
        if pool is not None:
            pool.close()
            pool.join()
    
    print(f"\n[HYBRID] Complete: {parallel_batches} parallel + {sequential_batches} sequential")
    log_ram_status("[HYBRID] Final ")
    
    return all_parquet_paths, label_distribution, total_rows


# =============================================================================
# MAIN PROCESSOR
# =============================================================================

def process_streaming(input_dir, output_dir):
    """Process all zips with hybrid memory-aware batch processing."""
    print("=" * 80)
    print("HYBRID MEMORY-AWARE PREPROCESSING")
    print("NO COMBINE STEP - Direct Daily Parquet Output")
    print("=" * 80)
    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}/")
    print(f"\nTime-Based Windows:")
    print(f"  Look-ahead:        {LOOK_AHEAD_SECONDS}s ({LOOK_AHEAD_SECONDS/60:.1f} min)")
    print(f"  Volatility window: {VOLATILITY_WINDOW_SEC}s ({VOLATILITY_WINDOW_SEC/60:.1f} min)")
    print(f"  Volume window:     {VOLUME_WINDOW_SEC}s ({VOLUME_WINDOW_SEC/60:.1f} min)")
    print(f"  Stats window:      {ROLLING_STATS_WINDOW_SEC}s ({ROLLING_STATS_WINDOW_SEC/60:.1f} min)")
    print(f"\nLabeling:")
    print(f"  Z-Score threshold: {Z_SCORE_THRESHOLD}")
    print(f"  Min profit:        {MIN_PROFIT_PCT*100:.3f}%")
    print(f"\nPerformance:")
    print(f"  Numba JIT:  {HAS_NUMBA}")
    print(f"  psutil:     {HAS_PSUTIL}")
    print(f"  Batch size: {BATCH_SIZE} days")
    print(f"  Workers:    {NUM_WORKERS}")
    print(f"  RAM threshold: {RAM_THRESHOLD_PERCENT}%")
    print("=" * 80 + "\n")
    
    zip_files = sorted(Path(input_dir).glob("*.zip"))
    if len(zip_files) == 0:
        raise FileNotFoundError(f"No zip files in {input_dir}")
    
    print(f"Found {len(zip_files)} zip files\n")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Process with hybrid approach
    all_parquet_paths, label_distribution, total_rows = process_batches_hybrid(
        zip_files, output_path, NUM_WORKERS
    )
    
    if len(all_parquet_paths) == 0:
        raise Exception("No days processed successfully")
    
    # Calculate total size
    total_size_bytes = sum(p.stat().st_size for p in all_parquet_paths)
    total_size_gb = total_size_bytes / 1e9
    
    # Save metadata for downstream consumption
    metadata = {
        'total_rows': int(total_rows),
        'total_files': len(all_parquet_paths),
        'total_size_gb': round(total_size_gb, 2),
        'label_distribution': {
            'down': int(label_distribution[0]),
            'neutral': int(label_distribution[1]),
            'up': int(label_distribution[2])
        },
        'config': {
            'look_ahead_seconds': LOOK_AHEAD_SECONDS,
            'volatility_window_sec': VOLATILITY_WINDOW_SEC,
            'z_score_threshold': Z_SCORE_THRESHOLD,
            'min_profit_pct': MIN_PROFIT_PCT
        },
        'files': [p.name for p in sorted(all_parquet_paths)]
    }
    
    metadata_file = output_path / "metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Final stats
    print("\n" + "=" * 80)
    print("PREPROCESSING COMPLETE")
    print("=" * 80)
    print(f"Output: {output_dir}/")
    print(f"Files: {len(all_parquet_paths)} daily parquets")
    print(f"Total rows: {total_rows:,}")
    print(f"Total size: {total_size_gb:.2f} GB")
    
    print("\nLabel Distribution:")
    total = sum(label_distribution.values())
    for lv in [0, 1, 2]:
        count = label_distribution[lv]
        pct = (count / total * 100) if total > 0 else 0
        label_name = ['DOWN', 'NEUTRAL', 'UP'][lv]
        print(f"  {label_name:>7}: {count:>12,} ({pct:>5.2f}%)")
    
    print(f"\nMetadata: {metadata_file}")
    print("=" * 80)
    
    return metadata


# =============================================================================
# MAIN
# =============================================================================

def main():
    start_time = datetime.now()
    print(f"\nStart time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    try:
        metadata = process_streaming(INPUT_DIR, OUTPUT_DIR)
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        print(f"\nDuration: {duration:.1f} seconds ({duration/60:.1f} minutes)")
        print(f"Throughput: {metadata['total_rows']/duration:,.0f} rows/second\n")
        
    except Exception as e:
        print("\n" + "=" * 80)
        print("ERROR OCCURRED")
        print("=" * 80)
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()