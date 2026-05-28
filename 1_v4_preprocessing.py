#!/usr/bin/env python3
"""
PREPROCESSING v4 — REGRESSION + PRICE-INVARIANT FEATURES
=========================================================
Target hardware : i9-14900K, 128 GB RAM, 2x A6000
Data            : Binance BTCUSDT tick trades (daily zips)

CHANGES FROM v3:
  P2. Made price-dependent features price-invariant:
      - tfi_quote_300s → divided by VWAP (now in BTC-equivalent units)
      - amihud_illiquidity → uses qty instead of quote_qty
      - avg_size_300s → normalized by 1h rolling mean (relative size)
  P3. Removed weight = abs(target) — no more double-penalty with sniper objective
  P7. Replaced redundant pairs with coefficient-of-variation ratios:
      - inter_time_mean + inter_time_std → inter_time_cv_300s
      - avg_size + size_std → size_cv_300s (no conflict with P2 normalization,
        CV = std/mean is already scale-invariant)

Final feature count: 12
  From v3 (kept):  price_volatility_300s, vol_imbalance_300s, vwap_dev_300s,
                   intensity_z_300s, feat_vol_force, feat_sma_trend, feat_rsi_norm
  Modified (P2):   tfi_quote_norm_300s, amihud_illiquidity_300s, rel_size_300s
  New (P7 CV):     inter_time_cv_300s, size_cv_300s
"""

import os, sys, zipfile, gc, time, json, warnings
import pandas as pd
import numpy as np
import pyarrow.parquet as pq
import pyarrow as pa
import pyarrow.csv as pacsv
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import multiprocessing as mp

warnings.filterwarnings('ignore')

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    print("[WARN] psutil not installed — RAM monitoring disabled")

try:
    from numba import jit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    print("[WARN] numba not installed — JIT disabled, will be SLOW")
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return decorator


# =============================================================================
# CONFIGURATION
# =============================================================================

INPUT_DIR  = "./btc_trade_data"
OUTPUT_DIR = "./processed_daily_parquets_v4"

# --- Regression Target ---
LOOK_AHEAD_SECONDS = 300          # 5-minute forward log-return

# --- Rolling Window (seconds) ---
WINDOW = 300

# --- Tick Subsampling ---
# Features are computed at tick level (accurate rolling windows), then we keep
# one row per SUBSAMPLE_SECONDS to eliminate redundancy.  Consecutive ticks
# share >99.9% of their feature/target windows — training on all of them
# inflates metrics and teaches the model nothing new.
SUBSAMPLE_SECONDS = 60            # 0 = keep every tick (not recommended)

# --- Date Filter ---
START_DATE = "2026-01-01"
END_DATE   = "2026-05-26"

# --- Smart Scheduler (i5-11400H, 16 GB RAM) ---
TARGET_BATCH_SIZE_MB  = 30
MAX_PARALLEL_SIZE_MB  = 80
OOM_SAFEGUARD_PERCENT = 75
RAM_SAFE_PERCENT      = 60
NUM_WORKERS           = 2

# --- Boundary Buffer ---
BOUNDARY_BUFFER_SEC = 600
BOUNDARY_BUFFER_US  = BOUNDARY_BUFFER_SEC * 1_000_000


# =============================================================================
# RAM UTILITIES
# =============================================================================

def get_ram_usage_percent():
    if HAS_PSUTIL:
        return psutil.virtual_memory().percent
    return 0.0

def is_ram_critical():
    return get_ram_usage_percent() > OOM_SAFEGUARD_PERCENT

def is_ram_safe():
    return get_ram_usage_percent() < RAM_SAFE_PERCENT

def log_ram_status(prefix=""):
    if HAS_PSUTIL:
        mem = psutil.virtual_memory()
        print(f"{prefix}RAM: {mem.percent:.1f}% ({mem.used/1e9:.1f}/{mem.total/1e9:.1f} GB)")


# =============================================================================
# NUMBA JIT ROLLING PRIMITIVES  (O(n) sliding window, causal: [t-w, t])
# =============================================================================

@jit(nopython=True, cache=True, nogil=True)
def _rolling_sum(ts_us, vals, win_sec):
    n = len(vals)
    out = np.empty(n, dtype=np.float64)
    win_us = win_sec * 1_000_000
    left = 0; s = 0.0
    for r in range(n):
        s += vals[r]
        lo = ts_us[r] - win_us
        while left < r and ts_us[left] < lo:
            s -= vals[left]; left += 1
        out[r] = s
    return out

@jit(nopython=True, cache=True, nogil=True)
def _rolling_mean(ts_us, vals, win_sec):
    n = len(vals)
    out = np.empty(n, dtype=np.float64)
    win_us = win_sec * 1_000_000
    left = 0; s = 0.0
    for r in range(n):
        s += vals[r]
        lo = ts_us[r] - win_us
        while left < r and ts_us[left] < lo:
            s -= vals[left]; left += 1
        out[r] = s / (r - left + 1)
    return out

@jit(nopython=True, cache=True, nogil=True)
def _rolling_std(ts_us, vals, win_sec):
    n = len(vals)
    out = np.empty(n, dtype=np.float64)
    win_us = win_sec * 1_000_000
    left = 0; s = 0.0; sq = 0.0
    for r in range(n):
        s += vals[r]; sq += vals[r] * vals[r]
        lo = ts_us[r] - win_us
        while left < r and ts_us[left] < lo:
            s -= vals[left]; sq -= vals[left] * vals[left]; left += 1
        cnt = r - left + 1
        if cnt >= 2:
            m = s / cnt
            var = sq / cnt - m * m
            out[r] = np.sqrt(max(0.0, var))
        else:
            out[r] = 0.0
    return out

@jit(nopython=True, cache=True, nogil=True)
def _rolling_count(ts_us, win_sec):
    n = len(ts_us)
    out = np.empty(n, dtype=np.float64)
    win_us = win_sec * 1_000_000
    left = 0
    for r in range(n):
        lo = ts_us[r] - win_us
        while left < r and ts_us[left] < lo:
            left += 1
        out[r] = float(r - left + 1)
    return out

@jit(nopython=True, cache=True, nogil=True)
def _rolling_vwap(ts_us, prices, qtys, win_sec):
    n = len(prices)
    out = np.empty(n, dtype=np.float64)
    win_us = win_sec * 1_000_000
    left = 0; s_pq = 0.0; s_q = 0.0
    for r in range(n):
        s_pq += prices[r] * qtys[r]
        s_q  += qtys[r]
        lo = ts_us[r] - win_us
        while left < r and ts_us[left] < lo:
            s_pq -= prices[left] * qtys[left]
            s_q  -= qtys[left]
            left += 1
        if s_q > 1e-15:
            out[r] = s_pq / s_q
        else:
            out[r] = prices[r]
    return out

@jit(nopython=True, cache=True, nogil=True)
def _find_future_price(ts_us, prices, look_ahead_us):
    n = len(ts_us)
    fp = np.full(n, np.nan)
    targets = ts_us + look_ahead_us
    idxs = np.searchsorted(ts_us, targets, side='left')
    for i in range(n):
        if idxs[i] < n:
            fp[i] = prices[idxs[i]]
    return fp

@jit(nopython=True, cache=True, nogil=True)
def _precompute_base_metrics(ts_us, prices, qty, side):
    n = len(ts_us)
    time_delta = np.zeros(n, dtype=np.float64)
    log_return = np.zeros(n, dtype=np.float64)
    buy_volume = np.zeros(n, dtype=np.float64)
    sell_volume = np.zeros(n, dtype=np.float64)
    price_diff = np.zeros(n, dtype=np.float64)
    for i in range(n):
        if i > 0:
            time_delta[i] = (ts_us[i] - ts_us[i-1]) / 1000000.0
            log_return[i] = np.log(prices[i] / prices[i-1])
            price_diff[i] = prices[i] - prices[i-1]
        if side[i] == 1.0:
            buy_volume[i] = qty[i]
        elif side[i] == -1.0:
            sell_volume[i] = qty[i]
    return time_delta, log_return, buy_volume, sell_volume, price_diff


# =============================================================================
# FEATURE COMPUTATION v4
# =============================================================================

def compute_features_regression(df, verbose=True):
    """
    Computes 12 features (v4):
      7 kept from v3, 3 modified for price-invariance, 2 new CV ratios.
    Weight column removed (P3).
    """
    n = len(df)

    # --- Extract numpy arrays ---
    ts_us     = df['timestamp'].values.astype(np.int64)
    prices    = df['price'].values.astype(np.float64)
    qty       = df['qty'].values.astype(np.float64)
    side      = df['side'].values.astype(np.float64)

    # --- Pre-compute base metrics ---
    time_delta, log_return, buy_volume, sell_volume, price_diff = _precompute_base_metrics(
        ts_us, prices, qty, side
    )

    w = 300  # 5 min window

    # =====================================================================
    # 1. TREND & PHYSICS FEATURES (kept from v3)
    # =====================================================================

    # Volume Force (mass * acceleration, smoothed 60s)
    raw_force = qty * price_diff
    feat_vol_force = _rolling_mean(ts_us, raw_force, 60)

    # SMA Trend Deviation (15 min / 900s)
    sma_900s = _rolling_mean(ts_us, prices, 900)
    feat_sma_trend = (prices - sma_900s) / (sma_900s + 1e-9)

    # Tick RSI (300s window)
    up_moves = np.maximum(price_diff, 0)
    down_moves = np.abs(np.minimum(price_diff, 0))
    avg_gain = _rolling_mean(ts_us, up_moves, w)
    avg_loss = _rolling_mean(ts_us, down_moves, w)
    rs = avg_gain / (avg_loss + 1e-9)
    feat_rsi = 100.0 - (100.0 / (1.0 + rs))
    feat_rsi_norm = (feat_rsi - 50.0) / 50.0

    # =====================================================================
    # 2. FLOW FEATURES
    # =====================================================================

    rb = _rolling_sum(ts_us, buy_volume, w)
    rs_vol = _rolling_sum(ts_us, sell_volume, w)
    denom = rb + rs_vol + 1e-9
    vol_imbalance_300s = (rb - rs_vol) / denom

    # [P2] Trade Flow Imbalance — normalized by VWAP to remove price-level dependency
    vwap_w = _rolling_vwap(ts_us, prices, qty, w)
    rb_q = _rolling_sum(ts_us, buy_volume * prices, w)
    rs_q = _rolling_sum(ts_us, sell_volume * prices, w)
    tfi_raw = rb_q - rs_q
    tfi_quote_norm_300s = tfi_raw / (vwap_w + 1e-9)  # Now in BTC-equivalent units

    # =====================================================================
    # 3. VOLATILITY & SPEED
    # =====================================================================

    price_volatility_300s = _rolling_std(ts_us, log_return, w)

    # [P7] Inter-arrival time: CV ratio instead of mean + std separately
    inter_time_mean = _rolling_mean(ts_us, time_delta, w)
    inter_time_std  = _rolling_std(ts_us, time_delta, w)
    inter_time_cv_300s = inter_time_std / (inter_time_mean + 1e-9)

    # Intensity Z
    count_w  = _rolling_count(ts_us, w)
    mean_cnt = _rolling_mean(ts_us, count_w, w)
    std_cnt  = _rolling_std(ts_us, count_w, w)
    intensity_z_300s = (count_w - mean_cnt) / (std_cnt + 1e-9)

    # =====================================================================
    # 4. RELATIVE VALUE
    # =====================================================================

    vwap_dev_300s = (prices - vwap_w) / (vwap_w + 1e-9)

    # [P2] Amihud illiquidity — use qty (BTC) instead of quote_qty ($) to remove
    # price-level dependency
    amihud_raw = np.abs(log_return) / (qty + 1e-9)
    amihud_illiquidity_300s = _rolling_mean(ts_us, amihud_raw, w)

    # =====================================================================
    # 5. SIZE FEATURES
    # =====================================================================

    avg_size_300s = _rolling_mean(ts_us, qty, w)
    size_std_300s = _rolling_std(ts_us, qty, w)

    # [P2] Relative size: normalize by 1h rolling mean to make scale-invariant
    avg_size_1h = _rolling_mean(ts_us, qty, 3600)
    rel_size_300s = avg_size_300s / (avg_size_1h + 1e-15)

    # [P7] Size CV ratio — already scale-invariant (std/mean), no conflict with P2
    size_cv_300s = size_std_300s / (avg_size_300s + 1e-9)

    # =====================================================================
    # TARGET (P3: no weight column)
    # =====================================================================

    look_ahead_us = LOOK_AHEAD_SECONDS * 1_000_000
    future_price = _find_future_price(ts_us, prices, look_ahead_us)
    target = np.log(future_price / prices)

    # =====================================================================
    # ASSEMBLE DATAFRAME
    # =====================================================================

    result = pd.DataFrame({
        'timestamp':                ts_us,
        'target':                   target,

        # --- 12 FEATURES ---
        # Kept from v3 (7)
        'price_volatility_300s':    price_volatility_300s,
        'vol_imbalance_300s':       vol_imbalance_300s,
        'vwap_dev_300s':            vwap_dev_300s,
        'intensity_z_300s':         intensity_z_300s,
        'feat_vol_force':           feat_vol_force,
        'feat_sma_trend':           feat_sma_trend,
        'feat_rsi_norm':            feat_rsi_norm,

        # Modified for price-invariance (P2) (3)
        'tfi_quote_norm_300s':      tfi_quote_norm_300s,
        'amihud_illiquidity_300s':  amihud_illiquidity_300s,
        'rel_size_300s':            rel_size_300s,

        # CV ratios replacing redundant pairs (P7) (2)
        'inter_time_cv_300s':       inter_time_cv_300s,
        'size_cv_300s':             size_cv_300s,
    })

    fcols = result.select_dtypes('float').columns
    result[fcols] = result[fcols].astype(np.float32)

    return result


# =============================================================================
# DATA I/O
# =============================================================================

def load_zip_to_df(zip_path):
    """Load a single Binance trade CSV from its zip using PyArrow for speed."""
    with zipfile.ZipFile(zip_path, 'r') as zf:
        csv_name = Path(zip_path).stem + '.csv'
        with zf.open(csv_name) as csvf:
            read_options = pacsv.ReadOptions(column_names=[
                'trade_id', 'price', 'qty', 'quote_qty',
                'timestamp', 'is_buyer_maker', 'is_best_match'
            ])
            parse_options = pacsv.ParseOptions(delimiter=',')
            table = pacsv.read_csv(csvf, read_options=read_options, parse_options=parse_options)
            df = table.to_pandas()
    # Binance 2026 trade data already uses microsecond timestamps (16 digits)
    # No conversion needed — rolling windows expect microseconds.
    return df


def process_single_day(zip_path, prev_tail_df=None, verbose=True):
    """
    Process one day: load -> concat tail -> features -> trim -> return.
    Returns (result_df, tail_df, boundary_raw_df) or None on error.
    """
    try:
        zip_path = Path(zip_path)
        if verbose:
            print(f"  Processing {zip_path.name}...")

        df = load_zip_to_df(zip_path)
        if len(df) == 0:
            return None

        df = df.sort_values('timestamp').reset_index(drop=True)
        df['side'] = df['is_buyer_maker'].map({True: -1, False: 1})
        df = df[['timestamp', 'price', 'qty', 'quote_qty', 'side']].copy()

        # Concat previous tail for rolling window continuity
        if prev_tail_df is not None and len(prev_tail_df) > 0:
            df_combined = pd.concat([prev_tail_df, df], ignore_index=True)
            tail_len = len(prev_tail_df)
        else:
            df_combined = df
            tail_len = 0

        # Compute features
        result = compute_features_regression(df_combined, verbose=verbose)

        # Remove prepended tail rows
        if tail_len > 0:
            result = result.iloc[tail_len:].copy()

        # Remove rows with no valid target (end of day, no future price)
        result = result.dropna(subset=['target'])

        if len(result) == 0:
            return None

        # Subsample: keep one row per SUBSAMPLE_SECONDS bucket.
        # Features were computed at full tick resolution (accurate rolling windows),
        # but adjacent ticks share >99% of feature/target content.
        tick_count = len(result)
        if SUBSAMPLE_SECONDS > 0:
            bucket = result['timestamp'] // (SUBSAMPLE_SECONDS * 1_000_000)
            result = result[~bucket.duplicated(keep='first')].copy()

        # Prepare tail for next day (last 15 min of raw data — increased for 1h rolling)
        buffer_us = 15 * 60 * 1_000_000
        last_ts = df_combined['timestamp'].iloc[-1]
        tail_mask = df_combined['timestamp'] >= (last_ts - buffer_us)
        tail_df = df_combined[tail_mask][['timestamp', 'price', 'qty', 'quote_qty', 'side']].copy()

        # Boundary raw for parallel correction (first 15 min of THIS day)
        first_ts = df['timestamp'].iloc[0]
        boundary_mask = df['timestamp'] <= (first_ts + BOUNDARY_BUFFER_US)
        boundary_raw_df = df[boundary_mask][['timestamp', 'price', 'qty', 'quote_qty', 'side']].copy()

        if verbose:
            mean_t = result['target'].mean()
            std_t  = result['target'].std()
            print(f"    Ticks: {tick_count:,} -> {len(result):,} rows "
                  f"({SUBSAMPLE_SECONDS}s) | target mean={mean_t:.6f} std={std_t:.6f}")

        return result, tail_df, boundary_raw_df

    except Exception as e:
        print(f"  [ERROR] {zip_path}: {e}")
        import traceback; traceback.print_exc()
        return None


# =============================================================================
# BOUNDARY CORRECTION
# =============================================================================

def recompute_initial_features(prev_tail_df, boundary_raw_df, current_result_df):
    if prev_tail_df is None or len(prev_tail_df) == 0:
        return current_result_df
    if boundary_raw_df is None or len(boundary_raw_df) == 0:
        return current_result_df

    combined_raw = pd.concat([prev_tail_df, boundary_raw_df], ignore_index=True)
    tail_len = len(prev_tail_df)

    recomputed = compute_features_regression(combined_raw, verbose=False)
    recomputed_boundary = recomputed.iloc[tail_len:].copy()
    recomputed_boundary = recomputed_boundary.dropna(subset=['target'])

    if len(recomputed_boundary) == 0:
        return current_result_df

    recomputed_ts = set(recomputed_boundary['timestamp'].values)
    mask = ~current_result_df['timestamp'].isin(recomputed_ts)
    result_without = current_result_df[mask]

    corrected = pd.concat([recomputed_boundary, result_without], ignore_index=True)
    corrected = corrected.sort_values('timestamp').reset_index(drop=True)
    return corrected


# =============================================================================
# BATCH PROCESSORS
# =============================================================================

def process_batch_sequential_with_disk_write(zip_paths, initial_tail, output_dir, start_file_idx):
    parquet_paths = []
    current_tail  = initial_tail
    total_rows    = 0

    for i, zip_path in enumerate(zip_paths):
        result = process_single_day(zip_path, current_tail, verbose=True)
        if result is not None:
            df, current_tail, _ = result
            total_rows += len(df)

            file_idx = start_file_idx + i
            path = output_dir / f"day_{file_idx:04d}.parquet"
            df.to_parquet(path, engine='pyarrow', compression='snappy', index=False)
            parquet_paths.append(path)

            del df; gc.collect()
        else:
            current_tail = None

    return parquet_paths, current_tail, total_rows


def process_batch_for_parallel(args):
    batch_idx, zip_paths, output_dir, start_file_idx = args

    parquet_paths = []
    boundary_raws = []
    current_tail  = None
    total_rows    = 0

    for i, zip_path in enumerate(zip_paths):
        result = process_single_day(zip_path, current_tail, verbose=True)
        if result is not None:
            df, current_tail, boundary_raw = result
            total_rows += len(df)

            file_idx = start_file_idx + i
            path = Path(output_dir) / f"day_{file_idx:04d}.parquet"
            df.to_parquet(path, engine='pyarrow', compression='snappy', index=False)
            parquet_paths.append(path)
            boundary_raws.append(boundary_raw)

            del df; gc.collect()
        else:
            current_tail = None
            boundary_raws.append(None)

    return batch_idx, parquet_paths, current_tail, boundary_raws, total_rows


# =============================================================================
# SMART SIZE-BASED SCHEDULER
# =============================================================================

def generate_smart_batches(zip_files):
    batches = []
    current_files = []
    current_size  = 0.0

    for f in zip_files:
        fsize = f.stat().st_size / (1024 * 1024)
        if current_files and (current_size + fsize > TARGET_BATCH_SIZE_MB):
            batches.append({
                'files':       current_files,
                'size_mb':     current_size,
                'is_oversized': len(current_files) == 1 and current_size > TARGET_BATCH_SIZE_MB,
            })
            current_files = []
            current_size  = 0.0
        current_files.append(f)
        current_size += fsize

    if current_files:
        batches.append({
            'files':       current_files,
            'size_mb':     current_size,
            'is_oversized': len(current_files) == 1 and current_size > TARGET_BATCH_SIZE_MB,
        })
    return batches


def process_batches_smart(zip_files, output_dir, num_workers):
    batches       = generate_smart_batches(zip_files)
    total_batches = len(batches)
    total_size_mb = sum(b['size_mb'] for b in batches)

    print(f"\n{'='*70}")
    print(f"SMART SCHEDULER")
    print(f"  Batches: {total_batches} | Total: {total_size_mb:.1f} MB")
    print(f"  Target/batch: {TARGET_BATCH_SIZE_MB} MB | Max parallel: {MAX_PARALLEL_SIZE_MB} MB")
    print(f"  Workers: {num_workers} | OOM safeguard: {OOM_SAFEGUARD_PERCENT}%")
    print(f"{'='*70}")

    for i, b in enumerate(batches[:5]):
        tag = " [OVERSIZED]" if b['is_oversized'] else ""
        print(f"  Batch {i}: {len(b['files'])} files, {b['size_mb']:.1f} MB{tag}")
    if total_batches > 5:
        print(f"  ... and {total_batches - 5} more")

    log_ram_status("\n[SCHEDULER] Initial ")

    # Assign global file indices
    batch_start_file_idx = {}
    cum = 0
    for i, b in enumerate(batches):
        batch_start_file_idx[i] = cum
        cum += len(b['files'])

    all_parquet_paths   = []
    total_rows          = 0
    current_active_mb   = 0.0
    pending             = []
    parallel_results    = {}
    sequential_tail     = None
    last_processed_idx  = -1
    parallel_count   = 0
    sequential_count = 0

    pool = None
    if HAS_PSUTIL and num_workers > 1:
        pool = mp.Pool(processes=num_workers)

    try:
        pbar = tqdm(total=total_batches, desc="Smart batching")
        queue_idx = 0

        while queue_idx < total_batches or pending:

            # A. Collect completed async results
            still_pending = []
            for bidx, ar, sz in pending:
                if ar.ready():
                    try:
                        res = ar.get(timeout=1)
                        # res = (batch_idx, paths, tail, boundary_raws, rows)
                        parallel_results[res[0]] = (res[1], res[2], res[3], res[4])
                    except Exception as e:
                        print(f"\n[ERROR] Batch {bidx} failed in worker: {e}")
                    current_active_mb -= sz
                else:
                    still_pending.append((bidx, ar, sz))
            pending = still_pending

            # B. Write completed results in order
            while last_processed_idx + 1 in parallel_results:
                next_idx = last_processed_idx + 1
                paths, final_tail, boundary_raws, batch_rows = parallel_results.pop(next_idx)

                all_parquet_paths.extend(paths)
                total_rows += batch_rows
                sequential_tail = final_tail
                last_processed_idx = next_idx
                pbar.update(1)
                gc.collect()

            # C. Submit next batch
            if queue_idx < total_batches:
                batch = batches[queue_idx]
                bsz   = batch['size_mb']

                if is_ram_critical():
                    gc.collect()
                    while not is_ram_safe():
                        time.sleep(1.0)
                        sp = []
                        for bidx, ar, sz in pending:
                            if ar.ready():
                                try:
                                    res = ar.get(timeout=1)
                                    parallel_results[res[0]] = (res[1], res[2], res[3], res[4])
                                except Exception as e:
                                    print(f"\n[ERROR] Batch {bidx}: {e}")
                                current_active_mb -= sz
                            else:
                                sp.append((bidx, ar, sz))
                        pending = sp

                if batch['is_oversized']:
                    # Drain pool, run in isolation
                    while pending:
                        time.sleep(0.5)
                        sp = []
                        for bidx, ar, sz in pending:
                            if ar.ready():
                                try:
                                    res = ar.get(timeout=1)
                                    parallel_results[res[0]] = (res[1], res[2], res[3], res[4])
                                except Exception as e:
                                    print(f"\n[ERROR] Batch {bidx}: {e}")
                                current_active_mb -= sz
                            else:
                                sp.append((bidx, ar, sz))
                        pending = sp

                    # Flush writes
                    while last_processed_idx + 1 in parallel_results:
                        next_idx = last_processed_idx + 1
                        paths, final_tail, boundary_raws, batch_rows = parallel_results.pop(next_idx)
                        all_parquet_paths.extend(paths)
                        total_rows += batch_rows
                        sequential_tail = final_tail
                        last_processed_idx = next_idx
                        pbar.update(1)
                        gc.collect()

                    sfidx = batch_start_file_idx[queue_idx]
                    paths, seq_tail, seq_rows = process_batch_sequential_with_disk_write(
                        batch['files'], sequential_tail, output_dir, sfidx
                    )
                    all_parquet_paths.extend(paths)
                    total_rows += seq_rows
                    sequential_tail = seq_tail
                    last_processed_idx = queue_idx
                    sequential_count += 1
                    queue_idx += 1
                    pbar.update(1)
                    gc.collect()

                elif pool is not None and current_active_mb + bsz <= MAX_PARALLEL_SIZE_MB:
                    sfidx = batch_start_file_idx[queue_idx]
                    ar = pool.apply_async(
                        process_batch_for_parallel,
                        args=((queue_idx, batch['files'], str(output_dir), sfidx),)
                    )
                    pending.append((queue_idx, ar, bsz))
                    current_active_mb += bsz
                    parallel_count += 1
                    queue_idx += 1

                elif pool is None or (current_active_mb + bsz > MAX_PARALLEL_SIZE_MB
                                      and len(pending) == 0):
                    sfidx = batch_start_file_idx[queue_idx]
                    paths, seq_tail, seq_rows = process_batch_sequential_with_disk_write(
                        batch['files'], sequential_tail, output_dir, sfidx
                    )
                    all_parquet_paths.extend(paths)
                    total_rows += seq_rows
                    sequential_tail = seq_tail
                    last_processed_idx = queue_idx
                    sequential_count += 1
                    queue_idx += 1
                    pbar.update(1)
                    gc.collect()
                else:
                    time.sleep(0.2)
            else:
                time.sleep(0.2)

        pbar.close()

    finally:
        if pool:
            pool.close()
            pool.join()

    print(f"\n[SCHEDULER] Done: {parallel_count} parallel + {sequential_count} sequential")
    log_ram_status("[SCHEDULER] Final ")

    return all_parquet_paths, total_rows


# =============================================================================
# MAIN ORCHESTRATOR
# =============================================================================

def build_feature_column_list():
    """Returns the 12 features for v4."""
    return [
        # Kept from v3
        'price_volatility_300s',
        'vol_imbalance_300s',
        'vwap_dev_300s',
        'intensity_z_300s',
        'feat_vol_force',
        'feat_sma_trend',
        'feat_rsi_norm',
        # Modified for price-invariance (P2)
        'tfi_quote_norm_300s',
        'amihud_illiquidity_300s',
        'rel_size_300s',
        # CV ratios (P7)
        'inter_time_cv_300s',
        'size_cv_300s',
    ]


def get_filtered_zip_files(root_dir, start_str, end_str):
    root = Path(root_dir)
    all_zips = sorted(root.rglob("*.zip"))

    start_dt = datetime.strptime(start_str, "%Y-%m-%d")
    end_dt   = datetime.strptime(end_str, "%Y-%m-%d")

    valid_files = []
    print(f"Scanning {len(all_zips)} files in {root_dir}...")

    for p in all_zips:
        try:
            date_part = p.stem.split('-trades-')[-1]
            file_dt = datetime.strptime(date_part, "%Y-%m-%d")
            if start_dt <= file_dt <= end_dt:
                valid_files.append(p)
        except ValueError:
            continue

    return sorted(valid_files)


def process_streaming(input_dir, output_dir):
    print("=" * 80)
    print("PREPROCESSING v4 — PRICE-INVARIANT FEATURES (12)")
    print(f"  Range: {START_DATE} to {END_DATE}")
    print(f"  Workers: {NUM_WORKERS}")
    print(f"  Batch target: {TARGET_BATCH_SIZE_MB} MB  |  Max parallel: {MAX_PARALLEL_SIZE_MB} MB")
    print(f"  Look-ahead: {LOOK_AHEAD_SECONDS}s  |  Window: {WINDOW}s  |  Subsample: {SUBSAMPLE_SECONDS}s")
    print("=" * 80)

    zip_files = get_filtered_zip_files(input_dir, START_DATE, END_DATE)
    if not zip_files:
        raise FileNotFoundError(f"No zip files found in {input_dir} between {START_DATE} and {END_DATE}")

    print(f"Selected {len(zip_files)} files to process.\n")

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    all_parquet_paths, total_rows = process_batches_smart(zip_files, output_path, NUM_WORKERS)

    if not all_parquet_paths:
        raise RuntimeError("No days processed successfully")

    total_size_gb = sum(p.stat().st_size for p in all_parquet_paths) / 1e9
    feature_columns = build_feature_column_list()

    metadata = {
        'version':          4,
        'task':             'regression',
        'total_rows':       int(total_rows),
        'total_files':      len(all_parquet_paths),
        'total_size_gb':    round(total_size_gb, 2),
        'config': {
            'look_ahead_seconds':    LOOK_AHEAD_SECONDS,
            'window':                WINDOW,
            'subsample_seconds':     SUBSAMPLE_SECONDS,
            'target_batch_size_mb':  TARGET_BATCH_SIZE_MB,
            'max_parallel_size_mb':  MAX_PARALLEL_SIZE_MB,
            'num_workers':           NUM_WORKERS,
        },
        'feature_columns':  feature_columns,
        'target_column':    'target',
        'files':            [p.name for p in sorted(all_parquet_paths)],
    }

    meta_path = output_path / "metadata.json"
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print("\n" + "=" * 80)
    print("PREPROCESSING v4 COMPLETE")
    print("=" * 80)
    print(f"  Output : {output_dir}/")
    print(f"  Files  : {len(all_parquet_paths)}")
    print(f"  Rows   : {total_rows:,}")
    print(f"  Size   : {total_size_gb:.2f} GB")
    print(f"  Features: {len(feature_columns)}")
    print(f"  Metadata: {meta_path}")
    print("=" * 80)

    return metadata


def main():
    start_time = datetime.now()
    print(f"\nStart: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    try:
        metadata = process_streaming(INPUT_DIR, OUTPUT_DIR)
        duration = (datetime.now() - start_time).total_seconds()
        print(f"\nDuration : {duration:.1f}s ({duration/60:.1f} min)")
        print(f"Throughput: {metadata['total_rows']/duration:,.0f} rows/s\n")

    except Exception as e:
        print("\n" + "=" * 80)
        import traceback; traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
