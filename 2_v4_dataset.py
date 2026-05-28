#!/usr/bin/env python3
"""
PARALLEL DATASET PREPARATION v4 (REGRESSION — NO WEIGHT)
=========================================================
Reads daily parquets -> Splits -> Normalizes -> Saves .npz

CHANGES FROM v3:
  - Removed weight column (P3: no more double-penalty with sniper objective)
  - .npz files now contain only X and y
  - Hardware config updated for i9 + 128GB RAM

HARDWARE TARGET: i9-14900K, 128GB RAM
"""

import numpy as np
import pyarrow.parquet as pq
import pickle
from pathlib import Path
from tqdm import tqdm
import gc
import json
from dataclasses import dataclass
import time
import multiprocessing as mp
import sys

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

# =============================================================================
# CONFIGURATION
# =============================================================================

INPUT_DIR   = "./processed_daily_parquets_v4"
OUTPUT_DIR  = "./dataset_xgb_regression_v4"
SCALER_FILE = "./scaler_regression_v4.pkl"

# Chronological splits (70% Train, 15% Val, 15% Test)
TRAIN_SPLIT = 0.80
VAL_SPLIT   = 0.15
TEST_SPLIT  = 0.05

# SMART SCHEDULER CONFIG (i5-11400H, 16 GB RAM)
TARGET_BATCH_MB       = 25
MAX_PARALLEL_SIZE_MB  = 60
OOM_SAFEGUARD_PERCENT = 75.0
NUM_WORKERS           = 2

# =============================================================================
# RAM UTILS
# =============================================================================

def get_ram_usage():
    if HAS_PSUTIL:
        return psutil.virtual_memory().percent
    return 0.0

def is_ram_critical():
    return get_ram_usage() > OOM_SAFEGUARD_PERCENT

# =============================================================================
# SCALER
# =============================================================================

@dataclass
class SimpleScaler:
    mean_: np.ndarray
    std_: np.ndarray

    def transform(self, X):
        return (X - self.mean_) / (self.std_ + 1e-8)

    def save(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump({'mean': self.mean_, 'std': self.std_}, f)

    @classmethod
    def load(cls, filepath):
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        return cls(mean_=data['mean'], std_=data['std'])

# =============================================================================
# SMART BATCH GENERATOR
# =============================================================================

def generate_smart_batches(files, target_mb=TARGET_BATCH_MB):
    batches = []
    current_files = []
    current_size = 0.0

    for f in files:
        size_mb = f.stat().st_size / (1024 * 1024)
        if current_files and (current_size + size_mb > target_mb):
            batches.append({'files': current_files, 'size_mb': current_size})
            current_files = []
            current_size = 0.0
        current_files.append(f)
        current_size += size_mb

    if current_files:
        batches.append({'files': current_files, 'size_mb': current_size})
    return batches

# =============================================================================
# WORKER FUNCTIONS
# =============================================================================

def _worker_compute_partial_stats(args):
    """Worker: Computes sum/sq_sum for a batch."""
    files, features = args
    n_features = len(features)

    local_sum = np.zeros(n_features, dtype=np.float64)
    local_sq_sum = np.zeros(n_features, dtype=np.float64)
    local_count = 0

    try:
        for p_file in files:
            table = pq.read_table(p_file, columns=features)
            for i, col in enumerate(features):
                arr = table.column(col).to_numpy().astype(np.float64)
                local_sum[i] += np.sum(arr)
                local_sq_sum[i] += np.sum(arr**2)
            local_count += table.num_rows
            del table
            gc.collect()
    except Exception as e:
        return None, str(e)

    return (local_sum, local_sq_sum, local_count), None


def _worker_process_batch(args):
    """Worker: Writes .npz files (no weight column)."""
    batch_idx, file_batch, scaler, features, out_dir = args

    saved_files = []
    local_rows = 0

    try:
        for p_file in file_batch:
            cols = features + ['target']
            table = pq.read_table(p_file, columns=cols)

            # Extract
            X_cols = [table.column(c).to_numpy().astype(np.float32) for c in features]
            X = np.column_stack(X_cols)
            y = table.column('target').to_numpy().astype(np.float32)

            # Scale
            X = scaler.transform(X)

            # Save — no weight
            out_path = out_dir / f"{p_file.stem}.npz"
            np.savez_compressed(out_path, X=X, y=y)

            saved_files.append(str(out_path))
            local_rows += len(y)

            del table, X, y
            gc.collect()

    except Exception as e:
        return batch_idx, None, str(e)

    return batch_idx, (saved_files, local_rows), None

# =============================================================================
# SMART PROCESSORS
# =============================================================================

def compute_statistics_smart(train_files, feature_columns):
    print(f"\n[STATS] Computing global statistics...")

    batches = generate_smart_batches(train_files, TARGET_BATCH_MB)
    tasks = [(b['files'], feature_columns) for b in batches]

    total_sum = np.zeros(len(feature_columns), dtype=np.float64)
    total_sq_sum = np.zeros(len(feature_columns), dtype=np.float64)
    total_count = 0

    with mp.Pool(NUM_WORKERS) as pool:
        for res, err in tqdm(pool.imap_unordered(_worker_compute_partial_stats, tasks), total=len(tasks)):
            if err: raise RuntimeError(f"Stats worker failed: {err}")
            l_sum, l_sq, l_count = res
            total_sum += l_sum
            total_sq_sum += l_sq
            total_count += l_count

    mean = (total_sum / total_count).astype(np.float32)
    var = (total_sq_sum / total_count) - (mean.astype(np.float64)**2)
    std = np.sqrt(np.maximum(var, 0)).astype(np.float32)

    print(f"[STATS] Done. Count: {total_count:,}")
    return mean, std


def process_split_smart_scheduler(split_name, files, scaler, features, output_dir):
    if not files: return [], 0

    split_dir = Path(output_dir) / split_name
    split_dir.mkdir(parents=True, exist_ok=True)

    batches = generate_smart_batches(files, TARGET_BATCH_MB)
    total_batches = len(batches)

    print(f"\n[PROCESS] {split_name}: {len(files)} files -> {total_batches} batches")

    queue_idx = 0
    active_mb = 0.0
    pending = []
    results_store = []
    total_rows = 0

    pool = mp.Pool(NUM_WORKERS)
    pbar = tqdm(total=total_batches, desc=f"Writing {split_name}")

    try:
        while queue_idx < total_batches or pending:
            still_pending = []
            for b_idx, ar, sz in pending:
                if ar.ready():
                    res_idx, res_data, err = ar.get()
                    if err: raise RuntimeError(f"Batch {res_idx} failed: {err}")
                    saved_files, batch_rows = res_data
                    results_store.extend(saved_files)
                    total_rows += batch_rows
                    active_mb -= sz
                    pbar.update(1)
                else:
                    still_pending.append((b_idx, ar, sz))
            pending = still_pending

            if queue_idx < total_batches:
                if is_ram_critical():
                    time.sleep(1)
                    continue

                batch = batches[queue_idx]
                bsz = batch['size_mb']

                if active_mb + bsz <= MAX_PARALLEL_SIZE_MB:
                    args = (queue_idx, batch['files'], scaler, features, split_dir)
                    ar = pool.apply_async(_worker_process_batch, (args,))
                    pending.append((queue_idx, ar, bsz))
                    active_mb += bsz
                    queue_idx += 1
                elif not pending:
                    args = (queue_idx, batch['files'], scaler, features, split_dir)
                    ar = pool.apply_async(_worker_process_batch, (args,))
                    pending.append((queue_idx, ar, bsz))
                    active_mb += bsz
                    queue_idx += 1
                else:
                    time.sleep(0.1)
            else:
                time.sleep(0.1)

    finally:
        pool.close()
        pool.join()
        pbar.close()

    results_store.sort()
    return results_store, total_rows

# =============================================================================
# MAIN
# =============================================================================

def main():
    start = time.time()
    print("="*80)
    print(f"DATASET GEN v4 (NO WEIGHT) | Limit: {MAX_PARALLEL_SIZE_MB}MB")
    print("="*80)

    # Load Meta
    meta_path = Path(INPUT_DIR) / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError("Run Stage 1 first!")
    with open(meta_path) as f: meta = json.load(f)
    features = meta['feature_columns']

    files = sorted(Path(INPUT_DIR).glob("day_*.parquet"))

    # Split chronologically
    n = len(files)
    i1 = int(n * TRAIN_SPLIT)
    i2 = i1 + int(n * VAL_SPLIT)
    splits = [('train', files[:i1]), ('val', files[i1:i2]), ('test', files[i2:])]

    print(f"\n  Total files: {n}")
    print(f"  Train: {len(splits[0][1])} | Val: {len(splits[1][1])} | Test: {len(splits[2][1])}")
    print(f"  Features: {len(features)}")

    # Compute scaler on training set only
    mean, std = compute_statistics_smart(splits[0][1], features)
    scaler = SimpleScaler(mean, std)
    scaler.save(SCALER_FILE)

    # Process all splits
    manifest = {'feature_columns': features}
    for name, f_list in splits:
        saved, rows = process_split_smart_scheduler(name, f_list, scaler, features, OUTPUT_DIR)
        manifest[name] = saved
        print(f"  > {name}: {rows:,} rows")

    with open(Path(OUTPUT_DIR) / "manifest.json", 'w') as f:
        json.dump(manifest, f, indent=2)

    print(f"\nDONE in {time.time()-start:.1f}s")


if __name__ == "__main__":
    mp.freeze_support()
    main()
