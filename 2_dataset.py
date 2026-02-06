#!/usr/bin/env python3
"""
PARALLEL DATASET PREPARATION - MULTI-CORE OPTIMIZED
====================================================
Reads individual daily parquet files -> Splits -> Normalizes -> Saves .npz

OPTIMIZATIONS:
1. Parallel Statistics Computation (Map-Reduce)
2. Parallel Dataset Generation (Multi-worker writing)
3. Zero-copy memory mapping where possible
4. Preserves exact output structure/order

HARDWARE TARGET: Intel i5-11400H, 8GB RAM
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
import psutil

# =============================================================================
# CONFIGURATION
# =============================================================================

INPUT_DIR = "./processed_daily_parquets"
OUTPUT_DIR = "./dataset_lgbm"
SCALER_FILE = "./scaler_lgbm.pkl"

# Chronological splits by DAYS
TRAIN_SPLIT = 0.70
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

# Hardware Tuning
NUM_WORKERS = 5  # Safe for 8GB RAM (leaves room for OS)
CHUNK_SIZE = 10  # Days per worker batch

FEATURE_COLUMNS = [
    'qty', 'time_delta', 'log_return',
    'trade_intensity', 'price_acceleration', 
    'volume_imbalance', 'rolling_signed_volume', 'side',
    'price_volatility', 'volume_volatility', 'time_delta_std',
    'price_distance_from_ma'
]

# =============================================================================
# SCALER CLASS
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
# WORKER FUNCTIONS (Must be top-level for multiprocessing)
# =============================================================================

def _worker_compute_partial_stats(args):
    """Worker: Computes sum and sum_sq for a batch of files."""
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
                s = np.sum(arr)
                ss = np.sum(arr**2)
                local_sum[i] += s
                local_sq_sum[i] += ss
            local_count += table.num_rows
    except Exception as e:
        return None, str(e)
        
    return (local_sum, local_sq_sum, local_count), None

def _worker_process_batch(args):
    """Worker: Loads, scales, and saves a batch of days."""
    file_batch, start_idx, scaler, features, out_dir = args
    
    saved_files = []
    local_rows = 0
    local_labels = {0: 0, 1: 0, 2: 0}
    
    try:
        for i, p_file in enumerate(file_batch):
            # Calculate correct output index
            current_idx = start_idx + i
            
            # Read
            table = pq.read_table(p_file, columns=features + ['label'])
            
            # Extract
            X = np.column_stack([
                table.column(c).to_numpy() for c in features
            ]).astype(np.float32)
            y = table.column('label').to_numpy().astype(np.int8)
            
            # Scale
            X = scaler.transform(X)
            
            # Stats
            local_rows += len(y)
            u, c = np.unique(y, return_counts=True)
            counts = dict(zip(u, c))
            for k in local_labels:
                local_labels[k] += counts.get(k, 0)
            
            # Save
            out_path = out_dir / f"day_{current_idx:04d}.npz"
            np.savez_compressed(out_path, X=X, y=y)
            saved_files.append(str(out_path))
            
    except Exception as e:
        return None, str(e)

    return (saved_files, local_rows, local_labels), None

# =============================================================================
# MAIN PARALLEL PROCESSORS
# =============================================================================

def compute_statistics_parallel(train_files, feature_columns):
    """Parallel Map-Reduce for statistics."""
    print("\n" + "=" * 80)
    print(f"COMPUTING STATISTICS (PARALLEL - {NUM_WORKERS} WORKERS)")
    print("=" * 80)
    
    # Split files into chunks
    chunks = [train_files[i:i + CHUNK_SIZE] for i in range(0, len(train_files), CHUNK_SIZE)]
    tasks = [(chunk, feature_columns) for chunk in chunks]
    
    n_features = len(feature_columns)
    total_sum = np.zeros(n_features, dtype=np.float64)
    total_sq_sum = np.zeros(n_features, dtype=np.float64)
    total_count = 0
    
    with mp.Pool(NUM_WORKERS) as pool:
        results = list(tqdm(
            pool.imap(_worker_compute_partial_stats, tasks),
            total=len(tasks),
            desc="Aggregating stats"
        ))
    
    # Reduce
    for res, err in results:
        if err:
            raise RuntimeError(f"Worker failed: {err}")
        l_sum, l_sq, l_count = res
        total_sum += l_sum
        total_sq_sum += l_sq
        total_count += l_count
        
    mean = (total_sum / total_count).astype(np.float32)
    mean_sq = total_sq_sum / total_count
    variance = np.maximum(mean_sq - (mean.astype(np.float64) ** 2), 0)
    std = np.sqrt(variance).astype(np.float32)
    
    print(f"[OK] Stats computed on {total_count:,} samples")
    return mean, std, total_count

def process_split_parallel(split_name, files, scaler, features, output_dir):
    """Parallel batch processing for dataset generation."""
    if not files:
        return [], 0, {0: 0, 1: 0, 2: 0}

    split_dir = Path(output_dir) / split_name
    split_dir.mkdir(parents=True, exist_ok=True)
    
    # Create tasks with explicit start indices to preserve naming order
    tasks = []
    for i in range(0, len(files), CHUNK_SIZE):
        batch = files[i:i + CHUNK_SIZE]
        tasks.append((batch, i, scaler, features, split_dir))
        
    print(f"\nProcessing {split_name} ({len(files)} days)...")
    
    all_saved_files = []
    total_rows = 0
    total_labels = {0: 0, 1: 0, 2: 0}
    
    with mp.Pool(NUM_WORKERS) as pool:
        results = list(tqdm(
            pool.imap(_worker_process_batch, tasks),
            total=len(tasks),
            desc=f"Writing {split_name}"
        ))
        
    for res, err in results:
        if err:
            raise RuntimeError(f"Worker failed: {err}")
        
        batch_files, batch_rows, batch_labels = res
        all_saved_files.extend(batch_files)
        total_rows += batch_rows
        for k, v in batch_labels.items():
            total_labels[k] += v
            
    # Sort to ensure manifest order is perfect (though tasks usually finish in order)
    all_saved_files.sort()
    
    return all_saved_files, total_rows, total_labels

# =============================================================================
# PIPELINE UTILS
# =============================================================================

def load_parquet_files(input_dir):
    input_path = Path(input_dir)
    metadata_file = input_path / "metadata.json"
    if metadata_file.exists():
        with open(metadata_file) as f:
            meta = json.load(f)
            print(f"[METADATA] Found {meta['total_files']} files")
            
    files = sorted(input_path.glob("day_*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquets in {input_dir}")
    return files

def compute_day_splits(files, train_pct, val_pct):
    n = len(files)
    t_end = int(n * train_pct)
    v_end = t_end + int(n * val_pct)
    return files[:t_end], files[t_end:v_end], files[v_end:]

# =============================================================================
# MAIN
# =============================================================================

def main():
    start_time = time.time()
    print("=" * 80)
    print(f"PARALLEL DATASET GEN | Workers: {NUM_WORKERS} | Batch: {CHUNK_SIZE}")
    print("=" * 80)
    
    # 1. Load Files
    parquet_files = load_parquet_files(INPUT_DIR)
    
    # 2. Split
    train, val, test = compute_day_splits(parquet_files, TRAIN_SPLIT, VAL_SPLIT)
    print(f"Split: {len(train)} Train, {len(val)} Val, {len(test)} Test")
    
    # 3. Parallel Stats
    mean, std, _ = compute_statistics_parallel(train, FEATURE_COLUMNS)
    scaler = SimpleScaler(mean, std)
    scaler.save(SCALER_FILE)
    
    # 4. Parallel Processing
    manifest = {}
    split_stats = {}
    
    for name, files in [('train', train), ('val', val), ('test', test)]:
        f_list, rows, labels = process_split_parallel(
            name, files, scaler, FEATURE_COLUMNS, OUTPUT_DIR
        )
        manifest[name] = f_list
        split_stats[name] = {'rows': rows, 'labels': labels}
    
    # 5. Save Manifest
    with open(Path(OUTPUT_DIR) / "manifest.json", 'w') as f:
        json.dump(manifest, f, indent=2)
        
    duration = time.time() - start_time
    print("\n" + "=" * 80)
    print(f"DONE in {duration:.1f}s ({duration/60:.1f} min)")
    print(f"Scaler saved to {SCALER_FILE}")
    print("=" * 80)

if __name__ == "__main__":
    # Windows/multiprocessing safeguard
    mp.freeze_support()
    main()