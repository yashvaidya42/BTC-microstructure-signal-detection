import numpy as np
import xgboost as xgb
import optuna
from optuna.samplers import TPESampler
from sklearn.model_selection import TimeSeriesSplit
import pickle
import json
from pathlib import Path
from typing import Tuple, List, Dict, Optional, Callable
import warnings
from tqdm import tqdm
import gc
from datetime import datetime
import os
import sys
import signal
import logging
import traceback
import threading
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)


# =============================================================================
# SNIPER OBJECTIVE
# DISABLED: sniper objective caused collapse with 166K rows. Using reg:absoluteerror + regime gate instead.
# =============================================================================

def sniper_objective(y_pred, dtrain):
    y_true = dtrain.get_label()
    residual = y_pred - y_true

    delta = 0.005  # Huber transition (50 bps) — covers ~99% of returns within L2 region
    abs_residual = np.abs(residual)

    # Huber gradient
    grad = np.where(abs_residual <= delta, residual, delta * np.sign(residual))

    # Hessian: must be 1.0 everywhere so min_child_weight works correctly.
    # The directional penalty on gradient alone is sufficient to bias learning.
    hess = np.ones_like(residual)

    # Directional penalty: 5x gradient on wrong-direction clear moves
    CLEAR_MOVE_THRESHOLD = 0.0007  # 7 bps
    clear_move = np.abs(y_true) > CLEAR_MOVE_THRESHOLD
    wrong_direction = np.sign(y_pred) != np.sign(y_true)
    penalty_mask = clear_move & wrong_direction

    grad[penalty_mask] *= 5.0

    return grad, hess


def sniper_eval(y_pred, dtrain):
    """Evaluation metric: MAE in bps (for early stopping)."""
    y_true = dtrain.get_label()
    mae_bps = float(np.mean(np.abs(y_true - y_pred)) * 10_000)
    return 'mae_bps', mae_bps


# =============================================================================
# CONFIGURATION
# =============================================================================
DATASET_DIR = "./dataset_xgb_regression_v4"
SCALER_FILE = "./scaler_regression_v4.pkl"
OUTPUT_DIR  = "./model_output_xgb_v4"

# --- Optuna (i5-11400H, 16 GB RAM) ---

N_OPTUNA_TRIALS     = 50
OPTUNA_TIMEOUT      = 7200          # 2 hours
OPTUNA_TS_SPLITS    = 2
MAX_SAMPLES_OPTUNA  = 500_000       # with 10s subsampling, total ~180K — use all
OPTUNA_SUBSAMPLE_PER_FILE = 100_000 # effectively no per-file cap after subsampling
OPTUNA_N_JOBS       = 1             # sequential trials (6 cores, no room for parallel)
OPTUNA_NTHREAD_PER_WORKER = 4       # CPU threads per Optuna worker

# --- Training ---
MAX_BOOST_ROUNDS    = 5000
EARLY_STOP_ROUNDS   = 100
MAX_TRAIN_ROWS      = 500_000       # safety cap — subsampled data is ~180K total

# --- Feature Selection ---
CORRELATION_THRESHOLD   = 0.90
CORRELATION_SAMPLE_SIZE = 200_000

# --- Purge (X6) ---
LOOK_AHEAD_SECONDS = 300
PURGE_GAP_ROWS     = 5             # 300s / 60s subsample interval

# --- Memory safety (i5-11400H, 16 GB RAM) ---
RAM_ABORT_PERCENT = 80
# --- Batched iterator ---
ITERATOR_FILES_PER_BATCH = 2  # ~300 MB per GPU batch, safe on 4 GB VRAM

# --- Crash recovery ---
CRASH_RECOVERY_FILE = None  # set in main() after OUTPUT_DIR is created
LOG_FILE            = None

# --- Optuna cache (Fix 4) ---
# Cache path is computed dynamically per keep_idx to avoid column-set mismatches (Bug 1).
# Stored in OUTPUT_DIR (not DATASET_DIR) to avoid writing to read-only input dirs (Warning 1).

# =============================================================================
# LOGGING SETUP
# =============================================================================

def setup_logging(output_dir: str):
    """Dual logging: console + file."""
    global LOG_FILE
    LOG_FILE = os.path.join(output_dir, "training_log.txt")
    logger = logging.getLogger('xgb_v4')
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    
    # ADDED: encoding='utf-8' to prevent Windows emoji crashes
    fh = logging.FileHandler(LOG_FILE, mode='w', encoding='utf-8')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        '%(asctime)s | %(levelname)-7s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    ))

    logger.addHandler(fh)
    # Console handler — INFO and above
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(ch)
    return logger
log = None  # initialized in main()
def L(msg, level='info'):
    """Convenience: log to both console and file."""
    if log is None:
        print(msg)
        return
    getattr(log, level)(msg)

# =============================================================================
# RAM SAFETY
# =============================================================================

def get_ram_info() -> Dict:

    try:
        import psutil
        mem = psutil.virtual_memory()
        return {
            'percent': mem.percent,
            'used_gb': mem.used / 1e9,
            'avail_gb': mem.available / 1e9,
            'total_gb': mem.total / 1e9,
        }
    except ImportError:
        return {'percent': 0, 'used_gb': 0, 'avail_gb': 999, 'total_gb': 0}

def check_ram_or_abort(needed_gb: float, context: str):
    """Hard abort if allocation would exceed safe RAM threshold."""
    info = get_ram_info()
    if needed_gb > info['avail_gb'] * 0.85:
        msg = (f"[RAM ABORT] {context}: need ~{needed_gb:.1f} GB "
               f"but only {info['avail_gb']:.1f} GB available "
               f"({info['percent']:.0f}% used). Stopping to prevent system harm.")
        L(msg, 'critical')
        sys.exit(1)
    if info['percent'] > RAM_ABORT_PERCENT:
        msg = (f"[RAM ABORT] {context}: RAM at {info['percent']:.0f}% "
               f"(>{RAM_ABORT_PERCENT}%). Stopping.")
        L(msg, 'critical')
        sys.exit(1)

def log_ram(prefix=""):
    info = get_ram_info()
    L(f"{prefix}RAM: {info['percent']:.1f}% "
      f"({info['used_gb']:.1f}/{info['total_gb']:.1f} GB, "
      f"{info['avail_gb']:.1f} GB free)")
# =============================================================================
# CRASH RECOVERY — dump Optuna state + model on any failure
# =============================================================================

class CrashRecovery:
    """Accumulates state that gets dumped to disk on crash or completion."""
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.filepath = os.path.join(output_dir, "crash_recovery.json")

        # Load existing data if it's there
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, 'r') as f:
                    self.data = json.load(f)
            except Exception:
                self._init_empty()
        else:
            self._init_empty()

    def _init_empty(self):
        self.data = {
            'started': datetime.now().isoformat(),
            'optuna_trials': [],
            'best_5_optuna': [],
            'gpu_config': {},
            'keep_idx': [],
            'training_params': {},
            'training_status': 'not_started',
            'errors': [],
        }
        self._save()

    def add_trial(self, trial_info: Dict):
        self.data['optuna_trials'].append(trial_info)
        self._update_best5()
        self._save()

    def _update_best5(self):
        completed = [t for t in self.data['optuna_trials']
                     if t.get('status') == 'complete' and t.get('value', 999) < 999]
        completed.sort(key=lambda t: t['value'])

        best5 = []
        for t in completed[:5]:
            entry = dict(t)  
            entry['mae_bps'] = entry.get('mae_bps', entry.get('value', 999))
            best5.append(entry)
        self.data['best_5_optuna'] = best5

    def set(self, key: str, value):
        self.data[key] = value
        self._save()

    def add_error(self, error_msg: str):
        self.data['errors'].append({
            'time': datetime.now().isoformat(),
            'error': error_msg,
        })
        self._save()

    def _save(self):
        try:
            with open(self.filepath, 'w') as f:
                json.dump(self.data, f, indent=2, default=str)
        except Exception:
            pass 

    def dump_final(self):
        self.data['finished'] = datetime.now().isoformat()
        self._save()
        L(f"  Recovery data saved to {self.filepath}")
recovery = None  # initialized in main()

def signal_handler(sig, frame):
    """On SIGINT/SIGTERM, dump whatever we have and exit."""
    L("\n[SIGNAL] Caught interrupt — saving recovery data...")
    if recovery:
        recovery.add_error(f"Interrupted by signal {sig}")
        recovery.dump_final()
    sys.exit(1)

# =============================================================================
# GPU DETECTION (X10)
# =============================================================================

def detect_gpu_config() -> Dict:

    L("\n" + "=" * 60)
    L("GPU CONFIGURATION")
    L("=" * 60)
    try:
        import subprocess
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,name,memory.total,memory.free',
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode != 0:
            raise RuntimeError("nvidia-smi failed")
        gpus = []
        for line in result.stdout.strip().split('\n'):
            parts = [p.strip() for p in line.split(',')]
            if len(parts) >= 4:
                gpus.append({
                    'index': int(parts[0]),
                    'name': parts[1],
                    'total_mb': int(parts[2]),
                    'free_mb': int(parts[3]),
                })
        if not gpus:
            raise RuntimeError("No GPUs detected")
        for g in gpus:
            L(f"  GPU {g['index']}: {g['name']} — "
              f"{g['free_mb']:,} MB free / {g['total_mb']:,} MB total")
        total_free_mb = sum(g['free_mb'] for g in gpus)
        if len(gpus) >= 2 and total_free_mb > 60_000:
            gpu_ids = ','.join(str(g['index']) for g in gpus)
            os.environ['CUDA_VISIBLE_DEVICES'] = gpu_ids
            config = {

                'tree_method': 'hist',
                'device': 'cuda',
                'max_bin': 256,
                'mode': f'multi-gpu ({len(gpus)} GPUs)',
                'n_gpus': len(gpus),
                'total_vram_mb': total_free_mb,
            }
            L(f"\n  -> MULTI-GPU: {len(gpus)} GPUs, {total_free_mb:,} MB combined free")
            return config
        if gpus and gpus[0]['free_mb'] > 2_000:
            os.environ['CUDA_VISIBLE_DEVICES'] = str(gpus[0]['index'])
            config = {
                'tree_method': 'hist',
                'device': 'cuda',
                'max_bin': 256 if gpus[0]['free_mb'] > 30_000 else 128,
                'mode': f"single-gpu ({gpus[0]['name']})",
                'n_gpus': 1,
                'total_vram_mb': gpus[0]['free_mb'],
            }
            L(f"\n  -> SINGLE GPU: {gpus[0]['name']}, {gpus[0]['free_mb']:,} MB free")
            return config
        raise RuntimeError(f"Insufficient VRAM: {total_free_mb} MB")
    except Exception as e:
        L(f"\n  GPU detection failed: {e}")
        L(f"  -> CPU FALLBACK")

        return {
            'tree_method': 'hist',
            'device': 'cpu',
            'max_bin': 256,
            'mode': 'cpu',
            'n_gpus': 0,
            'total_vram_mb': 0,
        }

# =============================================================================
# DATA LOADING
# =============================================================================

def load_manifest() -> Dict:
    with open(Path(DATASET_DIR) / "manifest.json", 'r') as f:
        return json.load(f)
    
# ---------------------------------------------------------------------------
# Batched .npz iterator (used as fallback for CPU ExtMem path)
# ---------------------------------------------------------------------------

class BatchedNpzIterator(xgb.DataIter):
    def __init__(self, file_list: List[str], keep_idx: Optional[List[int]] = None,
                 batch_size: int = ITERATOR_FILES_PER_BATCH, label: str = ""):
        self._file_list = file_list
        self._keep_idx = keep_idx
        self._batch_size = batch_size
        self._it = 0
        self._label = label or "DMatrix"
        self._total_batches = (len(file_list) + batch_size - 1) // batch_size
        self._batch_count = 0
        self._pbar = None
        super().__init__(cache_prefix=None, on_host=True)

    def next(self, input_data: Callable) -> bool:
        if self._it >= len(self._file_list):
            if self._pbar is not None:
                self._pbar.close()
                self._pbar = None
            return False

        # Lazy-init tqdm on first batch of each pass
        if self._pbar is None:
            self._pbar = tqdm(total=self._total_batches,
                              desc=f"Building {self._label}", unit="batch",
                              position=0, leave=False)

        end = min(self._it + self._batch_size, len(self._file_list))
        batch_files = self._file_list[self._it:end]

        # Fix D: Per-batch RAM check — rough estimate ~200k rows/file * n_feat * 4 bytes
        n_feat_est = len(self._keep_idx) if self._keep_idx else 15  # fallback to 15
        est_gb = len(batch_files) * 200_000 * n_feat_est * 4 / 1e9
        check_ram_or_abort(est_gb, f"Iterator batch ({len(batch_files)} files)")

        Xs, ys = [], []
        for f in batch_files:
            try:
                data = np.load(f)
                X = data['X'].astype(np.float32)
                y = data['y'].astype(np.float32)

                X[np.isinf(X)] = np.nan
                y[np.isinf(y)] = np.nan
                valid = ~np.isnan(y)
                X = X[valid]
                y = y[valid]

                if self._keep_idx is not None:
                    X = X[:, self._keep_idx]

                Xs.append(X)
                ys.append(y)
                del data, X, y
            except Exception as e:
                L(f"  [WARN] Iterator skipping {f}: {e}", 'warning')

        if Xs:
            X_batch = np.vstack(Xs)
            y_batch = np.concatenate(ys)
            input_data(data=X_batch, label=y_batch)
            del X_batch, y_batch

        self._it = end
        self._batch_count += 1
        if self._pbar is not None:
            self._pbar.update(1)

        return True

    def reset(self) -> None:
        """Idempotent reset — just rewind the file cursor."""
        if self._pbar is not None:
            self._pbar.close()
            self._pbar = None
        self._it = 0
        self._batch_count = 0

# ---------------------------------------------------------------------------

# Preallocate-then-fill for Optuna (small enough to fit in RAM)

# ---------------------------------------------------------------------------

def _optuna_cache_path(keep_idx: Optional[List[int]]) -> str:
    """Compute cache filename keyed to keep_idx identity (Bug 1 fix)."""
    import hashlib
    tag = hashlib.md5(str(keep_idx).encode()).hexdigest()[:10]
    return os.path.join(OUTPUT_DIR, f"optuna_cache_{tag}.npz")

def load_preallocated_for_optuna(
    file_list: List[str],
    keep_idx: Optional[List[int]] = None,
    max_rows: int = MAX_SAMPLES_OPTUNA,
    subsample_per_file: int = OPTUNA_SUBSAMPLE_PER_FILE,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load a time-stratified subsample for Optuna.
    Fix 4: One-time sequential cache — first run builds optuna_cache_<hash>.npz (~1.3 GB),
    subsequent runs load in <10s. Deterministic per-file seed (42 + file_idx) for
    bit-identical row order vs original sequential version.
    Bug 1 fix: cache filename includes keep_idx hash; keep_idx stored inside .npz
    and validated on load to prevent column-set mismatches.
    """
    cache_path = _optuna_cache_path(keep_idx)

    # --- Check for existing cache ---
    if os.path.exists(cache_path):
        L(f"  [Optuna] Loading cached subsample from {cache_path}")
        cached = np.load(cache_path, allow_pickle=True)
        # Validate keep_idx matches (Bug 1 + Bug 2 guard)
        cached_kidx = cached['keep_idx'].tolist() if 'keep_idx' in cached else None
        if cached_kidx != (list(keep_idx) if keep_idx else None):
            L("  [Optuna] Cache keep_idx mismatch — rebuilding.", 'warning')
        else:
            X, y = cached['X'], cached['y']
            L(f"  [Optuna] Cached: {X.shape[0]:,} rows x {X.shape[1]} features "
              f"({X.nbytes / 1e9:.1f} GB)")
            log_ram("  [Optuna-cache] ")
            return X, y

    # --- Build cache from scratch ---
    L(f"  [Optuna] No valid cache found, building from {len(file_list)} files...")
    est_rows = min(subsample_per_file * len(file_list), max_rows)
    # Get n_features from first file
    first = np.load(file_list[0])
    n_feat_raw = first['X'].shape[1]
    del first
    n_feat = len(keep_idx) if keep_idx else n_feat_raw
    est_gb = (est_rows * n_feat * 4 + est_rows * 4) / 1e9
    check_ram_or_abort(est_gb * 1.5, "Optuna data load")
    L(f"  [Optuna] Preallocating {est_rows:,} rows x {n_feat} features (~{est_gb:.1f} GB)")
    X = np.empty((est_rows, n_feat), dtype=np.float32)
    y = np.empty(est_rows, dtype=np.float32)
    offset = 0
    # Fix 4: Per-file deterministic seed for reproducibility across runs
    for file_idx, f in enumerate(tqdm(file_list, desc="Building Optuna cache", disable=False)):
        try:
            data = np.load(f)
            Xc = data['X'].astype(np.float32)
            yc = data['y'].astype(np.float32)
            Xc[np.isinf(Xc)] = np.nan
            yc[np.isinf(yc)] = np.nan
            valid = ~np.isnan(yc)
            Xc = Xc[valid]
            yc = yc[valid]
            if len(yc) > subsample_per_file:
                rng = np.random.RandomState(42 + file_idx)
                idx = rng.choice(len(yc), subsample_per_file, replace=False)
                idx.sort()
                Xc = Xc[idx]
                yc = yc[idx]
            if keep_idx:
                Xc = Xc[:, keep_idx]
            n = len(yc)
            if offset + n > est_rows:
                n = est_rows - offset
                Xc = Xc[:n]
                yc = yc[:n]
            X[offset:offset + n] = Xc
            y[offset:offset + n] = yc
            offset += n
            del data, Xc, yc
            gc.collect()
            if offset >= max_rows:
                break
        except Exception as e:
            L(f"  [WARN] Skipping {f}: {e}", 'warning')
    X = X[:offset]
    y = y[:offset]
    L(f"  [Optuna] Loaded {offset:,} rows ({offset * n_feat * 4 / 1e9:.1f} GB)")
    # Save cache with keep_idx embedded for validation (Bug 1 fix)
    try:
        L(f"  [Optuna] Saving cache to {cache_path}...")
        kidx_arr = np.array(keep_idx if keep_idx else [], dtype=np.int64)
        np.savez_compressed(cache_path, X=X, y=y, keep_idx=kidx_arr)
        L(f"  [Optuna] Cache saved.")
    except Exception as e:
        L(f"  [WARN] Cache save failed (read-only dir?): {e}", 'warning')
    log_ram("  [Optuna] ")
    return X, y

# ---------------------------------------------------------------------------
# Streaming evaluation (for test set — memory safe)
# ---------------------------------------------------------------------------
def stream_chunks_for_eval(file_list, keep_idx=None):
    for f in file_list:
        try:
            data = np.load(f)
            X = data['X'].astype(np.float32)
            yc = data['y'].astype(np.float32)
            X[np.isinf(X)] = np.nan
            yc[np.isinf(yc)] = np.nan
            valid = ~np.isnan(yc)
            X = X[valid]
            yc = yc[valid]
            if keep_idx:
                X = X[:, keep_idx]
            yield X, yc
            del data, X, yc
            gc.collect()
        except Exception as e:
            L(f"  [WARN] Eval skip {f}: {e}", 'warning')


# =============================================================================
# SUBSAMPLED DATA LOADER (for in-memory QuantileDMatrix)
# =============================================================================

def load_data_subsampled(
    file_list: List[str],
    keep_idx: List[int],
    max_rows: int = MAX_TRAIN_ROWS,
    label: str = "",
) -> Tuple[np.ndarray, np.ndarray]:
    total_rows = 0
    for f in file_list:
        data = np.load(f)
        total_rows += data['X'].shape[0]
        del data

    stride = max(1, total_rows // max_rows)
    expected = total_rows // stride
    L(f"  [{label}] {total_rows:,} rows, stride {stride} -> ~{expected:,} rows "
      f"({expected * len(keep_idx) * 4 / 1e9:.2f} GB)")

    Xs, ys = [], []
    for f in tqdm(file_list, desc=f"Loading {label}", leave=False):
        data = np.load(f)
        X = data['X'].astype(np.float32)
        y = data['y'].astype(np.float32)
        X[np.isinf(X)] = np.nan
        y[np.isinf(y)] = np.nan
        valid = ~np.isnan(y)
        X, y = X[valid], y[valid]
        if stride > 1:
            X, y = X[::stride], y[::stride]
        if keep_idx is not None:
            X = X[:, keep_idx]
        Xs.append(X)
        ys.append(y)
        del data

    X = np.vstack(Xs)
    y = np.concatenate(ys)
    del Xs, ys
    gc.collect()
    L(f"  [{label}] Loaded {len(y):,} rows")
    return X, y

# =============================================================================
# CORRELATION FILTERING
# =============================================================================

def compute_correlation_filter(

    file_list: List[str],
    feature_names: List[str],
    threshold: float = CORRELATION_THRESHOLD,
) -> List[int]:
    """Load a small subset for correlation analysis."""
    L("\n" + "=" * 60)
    L("CORRELATION FILTERING")
    L(f"  Threshold: {threshold}")
    L("=" * 60)
    # Load first 5 files (~500k-2M rows)
    Xs = []
    total = 0
    for f in file_list[:5]:
        data = np.load(f)
        Xs.append(data['X'].astype(np.float32))
        total += data['X'].shape[0]
        del data
    X = np.vstack(Xs)
    del Xs; gc.collect()
    n_feat = X.shape[1]
    if len(X) > CORRELATION_SAMPLE_SIZE:
        idx = np.random.choice(len(X), CORRELATION_SAMPLE_SIZE, replace=False)
        X_s = X[idx]
    else:
        X_s = X
    X_corr = np.nan_to_num(X_s, nan=0.0)
    L(f"  Computing on {len(X_corr):,} samples, {n_feat} features...")
    corr = np.corrcoef(X_corr, rowvar=False)
    to_drop = set()
    for i in range(n_feat):
        if i in to_drop:
            continue
        for j in range(i + 1, n_feat):
            if j in to_drop:
                continue
            if abs(corr[i, j]) > threshold:
                mi = np.mean(np.abs(corr[i, :]))
                mj = np.mean(np.abs(corr[j, :]))
                drop_idx = i if mi > mj else j
                keep_name = j if mi > mj else i
                to_drop.add(drop_idx)
                L(f"  DROP {feature_names[drop_idx]:<30} "
                  f"(corr={corr[i,j]:.4f} with {feature_names[keep_name]})")
    keep = [i for i in range(n_feat) if i not in to_drop]
    L(f"\n  Dropped {len(to_drop)}, keeping {len(keep)}")
    del X, X_s, X_corr
    gc.collect()
    return keep

# =============================================================================
# TRADER'S REPORT
# =============================================================================

def traders_report(y_true: np.ndarray, y_pred: np.ndarray, label: str = "") -> Dict:
    report = {}
    n = len(y_true)
    pred_std = np.std(y_pred)
    pred_range = np.ptp(y_pred)
    if pred_std < 1e-7 or pred_range < 1e-7:
        if label:
            L(f"  *** MODEL COLLAPSE DETECTED *** ({label})")
        report['collapsed'] = True
        report['mae_bps'] = 999.0
        return report
    report['collapsed'] = False
    mae_bps = np.mean(np.abs(y_true - y_pred)) * 10_000
    report['mae_bps'] = float(mae_bps)
    move_mask = np.abs(y_true) > 1e-4
    n_moves = move_mask.sum()
    dir_acc = float((np.sign(y_pred[move_mask]) == np.sign(y_true[move_mask])).mean()) if n_moves > 0 else 0.0
    report['directional_accuracy'] = dir_acc
    report['n_real_moves'] = int(n_moves)
    abs_pred = np.abs(y_pred)
    top10_thresh = np.percentile(abs_pred, 90)
    sniper_mask = abs_pred >= top10_thresh
    n_sniper = sniper_mask.sum()
    if n_sniper > 0:
        sniper_acc = float((np.sign(y_pred[sniper_mask]) == np.sign(y_true[sniper_mask])).mean())
        sniper_avg = float(np.mean(np.abs(y_true[sniper_mask])) * 10_000)
    else:
        sniper_acc = 0.0
        sniper_avg = 0.0
    report['sniper_accuracy'] = sniper_acc
    report['sniper_count'] = int(n_sniper)
    report['sniper_avg_move_bps'] = sniper_avg
    mean_pnl = float(np.mean(y_true[move_mask] * np.sign(y_pred[move_mask])) * 10_000) if n_moves > 0 else 0.0
    report['mean_pnl_bps'] = mean_pnl
    report['pred_mean'] = float(np.mean(y_pred))
    report['pred_std'] = float(np.std(y_pred))

    if label:
        L(f"\n{'='*60}")
        L(f"TRADER'S REPORT — {label}")
        L(f"{'='*60}")
        L(f"  Samples          : {n:,}")
        L(f"  MAE              : {mae_bps:.2f} bps")
        L(f"  Directional Acc  : {dir_acc:.4f}  ({n_moves:,} real moves)")
        L(f"  Sniper (top 10%) : {sniper_acc:.4f}  ({n_sniper:,} trades, avg |move| {sniper_avg:.1f} bps)")
        L(f"  Mean PnL/trade   : {mean_pnl:+.2f} bps")
        L(f"  Pred distribution: mean={report['pred_mean']:.6f}  std={report['pred_std']:.6f}")
    return report

# =============================================================================
# OPTUNA (X3, X4, X6)
# =============================================================================

def run_optuna(
    train_files: List[str],
    keep_idx: List[int],
    gpu_config: Dict,
) -> Tuple[Dict, int]:
    """
    Returns (best_params, best_num_boost_round).
    Dumps every trial to crash_recovery.json incrementally.
    Runs OPTUNA_N_JOBS trials in parallel on CPU (20M rows is too small to benefit
    from GPU — kernel launch overhead dominates). Thread lock protects shared state.
    """
    L("\n" + "=" * 60)
    L("HYPERPARAMETER OPTIMIZATION (Optuna)")
    L(f"  Trials: {N_OPTUNA_TRIALS} | Timeout: {OPTUNA_TIMEOUT}s")
    L(f"  Folds: {OPTUNA_TS_SPLITS} | Purge gap: {PURGE_GAP_ROWS:,} rows")
    L(f"  Subsample: {OPTUNA_SUBSAMPLE_PER_FILE:,}/file | Max: {MAX_SAMPLES_OPTUNA:,}")
    L(f"  Parallel workers: {OPTUNA_N_JOBS} (CPU, {OPTUNA_NTHREAD_PER_WORKER} threads each)")
    L("=" * 60)

    # Clear stale Optuna cache from previous failed runs
    for old_cache in Path(OUTPUT_DIR).glob("optuna_cache_*.npz"):
        old_cache.unlink()
        L(f"  [Optuna] Deleted stale cache: {old_cache}")

    X, y = load_preallocated_for_optuna(train_files, keep_idx=keep_idx)

    L(f"  Target stats: mean={y.mean():.6f}, std={y.std():.6f}, "
      f"min={y.min():.6f}, max={y.max():.6f}")

    L(f"  Purge gap: {PURGE_GAP_ROWS} rows ({PURGE_GAP_ROWS * 60}s at 60s subsample)")
    tscv = TimeSeriesSplit(n_splits=OPTUNA_TS_SPLITS, gap=PURGE_GAP_ROWS)
    folds = list(tscv.split(X))
    
    # ---> NEW: Pre-build GPU matrices ONCE before Optuna starts <---
    prebuilt_folds = []
    L("  [Optuna] Pre-building XGBoost QuantileDMatrices...")

    mb = gpu_config['max_bin']
    L(f"  [Optuna] Using max_bin={mb}")
    for i, (tr_idx, vl_idx) in enumerate(folds):
        L(f"  Pre-building Fold {i}: train={len(tr_idx):,}, val={len(vl_idx):,}")
        dtrain = xgb.QuantileDMatrix(X[tr_idx], label=y[tr_idx], max_bin=mb)
        dval   = xgb.QuantileDMatrix(X[vl_idx], label=y[vl_idx], ref=dtrain, max_bin=mb)
        prebuilt_folds.append((dtrain, dval, vl_idx))

    collapse_count = 0
    _optuna_lock = threading.Lock()

    def objective(trial: optuna.Trial) -> float:
        nonlocal collapse_count

        # Force CPU for Optuna — 20M rows is too small for GPU to help.
        # Each worker gets OPTUNA_NTHREAD_PER_WORKER threads.
        params = {
            'objective': 'reg:absoluteerror',
            'base_score': 0.0,
            'max_bin': gpu_config['max_bin'],
            'tree_method': gpu_config['tree_method'], # 'hist'
            'device': gpu_config['device'],           # 'cuda'
            'max_depth':        trial.suggest_int('max_depth', 3, 6),
            'learning_rate':    trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'gamma':            trial.suggest_float('gamma', 0.0, 2.0),
            'reg_alpha':        trial.suggest_float('reg_alpha', 1.0, 15.0, log=True),
            'reg_lambda':       trial.suggest_float('reg_lambda', 1.0, 15.0, log=True),
            'min_child_weight': trial.suggest_int('min_child_weight', 500, 5000),
            'subsample':        trial.suggest_float('subsample', 0.5, 0.8),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.8),
            'max_delta_step':   trial.suggest_float('max_delta_step', 0.0, 0.5),
            'nthread': OPTUNA_NTHREAD_PER_WORKER,
            'verbosity': 0,
        }

        num_rounds = trial.suggest_int('num_boost_round', 500, 3000)
        fold_maes = []

        for fold_i, (dtrain, dval, vl_idx) in enumerate(prebuilt_folds):

            try:
                model = xgb.train(
                    params, dtrain,
                    num_boost_round=num_rounds,
                    evals=[(dval, 'eval')],
                    early_stopping_rounds=50,
                    verbose_eval=False,
                )
            except Exception as e:
                L(f"  Trial {trial.number}: ERROR fold {fold_i}: {e}", 'warning')
                del dtrain, dval
                return 999.0

            preds = model.predict(dval)
            # Use original NumPy 'y' array so trader_report doesn't need to pull from GPU
            report = traders_report(y[vl_idx], preds)
            fold_mae = report.get('mae_bps', 999.0)
            fold_maes.append(fold_mae)

            # DO NOT delete dtrain or dval here, they are reused across trials!
            del model, preds

            if fold_mae >= 999.0 or report.get('collapsed', False):
                with _optuna_lock:
                    collapse_count += 1
                trial.report(999.0, fold_i)
                if trial.should_prune():
                    raise optuna.TrialPruned()
            else:
                trial.report(fold_mae, fold_i)
                if trial.should_prune():
                    raise optuna.TrialPruned()

        mean_mae = float(np.mean(fold_maes))

        # Dump to crash recovery (thread-safe)
        trial_info = {
            'number': trial.number,
            'value': mean_mae,
            'status': 'complete',
            'params': {k: v for k, v in trial.params.items()},
            'fold_maes': fold_maes,
        }
        with _optuna_lock:
            if recovery:
                recovery.add_trial(trial_info)

        L(f"  Trial {trial.number:>3d}: MAE={mean_mae:.2f}bps  "
          f"rounds={num_rounds} depth={params['max_depth']} "
          f"lr={params['learning_rate']:.4f} mcw={params['min_child_weight']}")

        return mean_mae

    sampler = TPESampler(seed=42)
    study = optuna.create_study(
        direction='minimize',
        sampler=sampler,
        pruner=optuna.pruners.MedianPruner(n_startup_trials=15, n_warmup_steps=0),
    )

    study.optimize(
        objective,
        n_trials=N_OPTUNA_TRIALS,
        timeout=OPTUNA_TIMEOUT,
        n_jobs=OPTUNA_N_JOBS,
        show_progress_bar=True,
        gc_after_trial=True,
    )



    completed = [t for t in study.trials
                 if t.state == optuna.trial.TrialState.COMPLETE]
    completed.sort(key=lambda t: t.value)

    L(f"\n{'='*60}")
    L("OPTUNA COMPLETE")
    L(f"{'='*60}")
    L(f"  Completed: {len(completed)} / {len(study.trials)}")
    L(f"  Collapses: {collapse_count}")
    if completed:
        L(f"  Best MAE: {study.best_value:.2f} bps")
        L(f"  Best params: {study.best_params}")

    for i, t in enumerate(completed[:5]):
        L(f"    #{i+1}: MAE={t.value:.2f}bps (trial {t.number})")

    if not completed:
        L("[FATAL] No Optuna trials completed successfully.", 'critical')
        if recovery:
            recovery.add_error("All Optuna trials failed or collapsed")
            recovery.dump_final()
        sys.exit(1)

    best = study.best_params.copy()
    num_boost_round = best.pop('num_boost_round', MAX_BOOST_ROUNDS)

    # Merge back GPU config for full training (Optuna used CPU, training uses GPU)
    best.update({
        'objective': 'reg:absoluteerror',
        'base_score': 0.0,
        'max_bin': gpu_config['max_bin'],
        'tree_method': gpu_config['tree_method'],
        'device': gpu_config['device'],
        'nthread': 0 if gpu_config['device'] == 'cuda' else 10,
        'verbosity': 0,
    })

    # Save best 5 to recovery
    if recovery:
        best5 = []
        for t in completed[:5]:
            p = t.params.copy()
            best5.append({'trial': t.number, 'mae_bps': t.value, 'params': p})
        recovery.set('best_5_optuna', best5)
        recovery.set('training_params', {k: v for k, v in best.items()})

    del X, y
    gc.collect()
    return best, num_boost_round

# =============================================================================
# SINGLE-MODEL TRAINING (X1, X2, X7, X10)
# =============================================================================

def train_single_model(
    train_files: List[str],
    val_files: List[str],
    keep_idx: List[int],
    params: Dict,
    num_boost_round: int,
    gpu_config: Dict,
) -> xgb.Booster:
    """
    Single xgb.train() with in-memory QuantileDMatrix (subsampled to fit VRAM).
    Full data, no subsampling, no walk-forward.
    """
    L("\n" + "=" * 60)
    L("SINGLE-MODEL TRAINING (in-memory QuantileDMatrix)")
    L(f"  Features     : {len(keep_idx)}")
    L(f"  Max rows     : {MAX_TRAIN_ROWS:,}")
    L(f"  Max rounds   : {num_boost_round}")
    L(f"  Early stop   : {EARLY_STOP_ROUNDS} rounds")
    L(f"  GPU          : {gpu_config['mode']}")
    L(f"  Train files  : {len(train_files)}")
    L(f"  Val files    : {len(val_files)}")
    L(f"  Objective    : reg:absoluteerror + regime gate")
    L("=" * 60)
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_ram("  [Pre-load] ")

    # --- Load training data with chronological subsampling ---
    X_train, y_train = load_data_subsampled(
        train_files, keep_idx, max_rows=MAX_TRAIN_ROWS, label="Train")
    check_ram_or_abort(X_train.nbytes / 1e9 + 1.0, "Training QuantileDMatrix")

    mb = params.get('max_bin', 256)
    L(f"\n  Building training QuantileDMatrix (max_bin={mb})...")
    dtrain = xgb.QuantileDMatrix(X_train, label=y_train, max_bin=mb)
    del X_train, y_train
    gc.collect()
    log_ram("  [Post-train-DMatrix] ")

    # --- Load validation data (no subsampling needed, ~10M rows fits) ---
    X_val, y_val = load_data_subsampled(
        val_files, keep_idx, max_rows=MAX_TRAIN_ROWS, label="Val")
    L(f"  Building validation QuantileDMatrix...")
    dval = xgb.QuantileDMatrix(X_val, label=y_val, ref=dtrain, max_bin=mb)
    del X_val, y_val
    gc.collect()
    log_ram("  [Post-val-DMatrix] ")

    # --- Train ---
    L(f"\n  Starting training: {num_boost_round} max rounds...")
    if recovery:
        recovery.set('training_status', 'training_started')
    evals_result = {}

    class TqdmTrainingCallback(xgb.callback.TrainingCallback):
        def __init__(self, total_rounds, update_interval=50):
            self._total = total_rounds
            self._interval = update_interval
            self._pbar = None

        def before_training(self, model):
            self._pbar = tqdm(total=self._total, desc="Training", unit="round")
            return model

        def after_iteration(self, model, epoch, evals_log):
            if self._pbar is not None:
                if (epoch + 1) % self._interval == 0:
                    self._pbar.update(self._interval)
            return False

        def after_training(self, model):
            if self._pbar is not None:
                remainder = self._total - self._pbar.n
                if remainder > 0:
                    self._pbar.update(remainder)
                self._pbar.close()
            return model

    training_callbacks = [TqdmTrainingCallback(num_boost_round, update_interval=50)]

    def _do_train(p, dt, dv):
        return xgb.train(
            p, dt,
            num_boost_round=num_boost_round,
            evals=[(dv, 'val')],
            evals_result=evals_result,
            verbose_eval=50,
            early_stopping_rounds=EARLY_STOP_ROUNDS,
            callbacks=training_callbacks,
        )

    try:
        model = _do_train(params, dtrain, dval)
    except Exception as e:
        error_str = str(e).lower()
        if 'cuda' in error_str or 'gpu' in error_str or 'device' in error_str:
            L(f"\n  [GPU FAIL] {e}")
            L(f"  [FALLBACK] Retrying with CPU...")
            if recovery:
                recovery.add_error(f"GPU training failed: {e}, falling back to CPU")
            params_cpu = params.copy()
            params_cpu['device'] = 'cpu'
            params_cpu['nthread'] = 10
            evals_result = {}
            training_callbacks[:] = [TqdmTrainingCallback(num_boost_round, update_interval=50)]
            model = _do_train(params_cpu, dtrain, dval)
        else:
            L(f"\n  [ERROR] Training failed: {e}", 'error')
            if recovery:
                recovery.add_error(f"Training failed: {e}\n{traceback.format_exc()}")
                recovery.dump_final()
            raise

    # --- Post-training ---
    best_iter = getattr(model, 'best_iteration', model.num_boosted_rounds())
    L(f"\n  Training complete.")
    L(f"  Best iteration : {best_iter}")
    L(f"  Total trees    : {model.num_boosted_rounds()}")
    model_path = str(output_dir / "model_sniper_v4.json")
    try:
        model.save_model(model_path)
    except Exception as e:
        L(f"  [WARN] Model save issue: {e}", 'warning')
    if best_iter < model.num_boosted_rounds():
        L(f"  Note: best_iteration={best_iter} < total={model.num_boosted_rounds()}; "
          f"iteration_range enforced at predict time.")
    L(f"  Model saved to {model_path}")
    if evals_result:
        curve_path = str(output_dir / "training_curve.json")
        with open(curve_path, 'w') as f:
            json.dump(evals_result, f, indent=2, default=float)
        L(f"  Training curve saved to {curve_path}")

    if recovery:
        recovery.set('training_status', 'training_complete')
        recovery.set('best_iteration', best_iter)
        recovery.set('total_trees', model.num_boosted_rounds())
    log_ram("  [Post-training] ")
    del dtrain, dval
    gc.collect()
    return model

# =============================================================================
# TEST EVALUATION
# =============================================================================

def evaluate_on_test(

    model: xgb.Booster,
    test_files: List[str],
    keep_idx: List[int],
) -> Dict:
    L("\n" + "=" * 60)
    L("TEST SET EVALUATION (streaming)")
    L("=" * 60)
    if model is None:
        L("  [ERROR] Model is None.", 'error')
        return {'error': 'model_is_none'}
    best_iter = getattr(model, 'best_iteration', None)
    all_y_true = []
    all_y_pred = []
    L(f"  Streaming {len(test_files)} test files...")
    for X_test, y_test in tqdm(
        stream_chunks_for_eval(test_files, keep_idx),
        total=len(test_files), desc="Test Eval"
    ):
        dtest = xgb.DMatrix(X_test)
        if best_iter is not None:
            preds = model.predict(dtest, iteration_range=(0, best_iter + 1))
        else:
            preds = model.predict(dtest)
        all_y_true.append(y_test)
        all_y_pred.append(preds)
        del X_test, dtest
        gc.collect()
    y_true = np.concatenate(all_y_true)
    y_pred = np.concatenate(all_y_pred)
    L(f"\n  Total test samples: {len(y_true):,}")
    report = traders_report(y_true, y_pred, label="TEST SET")
    del y_true, y_pred, all_y_true, all_y_pred
    gc.collect()
    return report

# =============================================================================
# SAVE ARTIFACTS
# =============================================================================

def save_artifacts(
    model: xgb.Booster,
    keep_idx: List[int],
    best_params: Dict,
    test_report: Dict,
    num_boost_round: int,
    feature_names: List[str],

):
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(exist_ok=True)
    # Model already saved in train_single_model; skip redundant write
    with open(output_dir / "keep_idx.pkl", 'wb') as f:
        pickle.dump(keep_idx, f)
    # Feature importance
    imp = {}
    if model is not None:
        try:
            imp_raw = model.get_score(importance_type='gain')
            for k, v in imp_raw.items():
                feat_idx = int(k.replace('f', ''))
                if feat_idx < len(keep_idx):
                    orig_idx = keep_idx[feat_idx]
                    if orig_idx < len(feature_names):
                        imp[feature_names[orig_idx]] = float(v)
            # Sort by importance
            imp = dict(sorted(imp.items(), key=lambda x: -x[1]))
            L(f"\n  Feature importance (top 10):")
            for i, (name, score) in enumerate(list(imp.items())[:10]):
                L(f"    {i+1:2d}. {name:<30} gain={score:.2f}")
        except Exception:
            pass

    artifacts = {
        'best_params': {k: v for k, v in best_params.items() if not callable(v)},
        'num_boost_round': num_boost_round,
        'test_report': test_report,
        'feature_importance': imp,
        'keep_idx': keep_idx,
        'kept_feature_names': [feature_names[i] for i in keep_idx if i < len(feature_names)],
        'xgboost_version': xgb.__version__,
        'timestamp': datetime.now().isoformat(),
    }

    with open(output_dir / "artifacts.json", 'w') as f:
        json.dump(artifacts, f, indent=2, default=str)
    L(f"\n  Artifacts saved to {output_dir}/")

# =============================================================================
# CRASH RECOVERY RESUME (Fix 5)
# =============================================================================

def try_load_from_recovery(recovery_path: str, gpu_config: Dict):
    """
    Fix 5: If crash_recovery.json has valid Optuna results + keep_idx,
    reconstruct best_params and skip both correlation filter and Optuna.
    Returns (best_params, num_boost_round, keep_idx) or None.
    """
    if not os.path.exists(recovery_path):
        return None

    try:
        with open(recovery_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        L(f"  [Recovery] Failed to load {recovery_path}: {e}", 'warning')
        return None

    # Fix C: Guard against GPU config mismatch (e.g., crashed on GPU, rerunning on CPU)
    saved_gpu = data.get('gpu_config', {})
    # if saved_gpu and saved_gpu != gpu_config:
    #     L(f"  [Recovery] GPU config mismatch — saved: {saved_gpu.get('mode','?')}, "
    #       f"current: {gpu_config.get('mode','?')}. Recomputing.", 'warning')
    #     return None

    best5 = data.get('best_5_optuna', [])
    if not best5:
        L("  [Recovery] No best_5_optuna in recovery file.")
        return None

    # Validate: at least one complete trial with reasonable MAE
    valid = [t for t in best5
             if t.get('status', t.get('mae_bps', 999)) != 999
             and isinstance(t.get('mae_bps', 999), (int, float))
             and t.get('mae_bps', 999) < 999]
    # best5 from the new format stores {'trial', 'mae_bps', 'params'}
    if not valid:
        # Try alternate format check
        valid = [t for t in best5
                 if isinstance(t.get('mae_bps', 999), (int, float))
                 and t['mae_bps'] < 999
                 and 'params' in t]
    if not valid:
        L("  [Recovery] No valid completed trials in recovery file.")
        return None

    keep_idx = data.get('keep_idx', [])
    if not keep_idx:
        L("  [Recovery] keep_idx missing in recovery file — will recompute.", 'warning')
        return None

    # Extract best params from top trial
    top = valid[0]
    params = top['params'].copy()
    num_boost_round = params.pop('num_boost_round', MAX_BOOST_ROUNDS)

    # Merge GPU overrides
    params.update({
        'objective': 'reg:absoluteerror',
        'base_score': 0.0,
        'max_bin': gpu_config['max_bin'],
        'tree_method': gpu_config['tree_method'],
        'device': gpu_config['device'],
        'nthread': 0 if gpu_config['device'] == 'cuda' else 10,
        'verbosity': 0,
    })

    L(f"  [Recovery] Loaded best params from recovery (MAE={top['mae_bps']:.2f} bps)")
    L(f"  [Recovery] keep_idx: {len(keep_idx)} features, num_boost_round: {num_boost_round}")
    return params, num_boost_round, keep_idx

# =============================================================================
# MAIN
# =============================================================================

def main():

    global log, recovery
    start = datetime.now()
    # Create output dir first
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    # Clear stale recovery from collapsed runs
    stale_recovery = Path(OUTPUT_DIR) / "crash_recovery.json"
    if stale_recovery.exists():
        stale_recovery.unlink()
        print("  [Cleanup] Deleted stale crash_recovery.json")
    # Setup logging
    log = setup_logging(OUTPUT_DIR)
    # Setup crash recovery
    recovery = CrashRecovery(OUTPUT_DIR)
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    L("\n" + "=" * 60)
    L("XGBOOST v4 — SINGLE-MODEL + EXTERNAL MEMORY + DUAL GPU")
    L("=" * 60)
    L(f"  Started   : {start}")
    L(f"  XGBoost   : {xgb.__version__}")
    L(f"  Objective : reg:absoluteerror + regime gate")
    L(f"  Optuna    : {OPTUNA_N_JOBS} parallel workers on CPU")
    L(f"  Output    : {OUTPUT_DIR}")
    log_ram("  [Start] ")

    try:
        # --- GPU detection ---
        gpu_config = detect_gpu_config()
        recovery.set('gpu_config', gpu_config)

        # --- Load manifest ---
        manifest = load_manifest()
        feature_names = manifest.get('feature_columns', [])
        L(f"\n  Train files : {len(manifest['train'])}")
        L(f"  Val files   : {len(manifest['val'])}")
        L(f"  Test files  : {len(manifest['test'])}")
        L(f"  Features    : {len(feature_names)}: {feature_names}")
        # --- Fix 5: Try resuming from crash_recovery.json ---
        recovered = try_load_from_recovery(
            os.path.join(OUTPUT_DIR, "crash_recovery.json"), gpu_config
        )

        if recovered is not None:
            best_params, num_boost_round, keep_idx = recovered
            L("  Resuming from recovery: skipping correlation filter + Optuna")
        else:
            # --- Step 1: Correlation filter ---
            keep_idx = compute_correlation_filter(
                manifest['train'], feature_names
            )
            recovery.set('keep_idx', keep_idx)
            kept_names = [feature_names[i] for i in keep_idx]
            L(f"  Kept features: {kept_names}")
            # --- Step 2: Optuna ---
            best_params, num_boost_round = run_optuna(
                manifest['train'], keep_idx, gpu_config
            )
        L(f"\n  Training with: {num_boost_round} rounds")
        L(f"  Params: {json.dumps({k:v for k,v in best_params.items() if k not in ('tree_method','device','nthread','verbosity','base_score','max_bin','disable_default_eval_metric','objective')}, indent=4, default=str)}")
        # --- Step 3: Train ---
        model = train_single_model(
            manifest['train'], manifest['val'], keep_idx,
            best_params, num_boost_round, gpu_config
        )
        # --- Step 4: Test ---
        test_report = evaluate_on_test(model, manifest['test'], keep_idx)
        # --- Step 5: Save ---
        save_artifacts(model, keep_idx, best_params, test_report,
                       num_boost_round, feature_names)
        recovery.set('training_status', 'complete')
        recovery.set('test_report', test_report)
        recovery.dump_final()
        duration = (datetime.now() - start).total_seconds()
        L(f"\n{'='*60}")
        L(f"COMPLETE — {duration/60:.1f} min ({duration/3600:.1f} hr)")
        L(f"{'='*60}")
        L(f"  Log file     : {LOG_FILE}")
        L(f"  Recovery file: {recovery.filepath}")
    except SystemExit:
        raise
    except Exception as e:
        L(f"\n[FATAL ERROR] {e}", 'critical')
        L(traceback.format_exc(), 'critical')
        if recovery:
            recovery.add_error(f"Fatal: {e}\n{traceback.format_exc()}")
            recovery.dump_final()
        raise
if __name__ == "__main__":
    main()
