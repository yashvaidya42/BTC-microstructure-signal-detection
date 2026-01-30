import numpy as np
import lightgbm as lgb
import json
from pathlib import Path
from tqdm import tqdm
import time
from datetime import datetime
import gc
from sklearn.metrics import classification_report, confusion_matrix
from collections import Counter

# =============================================================================
# CONFIGURATION
# =============================================================================

MANIFEST_FILE = "./dataset_lgbm/manifest.json"
MODEL_FILE = "./lgbm_model_profit_optimized.txt"

# Cache paths
TRAIN_BINARY_CACHE = "./dataset_lgbm/train_data.bin"
VAL_BINARY_CACHE = "./dataset_lgbm/val_data.bin"
CLASS_WEIGHTS_CACHE = "./dataset_lgbm/class_weights.json"

# LightGBM parameters - RAM optimized
LGBM_PARAMS = {
    'objective': 'multiclass',
    'num_class': 3,
    'metric': 'multi_logloss',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'max_depth': 6,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1,
    'num_threads': 12,
    'is_unbalance': False,
    'boost_from_average': False,
    'max_bin': 127,                    # OPT-5: Reduced bins
    'bin_construct_sample_cnt': 1000000,  # OPT-6: Sample 1M rows for bin boundaries (KEY FIX)
    'force_row_wise': True,            # Better for low RAM systems
}

# Training configuration
NUM_BOOST_ROUND = 500
EARLY_STOPPING_ROUNDS = 50

# Feature names
FEATURE_COLUMNS = [
    'qty', 'time_delta', 'log_return',
    'trade_intensity', 'price_acceleration', 
    'volume_imbalance', 'rolling_signed_volume', 'side',
    'price_volatility', 'volume_volatility', 'time_delta_std',
    'price_distance_from_ma'
]


# =============================================================================
# CLASS WEIGHTS (Labels only - minimal RAM)
# =============================================================================

def compute_class_weights(file_paths):
    """
    Compute class weights by scanning labels only.
    Memory: ~1 byte per sample for counting (not stored)
    """
    print("Computing class weights (labels only, streaming)...")
    
    global_counts = np.zeros(3, dtype=np.int64)
    total_samples = 0
    
    for fpath in tqdm(file_paths, desc="Counting"):
        data = np.load(fpath)
        y = data['y']
        for cls in range(3):
            global_counts[cls] += (y == cls).sum()
        total_samples += len(y)
        del data, y
    
    n_classes = 3
    class_weights = total_samples / (n_classes * global_counts.astype(np.float64))
    
    print(f"\nClass Distribution ({total_samples:,} samples):")
    for i in range(3):
        pct = (global_counts[i] / total_samples) * 100
        class_name = ['Down', 'Neutral', 'Up'][i]
        boost = class_weights[i] / class_weights[1]
        print(f"  {class_name:>7} ({i}): {global_counts[i]:>12,} ({pct:>5.2f}%) | weight: {class_weights[i]:.3f} ({boost:.1f}x)")
    
    return class_weights, global_counts, total_samples


def save_class_weights(class_weights, global_counts, total_samples, filepath):
    """Save class weights to JSON."""
    data = {
        'class_weights': class_weights.tolist(),
        'global_counts': global_counts.tolist(),
        'total_samples': int(total_samples)
    }
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"[CACHE] Saved: {filepath}")


def load_class_weights(filepath):
    """Load class weights from cache."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    print(f"[CACHE] Loaded: {filepath}")
    return np.array(data['class_weights'])


# =============================================================================
# TRUE STREAMING DATASET BUILDER (RAM-SAFE)
# =============================================================================

class StreamingSequence(lgb.Sequence):
    """
    True streaming sequence - loads ONE chunk at a time.
    LightGBM calls __getitem__ during construction/training.
    """
    def __init__(self, file_paths, class_weights):
        self.file_paths = file_paths
        self.class_weights = class_weights
        self._n_files = len(file_paths)
        
        # Pre-compute chunk sizes (small scan - only metadata)
        self._chunk_sizes = []
        for fpath in file_paths:
            data = np.load(fpath)
            self._chunk_sizes.append(len(data['y']))
            del data
        
        self._total_rows = sum(self._chunk_sizes)
    
    def __len__(self):
        return self._n_files
    
    def __getitem__(self, idx):
        """Load single chunk - called by LightGBM during iteration."""
        data = np.load(self.file_paths[idx])
        X = data['X'].astype(np.float32)
        return X
    
    @property
    def total_rows(self):
        return self._total_rows


def build_dataset_ram_safe(file_paths, class_weights, feature_names, reference=None, desc="Building"):
    """
    Build LightGBM dataset with TRUE streaming.
    
    Key insight: LightGBM needs labels/weights upfront, but with bin_construct_sample_cnt,
    it only samples a subset of features for bin boundary calculation.
    
    Memory strategy:
    - Labels: Load all (small - 1 byte per sample)
    - Weights: Compute from labels (small - 4 bytes per sample)  
    - Features: Stream via Sequence (only 1 chunk in RAM during construct)
    
    With 500M samples:
    - Labels: ~500MB
    - Weights: ~2GB
    - Features: ~120MB per chunk (NOT all 23GB)
    
    Total peak: ~3GB (fits in 8GB)
    """
    print(f"\n{'='*60}")
    print(f"{desc}: {len(file_paths)} files (RAM-safe streaming)")
    print(f"{'='*60}")
    
    # Step 1: Load labels (small)
    print("Step 1/3: Loading labels...")
    y_list = []
    for fpath in tqdm(file_paths, desc="Labels"):
        data = np.load(fpath)
        y_list.append(data['y'].astype(np.int8))  # Force int8 to save RAM
        del data
    
    y_all = np.concatenate(y_list)
    del y_list
    gc.collect()
    
    print(f"  Labels: {len(y_all):,} samples ({y_all.nbytes/1e6:.1f} MB)")
    
    # Step 2: Compute weights from labels (no extra file reads)
    print("Step 2/3: Computing sample weights...")
    weights_all = class_weights[y_all].astype(np.float32)
    print(f"  Weights: {weights_all.nbytes/1e6:.1f} MB")
    
    # Step 3: Create streaming sequence for features
    print("Step 3/3: Creating feature stream...")
    feature_stream = StreamingSequence(file_paths, class_weights)
    print(f"  Feature stream ready: {feature_stream.total_rows:,} rows")
    
    # Create dataset with streaming features
    print("\nCreating LightGBM Dataset...")
    print(f"  bin_construct_sample_cnt: {LGBM_PARAMS['bin_construct_sample_cnt']:,}")
    print(f"  (Only samples {LGBM_PARAMS['bin_construct_sample_cnt']/feature_stream.total_rows*100:.2f}% of data for binning)")
    
    dataset = lgb.Dataset(
        feature_stream,
        label=y_all,
        weight=weights_all,
        feature_name=feature_names,
        free_raw_data=True,
        params={
            'max_bin': LGBM_PARAMS['max_bin'],
            'bin_construct_sample_cnt': LGBM_PARAMS['bin_construct_sample_cnt'],
        }
    )
    
    if reference is not None:
        dataset.set_reference(reference)
    
    return dataset, y_all, weights_all


# =============================================================================
# OPTIMIZED TRAINING
# =============================================================================

def train_optimized(train_files, val_files):
    """
    Train with all optimizations:
    - OPT-1: Single-pass label scanning
    - OPT-2: Binary cache
    - OPT-3: Early stopping
    - OPT-4: free_raw_data=True
    - OPT-5: max_bin=127
    - OPT-6: bin_construct_sample_cnt=1M (KEY - enables true streaming)
    - OPT-7: force_row_wise=True (better for low RAM)
    """
    print("=" * 80)
    print("TRAINING PIPELINE (RAM-SAFE)")
    print("=" * 80)
    print(f"Train files: {len(train_files)}")
    print(f"Val files:   {len(val_files)}")
    print(f"Max rounds:  {NUM_BOOST_ROUND}")
    print(f"Early stop:  {EARLY_STOPPING_ROUNDS} rounds")
    print(f"Threads:     {LGBM_PARAMS['num_threads']}")
    print(f"Max bins:    {LGBM_PARAMS['max_bin']}")
    print(f"Bin sample:  {LGBM_PARAMS['bin_construct_sample_cnt']:,} rows")
    print("=" * 80 + "\n")
    
    # Check caches
    train_cache_exists = Path(TRAIN_BINARY_CACHE).exists()
    val_cache_exists = Path(VAL_BINARY_CACHE).exists()
    weights_cache_exists = Path(CLASS_WEIGHTS_CACHE).exists()
    
    # Load or compute class weights
    if weights_cache_exists:
        class_weights = load_class_weights(CLASS_WEIGHTS_CACHE)
    else:
        print("=" * 60)
        print("COMPUTING CLASS WEIGHTS (first run only)")
        print("=" * 60)
        class_weights, global_counts, total_samples = compute_class_weights(train_files)
        save_class_weights(class_weights, global_counts, total_samples, CLASS_WEIGHTS_CACHE)
        print()
    
    # Load from cache OR build datasets
    if train_cache_exists and val_cache_exists:
        print("[CACHE] Loading pre-built datasets...")
        print(f"  Train: {TRAIN_BINARY_CACHE}")
        print(f"  Val:   {VAL_BINARY_CACHE}")
        
        train_data = lgb.Dataset(TRAIN_BINARY_CACHE)
        val_data = lgb.Dataset(VAL_BINARY_CACHE, reference=train_data)
        
        print("[CACHE] Loaded (skipped construction)\n")
        
    else:
        print("[BUILD] No cache found - building datasets...\n")
        
        # Build train dataset (RAM-safe)
        train_data, y_train, w_train = build_dataset_ram_safe(
            train_files, class_weights, FEATURE_COLUMNS, 
            reference=None, desc="TRAIN DATASET"
        )
        
        # Construct (triggers histogram building with sampling)
        print("\nConstructing train histograms...")
        print("(Using sampled bin boundaries - NOT loading all features)")
        construct_start = time.time()
        train_data.construct()
        construct_time = time.time() - construct_start
        print(f"Construction time: {construct_time:.1f}s")
        
        # Save cache
        print(f"\n[CACHE] Saving: {TRAIN_BINARY_CACHE}")
        train_data.save_binary(TRAIN_BINARY_CACHE)
        
        # Cleanup
        del y_train, w_train
        gc.collect()
        
        # Build val dataset
        val_data, y_val, w_val = build_dataset_ram_safe(
            val_files, class_weights, FEATURE_COLUMNS,
            reference=train_data, desc="VAL DATASET"
        )
        
        print("\nConstructing val histograms...")
        val_data.construct()
        
        print(f"\n[CACHE] Saving: {VAL_BINARY_CACHE}")
        val_data.save_binary(VAL_BINARY_CACHE)
        
        del y_val, w_val
        gc.collect()
        
        print("\n[CACHE] Saved - next run will be ~10x faster\n")
    
    # Training
    print("=" * 80)
    print(f"TRAINING (max {NUM_BOOST_ROUND} rounds)")
    print("=" * 80)
    
    start_time = time.time()
    
    model = lgb.train(
        LGBM_PARAMS,
        train_data,
        num_boost_round=NUM_BOOST_ROUND,
        valid_sets=[train_data, val_data],
        valid_names=['train', 'val'],
        callbacks=[
            lgb.early_stopping(stopping_rounds=EARLY_STOPPING_ROUNDS),
            lgb.log_evaluation(period=25)
        ]
    )
    
    train_time = time.time() - start_time
    
    print("\n" + "=" * 80)
    print(f"Training complete: {train_time/60:.1f} minutes")
    print(f"Best iteration: {model.best_iteration}")
    print(f"Trees: {model.num_trees()}")
    if model.best_iteration < NUM_BOOST_ROUND:
        saved = NUM_BOOST_ROUND - model.best_iteration
        print(f"[OPT-3] Early stopping saved {saved} rounds")
    print("=" * 80 + "\n")
    
    gc.collect()
    return model, train_time, class_weights


# =============================================================================
# EVALUATION (Chunked - RAM safe)
# =============================================================================

def evaluate_profit_focused(model, file_paths, dataset_name, class_weights):
    """Evaluate in chunks to avoid RAM overflow."""
    print("=" * 80)
    print(f"{dataset_name.upper()} EVALUATION")
    print("=" * 80)
    
    y_true_list = []
    y_pred_list = []
    
    for fpath in tqdm(file_paths, desc="Evaluating"):
        data = np.load(fpath)
        X = data['X']
        y = data['y']
        
        pred_proba = model.predict(X)
        pred = np.argmax(pred_proba, axis=1)
        
        y_true_list.append(y)
        y_pred_list.append(pred)
        
        del X, y, pred_proba, pred, data
        gc.collect()
    
    y_true = np.concatenate(y_true_list)
    y_pred = np.concatenate(y_pred_list)
    del y_true_list, y_pred_list
    
    # Metrics
    accuracy = np.mean(y_true == y_pred)
    print(f"\nOverall Accuracy: {accuracy*100:.2f}%")
    
    # Prediction distribution
    print(f"\nPrediction Distribution:")
    pred_counts = np.bincount(y_pred, minlength=3)
    for i in range(3):
        pct = pred_counts[i] / len(y_pred) * 100
        name = ['Down', 'Neutral', 'Up'][i]
        print(f"  {name:>7}: {pred_counts[i]:>12,} ({pct:>5.2f}%)")
    
    # Per-class accuracy
    print(f"\nPer-Class Accuracy:")
    for cls in range(3):
        mask = y_true == cls
        if mask.sum() > 0:
            acc = np.mean(y_pred[mask] == cls)
            name = ['Down', 'Neutral', 'Up'][cls]
            print(f"  {name:>7}: {acc*100:>5.2f}%")
    
    # Confusion matrix
    print(f"\nConfusion Matrix:")
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    print("           Down      Neutral       Up")
    for i, name in enumerate(['Down', 'Neutral', 'Up']):
        row = f"{name:>7}  "
        for j in range(3):
            pct = cm[i, j] / cm[i].sum() * 100 if cm[i].sum() > 0 else 0
            row += f"{cm[i,j]:>8,} ({pct:>4.1f}%)  "
        print(row)
    
    # Classification report
    print("\n" + "=" * 80)
    print("SIGNAL METRICS")
    print("=" * 80)
    print(classification_report(y_true, y_pred, target_names=['Down', 'Neutral', 'Up'], digits=4))
    
    # Profit metrics
    print("=" * 80)
    print("PROFIT METRICS")
    print("=" * 80)
    
    # Down
    down_tp = np.sum((y_true == 0) & (y_pred == 0))
    down_fp = np.sum((y_true != 0) & (y_pred == 0))
    down_fn = np.sum((y_true == 0) & (y_pred != 0))
    down_prec = down_tp / (down_tp + down_fp) if (down_tp + down_fp) > 0 else 0
    down_rec = down_tp / (down_tp + down_fn) if (down_tp + down_fn) > 0 else 0
    down_f1 = 2 * down_prec * down_rec / (down_prec + down_rec) if (down_prec + down_rec) > 0 else 0
    
    print(f"\nDOWN Signal:")
    print(f"  Precision: {down_prec*100:>5.2f}%")
    print(f"  Recall:    {down_rec*100:>5.2f}%")
    print(f"  F1:        {down_f1:.4f}")
    
    # Up
    up_tp = np.sum((y_true == 2) & (y_pred == 2))
    up_fp = np.sum((y_true != 2) & (y_pred == 2))
    up_fn = np.sum((y_true == 2) & (y_pred != 2))
    up_prec = up_tp / (up_tp + up_fp) if (up_tp + up_fp) > 0 else 0
    up_rec = up_tp / (up_tp + up_fn) if (up_tp + up_fn) > 0 else 0
    up_f1 = 2 * up_prec * up_rec / (up_prec + up_rec) if (up_prec + up_rec) > 0 else 0
    
    print(f"\nUP Signal:")
    print(f"  Precision: {up_prec*100:>5.2f}%")
    print(f"  Recall:    {up_rec*100:>5.2f}%")
    print(f"  F1:        {up_f1:.4f}")
    
    # Combined
    avg_prec = (down_prec + up_prec) / 2
    avg_rec = (down_rec + up_rec) / 2
    avg_f1 = (down_f1 + up_f1) / 2
    
    print(f"\nCombined (Up+Down):")
    print(f"  Avg Precision: {avg_prec*100:>5.2f}%")
    print(f"  Avg Recall:    {avg_rec*100:>5.2f}%")
    print(f"  Avg F1:        {avg_f1:.4f}")
    print("=" * 80 + "\n")
    
    del y_true, y_pred
    gc.collect()
    
    return {
        'accuracy': accuracy,
        'down_precision': down_prec, 'down_recall': down_rec, 'down_f1': down_f1,
        'up_precision': up_prec, 'up_recall': up_rec, 'up_f1': up_f1,
        'avg_signal_precision': avg_prec, 'avg_signal_recall': avg_rec, 'avg_signal_f1': avg_f1
    }


# =============================================================================
# FEATURE IMPORTANCE
# =============================================================================

def analyze_feature_importance(model):
    """Feature importance analysis."""
    print("=" * 80)
    print("FEATURE IMPORTANCE")
    print("=" * 80)
    
    importance = model.feature_importance(importance_type='gain')
    pairs = sorted(zip(FEATURE_COLUMNS, importance), key=lambda x: x[1], reverse=True)
    total = sum(importance)
    
    print("\nRanking (by gain):")
    for i, (feat, imp) in enumerate(pairs, 1):
        pct = imp / total * 100 if total > 0 else 0
        print(f"{i:>2}. {feat:<30} {imp:>12,.0f} ({pct:>5.2f}%)")
    print("=" * 80 + "\n")


# =============================================================================
# MAIN
# =============================================================================

def main():
    start_time = datetime.now()
    
    print("\n" + "=" * 80)
    print("PROFIT-OPTIMIZED TRAINING (RAM-SAFE v3)")
    print("=" * 80)
    print(f"Start: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nOptimizations:")
    print(f"  OPT-1: Single-pass label scanning")
    print(f"  OPT-2: Binary cache")
    print(f"  OPT-3: Early stopping")
    print(f"  OPT-4: free_raw_data=True")
    print(f"  OPT-5: max_bin=127")
    print(f"  OPT-6: bin_construct_sample_cnt=1M (KEY)")
    print(f"  OPT-7: force_row_wise=True")
    print("=" * 80 + "\n")
    
    # Load manifest
    with open(MANIFEST_FILE) as f:
        manifest = json.load(f)
    
    train_files = manifest['train']
    val_files = manifest['val']
    test_files = manifest['test']
    
    print(f"Dataset: Train={len(train_files)}, Val={len(val_files)}, Test={len(test_files)}\n")
    
    # Train
    model, train_time, class_weights = train_optimized(train_files, val_files)
    
    # Save
    model.save_model(MODEL_FILE)
    print(f"[OK] Model saved: {MODEL_FILE}\n")
    
    # Analysis
    analyze_feature_importance(model)
    
    # Evaluate
    val_metrics = evaluate_profit_focused(model, val_files, "Validation", class_weights)
    test_metrics = evaluate_profit_focused(model, test_files, "Test", class_weights)
    
    # Summary
    end_time = datetime.now()
    total_time = (end_time - start_time).total_seconds()
    
    print("=" * 80)
    print("COMPLETE")
    print("=" * 80)
    print(f"\nTime: {total_time/60:.1f} min total, {train_time/60:.1f} min training")
    
    print(f"\nValidation: Acc={val_metrics['accuracy']*100:.2f}%, F1={val_metrics['avg_signal_f1']:.4f}")
    print(f"Test:       Acc={test_metrics['accuracy']*100:.2f}%, F1={test_metrics['avg_signal_f1']:.4f}")
    
    # Trading readiness
    print("\n" + "-" * 40)
    print("Trading Readiness:")
    if test_metrics['avg_signal_precision'] > 0.55:
        print("  [OK] Precision >55%")
    else:
        print("  [WARN] Precision <55%")
    
    if test_metrics['avg_signal_recall'] > 0.30:
        print("  [OK] Recall >30%")
    else:
        print("  [WARN] Recall <30%")
    
    if test_metrics['avg_signal_f1'] > 0.40:
        print("  [OK] F1 >0.40")
    else:
        print("  [WARN] F1 <0.40")
    
    # Cache status
    print("\n" + "-" * 40)
    print("Cache Status:")
    for name, path in [("Weights", CLASS_WEIGHTS_CACHE), ("Train", TRAIN_BINARY_CACHE), ("Val", VAL_BINARY_CACHE)]:
        status = "[OK]" if Path(path).exists() else "[MISSING]"
        print(f"  {name}: {status}")
    
    print("=" * 80 + "\n")
    
    return model, val_metrics, test_metrics


if __name__ == "__main__":
    model, val_metrics, test_metrics = main()