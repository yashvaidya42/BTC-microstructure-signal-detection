#!/usr/bin/env python3
"""
MEMORY-EFFICIENT LIGHTGBM TRAINING
==================================
Optimized for 8GB RAM systems.

Key optimizations:
1. Streaming data loading (chunk by chunk)
2. Uses LightGBM's native Dataset binning
3. Reduced Optuna trials with smart early pruning
4. Single-pass validation

IMPROVEMENTS:
- Stratified sampling / oversampling for UP/DOWN classes
- Configurable class weights (fixed or sqrt inverse freq)
- Per-class recall metrics during training
- Confusion matrix printed every 5 Optuna trials

Run this instead of train_lgbm.py if memory is constrained.
"""

import numpy as np
import lightgbm as lgb
import optuna
from optuna.samplers import TPESampler
import pickle
import json
from pathlib import Path
from typing import Tuple, List, Dict, Optional, Generator
from sklearn.metrics import f1_score, confusion_matrix, recall_score, precision_score
import warnings
from tqdm import tqdm
import gc
from datetime import datetime

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Check LightGBM version for API compatibility
from packaging import version
LGBM_VERSION = version.parse(lgb.__version__)
IS_LGBM_4X = LGBM_VERSION >= version.parse("4.0.0")
print(f"LightGBM version: {lgb.__version__} (4.x API: {IS_LGBM_4X})")


# =============================================================================
# CONFIGURATION
# =============================================================================

DATASET_DIR = "./dataset_lgbm"
SCALER_FILE = "./scaler_lgbm.pkl"
OUTPUT_DIR = "./model_output"

NUM_CLASSES = 3
LOOK_AHEAD = 1400  # Will be updated if preprocessing changes

# Memory optimization - Tuned for 8GB RAM
CHUNKS_FOR_OPTUNA = 10
VAL_CHUNKS = 15
MAX_SAMPLES_OPTUNA = 10_000_000

# Optuna
N_OPTUNA_TRIALS = 50
OPTUNA_TIMEOUT = 7200  # 2 hours

# Training
EARLY_STOPPING = 100

# Feature indices to log-transform
LOG_TRANSFORM_IDX = [3]  # trade_intensity

# =============================================================================
# CLASS WEIGHTS CONFIGURATION
# =============================================================================
# Options:
#   'fixed'      -> Use FIXED_CLASS_WEIGHTS directly
#   'sqrt'       -> Use sqrt of inverse frequency
#   'linear'     -> Use linear inverse frequency (original)

CLASS_WEIGHT_MODE = 'linear'  # Change this to experiment
FIXED_CLASS_WEIGHTS = np.array([1, 1, 1])  # [DOWN, NEUTRAL, UP]


# =============================================================================
# LIGHTGBM COMPATIBILITY (handles 3.x vs 4.x API)
# =============================================================================

def lgb_train_compat(
    params: Dict,
    train_set: lgb.Dataset,
    num_boost_round: int = 100,
    valid_sets=None,
    valid_names=None,
    fobj=None,
    feval=None,
    callbacks=None,
):
    """Train LightGBM with API compatibility for both 3.x and 4.x."""
    if callbacks is None:
        callbacks = []
    
    if IS_LGBM_4X:
        # LightGBM 4.x: custom objective via params['objective']
        if fobj is not None:
            params = params.copy()
            params['objective'] = fobj
        
        return lgb.train(
            params,
            train_set,
            num_boost_round=num_boost_round,
            valid_sets=valid_sets,
            valid_names=valid_names,
            feval=feval,
            callbacks=callbacks,
        )
    else:
        # LightGBM 3.x: custom objective via fobj parameter
        return lgb.train(
            params,
            train_set,
            num_boost_round=num_boost_round,
            valid_sets=valid_sets,
            valid_names=valid_names,
            fobj=fobj,
            feval=feval,
            callbacks=callbacks,
        )


# =============================================================================
# STREAMING DATA LOADING
# =============================================================================

def load_manifest() -> Dict:
    """Load dataset manifest."""
    with open(Path(DATASET_DIR) / "manifest.json", 'r') as f:
        return json.load(f)


def stream_chunks(file_list: List[str], max_chunks: Optional[int] = None) -> Generator:
    """
    Stream chunks from disk one at a time.
    
    Yields:
        (X, y) tuples for each chunk
    """
    files = file_list[:max_chunks] if max_chunks else file_list
    
    for f in files:
        data = np.load(f)
        X = data['X'].astype(np.float32)
        y = data['y'].astype(np.int32)
        
        # Apply log transform to trade_intensity
        for idx in LOG_TRANSFORM_IDX:
            col = X[:, idx]
            col_min = col.min()
            X[:, idx] = np.log(col - col_min + 1.0)
        
        yield X, y
        
        del data, X, y
        gc.collect()


def load_chunks_concat(file_list: List[str], max_chunks: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Load and concatenate chunks."""
    X_list = []
    y_list = []
    
    for X, y in stream_chunks(file_list, max_chunks):
        X_list.append(X)
        y_list.append(y)
    
    return np.vstack(X_list), np.concatenate(y_list)


# =============================================================================
# CLASS WEIGHTS COMPUTATION
# =============================================================================

def compute_class_weights(y: np.ndarray, mode: str = CLASS_WEIGHT_MODE) -> np.ndarray:
    """
    Compute class weights based on mode.
    
    Args:
        y: Label array
        mode: 'fixed', 'sqrt', or 'linear'
    
    Returns:
        Class weights array of shape (NUM_CLASSES,)
    """
    if mode == 'fixed':
        print(f"  Using FIXED class weights: {FIXED_CLASS_WEIGHTS}")
        return FIXED_CLASS_WEIGHTS.copy()
    
    counts = np.bincount(y.astype(int), minlength=NUM_CLASSES)
    
    if mode == 'sqrt':
        # Square root of inverse frequency - less aggressive
        weights = np.sqrt(len(y) / (NUM_CLASSES * counts + 1e-8))
        weights = weights / weights.sum() * NUM_CLASSES
        print(f"  Using SQRT inverse freq weights: {weights}")
        return weights.astype(np.float32)
    
    else:  # 'linear'
        weights = len(y) / (NUM_CLASSES * counts + 1e-8)
        weights = weights / weights.sum() * NUM_CLASSES
        print(f"  Using LINEAR inverse freq weights: {weights}")
        return weights.astype(np.float32)


# =============================================================================
# STRATIFIED SAMPLING / OVERSAMPLING
# =============================================================================

def stratified_oversample(
    X: np.ndarray, 
    y: np.ndarray, 
    target_ratio: Dict[int, float] = None,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Oversample minority classes (UP/DOWN) to improve balance.
    
    Args:
        X: Features
        y: Labels
        target_ratio: Target ratio for each class {0: 0.25, 1: 0.50, 2: 0.25}
                     If None, balances to match largest minority class
        random_state: Random seed
    
    Returns:
        Oversampled X, y
    """
    np.random.seed(random_state)
    
    counts = np.bincount(y.astype(int), minlength=NUM_CLASSES)
    
    if target_ratio is None:
        # Target: make UP and DOWN equal to 30% of NEUTRAL count each
        neutral_count = counts[1]
        target_minority = int(neutral_count * 0.8)
        target_counts = {
            0: max(counts[0], target_minority),  # DOWN
            1: counts[1],                         # NEUTRAL (unchanged)
            2: max(counts[2], target_minority),  # UP
        }
    else:
        total = len(y)
        target_counts = {c: int(total * r) for c, r in target_ratio.items()}
    
    # Collect indices for each class
    indices_by_class = {c: np.where(y == c)[0] for c in range(NUM_CLASSES)}
    
    # Oversample
    new_indices = []
    for c in range(NUM_CLASSES):
        class_indices = indices_by_class[c]
        current_count = len(class_indices)
        target_count = target_counts[c]
        
        if target_count > current_count:
            # Oversample with replacement
            extra_needed = target_count - current_count
            extra_indices = np.random.choice(class_indices, size=extra_needed, replace=True)
            new_indices.extend(class_indices.tolist())
            new_indices.extend(extra_indices.tolist())
        else:
            new_indices.extend(class_indices.tolist())
    
    # Shuffle
    np.random.shuffle(new_indices)
    new_indices = np.array(new_indices)
    
    X_oversampled = X[new_indices]
    y_oversampled = y[new_indices]
    
    # Print new distribution
    new_counts = np.bincount(y_oversampled.astype(int), minlength=NUM_CLASSES)
    print(f"  After oversampling: DOWN={new_counts[0]:,}, NEUTRAL={new_counts[1]:,}, UP={new_counts[2]:,}")
    
    return X_oversampled, y_oversampled


# =============================================================================
# SIMPLIFIED FOCAL LOSS (More stable)
# =============================================================================

def softmax(x: np.ndarray) -> np.ndarray:
    """Compute softmax along last axis."""
    exp_x = np.exp(x - x.max(axis=-1, keepdims=True))
    return exp_x / exp_x.sum(axis=-1, keepdims=True)


def focal_loss_objective(gamma: float, alpha: np.ndarray):
    """
    Create focal loss objective function.
    
    Simplified implementation that's more numerically stable.
    """
    def objective(y_pred: np.ndarray, dtrain: lgb.Dataset):
        y_true = dtrain.get_label().astype(int)
        n = len(y_true)
        
        # Reshape: (n * num_class,) -> (n, num_class)
        y_pred = y_pred.reshape((n, NUM_CLASSES), order='F')
        
        # Softmax probabilities
        p = softmax(y_pred)
        p = np.clip(p, 1e-15, 1 - 1e-15)
        
        # One-hot encode
        y_onehot = np.zeros_like(p)
        y_onehot[np.arange(n), y_true] = 1
        
        # p_t = probability of true class
        p_t = (p * y_onehot).sum(axis=1, keepdims=True)
        
        # Focal weight
        focal_w = (1 - p_t) ** gamma
        
        # Alpha weights per sample
        alpha_t = alpha[y_true].reshape(-1, 1)
        
        # Gradient: similar to cross-entropy but weighted
        grad = alpha_t * focal_w * (p - y_onehot)
        
        # Hessian (diagonal approximation)
        hess = alpha_t * focal_w * p * (1 - p) + 1e-8
        
        return grad.flatten('F'), hess.flatten('F')
    
    return objective


def focal_loss_metric(gamma: float, alpha: np.ndarray):
    """Create focal loss evaluation metric."""
    def metric(y_pred: np.ndarray, dtrain: lgb.Dataset):
        y_true = dtrain.get_label().astype(int)
        n = len(y_true)
        
        y_pred = y_pred.reshape((n, NUM_CLASSES), order='F')
        p = softmax(y_pred)
        p = np.clip(p, 1e-15, 1 - 1e-15)
        
        p_t = p[np.arange(n), y_true]
        focal_w = (1 - p_t) ** gamma
        alpha_t = alpha[y_true]
        
        loss = -alpha_t * focal_w * np.log(p_t)
        
        return 'focal_loss', float(loss.mean()), False
    
    return metric


# =============================================================================
# PER-CLASS RECALL METRIC
# =============================================================================

def per_class_recall_metric():
    """
    Create evaluation metric that returns per-class recall.
    
    This helps monitor if the model is learning to predict UP/DOWN.
    """
    def metric(y_pred: np.ndarray, dtrain: lgb.Dataset):
        y_true = dtrain.get_label().astype(int)
        n = len(y_true)
        
        y_pred = y_pred.reshape((n, NUM_CLASSES), order='F')
        y_pred_class = np.argmax(y_pred, axis=1)
        
        # Compute per-class recall
        recalls = recall_score(y_true, y_pred_class, average=None, zero_division=0)
        
        # Return average of UP and DOWN recall (what we care about)
        up_down_recall = (recalls[0] + recalls[2]) / 2
        
        return 'up_down_recall', float(up_down_recall), True  # Higher is better
    
    return metric


def compute_detailed_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    """Compute detailed metrics including per-class recall, precision, F1."""
    recalls = recall_score(y_true, y_pred, average=None, zero_division=0)
    precisions = precision_score(y_true, y_pred, average=None, zero_division=0)
    f1s = f1_score(y_true, y_pred, average=None, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    
    return {
        'recall': {'DOWN': recalls[0], 'NEUTRAL': recalls[1], 'UP': recalls[2]},
        'precision': {'DOWN': precisions[0], 'NEUTRAL': precisions[1], 'UP': precisions[2]},
        'f1': {'DOWN': f1s[0], 'NEUTRAL': f1s[1], 'UP': f1s[2]},
        'confusion_matrix': cm,
        'macro_f1': f1_score(y_true, y_pred, average='macro'),
        'up_down_recall_avg': (recalls[0] + recalls[2]) / 2,
    }


def print_confusion_matrix(cm: np.ndarray, title: str = "Confusion Matrix"):
    """Pretty print confusion matrix."""
    print(f"\n{title}")
    print("              Predicted")
    print("             DOWN   NEUT     UP")
    labels = ['DOWN', 'NEUT', '  UP']
    for i, name in enumerate(labels):
        print(f"Actual {name}: {cm[i][0]:6d} {cm[i][1]:6d} {cm[i][2]:6d}")


# =============================================================================
# PURGED VALIDATION SPLIT
# =============================================================================

def purged_train_val_split(
    X: np.ndarray, 
    y: np.ndarray, 
    val_ratio: float = 0.2,
    purge_gap: int = LOOK_AHEAD
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split data with purge gap between train and validation.
    """
    n = len(X)
    val_size = int(n * val_ratio)
    train_end = n - val_size - purge_gap
    val_start = n - val_size
    
    X_train = X[:train_end]
    y_train = y[:train_end]
    X_val = X[val_start:]
    y_val = y[val_start:]
    
    return X_train, y_train, X_val, y_val


# =============================================================================
# OPTUNA OPTIMIZATION
# =============================================================================

# Global storage for tracking trials (for confusion matrix printing)
_optuna_trial_results = []


def run_optuna(
    train_files: List[str],
    n_trials: int = N_OPTUNA_TRIALS,
    timeout: int = OPTUNA_TIMEOUT
) -> Dict:
    """
    Run Optuna hyperparameter optimization.
    
    Features:
    - Stratified oversampling of UP/DOWN
    - Per-class recall tracking
    - Confusion matrix every 5 trials
    """
    global _optuna_trial_results
    _optuna_trial_results = []
    
    print("=" * 60)
    print("HYPERPARAMETER OPTIMIZATION")
    print("=" * 60)
    
    # Load subset for tuning
    X, y = load_chunks_concat(train_files, max_chunks=CHUNKS_FOR_OPTUNA)
    
    # Limit if too large
    if len(y) > MAX_SAMPLES_OPTUNA:
        print(f"Limiting from {len(y):,} to {MAX_SAMPLES_OPTUNA:,} samples")
        X = X[:MAX_SAMPLES_OPTUNA]
        y = y[:MAX_SAMPLES_OPTUNA]
    
    print(f"Loaded {len(y):,} samples for optimization")
    
    # Print initial class distribution
    counts = np.bincount(y.astype(int), minlength=NUM_CLASSES)
    print(f"Initial distribution: DOWN={counts[0]:,}, NEUTRAL={counts[1]:,}, UP={counts[2]:,}")
    
    # Split with purge
    X_tr, y_tr, X_val, y_val = purged_train_val_split(X, y)
    print(f"Train: {len(y_tr):,}, Val: {len(y_val):,}")
    
    # Apply stratified oversampling to training data
    print("\nApplying stratified oversampling to training data...")
    X_tr, y_tr = stratified_oversample(X_tr, y_tr)
    
    # Compute class weights
    print("\nComputing class weights...")
    class_weights = compute_class_weights(y_tr)
    
    del X, y
    gc.collect()
    
    def objective(trial: optuna.Trial) -> float:
        global _optuna_trial_results
        
        params = {
            'objective': 'multiclass',
            'num_class': NUM_CLASSES,
            'metric': 'None',
            'boosting_type': 'gbdt',
            'verbosity': -1,
            'seed': 42,
            'num_threads': 4,
            
            'num_leaves': trial.suggest_int('num_leaves', 31, 127),
            'learning_rate': trial.suggest_float('learning_rate', 0.02, 0.2, log=True),
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 200, 1000),
            'lambda_l1': trial.suggest_float('lambda_l1', 1e-4, 1.0, log=True),
            'lambda_l2': trial.suggest_float('lambda_l2', 1e-4, 1.0, log=True),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 5),
            'max_depth': trial.suggest_int('max_depth', 6, 12),
        }
        
        gamma = trial.suggest_float('focal_gamma', 2.0, 5.0)  # Increased range
        
        fobj = focal_loss_objective(gamma, class_weights)
        feval = focal_loss_metric(gamma, class_weights)
        feval_recall = per_class_recall_metric()
        
        dtrain = lgb.Dataset(X_tr, label=y_tr, free_raw_data=True)
        dval = lgb.Dataset(X_val, label=y_val, free_raw_data=True)
        
        model = lgb_train_compat(
            params,
            dtrain,
            num_boost_round=500,
            valid_sets=[dval],
            fobj=fobj,
            feval=[feval, feval_recall],  # Multiple eval metrics
            callbacks=[
                lgb.early_stopping(50, verbose=False),
                lgb.log_evaluation(0)
            ]
        )
        
        y_pred_raw = model.predict(X_val, raw_score=True)
        y_pred = np.argmax(y_pred_raw, axis=1)
        
        # Compute detailed metrics
        metrics = compute_detailed_metrics(y_val, y_pred)
        
        # Store for printing
        _optuna_trial_results.append({
            'trial': trial.number,
            'macro_f1': metrics['macro_f1'],
            'up_down_recall': metrics['up_down_recall_avg'],
            'recall': metrics['recall'],
            'confusion_matrix': metrics['confusion_matrix'],
            'gamma': gamma,
        })
        
        # Print detailed class metrics every 5 trials
        if (trial.number + 1) % 1 == 0:
            print(f"\n{'='*60}")
            print(f"TRIAL {trial.number + 1} SUMMARY")
            print(f"{'='*60}")
            print(f"Macro F1 (Overall): {metrics['macro_f1']:.4f}")
            
            # Direct F1 values for each class
            print(f"Individual F1 Scores:")
            print(f"  - DOWN:    {metrics['f1']['DOWN']:.4f}")
            print(f"  - NEUTRAL: {metrics['f1']['NEUTRAL']:.4f}")
            print(f"  - UP:      {metrics['f1']['UP']:.4f}")
            
            print(f"\nPer-class Recall: DOWN={metrics['recall']['DOWN']:.3f}, "
                  f"NEUTRAL={metrics['recall']['NEUTRAL']:.3f}, "
                  f"UP={metrics['recall']['UP']:.3f}")
            
            print_confusion_matrix(metrics['confusion_matrix'], "Validation Confusion Matrix")
            print(f"{'='*60}\n")
        
        del dtrain, dval, model
        gc.collect()
        
        # Optimize for macro F1 but could also use up_down_recall
        return metrics['macro_f1']
    
    # Create study
    sampler = TPESampler(seed=42)
    study = optuna.create_study(
        direction='maximize',
        sampler=sampler,
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5)
    )
    
    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout,
        show_progress_bar=True,
        gc_after_trial=True
    )
    
    # Print final summary
    print(f"\n{'='*60}")
    print("OPTUNA OPTIMIZATION COMPLETE")
    print(f"{'='*60}")
    print(f"Best Macro F1: {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")
    
    # Find best trial's detailed metrics
    best_trial_idx = study.best_trial.number
    for result in _optuna_trial_results:
        if result['trial'] == best_trial_idx:
            print(f"\nBest Trial Metrics:")
            print(f"  UP+DOWN Recall Avg: {result['up_down_recall']:.4f}")
            print(f"  Per-class Recall: {result['recall']}")
            print_confusion_matrix(result['confusion_matrix'], "Best Trial Confusion Matrix")
            break
    
    del X_tr, y_tr, X_val, y_val
    gc.collect()
    
    return study.best_params


# =============================================================================
# FINAL TRAINING
# =============================================================================

def train_final_model(
    train_files: List[str],
    val_files: List[str],
    best_params: Dict
) -> lgb.Booster:
    """
    Train final model using incremental learning with stratified chunks.
    """
    print("\n" + "=" * 60)
    print("TRAINING FINAL MODEL (INCREMENTAL)")
    print("=" * 60)
    
    # Extract focal gamma
    gamma = best_params.pop('focal_gamma', 2.0)
    
    params = {
        'num_class': NUM_CLASSES,
        'metric': 'None',
        'boosting_type': 'gbdt',
        'verbosity': -1,
        'seed': 42,
        'num_threads': 6,
        **best_params
    }
    
    # Load validation data
    print("Loading validation data...")
    X_val, y_val = load_chunks_concat(val_files, max_chunks=VAL_CHUNKS)
    print(f"Val samples: {len(y_val):,}")
    
    dval = lgb.Dataset(X_val, label=y_val, free_raw_data=True)
    
    model = None
    total_trees = 0
    trees_per_chunk = 200
    
    # Compute class weights from first few chunks
    print("\nComputing class weights...")
    class_weights = compute_class_weights(np.array([0, 1, 2]))  # Will use fixed weights
    
    # Create loss functions
    fobj = focal_loss_objective(gamma, class_weights)
    feval = focal_loss_metric(gamma, class_weights)
    feval_recall = per_class_recall_metric()
    
    # Incremental training on chunks
    print(f"\nIncremental training on {len(train_files)} chunks...")
    print(f"Trees per chunk: {trees_per_chunk}")
    
    for chunk_idx, (X_chunk, y_chunk) in enumerate(tqdm(
        stream_chunks(train_files), 
        total=len(train_files),
        desc="Training"
    )):
        # Apply stratified oversampling to each chunk
        X_chunk, y_chunk = stratified_oversample(X_chunk, y_chunk, random_state=42 + chunk_idx)
        
        dtrain = lgb.Dataset(X_chunk, label=y_chunk, free_raw_data=True)
        
        if IS_LGBM_4X:
            params_with_obj = params.copy()
            params_with_obj['objective'] = fobj
            
            model = lgb.train(
                params_with_obj,
                dtrain,
                num_boost_round=trees_per_chunk,
                valid_sets=[dval],
                valid_names=['val'],
                feval=[feval, feval_recall],
                init_model=model,
                callbacks=[
                    lgb.early_stopping(30, verbose=False),
                    lgb.log_evaluation(0)
                ]
            )
        else:
            model = lgb.train(
                params,
                dtrain,
                num_boost_round=trees_per_chunk,
                valid_sets=[dval],
                valid_names=['val'],
                fobj=fobj,
                feval=[feval, feval_recall],
                init_model=model,
                callbacks=[
                    lgb.early_stopping(30, verbose=False),
                    lgb.log_evaluation(0)
                ]
            )
        
        total_trees = model.num_trees()
        
        # Early stop if we have enough trees
        if total_trees >= 3000:
            print(f"\nReached {total_trees} trees, stopping early")
            break
        
        # Progress update every 20 chunks
        if (chunk_idx + 1) % 20 == 0:
            y_pred_raw = model.predict(X_val, raw_score=True)
            y_pred = np.argmax(y_pred_raw, axis=1)
            metrics = compute_detailed_metrics(y_val, y_pred)
            print(f"\n  Chunk {chunk_idx+1}: {total_trees} trees")
            print(f"    Macro F1: {metrics['macro_f1']:.4f}")
            print(f"    Recall - DOWN: {metrics['recall']['DOWN']:.3f}, "
                  f"NEUTRAL: {metrics['recall']['NEUTRAL']:.3f}, "
                  f"UP: {metrics['recall']['UP']:.3f}")
        
        del dtrain, X_chunk, y_chunk
        gc.collect()
    
    print(f"\nFinal model: {model.num_trees()} trees")
    
    # Store metadata
    model._focal_gamma = gamma
    model._class_weights = class_weights
    
    return model, X_val, y_val


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate(model: lgb.Booster, X_test: np.ndarray, y_test: np.ndarray):
    """Evaluate model on test set with detailed metrics."""
    print("\n" + "=" * 60)
    print("TEST SET EVALUATION")
    print("=" * 60)
    
    y_pred_raw = model.predict(X_test, raw_score=True)
    y_pred = np.argmax(y_pred_raw, axis=1)
    
    # Detailed metrics
    metrics = compute_detailed_metrics(y_test, y_pred)
    
    print(f"\nAccuracy:   {(y_pred == y_test).mean():.4f}")
    print(f"Macro F1:   {metrics['macro_f1']:.4f}")
    print(f"UP+DOWN Recall Avg: {metrics['up_down_recall_avg']:.4f}")
    
    print("\nPer-class Metrics:")
    print(f"  {'Class':<10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print(f"  {'-'*40}")
    for cls in ['DOWN', 'NEUTRAL', 'UP']:
        print(f"  {cls:<10} {metrics['precision'][cls]:>10.4f} "
              f"{metrics['recall'][cls]:>10.4f} {metrics['f1'][cls]:>10.4f}")
    
    print_confusion_matrix(metrics['confusion_matrix'], "\nTest Set Confusion Matrix")
    
    return {
        'accuracy': (y_pred == y_test).mean(),
        'macro_f1': metrics['macro_f1'],
        'per_class_recall': metrics['recall'],
        'per_class_precision': metrics['precision'],
        'per_class_f1': metrics['f1'],
        'confusion_matrix': metrics['confusion_matrix'].tolist()
    }


# =============================================================================
# SAVE MODEL
# =============================================================================

def save_model(model: lgb.Booster, best_params: Dict, results: Dict):
    """Save model and artifacts."""
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(exist_ok=True)
    
    # Save model
    model.save_model(str(output_dir / "model_lgbm.txt"))
    
    # Save simple calibrator placeholder
    class SimpleCalibrator:
        def __init__(self):
            self.n_classes = NUM_CLASSES
            self.fitted = True
        def transform(self, p):
            return p / p.sum(axis=1, keepdims=True)
    
    with open(output_dir / "calibrator.pkl", 'wb') as f:
        pickle.dump(SimpleCalibrator(), f)
    
    # Save artifacts
    artifacts = {
        'best_params': best_params,
        'results': results,
        'config': {
            'feature_names': [
                'qty', 'time_delta', 'log_return', 'trade_intensity',
                'price_acceleration', 'volume_imbalance', 'rolling_signed_volume',
                'side', 'price_volatility', 'volume_volatility', 
                'time_delta_std', 'price_distance_from_ma'
            ],
            'log_transform_features': LOG_TRANSFORM_IDX,
            'confidence_threshold': 0.55,
            'class_weight_mode': CLASS_WEIGHT_MODE,
            'fixed_class_weights': FIXED_CLASS_WEIGHTS.tolist(),
        },
        'focal_gamma': getattr(model, '_focal_gamma', 2.0),
        'class_weights': getattr(model, '_class_weights', [1, 1, 1]).tolist() 
            if hasattr(model, '_class_weights') and isinstance(getattr(model, '_class_weights'), np.ndarray) 
            else [1, 1, 1],
        'timestamp': datetime.now().isoformat(),
    }
    
    with open(output_dir / "artifacts.json", 'w') as f:
        json.dump(artifacts, f, indent=2)
    
    print(f"\nModel saved to {output_dir}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("\n" + "=" * 60)
    print("MEMORY-EFFICIENT LGBM TRAINING")
    print("=" * 60)
    print(f"Started: {datetime.now()}")
    print(f"\nClass Weight Mode: {CLASS_WEIGHT_MODE}")
    if CLASS_WEIGHT_MODE == 'fixed':
        print(f"Fixed Weights: {FIXED_CLASS_WEIGHTS}")
    
    # Load manifest
    manifest = load_manifest()
    print(f"\nTrain files: {len(manifest['train'])}")
    print(f"Val files: {len(manifest['val'])}")
    print(f"Test files: {len(manifest['test'])}")
    
    # Hyperparameter optimization
    best_params = run_optuna(manifest['train'])
    
    # Train final model
    model, X_val, y_val = train_final_model(
        manifest['train'],
        manifest['val'],
        best_params.copy()
    )
    
    # Load test data
    print("\nLoading test data...")
    X_test, y_test = load_chunks_concat(manifest['test'], max_chunks=20)
    print(f"Test samples: {len(y_test):,}")
    
    # Evaluate
    results = evaluate(model, X_test, y_test)
    
    # Save
    save_model(model, best_params, results)
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Finished: {datetime.now()}")


if __name__ == "__main__":
    main()