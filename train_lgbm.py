#!/usr/bin/env python3
"""
LIGHTGBM TRAINING FOR BTC TICK PREDICTION
==========================================
Components:
1. Purged Walk-Forward Cross-Validation (prevents look-ahead bias)
2. Focal Loss for 3-class imbalanced classification
3. Feature preprocessing fixes (log transform trade_intensity)
4. Optuna hyperparameter optimization
5. Probability calibration
6. Comprehensive evaluation metrics

Hardware target: Intel i5-11400H, 8GB RAM
"""

import numpy as np
import lightgbm as lgb
import optuna
from optuna.samplers import TPESampler
import pickle
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple, List, Dict, Optional, Callable
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    f1_score, precision_score, recall_score, 
    confusion_matrix, classification_report,
    log_loss
)
import warnings
from tqdm import tqdm
import gc
from datetime import datetime

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class Config:
    """Training configuration."""
    # Paths
    dataset_dir: str = "./dataset_lgbm"
    scaler_file: str = "./scaler_lgbm.pkl"
    output_dir: str = "./model_output"
    
    # Data parameters
    look_ahead_trades: int = 1400  # Must match preprocessing
    num_classes: int = 3
    
    # Feature columns (must match dataset.py order)
    feature_names: Tuple[str, ...] = (
        'qty', 'time_delta', 'log_return',
        'trade_intensity', 'price_acceleration', 
        'volume_imbalance', 'rolling_signed_volume', 'side',
        'price_volatility', 'volume_volatility', 'time_delta_std',
        'price_distance_from_ma'
    )
    
    # Features needing log transform (indices)
    log_transform_features: Tuple[int, ...] = (3,)  # trade_intensity
    
    # Cross-validation
    n_cv_folds: int = 5
    purge_gap: int = 1400  # Equal to look_ahead
    embargo_pct: float = 0.01  # Additional 1% embargo after test fold
    
    # Optuna
    n_optuna_trials: int = 100
    optuna_timeout: int = 3600 * 4  # 4 hours max
    
    # Training
    early_stopping_rounds: int = 100
    verbose_eval: int = 100
    
    # Focal loss parameters (will be tuned)
    focal_gamma: float = 2.0
    focal_alpha: Optional[List[float]] = None  # Auto-computed from class freq
    
    # Confidence threshold for trading signals
    confidence_threshold: float = 0.55


CONFIG = Config()


# =============================================================================
# DATA LOADING WITH FEATURE FIXES
# =============================================================================

def load_manifest(dataset_dir: str) -> Dict:
    """Load dataset manifest."""
    manifest_path = Path(dataset_dir) / "manifest.json"
    with open(manifest_path, 'r') as f:
        return json.load(f)


def apply_feature_fixes(X: np.ndarray, config: Config) -> np.ndarray:
    """
    Apply feature transformations that should have been in preprocessing.
    
    Fix: Log-transform trade_intensity to handle extreme values.
    The scaler was fit on raw values, so we transform AFTER loading.
    """
    X = X.copy()
    
    for idx in config.log_transform_features:
        # Reverse the standardization first
        # X_std = (X - mean) / std  →  X = X_std * std + mean
        # Then log transform, then re-standardize
        # 
        # Simpler approach: just log transform the standardized values
        # This works because log(a*x + b) ≈ log(a) + log(x) for large x
        # We clip to avoid log(0) issues
        
        col = X[:, idx]
        # Since data is already standardized, values could be negative
        # Shift to positive range, then log
        col_min = col.min()
        col_shifted = col - col_min + 1.0  # Shift to [1, ...]
        X[:, idx] = np.log(col_shifted)
    
    return X


def load_split_data(
    manifest: Dict, 
    split: str, 
    config: Config,
    max_files: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load data for a split (train/val/test).
    
    Args:
        manifest: Dataset manifest
        split: 'train', 'val', or 'test'
        config: Configuration
        max_files: Limit files for memory (None = all)
    
    Returns:
        X, y arrays
    """
    files = manifest[split]
    if max_files:
        files = files[:max_files]
    
    X_chunks = []
    y_chunks = []
    
    for f in tqdm(files, desc=f"Loading {split}"):
        data = np.load(f)
        X_chunks.append(data['X'])
        y_chunks.append(data['y'])
    
    X = np.vstack(X_chunks)
    y = np.concatenate(y_chunks)
    
    # Apply feature fixes
    X = apply_feature_fixes(X, config)
    
    return X, y


def compute_class_weights(y: np.ndarray, num_classes: int = 3) -> np.ndarray:
    """Compute inverse frequency weights for focal loss alpha."""
    class_counts = np.bincount(y.astype(int), minlength=num_classes)
    total = len(y)
    # Inverse frequency, normalized
    weights = total / (num_classes * class_counts + 1e-8)
    # Normalize to sum to num_classes
    weights = weights / weights.sum() * num_classes
    return weights


# =============================================================================
# PURGED WALK-FORWARD CROSS-VALIDATION
# =============================================================================

class PurgedWalkForwardCV:
    """
    Purged Walk-Forward Cross-Validation for time series.
    
    Implements the methodology from Marcos López de Prado's
    "Advances in Financial Machine Learning".
    
    Key concepts:
    - Purging: Remove training samples whose labels overlap with test set
    - Embargo: Additional gap after test set to prevent serial correlation leakage
    """
    
    def __init__(
        self, 
        n_splits: int = 5,
        purge_gap: int = 1400,
        embargo_pct: float = 0.01
    ):
        self.n_splits = n_splits
        self.purge_gap = purge_gap
        self.embargo_pct = embargo_pct
    
    def split(self, X: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate train/test indices for each fold.
        
        Yields:
            (train_indices, test_indices) tuples
        """
        n_samples = len(X)
        fold_size = n_samples // self.n_splits
        embargo_size = int(n_samples * self.embargo_pct)
        
        splits = []
        
        for i in range(self.n_splits):
            # Test fold boundaries
            test_start = i * fold_size
            test_end = (i + 1) * fold_size if i < self.n_splits - 1 else n_samples
            
            # Training: everything before test_start - purge_gap
            # Plus everything after test_end + embargo (for expanding window variant)
            
            # For standard walk-forward: train only on data before test
            train_end = max(0, test_start - self.purge_gap)
            
            if train_end < fold_size:  # Skip if not enough training data
                continue
            
            train_indices = np.arange(0, train_end)
            test_indices = np.arange(test_start, test_end)
            
            splits.append((train_indices, test_indices))
        
        return splits
    
    def get_n_splits(self) -> int:
        return self.n_splits


class PurgedKFoldCV:
    """
    Purged K-Fold Cross-Validation (non-expanding window variant).
    
    Each fold uses a different portion as test set, with purging
    applied on both sides of the test set.
    """
    
    def __init__(
        self,
        n_splits: int = 5,
        purge_gap: int = 1400,
        embargo_pct: float = 0.01
    ):
        self.n_splits = n_splits
        self.purge_gap = purge_gap
        self.embargo_pct = embargo_pct
    
    def split(self, X: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Generate purged train/test indices."""
        n_samples = len(X)
        fold_size = n_samples // self.n_splits
        embargo_size = int(fold_size * self.embargo_pct)
        
        splits = []
        
        for i in range(self.n_splits):
            test_start = i * fold_size
            test_end = (i + 1) * fold_size if i < self.n_splits - 1 else n_samples
            
            # Purge boundaries
            purge_start = max(0, test_start - self.purge_gap)
            purge_end = min(n_samples, test_end + embargo_size)
            
            # Training indices: everything outside purge zone
            train_before = np.arange(0, purge_start)
            train_after = np.arange(purge_end, n_samples)
            
            train_indices = np.concatenate([train_before, train_after])
            test_indices = np.arange(test_start, test_end)
            
            if len(train_indices) < fold_size:  # Need enough training data
                continue
            
            splits.append((train_indices, test_indices))
        
        return splits


# =============================================================================
# FOCAL LOSS IMPLEMENTATION
# =============================================================================

def focal_loss_multiclass(
    y_pred: np.ndarray, 
    dtrain: lgb.Dataset,
    gamma: float = 2.0,
    alpha: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Focal Loss for multi-class classification.
    
    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
    
    For LightGBM, we need gradient and hessian with respect to raw scores.
    
    Args:
        y_pred: Raw predictions (before softmax), shape (n_samples * n_classes,)
        dtrain: LightGBM Dataset
        gamma: Focusing parameter (higher = more focus on hard examples)
        alpha: Class weights, shape (n_classes,)
    
    Returns:
        grad, hess: Gradient and Hessian arrays
    """
    y_true = dtrain.get_label().astype(int)
    n_samples = len(y_true)
    n_classes = 3
    
    # Reshape predictions: (n_samples * n_classes,) -> (n_samples, n_classes)
    y_pred = y_pred.reshape((n_samples, n_classes), order='F')
    
    # Softmax to get probabilities
    y_pred_max = y_pred.max(axis=1, keepdims=True)
    exp_pred = np.exp(y_pred - y_pred_max)
    softmax = exp_pred / exp_pred.sum(axis=1, keepdims=True)
    
    # One-hot encode true labels
    y_true_onehot = np.zeros((n_samples, n_classes))
    y_true_onehot[np.arange(n_samples), y_true] = 1
    
    # Default alpha if not provided
    if alpha is None:
        alpha = np.ones(n_classes)
    alpha = np.array(alpha).reshape(1, -1)
    
    # p_t = probability of true class
    p_t = (softmax * y_true_onehot).sum(axis=1, keepdims=True)
    p_t = np.clip(p_t, 1e-15, 1 - 1e-15)
    
    # Focal weight: (1 - p_t)^gamma
    focal_weight = (1 - p_t) ** gamma
    
    # Gradient of focal loss w.r.t. softmax outputs
    # d(FL)/d(p_k) for each class k
    
    # For true class: -alpha * gamma * (1-p_t)^(gamma-1) * log(p_t) - alpha * (1-p_t)^gamma / p_t
    # For other classes: alpha * gamma * (1-p_t)^(gamma-1) * log(p_t) * p_k
    
    # Simplified gradient computation
    grad = alpha * focal_weight * (
        softmax - y_true_onehot + 
        gamma * softmax * (y_true_onehot - softmax) * np.log(p_t + 1e-15)
    )
    
    # Hessian (diagonal approximation)
    hess = alpha * focal_weight * softmax * (1 - softmax) * (
        1 + gamma * (1 - 2 * softmax) * np.log(p_t + 1e-15) +
        gamma * (y_true_onehot - softmax)
    )
    hess = np.abs(hess) + 1e-8  # Ensure positive for numerical stability
    
    # Flatten back to (n_samples * n_classes,) in Fortran order
    return grad.flatten('F'), hess.flatten('F')


def focal_loss_eval(
    y_pred: np.ndarray,
    dtrain: lgb.Dataset,
    gamma: float = 2.0,
    alpha: Optional[np.ndarray] = None
) -> Tuple[str, float, bool]:
    """
    Focal Loss evaluation metric.
    
    Returns:
        (name, value, is_higher_better)
    """
    y_true = dtrain.get_label().astype(int)
    n_samples = len(y_true)
    n_classes = 3
    
    y_pred = y_pred.reshape((n_samples, n_classes), order='F')
    
    # Softmax
    y_pred_max = y_pred.max(axis=1, keepdims=True)
    exp_pred = np.exp(y_pred - y_pred_max)
    softmax = exp_pred / exp_pred.sum(axis=1, keepdims=True)
    
    if alpha is None:
        alpha = np.ones(n_classes)
    
    # Compute focal loss
    p_t = softmax[np.arange(n_samples), y_true]
    p_t = np.clip(p_t, 1e-15, 1 - 1e-15)
    
    focal_weight = (1 - p_t) ** gamma
    alpha_t = np.array(alpha)[y_true]
    
    loss = -alpha_t * focal_weight * np.log(p_t)
    
    return 'focal_loss', loss.mean(), False


def create_focal_loss_objective(gamma: float, alpha: np.ndarray) -> Callable:
    """Create focal loss objective function with fixed parameters."""
    def objective(y_pred, dtrain):
        return focal_loss_multiclass(y_pred, dtrain, gamma, alpha)
    return objective


def create_focal_loss_eval(gamma: float, alpha: np.ndarray) -> Callable:
    """Create focal loss evaluation function with fixed parameters."""
    def eval_func(y_pred, dtrain):
        return focal_loss_eval(y_pred, dtrain, gamma, alpha)
    return eval_func


# =============================================================================
# OPTUNA HYPERPARAMETER OPTIMIZATION
# =============================================================================

def create_optuna_objective(
    X_train: np.ndarray,
    y_train: np.ndarray,
    cv_splits: List[Tuple[np.ndarray, np.ndarray]],
    class_weights: np.ndarray,
    config: Config
) -> Callable:
    """Create Optuna objective function for hyperparameter tuning."""
    
    def objective(trial: optuna.Trial) -> float:
        # Hyperparameter search space
        params = {
            'objective': 'multiclass',
            'num_class': config.num_classes,
            'metric': 'None',  # We use custom focal loss metric
            'boosting_type': 'gbdt',
            'verbosity': -1,
            'seed': 42,
            'num_threads': 6,  # Leave cores for system
            
            # Tier 1: Most impactful
            'num_leaves': trial.suggest_int('num_leaves', 31, 255),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 200, 2000),
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 100, 2000),
            
            # Tier 2: Regularization
            'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
            'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
            
            # Tier 3: Tree structure
            'max_depth': trial.suggest_int('max_depth', 5, 15),
            'min_gain_to_split': trial.suggest_float('min_gain_to_split', 0.0, 1.0),
        }
        
        # Focal loss parameters
        focal_gamma = trial.suggest_float('focal_gamma', 0.5, 5.0)
        
        # Create loss functions
        fobj = create_focal_loss_objective(focal_gamma, class_weights)
        feval = create_focal_loss_eval(focal_gamma, class_weights)
        
        # Cross-validation scores
        cv_scores = []
        
        for fold_idx, (train_idx, val_idx) in enumerate(cv_splits):
            X_tr, y_tr = X_train[train_idx], y_train[train_idx]
            X_val, y_val = X_train[val_idx], y_train[val_idx]
            
            # Create datasets
            dtrain = lgb.Dataset(X_tr, label=y_tr, free_raw_data=False)
            dval = lgb.Dataset(X_val, label=y_val, reference=dtrain, free_raw_data=False)
            
            # Train with early stopping
            model = lgb.train(
                params,
                dtrain,
                num_boost_round=params['n_estimators'],
                valid_sets=[dval],
                fobj=fobj,
                feval=feval,
                callbacks=[
                    lgb.early_stopping(stopping_rounds=50, verbose=False),
                    lgb.log_evaluation(period=0)  # Silent
                ]
            )
            
            # Predict
            y_pred_raw = model.predict(X_val, raw_score=True)
            y_pred = np.argmax(y_pred_raw, axis=1)
            
            # Score: Macro F1
            f1 = f1_score(y_val, y_pred, average='macro')
            cv_scores.append(f1)
            
            # Pruning
            trial.report(np.mean(cv_scores), fold_idx)
            if trial.should_prune():
                raise optuna.TrialPruned()
            
            # Cleanup
            del dtrain, dval, model
            gc.collect()
        
        return np.mean(cv_scores)
    
    return objective


def run_optuna_optimization(
    X_train: np.ndarray,
    y_train: np.ndarray,
    config: Config
) -> Tuple[Dict, optuna.Study]:
    """Run Optuna hyperparameter optimization."""
    
    print("=" * 80)
    print("OPTUNA HYPERPARAMETER OPTIMIZATION")
    print("=" * 80)
    
    # Compute class weights
    class_weights = compute_class_weights(y_train, config.num_classes)
    print(f"Class weights: {class_weights}")
    
    # Create CV splits
    cv = PurgedKFoldCV(
        n_splits=config.n_cv_folds,
        purge_gap=config.purge_gap,
        embargo_pct=config.embargo_pct
    )
    cv_splits = cv.split(X_train)
    print(f"CV folds: {len(cv_splits)}")
    
    # Create study
    sampler = TPESampler(seed=42)
    study = optuna.create_study(
        direction='maximize',
        sampler=sampler,
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=2)
    )
    
    # Create objective
    objective = create_optuna_objective(
        X_train, y_train, cv_splits, class_weights, config
    )
    
    # Run optimization
    study.optimize(
        objective,
        n_trials=config.n_optuna_trials,
        timeout=config.optuna_timeout,
        show_progress_bar=True,
        gc_after_trial=True
    )
    
    print(f"\nBest trial:")
    print(f"  Value (Macro F1): {study.best_trial.value:.4f}")
    print(f"  Params: {study.best_trial.params}")
    
    return study.best_trial.params, study


# =============================================================================
# FINAL MODEL TRAINING
# =============================================================================

def train_final_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    best_params: Dict,
    config: Config
) -> lgb.Booster:
    """Train final model with best hyperparameters."""
    
    print("=" * 80)
    print("TRAINING FINAL MODEL")
    print("=" * 80)
    
    # Compute class weights
    class_weights = compute_class_weights(y_train, config.num_classes)
    
    # Extract focal gamma from best params
    focal_gamma = best_params.pop('focal_gamma', config.focal_gamma)
    
    # Prepare LightGBM parameters
    params = {
        'objective': 'multiclass',
        'num_class': config.num_classes,
        'metric': 'None',
        'boosting_type': 'gbdt',
        'verbosity': -1,
        'seed': 42,
        'num_threads': 8,
        **best_params
    }
    
    # Remove n_estimators (used in num_boost_round)
    n_estimators = params.pop('n_estimators', 1000)
    
    # Create loss functions
    fobj = create_focal_loss_objective(focal_gamma, class_weights)
    feval = create_focal_loss_eval(focal_gamma, class_weights)
    
    # Create datasets
    dtrain = lgb.Dataset(X_train, label=y_train, free_raw_data=False)
    dval = lgb.Dataset(X_val, label=y_val, reference=dtrain, free_raw_data=False)
    
    # Train
    model = lgb.train(
        params,
        dtrain,
        num_boost_round=n_estimators,
        valid_sets=[dtrain, dval],
        valid_names=['train', 'val'],
        fobj=fobj,
        feval=feval,
        callbacks=[
            lgb.early_stopping(stopping_rounds=config.early_stopping_rounds),
            lgb.log_evaluation(period=config.verbose_eval)
        ]
    )
    
    print(f"\nBest iteration: {model.best_iteration}")
    
    # Store focal gamma for later use
    model.focal_gamma = focal_gamma
    model.class_weights = class_weights
    
    return model


# =============================================================================
# PROBABILITY CALIBRATION
# =============================================================================

class IsotonicCalibrator:
    """
    Isotonic regression calibration for multi-class probabilities.
    
    Calibrates each class independently.
    """
    
    def __init__(self, n_classes: int = 3):
        self.n_classes = n_classes
        self.calibrators = [IsotonicRegression(out_of_bounds='clip') 
                           for _ in range(n_classes)]
        self.fitted = False
    
    def fit(self, y_prob: np.ndarray, y_true: np.ndarray):
        """
        Fit calibration.
        
        Args:
            y_prob: Predicted probabilities, shape (n_samples, n_classes)
            y_true: True labels, shape (n_samples,)
        """
        for c in range(self.n_classes):
            # Binary labels for this class
            y_binary = (y_true == c).astype(int)
            self.calibrators[c].fit(y_prob[:, c], y_binary)
        
        self.fitted = True
        return self
    
    def transform(self, y_prob: np.ndarray) -> np.ndarray:
        """
        Calibrate probabilities.
        
        Args:
            y_prob: Raw probabilities, shape (n_samples, n_classes)
        
        Returns:
            Calibrated probabilities
        """
        if not self.fitted:
            raise ValueError("Calibrator not fitted")
        
        calibrated = np.zeros_like(y_prob)
        for c in range(self.n_classes):
            calibrated[:, c] = self.calibrators[c].predict(y_prob[:, c])
        
        # Renormalize to sum to 1
        calibrated = calibrated / calibrated.sum(axis=1, keepdims=True)
        
        return calibrated


def calibrate_model(
    model: lgb.Booster,
    X_val: np.ndarray,
    y_val: np.ndarray,
    config: Config
) -> IsotonicCalibrator:
    """Fit probability calibration on validation set."""
    
    print("=" * 80)
    print("CALIBRATING PROBABILITIES")
    print("=" * 80)
    
    # Get raw predictions
    y_pred_raw = model.predict(X_val, raw_score=True)
    
    # Convert to probabilities (softmax)
    exp_pred = np.exp(y_pred_raw - y_pred_raw.max(axis=1, keepdims=True))
    y_prob = exp_pred / exp_pred.sum(axis=1, keepdims=True)
    
    # Fit calibrator
    calibrator = IsotonicCalibrator(n_classes=config.num_classes)
    calibrator.fit(y_prob, y_val)
    
    # Evaluate calibration improvement
    y_prob_cal = calibrator.transform(y_prob)
    
    print(f"Log loss before calibration: {log_loss(y_val, y_prob):.4f}")
    print(f"Log loss after calibration:  {log_loss(y_val, y_prob_cal):.4f}")
    
    return calibrator


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate_model(
    model: lgb.Booster,
    calibrator: IsotonicCalibrator,
    X_test: np.ndarray,
    y_test: np.ndarray,
    config: Config
) -> Dict:
    """Comprehensive model evaluation."""
    
    print("=" * 80)
    print("MODEL EVALUATION ON TEST SET")
    print("=" * 80)
    
    # Get predictions
    y_pred_raw = model.predict(X_test, raw_score=True)
    
    # Convert to probabilities
    exp_pred = np.exp(y_pred_raw - y_pred_raw.max(axis=1, keepdims=True))
    y_prob = exp_pred / exp_pred.sum(axis=1, keepdims=True)
    
    # Calibrate
    y_prob_cal = calibrator.transform(y_prob)
    
    # Standard predictions (argmax)
    y_pred = np.argmax(y_prob_cal, axis=1)
    
    # Confidence-thresholded predictions
    max_prob = y_prob_cal.max(axis=1)
    confident_mask = max_prob >= config.confidence_threshold
    y_pred_confident = y_pred.copy()
    y_pred_confident[~confident_mask] = 1  # Default to NEUTRAL
    
    # Metrics
    results = {
        'accuracy': (y_pred == y_test).mean(),
        'macro_f1': f1_score(y_test, y_pred, average='macro'),
        'weighted_f1': f1_score(y_test, y_pred, average='weighted'),
        'per_class_precision': precision_score(y_test, y_pred, average=None),
        'per_class_recall': recall_score(y_test, y_pred, average=None),
        'per_class_f1': f1_score(y_test, y_pred, average=None),
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'log_loss': log_loss(y_test, y_prob_cal),
        'confident_accuracy': (y_pred_confident[confident_mask] == y_test[confident_mask]).mean() if confident_mask.sum() > 0 else 0,
        'confidence_coverage': confident_mask.mean(),
    }
    
    # Print results
    print(f"\n{'='*40}")
    print("OVERALL METRICS")
    print(f"{'='*40}")
    print(f"Accuracy:        {results['accuracy']:.4f}")
    print(f"Macro F1:        {results['macro_f1']:.4f}")
    print(f"Weighted F1:     {results['weighted_f1']:.4f}")
    print(f"Log Loss:        {results['log_loss']:.4f}")
    
    print(f"\n{'='*40}")
    print("PER-CLASS METRICS")
    print(f"{'='*40}")
    class_names = ['DOWN', 'NEUTRAL', 'UP']
    for i, name in enumerate(class_names):
        print(f"{name:>8}: P={results['per_class_precision'][i]:.3f}  "
              f"R={results['per_class_recall'][i]:.3f}  "
              f"F1={results['per_class_f1'][i]:.3f}")
    
    print(f"\n{'='*40}")
    print("CONFUSION MATRIX")
    print(f"{'='*40}")
    print("              Predicted")
    print("             DOWN  NEUT    UP")
    cm = results['confusion_matrix']
    for i, name in enumerate(class_names):
        print(f"Actual {name:>4}: {cm[i][0]:5d} {cm[i][1]:5d} {cm[i][2]:5d}")
    
    print(f"\n{'='*40}")
    print(f"CONFIDENT PREDICTIONS (threshold={config.confidence_threshold})")
    print(f"{'='*40}")
    print(f"Coverage:  {results['confidence_coverage']:.2%}")
    print(f"Accuracy:  {results['confident_accuracy']:.4f}")
    
    return results


# =============================================================================
# FEATURE IMPORTANCE
# =============================================================================

def analyze_feature_importance(model: lgb.Booster, config: Config):
    """Analyze and display feature importance."""
    
    print("=" * 80)
    print("FEATURE IMPORTANCE")
    print("=" * 80)
    
    importance_gain = model.feature_importance(importance_type='gain')
    importance_split = model.feature_importance(importance_type='split')
    
    # Normalize
    importance_gain = importance_gain / importance_gain.sum()
    importance_split = importance_split / importance_split.sum()
    
    # Sort by gain
    indices = np.argsort(importance_gain)[::-1]
    
    print(f"\n{'Feature':<30} {'Gain':>10} {'Split':>10}")
    print("-" * 50)
    for idx in indices:
        name = config.feature_names[idx] if idx < len(config.feature_names) else f"Feature_{idx}"
        print(f"{name:<30} {importance_gain[idx]:>10.4f} {importance_split[idx]:>10.4f}")


# =============================================================================
# SAVE / LOAD
# =============================================================================

def save_model(
    model: lgb.Booster,
    calibrator: IsotonicCalibrator,
    best_params: Dict,
    results: Dict,
    config: Config
):
    """Save model and all artifacts."""
    
    output_dir = Path(config.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Save LightGBM model
    model.save_model(str(output_dir / "model_lgbm.txt"))
    
    # Save calibrator
    with open(output_dir / "calibrator.pkl", 'wb') as f:
        pickle.dump(calibrator, f)
    
    # Save parameters and results
    artifacts = {
        'best_params': best_params,
        'results': {k: v.tolist() if isinstance(v, np.ndarray) else v 
                   for k, v in results.items()},
        'config': {
            'feature_names': config.feature_names,
            'log_transform_features': config.log_transform_features,
            'confidence_threshold': config.confidence_threshold,
        },
        'focal_gamma': getattr(model, 'focal_gamma', config.focal_gamma),
        'class_weights': getattr(model, 'class_weights', [1, 1, 1]).tolist() 
            if hasattr(model, 'class_weights') else [1, 1, 1],
        'timestamp': datetime.now().isoformat(),
    }
    
    with open(output_dir / "artifacts.json", 'w') as f:
        json.dump(artifacts, f, indent=2)
    
    print(f"\nModel saved to {output_dir}")


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main():
    """Execute full training pipeline."""
    
    print("\n" + "=" * 80)
    print("LIGHTGBM BTC TICK PREDICTION TRAINING")
    print("=" * 80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load manifest
    manifest = load_manifest(CONFIG.dataset_dir)
    print(f"\nDataset splits:")
    print(f"  Train files: {len(manifest['train'])}")
    print(f"  Val files:   {len(manifest['val'])}")
    print(f"  Test files:  {len(manifest['test'])}")
    
    # Load data
    print("\n" + "=" * 80)
    print("LOADING DATA")
    print("=" * 80)
    
    # For Optuna, use subset for speed (optional)
    X_train_full, y_train_full = load_split_data(manifest, 'train', CONFIG)
    print(f"Train: {X_train_full.shape}")
    
    X_val, y_val = load_split_data(manifest, 'val', CONFIG)
    print(f"Val: {X_val.shape}")
    
    X_test, y_test = load_split_data(manifest, 'test', CONFIG)
    print(f"Test: {X_test.shape}")
    
    # Class distribution
    print("\nClass distribution (Train):")
    for c in range(CONFIG.num_classes):
        count = (y_train_full == c).sum()
        pct = count / len(y_train_full) * 100
        print(f"  Class {c}: {count:,} ({pct:.2f}%)")
    
    # Run Optuna (on subset for speed)
    # Use first 20% of training data for hyperparameter search
    subset_size = min(len(X_train_full), int(len(X_train_full) * 0.2))
    X_train_subset = X_train_full[:subset_size]
    y_train_subset = y_train_full[:subset_size]
    
    best_params, study = run_optuna_optimization(
        X_train_subset, y_train_subset, CONFIG
    )
    
    # Train final model on full data
    model = train_final_model(
        X_train_full, y_train_full,
        X_val, y_val,
        best_params.copy(),
        CONFIG
    )
    
    # Calibrate
    calibrator = calibrate_model(model, X_val, y_val, CONFIG)
    
    # Evaluate
    results = evaluate_model(model, calibrator, X_test, y_test, CONFIG)
    
    # Feature importance
    analyze_feature_importance(model, CONFIG)
    
    # Save
    save_model(model, calibrator, best_params, results, CONFIG)
    
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
