#!/usr/bin/env python3
"""
QUICK VALIDATION SCRIPT
=======================
Run this first to validate the pipeline works before full training.

Tests:
1. Data loading
2. Feature transformations
3. Focal loss computation
4. Single LightGBM training iteration
5. Prediction pipeline

Usage:
    python validate_pipeline.py
"""

import numpy as np
import lightgbm as lgb
import json
from pathlib import Path
import sys

# Check LightGBM version
from packaging import version
LGBM_VERSION = version.parse(lgb.__version__)
IS_LGBM_4X = LGBM_VERSION >= version.parse("4.0.0")
print(f"LightGBM version: {lgb.__version__} (4.x API: {IS_LGBM_4X})")


def test_data_loading():
    """Test data loading from manifest."""
    print("\n[1/5] Testing data loading...")
    
    manifest_path = Path("./dataset_lgbm/manifest.json")
    if not manifest_path.exists():
        print(f"  ERROR: {manifest_path} not found")
        return False
    
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)
    
    print(f"  Train files: {len(manifest['train'])}")
    print(f"  Val files: {len(manifest['val'])}")
    print(f"  Test files: {len(manifest['test'])}")
    
    # Load first chunk
    first_file = manifest['train'][0]
    if not Path(first_file).exists():
        print(f"  ERROR: {first_file} not found")
        return False
    
    data = np.load(first_file)
    X = data['X']
    y = data['y']
    
    print(f"  Sample chunk shape: X={X.shape}, y={y.shape}")
    print(f"  Features dtype: {X.dtype}")
    print(f"  Labels dtype: {y.dtype}")
    print(f"  Label distribution: {np.bincount(y.astype(int), minlength=3)}")
    
    return True


def test_feature_transforms():
    """Test feature transformations."""
    print("\n[2/5] Testing feature transformations...")
    
    # Load sample data
    with open("./dataset_lgbm/manifest.json", 'r') as f:
        manifest = json.load(f)
    
    data = np.load(manifest['train'][0])
    X = data['X'].copy()
    
    # Check for NaN/Inf before transform
    nan_count = np.isnan(X).sum()
    inf_count = np.isinf(X).sum()
    print(f"  Before transform - NaN: {nan_count}, Inf: {inf_count}")
    
    # Apply log transform to trade_intensity (index 3)
    col = X[:, 3]
    print(f"  trade_intensity stats before: min={col.min():.2e}, max={col.max():.2e}, mean={col.mean():.2e}")
    
    col_min = col.min()
    X[:, 3] = np.log(col - col_min + 1.0)
    
    col_after = X[:, 3]
    print(f"  trade_intensity stats after:  min={col_after.min():.2f}, max={col_after.max():.2f}, mean={col_after.mean():.2f}")
    
    # Check for NaN/Inf after transform
    nan_count = np.isnan(X).sum()
    inf_count = np.isinf(X).sum()
    print(f"  After transform - NaN: {nan_count}, Inf: {inf_count}")
    
    if nan_count > 0 or inf_count > 0:
        print("  WARNING: Transform introduced NaN/Inf values")
        return False
    
    return True


def test_focal_loss():
    """Test focal loss computation."""
    print("\n[3/5] Testing focal loss computation...")
    
    # Simulate predictions and labels
    n_samples = 1000
    n_classes = 3
    
    # Random raw scores
    y_pred = np.random.randn(n_samples * n_classes)
    y_true = np.random.randint(0, n_classes, n_samples)
    
    # Create mock dataset
    dtrain = lgb.Dataset(
        np.random.randn(n_samples, 10),
        label=y_true
    )
    dtrain.construct()
    
    # Focal loss parameters
    gamma = 2.0
    alpha = np.array([1.0, 1.0, 1.0])
    
    # Compute focal loss gradient/hessian
    y_pred_reshaped = y_pred.reshape((n_samples, n_classes), order='F')
    
    # Softmax
    exp_pred = np.exp(y_pred_reshaped - y_pred_reshaped.max(axis=1, keepdims=True))
    p = exp_pred / exp_pred.sum(axis=1, keepdims=True)
    p = np.clip(p, 1e-15, 1 - 1e-15)
    
    # One-hot
    y_onehot = np.zeros_like(p)
    y_onehot[np.arange(n_samples), y_true] = 1
    
    # p_t
    p_t = (p * y_onehot).sum(axis=1, keepdims=True)
    
    # Focal weight
    focal_w = (1 - p_t) ** gamma
    
    # Gradient
    alpha_t = alpha.reshape(1, -1)
    grad = alpha_t * focal_w * (p - y_onehot)
    
    # Hessian
    hess = alpha_t * focal_w * p * (1 - p) + 1e-8
    
    print(f"  Gradient shape: {grad.shape}")
    print(f"  Gradient stats: min={grad.min():.4f}, max={grad.max():.4f}")
    print(f"  Hessian stats: min={hess.min():.4f}, max={hess.max():.4f}")
    
    # Check for issues
    if np.isnan(grad).any() or np.isnan(hess).any():
        print("  ERROR: NaN in gradient/hessian")
        return False
    
    if (hess <= 0).any():
        print("  WARNING: Non-positive hessian values")
    
    return True


def test_lgbm_training():
    """Test a single LightGBM training iteration."""
    print("\n[4/5] Testing LightGBM training...")
    
    # Load small sample
    with open("./dataset_lgbm/manifest.json", 'r') as f:
        manifest = json.load(f)
    
    data = np.load(manifest['train'][0])
    X = data['X'][:10000].copy()  # Small subset
    y = data['y'][:10000].astype(int)
    
    # Apply transform
    col = X[:, 3]
    X[:, 3] = np.log(col - col.min() + 1.0)
    
    # Train/val split
    split = int(len(X) * 0.8)
    X_tr, y_tr = X[:split], y[:split]
    X_val, y_val = X[split:], y[split:]
    
    # Class weights
    counts = np.bincount(y_tr, minlength=3)
    weights = len(y_tr) / (3 * counts + 1e-8)
    weights = weights / weights.sum() * 3
    
    print(f"  Class weights: {weights}")
    
    # Create focal loss functions
    gamma = 2.0
    
    def fobj(y_pred, dtrain):
        y_true = dtrain.get_label().astype(int)
        n = len(y_true)
        
        y_pred = y_pred.reshape((n, 3), order='F')
        exp_pred = np.exp(y_pred - y_pred.max(axis=1, keepdims=True))
        p = exp_pred / exp_pred.sum(axis=1, keepdims=True)
        p = np.clip(p, 1e-15, 1 - 1e-15)
        
        y_onehot = np.zeros_like(p)
        y_onehot[np.arange(n), y_true] = 1
        
        p_t = (p * y_onehot).sum(axis=1, keepdims=True)
        focal_w = (1 - p_t) ** gamma
        alpha_t = weights.reshape(1, -1)
        
        grad = alpha_t * focal_w * (p - y_onehot)
        hess = alpha_t * focal_w * p * (1 - p) + 1e-8
        
        return grad.flatten('F'), hess.flatten('F')
    
    def feval(y_pred, dtrain):
        y_true = dtrain.get_label().astype(int)
        n = len(y_true)
        
        y_pred = y_pred.reshape((n, 3), order='F')
        exp_pred = np.exp(y_pred - y_pred.max(axis=1, keepdims=True))
        p = exp_pred / exp_pred.sum(axis=1, keepdims=True)
        p = np.clip(p, 1e-15, 1 - 1e-15)
        
        p_t = p[np.arange(n), y_true]
        focal_w = (1 - p_t) ** gamma
        alpha_t = weights[y_true]
        
        loss = -alpha_t * focal_w * np.log(p_t)
        return 'focal_loss', loss.mean(), False
    
    # Train
    params = {
        'objective': 'multiclass',
        'num_class': 3,
        'metric': 'None',
        'verbosity': -1,
        'num_leaves': 31,
        'learning_rate': 0.1,
        'seed': 42,
    }
    
    dtrain = lgb.Dataset(X_tr, label=y_tr)
    dval = lgb.Dataset(X_val, label=y_val)
    
    try:
        # Use appropriate API based on version
        if IS_LGBM_4X:
            # LightGBM 4.x: custom objective via params
            params_with_obj = params.copy()
            params_with_obj['objective'] = fobj
            
            model = lgb.train(
                params_with_obj,
                dtrain,
                num_boost_round=50,
                valid_sets=[dval],
                feval=feval,
                callbacks=[lgb.log_evaluation(0)]
            )
        else:
            # LightGBM 3.x: custom objective via fobj parameter
            model = lgb.train(
                params,
                dtrain,
                num_boost_round=50,
                valid_sets=[dval],
                fobj=fobj,
                feval=feval,
                callbacks=[lgb.log_evaluation(0)]
            )
        print(f"  Training completed: {model.num_trees()} trees")
    except Exception as e:
        print(f"  ERROR: Training failed - {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def test_prediction():
    """Test prediction pipeline."""
    print("\n[5/5] Testing prediction pipeline...")
    
    # Load sample
    with open("./dataset_lgbm/manifest.json", 'r') as f:
        manifest = json.load(f)
    
    data = np.load(manifest['train'][0])
    X = data['X'][:1000].copy()
    y = data['y'][:1000].astype(int)
    
    # Apply transform
    col = X[:, 3]
    X[:, 3] = np.log(col - col.min() + 1.0)
    
    # Quick train
    params = {
        'objective': 'multiclass',
        'num_class': 3,
        'verbosity': -1,
        'num_leaves': 15,
        'learning_rate': 0.1,
    }
    
    dtrain = lgb.Dataset(X, label=y)
    model = lgb.train(params, dtrain, num_boost_round=10)
    
    # Predict
    y_pred_raw = model.predict(X, raw_score=True)
    print(f"  Raw prediction shape: {y_pred_raw.shape}")
    
    # Softmax
    exp_pred = np.exp(y_pred_raw - y_pred_raw.max(axis=1, keepdims=True))
    y_prob = exp_pred / exp_pred.sum(axis=1, keepdims=True)
    print(f"  Probability shape: {y_prob.shape}")
    print(f"  Probability sums to 1: {np.allclose(y_prob.sum(axis=1), 1.0)}")
    
    # Argmax
    y_pred = np.argmax(y_prob, axis=1)
    print(f"  Prediction distribution: {np.bincount(y_pred, minlength=3)}")
    
    # Accuracy
    acc = (y_pred == y).mean()
    print(f"  Quick accuracy: {acc:.4f}")
    
    return True


def main():
    print("=" * 60)
    print("PIPELINE VALIDATION")
    print("=" * 60)
    
    tests = [
        ("Data Loading", test_data_loading),
        ("Feature Transforms", test_feature_transforms),
        ("Focal Loss", test_focal_loss),
        ("LightGBM Training", test_lgbm_training),
        ("Prediction", test_prediction),
    ]
    
    results = []
    for name, test_fn in tests:
        try:
            passed = test_fn()
            results.append((name, passed))
        except Exception as e:
            print(f"  EXCEPTION: {e}")
            results.append((name, False))
    
    print("\n" + "=" * 60)
    print("VALIDATION RESULTS")
    print("=" * 60)
    
    all_passed = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False
    
    print("=" * 60)
    
    if all_passed:
        print("\nAll tests passed! Ready for full training.")
        print("\nNext steps:")
        print("  1. Run: python train_lgbm_lite.py  (memory-efficient)")
        print("  2. Or:  python train_lgbm.py       (full features)")
        return 0
    else:
        print("\nSome tests failed. Fix issues before training.")
        return 1


if __name__ == "__main__":
    sys.exit(main())