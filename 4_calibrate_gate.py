#!/usr/bin/env python3
"""
GATE CALIBRATION — Find optimal regime filter thresholds from test set.
Run AFTER training (3_v4_xgboost2.py) and BEFORE live inference.

Usage: python 5_calibrate_gate.py
"""

import numpy as np
import xgboost as xgb
import pickle
import json
from pathlib import Path
from tqdm import tqdm
from itertools import product

# Paths — same as training script
DATASET_DIR  = "./dataset_xgb_regression_v4"
MODEL_PATH   = "./model_output_xgb_v4/model_sniper_v4.json"
SCALER_PATH  = "./scaler_regression_v4.pkl"
KEEPIDX_PATH = "./model_output_xgb_v4/keep_idx.pkl"
OUTPUT_PATH  = "./model_output_xgb_v4/gate_config.json"


def main():
    # Load artifacts
    model = xgb.Booster()
    model.load_model(MODEL_PATH)

    with open(SCALER_PATH, 'rb') as f:
        scaler_data = pickle.load(f)
    scaler_mean = scaler_data['mean']
    scaler_std  = scaler_data['std']

    with open(KEEPIDX_PATH, 'rb') as f:
        keep_idx = pickle.load(f)

    manifest_path = Path(DATASET_DIR) / "manifest.json"
    with open(manifest_path) as f:
        manifest = json.load(f)

    test_files = manifest.get('test', [])
    if not test_files:
        print("[ERROR] No test files found. Did you set TEST_SPLIT > 0 and re-run dataset generation?")
        return

    # Collect predictions on test set
    all_features_raw = []  # un-normalized features for gate analysis
    all_y_true = []
    all_y_pred = []

    for f in tqdm(test_files, desc="Loading test set"):
        data = np.load(f)
        X = data['X'].astype(np.float32)
        y = data['y'].astype(np.float32)

        X[np.isinf(X)] = np.nan
        y[np.isinf(y)] = np.nan
        valid = ~np.isnan(y)
        X, y = X[valid], y[valid]

        # Reverse normalization to get raw feature values (gate uses raw features)
        X_raw_file = X * (scaler_std + 1e-8) + scaler_mean
        all_features_raw.append(X_raw_file)

        # .npz already contains normalized data — select columns and predict directly
        X_sel = X[:, keep_idx]
        X_norm = np.nan_to_num(X_sel, nan=0.0, posinf=0.0, neginf=0.0)

        dmat = xgb.DMatrix(X_norm)
        preds = model.predict(dmat)

        all_y_true.append(y)
        all_y_pred.append(preds)

    X_raw = np.vstack(all_features_raw)
    y_true = np.concatenate(all_y_true)
    y_pred = np.concatenate(all_y_pred)

    print(f"\nTest set: {len(y_true):,} samples")

    # Ungated baseline
    moves = np.abs(y_true) > 1e-5
    if moves.sum() > 0:
        baseline_acc = float(np.mean(np.sign(y_pred[moves]) == np.sign(y_true[moves])))
    else:
        baseline_acc = 0.5
    print(f"Ungated directional accuracy: {baseline_acc:.4f}")

    # Feature index 0 = price_volatility_300s (in the FULL 12-feature vector)
    volatility = X_raw[:, 0]

    # Compute percentiles for volatility
    vol_percentiles = np.percentile(volatility, [10, 20, 30, 40, 50, 60, 70, 80, 90, 95])
    print(f"\nVolatility percentiles:")
    for p, v in zip([10, 20, 30, 40, 50, 60, 70, 80, 90, 95], vol_percentiles):
        print(f"  {p}th: {v:.8f}")

    # Grid search over gate thresholds
    vol_low_candidates  = np.percentile(volatility, [40, 50, 60, 70])
    vol_high_candidates = np.percentile(volatility, [85, 90, 95, 99])
    pred_bps_candidates = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.0, 10.0]

    best_score  = -1
    best_config = None

    n_combos = len(vol_low_candidates) * len(vol_high_candidates) * len(pred_bps_candidates)
    print(f"\nSearching {n_combos} threshold combinations...")

    pred_bps_arr = np.abs(y_pred) * 10_000

    for vl, vh, min_bps in product(vol_low_candidates, vol_high_candidates, pred_bps_candidates):
        if vl >= vh:
            continue

        mask = (
            (volatility >= vl) &
            (volatility <= vh) &
            (pred_bps_arr >= min_bps) &
            moves  # only count rows where actual return != 0
        )

        n_pass = mask.sum()
        if n_pass < 100:  # need minimum sample size
            continue

        acc = float(np.mean(np.sign(y_pred[mask]) == np.sign(y_true[mask])))

        # Score: expected total profit = edge_per_trade * n_trades
        # Only consider configs with acc > 51% (above noise floor)
        if acc > 0.51:
            avg_move_bps = float(np.mean(np.abs(y_true[mask])) * 10_000)
            edge_per_trade = (2 * acc - 1) * avg_move_bps  # expected bps per trade
            score = edge_per_trade * n_pass               # total expected profit in bps
        else:
            score = 0.0

        if score > best_score:
            best_score = score
            best_config = {
                'vol_low':        float(vl),
                'vol_high':       float(vh),
                'min_pred_bps':   float(min_bps),
                'gated_accuracy': float(acc),
                'gated_trades':   int(n_pass),
                'pass_rate':      float(n_pass / len(y_true)),
                'score':          float(score),
            }

    if best_config is None:
        print("[ERROR] No valid gate configuration found. Using defaults.")
        return

    # Compute expected daily trades
    # Test set subsampled at 60s: ~1440 samples/day
    samples_per_day = 1440  # 24h * 60min / 1min per sample
    est_trades_per_day = best_config['pass_rate'] * samples_per_day

    print(f"\n{'='*60}")
    print(f"BEST GATE CONFIGURATION")
    print(f"{'='*60}")
    print(f"  Volatility range : [{best_config['vol_low']:.8f}, {best_config['vol_high']:.8f}]")
    print(f"  Min prediction   : {best_config['min_pred_bps']:.1f} bps")
    print(f"  Gated accuracy   : {best_config['gated_accuracy']:.4f} (baseline: {baseline_acc:.4f})")
    print(f"  Gated trades     : {best_config['gated_trades']:,} / {len(y_true):,}")
    print(f"  Pass rate        : {best_config['pass_rate']*100:.1f}%")
    print(f"  Est. trades/day  : ~{est_trades_per_day:.0f}")
    print(f"  Score (exp.profit): {best_config['score']:.2f} bps")
    print(f"{'='*60}")

    # Build gate config for live_inference
    gate_config = {
        'vol_low':                float(best_config['vol_low']),
        'vol_high':               float(best_config['vol_high']),
        'vol_low_percentile':     'calibrated',
        'vol_high_percentile':    'calibrated',
        'min_trade_count_300s':   500,  # default, hard to calibrate from .npz
        'min_pred_bps':           float(best_config['min_pred_bps']),
        'agreement_enabled':      True,
        'directional_feature_indices': [1, 2, 5, 6, 7],
        'min_agreement_ratio':    0.6,
        'calibration_metadata': {
            'baseline_accuracy': baseline_acc,
            'gated_accuracy':    best_config['gated_accuracy'],
            'gated_trades':      best_config['gated_trades'],
            'test_samples':      len(y_true),
            'pass_rate':         best_config['pass_rate'],
        },
    }

    with open(OUTPUT_PATH, 'w') as f:
        json.dump(gate_config, f, indent=2)
    print(f"\nSaved gate config to {OUTPUT_PATH}")
    print("Now re-run live_inference.py — it will auto-load this config.")


if __name__ == "__main__":
    main()
