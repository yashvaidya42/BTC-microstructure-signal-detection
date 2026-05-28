# PROJECT HISTORY — BTC 5-Min Forward Return Prediction

## 1. Pipeline Evolution

### V1 — Classification (Deleted)
- **Target**: 3-class classification (UP / NEUTRAL / DOWN)
- **Result**: Failed. Classification labels discretize continuous returns, destroying the gradient signal that tree models need. NEUTRAL class dominated (~80%), producing a model that predicted NEUTRAL on everything.
- **Lesson**: Return prediction is fundamentally a regression problem.

### V2 — Regression + Walk-Forward (`v2/`)
- **Hardware**: i5-11400H, 8 GB RAM
- **Target**: 5-minute forward log-return (regression)
- **Features**: 12 "Champion" features at 300s window — pruned from a kitchen-sink of ~60 features across multiple time scales. The audit proved that multi-scale windows added noise, not signal. Single 300s window dominated.
- **Key features**: `inter_time_std` (panic detector, #1), `price_volatility` (#2), `vol_imbalance`, `tfi_quote`, `amihud_illiquidity`, `vwap_dev`, `intensity_z`, plus size and speed metrics.
- **Objective**: Custom `sniper_objective` — MAE gradient with 5x penalty when the model predicts the wrong direction on clear moves (>10 bps). Designed to bias the model toward directional accuracy on large moves rather than minimizing MAE on noise.
- **Training**: Walk-Forward incremental — evaluate chunk BEFORE training on it, producing true out-of-sample metrics at every step. Early stopping on held-out validation.
- **Optuna**: TimeSeriesSplit(n_splits=3) with purge gap. "Tank config" — shallow trees (3-5), high regularization (alpha/lambda 5-20), heavy subsampling.
- **Infrastructure**: Smart Size-Based Scheduler (batches by MB, not count), RAM hard-stop at 85%, Numba JIT rolling primitives.
- **Result**: Macro F1 ~0.41-0.43 (per git log). Directional accuracy above random but not tradeable. Walk-forward showed regime sensitivity.

### V3 — Quantile Experiment (`v3/`)
- **Change**: Trained 3 quantile models (q10, q50, q90) instead of single point estimate.
- **Preprocessing**: Identical to V2.
- **Result**: Added complexity without meaningful improvement. Per-fold collapse pruning helped Optuna stability but quantile approach was abandoned.

### V4 — GPU + Price-Invariant Features (Current: `1_v4_*.py`, `2_v4_*.py`, `3_v4_xgboost2.py`)
- **Hardware**: i9-14900K, 128 GB RAM, 2x NVIDIA A6000 (96 GB VRAM total)
- **Features**: 15 features (7 kept from v3, 3 modified, 2 new CV ratios, 4 time features):
  - **Price-invariance (P2)**: `tfi_quote` normalized by VWAP, `amihud` uses qty instead of quote_qty, `avg_size` normalized by 1h rolling mean → features no longer change meaning when BTC price doubles.
  - **CV ratios (P7)**: Replaced redundant mean+std pairs with coefficient-of-variation (`inter_time_cv`, `size_cv`) — same information, half the features, already scale-invariant.
  - **Time features (P1)**: `hour_sin/cos`, `dow_sin/cos` — cyclic encoding of UTC time-of-day and day-of-week.
  - **Removed weight (P3)**: Weight = abs(target) double-penalized large moves when combined with sniper objective.
- **Training**: Single-model training via `ExtMemQuantileDMatrix` (XGBoost 3.0+) — streams through all data files without loading into RAM.
- **GPU Objective**: CuPy-accelerated sniper_objective_v4 — chunked to 25M rows per GPU transfer to fit VRAM. Huber-like gradient (smooth near zero) with 5x directional penalty on wrong-direction predictions for moves > 3 bps.
- **Optuna**: Pre-built GPU QuantileDMatrix folds (built once, reused across all trials). TPE sampler, median pruner, wider search space than V2.
- **Infrastructure**: Crash recovery (CrashRecovery class dumps Optuna state + model on any failure/interrupt), dual logging (console + file), GPU auto-detection with CPU fallback, PyArrow CSV loading (2-3x faster than pandas).

### `without_sniper.py` — Ablation Variant
- Identical to `3_v4_xgboost2.py` but uses XGBoost's built-in `reg:absoluteerror` instead of custom sniper objective during both Optuna and training. Exists to answer: "Does the sniper objective actually help, or would standard MAE do the same?"

---

## 2. Dataset Requirements

- **Source**: Binance BTCUSDT spot tick trades, downloaded as daily ZIP archives from `data.binance.vision`.
- **Downloader**: `data/download_btc_trades_2025.py` — fetches daily ZIPs, skips existing files.
- **Format**: Each ZIP contains a headerless CSV: `trade_id, price, qty, quote_qty, timestamp_ms, is_buyer_maker, is_best_match`.
- **Volume**: ~3-5 million ticks/day at current BTC activity. ~40 GB/year compressed.
- **Preprocessing**: Each daily ZIP is loaded → side computed from `is_buyer_maker` → Numba JIT rolling windows compute 15 features → 5-min forward log-return target → saved as Parquet.
- **Day boundary handling**: Last 15 minutes of each day's raw data is carried as a "tail" to the next day for rolling window continuity. Boundary correction re-computes the first 10 minutes of each parallel batch using the prior batch's tail.

---

## 3. Core Hypothesis

### Why the Sniper Objective

Standard regression (MSE/MAE) treats all errors equally. A 0.1 bps error and a 5 bps error on a large move contribute the same to the loss. This produces "cowardly" models that predict near-zero for everything (minimizing MAE by hugging the mean).

The sniper objective introduces asymmetric penalties:
1. **Huber-like gradient**: Smooth near zero (L2), capped far from zero (L1). Prevents outlier moves from dominating training.
2. **Directional penalty**: When the model predicts the wrong direction on a "clear move" (|y_true| > 3 bps in v4), gradient and hessian are multiplied by 5x. This forces the model to "get the direction right or pay dearly."

**The hypothesis**: A model with 52% directional accuracy on the top-10% most confident predictions is more valuable than a model with 50.5% accuracy on all predictions but lower confidence dispersion. The sniper objective sacrifices global MAE to concentrate accuracy where it matters most.

### Why Price-Invariant Features

BTC price doubled in 2024. Features like raw `tfi_quote` (dollar volume imbalance) or `amihud_illiquidity` (|return|/dollar_volume) change meaning as the price level shifts — the same feature value means different things at $30K vs $100K. V4 normalizes these by VWAP or uses base-quantity denominators, making features stationary across price regimes.

### Where This Can Fail

1. **Regime breaks**: The model learns microstructure patterns from historical data. If Binance's matching engine changes, fee tiers shift, or a major market structure event occurs, learned patterns may invert.
2. **Latency**: Features are computed on 300s rolling windows. In a flash crash, the model won't react until the window fills with panic data — by which point the move is over.
3. **Overfitting to noise**: 3 bps threshold for "clear moves" is very small — this may be bid-ask bounce, not signal. V2 used 10 bps.
4. **Time features**: BTC trades 24/7 globally. Hour-of-day patterns (Asian session, US session) exist but are weak and may not generalize.
5. **Single-model vs Walk-Forward**: V4 dropped walk-forward for simplicity, but a single model can't adapt to regime shifts during the test period.

---

## 4. Honest Critique of V4 Architecture

### Strengths
- **Price-invariant features** are a genuine improvement — the model's predictions should be more stable across different price levels.
- **CV ratios** are elegant — same information content as mean+std pairs with half the parameters and built-in scale invariance.
- **ExtMemQuantileDMatrix** is the right approach for datasets that exceed RAM. The batched iterator pattern is clean.
- **Crash recovery** is production-grade — Optuna state, model checkpoints, and error logs are preserved on any failure.
- **GPU-chunked objective** is well-engineered — 25M row chunks keep VRAM usage bounded while maximizing throughput.

### Weaknesses & Technical Debt

1. **`live_inference.py` is incompatible with V4.** It computes 13 v3 features, not 15 v4 features. Model artifact paths, feature formulas (amihud, TFI), and the feature ordering are all wrong. **This must be fixed before any live testing.**

2. **`without_sniper.py` is a full copy of `3_v4_xgboost2.py`.** 1374 lines duplicated for a single flag change. Should be a command-line argument or config toggle.

3. **`4_model_test.py` points to v3 artifacts** (`model_artifacts/model_sniper_final.json`, `scaler_regression.pkl`). Hardcoded Windows paths. Will not work with v4 model output.

4. **Dead directories**: `v2/`, `v3/`, `study/`, `backup local/` contain superseded code. They add confusion without value.

5. **34/34/32 train/val/test split** in `2_v4_dataset.py` is unusual. Only 34% of data is used for training — this wastes data. Standard time-series split is 70/15/15 (as V2 used). The config comments still say "70% Train, 15% Val, 15% Test" but the actual values don't match.

6. **Sniper threshold drift**: V2 used `|y_true| > 0.001` (10 bps) for clear moves. V4 uses `|y_true| > 0.0003` (3 bps). This 3x reduction means the directional penalty fires on bid-ask bounce noise, potentially causing the model to overfit to micro-noise.

7. **Vestigial config**: `WINDOWS = [300]` and `LAG_STEPS = []` in preprocessing are config artifacts from the kitchen-sink era. The code no longer iterates over windows — it's hardcoded to `w = 300`.

8. **No feature importance feedback loop**: V2 had pilot feature selection (train a small model, drop bottom N%). V4 removed this, relying only on correlation filtering. The correlation filter (threshold=0.90) may be too aggressive — features with 0.91 correlation could carry complementary information that tree splits exploit.

9. **Numba `_precompute_base_metrics`** in v4 preprocessing is a single JIT function that computes `time_delta, log_return, buy_volume, sell_volume, price_diff`. The live engine computes these with vectorized NumPy instead. Not a bug (NumPy is fine for small buffers) but a divergence that makes it harder to verify feature parity.

---

## 5. File Map (Current V4)

| File | Purpose |
|------|---------|
| `1_v4_preprocessing.py` | Raw tick ZIPs → daily Parquets with 15 features + target |
| `2_v4_dataset.py` | Parquets → train/val/test .npz splits with scaler |
| `3_v4_xgboost2.py` | Optuna + GPU training + test evaluation (sniper objective) |
| `without_sniper.py` | Ablation: same as above but `reg:absoluteerror` objective |
| `4_model_test.py` | Independent model audit with plots (currently v3 paths) |
| `live_inference.py` | Real-time Binance WS → V4 feature computation → prediction (FIXED 2026-05-21) |
| `live_inference_v3_backup.py` | Backup of original v3-era live_inference.py |
| `live_dashboard.html` | Browser dashboard for live predictions (Lightweight Charts) |
| `data/download_btc_trades_2025.py` | Binance historical trade downloader |

---

## 6. Live Test Results (2026-05-21)

### Setup
- **Model**: `model_output_xgb_v4/model_sniper_v4.json` (989 trees, depth 4, trained with `reg:absoluteerror`)
- **Scaler**: `scaler_regression_v4.pkl` (16 features, z-score normalization)
- **Hardware**: i5-11400H, 16 GB RAM, no GPU needed for inference
- **Duration**: 13:13 → 13:28 UTC (15 min of active predictions within 30-min session)
- **Total predictions**: 83 (every ~11s)
- **Log file**: `live_test_results/predictions_20260521_131315.csv`

### Key Metrics

| Metric | Value | Training Baseline |
|--------|-------|-------------------|
| Predictions | 83 | — |
| Mean \|prediction\| | 0.59 bps | 0.71 bps (test MAE) |
| Prediction std | 3.37e-05 | 5.91e-05 |
| Prediction mean | +5.55e-05 | -4.69e-06 |
| Confident (>0.5 bps) | 74.7% | — |
| Near-zero (<0.1 bps) | 13.3% | — |
| Direction split | 88% UP / 12% DN | — |
| **Retrospective directional accuracy** | **85.5% (47/55)** | 93.4% (test set) |

### Interpretation

1. **The model is not collapsed.** Predictions have meaningful dispersion (std 3.37e-05 vs training 5.91e-05) and 74.7% of predictions exceed 0.5 bps — the model is making confident calls, not hugging zero. This confirms the sniper/MAE objective is working as intended.

2. **Retrospective directional accuracy of 85.5%** is strong. Of the 55 predictions where 5 minutes of future price data was available within the session, 47 correctly predicted the direction. However, this is measured during a trending UP session (+24 bps over 15 min) with an UP-biased model, so it's inflated.

3. **Directional bias is a concern.** 88% UP predictions during a session where BTC moved UP 24 bps could mean the model is genuinely reading microstructure bullishness — or it could mean the model has a persistent UP bias. A fair test needs a DOWN or FLAT session to distinguish. The live prediction mean (+5.55e-05) is an order of magnitude larger than the training test mean (-4.69e-06), suggesting some bias.

4. **Prediction magnitude is conservative.** Mean |prediction| of 0.59 bps is below the training test MAE of 0.71 bps, suggesting the model is slightly more conservative on live data — possibly because live feature distributions differ from training (newer price regime, different volatility).

5. **Infrastructure passed.** Buffer grew from 22K to 49K ticks, inference ran at a stable ~11s interval (target: 10s), no crashes, no memory issues, Binance WS stayed connected. The REST backfill correctly seeded 22,675 trades (3622s span) before first prediction.

### What Was Fixed for This Test

The original `live_inference.py` computed 13 v3 features — incompatible with the 16-feature V4 model. Six specific bugs were fixed:

1. TFI: raw dollar imbalance → VWAP-normalized (`tfi / vwap_300s`)
2. Amihud: `quote_qty` denominator → `qty` denominator
3. Missing: `inter_time_cv_300s`, `size_cv_300s` (CV ratios) — added
4. Missing: `rel_size_300s` (trade size vs 1h rolling mean) — added
5. Missing: `hour_sin/cos`, `dow_sin/cos` (time features) — added
6. Artifact paths: v3 paths → `model_output_xgb_v4/` paths

Backup of original: `live_inference_v3_backup.py`

### Recommendations for Next Test

1. **Run during a DOWN session** to test whether the UP bias is real or contextual.
2. **Run for 2+ hours** so enough predictions have full 5-min forward resolution for proper accuracy measurement.
3. **Log actual price at T+5min** by adding a deferred lookup — current CSV only has the prediction, not the realized return, so retrospective checks require price interpolation across rows.
4. **Install Numba** (`pip install numba`) — the test ran without it, which means features were computed with pure Python loops instead of JIT-compiled code. This is fine for correctness but adds latency.
5. **Test with the dashboard** — open `live_dashboard.html` during the session to verify the WebSocket push and chart rendering.
