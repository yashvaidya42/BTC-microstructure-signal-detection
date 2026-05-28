# BTC Microstructure Signal Detection

XGBoost model that predicts 5-minute BTC/USDT price direction from tick-level trade data, with a regime-aware gating system that filters for high-confidence trading windows.

---

## Key Results (Live Test — May 28, 2026)

True out-of-sample evaluation: Binance WebSocket, data the model and gate never saw.

| Metric | Value |
|--------|-------|
| Live session duration | 7.43 hours |
| Independent non-overlapping samples | 87 |
| Directional accuracy (independent) | **62.07%** |
| Binomial test p-value | **p = 0.031** |
| Accuracy — top 10% most confident predictions | 60.08% |
| Accuracy — moves > 1 bps | 57.18% |
| Accuracy — moves > 3 bps | 58.15% |
| Lookahead bias | None (verified by independent audit) |

Accuracy scales monotonically with actual move magnitude and model confidence — the hallmark of genuine signal rather than noise.

> **Note:** This is a proof-of-concept research project. The signal is statistically significant but not yet economically viable after exchange fees. See the full independent audit in [`live_test_results/QUANT_AUDIT_REPORT.md`](live_test_results/QUANT_AUDIT_REPORT.md).

---

## How It Works

The pipeline runs in five sequential stages from raw tick data to live predictions:

| Stage | Script | Description |
|-------|--------|-------------|
| 0 | `btc_trade_data/0_download_btc_data.py` | Download raw tick trades from Binance (daily zip files) |
| 1 | `1_v4_preprocessing.py` | Compute 12 price-invariant microstructure features from tick data; generate 5-min forward log-return targets |
| 2 | `2_v4_dataset.py` | Chronological train/val/test splits (80/15/5%), fit scaler on training data only |
| 3 | `3_v4_xgboost2.py` | XGBoost training with Optuna hyperparameter search (50 trials, TimeSeriesSplit, GPU-accelerated) |
| 4 | `4_calibrate_gate.py` | Calibrate regime gate thresholds — volatility range, prediction magnitude, feature agreement |
| 5 | `5_live_inference.py` | Live inference via Binance WebSocket with real-time predictions; broadcasts to dashboard |

### Regime Gate

The model generates a prediction every 10 seconds but only flags a **TRADE** signal when four independent conditions are met:

1. **Volatility gate** — price volatility is in the productive range (not too quiet, not extreme)
2. **Volume gate** — at least 500 trades in the last 5 minutes (liquid market)
3. **Magnitude gate** — predicted move exceeds minimum threshold (filters noise)
4. **Feature agreement gate** — ≥60% of directional features agree on direction

This reduces the raw prediction stream to approximately 212 actionable signals per day (~15% pass rate).

---

## Features (12 Price-Invariant Signals)

All features are computed as causal rolling statistics over the past 300 seconds of tick data. The 5-minute VWAP and 15-minute SMA windows use longer lookbacks.

| Feature | What It Measures | Category |
|---------|-----------------|----------|
| `price_volatility_300s` | Std dev of log-returns over 5 min | Volatility |
| `vol_imbalance_300s` | (buy volume − sell volume) / total volume | Microstructure |
| `vwap_dev_300s` | Price deviation from 5-min VWAP | Mean-reversion |
| `intensity_z_300s` | Trade arrival rate, z-scored vs recent history | Microstructure |
| `feat_vol_force` | Volume × price change (momentum force, 60s) | Microstructure |
| `feat_sma_trend` | Price deviation from 15-min SMA | Mean-reversion |
| `feat_rsi_norm` | Tick-level RSI, normalized to [−1, +1] | Mean-reversion |
| `tfi_quote_norm_300s` | Trade flow imbalance normalized by VWAP | Microstructure |
| `amihud_illiquidity_300s` | Price impact per unit BTC traded (Amihud ratio) | Microstructure |
| `rel_size_300s` | Avg trade size relative to 1h rolling mean | Microstructure |
| `inter_time_cv_300s` | Coefficient of variation of inter-trade arrival times | Volatility |
| `size_cv_300s` | Coefficient of variation of trade sizes | Microstructure |

All features are made price-invariant: TFI is normalized by VWAP, Amihud uses BTC quantity (not USD), relative size uses a 1h rolling baseline. This ensures the model generalizes across price levels.

---

## Project Structure

```
quant_ML_model/
├── btc_trade_data/
│   └── 0_download_btc_data.py      # Download Binance daily trade zips
├── 1_v4_preprocessing.py           # Tick data → 12 features + target (Numba-accelerated)
├── 2_v4_dataset.py                 # Chronological splits + StandardScaler
├── 3_v4_xgboost2.py                # Optuna HPO + XGBoost training (GPU)
├── 4_calibrate_gate.py             # Regime gate threshold calibration
├── 5_live_inference.py             # Live Binance WebSocket inference engine
├── live_dashboard.html             # Real-time price + prediction chart (browser)
├── PROJECT_HISTORY.md              # Development log and version history
├── model_output_xgb_v4/
│   ├── model_sniper_v4.json        # Trained XGBoost model
│   ├── gate_config.json            # Calibrated regime gate thresholds
│   └── artifacts.json             # Training metadata and feature importance
└── live_test_results/
    ├── QUANT_AUDIT_REPORT.md       # Full independent statistical audit
    ├── predictions_*.csv           # All predictions made during live test
    └── resolved_predictions_*.csv  # Predictions with realized outcomes
```

---

## Quick Start

### Requirements

```bash
pip install -r requirements.txt
```

Core dependencies: `xgboost`, `optuna`, `numpy`, `pandas`, `numba`, `pyarrow`, `scikit-learn`, `websockets`, `requests`, `scipy`

### Pipeline

```bash
# 1. Download raw data (2026-01-01 to present, ~4 GB)
python btc_trade_data/0_download_btc_data.py

# 2. Compute features and targets from tick data
python 1_v4_preprocessing.py

# 3. Create normalized train/val/test splits
python 2_v4_dataset.py

# 4. Train model (GPU recommended — runs on RTX 3050+)
python 3_v4_xgboost2.py

# 5. Calibrate regime gate
python 4_calibrate_gate.py

# 6. Start live inference
python 5_live_inference.py

# 7. Open dashboard in browser
#    open live_dashboard.html
```

**Hardware notes:**
- Training (steps 2–5): CUDA GPU recommended. Tested on RTX 3050 Laptop GPU (4 GB VRAM). Falls back to CPU automatically.
- Preprocessing (step 2): CPU-only, parallelized. Tested on i5-11400H.
- Live inference (step 6): CPU only, low resource usage (~1 core).

---

## Honest Assessment

| Claim | Status |
|-------|--------|
| Statistically significant signal (p < 0.05) | Yes — p = 0.031 on independent samples |
| Accuracy improves with move magnitude | Yes — hallmark of real signal |
| No lookahead bias | Confirmed by independent audit |
| Economically viable after fees | **No** — ~1.3 bps gross vs 10–20 bps round-trip fees |
| Regime-independent | **No** — works in Asian session, degrades at European open |
| Sufficient data for strong conclusion | **No** — need 200+ independent samples (~16+ hours) |

The model detects a genuine microstructure signal. The signal is too small to overcome exchange fees at current prediction frequency. The logical next step is either reducing trade frequency (gate more aggressively to select only the highest-edge moments) or extending the test to 16+ hours to characterize the edge more precisely.

**This is a research project and proof of concept. Nothing here is trading advice.**

---

## Full Audit

[`live_test_results/QUANT_AUDIT_REPORT.md`](live_test_results/QUANT_AUDIT_REPORT.md) contains the complete independent statistical audit including:

- Code review (lookahead bias, data leakage, train/test contamination)
- Optuna and training log analysis
- Permutation tests, binomial tests, Durbin-Watson autocorrelation analysis
- Hourly accuracy breakdown (time-of-day regime dependency)
- Economic viability analysis with fee modeling
- Sample size and power analysis
- Verdict and recommendations

---

## License

MIT
