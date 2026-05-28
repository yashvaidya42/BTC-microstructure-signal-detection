# QUANTITATIVE AUDIT REPORT — BTC 5-Min Prediction Model (v4)
**Date:** 2026-05-28 | **Auditor:** Independent Quant (30yr systematic trading)
**Live Session:** 2026-05-28 03:29:32 → 10:55:36 UTC (7.43 hours)
**Null hypothesis: This model is guessing.**

---

## PHASE 1: Code Audit

### 1.1 Target Computation (`1_v4_preprocessing.py`)

**Verdict: CLEAN. No lookahead bias in target.**

`_find_future_price(ts_us, prices, look_ahead_us)` uses `np.searchsorted` on the same day's timestamp array to find the first tick at or after `t + 300s`. This is the correct forward price for a 5-min log-return target. It is not used as a feature — only as the label to predict.

The feature rolling windows use `while left < r and ts_us[left] < lo` — strictly causal, window `[t-300s, t]` inclusive. No future data leaks into features.

**One minor concern:** `intensity_z_300s` is computed as a z-score of `count_w` using `mean_cnt` and `std_cnt` computed over the last 300s of `count_w` values. Since the current tick's own count is included in its window, there is marginal self-reference — but this is not exploitable and is present identically in live inference.

### 1.2 Feature List (12 features, v4)

All features are causal rolling statistics over the past 300 seconds (plus `feat_sma_trend` which uses 900s, and `rel_size_300s` which uses 3600s for normalization). The 1h normalization for `rel_size_300s` means the first hour of each day has a biased normalizer — but this is consistent between training and live inference (both start with a cold buffer). **No lookahead.**

### 1.3 Scaler / Data Leakage (`2_v4_dataset.py`)

```python
mean, std = compute_statistics_smart(splits[0][1], features)  # train files ONLY
scaler = SimpleScaler(mean, std)
```

Scaler is fit on training data only, then applied to val and test. **CLEAN.**

### 1.4 Train/Val/Test Split

Chronological split on sorted daily parquet files: **80% train / 15% val / 5% test.**

```
Total files: 146 (2026-01-01 to 2026-05-26)
Train: 116 files | Val: 21 files | Test: 9 files
```

Files are named `BTCUSDT-trades-YYYY-MM-DD.zip` and sorted lexicographically, which equals chronological order. No future data leaks into training. **CLEAN.**

Total rows: 209,510 at 60s subsampling (1 row/minute per day).

### 1.5 Training (`3_v4_xgboost2.py`)

Optuna hyperparameter search uses `TimeSeriesSplit(n_splits=2, gap=5)` — gap of 5 rows = 300s purge between train and validation in each fold. This prevents the target horizon from overlapping train/val. **CORRECT.**

The `reg:absoluteerror` objective (MAE loss) with `base_score=0.0` means the model's implicit null prediction is 0. With near-zero mean targets (mean=-0.000003), this is correct.

**Critical observation:** The validation MAE is reported as `0.00065` (6.5 bps) from round [0] through round [367] with essentially no visible improvement. The model's best iteration was 267, with early stopping firing 100 rounds later at 367. This means the model found its optimum very early and barely improved. Combined with heavily regularizing hyperparameters (gamma=1.99, min_child_weight=1745, max_depth=3), this is a conservative model that barely moves from its prior.

### 1.6 Gate Calibration (`4_calibrate_gate.py`) — RED FLAG

**This is the most significant methodological problem.**

The regime gate thresholds are calibrated via grid search over 160 combinations directly on the **held-out test set**. The test set is then reported to have 55.05% gated accuracy vs 52.34% ungated baseline. This improvement cannot be trusted — the gate has been optimized to maximize accuracy on the exact data used to measure it.

The "gated accuracy" of 55.05% on the test set is **the best of 160 configurations searched on that same data.** It is an in-sample metric masquerading as out-of-sample performance.

The live test, however, uses data the model and gate never saw — so the live results are genuinely out-of-sample.

**Note:** The gate status (TRADE/SKIP) is NOT logged in the predictions CSV, making it impossible to separately analyze gated vs non-gated performance in the live data.

### 1.7 Live Inference (`5_live_inference.py`)

**Feature computation:** The Numba functions are copy-pasted verbatim from `1_v4_preprocessing.py`. Column order matches training exactly. **CLEAN.**

**Timestamp handling:** REST and WebSocket both provide millisecond timestamps (`data['T']`). Converted to microseconds via `ts_us = ts * 1000`. Training used microsecond timestamps natively from Binance zip files. **Consistent.**

**Buffer warmup:** `MIN_BUFFER_SPAN_SEC = 900` (15 min) required before predicting, `BUFFER_MAX_SECONDS = 3600` (1h max buffer). This handles the 1h rolling window for `rel_size_300s`. However, the buffer evicts data older than 1h, so during a long session (as observed: 7.43h), the 1h rolling mean is fully populated after the first hour. **Consistent with training.**

**Resolution:** `realized = math.log(tick_price / price_at_pred)` where `tick_price` is the first market tick arriving at or after `t + 300s`. This exactly mirrors the training target. **CLEAN.**

---

## PHASE 2: Training Log Audit

### 2.1 Optuna

| Metric | Value |
|--------|-------|
| Trials attempted | 50 |
| Trials completed | 39 |
| Trials pruned | 11 (22%) |
| Collapses (pred_std ≈ 0) | 0 |
| Best CV MAE | 10.12 bps |
| Best params | max_depth=3, lr=0.011, gamma=1.99, reg_alpha=2.48, reg_lambda=1.22, min_child_weight=1745, subsample=0.77, colsample=0.54, rounds=2768 |

**Hyperparameter assessment:** gamma=1.99 (very high — tree splits must reduce loss by ≥2 bps to be made) and min_child_weight=1745 (very high — minimum 1745 samples per leaf) together produce an extremely regularized shallow forest. This is the expected behavior for low-SNR financial data. These hyperparameters are plausible and not obviously pathological.

**11 pruned trials (22%)** is normal with MedianPruner after 15 startup trials. No red flag.

### 2.2 Training Run

| Metric | Value |
|--------|-------|
| Train samples | 166,460 |
| Val samples | 30,135 |
| Test samples | 12,915 |
| Max rounds | 2,768 |
| Best iteration | 267 |
| Total trees | 368 |
| Early stopping fired | Yes, at round 367 (100 rounds after best=267) |

**Early stopping:** Fired at round 367, best at 267. The model stopped improving after round 267. This is NOT "very early" stopping — 267 rounds at depth-3 is a legitimate forest. However, the val-MAE was essentially **flat from round 0** (`0.00065` reported consistently), suggesting the model found its optimum almost immediately and only marginal gains exist. This is typical for near-noise data.

### 2.3 Test Set Performance

| Metric | Test Set |
|--------|----------|
| MAE | 6.53 bps |
| Directional accuracy | 52.48% (on 11,280 real moves) |
| Sniper accuracy (top 10%) | 57.20% (1,292 trades) |
| Mean PnL/trade | +0.26 bps |
| Pred mean | 0.000001 |
| **Pred std** | **0.000043 (0.43 bps)** |

**Prediction distribution is dangerously compressed.** Target std on training data is 0.0016 (16 bps). Prediction std is 0.000043 (0.43 bps). **Ratio: 0.027.** The model outputs values almost exclusively between -4.3 bps and +4.3 bps (≈99th percentile). This is a **sign-only model** — it outputs tiny magnitudes and relies entirely on getting the direction right. Any magnitude-based decision threshold (min_pred_bps) is largely decorative.

### 2.4 Feature Importance

| Rank | Feature | Gain |
|------|---------|------|
| 1 | feat_sma_trend (15-min deviation from SMA) | 27.55 |
| 2 | vwap_dev_300s (deviation from 5-min VWAP) | 15.97 |
| 3 | tfi_quote_norm_300s (normalized trade flow imbalance) | 13.81 |
| 4 | feat_rsi_norm (normalized RSI) | 12.58 |
| 5 | intensity_z_300s | 10.27 |

**The top 4 features are all mean-reversion signals** (price above SMA → predict down, price above VWAP → predict down, overbought RSI → predict down). The model is a mean-reversion predictor. Features 6-10 have gains of 8.75–10.21, suggesting reasonable diversification across feature types. No single feature dominates catastrophically.

---

## PHASE 3: Live Results Analysis

### 3.A Basic Statistics

| Metric | Value |
|--------|-------|
| Total predictions | 2,589 |
| Resolved predictions | 2,559 |
| Unresolved predictions | 30 |
| Live session duration | 7.43 hours |
| Prediction mean | -0.0748 bps (slight DN bias) |
| Prediction std | 0.85 bps |
| Prediction min | -3.46 bps |
| Prediction max | +2.91 bps |
| Percentiles (5/25/50/75/95) | -1.60, -0.35, -0.07, +0.35, +1.02 bps |

**One-sample t-test on predictions (H0: mean=0):** t=-4.49, **p=0.000008.**
The model has a statistically significant downward bias during this session. 55.7% of all predictions are DN, 44.3% UP. This could reflect genuine market structure (BTC fell during the session) or a systematic bias in the features. BTC opened at ~$74,495 and was at ~$72,983 at session end (a -2% move), so a sustained DN bias is directionally plausible.

### 3.B Directional Accuracy

| Condition | Correct | Total | Accuracy |
|-----------|---------|-------|----------|
| All non-flat | 1,444 | 2,545 | **56.74%** |
| Real moves (>1 bps actual) | 1,362 | 2,382 | **57.18%** |
| Large moves (>3 bps actual) | 1,184 | 2,036 | **58.15%** |
| Top 10% by \|pred\| (>1.43 bps) | 152 | 253 | **60.08%** |

**The accuracy improves with both actual move size AND model confidence.** This is the hallmark of genuine signal — a model that is more right when the market actually moves, and more right when it is more confident. A pure guesser would show flat accuracy across all these cuts.

**Binomial test (overall, two-sided): p < 0.000001.** Highly significant, but this is misleading due to massive autocorrelation — see Section 3.C.

### 3.C Non-Overlapping (Independent) Accuracy — THE ONLY NUMBER THAT MATTERS

Predictions are made every 10 seconds. The target is 5-minute forward returns. Consecutive predictions overlap by up to 290 seconds of the 300-second horizon. The effective sample size is far smaller than 2,545.

**Lag-1 autocorrelation of the correct/wrong series: 0.777.** This is massive. Consecutive predictions share nearly identical outcomes because they're resolving against almost identical future windows. The Durbin-Watson statistic is 0.193 (vs 2.0 for independence). **The 2,545 predictions contain approximately 87 truly independent observations.**

**Independent samples (one per 5-min window, non-flat): 54/87 = 62.07%**
- Binomial test (two-sided): **p = 0.031**
- Binomial test (one-sided, greater than 50%): **p = 0.016**

**This IS statistically significant at the 5% level, on truly independent samples.** The null hypothesis of 50% accuracy can be rejected, but barely.

**Power analysis:**
| Target accuracy | Samples needed (p<0.05, power=80%) | Have |
|-----------------|-------------------------------------|------|
| 52% | 4,904 | 87 |
| 55% | 783 | 87 |
| 60% | 194 | 87 |

We have 87 samples. To confirm that 62% accuracy is real (not a 7.43-hour lucky run), we need 194 independent samples — roughly **16 hours** of additional live testing at current prediction frequency.

### 3.D Regime Gate Performance

**The gate status was not logged in the output CSV files.** The prediction logger (`PredictionLogger`) writes all predictions regardless of gate decision, and `ResolvedLogger` resolves all predictions regardless of gate. There is no TRADE/SKIP column in any of the output files.

**Consequence:** We cannot separately evaluate gated vs non-gated accuracy from the live data. This is a logging gap. The gate's live performance is unknown.

**What we know from calibration:** The gate was calibrated to pass 14.7% of test set samples (est. ~212 trades/day). At 10s prediction intervals, the live session generated ~2,589 predictions, so approximately 380 TRADE signals would be expected over 7.43 hours.

### 3.E Profit Analysis

| Metric | Value |
|--------|-------|
| Cumulative gross PnL (all predictions) | **+3,309 bps** |
| Mean gross PnL per prediction | **+1.29 bps** |
| Std PnL per prediction | 14.22 bps |
| Annualized Sharpe (360 pred/hr) | 32.96 |

**The Sharpe ratio of 32.96 is an artifact and should be ignored.** It is computed assuming 2,589 predictions × 365/7.43h annual scaling with mean/std from highly autocorrelated data. The effective sample size is 87, which yields a Sharpe of approximately:
`(1.29/14.22) × sqrt(87 × 365 / (7.43/24)) ≈ 0.091 × sqrt(102,640) ≈ 29` — still too high because the formula assumes i.i.d. trades.

**PnL consistency check:**
- Total PnL: +3,309 bps
- Without top 5 trades: +3,078 bps (−7%)
- Without top 10 trades: +2,871 bps (−13%)

The PnL is NOT driven by a few lucky outliers — removing the top 10 trades still leaves 87% of the total. This is a good sign. The edge is distributed.

**However, the bottom 5 trades lost 79, 78, 77, 77, 75 bps respectively** — massive individual losses. This asymmetry is concerning: the gains are many small wins (+1.29 bps average), losses can be catastrophic in magnitude. This is the signature of a mean-reversion strategy in trending markets.

**Critical observation — hour of day:** Accuracy collapses after UTC 07:00:
| Hour UTC | Accuracy |
|----------|----------|
| 03:00 | 57.2% |
| 04:00 | 58.2% |
| 05:00 | **67.6%** |
| 06:00 | 59.8% |
| 07:00 | 62.4% |
| 08:00 | **45.8%** ← BELOW CHANCE |
| 09:00 | 49.5% |
| 10:00 | 51.9% |

The model's "edge" exists almost entirely in the early UTC session (03:00–07:00 = Asian/pre-European). It loses money in the European morning (08:00 UTC = 09:00 London). Hour 08 UTC is statistically below 50%. **This is a severe regime dependency warning.**

### 3.F Is It Luck?

**Permutation test (10,000 shuffles):**
- Observed accuracy: 56.74%
- Permutation distribution mean: 50.51%, std: 0.98%
- Real accuracy is at the **100th percentile** of the permutation distribution (0 shuffles beat it)
- Permutation p-value: **p = 0.0000**

The permutation test confirms that the observed accuracy is not explained by random prediction-outcome pairing. There is a real relationship between predictions and outcomes.

**However:** The permutation test is computed on ALL 2,545 non-flat observations, which are NOT independent. The correct permutation would block-shuffle 5-minute windows, preserving intra-window autocorrelation. The raw permutation test over-states significance. The independent-sample test (p=0.031) is the correct benchmark.

**Autocorrelation structure:**
- Lag-1 autocorrelation: 0.777
- Durbin-Watson: 0.193

When the model is correct, it tends to stay correct for extended periods (a correct streak). When it's wrong, it tends to stay wrong. This suggests the model tracks regimes (trending vs mean-reverting) rather than predicting individual 5-minute candles. During trend periods, mean-reversion predictions are systematically wrong; during quiet/choppy periods, they are systematically right.

**Do we have enough data for a conclusive test?**
- With 87 independent samples: **No.** We can reject H0 at p<0.05, but we cannot confirm the true accuracy is 62% vs 55% vs 52% with useful precision.
- 95% confidence interval for the independent accuracy: `0.621 ± 1.96 × sqrt(0.621×0.379/87)` = **[0.519, 0.723]**
- The confidence interval spans from "possibly significant" to "surprisingly strong." We need more data.

### 3.G Prediction Compression (Critical)

| Metric | Training | Live |
|--------|----------|------|
| Prediction std | 0.43 bps (test) | 0.85 bps (live) |
| Realized return std | ~16 bps (target) | **14.06 bps** |
| Ratio | 0.027 | **0.061** |

The model predicts within ±0.85 bps standard deviation while actual returns have a 14 bps standard deviation. **The model's magnitude is essentially meaningless.** Every prediction should be treated as a ±1 sign prediction, not a return forecast. The "pred_bps" column in the CSV is close to zero for all rows — the mean absolute prediction is 0.7 bps.

This creates a fundamental problem for the gate calibration: `min_pred_bps = 0.5 bps` is supposed to filter low-confidence predictions, but almost all predictions (>90th percentile threshold = 1.43 bps) are within 0–1.5 bps of zero. The gate barely filters on magnitude.

---

## PHASE 4: VERDICT

---

### Signal or Noise?
## **INCONCLUSIVE — Leaning toward weak SIGNAL**
### Confidence: LOW

---

### Evidence FOR Signal (3 strongest):

1. **Independent-sample binomial test: p=0.031.** On 87 truly non-overlapping 5-minute windows, the model called direction correctly 62% of the time (54/87). This exceeds chance at the 5% level even after controlling for the 290-second overlap between adjacent predictions. A pure guesser would need astronomical luck to achieve this.

2. **Accuracy scales correctly with move magnitude.** The model shows 56.7% on all moves, 57.2% on moves >1 bps, 58.2% on moves >3 bps, and 60.1% on the model's own top-confidence predictions. This monotonic improvement is the signature of real signal — if accuracy were noise, these slices would scatter randomly around 50%.

3. **Permutation test p=0.0000.** When prediction-outcome pairs are randomly shuffled 10,000 times, none of the permutations achieves the observed accuracy. There is a real statistical relationship between predictions and outcomes, not explainable by chance.

---

### Evidence AGAINST Signal (3 strongest):

1. **87 independent samples — far too few for a conclusive test.** The 95% confidence interval on true accuracy is [51.9%, 72.3%] — a 21-percentage-point range that spans "barely above noise" to "strong signal." We cannot distinguish these cases. The p=0.031 result is real but fragile — a single unlucky 5-minute run changes everything. The null hypothesis can be rejected but a definitive estimate of edge size cannot be made.

2. **The model is completely non-viable economically.** Mean gross PnL per prediction = 1.29 bps. Binance spot taker fees = 10 bps/side = 20 bps round-trip. Even at best-case maker fees (5 bps/side = 10 bps RT), the model loses on average **8.71 bps per trade after fees.** Only 6.3% of all predictions produce >20 bps of gross return. The signal, if real, is far too small to overcome any realistic transaction cost. This is not a tradeable strategy.

3. **Catastrophic regime dependency: hour 08 UTC has 45.8% accuracy (below chance).** The model's "edge" is confined to the Asian session (03:00–07:00 UTC). During the first 2 hours of European trading, the model performed worse than random. A model that is 67% accurate for 2 hours and 46% accurate for the next 2 hours has a structural problem — it may have learned Asian-session mean-reversion patterns that fail in a trending European open. The training data covers Jan–May 2026; if the market regime has shifted, live performance will degrade.

**Which side wins?**
The evidence leans toward "there is SOMETHING here" — the signal is real in a statistical sense. However, it is so small that it is economically irrelevant, highly regime-dependent (breaks down in trending European sessions), and based on an insufficient sample for confidence. The correct label is **INCONCLUSIVE**, not SIGNAL, because we cannot yet determine if the signal is persistent, large enough to matter, or confined to a single 7-hour cherry-picked session.

---

### Statistical Significance

| Test | Statistic | p-value | Interpretation |
|------|-----------|---------|---------------|
| Overall accuracy binomial | 56.74% (N=2545) | < 0.000001 | Misleading — autocorrelation inflates N |
| Independent accuracy binomial | 62.07% (N=87) | 0.031 | Valid, but barely significant |
| Permutation test | 100th percentile | 0.000 | Valid on aggregate; block permutation needed |
| Hour 08 UTC accuracy | 45.8% | ~0.20 | Below chance; not sig. but directionally bad |

**Do we have enough data for a conclusive test?** No.
- For 55% true accuracy: need 783 independent samples (~65 hours)
- For 60% true accuracy: need 194 independent samples (~16 hours)
- We have 87 samples from 7.43 hours.

**To reach a verdict, run at least 16–65 more hours of live testing.**

---

### Economic Viability

| Scenario | Gross PnL/trade | Fee (RT) | Net PnL |
|----------|----------------|----------|---------|
| Taker/taker | +1.29 bps | 20 bps | **-18.71 bps** |
| Maker/maker | +1.29 bps | 10 bps | **-8.71 bps** |
| Best case | +1.29 bps | 0 bps | +1.29 bps |

**The model is not economically viable at any realistic fee level.** The signal (if real) would need to be 7–15× larger to overcome fees. There is no position size or leverage that makes this work — leverage amplifies both gain and loss proportionally, and the loss from fees is systematic.

**What would viable look like?** For a taker strategy to break even, mean gross PnL would need to exceed 20 bps. For a maker strategy, >10 bps. The current 1.29 bps mean is 8–15× too small. Alternatively, the gate would need to select a subset of predictions with ≥15 bps expected gross edge while rejecting the rest — implying the gate would trade perhaps 1–5% of predictions. This is theoretically possible if the gate can identify truly high-conviction moments, but the current gate calibration is overfit to the test set and cannot be trusted.

---

### Recommendations

**If signal exists (conditional on confirming it):**

1. **Run 50+ hours of live testing before drawing conclusions.** 7.43 hours with 87 independent samples is preliminary. You need 194 samples minimum for 80% power at 60% accuracy, or 783 for 55%. Run continuously for at least 3 days.

2. **Log gate status in the CSV.** Add a `gate_decision` column and `rejection_reason` to both the predictions and resolved CSV files. Without this, you cannot evaluate whether the regime gate adds value in live markets vs its test-set calibration.

3. **Analyze regime dependency rigorously.** Hour-of-day accuracy varies from 46% to 68%. Train separate models for Asian/European/US sessions, or add a time-of-day feature. Restrict live trading to hours 03:00–07:00 UTC pending further data.

4. **Recalibrate gate on a held-out forward validation set, not the test set.** The current gate is overfit. Use walk-forward validation — train gate on first N months, validate on next M months, test on final K months.

5. **Address prediction compression.** The model outputs 0.0606 times the realized return std. Consider a multi-objective loss (e.g., calibration penalty), quantile regression, or post-processing calibration to scale predictions to expected return magnitudes.

**If confirmed as noise (if extended test shows p>0.10):**

1. **Change the feature horizon.** 5-minute prediction is extremely competitive. The mean-reversion signals in the top features (SMA trend, VWAP deviation, RSI) are likely exploited by market-making algorithms at sub-second frequency. Try 15-minute or 1-hour prediction horizons where market microstructure noise is less dominant.

2. **Add regime conditioning.** Rather than a single model, separate the problem: detect trending vs mean-reverting regimes explicitly (using volatility, autocorrelation of returns, ADX), then apply separate prediction models within each regime.

3. **Consider the European session specifically.** The hour-08 UTC failure is a strong signal that mean-reversion fails during directional trending (European open typically sees momentum). A momentum model for 08:00–10:00 UTC could complement the mean-reversion model.

4. **Do not deploy with leverage.** At current performance levels, live trading with any meaningful position size would be reckless. The model's signal strength cannot justify risk-taking.

---

## AUDIT SUMMARY TABLE

| Category | Finding | Severity |
|----------|---------|----------|
| Lookahead bias in features | None detected | CLEAN |
| Lookahead bias in target | None detected | CLEAN |
| Scaler fitted on train only | Confirmed | CLEAN |
| Chronological split | Confirmed | CLEAN |
| Live/train feature parity | Identical Numba code | CLEAN |
| Gate calibrated on test set | **YES — overfit to test** | RED FLAG |
| Gate status not logged | Cannot evaluate live gate | YELLOW FLAG |
| Prediction compression | 0.06× realized std | YELLOW FLAG |
| Independent accuracy (87 samples) | 62.07%, p=0.031 | WEAK SIGNAL |
| Hour-of-day degradation | 45.8% at 08:00 UTC | RED FLAG |
| Economic viability | -8.71 to -18.71 bps/trade net | NOT VIABLE |
| Sample size sufficiency | 87/194 needed | INSUFFICIENT |

**Bottom line:** There may be a genuine, small edge in the 03:00–07:00 UTC window. The statistics marginally support this. The signal is too small to trade profitably after fees. The hour-08 breakdown is a serious warning. The gate is overfit and its live contribution is unknown. Do not trade live capital. Collect more data.

---
*Report generated: 2026-05-28 | All statistics computed from live test files in `live_test_results/`*
