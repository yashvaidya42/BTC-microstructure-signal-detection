import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Load CSVs
resolved = pd.read_csv(r'D:\ml\quant_ML_model\live_test_results\resolved_predictions_20260528_032904.csv')
all_preds = pd.read_csv(r'D:\ml\quant_ML_model\live_test_results\predictions_20260528_032904.csv')
unresolved = pd.read_csv(r'D:\ml\quant_ML_model\live_test_results\unresolved_predictions_20260528_032904.csv')

print("=== BASIC STATS ===")
print(f"Total predictions: {len(all_preds)}")
print(f"Resolved predictions: {len(resolved)}")
print(f"Unresolved predictions: {len(unresolved)}")

# Time span
all_preds['wall_clock'] = pd.to_datetime(all_preds['wall_clock'])
resolved['wall_clock'] = pd.to_datetime(resolved['wall_clock'])
t_start = all_preds['wall_clock'].min()
t_end = all_preds['wall_clock'].max()
print(f"Time span: {t_start} to {t_end}")
print(f"Duration: {(t_end - t_start).total_seconds()/3600:.2f} hours")

# Prediction distribution
preds = all_preds['pred_log_return'].astype(float)
print(f"\nPrediction distribution (log-return):")
print(f"  mean: {preds.mean():.8f}")
print(f"  std:  {preds.std():.8f}")
print(f"  min:  {preds.min():.8f}")
print(f"  max:  {preds.max():.8f}")
percs = np.percentile(preds, [5,25,50,75,95])
print(f"  5th/25th/50th/75th/95th: {percs}")

# One-sample t-test
t_stat, p_val = stats.ttest_1samp(preds, 0)
print(f"\nPrediction t-test (H0: mean=0): t={t_stat:.4f}, p={p_val:.6f}")
print(f"Mean pred (bps): {preds.mean()*10000:.4f}")

# Direction counts
print(f"\nDirection distribution:")
print(resolved['direction'].value_counts() if 'direction' in resolved.columns else "no direction col")

print("\n=== DIRECTIONAL ACCURACY ===")
# Filter FLAT
non_flat = resolved[resolved['correct_direction'] != 'FLAT'].copy()
n_non_flat = len(non_flat)
n_correct = (non_flat['correct_direction'] == 'CORRECT').sum()
n_wrong = (non_flat['correct_direction'] == 'WRONG').sum()
overall_acc = n_correct / n_non_flat if n_non_flat > 0 else 0
print(f"Overall (non-flat): {n_correct}/{n_non_flat} = {overall_acc:.4f}")

# On real moves >1 bps
real_moves = resolved[np.abs(resolved['realized_return'].astype(float)) > 0.0001]
real_moves_nf = real_moves[real_moves['correct_direction'] != 'FLAT']
rm_correct = (real_moves_nf['correct_direction'] == 'CORRECT').sum()
rm_acc = rm_correct / len(real_moves_nf) if len(real_moves_nf) > 0 else 0
print(f"On real moves >1bps: {rm_correct}/{len(real_moves_nf)} = {rm_acc:.4f}")

# On large moves >3 bps
large_moves = resolved[np.abs(resolved['realized_return'].astype(float)) > 0.0003]
lm_nf = large_moves[large_moves['correct_direction'] != 'FLAT']
lm_correct = (lm_nf['correct_direction'] == 'CORRECT').sum()
lm_acc = lm_correct / len(lm_nf) if len(lm_nf) > 0 else 0
print(f"On large moves >3bps: {lm_correct}/{len(lm_nf)} = {lm_acc:.4f}")

# Top 10% by abs prediction magnitude
abs_pred = np.abs(resolved['pred_log_return'].astype(float))
thresh_90 = np.percentile(abs_pred, 90)
top10 = resolved[abs_pred >= thresh_90]
top10_nf = top10[top10['correct_direction'] != 'FLAT']
t10_correct = (top10_nf['correct_direction'] == 'CORRECT').sum()
t10_acc = t10_correct / len(top10_nf) if len(top10_nf) > 0 else 0
print(f"Top 10% by |pred| (thresh={thresh_90*10000:.2f}bps): {t10_correct}/{len(top10_nf)} = {t10_acc:.4f}")

# Binomial test overall
bt_overall = stats.binomtest(n_correct, n_non_flat, 0.5, alternative='two-sided')
print(f"\nBinomial test (overall, two-sided): p={bt_overall.pvalue:.6f}")

print("\n=== NON-OVERLAPPING (INDEPENDENT) ACCURACY ===")
# Select one prediction per 5-min window (300s = 300000ms)
resolved_sorted = resolved.sort_values('timestamp_ms').copy()
resolved_sorted['timestamp_ms'] = resolved_sorted['timestamp_ms'].astype(np.int64)

# Non-overlapping sampling: one per 300s window
indep_rows = []
next_allowed_ts = resolved_sorted['timestamp_ms'].iloc[0]
for _, row in resolved_sorted.iterrows():
    if row['timestamp_ms'] >= next_allowed_ts:
        indep_rows.append(row)
        next_allowed_ts = row['timestamp_ms'] + 300_000  # 300 seconds in ms

indep_df = pd.DataFrame(indep_rows)
indep_nf = indep_df[indep_df['correct_direction'] != 'FLAT']
indep_correct = (indep_nf['correct_direction'] == 'CORRECT').sum()
indep_n = len(indep_nf)
indep_acc = indep_correct / indep_n if indep_n > 0 else 0
print(f"Independent samples (non-overlapping, non-flat): {indep_correct}/{indep_n} = {indep_acc:.4f}")

bt_indep = stats.binomtest(int(indep_correct), int(indep_n), 0.5, alternative='two-sided')
bt_indep_gt = stats.binomtest(int(indep_correct), int(indep_n), 0.5, alternative='greater')
print(f"Binomial test (indep, two-sided): p={bt_indep.pvalue:.6f}")
print(f"Binomial test (indep, greater): p={bt_indep_gt.pvalue:.6f}")

# Power analysis: how many samples needed to confirm 55% at p<0.05
from scipy.stats import norm
def samples_needed(p_true, alpha=0.05, power=0.80):
    z_alpha = norm.ppf(1 - alpha/2)
    z_beta = norm.ppf(power)
    p0 = 0.5
    n = ((z_alpha * np.sqrt(p0*(1-p0)) + z_beta * np.sqrt(p_true*(1-p_true)))**2) / (p_true - p0)**2
    return int(np.ceil(n))

print(f"\nSamples needed for 52% accuracy at p<0.05: {samples_needed(0.52)}")
print(f"Samples needed for 55% accuracy at p<0.05: {samples_needed(0.55)}")
print(f"Samples needed for 60% accuracy at p<0.05: {samples_needed(0.60)}")
print(f"We have {indep_n} independent samples")

print("\n=== PnL ANALYSIS ===")
resolved['pred_lr'] = resolved['pred_log_return'].astype(float)
resolved['realized_lr'] = resolved['realized_return'].astype(float)
resolved['trade_pnl_bps'] = np.sign(resolved['pred_lr']) * resolved['realized_lr'] * 10000

cum_pnl = resolved['trade_pnl_bps'].sum()
print(f"Total cumulative PnL (all predictions): {cum_pnl:.2f} bps")
print(f"Mean PnL per prediction: {resolved['trade_pnl_bps'].mean():.4f} bps")
print(f"Std PnL per prediction: {resolved['trade_pnl_bps'].std():.4f} bps")

# Sharpe ratio (annualized)
n_per_day = 6 * 60  # 6 predictions/min * 60 min (10s interval)
sharpe_annual = (resolved['trade_pnl_bps'].mean() / resolved['trade_pnl_bps'].std()) * np.sqrt(n_per_day * 365)
print(f"Sharpe ratio (annualized, assumes 360 pred/hr): {sharpe_annual:.4f}")

# Is PnL from a few outliers?
pnl_sorted = resolved['trade_pnl_bps'].sort_values(ascending=False)
print(f"\nTop 5 PnL trades: {pnl_sorted.head(5).values}")
print(f"Bottom 5 PnL trades: {pnl_sorted.tail(5).values}")
print(f"PnL without top 5 trades: {pnl_sorted.iloc[5:].sum():.2f} bps")
print(f"PnL without top 10 trades: {pnl_sorted.iloc[10:].sum():.2f} bps")

print("\n=== AUTOCORRELATION IN CORRECTNESS ===")
correct_series = (non_flat['correct_direction'] == 'CORRECT').astype(int).values
if len(correct_series) > 10:
    try:
        from statsmodels.stats.stattools import durbin_watson
        dw = durbin_watson(correct_series)
        print(f"Durbin-Watson statistic: {dw:.4f} (2.0=no autocorrel, <2=pos, >2=neg)")
    except ImportError:
        # Manual Durbin-Watson: DW = sum((e_t - e_{t-1})^2) / sum(e_t^2)
        e = correct_series.astype(float)
        dw = np.sum(np.diff(e)**2) / np.sum(e**2)
        print(f"Durbin-Watson statistic (manual): {dw:.4f} (2.0=no autocorrel, <2=pos, >2=neg)")

    # Lag-1 autocorrelation
    ac1 = np.corrcoef(correct_series[:-1], correct_series[1:])[0,1]
    print(f"Lag-1 autocorrelation of correct/wrong: {ac1:.4f}")

print("\n=== PERMUTATION TEST ===")
np.random.seed(42)
n_perm = 10000
realized_arr = non_flat['realized_return'].astype(float).values
pred_arr = non_flat['pred_log_return'].astype(float).values
real_acc = (np.sign(pred_arr) == np.sign(realized_arr)).mean()

perm_accs = np.zeros(n_perm)
for i in range(n_perm):
    shuffled_preds = np.random.permutation(pred_arr)
    perm_accs[i] = (np.sign(shuffled_preds) == np.sign(realized_arr)).mean()

perm_p = (perm_accs >= real_acc).mean()
print(f"Observed accuracy: {real_acc:.4f}")
print(f"Permutation test p-value: {perm_p:.4f}")
print(f"Permutation mean: {perm_accs.mean():.4f}, std: {perm_accs.std():.4f}")
print(f"Percentile of real acc in permutation dist: {(perm_accs < real_acc).mean()*100:.1f}th")

print("\n=== TIME OF DAY ACCURACY ===")
resolved['hour'] = pd.to_datetime(resolved['wall_clock']).dt.hour
hourly = resolved[resolved['correct_direction'] != 'FLAT'].groupby('hour').apply(
    lambda x: (x['correct_direction'] == 'CORRECT').sum() / len(x)
)
print("Hourly accuracy:")
print(hourly)

print("\n=== PREDICTION STD vs TARGET STD ===")
pred_std = resolved['pred_log_return'].astype(float).std()
target_std = resolved['realized_return'].astype(float).std()
print(f"Prediction std: {pred_std:.8f} ({pred_std*10000:.4f} bps)")
print(f"Realized return std: {target_std:.8f} ({target_std*10000:.4f} bps)")
print(f"Ratio pred_std/target_std: {pred_std/target_std:.4f}")

print("\n=== DIRECTION BIAS ===")
print(f"UP predictions: {(all_preds['direction']=='UP').sum()}")
print(f"DN predictions: {(all_preds['direction']=='DN').sum()}")
print(f"UP fraction: {(all_preds['direction']=='UP').mean():.4f}")

print("\n=== ECONOMIC VIABILITY ===")
# Binance spot taker fee: 0.1% = 10 bps round trip
# Maker fee: 0.1% = 10 bps round trip (0.05% each way)
mean_pnl_bps = resolved['trade_pnl_bps'].mean()
taker_cost = 20  # 20 bps round trip taker
maker_cost = 10  # 10 bps round trip maker
print(f"Mean gross PnL per prediction: {mean_pnl_bps:.4f} bps")
print(f"After taker fees (20bps RT): {mean_pnl_bps - taker_cost:.4f} bps")
print(f"After maker fees (10bps RT): {mean_pnl_bps - maker_cost:.4f} bps")

# What fraction of predictions would be profitable after fees?
pos_pnl = (resolved['trade_pnl_bps'] > 20).sum() / len(resolved)
print(f"Fraction of predictions profitable after 20bps: {pos_pnl:.4f}")
