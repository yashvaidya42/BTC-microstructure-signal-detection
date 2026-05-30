#!/usr/bin/env python3
"""
LIVE INFERENCE ENGINE v4 — BTC 5-Min Forward Prediction
========================================================
Architecture:
  1. Start local WebSocket server (pushes data to dashboard)
  2. Connect to Binance WebSocket (btcusdt@trade)
  3. Backfill 60 min of history from REST API
  4. Merge (dedupe by trade_id), fill circular buffer
  5. Compute 12 V4 features using EXACT same Numba functions as training
  6. Normalize with saved scaler, predict with saved XGBoost model
  7. Push (actual_price, predicted_price_5m) to dashboard
  8. Log all predictions to CSV for post-analysis
  9. Auto-shutdown after RUN_DURATION_MINUTES

Usage:
    python live_inference.py

Requirements:
    pip install websockets xgboost numba numpy requests
"""

import asyncio
import json
import time
import csv
import threading
import pickle
import signal
import sys
import os
from collections import deque
from pathlib import Path
from datetime import datetime, timezone

import math
import numpy as np
import xgboost as xgb
import requests

try:
    import websockets
    import websockets.server
except ImportError:
    print("[FATAL] pip install websockets")
    sys.exit(1)

try:
    from numba import jit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    print("[WARN] numba not installed — will be SLOW")
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return decorator

# =============================================================================
# CONFIGURATION
# =============================================================================

# --- Model Artifacts (V4) ---
# BTC_BASE_DIR env var lets the same script run on both Windows (local) and
# Linux (server) without modification. Systemd sets it to /home/ubuntu/btc_model.
BASE_DIR     = Path(os.environ.get('BTC_BASE_DIR', str(Path(__file__).resolve().parent)))
MODEL_PATH   = os.environ.get('BTC_MODEL_PATH',
                   str(BASE_DIR / "model_output_xgb_v4" / "model_sniper_v4.json"))
SCALER_PATH  = os.environ.get('BTC_SCALER_PATH',
                   str(BASE_DIR / "model_output_xgb_v4" / "scaler_regression_v4.pkl"))
KEEPIDX_PATH = os.environ.get('BTC_KEEPIDX_PATH',
                   str(BASE_DIR / "model_output_xgb_v4" / "keep_idx.pkl"))
GATE_CONFIG_PATH = os.environ.get('BTC_GATE_CONFIG_PATH',
                   str(BASE_DIR / "model_output_xgb_v4" / "gate_config.json"))

# --- Binance ---
BINANCE_WS_URL   = "wss://stream.binance.com:9443/ws/btcusdt@trade"
SYMBOL           = "BTCUSDT"
REST_LIMIT       = 1000

# --- Buffer ---
BUFFER_MAX_SECONDS = 3600   # 1 hour — needed for rel_size_300s (1h rolling mean)
BUFFER_MAX_ROWS    = 600_000

# --- Inference ---
INFERENCE_INTERVAL_SEC = 10  # Predict every 10s (safe for i5-11400H)
LOOK_AHEAD_SECONDS     = 300
MIN_BUFFER_SPAN_SEC    = 900  # Need 15 min for SMA trend feature

# --- Auto-shutdown (0 = run indefinitely until Ctrl+C) ---
RUN_DURATION_MINUTES = 0

# --- Dashboard Server ---
DASHBOARD_HOST = "0.0.0.0"
DASHBOARD_PORT = 8765

# --- Logging ---
LOG_DIR = str(BASE_DIR / "live_test_results")
LOG_CSV = None  # set in main()

# --- V4 Feature List (EXACT order matching training, 12 features) ---
FEATURE_COLUMNS = [
    'price_volatility_300s',
    'vol_imbalance_300s',
    'vwap_dev_300s',
    'intensity_z_300s',
    'feat_vol_force',
    'feat_sma_trend',
    'feat_rsi_norm',
    'tfi_quote_norm_300s',
    'amihud_illiquidity_300s',
    'rel_size_300s',
    'inter_time_cv_300s',
    'size_cv_300s',
]


# =============================================================================
# NUMBA ROLLING PRIMITIVES (EXACT COPY from 1_v4_preprocessing.py)
# =============================================================================

@jit(nopython=True, cache=True)
def _rolling_sum(ts_us, vals, win_sec):
    n = len(vals)
    out = np.empty(n, dtype=np.float64)
    win_us = win_sec * 1_000_000
    left = 0; s = 0.0
    for r in range(n):
        s += vals[r]
        lo = ts_us[r] - win_us
        while left < r and ts_us[left] < lo:
            s -= vals[left]; left += 1
        out[r] = s
    return out

@jit(nopython=True, cache=True)
def _rolling_mean(ts_us, vals, win_sec):
    n = len(vals)
    out = np.empty(n, dtype=np.float64)
    win_us = win_sec * 1_000_000
    left = 0; s = 0.0
    for r in range(n):
        s += vals[r]
        lo = ts_us[r] - win_us
        while left < r and ts_us[left] < lo:
            s -= vals[left]; left += 1
        out[r] = s / (r - left + 1)
    return out

@jit(nopython=True, cache=True)
def _rolling_std(ts_us, vals, win_sec):
    n = len(vals)
    out = np.empty(n, dtype=np.float64)
    win_us = win_sec * 1_000_000
    left = 0; s = 0.0; sq = 0.0
    for r in range(n):
        s += vals[r]; sq += vals[r] * vals[r]
        lo = ts_us[r] - win_us
        while left < r and ts_us[left] < lo:
            s -= vals[left]; sq -= vals[left] * vals[left]; left += 1
        cnt = r - left + 1
        if cnt >= 2:
            m = s / cnt
            var = sq / cnt - m * m
            out[r] = np.sqrt(max(0.0, var))
        else:
            out[r] = 0.0
    return out

@jit(nopython=True, cache=True)
def _rolling_count(ts_us, win_sec):
    n = len(ts_us)
    out = np.empty(n, dtype=np.float64)
    win_us = win_sec * 1_000_000
    left = 0
    for r in range(n):
        lo = ts_us[r] - win_us
        while left < r and ts_us[left] < lo:
            left += 1
        out[r] = float(r - left + 1)
    return out

@jit(nopython=True, cache=True)
def _rolling_vwap(ts_us, prices, qtys, win_sec):
    n = len(prices)
    out = np.empty(n, dtype=np.float64)
    win_us = win_sec * 1_000_000
    left = 0; s_pq = 0.0; s_q = 0.0
    for r in range(n):
        s_pq += prices[r] * qtys[r]
        s_q  += qtys[r]
        lo = ts_us[r] - win_us
        while left < r and ts_us[left] < lo:
            s_pq -= prices[left] * qtys[left]
            s_q  -= qtys[left]
            left += 1
        if s_q > 1e-15:
            out[r] = s_pq / s_q
        else:
            out[r] = prices[r]
    return out


# =============================================================================
# V4 FEATURE COMPUTATION (matches 1_v4_preprocessing.py EXACTLY)
# =============================================================================

def compute_features_v4(ts_us, prices, qty, side):
    """
    Computes the 12 V4 features for the LAST row of the buffer.
    Returns: np.ndarray of shape (12,) in EXACT training column order.
    """
    n = len(ts_us)
    w = 300  # 5-min window

    # --- Base metrics (vectorized, fine for buffer size) ---
    time_delta = np.zeros(n, dtype=np.float64)
    time_delta[1:] = (ts_us[1:] - ts_us[:-1]) / 1e6

    log_return = np.zeros(n, dtype=np.float64)
    log_return[1:] = np.log(prices[1:] / prices[:-1])

    price_diff = np.zeros(n, dtype=np.float64)
    price_diff[1:] = prices[1:] - prices[:-1]

    buy_volume  = qty * (side == 1).astype(np.float64)
    sell_volume = qty * (side == -1).astype(np.float64)

    # === 1. TREND & PHYSICS (kept from v3) ===

    # Volume Force (mass * acceleration, smoothed 60s)
    raw_force = qty * price_diff
    feat_vol_force = _rolling_mean(ts_us, raw_force, 60)

    # SMA Trend Deviation (15 min / 900s)
    sma_900s = _rolling_mean(ts_us, prices, 900)
    feat_sma_trend = (prices - sma_900s) / (sma_900s + 1e-9)

    # Tick RSI (300s window)
    up_moves = np.maximum(price_diff, 0)
    down_moves = np.abs(np.minimum(price_diff, 0))
    avg_gain = _rolling_mean(ts_us, up_moves, w)
    avg_loss = _rolling_mean(ts_us, down_moves, w)
    rs = avg_gain / (avg_loss + 1e-9)
    feat_rsi = 100.0 - (100.0 / (1.0 + rs))
    feat_rsi_norm = (feat_rsi - 50.0) / 50.0

    # === 2. FLOW FEATURES ===

    rb = _rolling_sum(ts_us, buy_volume, w)
    rs_vol = _rolling_sum(ts_us, sell_volume, w)
    denom = rb + rs_vol + 1e-9
    vol_imbalance_300s = (rb - rs_vol) / denom

    # [V4 P2] TFI normalized by VWAP (price-invariant)
    vwap_w = _rolling_vwap(ts_us, prices, qty, w)
    rb_q = _rolling_sum(ts_us, buy_volume * prices, w)
    rs_q = _rolling_sum(ts_us, sell_volume * prices, w)
    tfi_raw = rb_q - rs_q
    tfi_quote_norm_300s = tfi_raw / (vwap_w + 1e-9)

    # === 3. VOLATILITY & SPEED ===

    price_volatility_300s = _rolling_std(ts_us, log_return, w)

    # [V4 P7] Inter-arrival time: CV ratio
    inter_time_mean = _rolling_mean(ts_us, time_delta, w)
    inter_time_std  = _rolling_std(ts_us, time_delta, w)
    inter_time_cv_300s = inter_time_std / (inter_time_mean + 1e-9)

    # Intensity Z
    count_w  = _rolling_count(ts_us, w)
    mean_cnt = _rolling_mean(ts_us, count_w, w)
    std_cnt  = _rolling_std(ts_us, count_w, w)
    intensity_z_300s = (count_w - mean_cnt) / (std_cnt + 1e-9)

    # === 4. RELATIVE VALUE ===

    vwap_dev_300s = (prices - vwap_w) / (vwap_w + 1e-9)

    # [V4 P2] Amihud — use qty (BTC), not quote_qty ($)
    amihud_raw = np.abs(log_return) / (qty + 1e-9)
    amihud_illiquidity_300s = _rolling_mean(ts_us, amihud_raw, w)

    # === 5. SIZE FEATURES ===

    avg_size_300s = _rolling_mean(ts_us, qty, w)
    size_std_300s = _rolling_std(ts_us, qty, w)

    # [V4 P2] Relative size: normalize by 1h rolling mean
    avg_size_1h = _rolling_mean(ts_us, qty, 3600)
    rel_size_300s = avg_size_300s / (avg_size_1h + 1e-15)

    # [V4 P7] Size CV ratio
    size_cv_300s = size_std_300s / (avg_size_300s + 1e-9)

    # === ASSEMBLE: LAST ROW ONLY, in EXACT training column order (12 features) ===
    features = np.array([
        price_volatility_300s[-1],     # 0
        vol_imbalance_300s[-1],        # 1
        vwap_dev_300s[-1],             # 2
        intensity_z_300s[-1],          # 3
        feat_vol_force[-1],            # 4
        feat_sma_trend[-1],            # 5
        feat_rsi_norm[-1],             # 6
        tfi_quote_norm_300s[-1],       # 7
        amihud_illiquidity_300s[-1],   # 8
        rel_size_300s[-1],             # 9
        inter_time_cv_300s[-1],        # 10
        size_cv_300s[-1],              # 11
    ], dtype=np.float32)

    return features


# =============================================================================
# SCALER (matches SimpleScaler from 2_v4_dataset.py)
# =============================================================================

class SimpleScaler:
    def __init__(self, mean_, std_):
        self.mean_ = mean_
        self.std_  = std_

    def transform(self, X):
        return (X - self.mean_) / (self.std_ + 1e-8)

    @classmethod
    def load(cls, filepath):
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        return cls(mean_=data['mean'], std_=data['std'])


# =============================================================================
# REGIME GATE
# =============================================================================

class RegimeGate:
    """
    Filters predictions through regime conditions.
    Only signals "trade" when all gates pass:
      1. Volatility is in the productive range (not too low, not too high)
      2. Volume is sufficient (not thin market)
      3. Prediction magnitude exceeds cost threshold
      4. Top features agree on direction

    Thresholds are calibrated from test set evaluation (see gate_calibration.json).
    Defaults are conservative starting points.
    """

    DEFAULT_CONFIG = {
        # Volatility gate: price_volatility_300s percentile range
        # Only trade when vol is elevated but not extreme
        'vol_low_percentile': 60,    # below this = too quiet, noise dominates
        'vol_high_percentile': 90,   # above this = momentum/news, mean reversion fails
        'vol_low': 0.0,              # actual threshold (set during calibration)
        'vol_high': 999.0,           # actual threshold (set during calibration)

        # Volume gate: rolling 5min trade count must exceed this
        'min_trade_count_300s': 500,  # conservative default

        # Prediction magnitude gate: |pred| must exceed round-trip cost
        'min_pred_bps': 3.0,         # 3 bps minimum (covers ~1.5 bps each way)

        # Feature agreement gate: top N features must agree on direction
        'agreement_enabled': True,
        # Indices of key directional features in the 12-feature vector:
        # vol_imbalance(1), vwap_dev(2), feat_sma_trend(5), feat_rsi_norm(6), tfi_quote_norm(7)
        'directional_feature_indices': [1, 2, 5, 6, 7],
        'min_agreement_ratio': 0.6,  # at least 60% of directional features agree
    }

    def __init__(self, config_path=None):
        self.config = self.DEFAULT_CONFIG.copy()
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                saved = json.load(f)
                self.config.update(saved)
            print(f"[GATE] Loaded config from {config_path}")
        else:
            print(f"[GATE] Using default config (no calibration file found)")

        print(f"[GATE] Thresholds:")
        print(f"  vol range    : [{self.config['vol_low']:.8f}, {self.config['vol_high']:.8f}]")
        print(f"  min_pred_bps : {self.config['min_pred_bps']:.1f} bps")
        print(f"  min_trades   : {self.config['min_trade_count_300s']}")
        print(f"  agreement    : {'enabled' if self.config['agreement_enabled'] else 'disabled'}"
              f"  (indices={self.config['directional_feature_indices']}, "
              f"min_ratio={self.config['min_agreement_ratio']})")

        # Stats tracking
        self.total_predictions = 0
        self.trades_passed = 0
        self.gate_rejections = {
            'volatility_low': 0,
            'volatility_high': 0,
            'volume': 0,
            'magnitude': 0,
            'agreement': 0,
        }

    def check(self, features_raw, pred_log_return, trade_count_300s):
        """
        Returns (should_trade: bool, rejection_reason: str or None, gate_details: dict)

        Args:
            features_raw: np.ndarray shape (12,) — raw (un-normalized) feature values
            pred_log_return: float — model's prediction
            trade_count_300s: float — number of trades in last 300s
        """
        self.total_predictions += 1
        pred_bps = abs(pred_log_return) * 10_000

        details = {
            'pred_bps': round(pred_bps, 2),
            'volatility': round(float(features_raw[0]), 8),  # price_volatility_300s
            'trade_count': int(trade_count_300s),
        }

        # Gate 1: Volatility regime
        vol = float(features_raw[0])  # price_volatility_300s is index 0
        if vol < self.config['vol_low']:
            self.gate_rejections['volatility_low'] += 1
            return False, 'volatility_low', details
        if vol > self.config['vol_high']:
            self.gate_rejections['volatility_high'] += 1
            return False, 'volatility_high', details

        # Gate 2: Volume
        if trade_count_300s < self.config['min_trade_count_300s']:
            self.gate_rejections['volume'] += 1
            return False, 'volume', details

        # Gate 3: Prediction magnitude
        if pred_bps < self.config['min_pred_bps']:
            self.gate_rejections['magnitude'] += 1
            return False, 'magnitude', details

        # Gate 4: Feature agreement
        if self.config['agreement_enabled']:
            direction = np.sign(pred_log_return)
            indices = self.config['directional_feature_indices']
            # For mean-reversion: negative value suggests price will go UP
            # vwap_dev > 0 = price above vwap = mean revert down
            # sma_trend > 0 = price above sma = mean revert down
            # rsi_norm > 0 = overbought = mean revert down
            feature_signs = np.sign(features_raw[indices])
            # Flip mean-reverting features: positive feature → expect DOWN
            for i, idx in enumerate(indices):
                if idx in [2, 5, 6]:  # vwap_dev, sma_trend, rsi_norm
                    feature_signs[i] = -feature_signs[i]

            agreement = float(np.mean(feature_signs == direction))
            details['feature_agreement'] = round(agreement, 2)

            if agreement < self.config['min_agreement_ratio']:
                self.gate_rejections['agreement'] += 1
                return False, 'agreement', details

        self.trades_passed += 1
        return True, None, details

    def pass_rate(self):
        if self.total_predictions == 0:
            return 0.0
        return self.trades_passed / self.total_predictions

    def report(self):
        print(f"\n{'='*60}")
        print(f"  REGIME GATE STATS")
        print(f"  Total predictions: {self.total_predictions}")
        print(f"  Trades passed:     {self.trades_passed} ({self.pass_rate()*100:.1f}%)")
        print(f"  Rejections:")
        for reason, count in self.gate_rejections.items():
            pct = count / max(1, self.total_predictions) * 100
            print(f"    {reason:<20}: {count:>6} ({pct:.1f}%)")
        print(f"{'='*60}\n")


# =============================================================================
# TICK BUFFER
# =============================================================================

class TickBuffer:
    def __init__(self, max_seconds=BUFFER_MAX_SECONDS, max_rows=BUFFER_MAX_ROWS):
        self.max_seconds = max_seconds
        self.max_rows = max_rows

        self.trade_ids    = deque(maxlen=max_rows)
        self.timestamps   = deque(maxlen=max_rows)
        self.prices       = deque(maxlen=max_rows)
        self.qtys         = deque(maxlen=max_rows)
        self.sides        = deque(maxlen=max_rows)

        self._trade_id_set = set()
        self._lock = threading.Lock()

    def add_tick(self, trade_id, timestamp_ms, price, qty, is_buyer_maker):
        with self._lock:
            if trade_id in self._trade_id_set:
                return
            self.trade_ids.append(trade_id)
            self.timestamps.append(timestamp_ms)
            self.prices.append(price)
            self.qtys.append(qty)
            self.sides.append(-1 if is_buyer_maker else 1)
            self._trade_id_set.add(trade_id)
            self._evict()

    def add_ticks_bulk(self, ticks):
        with self._lock:
            for trade_id, ts_ms, price, qty, is_buyer_maker in ticks:
                if trade_id in self._trade_id_set:
                    continue
                self.trade_ids.append(trade_id)
                self.timestamps.append(ts_ms)
                self.prices.append(price)
                self.qtys.append(qty)
                self.sides.append(-1 if is_buyer_maker else 1)
                self._trade_id_set.add(trade_id)
            self._evict()

    def _evict(self):
        if not self.timestamps:
            return
        cutoff_ms = max(self.timestamps) - (self.max_seconds * 1000)
        while self.timestamps and self.timestamps[0] < cutoff_ms:
            old_id = self.trade_ids.popleft()
            self.timestamps.popleft()
            self.prices.popleft()
            self.qtys.popleft()
            self.sides.popleft()
            self._trade_id_set.discard(old_id)

    def get_arrays(self):
        with self._lock:
            if len(self.timestamps) < 100:
                return None

            ts   = np.array(self.timestamps, dtype=np.int64)
            p    = np.array(self.prices, dtype=np.float64)
            q    = np.array(self.qtys, dtype=np.float64)
            s    = np.array(self.sides, dtype=np.float64)

            order = np.argsort(ts)
            ts = ts[order]
            p  = p[order]
            q  = q[order]
            s  = s[order]

            ts_us = ts * 1000
            return ts_us, p, q, s

    def __len__(self):
        return len(self.timestamps)

    @property
    def time_span_seconds(self):
        with self._lock:
            if len(self.timestamps) < 2:
                return 0
            return (max(self.timestamps) - min(self.timestamps)) / 1000


# =============================================================================
# BINANCE REST BACKFILL
# =============================================================================

def fetch_historical_trades(n_minutes=60):
    end_ms   = int(time.time() * 1000)
    start_ms = end_ms - (n_minutes * 60 * 1000)

    all_trades = []
    current_start = start_ms

    print(f"[REST] Fetching {n_minutes} min of history...")

    while current_start < end_ms:
        params = {
            'symbol':    SYMBOL,
            'startTime': current_start,
            'endTime':   end_ms,
            'limit':     REST_LIMIT,
        }
        try:
            resp = requests.get(
                "https://api.binance.com/api/v3/aggTrades",
                params=params, timeout=10,
            )
            resp.raise_for_status()
            trades = resp.json()
        except Exception as e:
            print(f"[REST] Error: {e}")
            break

        if not trades:
            break

        for t in trades:
            all_trades.append((
                t['a'],
                t['T'],
                float(t['p']),
                float(t['q']),
                t['m'],
            ))

        last_ts = trades[-1]['T']
        if last_ts <= current_start:
            break
        current_start = last_ts + 1
        time.sleep(0.1)

    print(f"[REST] Done. {len(all_trades)} trades fetched.")
    return all_trades


# =============================================================================
# PREDICTION LOGGER (CSV for post-analysis)
# =============================================================================

class PredictionLogger:
    def __init__(self, filepath):
        self.filepath = filepath
        self._file = open(filepath, 'w', newline='', encoding='utf-8')
        self._writer = csv.writer(self._file)
        self._writer.writerow([
            'wall_clock', 'timestamp_ms', 'current_price',
            'predicted_price_5m', 'pred_log_return', 'pred_bps',
            'direction', 'buffer_ticks', 'buffer_span_s',
            'gate_decision', 'rejection_reason',
        ])
        self._file.flush()
        self.count = 0

    def log(self, timestamp_ms, current_price, predicted_price,
            pred_lr, buffer_size, buffer_span,
            gate_decision, rejection_reason):
        bps = abs(pred_lr) * 10000
        direction = "UP" if pred_lr > 0 else "DN"
        self._writer.writerow([
            datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            timestamp_ms,
            f"{current_price:.2f}",
            f"{predicted_price:.2f}",
            f"{pred_lr:.10f}",
            f"{bps:.4f}",
            direction,
            buffer_size,
            f"{buffer_span:.0f}",
            gate_decision,
            rejection_reason if rejection_reason else '',
        ])
        self.count += 1
        if self.count % 10 == 0:
            self._file.flush()

    def close(self):
        self._file.flush()
        self._file.close()


# =============================================================================
# LIVE STATS TRACKER (non-overlapping accuracy for real signal measurement)
# =============================================================================

STATS_REPORT_INTERVAL = 60  # print stats every 60 seconds

class LiveStatsTracker:
    """Tracks both raw and non-overlapping (independent) accuracy in real-time."""
    def __init__(self):
        self.all_correct = 0
        self.all_wrong = 0
        self.all_flat = 0
        self.cum_pnl_bps = 0.0

        self._next_independent_ts_ms = 0
        self.indep_correct = 0
        self.indep_wrong = 0
        self.indep_flat = 0
        self.indep_pnl_bps = 0.0

        self._last_report_time = 0

    def record(self, pred_lr, realized_return, timestamp_ms):
        if abs(realized_return) < 1e-5:
            tag = "FLAT"
        elif (realized_return > 0) == (pred_lr > 0):
            tag = "CORRECT"
        else:
            tag = "WRONG"

        trade_pnl = (1 if pred_lr > 0 else -1) * realized_return * 10_000

        if tag == "CORRECT":
            self.all_correct += 1
        elif tag == "WRONG":
            self.all_wrong += 1
        else:
            self.all_flat += 1
        self.cum_pnl_bps += trade_pnl

        if timestamp_ms >= self._next_independent_ts_ms:
            if tag == "CORRECT":
                self.indep_correct += 1
            elif tag == "WRONG":
                self.indep_wrong += 1
            else:
                self.indep_flat += 1
            self.indep_pnl_bps += trade_pnl
            self._next_independent_ts_ms = timestamp_ms + (LOOK_AHEAD_SECONDS * 1000)

    def should_report(self):
        now = time.time()
        if now - self._last_report_time >= STATS_REPORT_INTERVAL:
            self._last_report_time = now
            return True
        return False

    def report(self):
        n_all = self.all_correct + self.all_wrong
        n_indep = self.indep_correct + self.indep_wrong

        all_acc = self.all_correct / n_all * 100 if n_all > 0 else 0
        indep_acc = self.indep_correct / n_indep * 100 if n_indep > 0 else 0

        p_val_str = ""
        if n_indep >= 5:
            from scipy.stats import binomtest
            bt = binomtest(self.indep_correct, n_indep, 0.5, alternative='greater')
            p_val_str = f"  p={bt.pvalue:.4f}"
            if bt.pvalue < 0.05:
                p_val_str += " *SIG*"

        needed = 42
        hours_left = max(0, (needed - n_indep) * 5 / 60)

        print(f"\n{'='*70}")
        print(f"  LIVE STATS @ {datetime.now().strftime('%H:%M:%S')}")
        print(f"  Raw:         {self.all_correct}/{n_all} = {all_acc:.1f}%  "
              f"(flat={self.all_flat})  PnL={self.cum_pnl_bps:+.1f}bps")
        print(f"  Independent: {self.indep_correct}/{n_indep} = {indep_acc:.1f}%  "
              f"PnL={self.indep_pnl_bps:+.1f}bps{p_val_str}")
        print(f"  Need {needed} indep samples for verdict. "
              f"Have {n_indep}. ~{hours_left:.1f}h remaining.")
        print(f"{'='*70}\n")


# =============================================================================
# RESOLVED PREDICTION LOGGER
# =============================================================================

RESOLVED_CSV_COLUMNS = [
    'wall_clock', 'timestamp_ms', 'current_price',
    'predicted_price_5m', 'pred_log_return', 'pred_bps',
    'direction', 'buffer_ticks', 'buffer_span_s',
    'gate_decision', 'rejection_reason',
    'realized_return', 'price_at_t_plus_5', 'correct_direction', 'abs_error_bps',
]


class ResolvedLogger:
    def __init__(self, filepath):
        self.filepath = filepath
        self._file = open(filepath, 'w', newline='', encoding='utf-8')
        self._writer = csv.writer(self._file)
        self._writer.writerow(RESOLVED_CSV_COLUMNS)
        self._file.flush()
        self.count = 0

    def log(self, original_row, realized_return, price_at_t_plus_5):
        pred_lr = original_row['pred_log_return']
        abs_error = abs(realized_return - pred_lr) * 10_000

        if abs(realized_return) < 1e-5:
            correct_direction = "FLAT"
        elif (realized_return > 0) == (pred_lr > 0):
            correct_direction = "CORRECT"
        else:
            correct_direction = "WRONG"

        self._writer.writerow([
            original_row['wall_clock'],
            original_row['timestamp_ms'],
            original_row['current_price'],
            original_row['predicted_price_5m'],
            f"{pred_lr:.10f}",
            original_row['pred_bps'],
            original_row['direction'],
            original_row['buffer_ticks'],
            original_row['buffer_span_s'],
            original_row.get('gate_decision', ''),
            original_row.get('rejection_reason', ''),
            f"{realized_return:.10f}",
            f"{price_at_t_plus_5:.2f}",
            correct_direction,
            f"{abs_error:.4f}",
        ])
        self.count += 1
        if self.count % 5 == 0:
            self._file.flush()

    def close(self):
        self._file.flush()
        self._file.close()


def write_unresolved_csv(filepath, pending_predictions):
    if not pending_predictions:
        return
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            'wall_clock', 'timestamp_ms', 'current_price',
            'predicted_price_5m', 'pred_log_return', 'pred_bps',
            'direction', 'buffer_ticks', 'buffer_span_s',
            'gate_decision', 'rejection_reason',
        ])
        for entry in pending_predictions:
            row = entry['row']
            writer.writerow([
                row['wall_clock'], row['timestamp_ms'], row['current_price'],
                row['predicted_price_5m'], f"{row['pred_log_return']:.10f}",
                row['pred_bps'], row['direction'],
                row['buffer_ticks'], row['buffer_span_s'],
                row.get('gate_decision', ''), row.get('rejection_reason', ''),
            ])
    print(f"[SHUTDOWN] {len(pending_predictions)} unresolved predictions saved: {filepath}")


# =============================================================================
# LIVE ENGINE
# =============================================================================

class LiveEngine:
    def __init__(self):
        self.buffer = TickBuffer()
        self.model  = None
        self.scaler = None
        self.keep_idx = None
        self.running  = False
        self.start_time = None
        self.pred_logger = None

        self.ws_clients = set()
        self._ws_lock = threading.Lock()

        self.prediction_history = deque(maxlen=2000)
        self.price_history      = deque(maxlen=2000)

        self._pending_predictions = deque()
        self._pending_lock = threading.Lock()
        self.resolved_logger = None
        self._unresolved_path = None
        self.stats = LiveStatsTracker()

        self.regime_gate = RegimeGate(GATE_CONFIG_PATH)

    def load_artifacts(self):
        print("[MODEL] Loading V4 artifacts...")

        self.model = xgb.Booster()
        self.model.load_model(MODEL_PATH)
        print(f"  Model: {MODEL_PATH}")
        print(f"  Trees: {self.model.num_boosted_rounds()}")

        self.scaler = SimpleScaler.load(SCALER_PATH)
        print(f"  Scaler: {SCALER_PATH}")
        print(f"    mean shape: {self.scaler.mean_.shape}")
        print(f"    std  shape: {self.scaler.std_.shape}")

        with open(KEEPIDX_PATH, 'rb') as f:
            self.keep_idx = pickle.load(f)
        print(f"  keep_idx: {len(self.keep_idx)} features kept from 12")
        print(f"  Indices: {self.keep_idx}")

        n_expected = len(self.keep_idx)
        n_scaler = len(self.scaler.mean_)
        if n_expected != n_scaler:
            print(f"  [WARN] Feature count mismatch: keep_idx={n_expected}, scaler={n_scaler}")

    def predict(self):
        arrays = self.buffer.get_arrays()
        if arrays is None:
            return None

        ts_us, prices, qty, side = arrays

        span_sec = (ts_us[-1] - ts_us[0]) / 1e6
        if span_sec < MIN_BUFFER_SPAN_SEC:
            return None

        features = compute_features_v4(ts_us, prices, qty, side)

        # Compute trade count in last 300s for volume gate
        w_us = 300 * 1_000_000
        trade_count_300s = float(np.sum(ts_us >= (ts_us[-1] - w_us)))

        features_selected = features[self.keep_idx]
        features_norm = self.scaler.transform(features_selected.reshape(1, -1))
        features_norm = np.nan_to_num(features_norm, nan=0.0, posinf=0.0, neginf=0.0)

        dmat = xgb.DMatrix(features_norm.astype(np.float32))
        pred_log_return = float(self.model.predict(dmat)[0])

        # --- Regime Gate ---
        should_trade, rejection_reason, gate_details = self.regime_gate.check(
            features, pred_log_return, trade_count_300s
        )

        current_price = float(prices[-1])
        predicted_price = current_price * np.exp(pred_log_return)
        timestamp_ms = int(ts_us[-1] / 1000)

        return current_price, predicted_price, pred_log_return, timestamp_ms, should_trade, rejection_reason, gate_details

    def _enqueue_prediction(self, timestamp_ms, current_price, predicted_price,
                            pred_lr, buffer_size, buffer_span,
                            gate_decision, rejection_reason):
        bps = abs(pred_lr) * 10000
        direction = "UP" if pred_lr > 0 else "DN"
        entry = {
            'resolve_after_ms': timestamp_ms + (LOOK_AHEAD_SECONDS * 1000),
            'price_at_prediction': current_price,
            'row': {
                'wall_clock': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'timestamp_ms': timestamp_ms,
                'current_price': f"{current_price:.2f}",
                'predicted_price_5m': f"{predicted_price:.2f}",
                'pred_log_return': pred_lr,
                'pred_bps': f"{bps:.4f}",
                'direction': direction,
                'buffer_ticks': buffer_size,
                'buffer_span_s': f"{buffer_span:.0f}",
                'gate_decision': gate_decision,
                'rejection_reason': rejection_reason if rejection_reason else '',
            },
        }
        with self._pending_lock:
            self._pending_predictions.append(entry)
            qlen = len(self._pending_predictions)
        if qlen > 1000:
            print(f"[WARN] Pending resolution queue is large: {qlen} entries")

    def _check_resolutions(self, tick_timestamp_ms, tick_price):
        with self._pending_lock:
            while self._pending_predictions:
                head = self._pending_predictions[0]
                if tick_timestamp_ms < head['resolve_after_ms']:
                    break
                self._pending_predictions.popleft()

                price_at_pred = head['price_at_prediction']
                realized = math.log(tick_price / price_at_pred)
                pred_lr = head['row']['pred_log_return']

                if self.resolved_logger:
                    self.resolved_logger.log(head['row'], realized, tick_price)

                self.stats.record(pred_lr, realized, head['row']['timestamp_ms'])

                realized_bps = realized * 10_000
                pred_bps = pred_lr * 10_000
                if abs(realized) < 1e-5:
                    dir_tag = "FLAT"
                elif (realized > 0) == (pred_lr > 0):
                    dir_tag = "CORRECT"
                else:
                    dir_tag = "WRONG"

                pred_time = head['row']['timestamp_ms']
                print(f"[RESOLVED] T={pred_time} pred={pred_bps:+.2f}bps "
                      f"real={realized_bps:+.2f}bps dir={dir_tag}")

        if self.stats.should_report():
            self.stats.report()

    async def broadcast(self, message):
        with self._ws_lock:
            clients = self.ws_clients.copy()
        if clients:
            msg = json.dumps(message)
            dead = set()
            for ws in clients:
                try:
                    await ws.send(msg)
                except Exception:
                    dead.add(ws)
            if dead:
                with self._ws_lock:
                    self.ws_clients -= dead

    async def ws_handler(self, websocket):
        with self._ws_lock:
            self.ws_clients.add(websocket)
        print(f"[DASH] Client connected ({len(self.ws_clients)} total)")
        try:
            history_msg = {
                'type': 'history',
                'prices': list(self.price_history),
                'predictions': list(self.prediction_history),
            }
            await websocket.send(json.dumps(history_msg))
            async for _ in websocket:
                pass
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            with self._ws_lock:
                self.ws_clients.discard(websocket)

    async def binance_ws_listener(self):
        import websockets as ws_lib

        while self.running:
            try:
                print(f"[WS] Connecting to Binance...")
                async with ws_lib.connect(BINANCE_WS_URL, ping_interval=20, ping_timeout=10) as ws:
                    print(f"[WS] Connected.")
                    async for msg in ws:
                        if not self.running:
                            break
                        try:
                            data = json.loads(msg)
                            tick_ts = data['T']
                            tick_price = float(data['p'])
                            self.buffer.add_tick(
                                trade_id=data['t'],
                                timestamp_ms=tick_ts,
                                price=tick_price,
                                qty=float(data['q']),
                                is_buyer_maker=data['m'],
                            )
                            self._check_resolutions(tick_ts, tick_price)
                        except (KeyError, ValueError):
                            pass
            except Exception as e:
                if not self.running:
                    break
                print(f"[WS] Disconnected: {e}. Reconnecting in 3s...")
                await asyncio.sleep(3)

    async def inference_loop(self):
        if RUN_DURATION_MINUTES > 0:
            print(f"[INFER] Starting inference loop (every {INFERENCE_INTERVAL_SEC}s)")
            print(f"[INFER] Auto-shutdown in {RUN_DURATION_MINUTES} minutes")
        else:
            print(f"[INFER] Starting inference loop (every {INFERENCE_INTERVAL_SEC}s)")
            print(f"[INFER] Running indefinitely — Ctrl+C to stop")
            print(f"[INFER] Need ~42 independent samples (~3.5h) for statistical verdict")

        while self.running and self.buffer.time_span_seconds < MIN_BUFFER_SPAN_SEC:
            span = self.buffer.time_span_seconds
            print(f"[INFER] Buffer: {len(self.buffer)} ticks, {span:.0f}s span. "
                  f"Need {MIN_BUFFER_SPAN_SEC}s. Waiting...")
            await asyncio.sleep(5)

        if not self.running:
            return

        print(f"[INFER] Buffer ready. Starting predictions.")
        self.start_time = time.time()
        pred_count = 0

        while self.running:
            if RUN_DURATION_MINUTES > 0:
                elapsed = time.time() - self.start_time
                if elapsed >= RUN_DURATION_MINUTES * 60:
                    break

            try:
                result = self.predict()
                if result is not None:
                    current_price, predicted_price, pred_lr, timestamp_ms, should_trade, rejection_reason, gate_details = result
                    pred_time_ms = timestamp_ms + (LOOK_AHEAD_SECONDS * 1000)
                    pred_count += 1

                    price_point = {
                        'time': timestamp_ms,
                        'price': round(current_price, 2),
                    }
                    pred_point = {
                        'time': pred_time_ms,
                        'price': round(predicted_price, 2),
                        'made_at': timestamp_ms,
                        'log_return': round(pred_lr, 8),
                        'should_trade': should_trade,
                        'rejection_reason': rejection_reason,
                    }

                    self.price_history.append(price_point)
                    self.prediction_history.append(pred_point)

                    await self.broadcast({
                        'type': 'tick',
                        'price': price_point,
                        'prediction': pred_point,
                        'buffer_size': len(self.buffer),
                        'buffer_span_s': round(self.buffer.time_span_seconds, 0),
                    })

                    gate_decision = "TRADE" if should_trade else "SKIP"

                    if self.pred_logger:
                        self.pred_logger.log(
                            timestamp_ms, current_price, predicted_price,
                            pred_lr, len(self.buffer),
                            self.buffer.time_span_seconds,
                            gate_decision, rejection_reason,
                        )

                    self._enqueue_prediction(
                        timestamp_ms, current_price, predicted_price,
                        pred_lr, len(self.buffer),
                        self.buffer.time_span_seconds,
                        gate_decision, rejection_reason,
                    )

                    direction = "UP" if pred_lr > 0 else "DN"
                    bps = abs(pred_lr) * 10000
                    elapsed_min = (time.time() - self.start_time) / 60
                    now_str = datetime.now().strftime('%H:%M:%S')
                    trade_tag = "TRADE" if should_trade else f"SKIP({rejection_reason})"

                    if pred_count <= 5 or pred_count % 6 == 0:
                        n_indep = self.stats.indep_correct + self.stats.indep_wrong + self.stats.indep_flat
                        print(f"[{now_str}] #{pred_count} ${current_price:,.2f} -> "
                              f"${predicted_price:,.2f} ({direction} {bps:.1f}bps) [{trade_tag}] | "
                              f"buf={len(self.buffer)} | {elapsed_min:.0f}m")

            except Exception as e:
                print(f"[INFER] Error: {e}")

            await asyncio.sleep(INFERENCE_INTERVAL_SEC)

        elapsed_total = (time.time() - self.start_time) / 60
        print(f"\n[DONE] {pred_count} predictions in {elapsed_total:.1f} minutes")
        self.stats.report()
        if self.pred_logger:
            print(f"[DONE] Log saved: {self.pred_logger.filepath}")
            self.pred_logger.close()
        if self.resolved_logger:
            print(f"[DONE] Resolved predictions ({self.resolved_logger.count}): "
                  f"{self.resolved_logger.filepath}")
            self.resolved_logger.close()
        with self._pending_lock:
            remaining_pending = list(self._pending_predictions)
        if remaining_pending and self._unresolved_path:
            write_unresolved_csv(self._unresolved_path, remaining_pending)
        self.regime_gate.report()
        self.running = False

    async def run(self):
        self.running = True

        self.load_artifacts()

        print("\n[STARTUP] Phase 1: Starting Binance WebSocket...")
        ws_task = asyncio.create_task(self.binance_ws_listener())
        await asyncio.sleep(2)

        print("[STARTUP] Phase 2: Fetching 60 min history from REST...")
        hist_trades = await asyncio.get_event_loop().run_in_executor(
            None, fetch_historical_trades, 60
        )

        print("[STARTUP] Phase 3: Merging into buffer...")
        self.buffer.add_ticks_bulk(hist_trades)
        print(f"  Buffer: {len(self.buffer)} ticks, "
              f"{self.buffer.time_span_seconds:.0f}s span")

        arrays = self.buffer.get_arrays()
        if arrays is not None:
            ts_us, prices, _, _ = arrays
            step = max(1, len(prices) // 360)
            for i in range(0, len(prices), step):
                self.price_history.append({
                    'time': int(ts_us[i] / 1000),
                    'price': round(float(prices[i]), 2),
                })

        print(f"\n[DASH] Starting dashboard server on ws://{DASHBOARD_HOST}:{DASHBOARD_PORT}")
        server = await websockets.serve(
            self.ws_handler, DASHBOARD_HOST, DASHBOARD_PORT
        )

        infer_task = asyncio.create_task(self.inference_loop())

        duration_str = f"{RUN_DURATION_MINUTES} minutes" if RUN_DURATION_MINUTES > 0 else "INDEFINITE (Ctrl+C to stop)"
        print(f"\n{'='*60}")
        print(f"LIVE ENGINE v4 — CONTINUOUS EVALUATION")
        print(f"  Model     : {MODEL_PATH}")
        print(f"  Features  : {len(FEATURE_COLUMNS)} (V4, 12 features)")
        print(f"  Interval  : {INFERENCE_INTERVAL_SEC}s")
        print(f"  Duration  : {duration_str}")
        print(f"  Target    : 42 independent samples (~3.5 hours)")
        print(f"  Stats     : printed every {STATS_REPORT_INTERVAL}s")
        print(f"  Dashboard : open live_dashboard.html in browser")
        print(f"  Log       : {LOG_CSV}")
        print(f"  Ctrl+C to stop (data is saved on shutdown)")
        print(f"{'='*60}\n")

        try:
            await asyncio.gather(ws_task, infer_task)
        except asyncio.CancelledError:
            pass
        finally:
            self.running = False
            server.close()
            await server.wait_closed()


# =============================================================================
# ENTRY POINT
# =============================================================================

def main():
    global LOG_CSV

    os.makedirs(LOG_DIR, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    LOG_CSV = os.path.join(LOG_DIR, f"predictions_{timestamp}.csv")

    resolved_csv = os.path.join(LOG_DIR, f"resolved_predictions_{timestamp}.csv")
    unresolved_csv = os.path.join(LOG_DIR, f"unresolved_predictions_{timestamp}.csv")

    engine = LiveEngine()
    engine.pred_logger = PredictionLogger(LOG_CSV)
    engine.resolved_logger = ResolvedLogger(resolved_csv)
    engine._unresolved_path = unresolved_csv

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    def shutdown(sig, frame):
        print("\n[SHUTDOWN] Stopping...")
        engine.running = False
        if engine.pred_logger:
            engine.pred_logger.close()
        if engine.resolved_logger:
            engine.resolved_logger.close()
        with engine._pending_lock:
            remaining = list(engine._pending_predictions)
        if remaining and engine._unresolved_path:
            write_unresolved_csv(engine._unresolved_path, remaining)
        for task in asyncio.all_tasks(loop):
            task.cancel()

    signal.signal(signal.SIGINT, shutdown)

    try:
        loop.run_until_complete(engine.run())
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    finally:
        if engine.pred_logger:
            try:
                engine.pred_logger.close()
            except Exception:
                pass
        if engine.resolved_logger:
            try:
                engine.resolved_logger.close()
            except Exception:
                pass
        loop.close()
        print("[SHUTDOWN] Done.")


if __name__ == "__main__":
    main()
