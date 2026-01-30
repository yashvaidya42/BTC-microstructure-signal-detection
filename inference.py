#!/usr/bin/env python3
"""
INFERENCE MODULE FOR BTC TICK PREDICTION
=========================================
Use this module to make predictions with the trained LightGBM model.

Features:
- Load trained model and calibrator
- Apply same feature transformations as training
- Return calibrated probabilities and trading signals
"""

import numpy as np
import lightgbm as lgb
import pickle
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple, Optional, Dict
from enum import IntEnum


class Signal(IntEnum):
    """Trading signal enum."""
    SHORT = 0   # DOWN prediction
    HOLD = 1    # NEUTRAL prediction
    LONG = 2    # UP prediction


@dataclass
class PredictionResult:
    """Container for prediction results."""
    signal: Signal
    probabilities: np.ndarray  # [p_down, p_neutral, p_up]
    confidence: float
    raw_prediction: int
    is_confident: bool


class BTCPredictor:
    """
    BTC Tick Prediction Inference Class.
    
    Usage:
        predictor = BTCPredictor.load("./model_output")
        
        # Single prediction
        features = np.array([...])  # Shape (12,)
        result = predictor.predict_single(features)
        
        # Batch prediction
        features_batch = np.array([...])  # Shape (n, 12)
        results = predictor.predict_batch(features_batch)
    """
    
    def __init__(
        self,
        model: lgb.Booster,
        calibrator,
        config: Dict,
        scaler_mean: Optional[np.ndarray] = None,
        scaler_std: Optional[np.ndarray] = None
    ):
        self.model = model
        self.calibrator = calibrator
        self.config = config
        self.scaler_mean = scaler_mean
        self.scaler_std = scaler_std
        
        self.confidence_threshold = config.get('confidence_threshold', 0.55)
        self.log_transform_features = config.get('log_transform_features', (3,))
        self.feature_names = config.get('feature_names', [])
    
    @classmethod
    def load(cls, model_dir: str, scaler_path: Optional[str] = None) -> 'BTCPredictor':
        """
        Load predictor from saved artifacts.
        
        Args:
            model_dir: Directory containing model files
            scaler_path: Path to scaler pickle (optional, for raw feature input)
        
        Returns:
            BTCPredictor instance
        """
        model_dir = Path(model_dir)
        
        # Load model
        model = lgb.Booster(model_file=str(model_dir / "model_lgbm.txt"))
        
        # Load calibrator
        with open(model_dir / "calibrator.pkl", 'rb') as f:
            calibrator = pickle.load(f)
        
        # Load config
        with open(model_dir / "artifacts.json", 'r') as f:
            artifacts = json.load(f)
        
        config = artifacts.get('config', {})
        
        # Load scaler if provided
        scaler_mean = None
        scaler_std = None
        if scaler_path:
            with open(scaler_path, 'rb') as f:
                scaler_data = pickle.load(f)
                scaler_mean = scaler_data['mean']
                scaler_std = scaler_data['std']
        
        return cls(model, calibrator, config, scaler_mean, scaler_std)
    
    def _apply_feature_fixes(self, X: np.ndarray) -> np.ndarray:
        """Apply feature transformations (log transform trade_intensity)."""
        X = X.copy()
        
        for idx in self.log_transform_features:
            if idx < X.shape[-1]:
                if X.ndim == 1:
                    col = X[idx]
                    col_min = col if isinstance(col, (int, float)) else col.min()
                    X[idx] = np.log(col - col_min + 1.0)
                else:
                    col = X[:, idx]
                    col_min = col.min()
                    X[:, idx] = np.log(col - col_min + 1.0)
        
        return X
    
    def _normalize(self, X: np.ndarray) -> np.ndarray:
        """Normalize features using scaler if available."""
        if self.scaler_mean is not None and self.scaler_std is not None:
            return (X - self.scaler_mean) / (self.scaler_std + 1e-8)
        return X
    
    def _raw_to_probs(self, raw_scores: np.ndarray) -> np.ndarray:
        """Convert raw LightGBM scores to probabilities (softmax)."""
        if raw_scores.ndim == 1:
            raw_scores = raw_scores.reshape(1, -1)
        
        exp_scores = np.exp(raw_scores - raw_scores.max(axis=1, keepdims=True))
        return exp_scores / exp_scores.sum(axis=1, keepdims=True)
    
    def predict_single(
        self, 
        features: np.ndarray,
        is_normalized: bool = True
    ) -> PredictionResult:
        """
        Make prediction for a single sample.
        
        Args:
            features: Feature array of shape (12,)
            is_normalized: Whether features are already normalized
        
        Returns:
            PredictionResult with signal, probabilities, and confidence
        """
        features = np.asarray(features).reshape(1, -1)
        results = self.predict_batch(features, is_normalized)
        return results[0]
    
    def predict_batch(
        self,
        features: np.ndarray,
        is_normalized: bool = True
    ) -> list:
        """
        Make predictions for multiple samples.
        
        Args:
            features: Feature array of shape (n_samples, 12)
            is_normalized: Whether features are already normalized
        
        Returns:
            List of PredictionResult objects
        """
        features = np.asarray(features)
        if features.ndim == 1:
            features = features.reshape(1, -1)
        
        # Normalize if needed
        if not is_normalized:
            features = self._normalize(features)
        
        # Apply feature fixes
        features = self._apply_feature_fixes(features)
        
        # Get raw predictions
        raw_scores = self.model.predict(features, raw_score=True)
        
        # Convert to probabilities
        probs_raw = self._raw_to_probs(raw_scores)
        
        # Calibrate
        probs_cal = self.calibrator.transform(probs_raw)
        
        # Generate results
        results = []
        for i in range(len(features)):
            probs = probs_cal[i]
            raw_pred = int(np.argmax(probs))
            confidence = float(probs.max())
            is_confident = confidence >= self.confidence_threshold
            
            # Signal: only trade if confident
            if is_confident:
                signal = Signal(raw_pred)
            else:
                signal = Signal.HOLD
            
            results.append(PredictionResult(
                signal=signal,
                probabilities=probs,
                confidence=confidence,
                raw_prediction=raw_pred,
                is_confident=is_confident
            ))
        
        return results
    
    def get_trading_signal(
        self,
        features: np.ndarray,
        is_normalized: bool = True,
        up_threshold: float = 0.55,
        down_threshold: float = 0.55
    ) -> Tuple[Signal, Dict]:
        """
        Get trading signal with custom thresholds.
        
        Args:
            features: Feature array
            is_normalized: Whether features are already normalized
            up_threshold: Minimum probability to signal LONG
            down_threshold: Minimum probability to signal SHORT
        
        Returns:
            (Signal, info_dict)
        """
        result = self.predict_single(features, is_normalized)
        
        info = {
            'probabilities': {
                'down': float(result.probabilities[0]),
                'neutral': float(result.probabilities[1]),
                'up': float(result.probabilities[2]),
            },
            'confidence': result.confidence,
            'raw_prediction': result.raw_prediction,
        }
        
        # Apply custom thresholds
        if result.probabilities[2] >= up_threshold:
            signal = Signal.LONG
        elif result.probabilities[0] >= down_threshold:
            signal = Signal.SHORT
        else:
            signal = Signal.HOLD
        
        return signal, info


# =============================================================================
# FEATURE COMPUTATION FOR LIVE DATA
# =============================================================================

class LiveFeatureComputer:
    """
    Compute features from live trade data.
    
    This class maintains state to compute rolling features
    from a stream of trade data.
    """
    
    def __init__(
        self,
        volatility_window: int = 1400,
        volume_window: int = 2000,
        rolling_stats_window: int = 150
    ):
        self.volatility_window = volatility_window
        self.volume_window = volume_window
        self.rolling_stats_window = rolling_stats_window
        
        # Buffers
        self.prices = []
        self.qtys = []
        self.timestamps = []
        self.sides = []
        
        self.max_buffer = max(volatility_window, volume_window) + 100
    
    def update(
        self,
        price: float,
        qty: float,
        timestamp: int,  # microseconds
        is_buyer_maker: bool
    ):
        """Add a new trade to the buffer."""
        self.prices.append(price)
        self.qtys.append(qty)
        self.timestamps.append(timestamp)
        self.sides.append(-1 if is_buyer_maker else 1)
        
        # Trim buffers
        if len(self.prices) > self.max_buffer:
            self.prices = self.prices[-self.max_buffer:]
            self.qtys = self.qtys[-self.max_buffer:]
            self.timestamps = self.timestamps[-self.max_buffer:]
            self.sides = self.sides[-self.max_buffer:]
    
    def compute_features(self) -> Optional[np.ndarray]:
        """
        Compute feature vector from current buffer.
        
        Returns:
            Feature array of shape (12,) or None if insufficient data
        """
        n = len(self.prices)
        
        if n < max(self.volatility_window, self.volume_window):
            return None
        
        prices = np.array(self.prices)
        qtys = np.array(self.qtys)
        timestamps = np.array(self.timestamps)
        sides = np.array(self.sides)
        
        # Current values (last trade)
        current_qty = qtys[-1]
        current_side = sides[-1]
        
        # Time delta
        if n >= 2:
            time_delta = (timestamps[-1] - timestamps[-2]) / 1e6  # Convert to seconds
        else:
            time_delta = 0.0
        
        # Log return
        if n >= 2 and prices[-2] > 0:
            log_return = np.log(prices[-1] / prices[-2])
        else:
            log_return = 0.0
        
        # Trade intensity
        trade_intensity = 1.0 / (time_delta + 1e-9)
        
        # Price acceleration
        if n >= 3 and prices[-3] > 0:
            log_return_prev = np.log(prices[-2] / prices[-3])
            price_acceleration = log_return - log_return_prev
        else:
            price_acceleration = 0.0
        
        # Volume imbalance (rolling)
        rs = self.rolling_stats_window
        buy_vol = np.sum(qtys[-rs:] * (sides[-rs:] == 1))
        sell_vol = np.sum(qtys[-rs:] * (sides[-rs:] == -1))
        volume_imbalance = (buy_vol - sell_vol) / (buy_vol + sell_vol + 1e-9)
        
        # Rolling signed volume
        vw = self.volume_window
        signed_vol = qtys[-vw:] * sides[-vw:]
        rolling_signed_volume = np.sum(signed_vol)
        
        # Price volatility (std of log returns)
        vw = self.volatility_window
        if n >= vw + 1:
            log_returns = np.log(prices[-vw:] / prices[-vw-1:-1])
            price_volatility = np.std(log_returns)
        else:
            price_volatility = 0.0
        
        # Volume volatility
        volume_volatility = np.std(qtys[-rs:]) if n >= rs else 0.0
        
        # Time delta std
        if n >= rs + 1:
            time_deltas = (timestamps[-rs:] - timestamps[-rs-1:-1]) / 1e6
            time_delta_std = np.std(time_deltas)
        else:
            time_delta_std = 0.0
        
        # Price distance from MA
        price_ma = np.mean(prices[-rs:])
        price_distance_from_ma = (prices[-1] - price_ma) / price_ma if price_ma > 0 else 0.0
        
        # Build feature vector (same order as training)
        features = np.array([
            current_qty,           # 0: qty
            time_delta,            # 1: time_delta
            log_return,            # 2: log_return
            trade_intensity,       # 3: trade_intensity
            price_acceleration,    # 4: price_acceleration
            volume_imbalance,      # 5: volume_imbalance
            rolling_signed_volume, # 6: rolling_signed_volume
            current_side,          # 7: side
            price_volatility,      # 8: price_volatility
            volume_volatility,     # 9: volume_volatility
            time_delta_std,        # 10: time_delta_std
            price_distance_from_ma # 11: price_distance_from_ma
        ], dtype=np.float32)
        
        return features
    
    def clear(self):
        """Clear all buffers."""
        self.prices = []
        self.qtys = []
        self.timestamps = []
        self.sides = []


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

def example_usage():
    """Example of how to use the inference module."""
    
    # Load predictor
    predictor = BTCPredictor.load(
        model_dir="./model_output",
        scaler_path="./scaler_lgbm.pkl"
    )
    
    # Create live feature computer
    feature_computer = LiveFeatureComputer()
    
    # Simulate receiving trade data
    # In production, this would come from a WebSocket feed
    sample_trades = [
        {'price': 45000.0, 'qty': 0.1, 'timestamp': 1700000000000000, 'is_buyer_maker': False},
        {'price': 45001.0, 'qty': 0.2, 'timestamp': 1700000000100000, 'is_buyer_maker': True},
        # ... more trades ...
    ]
    
    for trade in sample_trades:
        feature_computer.update(
            price=trade['price'],
            qty=trade['qty'],
            timestamp=trade['timestamp'],
            is_buyer_maker=trade['is_buyer_maker']
        )
    
    # Compute features (need enough data in buffer)
    features = feature_computer.compute_features()
    
    if features is not None:
        # Get prediction
        signal, info = predictor.get_trading_signal(
            features,
            is_normalized=False,  # Raw features need normalization
            up_threshold=0.55,
            down_threshold=0.55
        )
        
        print(f"Signal: {signal.name}")
        print(f"Probabilities: {info['probabilities']}")
        print(f"Confidence: {info['confidence']:.2%}")


if __name__ == "__main__":
    example_usage()
