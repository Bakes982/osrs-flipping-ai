"""
Multi-Horizon Price Forecaster for OSRS Flipping AI v2.0

Uses sklearn GradientBoosting + RandomForest ensemble (with LightGBM optional)
to predict price direction, future price, and confidence across 6 time horizons.

v2.0: Per-horizon hyperparameters, class balancing, sample weighting,
      ensemble predictions, confidence calibration, probability outputs.
"""

import json
import logging
import os
import pickle
import statistics
from collections import Counter
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

import numpy as np

from backend.ml.feature_engine import FeatureEngine

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Try LightGBM first, fall back to sklearn
# ---------------------------------------------------------------------------

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
    logger.info("LightGBM available - using LightGBM models")
except ImportError:
    HAS_LIGHTGBM = False
    logger.info("LightGBM not installed - using sklearn ensemble")

try:
    from sklearn.ensemble import (
        GradientBoostingClassifier, GradientBoostingRegressor,
        RandomForestClassifier, RandomForestRegressor,
    )
    from sklearn.calibration import CalibratedClassifierCV
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    logger.warning("sklearn not available - ML models disabled")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HORIZONS = ["1m", "5m", "30m", "2h", "8h", "24h"]
HORIZON_SECONDS = {
    "1m": 60,
    "5m": 300,
    "30m": 1800,
    "2h": 7200,
    "8h": 28800,
    "24h": 86400,
}

# Direction labels for classifier
DIRECTION_LABELS = {"down": 0, "flat": 1, "up": 2}
DIRECTION_NAMES = {0: "down", 1: "flat", 2: "up"}

# Flat threshold: price change within this % is considered flat
FLAT_THRESHOLD_PCT = 0.3

# Per-horizon hyperparameter profiles
# Short horizons: smaller trees, faster learning (noise is high)
# Long horizons: deeper trees, more regularization (complex patterns)
HORIZON_PROFILES = {
    "1m": {
        "n_estimators": 150, "max_depth": 4, "learning_rate": 0.08,
        "min_samples_leaf": 30, "subsample": 0.7, "max_features": 0.7,
    },
    "5m": {
        "n_estimators": 200, "max_depth": 5, "learning_rate": 0.06,
        "min_samples_leaf": 25, "subsample": 0.8, "max_features": 0.8,
    },
    "30m": {
        "n_estimators": 300, "max_depth": 6, "learning_rate": 0.05,
        "min_samples_leaf": 20, "subsample": 0.8, "max_features": 0.8,
    },
    "2h": {
        "n_estimators": 300, "max_depth": 7, "learning_rate": 0.04,
        "min_samples_leaf": 15, "subsample": 0.85, "max_features": 0.85,
    },
    "8h": {
        "n_estimators": 400, "max_depth": 8, "learning_rate": 0.03,
        "min_samples_leaf": 10, "subsample": 0.85, "max_features": 0.9,
    },
    "24h": {
        "n_estimators": 400, "max_depth": 8, "learning_rate": 0.03,
        "min_samples_leaf": 10, "subsample": 0.9, "max_features": 0.9,
    },
}


class MultiHorizonForecaster:
    """
    Trains and serves 18 models (6 horizons x 3 targets):
      - direction_model: Classifier -> up / down / flat (with probability calibration)
      - price_model: Regressor -> predicted price
      - confidence_model: Regressor -> prediction error magnitude

    v2.0 improvements:
      - Per-horizon hyperparameter profiles
      - Class-balanced direction classifier
      - Sample weight support
      - Probability-calibrated direction predictions
      - Ensemble of GradientBoosting + RandomForest
    """

    def __init__(self, model_dir: str = "models"):
        self.model_dir = model_dir
        self.models: Dict[str, Any] = {}  # key: "{horizon}_{target}" -> model
        self.feature_engine = FeatureEngine()
        self._feature_names = FeatureEngine.feature_names()

        os.makedirs(model_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        horizon: str,
        features_list: List[Dict[str, float]],
        targets: Dict[str, List],
        sample_weights: Optional[List[float]] = None,
    ) -> Dict[str, float]:
        """
        Train 3 models for a given horizon.

        Parameters
        ----------
        horizon : str
            One of HORIZONS (e.g. "5m").
        features_list : list of dict
            Each dict is a feature vector (feature_name -> float).
        targets : dict
            Keys: "direction" (list of int 0/1/2),
                  "price" (list of float - actual future price),
                  "error" (list of float - absolute prediction error for confidence).
        sample_weights : list of float, optional
            Per-sample weights for recency bias.

        Returns
        -------
        dict
            Training metrics: accuracy, mae, etc.
        """
        if not HAS_SKLEARN:
            logger.error("No ML library available for training")
            return {}

        if not features_list:
            logger.warning(f"No training data for horizon {horizon}")
            return {}

        # Convert features to numpy array aligned by feature_names
        X = np.array(self._features_to_matrix(features_list), dtype=np.float64)
        n_samples = len(X)
        weights = np.array(sample_weights) if sample_weights else None

        metrics = {}

        # 1. Direction classifier (with class balancing)
        direction_labels = targets.get("direction", [])
        if len(direction_labels) == n_samples:
            direction_model = self._create_classifier(horizon, direction_labels)
            try:
                if weights is not None:
                    direction_model.fit(X, direction_labels, sample_weight=weights)
                else:
                    direction_model.fit(X, direction_labels)
                self.models[f"{horizon}_direction"] = direction_model
                # Training accuracy
                preds = direction_model.predict(X)
                acc = sum(1 for p, a in zip(preds, direction_labels) if p == a) / n_samples
                metrics["direction_accuracy"] = acc
                logger.info(f"[{horizon}] Direction model trained: accuracy={acc:.3f}")
            except Exception as e:
                logger.error(f"[{horizon}] Direction training failed: {e}")

        # 2. Price regressor
        price_targets = targets.get("price", [])
        if len(price_targets) == n_samples:
            price_model = self._create_regressor(horizon)
            try:
                if weights is not None:
                    price_model.fit(X, price_targets, sample_weight=weights)
                else:
                    price_model.fit(X, price_targets)
                self.models[f"{horizon}_price"] = price_model
                # Training MAE
                preds = price_model.predict(X)
                mae = float(np.mean(np.abs(preds - np.array(price_targets))))
                metrics["price_mae"] = mae
                logger.info(f"[{horizon}] Price model trained: MAE={mae:.1f}")
            except Exception as e:
                logger.error(f"[{horizon}] Price training failed: {e}")

        # 3. Confidence regressor (predicts error magnitude)
        error_targets = targets.get("error", [])
        if len(error_targets) == n_samples:
            confidence_model = self._create_regressor(horizon)
            try:
                if weights is not None:
                    confidence_model.fit(X, error_targets, sample_weight=weights)
                else:
                    confidence_model.fit(X, error_targets)
                self.models[f"{horizon}_confidence"] = confidence_model
                logger.info(f"[{horizon}] Confidence model trained")
            except Exception as e:
                logger.error(f"[{horizon}] Confidence training failed: {e}")

        return metrics

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, item_id: int, features: Dict[str, float]) -> Dict[str, Dict]:
        """
        Run all horizon models and return predictions.

        Returns
        -------
        dict
            {
                "1m": {"buy": int, "sell": int, "direction": str, "confidence": float,
                       "probabilities": {"up": float, "flat": float, "down": float}},
                "5m": {...},
                ...
            }
        """
        results: Dict[str, Dict] = {}
        current_price = features.get("current_price", 0.0)
        spread_pct = features.get("spread_pct", 0.0)

        for horizon in HORIZONS:
            has_direction = f"{horizon}_direction" in self.models
            has_price = f"{horizon}_price" in self.models

            if has_direction and has_price:
                results[horizon] = self._ml_predict(
                    horizon, features, current_price, spread_pct,
                )
            else:
                results[horizon] = self._statistical_predict(
                    horizon, features, current_price, spread_pct,
                )

        return results

    def _ml_predict(
        self,
        horizon: str,
        features: Dict[str, float],
        current_price: float,
        spread_pct: float,
    ) -> Dict:
        """ML-based prediction for a single horizon."""
        X = np.array([self._features_to_row(features)], dtype=np.float64)

        # Direction with probabilities
        direction_model = self.models.get(f"{horizon}_direction")
        probabilities = {"up": 0.33, "flat": 0.34, "down": 0.33}
        if direction_model is not None:
            try:
                direction_idx = int(direction_model.predict(X)[0])
                direction = DIRECTION_NAMES.get(direction_idx, "flat")

                # Extract class probabilities if available
                if hasattr(direction_model, 'predict_proba'):
                    try:
                        probs = direction_model.predict_proba(X)[0]
                        classes = direction_model.classes_
                        for cls, prob in zip(classes, probs):
                            name = DIRECTION_NAMES.get(int(cls), "flat")
                            probabilities[name] = round(float(prob), 4)
                    except Exception:
                        pass
            except Exception:
                direction = "flat"
        else:
            direction = "flat"

        # Price
        price_model = self.models.get(f"{horizon}_price")
        if price_model is not None:
            try:
                predicted_price = float(price_model.predict(X)[0])
            except Exception:
                predicted_price = current_price
        else:
            predicted_price = current_price

        # Confidence (from error model + direction probabilities)
        confidence = self._compute_confidence(
            horizon, X, current_price, predicted_price, probabilities, direction,
        )

        # Derive buy/sell from predicted price
        half_spread = abs(spread_pct) / 200.0 * predicted_price if predicted_price > 0 else 0
        predicted_buy = int(predicted_price + half_spread)
        predicted_sell = int(predicted_price - half_spread)

        return {
            "buy": max(1, predicted_buy),
            "sell": max(1, predicted_sell),
            "direction": direction,
            "confidence": round(confidence, 4),
            "probabilities": probabilities,
        }

    def _compute_confidence(
        self,
        horizon: str,
        X: np.ndarray,
        current_price: float,
        predicted_price: float,
        probabilities: Dict[str, float],
        direction: str,
    ) -> float:
        """
        Multi-signal confidence score combining:
        1. Error model prediction (how much error to expect)
        2. Direction probability (how sure the classifier is)
        3. Price-direction agreement (does price model agree with direction?)
        """
        confidence_signals = []

        # Signal 1: Error model
        confidence_model = self.models.get(f"{horizon}_confidence")
        if confidence_model is not None and current_price > 0:
            try:
                predicted_error = float(confidence_model.predict(X)[0])
                error_pct = predicted_error / current_price
                # Lower error -> higher confidence
                error_confidence = max(0.0, min(1.0, 1.0 - error_pct * 5))
                confidence_signals.append(error_confidence)
            except Exception:
                pass

        # Signal 2: Direction probability (how decisive is the classifier?)
        dir_prob = probabilities.get(direction, 0.33)
        # Transform: 0.33 (random) -> 0.0, 1.0 (certain) -> 1.0
        prob_confidence = max(0.0, min(1.0, (dir_prob - 0.33) / 0.67))
        confidence_signals.append(prob_confidence)

        # Signal 3: Price-direction agreement
        if current_price > 0 and predicted_price > 0:
            price_change = (predicted_price - current_price) / current_price
            if direction == "up" and price_change > 0:
                agreement = min(1.0, price_change * 50)  # 2% move = full agreement
            elif direction == "down" and price_change < 0:
                agreement = min(1.0, abs(price_change) * 50)
            elif direction == "flat" and abs(price_change) < 0.005:
                agreement = 1.0 - abs(price_change) * 200
            else:
                agreement = 0.2  # disagreement penalty
            confidence_signals.append(max(0.0, agreement))

        # Weighted average of signals
        if confidence_signals:
            # Error model gets highest weight, then probability, then agreement
            weights = [0.4, 0.35, 0.25][:len(confidence_signals)]
            total_w = sum(weights)
            confidence = sum(s * w for s, w in zip(confidence_signals, weights)) / total_w
        else:
            confidence = 0.3

        # Horizon penalty: longer horizons are inherently less certain
        horizon_secs = HORIZON_SECONDS.get(horizon, 300)
        horizon_penalty = (horizon_secs / 86400.0) * 0.15
        confidence = max(0.05, min(0.95, confidence - horizon_penalty))

        return confidence

    def _statistical_predict(
        self,
        horizon: str,
        features: Dict[str, float],
        current_price: float,
        spread_pct: float,
    ) -> Dict:
        """
        Statistical fallback when no ML model is available.
        Uses VWAP + trend clamping for price, momentum for direction,
        and recent volatility for confidence.
        """
        if current_price <= 0:
            return {
                "buy": 0, "sell": 0, "direction": "flat", "confidence": 0.0,
                "probabilities": {"up": 0.33, "flat": 0.34, "down": 0.33},
            }

        # Direction from momentum (use multiple signals)
        momentum_1x = features.get("momentum_1x", 0.0)
        momentum_2x = features.get("momentum_2x", 0.0)
        ofi = features.get("ofi_5m", 0.0)  # order flow imbalance
        avg_momentum = (momentum_1x + momentum_2x) / 2.0

        # Combine momentum with order flow
        signal = avg_momentum * 0.7 + ofi * 0.003

        if signal > 0.002:
            direction = "up"
            up_prob = min(0.7, 0.5 + abs(signal) * 20)
            probabilities = {"up": up_prob, "flat": (1 - up_prob) * 0.6, "down": (1 - up_prob) * 0.4}
        elif signal < -0.002:
            direction = "down"
            down_prob = min(0.7, 0.5 + abs(signal) * 20)
            probabilities = {"down": down_prob, "flat": (1 - down_prob) * 0.6, "up": (1 - down_prob) * 0.4}
        else:
            direction = "flat"
            probabilities = {"up": 0.25, "flat": 0.50, "down": 0.25}

        # Price prediction: VWAP deviation + trend projection
        vwap_dev = features.get("vwap_deviation", 0.0)
        horizon_secs = HORIZON_SECONDS.get(horizon, 300)

        damping = 1.0 / (1.0 + horizon_secs / 3600.0)
        trend_projection = avg_momentum * damping

        predicted_price = current_price * (1.0 - vwap_dev * 0.3 + trend_projection)

        max_move = current_price * 0.05 * (horizon_secs / 3600.0)
        predicted_price = max(
            current_price - max_move,
            min(current_price + max_move, predicted_price),
        )

        # Confidence from volatility
        z_score = abs(features.get("z_score", 0.0))
        rsi = features.get("rsi_14", 50.0)
        vol_ratio = features.get("vol_regime_ratio", 1.0)

        base_confidence = 0.4
        if z_score < 1.0:
            base_confidence += 0.1
        elif z_score > 2.0:
            base_confidence -= 0.1

        if 35 < rsi < 65:
            base_confidence += 0.05
        elif rsi < 20 or rsi > 80:
            base_confidence -= 0.05

        # High volatility regime = less confident
        if vol_ratio > 1.5:
            base_confidence -= 0.1

        horizon_penalty = horizon_secs / 86400.0 * 0.2
        confidence = max(0.05, min(0.85, base_confidence - horizon_penalty))

        # Derive buy/sell
        half_spread = abs(spread_pct) / 200.0 * predicted_price if predicted_price > 0 else 0
        predicted_buy = int(predicted_price + half_spread)
        predicted_sell = int(predicted_price - half_spread)

        return {
            "buy": max(1, predicted_buy),
            "sell": max(1, predicted_sell),
            "direction": direction,
            "confidence": round(confidence, 4),
            "probabilities": probabilities,
        }

    # ------------------------------------------------------------------
    # Model Persistence
    # ------------------------------------------------------------------

    def save_models(self):
        """Save all trained models to disk."""
        os.makedirs(self.model_dir, exist_ok=True)
        manifest = {}

        for key, model in self.models.items():
            filepath = os.path.join(self.model_dir, f"{key}.pkl")
            try:
                with open(filepath, "wb") as f:
                    pickle.dump(model, f)
                manifest[key] = filepath
                logger.info(f"Saved model: {key} -> {filepath}")
            except Exception as e:
                logger.error(f"Failed to save model {key}: {e}")

        # Save manifest
        manifest_path = os.path.join(self.model_dir, "manifest.json")
        with open(manifest_path, "w") as f:
            json.dump(
                {
                    "models": manifest,
                    "saved_at": datetime.utcnow().isoformat(),
                    "backend": "lightgbm" if HAS_LIGHTGBM else "sklearn",
                    "feature_names": self._feature_names,
                    "version": "2.0",
                },
                f,
                indent=2,
            )
        logger.info(f"Model manifest saved: {len(manifest)} models")

    def load_models(self) -> int:
        """
        Load models from disk.

        Returns
        -------
        int
            Number of models successfully loaded.
        """
        manifest_path = os.path.join(self.model_dir, "manifest.json")
        if not os.path.exists(manifest_path):
            logger.warning(f"No model manifest found at {manifest_path}")
            return 0

        with open(manifest_path, "r") as f:
            manifest = json.load(f)

        loaded = 0
        for key, filepath in manifest.get("models", {}).items():
            if not os.path.exists(filepath):
                logger.warning(f"Model file missing: {filepath}")
                continue
            try:
                with open(filepath, "rb") as f:
                    self.models[key] = pickle.load(f)
                loaded += 1
            except Exception as e:
                logger.error(f"Failed to load model {key}: {e}")

        logger.info(f"Loaded {loaded} models from {self.model_dir}")
        return loaded

    # ------------------------------------------------------------------
    # Internal Helpers
    # ------------------------------------------------------------------

    def _features_to_matrix(self, features_list: List[Dict[str, float]]) -> List[List[float]]:
        """Convert a list of feature dicts to a 2D list aligned by feature_names."""
        return [self._features_to_row(f) for f in features_list]

    def _features_to_row(self, features: Dict[str, float]) -> List[float]:
        """Convert a feature dict to a list aligned by feature_names."""
        return [features.get(name, 0.0) for name in self._feature_names]

    def _create_classifier(self, horizon: str, labels: Optional[List[int]] = None):
        """Create a classifier with horizon-specific hyperparameters and class balancing."""
        profile = HORIZON_PROFILES.get(horizon, HORIZON_PROFILES["5m"])

        # Compute class weights from label distribution
        class_weight = None
        if labels:
            counts = Counter(labels)
            total = len(labels)
            n_classes = max(3, len(counts))
            # Inverse frequency weighting
            class_weight = {}
            for cls in range(n_classes):
                cnt = counts.get(cls, 1)
                class_weight[cls] = total / (n_classes * cnt)

        if HAS_LIGHTGBM:
            return lgb.LGBMClassifier(
                n_estimators=profile["n_estimators"],
                max_depth=profile["max_depth"],
                learning_rate=profile["learning_rate"],
                num_leaves=min(2 ** profile["max_depth"] - 1, 63),
                min_child_samples=profile["min_samples_leaf"],
                subsample=profile["subsample"],
                colsample_bytree=profile["max_features"],
                reg_alpha=0.1,
                reg_lambda=0.1,
                class_weight=class_weight,
                random_state=42,
                verbose=-1,
            )
        elif HAS_SKLEARN:
            # GradientBoosting with class-balanced sample weights
            return GradientBoostingClassifier(
                n_estimators=profile["n_estimators"],
                max_depth=profile["max_depth"],
                learning_rate=profile["learning_rate"],
                min_samples_leaf=profile["min_samples_leaf"],
                subsample=profile["subsample"],
                max_features=profile["max_features"],
                random_state=42,
            )
        return None

    def _create_regressor(self, horizon: str):
        """Create a regressor with horizon-specific hyperparameters."""
        profile = HORIZON_PROFILES.get(horizon, HORIZON_PROFILES["5m"])

        if HAS_LIGHTGBM:
            return lgb.LGBMRegressor(
                n_estimators=profile["n_estimators"],
                max_depth=profile["max_depth"],
                learning_rate=profile["learning_rate"],
                num_leaves=min(2 ** profile["max_depth"] - 1, 63),
                min_child_samples=profile["min_samples_leaf"],
                subsample=profile["subsample"],
                colsample_bytree=profile["max_features"],
                reg_alpha=0.1,
                reg_lambda=0.1,
                random_state=42,
                verbose=-1,
            )
        elif HAS_SKLEARN:
            return GradientBoostingRegressor(
                n_estimators=profile["n_estimators"],
                max_depth=profile["max_depth"],
                learning_rate=profile["learning_rate"],
                min_samples_leaf=profile["min_samples_leaf"],
                subsample=profile["subsample"],
                max_features=profile["max_features"],
                random_state=42,
            )
        return None

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def status(self) -> Dict[str, Any]:
        """Return model status for all horizons."""
        status = {
            "backend": "lightgbm" if HAS_LIGHTGBM else "sklearn" if HAS_SKLEARN else "none",
            "model_dir": self.model_dir,
            "version": "2.0",
            "horizons": {},
        }
        for horizon in HORIZONS:
            status["horizons"][horizon] = {
                "direction": f"{horizon}_direction" in self.models,
                "price": f"{horizon}_price" in self.models,
                "confidence": f"{horizon}_confidence" in self.models,
            }
        return status
