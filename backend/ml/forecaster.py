"""
Multi-Horizon Price Forecaster for OSRS Flipping AI
Uses LightGBM (with sklearn fallback) to predict price direction,
future price, and confidence across 6 time horizons.
"""

import json
import logging
import os
import pickle
import statistics
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

from backend.ml.feature_engine import FeatureEngine

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Try LightGBM first, fall back to sklearn
# ---------------------------------------------------------------------------

try:
    import lightgbm as lgb

    LGBMClassifier = lgb.LGBMClassifier
    LGBMRegressor = lgb.LGBMRegressor
    HAS_LIGHTGBM = True
    logger.info("LightGBM available - using LightGBM models")
except ImportError:
    try:
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

        LGBMClassifier = RandomForestClassifier  # type: ignore[misc]
        LGBMRegressor = RandomForestRegressor  # type: ignore[misc]
        HAS_LIGHTGBM = False
        logger.info("LightGBM not installed - falling back to sklearn RandomForest")
    except ImportError:
        LGBMClassifier = None  # type: ignore[misc]
        LGBMRegressor = None  # type: ignore[misc]
        HAS_LIGHTGBM = False
        logger.warning("Neither LightGBM nor sklearn available - ML models disabled")


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


class MultiHorizonForecaster:
    """
    Trains and serves 18 models (6 horizons x 3 targets):
      - direction_model: LGBMClassifier -> up / down / flat
      - price_model: LGBMRegressor -> predicted price
      - confidence_model: LGBMRegressor -> prediction error magnitude
    """

    # Track dirs we already warned about to avoid repeating the same warning
    # across many short-lived predictor instances.
    _missing_manifest_warned_dirs: set[str] = set()

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

        Returns
        -------
        dict
            Training metrics: accuracy, mae, etc.
        """
        if LGBMClassifier is None:
            logger.error("No ML library available for training")
            return {}

        if not features_list:
            logger.warning(f"No training data for horizon {horizon}")
            return {}

        # Convert features to 2D list aligned by feature_names
        X = self._features_to_matrix(features_list)
        n_samples = len(X)

        metrics = {}

        # 1. Direction classifier
        direction_labels = targets.get("direction", [])
        if len(direction_labels) == n_samples:
            direction_model = self._create_classifier(horizon)
            try:
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
                price_model.fit(X, price_targets)
                self.models[f"{horizon}_price"] = price_model
                # Training MAE
                preds = price_model.predict(X)
                mae = statistics.mean(abs(p - a) for p, a in zip(preds, price_targets))
                metrics["price_mae"] = mae
                logger.info(f"[{horizon}] Price model trained: MAE={mae:.1f}")
            except Exception as e:
                logger.error(f"[{horizon}] Price training failed: {e}")

        # 3. Confidence regressor (predicts error magnitude)
        error_targets = targets.get("error", [])
        if len(error_targets) == n_samples:
            confidence_model = self._create_regressor(horizon)
            try:
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

        Parameters
        ----------
        item_id : int
            Item ID (used for context in fallback).
        features : dict
            Feature vector from FeatureEngine.

        Returns
        -------
        dict
            {
                "1m": {"buy": int, "sell": int, "direction": str, "confidence": float},
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
            has_confidence = f"{horizon}_confidence" in self.models

            if has_direction and has_price:
                # ML prediction
                results[horizon] = self._ml_predict(
                    horizon, features, current_price, spread_pct,
                )
            else:
                # Statistical fallback
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
        X = [self._features_to_row(features)]

        # Direction
        direction_model = self.models.get(f"{horizon}_direction")
        if direction_model is not None:
            try:
                direction_idx = int(direction_model.predict(X)[0])
                direction = DIRECTION_NAMES.get(direction_idx, "flat")
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

        # Confidence
        confidence_model = self.models.get(f"{horizon}_confidence")
        if confidence_model is not None:
            try:
                predicted_error = float(confidence_model.predict(X)[0])
                # Convert error magnitude to confidence: lower error -> higher confidence
                # Normalize: error as percentage of price
                if current_price > 0:
                    error_pct = predicted_error / current_price
                    confidence = max(0.0, min(1.0, 1.0 - error_pct * 10))
                else:
                    confidence = 0.5
            except Exception:
                confidence = 0.5
        else:
            confidence = 0.5

        # Derive buy/sell from predicted price
        half_spread = abs(spread_pct) / 200.0 * predicted_price if predicted_price > 0 else 0
        predicted_buy = int(predicted_price + half_spread)
        predicted_sell = int(predicted_price - half_spread)

        return {
            "buy": max(1, predicted_buy),
            "sell": max(1, predicted_sell),
            "direction": direction,
            "confidence": round(confidence, 4),
        }

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
            return {"buy": 0, "sell": 0, "direction": "flat", "confidence": 0.0}

        # Direction from momentum
        momentum_1x = features.get("momentum_1x", 0.0)
        momentum_2x = features.get("momentum_2x", 0.0)
        avg_momentum = (momentum_1x + momentum_2x) / 2.0

        if avg_momentum > 0.002:
            direction = "up"
        elif avg_momentum < -0.002:
            direction = "down"
        else:
            direction = "flat"

        # Price prediction: VWAP deviation + trend projection
        vwap_dev = features.get("vwap_deviation", 0.0)
        horizon_secs = HORIZON_SECONDS.get(horizon, 300)

        # Project trend forward, with damping for longer horizons
        damping = 1.0 / (1.0 + horizon_secs / 3600.0)
        trend_projection = avg_momentum * damping

        # Predicted price: revert toward VWAP + add trend
        predicted_price = current_price * (1.0 - vwap_dev * 0.3 + trend_projection)

        # Clamp: don't predict more than 5% move per horizon
        max_move = current_price * 0.05 * (horizon_secs / 3600.0)
        predicted_price = max(
            current_price - max_move,
            min(current_price + max_move, predicted_price),
        )

        # Confidence from volatility (z_score, atr)
        z_score = abs(features.get("z_score", 0.0))
        rsi = features.get("rsi_14", 50.0)

        # Lower confidence when z_score is extreme or RSI is extreme
        base_confidence = 0.5
        if z_score < 1.0:
            base_confidence += 0.15
        elif z_score > 2.0:
            base_confidence -= 0.15

        if 35 < rsi < 65:
            base_confidence += 0.1
        elif rsi < 20 or rsi > 80:
            base_confidence -= 0.1

        # Longer horizons inherently less confident
        horizon_penalty = horizon_secs / 86400.0 * 0.2
        confidence = max(0.05, min(0.95, base_confidence - horizon_penalty))

        # Derive buy/sell
        half_spread = abs(spread_pct) / 200.0 * predicted_price if predicted_price > 0 else 0
        predicted_buy = int(predicted_price + half_spread)
        predicted_sell = int(predicted_price - half_spread)

        return {
            "buy": max(1, predicted_buy),
            "sell": max(1, predicted_sell),
            "direction": direction,
            "confidence": round(confidence, 4),
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
            if self.model_dir not in MultiHorizonForecaster._missing_manifest_warned_dirs:
                logger.warning(f"No model manifest found at {manifest_path}")
                MultiHorizonForecaster._missing_manifest_warned_dirs.add(self.model_dir)
            else:
                logger.debug(f"No model manifest found at {manifest_path}")
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

    def _create_classifier(self, horizon: str):
        """Create a classifier with horizon-appropriate hyperparameters."""
        if HAS_LIGHTGBM:
            return LGBMClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.05,
                num_leaves=31,
                min_child_samples=20,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=0.1,
                random_state=42,
                verbose=-1,
            )
        elif LGBMClassifier is not None:
            # sklearn RandomForestClassifier fallback
            return LGBMClassifier(
                n_estimators=200,
                max_depth=8,
                min_samples_leaf=20,
                random_state=42,
                n_jobs=-1,
            )
        return None

    def _create_regressor(self, horizon: str):
        """Create a regressor with horizon-appropriate hyperparameters."""
        if HAS_LIGHTGBM:
            return LGBMRegressor(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.05,
                num_leaves=31,
                min_child_samples=20,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=0.1,
                random_state=42,
                verbose=-1,
            )
        elif LGBMRegressor is not None:
            return LGBMRegressor(
                n_estimators=200,
                max_depth=8,
                min_samples_leaf=20,
                random_state=42,
                n_jobs=-1,
            )
        return None

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def status(self) -> Dict[str, Any]:
        """Return model status for all horizons."""
        status = {
            "backend": "lightgbm" if HAS_LIGHTGBM else "sklearn" if LGBMClassifier else "none",
            "model_dir": self.model_dir,
            "horizons": {},
        }
        for horizon in HORIZONS:
            status["horizons"][horizon] = {
                "direction": f"{horizon}_direction" in self.models,
                "price": f"{horizon}_price" in self.models,
                "confidence": f"{horizon}_confidence" in self.models,
            }
        return status
