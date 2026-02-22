"""
Inference Engine for OSRS Flipping AI
Loads trained models, computes features from current price data,
and returns multi-horizon predictions with statistical fallback.
"""

import logging
import statistics
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

from backend.database import (
    get_db, PriceSnapshot, FlipHistory, Prediction,
    get_price_history, get_item_flips, get_latest_price,
    insert_prediction, find_pending_predictions,
    update_prediction_outcome, find_snapshot_near_time,
)
from backend.ml.feature_engine import FeatureEngine, HORIZONS, HORIZON_SECONDS
from backend.ml.forecaster import MultiHorizonForecaster

logger = logging.getLogger(__name__)


class Predictor:
    """
    Inference engine that:
    - Loads trained models from disk
    - Computes features for a given item using current price data
    - Returns multi-horizon predictions
    - Falls back to statistical methods if no trained models exist
    """

    def __init__(self, model_dir: str = "models"):
        self.feature_engine = FeatureEngine()
        self.forecaster = MultiHorizonForecaster(model_dir=model_dir)
        self._models_loaded = False
        # Avoid retrying model discovery on every prediction when no manifest/models exist.
        self._models_checked = False
        self._feature_cache: Dict[int, Dict[str, Any]] = {}  # item_id -> {features, ts}
        self._cache_ttl = 60  # seconds

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def load_models(self) -> int:
        """
        Load trained models from disk.

        Returns
        -------
        int
            Number of models loaded.
        """
        count = self.forecaster.load_models()
        self._models_checked = True
        self._models_loaded = count > 0
        logger.info(
            f"Predictor initialized: {count} models loaded, "
            f"ML {'enabled' if self._models_loaded else 'disabled (using statistical fallback)'}"
        )
        return count

    # ------------------------------------------------------------------
    # Main Prediction Interface
    # ------------------------------------------------------------------

    def predict_item(
        self,
        item_id: int,
        snapshots: Optional[List[PriceSnapshot]] = None,
        flips: Optional[List[FlipHistory]] = None,
        save_to_db: bool = True,
    ) -> Dict[str, Dict]:
        """
        Generate multi-horizon predictions for an item.

        Parameters
        ----------
        item_id : int
            OSRS item ID.
        snapshots : list of PriceSnapshot, optional
            Price history. Fetched from DB if None.
        flips : list of FlipHistory, optional
            Flip history. Fetched from DB if None.
        save_to_db : bool
            Whether to log predictions to the Prediction table.

        Returns
        -------
        dict
            {
                "1m": {"buy": int, "sell": int, "direction": str, "confidence": float},
                "5m": {...},
                ...
                "_meta": {"item_id": int, "timestamp": str, "method": str, "features": dict}
            }
        """
        start = time.time()

        # Load models on first call (or if not checked yet) to avoid repeated
        # manifest scans/log spam when ML artifacts are absent.
        if not self._models_checked:
            self.load_models()

        # Compute features (with caching)
        features = self._get_features(item_id, snapshots, flips)

        if not features or features.get("current_price", 0) <= 0:
            logger.warning(f"Item {item_id}: No valid price data for prediction")
            return self._empty_prediction(item_id)

        # Run prediction through forecaster (handles ML vs fallback internally)
        predictions = self.forecaster.predict(item_id, features)

        # Determine method used
        has_ml = any(
            f"{h}_direction" in self.forecaster.models for h in HORIZONS
        )
        method = "ml" if has_ml else "statistical"

        # Add metadata
        predictions["_meta"] = {
            "item_id": item_id,
            "timestamp": datetime.utcnow().isoformat(),
            "method": method,
            "current_price": features.get("current_price", 0),
            "inference_ms": round((time.time() - start) * 1000, 1),
        }

        # Save predictions to DB
        if save_to_db:
            self._save_predictions(item_id, predictions, method)

        return predictions

    def predict_batch(
        self,
        item_ids: List[int],
        save_to_db: bool = True,
    ) -> Dict[int, Dict[str, Dict]]:
        """
        Predict for multiple items at once.

        Parameters
        ----------
        item_ids : list of int
            Item IDs to predict for.
        save_to_db : bool
            Whether to log predictions.

        Returns
        -------
        dict
            {item_id: prediction_dict}
        """
        results: Dict[int, Dict[str, Dict]] = {}
        start = time.time()

        for item_id in item_ids:
            try:
                results[item_id] = self.predict_item(
                    item_id, save_to_db=save_to_db,
                )
            except Exception as e:
                logger.error(f"Prediction failed for item {item_id}: {e}")
                results[item_id] = self._empty_prediction(item_id)

        elapsed = time.time() - start
        logger.info(
            f"Batch prediction: {len(item_ids)} items in {elapsed:.1f}s "
            f"({elapsed / max(len(item_ids), 1) * 1000:.0f}ms/item)"
        )

        return results

    # ------------------------------------------------------------------
    # Feature Computation with Caching
    # ------------------------------------------------------------------

    def _get_features(
        self,
        item_id: int,
        snapshots: Optional[List[PriceSnapshot]] = None,
        flips: Optional[List[FlipHistory]] = None,
    ) -> Dict[str, float]:
        """Get features with TTL-based caching."""
        now = time.time()

        # Check cache
        cached = self._feature_cache.get(item_id)
        if cached and (now - cached["ts"]) < self._cache_ttl:
            return cached["features"]

        # Compute fresh features
        features = self.feature_engine.compute_features(item_id, snapshots, flips)

        # Cache
        self._feature_cache[item_id] = {"features": features, "ts": now}

        return features

    def invalidate_cache(self, item_id: Optional[int] = None):
        """Clear the feature cache, optionally for a specific item."""
        if item_id is not None:
            self._feature_cache.pop(item_id, None)
        else:
            self._feature_cache.clear()

    # ------------------------------------------------------------------
    # Prediction Persistence
    # ------------------------------------------------------------------

    def _save_predictions(
        self,
        item_id: int,
        predictions: Dict[str, Dict],
        method: str,
    ):
        """Log predictions to the Prediction table for backtesting."""
        db = get_db()
        try:
            now = datetime.utcnow()
            for horizon in HORIZONS:
                pred_data = predictions.get(horizon)
                if not pred_data:
                    continue

                record = Prediction(
                    item_id=item_id,
                    timestamp=now,
                    horizon=horizon,
                    predicted_buy=pred_data.get("buy"),
                    predicted_sell=pred_data.get("sell"),
                    predicted_direction=pred_data.get("direction"),
                    confidence=pred_data.get("confidence"),
                    model_version=f"{method}_v1",
                )
                insert_prediction(db, record)
        except Exception as e:
            logger.error(f"Failed to save predictions for item {item_id}: {e}")
        finally:
            db.close()

    # ------------------------------------------------------------------
    # Outcome Recording (for accuracy tracking)
    # ------------------------------------------------------------------

    def record_outcomes(self, item_id: int):
        """
        Check past predictions whose horizon has elapsed and fill in
        actual outcomes. This enables accuracy tracking.
        """
        db = get_db()
        try:
            now = datetime.utcnow()

            for horizon in HORIZONS:
                horizon_secs = HORIZON_SECONDS[horizon]
                horizon_delta = timedelta(seconds=horizon_secs)

                # Find predictions that should have matured by now
                cutoff = now - horizon_delta
                pending = find_pending_predictions(
                    db, item_id, horizon, cutoff,
                )

                for pred in pending:
                    target_time = pred.timestamp + horizon_delta
                    # Find the actual price at target_time
                    actual = self._find_actual_price(
                        db, item_id, target_time,
                    )
                    if actual is None:
                        continue

                    actual_buy, actual_sell = actual

                    # Determine actual direction
                    actual_direction = None
                    if pred.predicted_buy and actual_buy:
                        # Compare predicted price to actual
                        # Use the feature's current_price at prediction time
                        # For simplicity, compare predicted_buy to actual_buy
                        if actual_buy > pred.predicted_buy * 1.003:
                            actual_direction = "up"
                        elif actual_buy < pred.predicted_buy * 0.997:
                            actual_direction = "down"
                        else:
                            actual_direction = "flat"

                    update_prediction_outcome(
                        db, pred.id, actual_buy, actual_sell, actual_direction,
                    )

                if pending:
                    logger.info(
                        f"[{item_id}][{horizon}] Recorded {len(pending)} outcomes"
                    )

        except Exception as e:
            logger.error(f"Failed to record outcomes for item {item_id}: {e}")
        finally:
            db.close()

    def _find_actual_price(
        self,
        db,
        item_id: int,
        target_time: datetime,
    ) -> Optional[tuple]:
        """Find the actual price closest to target_time."""
        snap = find_snapshot_near_time(db, item_id, target_time)
        if snap and snap.instant_buy and snap.instant_sell:
            return (snap.instant_buy, snap.instant_sell)
        return None

    # ------------------------------------------------------------------
    # Empty / Fallback Results
    # ------------------------------------------------------------------

    def _empty_prediction(self, item_id: int) -> Dict[str, Dict]:
        """Return an empty prediction dict when no data is available."""
        result: Dict[str, Any] = {}
        for horizon in HORIZONS:
            result[horizon] = {
                "buy": 0,
                "sell": 0,
                "direction": "flat",
                "confidence": 0.0,
            }
        result["_meta"] = {
            "item_id": item_id,
            "timestamp": datetime.utcnow().isoformat(),
            "method": "none",
            "current_price": 0,
            "inference_ms": 0,
        }
        return result

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def status(self) -> Dict[str, Any]:
        """Return predictor status."""
        return {
            "models_loaded": self._models_loaded,
            "cached_items": len(self._feature_cache),
            "cache_ttl_seconds": self._cache_ttl,
            "forecaster": self.forecaster.status(),
        }
