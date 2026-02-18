"""
Model Training Pipeline for OSRS Flipping AI
Queries historical price data, constructs training examples with
horizon-specific targets, trains models, and saves metrics.
"""

import logging
import statistics
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any

from backend.database import (
    get_db, PriceSnapshot, FlipHistory, ModelMetrics,
    get_price_history, get_item_flips, get_tracked_item_ids,
    insert_model_metrics, get_model_metrics_latest,
)
from backend.ml.feature_engine import FeatureEngine, HORIZONS, HORIZON_SECONDS
from backend.ml.forecaster import (
    MultiHorizonForecaster, DIRECTION_LABELS, FLAT_THRESHOLD_PCT,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Minimum snapshots required to build a training example
MIN_SNAPSHOTS_PER_ITEM = 100

# Train/validation split ratio
VALIDATION_SPLIT = 0.2

# Minimum training examples required to train a model
MIN_TRAINING_EXAMPLES = 50

# Model version string
MODEL_VERSION = "v1.0"


class ModelTrainer:
    """
    End-to-end training pipeline:
    1. Query historical price data from SQLite
    2. Construct training examples with horizon-specific targets
    3. Split into train/validation
    4. Train models and evaluate
    5. Save model metrics to ModelMetrics table
    """

    def __init__(self, model_dir: str = "models"):
        self.feature_engine = FeatureEngine()
        self.forecaster = MultiHorizonForecaster(model_dir=model_dir)

    # ------------------------------------------------------------------
    # Main Entry Point
    # ------------------------------------------------------------------

    def train_all_horizons(
        self,
        hours_of_data: int = 168,
        item_ids: Optional[List[int]] = None,
    ) -> Dict[str, Dict[str, float]]:
        """
        Train models for all horizons using historical data.

        Parameters
        ----------
        hours_of_data : int
            How many hours of historical data to use (default: 168 = 7 days).
        item_ids : list of int, optional
            Specific items to train on. If None, uses all tracked items.

        Returns
        -------
        dict
            {horizon: {metric_name: value}} for each horizon.
        """
        start_time = time.time()
        logger.info("=" * 60)
        logger.info("Starting model training pipeline")
        logger.info("=" * 60)

        # Step 1: Get item IDs
        db = get_db()
        try:
            if item_ids is None:
                item_ids = get_tracked_item_ids(db)
            logger.info(f"Training on {len(item_ids)} items with {hours_of_data}h of data")
        finally:
            db.close()

        if not item_ids:
            logger.warning("No items to train on")
            return {}

        # Step 2: Build training dataset for each horizon
        all_metrics: Dict[str, Dict[str, float]] = {}

        for horizon in HORIZONS:
            logger.info(f"\n--- Training horizon: {horizon} ---")
            horizon_start = time.time()

            features_list, targets = self._build_training_data(
                item_ids, horizon, hours_of_data,
            )

            if len(features_list) < MIN_TRAINING_EXAMPLES:
                logger.warning(
                    f"[{horizon}] Only {len(features_list)} examples "
                    f"(need {MIN_TRAINING_EXAMPLES}), skipping"
                )
                continue

            # Step 3: Train/validation split
            train_features, train_targets, val_features, val_targets = (
                self._train_val_split(features_list, targets)
            )

            logger.info(
                f"[{horizon}] Training: {len(train_features)} examples, "
                f"Validation: {len(val_features)} examples"
            )

            # Step 4: Train
            train_metrics = self.forecaster.train(horizon, train_features, train_targets)

            # Step 5: Evaluate on validation set
            val_metrics = self._evaluate(horizon, val_features, val_targets)

            # Merge metrics
            combined = {**train_metrics}
            for k, v in val_metrics.items():
                combined[f"val_{k}"] = v

            all_metrics[horizon] = combined

            # Step 6: Save metrics to DB
            self._save_metrics(horizon, combined)

            elapsed = time.time() - horizon_start
            logger.info(f"[{horizon}] Completed in {elapsed:.1f}s: {combined}")

        # Save all models to disk
        self.forecaster.save_models()

        total_time = time.time() - start_time
        logger.info(f"\nTraining pipeline completed in {total_time:.1f}s")
        logger.info(f"Metrics: {all_metrics}")

        return all_metrics

    # ------------------------------------------------------------------
    # Training Data Construction
    # ------------------------------------------------------------------

    def _build_training_data(
        self,
        item_ids: List[int],
        horizon: str,
        hours_of_data: int,
    ) -> Tuple[List[Dict[str, float]], Dict[str, List]]:
        """
        Build training examples for a single horizon.

        For each snapshot at time T, the target is the actual price at
        T + horizon_seconds. We only create examples where we have both
        the feature snapshot and the target price.
        """
        horizon_secs = HORIZON_SECONDS[horizon]
        horizon_delta = timedelta(seconds=horizon_secs)

        all_features: List[Dict[str, float]] = []
        all_direction: List[int] = []
        all_price: List[float] = []
        all_error: List[float] = []

        db = get_db()
        try:
            for item_id in item_ids:
                snapshots = get_price_history(db, item_id, hours=hours_of_data)
                flips = get_item_flips(db, item_id, days=min(hours_of_data // 24 + 1, 90))

                if len(snapshots) < MIN_SNAPSHOTS_PER_ITEM:
                    continue

                # Build a time-indexed lookup for finding target prices
                price_lookup = self._build_price_lookup(snapshots)

                # Slide through snapshots, creating training examples
                for i, snap in enumerate(snapshots):
                    # We need enough history before this point for features
                    if i < 30:
                        continue

                    current_price = snap.instant_buy
                    if not current_price or current_price <= 0:
                        continue

                    # Find target price at T + horizon
                    target_time = snap.timestamp + horizon_delta
                    target_price = self._find_price_at_time(
                        snapshots, price_lookup, target_time,
                    )
                    if target_price is None or target_price <= 0:
                        continue

                    # Compute features using snapshots up to this point
                    history_slice = snapshots[:i + 1]
                    try:
                        features = self.feature_engine.compute_features(
                            item_id, history_slice, flips,
                            reference_time=snap.timestamp,
                        )
                    except Exception as e:
                        logger.debug(f"Feature computation failed for item {item_id}: {e}")
                        continue

                    # Compute targets
                    price_change_pct = (target_price - current_price) / current_price * 100

                    if price_change_pct > FLAT_THRESHOLD_PCT:
                        direction = DIRECTION_LABELS["up"]
                    elif price_change_pct < -FLAT_THRESHOLD_PCT:
                        direction = DIRECTION_LABELS["down"]
                    else:
                        direction = DIRECTION_LABELS["flat"]

                    # Error magnitude: how far actual is from a naive prediction
                    naive_error = abs(target_price - current_price)

                    all_features.append(features)
                    all_direction.append(direction)
                    all_price.append(float(target_price))
                    all_error.append(float(naive_error))
        finally:
            db.close()

        targets = {
            "direction": all_direction,
            "price": all_price,
            "error": all_error,
        }

        logger.info(
            f"[{horizon}] Built {len(all_features)} training examples "
            f"from {len(item_ids)} items"
        )

        return all_features, targets

    def _build_price_lookup(
        self, snapshots: List[PriceSnapshot],
    ) -> Dict[int, int]:
        """
        Build a lookup from unix timestamp (rounded to 10s) to index
        in the snapshots list, for fast target price finding.
        """
        lookup: Dict[int, int] = {}
        for i, s in enumerate(snapshots):
            ts = int(s.timestamp.timestamp())
            # Round to nearest 10 seconds
            ts_rounded = (ts // 10) * 10
            lookup[ts_rounded] = i
        return lookup

    def _find_price_at_time(
        self,
        snapshots: List[PriceSnapshot],
        lookup: Dict[int, int],
        target_time: datetime,
    ) -> Optional[float]:
        """
        Find the price closest to target_time.
        Searches within a tolerance window of +/- 30 seconds.
        """
        target_ts = int(target_time.timestamp())
        target_rounded = (target_ts // 10) * 10

        # Search within +/- 30 seconds (3 slots of 10s each)
        for offset in [0, 10, -10, 20, -20, 30, -30]:
            idx = lookup.get(target_rounded + offset)
            if idx is not None:
                price = snapshots[idx].instant_buy
                if price and price > 0:
                    return float(price)

        return None

    # ------------------------------------------------------------------
    # Train/Validation Split
    # ------------------------------------------------------------------

    def _train_val_split(
        self,
        features_list: List[Dict[str, float]],
        targets: Dict[str, List],
    ) -> Tuple[List[Dict], Dict[str, List], List[Dict], Dict[str, List]]:
        """
        Time-series aware split: first (1 - VALIDATION_SPLIT) for training,
        last VALIDATION_SPLIT for validation (no shuffling to avoid lookahead).
        """
        n = len(features_list)
        split_idx = int(n * (1 - VALIDATION_SPLIT))

        train_features = features_list[:split_idx]
        val_features = features_list[split_idx:]

        train_targets: Dict[str, List] = {}
        val_targets: Dict[str, List] = {}

        for key, values in targets.items():
            train_targets[key] = values[:split_idx]
            val_targets[key] = values[split_idx:]

        return train_features, train_targets, val_features, val_targets

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def _evaluate(
        self,
        horizon: str,
        val_features: List[Dict[str, float]],
        val_targets: Dict[str, List],
    ) -> Dict[str, float]:
        """Evaluate models on validation data."""
        metrics: Dict[str, float] = {}

        if not val_features:
            return metrics

        X = self.forecaster._features_to_matrix(val_features)

        # Direction accuracy
        direction_model = self.forecaster.models.get(f"{horizon}_direction")
        direction_labels = val_targets.get("direction", [])
        if direction_model is not None and direction_labels:
            try:
                preds = direction_model.predict(X)
                correct = sum(
                    1 for p, a in zip(preds, direction_labels) if int(p) == int(a)
                )
                metrics["direction_accuracy"] = correct / len(direction_labels)
            except Exception as e:
                logger.error(f"Direction evaluation failed: {e}")

        # Price MAE and MAPE
        price_model = self.forecaster.models.get(f"{horizon}_price")
        price_targets = val_targets.get("price", [])
        if price_model is not None and price_targets:
            try:
                preds = price_model.predict(X)
                errors = [abs(p - a) for p, a in zip(preds, price_targets)]
                metrics["price_mae"] = statistics.mean(errors)

                # MAPE
                pct_errors = [
                    abs(p - a) / a * 100
                    for p, a in zip(preds, price_targets)
                    if a > 0
                ]
                metrics["price_mape"] = (
                    statistics.mean(pct_errors) if pct_errors else 0.0
                )
            except Exception as e:
                logger.error(f"Price evaluation failed: {e}")

        # Profitable prediction accuracy
        if direction_model is not None and price_targets and direction_labels:
            try:
                dir_preds = direction_model.predict(X)
                profitable_correct = 0
                profitable_total = 0
                for pred_dir, actual_price, features in zip(
                    dir_preds, price_targets, val_features
                ):
                    current = features.get("current_price", 0)
                    if current <= 0:
                        continue
                    # "Profitable" prediction: predicted up and actually went up
                    actual_change = (actual_price - current) / current
                    if int(pred_dir) == DIRECTION_LABELS["up"]:
                        profitable_total += 1
                        if actual_change > 0:
                            profitable_correct += 1
                    elif int(pred_dir) == DIRECTION_LABELS["down"]:
                        profitable_total += 1
                        if actual_change < 0:
                            profitable_correct += 1

                if profitable_total > 0:
                    metrics["profit_accuracy"] = profitable_correct / profitable_total
            except Exception as e:
                logger.error(f"Profit accuracy evaluation failed: {e}")

        return metrics

    # ------------------------------------------------------------------
    # Metrics Persistence
    # ------------------------------------------------------------------

    def _save_metrics(self, horizon: str, metrics: Dict[str, float]):
        """Save training metrics to the ModelMetrics table."""
        db = get_db()
        try:
            record = ModelMetrics(
                horizon=horizon,
                model_version=MODEL_VERSION,
                timestamp=datetime.utcnow(),
                direction_accuracy=metrics.get("val_direction_accuracy"),
                price_mae=metrics.get("val_price_mae"),
                price_mape=metrics.get("val_price_mape"),
                profit_accuracy=metrics.get("val_profit_accuracy"),
                sample_count=int(metrics.get("sample_count", 0)),
            )
            insert_model_metrics(db, record)
            logger.info(f"[{horizon}] Metrics saved to DB")
        except Exception as e:
            logger.error(f"Failed to save metrics for {horizon}: {e}")
        finally:
            db.close()

    # ------------------------------------------------------------------
    # Convenience: Train a Single Horizon
    # ------------------------------------------------------------------

    def train_horizon(
        self,
        horizon: str,
        hours_of_data: int = 168,
        item_ids: Optional[List[int]] = None,
    ) -> Dict[str, float]:
        """
        Train models for a single horizon.

        Parameters
        ----------
        horizon : str
            One of HORIZONS.
        hours_of_data : int
            Hours of historical data.
        item_ids : list of int, optional
            Items to train on.

        Returns
        -------
        dict
            Metrics for this horizon.
        """
        if horizon not in HORIZONS:
            raise ValueError(f"Invalid horizon: {horizon}. Must be one of {HORIZONS}")

        db = get_db()
        try:
            if item_ids is None:
                item_ids = get_tracked_item_ids(db)
        finally:
            db.close()

        features_list, targets = self._build_training_data(
            item_ids, horizon, hours_of_data,
        )

        if len(features_list) < MIN_TRAINING_EXAMPLES:
            logger.warning(f"[{horizon}] Insufficient data: {len(features_list)} examples")
            return {}

        train_features, train_targets, val_features, val_targets = (
            self._train_val_split(features_list, targets)
        )

        train_metrics = self.forecaster.train(horizon, train_features, train_targets)
        val_metrics = self._evaluate(horizon, val_features, val_targets)

        combined = {**train_metrics}
        for k, v in val_metrics.items():
            combined[f"val_{k}"] = v

        self._save_metrics(horizon, combined)
        self.forecaster.save_models()

        return combined

    # ------------------------------------------------------------------
    # Retraining Schedule Helper
    # ------------------------------------------------------------------

    def should_retrain(self, horizon: str, max_age_hours: int = 24) -> bool:
        """Check if a horizon's model is stale and should be retrained."""
        db = get_db()
        try:
            latest = get_model_metrics_latest(db, horizon)
            if latest is None:
                return True

            age = datetime.utcnow() - latest.timestamp
            return age > timedelta(hours=max_age_hours)
        finally:
            db.close()
