"""
Model Training Pipeline for OSRS Flipping AI
Queries historical price data, constructs training examples with
horizon-specific targets, trains models, and saves metrics.

v2.0: Walk-forward cross-validation, recency weighting, feature importance,
      per-horizon hyperparameter profiles, concept drift detection.
"""

import logging
import math
import statistics
import time
from collections import Counter
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

from backend.database import (
    get_db, PriceSnapshot, FlipHistory, ModelMetrics,
    get_price_history, get_item_flips, get_tracked_item_ids,
    insert_model_metrics, get_model_metrics_latest,
)
from backend.ml.feature_engine import FeatureEngine, HORIZONS, HORIZON_SECONDS
from backend.ml.forecaster import (
    MultiHorizonForecaster, DIRECTION_LABELS, DIRECTION_NAMES, FLAT_THRESHOLD_PCT,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Minimum snapshots required to build a training example
MIN_SNAPSHOTS_PER_ITEM = 100

# Walk-forward cross-validation folds
N_FOLDS = 3

# Minimum training examples required to train a model
MIN_TRAINING_EXAMPLES = 50

# Model version string
MODEL_VERSION = "v2.0"

# Recency weighting: how much to favour recent samples (higher = more recency bias)
RECENCY_HALF_LIFE_HOURS = 48  # samples from 48h ago get half the weight


class ModelTrainer:
    """
    End-to-end training pipeline v2.0:
    1. Query historical price data
    2. Construct training examples with horizon-specific targets
    3. Walk-forward cross-validation (time-series aware)
    4. Recency-weighted sample training
    5. Per-horizon hyperparameter profiles
    6. Feature importance tracking
    7. Concept drift detection
    """

    def __init__(self, model_dir: str = "models"):
        self.feature_engine = FeatureEngine()
        self.forecaster = MultiHorizonForecaster(model_dir=model_dir)
        self._feature_importances: Dict[str, Dict[str, float]] = {}

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
        logger.info("Starting model training pipeline v2.0")
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

            features_list, targets, timestamps = self._build_training_data(
                item_ids, horizon, hours_of_data,
            )

            if len(features_list) < MIN_TRAINING_EXAMPLES:
                logger.warning(
                    f"[{horizon}] Only {len(features_list)} examples "
                    f"(need {MIN_TRAINING_EXAMPLES}), skipping"
                )
                continue

            # Log class distribution
            dir_counts = Counter(targets["direction"])
            logger.info(
                f"[{horizon}] Class distribution: "
                f"down={dir_counts.get(0, 0)}, flat={dir_counts.get(1, 0)}, up={dir_counts.get(2, 0)}"
            )

            # Step 3: Compute sample weights (recency bias)
            sample_weights = self._compute_sample_weights(timestamps)

            # Step 4: Walk-forward cross-validation
            cv_metrics = self._walk_forward_cv(
                horizon, features_list, targets, timestamps, sample_weights,
            )

            # Step 5: Final training on all data (for production model)
            train_metrics = self.forecaster.train(
                horizon, features_list, targets,
                sample_weights=sample_weights,
            )

            # Step 6: Track feature importance
            self._track_feature_importance(horizon)

            # Merge metrics
            combined = {**train_metrics}
            for k, v in cv_metrics.items():
                combined[k] = v
            combined["sample_count"] = len(features_list)

            all_metrics[horizon] = combined

            # Step 7: Save metrics to DB
            self._save_metrics(horizon, combined)

            # Step 8: Check for concept drift
            self._check_drift(horizon, combined)

            elapsed = time.time() - horizon_start
            logger.info(f"[{horizon}] Completed in {elapsed:.1f}s")
            self._log_metrics_summary(horizon, combined)

        # Save all models to disk
        self.forecaster.save_models()

        total_time = time.time() - start_time
        logger.info(f"\nTraining pipeline completed in {total_time:.1f}s")

        return all_metrics

    # ------------------------------------------------------------------
    # Training Data Construction
    # ------------------------------------------------------------------

    def _build_training_data(
        self,
        item_ids: List[int],
        horizon: str,
        hours_of_data: int,
    ) -> Tuple[List[Dict[str, float]], Dict[str, List], List[datetime]]:
        """
        Build training examples for a single horizon.

        Returns features, targets, and timestamps for each example.
        """
        horizon_secs = HORIZON_SECONDS[horizon]
        horizon_delta = timedelta(seconds=horizon_secs)

        all_features: List[Dict[str, float]] = []
        all_direction: List[int] = []
        all_price: List[float] = []
        all_error: List[float] = []
        all_timestamps: List[datetime] = []

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
                # Use adaptive step to avoid oversampling (skip every N for longer horizons)
                step = max(1, horizon_secs // 120)  # ~1 sample per 2 min of horizon
                for i in range(30, len(snapshots), step):
                    snap = snapshots[i]
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
                        )
                    except Exception as e:
                        logger.debug(f"Feature computation failed for item {item_id}: {e}")
                        continue

                    # Compute targets with dynamic flat threshold
                    price_change_pct = (target_price - current_price) / current_price * 100

                    # Use horizon-adaptive flat threshold: longer horizons need larger
                    # moves to count as directional (noise increases with time)
                    flat_thresh = self._adaptive_flat_threshold(horizon, current_price)

                    if price_change_pct > flat_thresh:
                        direction = DIRECTION_LABELS["up"]
                    elif price_change_pct < -flat_thresh:
                        direction = DIRECTION_LABELS["down"]
                    else:
                        direction = DIRECTION_LABELS["flat"]

                    # Error magnitude: how far actual is from a naive prediction
                    naive_error = abs(target_price - current_price)

                    all_features.append(features)
                    all_direction.append(direction)
                    all_price.append(float(target_price))
                    all_error.append(float(naive_error))
                    all_timestamps.append(snap.timestamp)
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

        return all_features, targets, all_timestamps

    def _adaptive_flat_threshold(self, horizon: str, current_price: float) -> float:
        """
        Compute a flat threshold that adapts to the horizon length.
        Short horizons: tight threshold (small moves matter).
        Long horizons: wider threshold (need bigger moves to be meaningful).
        """
        base = FLAT_THRESHOLD_PCT  # 0.3%
        horizon_secs = HORIZON_SECONDS[horizon]

        # Scale: 1m=0.15%, 5m=0.3%, 30m=0.5%, 2h=0.8%, 8h=1.2%, 24h=1.5%
        scale = math.log2(max(1, horizon_secs / 60)) / 10.0
        return max(0.15, base + scale)

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

        for offset in [0, 10, -10, 20, -20, 30, -30]:
            idx = lookup.get(target_rounded + offset)
            if idx is not None:
                price = snapshots[idx].instant_buy
                if price and price > 0:
                    return float(price)

        return None

    # ------------------------------------------------------------------
    # Sample Weighting (Recency Bias)
    # ------------------------------------------------------------------

    def _compute_sample_weights(self, timestamps: List[datetime]) -> List[float]:
        """
        Compute exponential decay weights favouring recent data.
        More recent data gets higher weight so the model adapts to
        current market conditions rather than overfitting old patterns.
        """
        if not timestamps:
            return []

        now = datetime.utcnow()
        half_life_secs = RECENCY_HALF_LIFE_HOURS * 3600
        weights = []

        for ts in timestamps:
            age_secs = max(0, (now - ts).total_seconds())
            # Exponential decay: weight = 2^(-age / half_life)
            weight = 2.0 ** (-age_secs / half_life_secs)
            weights.append(max(0.1, weight))  # floor at 0.1 so old data isn't ignored

        # Normalize so mean weight = 1.0
        mean_w = statistics.mean(weights) if weights else 1.0
        if mean_w > 0:
            weights = [w / mean_w for w in weights]

        return weights

    # ------------------------------------------------------------------
    # Walk-Forward Cross-Validation
    # ------------------------------------------------------------------

    def _walk_forward_cv(
        self,
        horizon: str,
        features_list: List[Dict[str, float]],
        targets: Dict[str, List],
        timestamps: List[datetime],
        sample_weights: List[float],
    ) -> Dict[str, float]:
        """
        Walk-forward cross-validation: train on expanding windows,
        validate on the next temporal segment. This prevents lookahead
        bias and gives a realistic estimate of out-of-sample performance.
        """
        n = len(features_list)
        if n < MIN_TRAINING_EXAMPLES * 2:
            # Not enough data for CV, do simple split
            return self._simple_validate(horizon, features_list, targets, sample_weights)

        # Divide into N_FOLDS + 1 temporal segments
        # Train on segments [0..k], validate on segment [k+1]
        fold_size = n // (N_FOLDS + 1)
        if fold_size < MIN_TRAINING_EXAMPLES // 2:
            return self._simple_validate(horizon, features_list, targets, sample_weights)

        fold_metrics: List[Dict[str, float]] = []

        for fold in range(N_FOLDS):
            # Training: all data up to fold boundary
            train_end = fold_size * (fold + 1)
            val_start = train_end
            val_end = min(train_end + fold_size, n)

            if val_end - val_start < 10:
                continue

            train_features = features_list[:train_end]
            train_targets = {k: v[:train_end] for k, v in targets.items()}
            train_weights = sample_weights[:train_end]

            val_features = features_list[val_start:val_end]
            val_targets = {k: v[val_start:val_end] for k, v in targets.items()}

            # Create a temporary forecaster for this fold
            temp_forecaster = MultiHorizonForecaster(model_dir=self.forecaster.model_dir)
            temp_forecaster.train(horizon, train_features, train_targets, sample_weights=train_weights)

            # Evaluate on validation fold
            fold_metric = self._evaluate_fold(horizon, temp_forecaster, val_features, val_targets)
            if fold_metric:
                fold_metrics.append(fold_metric)
                logger.info(
                    f"[{horizon}] CV fold {fold + 1}/{N_FOLDS}: "
                    f"dir_acc={fold_metric.get('direction_accuracy', 0):.3f}, "
                    f"profit_acc={fold_metric.get('profit_accuracy', 0):.3f}"
                )

        # Average metrics across folds
        if not fold_metrics:
            return {}

        avg_metrics: Dict[str, float] = {}
        all_keys = set()
        for fm in fold_metrics:
            all_keys.update(fm.keys())

        for key in all_keys:
            values = [fm[key] for fm in fold_metrics if key in fm]
            if values:
                avg_metrics[f"cv_{key}"] = statistics.mean(values)
                if len(values) >= 2:
                    avg_metrics[f"cv_{key}_std"] = statistics.stdev(values)

        logger.info(
            f"[{horizon}] CV summary ({len(fold_metrics)} folds): "
            f"dir_acc={avg_metrics.get('cv_direction_accuracy', 0):.3f}, "
            f"profit_acc={avg_metrics.get('cv_profit_accuracy', 0):.3f}"
        )

        return avg_metrics

    def _simple_validate(
        self,
        horizon: str,
        features_list: List[Dict[str, float]],
        targets: Dict[str, List],
        sample_weights: List[float],
    ) -> Dict[str, float]:
        """Simple 80/20 validation when not enough data for walk-forward CV."""
        n = len(features_list)
        split_idx = int(n * 0.8)

        val_features = features_list[split_idx:]
        val_targets = {k: v[split_idx:] for k, v in targets.items()}

        # Train a temp model on first 80%
        train_features = features_list[:split_idx]
        train_targets = {k: v[:split_idx] for k, v in targets.items()}
        train_weights = sample_weights[:split_idx]

        temp_forecaster = MultiHorizonForecaster(model_dir=self.forecaster.model_dir)
        temp_forecaster.train(horizon, train_features, train_targets, sample_weights=train_weights)

        metrics = self._evaluate_fold(horizon, temp_forecaster, val_features, val_targets)
        return {f"cv_{k}": v for k, v in metrics.items()} if metrics else {}

    def _evaluate_fold(
        self,
        horizon: str,
        forecaster: MultiHorizonForecaster,
        val_features: List[Dict[str, float]],
        val_targets: Dict[str, List],
    ) -> Dict[str, float]:
        """Evaluate a forecaster on validation data."""
        metrics: Dict[str, float] = {}

        if not val_features:
            return metrics

        X = forecaster._features_to_matrix(val_features)

        # Direction accuracy
        direction_model = forecaster.models.get(f"{horizon}_direction")
        direction_labels = val_targets.get("direction", [])
        if direction_model is not None and direction_labels:
            try:
                preds = direction_model.predict(X)
                correct = sum(
                    1 for p, a in zip(preds, direction_labels) if int(p) == int(a)
                )
                metrics["direction_accuracy"] = correct / len(direction_labels)

                # Per-class accuracy
                for label_name, label_id in DIRECTION_LABELS.items():
                    class_total = sum(1 for a in direction_labels if int(a) == label_id)
                    class_correct = sum(
                        1 for p, a in zip(preds, direction_labels)
                        if int(p) == label_id and int(a) == label_id
                    )
                    if class_total > 0:
                        metrics[f"{label_name}_precision"] = class_correct / max(
                            1, sum(1 for p in preds if int(p) == label_id)
                        )
                        metrics[f"{label_name}_recall"] = class_correct / class_total
            except Exception as e:
                logger.error(f"Direction evaluation failed: {e}")

        # Price MAE and MAPE
        price_model = forecaster.models.get(f"{horizon}_price")
        price_targets = val_targets.get("price", [])
        if price_model is not None and price_targets:
            try:
                preds = price_model.predict(X)
                errors = [abs(p - a) for p, a in zip(preds, price_targets)]
                metrics["price_mae"] = statistics.mean(errors)

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

        # Profitable prediction accuracy (the metric that matters most for flipping)
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
    # Feature Importance Tracking
    # ------------------------------------------------------------------

    def _track_feature_importance(self, horizon: str):
        """Extract and log feature importance from trained models."""
        feature_names = FeatureEngine.feature_names()

        direction_model = self.forecaster.models.get(f"{horizon}_direction")
        if direction_model is None:
            return

        try:
            # sklearn-compatible feature importance
            if hasattr(direction_model, 'feature_importances_'):
                importances = direction_model.feature_importances_
                importance_dict = dict(zip(feature_names, importances))
                self._feature_importances[horizon] = importance_dict

                # Log top 10 features
                sorted_feats = sorted(
                    importance_dict.items(), key=lambda x: x[1], reverse=True
                )[:10]
                logger.info(f"[{horizon}] Top features: {sorted_feats}")

                # Warn about near-zero importance features
                zero_feats = [
                    name for name, imp in importance_dict.items() if imp < 0.001
                ]
                if zero_feats:
                    logger.info(
                        f"[{horizon}] Low-importance features ({len(zero_feats)}): "
                        f"{zero_feats[:5]}..."
                    )
        except Exception as e:
            logger.debug(f"Feature importance extraction failed: {e}")

    # ------------------------------------------------------------------
    # Concept Drift Detection
    # ------------------------------------------------------------------

    def _check_drift(self, horizon: str, current_metrics: Dict[str, float]):
        """
        Compare current training metrics against historical metrics to
        detect concept drift (model degradation over time).
        """
        db = get_db()
        try:
            latest = get_model_metrics_latest(db, horizon)
            if latest is None:
                return

            # Compare key metrics
            prev_dir_acc = latest.direction_accuracy or 0
            curr_dir_acc = current_metrics.get("cv_direction_accuracy", 0)
            prev_profit_acc = latest.profit_accuracy or 0
            curr_profit_acc = current_metrics.get("cv_profit_accuracy", 0)

            # Detect significant drops
            if prev_dir_acc > 0 and curr_dir_acc > 0:
                dir_drop = prev_dir_acc - curr_dir_acc
                if dir_drop > 0.05:  # >5% drop
                    logger.warning(
                        f"[{horizon}] DRIFT DETECTED: Direction accuracy dropped "
                        f"{dir_drop:.1%} ({prev_dir_acc:.1%} -> {curr_dir_acc:.1%})"
                    )

            if prev_profit_acc > 0 and curr_profit_acc > 0:
                profit_drop = prev_profit_acc - curr_profit_acc
                if profit_drop > 0.05:
                    logger.warning(
                        f"[{horizon}] DRIFT DETECTED: Profit accuracy dropped "
                        f"{profit_drop:.1%} ({prev_profit_acc:.1%} -> {curr_profit_acc:.1%})"
                    )
        except Exception as e:
            logger.debug(f"Drift check failed: {e}")
        finally:
            db.close()

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
                direction_accuracy=metrics.get("cv_direction_accuracy"),
                price_mae=metrics.get("cv_price_mae"),
                price_mape=metrics.get("cv_price_mape"),
                profit_accuracy=metrics.get("cv_profit_accuracy"),
                sample_count=int(metrics.get("sample_count", 0)),
            )
            insert_model_metrics(db, record)
            logger.info(f"[{horizon}] Metrics saved to DB")
        except Exception as e:
            logger.error(f"Failed to save metrics for {horizon}: {e}")
        finally:
            db.close()

    def _log_metrics_summary(self, horizon: str, metrics: Dict[str, float]):
        """Log a clean summary of training results."""
        lines = [f"[{horizon}] Training Summary:"]
        key_metrics = [
            ("CV Direction Acc", "cv_direction_accuracy"),
            ("CV Profit Acc", "cv_profit_accuracy"),
            ("CV Price MAPE", "cv_price_mape"),
            ("Train Direction Acc", "direction_accuracy"),
            ("Samples", "sample_count"),
        ]
        for label, key in key_metrics:
            val = metrics.get(key)
            if val is not None:
                if "acc" in key.lower():
                    lines.append(f"  {label}: {val:.1%}")
                elif "mape" in key.lower():
                    lines.append(f"  {label}: {val:.2f}%")
                else:
                    lines.append(f"  {label}: {val}")

        logger.info("\n".join(lines))

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

        features_list, targets, timestamps = self._build_training_data(
            item_ids, horizon, hours_of_data,
        )

        if len(features_list) < MIN_TRAINING_EXAMPLES:
            logger.warning(f"[{horizon}] Insufficient data: {len(features_list)} examples")
            return {}

        sample_weights = self._compute_sample_weights(timestamps)

        # Walk-forward CV
        cv_metrics = self._walk_forward_cv(
            horizon, features_list, targets, timestamps, sample_weights,
        )

        # Final train on all data
        train_metrics = self.forecaster.train(
            horizon, features_list, targets, sample_weights=sample_weights,
        )

        self._track_feature_importance(horizon)

        combined = {**train_metrics, **cv_metrics, "sample_count": len(features_list)}
        self._save_metrics(horizon, combined)
        self._check_drift(horizon, combined)
        self.forecaster.save_models()

        self._log_metrics_summary(horizon, combined)
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

    # ------------------------------------------------------------------
    # Public API: Feature Importances
    # ------------------------------------------------------------------

    def get_feature_importances(self, horizon: str) -> Dict[str, float]:
        """Return feature importance dict for a horizon (empty if not trained)."""
        return self._feature_importances.get(horizon, {})
