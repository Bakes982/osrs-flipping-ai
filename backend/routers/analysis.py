"""
ML Analysis Endpoints for OSRS Flipping AI
GET /api/predict/{item_id}  - multi-horizon price predictions (ML-backed)
GET /api/model/metrics      - model accuracy metrics per horizon
GET /api/model/status       - ML pipeline status & learning progress
"""

import asyncio
import sys
import os
from typing import Optional

from fastapi import APIRouter, Query, HTTPException

# Ensure project root is on sys.path
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from backend.database import (
    get_db,
    get_price_history,
    get_latest_price,
    get_item,
    get_model_metrics_latest,
    Prediction,
    PriceSnapshot,
)
from backend.smart_pricer import SmartPricer

router = APIRouter(tags=["analysis"])

_pricer = SmartPricer()

# Lazily-initialized ML predictor (shared across requests)
_ml_predictor = None

def _get_ml_predictor():
    """Lazily initialize and return the ML predictor singleton."""
    global _ml_predictor
    if _ml_predictor is None:
        try:
            from backend.ml.predictor import Predictor
            _ml_predictor = Predictor()
            count = _ml_predictor.load_models()
            import logging
            logging.getLogger(__name__).info(
                "ML Predictor initialized: %d models loaded", count
            )
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(
                "ML Predictor unavailable: %s — using heuristic fallback", e
            )
    return _ml_predictor

# Horizons the system supports
ALL_HORIZONS = ["1m", "5m", "30m", "2h", "8h", "24h"]


def _direction(current: Optional[int], predicted: Optional[int]) -> str:
    """Return 'up', 'down', or 'flat' comparing two prices."""
    if current is None or predicted is None:
        return "flat"
    if predicted > current:
        return "up"
    elif predicted < current:
        return "down"
    return "flat"


def _fetch_wiki_snapshot(item_id: int):
    """Fetch current prices from the OSRS Wiki API and return a synthetic PriceSnapshot."""
    import requests
    from datetime import datetime

    try:
        url = "https://prices.runescape.wiki/api/v1/osrs/latest"
        headers = {"User-Agent": "osrs-flipping-ai"}
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        data = resp.json().get("data", {}).get(str(item_id))
        if not data:
            return None

        high = data.get("high")  # instant-buy price (what buyers pay)
        low = data.get("low")    # instant-sell price (what sellers receive)
        if not high or not low:
            return None

        return PriceSnapshot(
            item_id=item_id,
            instant_buy=high,
            instant_sell=low,
            timestamp=datetime.utcnow(),
            buy_volume=data.get("highPriceVolume", 0),
            sell_volume=data.get("lowPriceVolume", 0),
        )
    except Exception:
        return None


def _fetch_wiki_item_name(item_id: int) -> str:
    """Fetch item name from the OSRS Wiki mapping API."""
    import requests

    try:
        url = "https://prices.runescape.wiki/api/v1/osrs/mapping"
        headers = {"User-Agent": "osrs-flipping-ai"}
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        for item in resp.json():
            if item.get("id") == item_id:
                return item.get("name", f"Item {item_id}")
    except Exception:
        pass
    return f"Item {item_id}"


def _fetch_wiki_timeseries(item_id: int, timestep: str = "1h"):
    """Fetch historical timeseries data from the OSRS Wiki API.

    Returns up to 365 data points with avgHighPrice, avgLowPrice, volumes.
    Valid timesteps: 5m, 1h, 6h, 24h
    """
    import requests

    valid = {"5m", "1h", "6h", "24h"}
    if timestep not in valid:
        timestep = "1h"

    try:
        url = f"https://prices.runescape.wiki/api/v1/osrs/timeseries?timestep={timestep}&id={item_id}"
        headers = {"User-Agent": "osrs-flipping-ai"}
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
        return resp.json().get("data", [])
    except Exception:
        return []


# ---------------------------------------------------------------------------
# GET /api/prices/{item_id}/history  –  GE price history from Wiki API
# ---------------------------------------------------------------------------

VALID_TIMESTEPS = ["5m", "1h", "6h", "24h"]


@router.get("/api/prices/{item_id}/history")
async def get_price_history_endpoint(
    item_id: int,
    timestep: str = Query("1h", description="Timestep: 5m, 1h, 6h, 24h"),
):
    """Return GE price history for an item from the OSRS Wiki timeseries API.

    This is global Grand Exchange data — not user-specific trades.
    """
    if timestep not in VALID_TIMESTEPS:
        raise HTTPException(400, f"Invalid timestep. Choose from: {VALID_TIMESTEPS}")

    def _sync():
        raw = _fetch_wiki_timeseries(item_id, timestep)
        if not raw:
            return None

        # Get item name
        item_name = _fetch_wiki_item_name(item_id)

        # Clean up data points — filter out entries where both prices are null
        points = []
        for pt in raw:
            high = pt.get("avgHighPrice")
            low = pt.get("avgLowPrice")
            if high is None and low is None:
                continue
            points.append({
                "timestamp": pt["timestamp"],
                "high": high,      # avg instant-buy price for this period
                "low": low,        # avg instant-sell price for this period
                "highVol": pt.get("highPriceVolume", 0),
                "lowVol": pt.get("lowPriceVolume", 0),
            })

        if not points:
            return None

        return {
            "item_id": item_id,
            "item_name": item_name,
            "timestep": timestep,
            "data": points,
        }

    result = await asyncio.to_thread(_sync)
    if result is None:
        raise HTTPException(404, f"No price history for item {item_id}")
    return result


# ---------------------------------------------------------------------------
# GET /api/predict/{item_id}
# ---------------------------------------------------------------------------

@router.get("/api/predict/{item_id}")
async def predict_item(
    item_id: int,
    horizon: Optional[str] = Query(None, description="Specific horizon e.g. 5m, 30m. Omit for all."),
):
    """Return multi-horizon price predictions for an item.

    Uses the ML pipeline (LightGBM / sklearn RandomForest) when trained
    models are available, with a statistical + heuristic fallback.
    The ML models retrain every 6 hours on live Wiki price data, so
    predictions get smarter over time.
    """
    def _sync():
        db = get_db()
        try:
            snapshots = get_price_history(db, item_id, hours=4)

            # Fall back to Wiki API if no DB history
            if not snapshots:
                wiki_snap = _fetch_wiki_snapshot(item_id)
                if not wiki_snap:
                    return None  # signal 404
                snapshots = [wiki_snap]

            # Item name — try DB first, then Wiki
            item_row = get_item(db, item_id)
            if item_row and item_row.name:
                item_name = item_row.name
            else:
                item_name = _fetch_wiki_item_name(item_id)

            # --- Try ML pipeline first ---
            ml_predictor = _get_ml_predictor()
            ml_predictions = None
            ml_method = "heuristic"

            if ml_predictor is not None:
                try:
                    from backend.database import get_item_flips
                    flips = get_item_flips(db, item_id, days=30)
                    ml_result = ml_predictor.predict_item(
                        item_id, snapshots=snapshots, flips=flips,
                        save_to_db=True,
                    )
                    meta = ml_result.get("_meta", {})
                    ml_method = meta.get("method", "statistical")

                    # Only use ML predictions if they have real values
                    if any(
                        ml_result.get(h, {}).get("buy", 0) > 0
                        for h in ALL_HORIZONS
                    ):
                        ml_predictions = ml_result
                except Exception:
                    pass  # fall through to heuristic

            # --- SmartPricer heuristic (always computed for suggested_action) ---
            rec = _pricer.price_item(item_id, snapshots=snapshots)
            current_buy = rec.instant_buy
            current_sell = rec.instant_sell
            spread = (current_buy - current_sell) if (current_buy and current_sell) else 0

            # Build predictions — prefer ML, fall back to heuristic
            horizons_to_compute = [horizon] if horizon and horizon in ALL_HORIZONS else ALL_HORIZONS
            predictions = {}

            for h in horizons_to_compute:
                if ml_predictions and h in ml_predictions:
                    ml_pred = ml_predictions[h]
                    predictions[h] = {
                        "buy": ml_pred.get("buy"),
                        "sell": ml_pred.get("sell"),
                        "direction": ml_pred.get("direction", "flat"),
                        "confidence": ml_pred.get("confidence", 0.5),
                        "method": ml_method,
                    }
                else:
                    # Heuristic fallback (spread-based mean-reversion)
                    scale = _horizon_scale(h)
                    minutes = _horizon_minutes(h)
                    momentum_offset = int(rec.momentum * minutes)

                    if abs(momentum_offset) < 1 and spread > 0:
                        compression = min(0.5, scale * 0.35)
                        spread_shift = int(spread * compression)
                        pred_buy = int(current_buy - spread_shift * 0.6) if current_buy else None
                        pred_sell = int(current_sell + spread_shift * 0.4) if current_sell else None
                    else:
                        pred_buy = int(current_buy + momentum_offset * scale) if current_buy else None
                        pred_sell = int(current_sell + momentum_offset * scale) if current_sell else None

                    predictions[h] = {
                        "buy": pred_buy,
                        "sell": pred_sell,
                        "direction": _direction(current_buy, pred_buy),
                        "confidence": round(max(0.1, rec.confidence * (1.0 - scale * 0.15)), 3),
                        "method": "heuristic",
                    }

            return {
                "item_id": item_id,
                "item_name": item_name,
                "current_buy": current_buy,
                "current_sell": current_sell,
                "prediction_method": ml_method if ml_predictions else "heuristic",
                "predictions": predictions,
                "suggested_action": {
                    "buy_at": rec.recommended_buy,
                    "sell_at": rec.recommended_sell,
                    "expected_profit": rec.expected_profit,
                    "expected_profit_pct": round(rec.expected_profit_pct, 2) if rec.expected_profit_pct else None,
                    "tax": rec.tax,
                    "trend": rec.trend.value,
                    "confidence": round(rec.confidence, 3),
                    "reason": rec.reason,
                },
            }
        finally:
            db.close()

    result = await asyncio.to_thread(_sync)
    if result is None:
        raise HTTPException(status_code=404, detail=f"No price data for item {item_id}")
    return result


def _horizon_minutes(h: str) -> float:
    """Convert horizon string to minutes."""
    mapping = {
        "1m": 1,
        "5m": 5,
        "30m": 30,
        "2h": 120,
        "8h": 480,
        "24h": 1440,
    }
    return mapping.get(h, 5)


def _horizon_scale(h: str) -> float:
    """Confidence decay scale per horizon (longer = less certain)."""
    mapping = {
        "1m": 0.2,
        "5m": 0.4,
        "30m": 0.6,
        "2h": 0.8,
        "8h": 0.9,
        "24h": 1.0,
    }
    return mapping.get(h, 0.5)


# ---------------------------------------------------------------------------
# GET /api/model/metrics
# ---------------------------------------------------------------------------

@router.get("/api/model/metrics")
async def get_model_metrics():
    """Return the latest model accuracy metrics per horizon."""
    def _sync():
        db = get_db()
        try:
            results = {}
            for h in ALL_HORIZONS:
                row = get_model_metrics_latest(db, h)
                if row:
                    results[h] = {
                        "model_version": row.model_version,
                        "direction_accuracy": row.direction_accuracy,
                        "price_mae": row.price_mae,
                        "price_mape": row.price_mape,
                        "profit_accuracy": row.profit_accuracy,
                        "sample_count": row.sample_count,
                        "last_evaluated": row.timestamp.isoformat() if row.timestamp else None,
                    }
                else:
                    results[h] = {
                        "model_version": None,
                        "direction_accuracy": None,
                        "price_mae": None,
                        "price_mape": None,
                        "profit_accuracy": None,
                        "sample_count": 0,
                        "last_evaluated": None,
                    }

            return {"horizons": results}
        finally:
            db.close()

    return await asyncio.to_thread(_sync)


# ---------------------------------------------------------------------------
# GET /api/model/status  –  Full ML pipeline status & learning progress
# ---------------------------------------------------------------------------

@router.get("/api/model/status")
async def get_model_status():
    """Return comprehensive ML pipeline status including:
    - Whether ML models are trained and active
    - Prediction accuracy over time (learning curve)
    - Data collection stats (how much training data we have)
    - Next retrain schedule
    """
    def _sync():
        db = get_db()
        try:
            from datetime import datetime, timedelta

            # --- Model status per horizon ---
            horizons = {}
            any_trained = False
            for h in ALL_HORIZONS:
                row = get_model_metrics_latest(db, h)
                if row and row.direction_accuracy is not None:
                    any_trained = True
                    horizons[h] = {
                        "trained": True,
                        "direction_accuracy": round(row.direction_accuracy * 100, 1) if row.direction_accuracy else None,
                        "price_mae": round(row.price_mae, 0) if row.price_mae else None,
                        "price_mape": round(row.price_mape, 2) if row.price_mape else None,
                        "profit_accuracy": round(row.profit_accuracy * 100, 1) if row.profit_accuracy else None,
                        "sample_count": row.sample_count or 0,
                        "last_trained": row.timestamp.isoformat() if row.timestamp else None,
                    }
                else:
                    horizons[h] = {
                        "trained": False,
                        "direction_accuracy": None,
                        "price_mae": None,
                        "price_mape": None,
                        "profit_accuracy": None,
                        "sample_count": 0,
                        "last_trained": None,
                    }

            # --- Prediction accuracy tracking ---
            # Count predictions with recorded outcomes
            total_predictions = db.predictions.count_documents({})
            predictions_with_outcomes = db.predictions.count_documents({
                "actual_buy": {"$ne": None}
            })

            # Direction accuracy from recorded outcomes
            correct_direction = 0
            total_graded = 0
            if predictions_with_outcomes > 0:
                pipeline = [
                    {"$match": {"actual_direction": {"$ne": None}, "predicted_direction": {"$ne": None}}},
                    {"$group": {
                        "_id": None,
                        "total": {"$sum": 1},
                        "correct": {"$sum": {
                            "$cond": [{"$eq": ["$predicted_direction", "$actual_direction"]}, 1, 0]
                        }},
                    }},
                ]
                agg_result = list(db.predictions.aggregate(pipeline))
                if agg_result:
                    total_graded = agg_result[0]["total"]
                    correct_direction = agg_result[0]["correct"]

            live_accuracy = round(correct_direction / total_graded * 100, 1) if total_graded > 0 else None

            # --- Accuracy over time (learning curve) ---
            # Group by date to show accuracy improving over days
            learning_curve = []
            if total_graded > 0:
                curve_pipeline = [
                    {"$match": {"actual_direction": {"$ne": None}, "predicted_direction": {"$ne": None}}},
                    {"$group": {
                        "_id": {
                            "$dateToString": {"format": "%Y-%m-%d", "date": "$timestamp"}
                        },
                        "total": {"$sum": 1},
                        "correct": {"$sum": {
                            "$cond": [{"$eq": ["$predicted_direction", "$actual_direction"]}, 1, 0]
                        }},
                    }},
                    {"$sort": {"_id": 1}},
                    {"$limit": 30},  # last 30 days
                ]
                for row in db.predictions.aggregate(curve_pipeline):
                    if row["total"] >= 5:  # min 5 predictions per day
                        learning_curve.append({
                            "date": row["_id"],
                            "accuracy": round(row["correct"] / row["total"] * 100, 1),
                            "predictions": row["total"],
                        })

            # --- Data collection stats ---
            snapshot_count = db.price_snapshots.count_documents({})
            tracked_items = len(db.price_snapshots.distinct("item_id"))
            oldest_snapshot = db.price_snapshots.find_one(
                {}, sort=[("timestamp", 1)], projection={"timestamp": 1}
            )
            newest_snapshot = db.price_snapshots.find_one(
                {}, sort=[("timestamp", -1)], projection={"timestamp": 1}
            )

            data_age_hours = 0
            if oldest_snapshot and newest_snapshot:
                data_span = newest_snapshot["timestamp"] - oldest_snapshot["timestamp"]
                data_age_hours = round(data_span.total_seconds() / 3600, 1)

            # Feature cache count
            feature_count = db.item_features.count_documents({})

            # --- ML predictor status ---
            ml_predictor = _get_ml_predictor()
            predictor_status = None
            if ml_predictor is not None:
                try:
                    predictor_status = ml_predictor.status()
                except Exception:
                    pass

            return {
                "ml_active": any_trained,
                "prediction_method": "ml" if any_trained else "statistical",
                "horizons": horizons,
                "live_accuracy": {
                    "direction_accuracy_pct": live_accuracy,
                    "total_predictions": total_predictions,
                    "graded_predictions": total_graded,
                    "correct_predictions": correct_direction,
                },
                "learning_curve": learning_curve,
                "data_collection": {
                    "total_snapshots": snapshot_count,
                    "tracked_items": tracked_items,
                    "data_span_hours": data_age_hours,
                    "feature_vectors_cached": feature_count,
                },
                "predictor": predictor_status,
                "pipeline_info": {
                    "price_collection_interval": "10s",
                    "feature_computation_interval": "60s",
                    "prediction_interval": "60s",
                    "retrain_interval": "6h",
                    "data_retention_raw": "7 days",
                    "data_retention_aggregated": "30+ days",
                },
            }
        except Exception as e:
            import logging
            logging.getLogger(__name__).error("Model status error: %s", e)
            return {
                "ml_active": False,
                "prediction_method": "statistical",
                "error": str(e),
            }
        finally:
            db.close()

    return await asyncio.to_thread(_sync)
