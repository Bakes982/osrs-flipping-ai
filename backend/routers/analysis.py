"""
ML Analysis Endpoints for OSRS Flipping AI
GET /api/predict/{item_id}  - multi-horizon price predictions
GET /api/model/metrics      - model accuracy metrics per horizon
"""

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
    Item,
    ModelMetrics,
    Prediction,
)
from backend.smart_pricer import SmartPricer

router = APIRouter(tags=["analysis"])

_pricer = SmartPricer()

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


# ---------------------------------------------------------------------------
# GET /api/predict/{item_id}
# ---------------------------------------------------------------------------

@router.get("/api/predict/{item_id}")
async def predict_item(
    item_id: int,
    horizon: Optional[str] = Query(None, description="Specific horizon e.g. 5m, 30m. Omit for all."),
):
    """Return multi-horizon price predictions for an item.

    Currently powered by SmartPricer trend analysis.  Once ML models are
    trained, this endpoint will switch to model inference.

    Response format:
    {
        "item_id": 13652,
        "item_name": "Dragon claws",
        "current_buy": 58000000,
        "current_sell": 57500000,
        "predictions": {
            "1m":  {"buy": ..., "sell": ..., "direction": "up", "confidence": 0.72},
            "5m":  {...},
            ...
        },
        "suggested_action": {"buy_at": ..., "sell_at": ..., ...}
    }
    """
    db = get_db()
    try:
        snapshots = get_price_history(db, item_id, hours=4)
        if not snapshots:
            raise HTTPException(status_code=404, detail=f"No price data for item {item_id}")

        rec = _pricer.price_item(item_id, snapshots=snapshots)

        # Item name
        item_row = db.query(Item).filter(Item.id == item_id).first()
        item_name = item_row.name if item_row else f"Item {item_id}"

        current_buy = rec.instant_buy
        current_sell = rec.instant_sell

        # Build predictions per horizon using SmartPricer extrapolations
        horizons_to_compute = [horizon] if horizon and horizon in ALL_HORIZONS else ALL_HORIZONS

        predictions = {}
        for h in horizons_to_compute:
            # Scale the SmartPricer recommendation by horizon length
            # Shorter horizons stay closer to current price; longer horizons
            # extrapolate further along the detected trend.
            scale = _horizon_scale(h)
            momentum_offset = int(rec.momentum * _horizon_minutes(h))

            pred_buy = int(current_buy + momentum_offset * scale) if current_buy else None
            pred_sell = int(current_sell + momentum_offset * scale) if current_sell else None

            predictions[h] = {
                "buy": pred_buy,
                "sell": pred_sell,
                "direction": _direction(current_buy, pred_buy),
                "confidence": round(max(0.1, rec.confidence * (1.0 - scale * 0.15)), 3),
            }

        return {
            "item_id": item_id,
            "item_name": item_name,
            "current_buy": current_buy,
            "current_sell": current_sell,
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
    """Return the latest model accuracy metrics per horizon.

    Each horizon has: direction_accuracy, price_mae, price_mape,
    profit_accuracy, sample_count.
    """
    db = get_db()
    try:
        # Get the most recent metrics row for each horizon
        results = {}
        for h in ALL_HORIZONS:
            row = (
                db.query(ModelMetrics)
                .filter(ModelMetrics.horizon == h)
                .order_by(ModelMetrics.timestamp.desc())
                .first()
            )
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
