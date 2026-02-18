"""
ML Analysis Endpoints for OSRS Flipping AI
GET /api/predict/{item_id}  - multi-horizon price predictions
GET /api/model/metrics      - model accuracy metrics per horizon
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
    from datetime import datetime, timezone

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
            timestamp=datetime.now(timezone.utc),
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
    """Return multi-horizon price predictions for an item."""
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

            rec = _pricer.price_item(item_id, snapshots=snapshots)

            # Item name — try DB first, then Wiki
            item_row = get_item(db, item_id)
            if item_row and item_row.name:
                item_name = item_row.name
            else:
                item_name = _fetch_wiki_item_name(item_id)

            current_buy = rec.instant_buy
            current_sell = rec.instant_sell
            spread = (current_buy - current_sell) if (current_buy and current_sell) else 0

            # Build predictions per horizon
            horizons_to_compute = [horizon] if horizon and horizon in ALL_HORIZONS else ALL_HORIZONS

            predictions = {}
            for h in horizons_to_compute:
                scale = _horizon_scale(h)
                minutes = _horizon_minutes(h)
                momentum_offset = int(rec.momentum * minutes)

                # When momentum is ~0 (e.g. single snapshot), use spread-based
                # mean-reversion model: longer horizons → spread compresses
                if abs(momentum_offset) < 1 and spread > 0:
                    # Spread compression factor: longer horizons tend toward
                    # tighter spreads as more offers arrive
                    compression = min(0.5, scale * 0.35)  # up to 50% spread compression
                    spread_shift = int(spread * compression)

                    # Buyer pays less over time (buy price drifts down)
                    pred_buy = int(current_buy - spread_shift * 0.6) if current_buy else None
                    # Seller gets more over time (sell price drifts up)
                    pred_sell = int(current_sell + spread_shift * 0.4) if current_sell else None
                else:
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
