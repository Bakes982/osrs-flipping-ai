"""
Opportunity Endpoints for OSRS Flipping AI
GET /api/opportunities       - list flip opportunities
GET /api/opportunities/{id}  - detailed analysis for a single item
"""

import sys
import os
from typing import Optional

from fastapi import APIRouter, Query, HTTPException

# Ensure project root is on sys.path so we can import top-level modules
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from backend.database import get_db, get_price_history, get_latest_price, get_item_flips, Item
from backend.smart_pricer import SmartPricer
from ai_strategist import scan_all_items_for_flips, analyze_single_item

router = APIRouter(prefix="/api/opportunities", tags=["opportunities"])

_pricer = SmartPricer()


@router.get("")
async def list_opportunities(
    min_profit: int = Query(0, description="Minimum net profit in GP"),
    max_risk: int = Query(8, description="Maximum risk score (1-10)"),
    min_volume: int = Query(0, description="Minimum 5-minute volume"),
    sort_by: str = Query("expected_profit", description="Sort field: expected_profit, margin_pct, risk_score, volume_5m"),
    limit: int = Query(50, ge=1, le=200, description="Max results"),
):
    """Return a ranked list of current flip opportunities.

    Scans items using the ai_strategist scan logic and enriches each
    result with SmartPricer recommended buy/sell prices.
    """
    try:
        # Run the scan (this hits the Wiki API for live prices)
        raw = scan_all_items_for_flips(
            min_price=10_000,
            max_price=500_000_000,
            min_margin_pct=0.3,
            max_risk=max_risk,
            limit=limit * 3,  # over-fetch so we can filter
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Failed to fetch market data: {e}")

    # Post-filter
    results = []
    for item in raw:
        if item["expected_profit"] < min_profit:
            continue
        if item.get("volume_5m", 0) < min_volume:
            continue

        # Enrich with SmartPricer recommendation
        try:
            rec = _pricer.price_item(item["item_id"])
            item["smart_buy"] = rec.recommended_buy
            item["smart_sell"] = rec.recommended_sell
            item["smart_profit"] = rec.expected_profit
            item["trend"] = rec.trend.value
            item["confidence"] = round(rec.confidence, 3)
            item["smart_reason"] = rec.reason
        except Exception:
            item["smart_buy"] = None
            item["smart_sell"] = None
            item["smart_profit"] = None
            item["trend"] = None
            item["confidence"] = None
            item["smart_reason"] = None

        results.append(item)

    # Sort
    valid_sort_keys = {"expected_profit", "margin_pct", "risk_score", "volume_5m", "confidence"}
    key = sort_by if sort_by in valid_sort_keys else "expected_profit"
    reverse = key != "risk_score"  # lower risk is better
    results.sort(key=lambda x: x.get(key) or 0, reverse=reverse)

    return results[:limit]


@router.get("/{item_id}")
async def get_opportunity_detail(item_id: int):
    """Return a full analysis for a single item.

    Combines quant analysis, SmartPricer recommendation, and any
    available predictions.
    """
    db = get_db()
    try:
        # SmartPricer recommendation
        snapshots = get_price_history(db, item_id, hours=4)
        rec = _pricer.price_item(item_id, snapshots=snapshots)

        # Quant analysis (text report)
        try:
            quant_report = analyze_single_item(item_id)
        except Exception:
            quant_report = None

        # Recent flips for this item
        flips = get_item_flips(db, item_id, days=30)
        flip_data = [
            {
                "buy_price": f.buy_price,
                "sell_price": f.sell_price,
                "quantity": f.quantity,
                "net_profit": f.net_profit,
                "margin_pct": round(f.margin_pct, 2),
                "duration_seconds": f.duration_seconds,
                "sell_time": f.sell_time.isoformat() if f.sell_time else None,
            }
            for f in flips[:20]
        ]

        # Item metadata
        item_row = db.query(Item).filter(Item.id == item_id).first()
        item_name = item_row.name if item_row else f"Item {item_id}"

        return {
            "item_id": item_id,
            "item_name": item_name,
            "current_buy": rec.instant_buy,
            "current_sell": rec.instant_sell,
            "recommendation": {
                "recommended_buy": rec.recommended_buy,
                "recommended_sell": rec.recommended_sell,
                "expected_profit": rec.expected_profit,
                "expected_profit_pct": round(rec.expected_profit_pct, 2) if rec.expected_profit_pct else None,
                "tax": rec.tax,
                "trend": rec.trend.value,
                "momentum": round(rec.momentum, 2),
                "confidence": round(rec.confidence, 3),
                "reason": rec.reason,
                "stale_data": rec.stale_data,
                "anomalous_spread": rec.anomalous_spread,
            },
            "vwap": {
                "1m": round(rec.vwap_1m, 2) if rec.vwap_1m else None,
                "5m": round(rec.vwap_5m, 2) if rec.vwap_5m else None,
                "30m": round(rec.vwap_30m, 2) if rec.vwap_30m else None,
                "2h": round(rec.vwap_2h, 2) if rec.vwap_2h else None,
            },
            "bollinger": {
                "upper": round(rec.bb_upper, 2) if rec.bb_upper else None,
                "middle": round(rec.bb_middle, 2) if rec.bb_middle else None,
                "lower": round(rec.bb_lower, 2) if rec.bb_lower else None,
                "position": round(rec.bb_position, 3) if rec.bb_position is not None else None,
            },
            "volume_5m": rec.volume_5m,
            "quant_report": quant_report,
            "recent_flips": flip_data,
        }
    finally:
        db.close()
