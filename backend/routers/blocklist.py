"""
Blocklist Endpoints for OSRS Flipping AI

GET  /api/blocklist          - return current blocklist with item names
POST /api/blocklist          - set the full blocklist (array of item IDs)
POST /api/blocklist/analyze  - statistical analysis of trade history → suggestions
"""

import asyncio
from collections import defaultdict
from typing import List

from fastapi import APIRouter

from backend.database import (
    get_db,
    get_setting,
    set_setting,
    find_all_flips,
    get_item,
)

router = APIRouter(prefix="/api/blocklist", tags=["blocklist"])

_SETTING_KEY = "blacklisted_item_ids"


def _resolve_names(db, item_ids: list) -> dict:
    """Return {item_id: name} for a list of IDs, using DB first then Wiki cache."""
    names = {}
    for iid in item_ids:
        row = get_item(db, iid)
        if row and row.name and not row.name.startswith("Item "):
            names[iid] = row.name
        else:
            names[iid] = f"Item {iid}"
    return names


@router.get("")
async def get_blocklist():
    """Return the current blocklist with item names resolved."""
    def _sync():
        db = get_db()
        try:
            ids: List[int] = get_setting(db, _SETTING_KEY) or []
            names = _resolve_names(db, ids)
            return [{"item_id": iid, "name": names.get(iid, f"Item {iid}")} for iid in ids]
        finally:
            db.close()

    items = await asyncio.to_thread(_sync)
    return {"items": items, "total": len(items)}


@router.post("")
async def set_blocklist(body: dict):
    """Replace the full blocklist with the provided array of item IDs."""
    item_ids = [int(x) for x in (body.get("item_ids") or []) if x is not None]

    def _sync():
        db = get_db()
        try:
            set_setting(db, _SETTING_KEY, item_ids)
        finally:
            db.close()

    await asyncio.to_thread(_sync)
    return {"status": "ok", "count": len(item_ids)}


@router.post("/analyze")
async def analyze_blocklist():
    """Statistical analysis of flip history to generate blocklist suggestions.

    Suggestion logic (per item):
    - "block"   → win_rate < 40%  OR  (total_flips >= 3 AND total_pnl < 0)
    - "monitor" → win_rate < 55%  AND  total_flips >= 5
    - "keep"    → otherwise
    """
    def _sync():
        db = get_db()
        try:
            flips = find_all_flips(db, limit=10_000)

            # Aggregate stats per item
            stats: dict = defaultdict(lambda: {
                "item_id": 0, "name": "", "total_flips": 0,
                "wins": 0, "total_pnl": 0, "margins": [],
            })

            for f in flips:
                s = stats[f.item_id]
                s["item_id"] = f.item_id
                s["name"] = f.item_name or f"Item {f.item_id}"
                s["total_flips"] += 1
                profit = getattr(f, "net_profit", None)
                if profit is None:
                    # Compute from raw fields if net_profit not stored
                    tax = int(min(f.sell_price * 0.02, 5_000_000))
                    profit = (f.sell_price - f.buy_price - tax) * f.quantity
                s["total_pnl"] += profit
                if profit > 0:
                    s["wins"] += 1
                margin_pct = getattr(f, "margin_pct", None)
                if margin_pct:
                    s["margins"].append(margin_pct)

            results = []
            for iid, s in stats.items():
                n = s["total_flips"]
                win_rate = s["wins"] / n if n > 0 else 0
                avg_margin = sum(s["margins"]) / len(s["margins"]) if s["margins"] else 0

                if win_rate < 0.40 or (n >= 3 and s["total_pnl"] < 0):
                    suggestion = "block"
                elif win_rate < 0.55 and n >= 5:
                    suggestion = "monitor"
                else:
                    suggestion = "keep"

                results.append({
                    "item_id": iid,
                    "name": s["name"],
                    "total_flips": n,
                    "win_rate": round(win_rate * 100, 1),
                    "total_pnl": int(s["total_pnl"]),
                    "avg_margin_pct": round(avg_margin, 2),
                    "suggestion": suggestion,
                })

            results.sort(key=lambda x: (x["suggestion"] != "block", -x["total_flips"]))
            return results
        finally:
            db.close()

    items = await asyncio.to_thread(_sync)
    return {"items": items, "total": len(items)}
