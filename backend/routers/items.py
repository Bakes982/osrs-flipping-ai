"""Item search + detail endpoints."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from typing import Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query

from backend.alerts import item_name_resolver
from backend.cache import get_redis
from backend.database import get_db, get_item
from backend.services.prices import (
    get_graph_points,
    get_latest_quote,
    get_price_series,
    normalize_graph_range,
    normalize_range,
)

router = APIRouter(prefix="/items", tags=["items"])
logger = logging.getLogger(__name__)


def _load_wiki_mapping() -> Dict[int, str]:
    """Return cached wiki mapping {item_id: name} if available."""
    try:
        mapping = item_name_resolver._ensure_cache_fresh()  # type: ignore[attr-defined]
        if isinstance(mapping, dict):
            return {int(k): str(v) for k, v in mapping.items()}
    except Exception:
        pass
    return {}


def _clean_name(item_id: int, fallback: Optional[str]) -> str:
    return item_name_resolver.resolve_item_name(item_id, fallback=fallback)


def _search_items_sync(q: str, limit: int) -> List[Dict[str, object]]:
    query = (q or "").strip()
    if not query:
        return []

    q_lower = query.lower()
    is_numeric = query.isdigit()
    exact_item_id = int(query) if is_numeric else None
    max_scan = max(40, limit * 4)

    found: Dict[int, str] = {}

    def add_result(item_id: int, name: Optional[str]) -> None:
        if item_id <= 0:
            return
        clean = str(name or "").strip()
        if not clean or clean.lower().startswith("item "):
            clean = _clean_name(item_id, clean or None)
        current = found.get(item_id)
        if current is None or current.lower().startswith("item "):
            found[item_id] = clean

    mapping = _load_wiki_mapping()

    db = get_db()
    try:
        regex = {"$regex": re.escape(query), "$options": "i"}

        if exact_item_id is not None:
            row = get_item(db, exact_item_id)
            if row and row.name:
                add_result(row.id, row.name)
            docs = db.items.find(
                {"$or": [{"_id": exact_item_id}, {"name": regex}]},
                {"_id": 1, "name": 1},
            ).limit(max_scan)
        else:
            docs = db.items.find({"name": regex}, {"_id": 1, "name": 1}).limit(max_scan)

        for doc in docs:
            add_result(int(doc.get("_id") or 0), doc.get("name"))

        trade_docs = db.trades.find(
            {"item_name": regex},
            {"item_id": 1, "item_name": 1},
        ).limit(max_scan)
        for doc in trade_docs:
            add_result(int(doc.get("item_id") or 0), doc.get("item_name"))
    finally:
        db.close()

    if exact_item_id is not None and exact_item_id in mapping:
        add_result(exact_item_id, mapping.get(exact_item_id))

    for item_id, name in mapping.items():
        if is_numeric:
            if item_id == exact_item_id or q_lower in name.lower():
                add_result(item_id, name)
        else:
            if q_lower in name.lower():
                add_result(item_id, name)

    def rank(item_id: int) -> tuple[int, str]:
        name = found[item_id].lower()
        if exact_item_id is not None and item_id == exact_item_id:
            return (0, name)
        if name.startswith(q_lower):
            return (1, name)
        if q_lower in name:
            return (2, name)
        return (3, name)

    ordered_ids = sorted(found.keys(), key=rank)
    return [{"item_id": item_id, "name": found[item_id]} for item_id in ordered_ids[:limit]]


def _compute_trend(points: List[Dict[str, int]]) -> str:
    if len(points) < 2:
        return "flat"
    prev = points[-2].get("sell") or points[-2].get("buy") or 0
    curr = points[-1].get("sell") or points[-1].get("buy") or 0
    if prev <= 0 or curr <= 0:
        return "flat"
    if curr > prev:
        return "up"
    if curr < prev:
        return "down"
    return "flat"


def _build_item_graph(item_id: int, range_key: str) -> Dict[str, object]:
    normalized = normalize_graph_range(range_key)
    mapping = _load_wiki_mapping()

    db = get_db()
    try:
        item_row = get_item(db, item_id)
        item_name = item_row.name if item_row and item_row.name else None
    finally:
        db.close()

    points = get_graph_points(item_id, normalized)
    latest_quote = get_latest_quote(item_id)
    if item_name is None and item_id not in mapping and not points and not latest_quote:
        raise HTTPException(status_code=404, detail=f"Item {item_id} not found")

    name = mapping.get(item_id) or _clean_name(item_id, item_name)

    if latest_quote:
        buy_price = int(latest_quote.get("buy") or 0)
        sell_price = int(latest_quote.get("sell") or 0)
        volume_5m = int(latest_quote.get("volume_5m") or 0)
        updated_at = int(time.time())
    elif points:
        last = points[-1]
        buy_price = int(last.get("buy") or 0)
        sell_price = int(last.get("sell") or 0)
        volume_5m = int(last.get("volume") or 0)
        updated_at = int(last.get("ts") or time.time())
    else:
        buy_price = 0
        sell_price = 0
        volume_5m = 0
        updated_at = int(time.time())

    margin_gp = max(0, buy_price - sell_price)
    trend = _compute_trend(points)

    return {
        "item_id": item_id,
        "name": name,
        "latest": {
            "buy_price": buy_price,
            "sell_price": sell_price,
            "margin_gp": margin_gp,
            "volume_5m": volume_5m,
            "trend": trend,
            "updated_at": updated_at,
        },
        "points": points,
    }


def _build_item_detail(item_id: int, range_key: str) -> Dict[str, object]:
    normalized = normalize_range(range_key)

    db = get_db()
    try:
        item_row = get_item(db, item_id)
        fallback_name = item_row.name if item_row and item_row.name else None
    finally:
        db.close()

    name = _clean_name(item_id, fallback_name)
    series = get_price_series(item_id, normalized)
    latest_quote = get_latest_quote(item_id)

    if latest_quote:
        latest_buy = int(latest_quote.get("buy") or 0)
        latest_sell = int(latest_quote.get("sell") or 0)
        latest_volume = int(latest_quote.get("volume_5m") or 0)
        source = "wiki" if series.get("ts") else "ge"
    elif series.get("ts"):
        latest_buy = int(series["buy"][-1])
        latest_sell = int(series["sell"][-1])
        latest_volume = int(series["volume"][-1]) if series.get("volume") else 0
        source = "wiki"
    else:
        latest_buy = 0
        latest_sell = 0
        latest_volume = 0
        source = "ge"

    spread_gp = max(0, latest_buy - latest_sell)
    roi_pct = (spread_gp / latest_sell * 100.0) if latest_sell > 0 else 0.0

    return {
        "item_id": item_id,
        "name": name,
        "latest": {
            "buy": latest_buy,
            "sell": latest_sell,
            "spread_gp": spread_gp,
            "roi_pct": round(roi_pct, 3),
            "volume_5m": latest_volume,
        },
        "series": {
            "ts": series.get("ts", []),
            "buy": series.get("buy", []),
            "sell": series.get("sell", []),
            "volume": series.get("volume", []),
        },
        "meta": {
            "range": normalized,
            "points": len(series.get("ts", [])),
            "source": source,
            "generated_at": int(time.time()),
        },
    }


@router.get("/search")
async def search_items(
    q: str = Query("", max_length=120),
    limit: int = Query(20, ge=1, le=50),
):
    return await asyncio.to_thread(_search_items_sync, q, limit)


@router.get("/{item_id}/graph")
async def get_item_graph(
    item_id: int,
    range: str = Query("24h", pattern="^(1h|6h|24h|7d)$"),
):
    normalized = normalize_graph_range(range)
    key = f"item_graph:{item_id}:range:{normalized}"
    redis = get_redis()

    raw = redis.get(key)
    if raw:
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8", errors="ignore")
        try:
            payload = json.loads(raw)
            if isinstance(payload, dict):
                return payload
        except Exception:
            pass

    payload = await asyncio.to_thread(_build_item_graph, item_id, normalized)
    try:
        redis.set(key, json.dumps(payload), ex=60)
    except Exception:
        pass
    return payload


@router.get("/{item_id}")
async def get_item_detail(
    item_id: int,
    range: str = Query("15d", pattern="^(15d|30h|3m|1y)$"),
):
    normalized = normalize_range(range)
    key = f"item:{item_id}:range:{normalized}"
    redis = get_redis()

    raw = redis.get(key)
    if raw:
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8", errors="ignore")
        try:
            payload = json.loads(raw)
            if isinstance(payload, dict):
                if isinstance(payload.get("meta"), dict):
                    payload["meta"]["source"] = "cache"
                points = len((payload.get("series") or {}).get("ts") or [])
                logger.info("ITEM_DETAIL key=%s hit=%s points=%d", key, True, points)
                return payload
        except Exception:
            pass

    payload = await asyncio.to_thread(_build_item_detail, item_id, normalized)
    try:
        redis.set(key, json.dumps(payload), ex=60)
    except Exception:
        pass
    points = len((payload.get("series") or {}).get("ts") or [])
    logger.info("ITEM_DETAIL key=%s hit=%s points=%d", key, False, points)
    return payload
