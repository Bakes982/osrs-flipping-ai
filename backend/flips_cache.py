"""
Shared flip computation + cache warming helpers.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Dict, List, Sequence

from backend import config
from backend.cache_backend import get_cache_backend

logger = logging.getLogger(__name__)
from backend.database import get_db, get_item, get_item_flips, get_price_history, get_tracked_item_ids
from backend.prediction.scoring import calculate_flip_metrics
from backend.alerts.item_name_resolver import resolve_item_limit

# Live in-memory caches populated by PriceCollector every 10s/60s.
# Imported here so the margin scan can see ALL ~6000 OSRS items without
# needing them to be stored in MongoDB first.
try:
    from backend.tasks import _price_cache, _5m_cache  # type: ignore[attr-defined]
except Exception:  # worker not started yet (tests / import time)
    _price_cache: Dict = {}
    _5m_cache: Dict = {}


def _snapshot_ts(snapshot) -> datetime | None:
    for attr in ("timestamp", "ts", "time"):
        v = getattr(snapshot, attr, None) if not isinstance(snapshot, dict) else snapshot.get(attr)
        if isinstance(v, datetime):
            return v if v.tzinfo else v.replace(tzinfo=timezone.utc)
    return None


def _live_margin_scan(
    exclude_item_ids: set,
    min_margin_gp: int = 1_000,
) -> List[dict]:
    """Scan ALL live prices for items with a positive margin after GE tax.

    Uses the in-memory ``_price_cache`` populated every 10 s by
    PriceCollector — no MongoDB required.  Any item in ``exclude_item_ids``
    is skipped (already covered by the full scoring pass from MongoDB).

    Scoring is intentionally simpler than ``calculate_flip_metrics``:
    no trend, volatility, or fill-probability analysis — just margin, ROI,
    and a small volume bonus.  Scores are capped at 65 so fully-scored
    MongoDB items always rank first when both are present.
    """
    if not _price_cache:
        return []

    db = get_db()
    try:
        # One bulk DB query for item names + buy limits.
        item_meta: Dict[int, dict] = {}
        try:
            for doc in db.items.find({}, {"item_id": 1, "name": 1, "buy_limit": 1}):
                item_meta[int(doc["item_id"])] = {
                    "name": doc.get("name") or f"Item {doc.get('item_id', '?')}",
                    "buy_limit": int(doc.get("buy_limit") or 100),
                }
        except Exception:
            pass  # proceed with "Item {id}" fallback names

        results: List[dict] = []
        for item_id_str, instant in _price_cache.items():
            try:
                item_id = int(item_id_str)
                if item_id in exclude_item_ids:
                    continue

                avg = _5m_cache.get(item_id_str, {})

                # Use /latest instant prices; fall back to 5m averages for
                # expensive items that trade infrequently (may have low=0 or high=0
                # in the latest poll even though they're perfectly tradeable).
                high = int(instant.get("high") or avg.get("avgHighPrice") or 0)
                low  = int(instant.get("low")  or avg.get("avgLowPrice")  or 0)
                if high <= 0 or low <= 0:
                    continue
                if high < low:
                    high, low = low, high

                tax = min(int(high * 0.01), 5_000_000)
                margin = high - low - tax
                if margin < min_margin_gp:
                    continue

                vol = int((avg.get("highPriceVolume") or 0) + (avg.get("lowPriceVolume") or 0))

                meta = item_meta.get(item_id, {})
                name = meta.get("name") or f"Item {item_id}"
                # Prefer Wiki buy limit (accurate), then DB value, then safe 125 default.
                # 125 is a conservative guess — most tradeable items have limit ≥ 125.
                wiki_limit = resolve_item_limit(item_id)
                db_limit = int(meta.get("buy_limit") or 0)
                buy_limit = wiki_limit or db_limit or 125

                roi_pct = round(margin / max(low, 1) * 100.0, 2)
                # Conservative qty for expected-profit calc (10 M capital budget)
                qty = min(buy_limit, max(1, int(10_000_000 // max(low, 1))))
                expected_profit = margin * qty

                # Simple score: weighted by expected GP profit and ROI.
                # Volume is only a small tiebreaker — presence of zero volume
                # does NOT veto the item (unlike the complex scorer).
                # Cap at 65 so MongoDB-scored items always rank higher.
                gp_score = min(1.0, expected_profit / 5_000_000.0)
                roi_score = min(1.0, roi_pct / 5.0)
                vol_score = min(1.0, vol / 100.0) if vol > 0 else 0.0
                base = 0.55 * gp_score + 0.30 * roi_score + 0.15 * vol_score
                total_score = round(base * 65.0)

                results.append({
                    # Identity
                    "item_id": item_id,
                    "item_name": name,
                    "name": name,
                    # Prices
                    "instant_buy": high,
                    "instant_sell": low,
                    "buy_price": low,
                    "sell_price": high,
                    "recommended_buy": low,
                    "recommended_sell": high,
                    # Margin
                    "spread": margin,            # raw GP spread (pre-tax)
                    "margin": margin,
                    "spread_pct": roi_pct,
                    "margin_pct": roi_pct,
                    "roi_pct": roi_pct,
                    "tax": tax,
                    # Profit
                    "margin_after_tax": margin,
                    "net_profit": expected_profit,
                    "potential_profit": expected_profit,
                    "expected_profit": expected_profit,
                    "qty_suggested": qty,
                    # Volume
                    "volume_5m": vol,
                    "volume": vol,
                    # Scores
                    "total_score": float(total_score),
                    "flip_score": float(total_score),
                    # Trend / confidence (no history available)
                    "trend": "NEUTRAL",
                    "confidence": 0.4,
                    "confidence_pct": 40,
                    "fill_probability": 0.5 if vol > 0 else 0.3,
                    "risk_level": "MEDIUM",
                    "risk_score": 5.0,
                    "stability_score": 50,
                    # Flag so callers know this item lacks historical analysis
                    "from_margin_scan": True,
                    "vetoed": False,
                })
            except Exception:
                continue

        results.sort(key=lambda m: m.get("total_score", 0.0), reverse=True)
        return results
    finally:
        db.close()


def compute_scored_opportunities_sync(
    limit: int = 100,
    min_score: float = 45.0,
    profile: str = "balanced",
    min_confidence_pct: float = 0.0,
    user_capital: int = 10_000_000,
) -> List[dict]:
    db = get_db()
    results: List[dict] = []
    try:
        item_ids = get_tracked_item_ids(db)
        for item_id in item_ids[:400]:
            try:
                snaps = get_price_history(db, item_id, hours=4)
                if not snaps:
                    continue
                flips = get_item_flips(db, item_id, days=30)
                item = get_item(db, item_id)
                latest = snaps[-1]
                latest_ts = _snapshot_ts(latest)
                if latest_ts is not None:
                    age_min = (datetime.now(timezone.utc) - latest_ts).total_seconds() / 60.0
                    if age_min > max(1, int(config.SCORE_STALE_MAX_MINUTES)):
                        continue
                if not latest.instant_buy or not latest.instant_sell:
                    continue
                metrics = calculate_flip_metrics(
                    {
                        "item_id": item_id,
                        "item_name": item.name if item else f"Item {item_id}",
                        "instant_buy": latest.instant_buy,
                        "instant_sell": latest.instant_sell,
                        "volume_5m": (latest.buy_volume or 0) + (latest.sell_volume or 0),
                        "buy_time": latest.buy_time,
                        "sell_time": latest.sell_time,
                        "snapshots": snaps,
                        "flip_history": flips,
                        "risk_profile": profile,
                        "item_limit": getattr(item, "buy_limit", None) if item else None,
                        "user_capital": user_capital,
                    }
                )
                confidence = metrics.get("confidence", 0.0) or 0.0
                conf_pct = confidence * 100.0 if confidence <= 1.0 else confidence
                if (
                    not metrics.get("vetoed", False)
                    and (metrics.get("total_score", 0.0) or 0.0) >= min_score
                    and conf_pct >= min_confidence_pct
                ):
                    results.append(metrics)
            except Exception:
                continue
    finally:
        db.close()

    results.sort(key=lambda m: m.get("total_score", 0.0), reverse=True)
    return results[:limit]


async def compute_scored_opportunities(
    limit: int = 100,
    min_score: float = 45.0,
    profile: str = "balanced",
    min_confidence_pct: float = 0.0,
    user_capital: int = 10_000_000,
) -> List[dict]:
    return await asyncio.to_thread(
        compute_scored_opportunities_sync,
        limit,
        min_score,
        profile,
        min_confidence_pct,
        user_capital,
    )


def warm_flip_caches_sync(
    profiles: Sequence[str] = ("conservative", "balanced", "aggressive"),
    min_score: float = 45.0,
    min_confidence_pct: float = 0.0,
) -> Dict[str, int]:
    cache = get_cache_backend()
    ttl = max(30, config.FLIPS_CACHE_TTL_SECONDS)
    warmed_counts: Dict[str, int] = {}

    now = datetime.now(timezone.utc)
    cache_ts = now.isoformat()
    # Mark cache activity early so status endpoints can detect an active worker
    # even while warming profiles in sequence.
    cache.set("flips:last_updated_ts", cache_ts, ttl_seconds=ttl)

    # Run the live margin scan once (it's profile-neutral).
    # This finds EVERY item with a positive margin across all ~6000 OSRS
    # items regardless of whether they're stored in MongoDB.
    try:
        all_margin_items = _live_margin_scan(exclude_item_ids=set())
        logger.info("MARGIN_SCAN found %d positive-margin items", len(all_margin_items))
    except Exception as exc:
        logger.warning("MARGIN_SCAN failed: %s", exc)
        all_margin_items = []

    for profile in profiles:
        try:
            top100 = compute_scored_opportunities_sync(
                limit=100,
                min_score=min_score,
                profile=profile,
                min_confidence_pct=min_confidence_pct,
            )

            # Merge: MongoDB-scored items always take priority (richer analysis).
            # Append any positive-margin items not already covered.
            scored_ids = {m["item_id"] for m in top100}
            extras = [m for m in all_margin_items if m["item_id"] not in scored_ids]
            merged = top100 + extras
            merged.sort(key=lambda m: m.get("total_score", 0.0), reverse=True)

            top5 = merged[:5]

            cache.set_json(f"flips:top5:{profile}", {"ts": cache_ts, "flips": top5}, ttl_seconds=ttl)
            logger.info("OPP_CACHE_WRITE flips:top5:%s ttl=%ds items=%d", profile, ttl, len(top5))
            cache.set_json(f"flips:top100:{profile}", {"ts": cache_ts, "flips": merged}, ttl_seconds=ttl)
            logger.info("OPP_CACHE_WRITE flips:top100:%s ttl=%ds items=%d", profile, ttl, len(merged))
            cache.set_json(
                f"flips:stats:{profile}",
                {"ts": cache_ts, "count": len(merged), "top_score": (merged[0]["total_score"] if merged else 0)},
                ttl_seconds=ttl,
            )
            warmed_counts[profile] = len(merged)
            cache.set("flips:last_updated_ts", cache_ts, ttl_seconds=ttl)
            logger.info("OPP_CACHE_WRITE flips:last_updated_ts ttl=%ds", ttl)
        except Exception:
            warmed_counts[profile] = 0
            continue

    return warmed_counts


async def warm_flip_caches(
    profiles: Sequence[str] = ("conservative", "balanced", "aggressive"),
    min_score: float = 45.0,
    min_confidence_pct: float = 0.0,
) -> Dict[str, int]:
    return await asyncio.to_thread(warm_flip_caches_sync, profiles, min_score, min_confidence_pct)
