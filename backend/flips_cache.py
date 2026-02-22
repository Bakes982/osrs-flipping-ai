"""
Shared flip computation + cache warming helpers.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Dict, List, Sequence

from backend import config
from backend.cache_backend import get_cache_backend
from backend.database import get_db, get_item, get_item_flips, get_price_history, get_tracked_item_ids
from backend.prediction.scoring import calculate_flip_metrics


def _snapshot_ts(snapshot) -> datetime | None:
    for attr in ("timestamp", "ts", "time"):
        v = getattr(snapshot, attr, None) if not isinstance(snapshot, dict) else snapshot.get(attr)
        if isinstance(v, datetime):
            return v if v.tzinfo else v.replace(tzinfo=timezone.utc)
    return None


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
    for profile in profiles:
        try:
            top100 = compute_scored_opportunities_sync(
                limit=100,
                min_score=min_score,
                profile=profile,
                min_confidence_pct=min_confidence_pct,
            )
            top5 = top100[:5]

            cache.set_json(f"flips:top5:{profile}", {"ts": cache_ts, "flips": top5}, ttl_seconds=ttl)
            cache.set_json(f"flips:top100:{profile}", {"ts": cache_ts, "flips": top100}, ttl_seconds=ttl)
            cache.set_json(
                f"flips:stats:{profile}",
                {"ts": cache_ts, "count": len(top100), "top_score": (top100[0]["total_score"] if top100 else 0)},
                ttl_seconds=ttl,
            )
            warmed_counts[profile] = len(top100)
            cache.set("flips:last_updated_ts", cache_ts, ttl_seconds=ttl)
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
