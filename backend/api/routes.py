"""
OSRS Flipping AI — Centralised router registration (Phase 4 + 7 + 8).

This module is the single place where every APIRouter is mounted onto the
FastAPI application.  Import and call ``register_routes(app)`` once in
``backend.app``.

New endpoints added here (beyond the existing routers):

  GET  /flips/top                — top N opportunities, full schema
  GET  /flips/top5               — RuneLite-optimised payload (Phase 7)
  GET  /flips/filtered           — filtered/sorted opportunities
  POST /portfolio/optimize       — portfolio allocation plan (Phase 5)
  GET  /health                   — health check (Phase 8)
"""

from __future__ import annotations

import asyncio
import logging
import time
import threading
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

from fastapi import APIRouter, FastAPI, HTTPException, Query, Request

from backend import config
from backend.api.schemas import (
    FlipMetricsResponse,
    FlipSummary,
    FlipsTopResponse,
    RuneLiteFlip,
    RuneLiteTop5Response,
    PortfolioAllocationResponse,
    OptimizePortfolioRequest,
    HealthResponse,
)
from backend.cache_backend import get_cache_backend
from backend.flips_cache import compute_scored_opportunities

logger = logging.getLogger(__name__)

# Module-level start time for uptime reporting
_START_TIME: float = time.time()
_FRESH_RATE_STATE: Dict[Tuple[str, str], Tuple[int, int]] = {}
_FRESH_RATE_LOCK = threading.Lock()


# ---------------------------------------------------------------------------
# Flip opportunity endpoints (Phases 4 & 7)
# ---------------------------------------------------------------------------

flip_router = APIRouter(prefix="/flips", tags=["flips"])


def _volume_rating(score_volume: float) -> str:
    if score_volume >= 75:
        return "HIGH"
    if score_volume >= 40:
        return "MEDIUM"
    return "LOW"


def _meets_filter(
    metrics: dict,
    min_roi: float,
    max_risk: float,
    min_volume_rating: str,
    min_price: int,
    max_price: int,
) -> bool:
    if metrics.get("vetoed"):
        return False
    if metrics.get("roi_pct", 0) < min_roi:
        return False
    if metrics.get("risk_score", 10) > max_risk:
        return False
    rating = _volume_rating(metrics.get("score_volume", 0))
    rating_order = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
    if rating_order.get(rating, 0) < rating_order.get(min_volume_rating, 0):
        return False
    buy = metrics.get("recommended_buy", 0)
    if min_price > 0 and buy < min_price:
        return False
    if max_price > 0 and buy > max_price:
        return False
    return True


def _to_summary(m: dict) -> FlipSummary:
    conf = m.get("confidence", 0.0) or 0.0
    conf_pct = conf * 100.0 if conf <= 1.0 else conf
    return FlipSummary(
        item_id=m["item_id"],
        item_name=m.get("item_name", ""),
        name=m.get("item_name", ""),
        buy=m.get("recommended_buy", 0),
        sell=m.get("recommended_sell", 0),
        margin=m.get("net_profit", 0),
        margin_after_tax=m.get("margin_after_tax", m.get("net_profit", 0)),
        roi=m.get("roi_pct", 0),
        roi_pct=m.get("roi_pct", 0),
        volatility_1h=m.get("volatility_1h", 0.0),
        volatility_24h=m.get("volatility_24h", 0.0),
        liquidity_score=(m.get("liquidity_score", 0.0) or 0.0) * 100.0 if (m.get("liquidity_score", 0.0) or 0.0) <= 1.0 else (m.get("liquidity_score", 0.0) or 0.0),
        fill_probability=m.get("fill_probability", 0.0),
        est_fill_time_minutes=m.get("est_fill_time_minutes", m.get("estimated_hold_time", 0)),
        trend_score=(m.get("trend_score", 0.0) or 0.0) * 100.0 if (m.get("trend_score", 0.0) or 0.0) <= 1.0 else (m.get("trend_score", 0.0) or 0.0),
        decay_penalty=m.get("decay_penalty", m.get("spread_compression", 0.0)),
        risk_level=m.get("risk_level", "MEDIUM"),
        confidence_pct=conf_pct,
        qty_suggested=m.get("qty_suggested", 0),
        expected_profit_personal=m.get("expected_profit_personal", m.get("expected_profit", 0)),
        risk_adjusted_gph_personal=m.get("risk_adjusted_gph_personal", m.get("risk_adjusted_gp_per_hour", 0.0)),
        final_score=m.get("final_score", m.get("total_score", 0)),
        score=m.get("total_score", 0),
        risk=m.get("risk_score", 5),
        confidence=conf_pct,
        volume_rating=_volume_rating(m.get("score_volume", 0)),
        estimated_hold_time=m.get("estimated_hold_time", 0),
        gp_per_hour=m.get("gp_per_hour", 0),
        trend=m.get("trend", "NEUTRAL"),
        vetoed=m.get("vetoed", False),
    )


def _parse_cache_ts(ts_value: object) -> Optional[datetime]:
    if ts_value is None:
        return None
    if isinstance(ts_value, datetime):
        return ts_value if ts_value.tzinfo else ts_value.replace(tzinfo=timezone.utc)
    if isinstance(ts_value, (int, float)):
        try:
            return datetime.fromtimestamp(float(ts_value), tz=timezone.utc)
        except Exception:
            return None
    if isinstance(ts_value, str):
        try:
            normalized = ts_value.replace("Z", "+00:00")
            parsed = datetime.fromisoformat(normalized)
            return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
        except Exception:
            return None
    return None


def _cache_age_seconds(cache_ts: Optional[datetime]) -> Optional[int]:
    if cache_ts is None:
        return None
    return max(0, int((datetime.now(timezone.utc) - cache_ts).total_seconds()))


def _load_cached_flip_payload(key: str) -> Tuple[Optional[datetime], Optional[List[dict]]]:
    cache = get_cache_backend()
    payload = cache.get_json(key)
    if not isinstance(payload, dict):
        return None, None
    ts = _parse_cache_ts(payload.get("ts"))
    flips = payload.get("flips")
    if not isinstance(flips, list):
        return ts, None
    return ts, flips


def _fresh_rate_limit_key(request: Request, profile: str) -> Tuple[str, str]:
    client_host = request.client.host if request.client else "unknown"
    return client_host, profile


def _enforce_fresh_rate_limit(request: Request, profile: str) -> None:
    max_per_minute = max(1, int(config.FLIPS_FRESH_MAX_PER_MINUTE))
    now_bucket = int(time.time() // 60)
    key = _fresh_rate_limit_key(request, profile)
    with _FRESH_RATE_LOCK:
        bucket, count = _FRESH_RATE_STATE.get(key, (now_bucket, 0))
        if bucket != now_bucket:
            bucket, count = now_bucket, 0
        if count >= max_per_minute:
            raise HTTPException(status_code=429, detail="fresh=1 rate limit exceeded; retry in under a minute")
        _FRESH_RATE_STATE[key] = (bucket, count + 1)


def _reset_fresh_rate_limit_for_tests() -> None:
    with _FRESH_RATE_LOCK:
        _FRESH_RATE_STATE.clear()


@flip_router.get(
    "/top",
    response_model=FlipsTopResponse,
    summary="Top flip opportunities",
    description=(
        "Returns the top N scored flip opportunities.  Supports filtering by "
        "min ROI, max risk, volume rating, and price range."
    ),
)
async def get_top_flips(
    request: Request,
    limit: int = Query(20, ge=1, le=100),
    profile: str = Query("balanced", pattern="^(conservative|balanced|aggressive)$"),
    fresh: int = Query(0, ge=0, le=1),
    min_score: float = Query(45.0, ge=0, le=100),
    min_roi: float = Query(0.0, ge=0),
    min_confidence: float = Query(0.0, ge=0, le=100),
    max_risk: float = Query(10.0, ge=0, le=10),
    min_volume: str = Query("LOW", pattern="^(LOW|MEDIUM|HIGH)$"),
    min_price: int = Query(0, ge=0),
    max_price: int = Query(0, ge=0),
    sort_by: str = Query("score", pattern="^(score|roi|gp_per_hour)$"),
):
    ctx = getattr(request.state, "user_ctx", None)
    active_profile = profile
    if profile == "balanced" and getattr(ctx, "risk_profile", None) is not None:
        active_profile = ctx.risk_profile.value

    cache = get_cache_backend()
    cache_ts: Optional[datetime] = None

    if fresh == 1:
        _enforce_fresh_rate_limit(request, active_profile)
        all_scored = await compute_scored_opportunities(
            limit=100,
            min_score=min_score,
            profile=active_profile,
            min_confidence_pct=min_confidence,
        )
        cache_ts = datetime.now(timezone.utc)
        ts_iso = cache_ts.isoformat()
        ttl = max(30, int(config.FLIPS_CACHE_TTL_SECONDS))
        cache.set_json(f"flips:top100:{active_profile}", {"ts": ts_iso, "flips": all_scored}, ttl_seconds=ttl)
        cache.set_json(f"flips:top5:{active_profile}", {"ts": ts_iso, "flips": all_scored[:5]}, ttl_seconds=ttl)
        cache.set("flips:last_updated_ts", ts_iso, ttl_seconds=ttl)
    else:
        cache_ts, all_scored = _load_cached_flip_payload(f"flips:top100:{active_profile}")
        if all_scored is None:
            raise HTTPException(
                status_code=503,
                detail="Top list cache is not ready yet. Retry shortly or use fresh=1.",
            )

    filtered = [
        m for m in all_scored
        if _meets_filter(m, min_roi, max_risk, min_volume, min_price, max_price)
    ]

    _sort_key = {"score": "total_score", "roi": "roi_pct", "gp_per_hour": "gp_per_hour"}.get(sort_by, "total_score")
    filtered.sort(key=lambda m: m.get(_sort_key, 0), reverse=True)

    flips = [_to_summary(m) for m in filtered[:limit]]
    return FlipsTopResponse(
        count=len(flips),
        generated_at=datetime.now(timezone.utc),
        flips=flips,
        cache_ts=cache_ts,
        cache_age_seconds=_cache_age_seconds(cache_ts),
        profile_used=active_profile,
    )


@flip_router.get(
    "/top5",
    response_model=RuneLiteTop5Response,
    summary="Top-5 for RuneLite plugin (Phase 7)",
    description=(
        "Minimal-payload endpoint optimised for RuneLite plugin polling. "
        "Target response time < 200 ms.  Returns compact field names."
    ),
)
async def get_top5_runelite(
    request: Request,
    profile: str = Query("balanced", pattern="^(conservative|balanced|aggressive)$"),
    min_score: float = Query(45.0, ge=0, le=100),
    min_confidence: float = Query(0.0, ge=0, le=100),
):
    ctx = getattr(request.state, "user_ctx", None)
    active_profile = profile
    if profile == "balanced" and getattr(ctx, "risk_profile", None) is not None:
        active_profile = ctx.risk_profile.value

    cache_ts, scored = _load_cached_flip_payload(f"flips:top5:{active_profile}")
    if scored is None:
        raise HTTPException(status_code=503, detail="Top-5 cache not warmed yet")

    flips = [
        RuneLiteFlip(
            item_id=m["item_id"],
            item_name=m.get("item_name", ""),
            recommended_buy=m.get("recommended_buy", 0),
            recommended_sell=m.get("recommended_sell", 0),
            net_profit=m.get("net_profit", 0),
            roi_pct=m.get("roi_pct", 0),
            total_score=m.get("total_score", 0),
            confidence_pct=((m.get("confidence", 0.0) or 0.0) * 100.0) if (m.get("confidence", 0.0) or 0.0) <= 1.0 else (m.get("confidence", 0.0) or 0.0),
            risk_level=m.get("risk_level", "MEDIUM"),
        )
        for m in scored[:5]
    ]
    return RuneLiteTop5Response(
        ts=int(time.time()),
        flips=flips,
        cache_ts=cache_ts,
        cache_age_seconds=_cache_age_seconds(cache_ts),
        profile_used=active_profile,
    )


@flip_router.get(
    "/filtered",
    response_model=FlipsTopResponse,
    summary="Filtered flip opportunities",
    description="Same as /flips/top but with explicit query parameters for every filter.",
)
async def get_filtered_flips(
    request: Request,
    profile: str = Query("balanced", pattern="^(conservative|balanced|aggressive)$"),
    limit: int = Query(20, ge=1, le=100),
    min_roi: float = Query(0.0, ge=0),
    min_confidence: float = Query(0.0, ge=0, le=100),
    max_risk: float = Query(10.0, ge=0, le=10),
    min_volume: str = Query("LOW", pattern="^(LOW|MEDIUM|HIGH)$"),
    min_price: int = Query(0, ge=0),
    max_price: int = Query(0, ge=0),
    sort_by: str = Query("score", pattern="^(score|roi|gp_per_hour)$"),
    min_score: float = Query(0.0, ge=0, le=100),
):
    return await get_top_flips(
        request=request,
        limit=limit,
        profile=profile,
        min_score=min_score,
        min_roi=min_roi,
        min_confidence=min_confidence,
        max_risk=max_risk,
        min_volume=min_volume,
        min_price=min_price,
        max_price=max_price,
        sort_by=sort_by,
    )


# ---------------------------------------------------------------------------
# Portfolio optimizer (Phase 5)
# ---------------------------------------------------------------------------

portfolio_router = APIRouter(prefix="/portfolio", tags=["portfolio"])


@portfolio_router.post(
    "/optimize",
    response_model=PortfolioAllocationResponse,
    summary="Generate optimal portfolio allocation (Phase 5)",
)
async def optimize_portfolio(body: OptimizePortfolioRequest):
    from backend.portfolio.optimizer import generate_optimal_portfolio

    if body.capital <= 0:
        raise HTTPException(status_code=400, detail="capital must be positive")
    if body.ge_slots < 1 or body.ge_slots > 8:
        raise HTTPException(status_code=400, detail="ge_slots must be 1–8")

    plan = await asyncio.to_thread(
        generate_optimal_portfolio,
        body.capital,
        body.ge_slots,
        body.min_score,
        body.risk_tolerance,
    )
    return PortfolioAllocationResponse(**plan.to_dict())


# ---------------------------------------------------------------------------
# Health check (Phase 8)
# ---------------------------------------------------------------------------

system_router = APIRouter(tags=["system"])


@system_router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check (Phase 8)",
)
async def health_check():
    db_status = "ok"
    try:
        from backend.database import get_db
        db = get_db()
        # Database wrapper stores pymongo database on `_db`.
        if hasattr(db, "_db"):
            db._db.command("ping")
        elif hasattr(db, "db"):
            db.db.command("ping")
        else:
            raise RuntimeError("Unsupported database wrapper (missing _db/db)")
        db.close()
    except Exception as exc:
        db_status = f"error: {exc}"

    from backend.tasks import _tasks  # noqa: PLC0415
    return HealthResponse(
        status="ok" if db_status == "ok" else "degraded",
        db=db_status,
        background_tasks=len(_tasks),
        uptime_seconds=round(time.time() - _START_TIME, 1),
    )


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def register_routes(app: FastAPI) -> None:
    """Mount all routers onto ``app``.

    Call this once from ``backend.app`` after creating the FastAPI instance.
    """
    # Existing routers (retain their own prefixes)
    from backend.routers import opportunities, portfolio, analysis, settings, alerts
    from backend.routers import blocklist as blocklist_router
    from backend.routers import user_profile

    app.include_router(opportunities.router)
    app.include_router(portfolio.router)
    app.include_router(analysis.router)
    app.include_router(settings.router)
    app.include_router(alerts.router)
    app.include_router(blocklist_router.router)
    app.include_router(user_profile.router)

    # New unified endpoints
    app.include_router(flip_router)
    app.include_router(portfolio_router)
    app.include_router(system_router)

    logger.info(
        "Routes registered: %d total endpoints",
        sum(len(r.routes) for r in app.routes if hasattr(r, "routes")),
    )
