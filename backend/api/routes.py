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
from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, FastAPI, HTTPException, Query, Request

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

logger = logging.getLogger(__name__)

# Module-level start time for uptime reporting
_START_TIME: float = time.time()


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
    return FlipSummary(
        item_id=m["item_id"],
        item_name=m.get("item_name", ""),
        buy=m.get("recommended_buy", 0),
        sell=m.get("recommended_sell", 0),
        margin=m.get("net_profit", 0),
        roi=m.get("roi_pct", 0),
        score=m.get("total_score", 0),
        risk=m.get("risk_score", 5),
        confidence=m.get("confidence", 1),
        volume_rating=_volume_rating(m.get("score_volume", 0)),
        estimated_hold_time=m.get("estimated_hold_time", 0),
        gp_per_hour=m.get("gp_per_hour", 0),
        trend=m.get("trend", "NEUTRAL"),
        vetoed=m.get("vetoed", False),
    )


async def _fetch_scored_opportunities(
    limit: int = 50,
    min_score: float = 45.0,
) -> List[dict]:
    """Pull currently-scored opportunities from the database."""
    from backend.database import get_db, get_tracked_item_ids, get_price_history, get_item_flips, get_item
    from backend.prediction.scoring import calculate_flip_metrics

    def _sync() -> List[dict]:
        db = get_db()
        results = []
        try:
            item_ids = get_tracked_item_ids(db)
            for item_id in item_ids[:200]:
                try:
                    snaps = get_price_history(db, item_id, hours=4)
                    if not snaps:
                        continue
                    flips = get_item_flips(db, item_id, days=30)
                    item = get_item(db, item_id)
                    latest = snaps[-1]
                    metrics = calculate_flip_metrics({
                        "item_id": item_id,
                        "item_name": item.name if item else f"Item {item_id}",
                        "instant_buy": latest.instant_buy,
                        "instant_sell": latest.instant_sell,
                        "volume_5m": (latest.buy_volume or 0) + (latest.sell_volume or 0),
                        "buy_time": latest.buy_time,
                        "sell_time": latest.sell_time,
                        "snapshots": snaps,
                        "flip_history": flips,
                    })
                    if not metrics["vetoed"] and metrics["total_score"] >= min_score:
                        results.append(metrics)
                except Exception:
                    continue
        finally:
            db.close()

        results.sort(key=lambda m: m.get("total_score", 0), reverse=True)
        return results[:limit]

    return await asyncio.to_thread(_sync)


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
    limit: int = Query(20, ge=1, le=100),
    min_score: float = Query(45.0, ge=0, le=100),
    min_roi: float = Query(0.0, ge=0),
    max_risk: float = Query(10.0, ge=0, le=10),
    min_volume: str = Query("LOW", pattern="^(LOW|MEDIUM|HIGH)$"),
    min_price: int = Query(0, ge=0),
    max_price: int = Query(0, ge=0),
    sort_by: str = Query("score", pattern="^(score|roi|gp_per_hour)$"),
):
    all_scored = await _fetch_scored_opportunities(limit=limit * 2, min_score=min_score)

    filtered = [
        m for m in all_scored
        if _meets_filter(m, min_roi, max_risk, min_volume, min_price, max_price)
    ]

    _sort_key = {"score": "total_score", "roi": "roi_pct", "gp_per_hour": "gp_per_hour"}.get(sort_by, "total_score")
    filtered.sort(key=lambda m: m.get(_sort_key, 0), reverse=True)

    flips = [_to_summary(m) for m in filtered[:limit]]
    return FlipsTopResponse(count=len(flips), generated_at=datetime.utcnow(), flips=flips)


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
    min_score: float = Query(45.0, ge=0, le=100),
):
    scored = await _fetch_scored_opportunities(limit=5, min_score=min_score)
    flips = [
        RuneLiteFlip(
            item_id=m["item_id"],
            item_name=m.get("item_name", ""),
            recommended_buy=m.get("recommended_buy", 0),
            recommended_sell=m.get("recommended_sell", 0),
            net_profit=m.get("net_profit", 0),
            roi_pct=m.get("roi_pct", 0),
            total_score=m.get("total_score", 0),
        )
        for m in scored[:5]
    ]
    return RuneLiteTop5Response(ts=int(time.time()), flips=flips)


@flip_router.get(
    "/filtered",
    response_model=FlipsTopResponse,
    summary="Filtered flip opportunities",
    description="Same as /flips/top but with explicit query parameters for every filter.",
)
async def get_filtered_flips(
    limit: int = Query(20, ge=1, le=100),
    min_roi: float = Query(0.0, ge=0),
    max_risk: float = Query(10.0, ge=0, le=10),
    min_volume: str = Query("LOW", pattern="^(LOW|MEDIUM|HIGH)$"),
    min_price: int = Query(0, ge=0),
    max_price: int = Query(0, ge=0),
    sort_by: str = Query("score", pattern="^(score|roi|gp_per_hour)$"),
    min_score: float = Query(0.0, ge=0, le=100),
):
    return await get_top_flips(
        limit=limit,
        min_score=min_score,
        min_roi=min_roi,
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
        db.db.command("ping")
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
