"""
Backtest endpoint — GET /backtest/run (PR12).

Rate-limited to 1 request per 60 seconds per IP (in-memory counter).
Requires BACKTEST_ADMIN_KEY header when the env var is configured.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel

from backend import config as _cfg

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/backtest", tags=["backtest"])


# ---------------------------------------------------------------------------
# Simple in-memory rate limiter
# ---------------------------------------------------------------------------

_rate_limit_state: Dict[str, float] = {}   # ip → last_request_ts
_RATE_LIMIT_SECONDS = 60


def _check_rate_limit(request: Request) -> None:
    ip  = request.client.host if request.client else "unknown"
    now = time.time()
    last = _rate_limit_state.get(ip, 0.0)
    if now - last < _RATE_LIMIT_SECONDS:
        wait = int(_RATE_LIMIT_SECONDS - (now - last))
        raise HTTPException(
            status_code=429,
            detail=f"Rate limited: please wait {wait}s before running another backtest",
        )
    _rate_limit_state[ip] = now


# ---------------------------------------------------------------------------
# Admin key auth
# ---------------------------------------------------------------------------

def _check_admin_key(request: Request) -> None:
    required = _cfg.BACKTEST_ADMIN_KEY.strip()
    if not required:
        return   # no key configured → open in dev
    provided = request.headers.get("X-Backtest-Key", "")
    if provided != required:
        raise HTTPException(status_code=403, detail="Invalid or missing X-Backtest-Key header")


# ---------------------------------------------------------------------------
# Response schema
# ---------------------------------------------------------------------------

class BacktestResponse(BaseModel):
    days:               int
    profile:            str
    strategy:           str
    steps_simulated:    int
    avg_gp_per_hour:    float
    std_gp_per_hour:    float
    fail_to_fill_rate:  float
    avg_hold_time:      float
    spice_contribution_pct: float
    total_gp:           float
    spice_gp:           float
    top_picks:          list


# ---------------------------------------------------------------------------
# Route
# ---------------------------------------------------------------------------

@router.get(
    "/run",
    response_model=BacktestResponse,
    summary="Backtest simulator (PR12)",
    description=(
        "Simulate a strategy_mode + risk_profile over historical snapshots. "
        "Heavily rate-limited. Requires X-Backtest-Key header when "
        "BACKTEST_ADMIN_KEY env var is set."
    ),
)
async def run_backtest(
    request: Request,
    days:     int = Query(7, ge=1, le=30,
                          description="Days of history to simulate (max 30)"),
    profile:  str = Query("balanced",
                          pattern="^(balanced|conservative|aggressive)$"),
    strategy: str = Query("steady_spice",
                          pattern="^(steady|steady_spice|spice_only)$"),
):
    _check_admin_key(request)
    _check_rate_limit(request)

    days = min(days, _cfg.BACKTEST_MAX_DAYS)

    def _sync() -> Dict[str, Any]:
        from backend.database import (
            get_db, get_tracked_item_ids, get_price_history, get_item,
        )
        from backend.backtest.simulator import run_backtest as _run

        db = get_db()
        snapshots_by_item: Dict[int, list] = {}
        item_names:        Dict[int, str]  = {}

        try:
            item_ids = get_tracked_item_ids(db)
            for iid in item_ids[:50]:   # cap at 50 items for performance
                try:
                    snaps = get_price_history(db, iid, hours=days * 24)
                    if snaps:
                        snapshots_by_item[iid] = snaps
                    item = get_item(db, iid)
                    item_names[iid] = item.name if item else f"Item {iid}"
                except Exception:
                    continue
        finally:
            db.close()

        if not snapshots_by_item:
            raise HTTPException(
                status_code=503,
                detail="No historical price data available for backtest",
            )

        result = _run(
            snapshots_by_item=snapshots_by_item,
            item_names=item_names,
            days=days,
            profile=profile,
            strategy=strategy,
        )
        return result.to_dict()

    data = await asyncio.to_thread(_sync)
    return BacktestResponse(**data)
