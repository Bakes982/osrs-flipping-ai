"""
User profile endpoints — risk profile selection, watchlist management,
and personalisation data access.

Routes:
    GET  /api/user/profile          — return current user profile
    POST /api/user/profile          — update risk profile / alert prefs
    GET  /api/user/calibration      — return calibration multipliers
    GET  /api/user/affinity         — return item affinity boosts
    POST /api/user/flip-outcome     — record a flip outcome for personalisation
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from backend.domain.enums import RiskProfile
from backend.domain.models import UserContext

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/user", tags=["user"])


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class UserProfileResponse(BaseModel):
    user_id:          Optional[str]
    username:         Optional[str]
    risk_profile:     str
    profit_multiplier: float
    hold_multiplier:   float
    watchlist:         List[int]
    subscription_tier: str


class UpdateProfileRequest(BaseModel):
    risk_profile: Optional[str] = None
    watchlist:    Optional[List[int]] = None
    alert_margin_threshold: Optional[int] = None
    alert_volume_spike_x:   Optional[float] = None


class RecordFlipOutcomeRequest(BaseModel):
    item_id:        int
    item_name:      str = ""
    buy_target:     int
    sell_target:    int
    qty:            int = 1
    expected_profit: int = 0
    est_hold_minutes: float = 60.0
    feature_snapshot: Dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ctx(request: Request) -> UserContext:
    return getattr(request.state, "user_ctx", UserContext.anonymous())


def _require_auth(ctx: UserContext) -> None:
    if ctx.user_id is None:
        raise HTTPException(status_code=401, detail="Authentication required")


def _upsert_user(db, ctx: UserContext) -> None:
    """Ensure a user document exists for this Discord user."""
    db.db["users"].update_one(
        {"_id": ctx.user_id},
        {"$setOnInsert": {
            "_id":               ctx.user_id,
            "username":          ctx.username or "",
            "risk_profile":      ctx.risk_profile.value,
            "profit_multiplier": 1.0,
            "hold_multiplier":   1.0,
            "item_affinity":     {},
            "category_affinity": {},
            "watchlist":         [],
            "subscription_tier": "free",
            "created_at":        datetime.utcnow(),
            "updated_at":        datetime.utcnow(),
        }},
        upsert=True,
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.get("/profile", response_model=UserProfileResponse)
async def get_profile(request: Request):
    """Return the authenticated user's profile."""
    ctx = _ctx(request)
    _require_auth(ctx)

    def _sync():
        from backend.database import get_db
        db = get_db()
        try:
            _upsert_user(db, ctx)
            doc = db.db["users"].find_one({"_id": ctx.user_id}) or {}
            return {
                "user_id":          ctx.user_id,
                "username":         doc.get("username", ctx.username or ""),
                "risk_profile":     doc.get("risk_profile", "balanced"),
                "profit_multiplier": doc.get("profit_multiplier", 1.0),
                "hold_multiplier":   doc.get("hold_multiplier", 1.0),
                "watchlist":         doc.get("watchlist", []),
                "subscription_tier": doc.get("subscription_tier", "free"),
            }
        finally:
            db.close()

    return await asyncio.to_thread(_sync)


@router.post("/profile")
async def update_profile(request: Request, body: UpdateProfileRequest):
    """Update risk profile and / or alert preferences."""
    ctx = _ctx(request)
    _require_auth(ctx)

    def _sync():
        from backend.database import get_db
        db = get_db()
        try:
            _upsert_user(db, ctx)
            updates: Dict[str, Any] = {"updated_at": datetime.utcnow()}

            if body.risk_profile is not None:
                try:
                    RiskProfile(body.risk_profile)   # validate
                except ValueError:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Invalid risk_profile '{body.risk_profile}'. "
                               f"Choose: {[p.value for p in RiskProfile]}",
                    )
                updates["risk_profile"] = body.risk_profile

            if body.watchlist is not None:
                updates["watchlist"] = body.watchlist

            if body.alert_margin_threshold is not None:
                updates["alert_margin_threshold"] = body.alert_margin_threshold

            if body.alert_volume_spike_x is not None:
                updates["alert_volume_spike_x"] = body.alert_volume_spike_x

            db.db["users"].update_one({"_id": ctx.user_id}, {"$set": updates})
            return {"ok": True}
        finally:
            db.close()

    return await asyncio.to_thread(_sync)


@router.get("/calibration")
async def get_calibration(request: Request):
    """Return the user's current calibration multipliers and sample size."""
    ctx = _ctx(request)
    _require_auth(ctx)

    def _sync():
        from backend.database import get_db
        db = get_db()
        try:
            doc = db.db["users"].find_one(
                {"_id": ctx.user_id},
                {"profit_multiplier": 1, "hold_multiplier": 1,
                 "calibration_sample_size": 1},
            ) or {}
            return {
                "profit_multiplier":      doc.get("profit_multiplier", 1.0),
                "hold_multiplier":        doc.get("hold_multiplier", 1.0),
                "calibration_sample_size": doc.get("calibration_sample_size", 0),
            }
        finally:
            db.close()

    return await asyncio.to_thread(_sync)


@router.get("/affinity")
async def get_affinity(request: Request):
    """Return per-item score boosts for this user."""
    ctx = _ctx(request)
    _require_auth(ctx)

    def _sync():
        from backend.database import get_db
        db = get_db()
        try:
            doc = db.db["users"].find_one(
                {"_id": ctx.user_id},
                {"item_affinity": 1},
            ) or {}
            raw = doc.get("item_affinity", {}) or {}
            # Return as {item_id_int: boost_float}
            return {int(k): float(v) for k, v in raw.items() if k.isdigit()}
        finally:
            db.close()

    return await asyncio.to_thread(_sync)


@router.post("/flip-outcome", status_code=201)
async def record_flip_outcome(request: Request, body: RecordFlipOutcomeRequest):
    """
    Record a new open flip outcome for personalisation.

    Returns the outcome document ID to use when closing the flip later.
    """
    ctx = _ctx(request)
    _require_auth(ctx)

    def _sync():
        from backend.database import get_db
        from backend.personalization.outcomes import open_flip
        db = get_db()
        try:
            outcome_id = open_flip(
                db=db,
                user_id=ctx.user_id,
                item_id=body.item_id,
                item_name=body.item_name,
                buy_target=body.buy_target,
                sell_target=body.sell_target,
                qty=body.qty,
                expected_profit=body.expected_profit,
                est_hold_minutes=body.est_hold_minutes,
                risk_profile=ctx.risk_profile.value,
                feature_snapshot=body.feature_snapshot,
            )
            return {"outcome_id": outcome_id}
        finally:
            db.close()

    return await asyncio.to_thread(_sync)
