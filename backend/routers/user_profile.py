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
import statistics
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


class RiskProfileRequest(BaseModel):
    risk_profile: str


class FlipImportRecord(BaseModel):
    item_id: int
    ts_open: datetime
    ts_close: datetime
    qty: int = 1
    buy_filled_avg: int
    sell_filled_avg: int
    status: str = "completed"
    expected_profit_at_open: Optional[int] = None
    est_hold_minutes_at_open: Optional[float] = None
    feature_snapshot_at_open: Optional[Dict[str, Any]] = None


class FlipImportRequest(BaseModel):
    records: List[FlipImportRecord]


class AlertSettingsPatch(BaseModel):
    enabled: Optional[bool] = None
    webhook_url: Optional[str] = None
    profile_thresholds: Optional[Dict[str, Dict[str, float]]] = None


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
            "alerts": {
                "enabled": False,
                "webhook_url": "",
                "profile_thresholds": {
                    "conservative": {"min_confidence": 70, "max_risk": 4},
                    "balanced": {"min_confidence": 50, "max_risk": 7},
                    "aggressive": {"min_confidence": 30, "max_risk": 10},
                },
            },
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


@router.post("/risk_profile")
async def set_risk_profile(request: Request, body: RiskProfileRequest):
    """Set user risk profile using the dedicated contract endpoint."""
    return await update_profile(request, UpdateProfileRequest(risk_profile=body.risk_profile))


@router.post("/flips/import")
async def import_flip_history(request: Request, body: FlipImportRequest):
    """Bulk import completed/open outcomes for personalization calibration."""
    ctx = _ctx(request)
    _require_auth(ctx)

    def _sync():
        from backend.database import get_db, get_price_history
        from backend.prediction.scoring import calculate_flip_metrics

        db = get_db()
        imported = 0
        try:
            _upsert_user(db, ctx)
            col = db.db["flip_outcomes"]
            for rec in body.records:
                qty = max(int(rec.qty), 1)
                realized_profit = int((rec.sell_filled_avg - rec.buy_filled_avg) * qty)
                hold_minutes = max((rec.ts_close - rec.ts_open).total_seconds() / 60.0, 0.0)
                snapshot = rec.feature_snapshot_at_open or {}

                if not snapshot:
                    snaps = get_price_history(db, rec.item_id, hours=24)
                    if snaps:
                        nearest = min(
                            snaps,
                            key=lambda s: abs((s.timestamp - rec.ts_open).total_seconds()),
                        )
                        snapshot = calculate_flip_metrics(
                            {
                                "item_id": rec.item_id,
                                "instant_buy": nearest.instant_buy,
                                "instant_sell": nearest.instant_sell,
                                "volume_5m": (nearest.buy_volume or 0) + (nearest.sell_volume or 0),
                                "buy_time": nearest.buy_time,
                                "sell_time": nearest.sell_time,
                                "snapshots": snaps,
                                "risk_profile": ctx.risk_profile.value,
                            }
                        )

                expected_profit_open = (
                    int(rec.expected_profit_at_open)
                    if rec.expected_profit_at_open is not None
                    else int(snapshot.get("expected_profit", 0))
                )
                est_hold_open = (
                    float(rec.est_hold_minutes_at_open)
                    if rec.est_hold_minutes_at_open is not None
                    else float(snapshot.get("est_fill_time_minutes", snapshot.get("estimated_hold_time", 60.0)))
                )

                doc = {
                    "user_id": ctx.user_id,
                    "item_id": rec.item_id,
                    "item_name": snapshot.get("item_name", f"Item {rec.item_id}"),
                    "ts_open": rec.ts_open,
                    "ts_close": rec.ts_close,
                    "qty": qty,
                    "buy_target": rec.buy_filled_avg,
                    "buy_filled_avg": rec.buy_filled_avg,
                    "sell_target": rec.sell_filled_avg,
                    "sell_filled_avg": rec.sell_filled_avg,
                    "expected_profit_at_open": expected_profit_open,
                    "realized_profit": realized_profit,
                    "est_hold_minutes_at_open": est_hold_open,
                    "realized_hold_minutes": hold_minutes,
                    "status": rec.status,
                    "risk_profile_used": ctx.risk_profile.value,
                    "feature_snapshot_at_open": snapshot,
                }
                col.insert_one(doc)
                imported += 1

            return {"imported": imported}
        finally:
            db.close()

    return await asyncio.to_thread(_sync)


@router.get("/insights")
async def get_user_insights(request: Request):
    """Return personalization multipliers and realized-vs-expected outcomes."""
    ctx = _ctx(request)
    _require_auth(ctx)

    def _sync():
        from backend.database import get_db
        from backend.personalization.calibration import compute_calibration

        db = get_db()
        try:
            _upsert_user(db, ctx)
            cal = compute_calibration(db, ctx.user_id)
            outcomes = list(
                db.db["flip_outcomes"]
                .find({"user_id": ctx.user_id, "status": "completed"})
                .sort("ts_close", -1)
                .limit(500)
            )
            if not outcomes:
                return {
                    "profit_multiplier_user": cal["profit_multiplier"],
                    "hold_multiplier_user": cal["hold_multiplier"],
                    "best_categories": [],
                    "best_risk_mode_last_30d": ctx.risk_profile.value,
                    "realized_vs_expected": {"expected_profit_sum": 0, "realized_profit_sum": 0, "ratio": 1.0},
                }

            expected_sum = sum(int(o.get("expected_profit_at_open", 0) or 0) for o in outcomes)
            realized_sum = sum(int(o.get("realized_profit", 0) or 0) for o in outcomes)
            ratio = safe_ratio = (realized_sum / expected_sum) if expected_sum else 1.0

            # Best category proxy: use item categories from items_meta when available.
            item_ids = list({int(o.get("item_id", 0)) for o in outcomes if o.get("item_id")})
            meta = {
                d.get("item_id"): d.get("category", "unknown")
                for d in db.db["items_meta"].find({"item_id": {"$in": item_ids}}, {"item_id": 1, "category": 1})
            }
            by_cat: Dict[str, List[float]] = {}
            for o in outcomes:
                cat = meta.get(int(o.get("item_id", 0)), "unknown")
                hold = float(o.get("realized_hold_minutes", 0) or 0)
                if hold <= 0:
                    continue
                gph = (float(o.get("realized_profit", 0) or 0) / hold) * 60.0
                by_cat.setdefault(cat, []).append(gph)
            best_categories = sorted(
                (
                    {"category": c, "median_realized_gph": round(statistics.median(vals), 2), "sample_size": len(vals)}
                    for c, vals in by_cat.items()
                    if vals
                ),
                key=lambda x: x["median_realized_gph"],
                reverse=True,
            )[:5]

            # Best risk mode over last 30d by realized gph.
            cutoff = datetime.utcnow().timestamp() - 30 * 24 * 3600
            by_mode: Dict[str, List[float]] = {}
            for o in outcomes:
                ts_close = o.get("ts_close")
                if not ts_close or ts_close.timestamp() < cutoff:
                    continue
                mode = o.get("risk_profile_used", "balanced")
                hold = float(o.get("realized_hold_minutes", 0) or 0)
                if hold <= 0:
                    continue
                by_mode.setdefault(mode, []).append((float(o.get("realized_profit", 0) or 0) / hold) * 60.0)
            if by_mode:
                best_mode = max(by_mode.items(), key=lambda kv: statistics.mean(kv[1]))[0]
            else:
                best_mode = ctx.risk_profile.value

            return {
                "profit_multiplier_user": cal["profit_multiplier"],
                "hold_multiplier_user": cal["hold_multiplier"],
                "best_categories": best_categories,
                "best_risk_mode_last_30d": best_mode,
                "realized_vs_expected": {
                    "expected_profit_sum": expected_sum,
                    "realized_profit_sum": realized_sum,
                    "ratio": round(safe_ratio, 4),
                },
            }
        finally:
            db.close()

    return await asyncio.to_thread(_sync)


@router.patch("/alerts/settings")
async def patch_alert_settings(request: Request, body: AlertSettingsPatch):
    """Patch per-user alert settings and profile thresholds."""
    ctx = _ctx(request)
    _require_auth(ctx)

    def _sync():
        from backend.database import get_db
        db = get_db()
        try:
            _upsert_user(db, ctx)
            set_map: Dict[str, Any] = {"updated_at": datetime.utcnow()}
            if body.enabled is not None:
                set_map["alerts.enabled"] = bool(body.enabled)
            if body.webhook_url is not None:
                set_map["alerts.webhook_url"] = body.webhook_url.strip()
            if body.profile_thresholds is not None:
                set_map["alerts.profile_thresholds"] = body.profile_thresholds
            db.db["users"].update_one({"_id": ctx.user_id}, {"$set": set_map})
            return {"ok": True}
        finally:
            db.close()

    return await asyncio.to_thread(_sync)


@router.get("/alerts/test")
async def alerts_test_ping(request: Request):
    """Send a test alert to the user's configured webhook."""
    ctx = _ctx(request)
    _require_auth(ctx)

    def _sync():
        import requests
        from backend.database import get_db

        db = get_db()
        try:
            doc = db.db["users"].find_one({"_id": ctx.user_id}, {"alerts": 1}) or {}
            alerts = doc.get("alerts", {}) or {}
            webhook = (alerts.get("webhook_url") or "").strip()
            enabled = bool(alerts.get("enabled", False))
            if not enabled:
                raise HTTPException(status_code=400, detail="Alerts are disabled for this user")
            if not webhook:
                raise HTTPException(status_code=400, detail="No webhook_url configured")

            resp = requests.post(
                webhook,
                json={"content": f"OSRS Flipping AI test alert for {ctx.username or ctx.user_id}"},
                timeout=10,
            )
            if resp.status_code >= 400:
                raise HTTPException(status_code=502, detail=f"Webhook failed with HTTP {resp.status_code}")
            return {"ok": True}
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
