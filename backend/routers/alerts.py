"""
Alert Endpoints for OSRS Flipping AI
GET  /api/alerts              - list recent alerts
POST /api/alerts/acknowledge  - mark alerts as acknowledged
POST /api/alerts/price-target - create a price target alert
GET  /api/alerts/price-targets - list active price targets
DELETE /api/alerts/price-target/{item_id} - remove a price target
"""

import asyncio
from typing import Optional, List

from fastapi import APIRouter, Query, HTTPException
from pydantic import BaseModel

from backend.database import get_db, find_alerts, acknowledge_alerts as db_acknowledge_alerts, get_setting, set_setting

router = APIRouter(prefix="/api/alerts", tags=["alerts"])


class PriceTargetCreate(BaseModel):
    """Body for creating a price target alert."""
    item_id: int
    item_name: str = ""
    target_price: int
    direction: str = "below"  # "below" or "above"


class AcknowledgeRequest(BaseModel):
    """Body for acknowledging alerts."""
    alert_ids: List[str] = []
    acknowledge_all: bool = False


@router.get("")
async def list_alerts(
    limit: int = Query(50, ge=1, le=200),
    alert_type: Optional[str] = Query(None, description="Filter by type: price_target, dump, opportunity, ml_signal"),
    unacknowledged_only: bool = Query(False),
    hours: int = Query(24, ge=1, le=168),
):
    """Return recent alerts, newest first."""
    def _sync():
        db = get_db()
        try:
            rows = find_alerts(db, hours=hours, alert_type=alert_type,
                               unacknowledged_only=unacknowledged_only, limit=limit)

            return {
                "alerts": [
                    {
                        "id": a.id,
                        "item_id": a.item_id,
                        "item_name": a.item_name,
                        "alert_type": a.alert_type,
                        "message": a.message,
                        "data": a.data,
                        "timestamp": a.timestamp.isoformat() if a.timestamp else None,
                        "acknowledged": a.acknowledged,
                    }
                    for a in rows
                ],
                "total": len(rows),
                "unacknowledged": sum(1 for a in rows if not a.acknowledged),
            }
        finally:
            db.close()

    return await asyncio.to_thread(_sync)


@router.post("/acknowledge")
async def acknowledge_alerts(body: AcknowledgeRequest):
    """Mark alerts as acknowledged (hides from notification badge)."""
    if not body.acknowledge_all and not body.alert_ids:
        raise HTTPException(status_code=400, detail="Provide alert_ids or acknowledge_all")

    def _sync():
        db = get_db()
        try:
            updated = db_acknowledge_alerts(db, alert_ids=body.alert_ids, all_unack=body.acknowledge_all)
            return {"status": "ok", "acknowledged": updated}
        finally:
            db.close()

    return await asyncio.to_thread(_sync)


@router.post("/price-target")
async def create_price_target(body: PriceTargetCreate):
    """Add a price target alert that triggers when the item hits the target price."""
    if body.direction not in ("below", "above"):
        raise HTTPException(status_code=400, detail="direction must be 'below' or 'above'")
    if body.target_price <= 0:
        raise HTTPException(status_code=400, detail="target_price must be positive")

    def _sync():
        db = get_db()
        try:
            targets = get_setting(db, "price_alerts", default=[])

            # Prevent duplicate targets for same item + direction
            for t in targets:
                if t.get("item_id") == body.item_id and t.get("direction") == body.direction:
                    # Update existing target
                    t["target_price"] = body.target_price
                    t["item_name"] = body.item_name
                    set_setting(db, "price_alerts", targets)
                    return {"status": "updated", "target": t}

            new_target = {
                "item_id": body.item_id,
                "item_name": body.item_name,
                "target_price": body.target_price,
                "direction": body.direction,
            }
            targets.append(new_target)
            set_setting(db, "price_alerts", targets)

            return {"status": "created", "target": new_target}
        finally:
            db.close()

    return await asyncio.to_thread(_sync)


@router.get("/price-targets")
async def list_price_targets():
    """Return all active price target alerts."""
    def _sync():
        db = get_db()
        try:
            targets = get_setting(db, "price_alerts", default=[])
            return {"targets": targets, "total": len(targets)}
        finally:
            db.close()

    return await asyncio.to_thread(_sync)


@router.delete("/price-target/{item_id}")
async def delete_price_target(item_id: int, direction: Optional[str] = Query(None)):
    """Remove a price target alert."""
    def _sync():
        db = get_db()
        try:
            targets = get_setting(db, "price_alerts", default=[])
            before = len(targets)
            if direction:
                targets = [t for t in targets if not (t.get("item_id") == item_id and t.get("direction") == direction)]
            else:
                targets = [t for t in targets if t.get("item_id") != item_id]

            set_setting(db, "price_alerts", targets)
            removed = before - len(targets)
            return {"status": "ok", "removed": removed}
        finally:
            db.close()

    return await asyncio.to_thread(_sync)


# ---------------------------------------------------------------------------
# Discord top-5 opportunity digest
# ---------------------------------------------------------------------------


@router.post("/send-top5")
async def send_top5_now():
    """Kick off a Discord notification with the current top 5 opportunities.

    Returns immediately — scanning + chart generation + posting runs in
    the background.  Poll GET /api/alerts/send-top5/status to track progress.
    """
    from backend.tasks import get_opportunity_notifier
    notifier = get_opportunity_notifier()
    result = await notifier.send_now()
    if not result.get("ok"):
        raise HTTPException(status_code=400, detail=result.get("error", "Unknown error"))
    return result


@router.get("/send-top5/status")
async def send_top5_status():
    """Poll the status of the background send-top5 job.

    Returns status: idle | scanning | sending | done | error
    """
    from backend.tasks import get_opportunity_notifier
    notifier = get_opportunity_notifier()
    return notifier.get_send_status()


@router.post("/test-webhook")
async def test_webhook():
    """Send a short test message to the configured Discord webhook.

    Useful for verifying the webhook URL works before enabling auto-notifications.
    """
    import requests as _requests
    from datetime import datetime as _dt

    def _sync():
        db = get_db()
        try:
            wh = get_setting(db, "discord_webhook")
            url = None
            if isinstance(wh, dict):
                url = wh.get("url")
            elif isinstance(wh, str):
                url = wh
            if not url:
                url = get_setting(db, "discord_webhook_url")
            if not url:
                return {"ok": False, "error": "No webhook URL configured"}

            payload = {
                "embeds": [{
                    "title": "✅ Webhook Test — OSRS AI Flipper",
                    "description": "Your webhook is connected and working!",
                    "color": 0x00FF00,
                    "timestamp": _dt.utcnow().isoformat(),
                    "footer": {"text": "OSRS AI Flipper"}
                }]
            }
            r = _requests.post(url, json=payload, timeout=10)
            return {"ok": r.status_code in (200, 204), "status_code": r.status_code}
        finally:
            db.close()

    return await asyncio.to_thread(_sync)
