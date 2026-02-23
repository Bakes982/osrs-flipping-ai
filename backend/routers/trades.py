"""Trade workflow endpoints (slot-aware GE trade lifecycle)."""

from __future__ import annotations

import asyncio
from datetime import datetime
import os
from typing import Any, Dict, Optional
from uuid import uuid4

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from backend.database import get_db, get_setting

router = APIRouter(prefix="/api/trades", tags=["trades"])

ACTIVE_STATES = {"BUY_PENDING", "BUYING", "HOLDING", "SELLING"}
ALL_STATES = ACTIVE_STATES | {"CLOSED", "CANCELLED"}
EVENT_TO_STATE = {
    "BUY_PLACED": "BUYING",
    "BUY_FILLED": "HOLDING",
    "SELL_PLACED": "SELLING",
    "SOLD": "CLOSED",
    "CANCEL": "CANCELLED",
}


class AcceptTradeRequest(BaseModel):
    item_id: int
    name: str
    buy_target: int = Field(ge=0)
    sell_target: int = Field(ge=0)
    qty_target: int = Field(ge=1)
    max_invest_gp: int = Field(ge=0)
    type: str = Field(default="normal", pattern="^(normal|dump)$")
    volume_5m: Optional[int] = Field(default=None, ge=0)
    replace_trade_id: Optional[str] = None


class TradeEventRequest(BaseModel):
    event: str = Field(pattern="^(BUY_PLACED|BUY_FILLED|SELL_PLACED|SOLD|CANCEL)$")
    note: Optional[str] = None


def _trades_collection(db):
    return db._db["strategy_trades"]


def _now() -> datetime:
    return datetime.utcnow()


def _get_positions_webhook_sync(db) -> tuple[Optional[str], Optional[str]]:
    env_url = os.environ.get("DISCORD_WEBHOOK_POSITIONS", "").strip()
    if env_url:
        return env_url, "env"

    for key in ("positions_webhook_url", "sell_alert_webhook_url"):
        url = (get_setting(db, key) or "").strip()
        if url:
            return url, "db"
    return None, None


def _notify_dump_trade_accepted(db, trade: Dict[str, Any]) -> None:
    webhook_url, source = _get_positions_webhook_sync(db)
    if not webhook_url:
        return

    try:
        import requests

        embed = {
            "title": f"⚠️ Accepted dump trade: {trade['name']}",
            "color": 0xF59E0B,
            "fields": [
                {"name": "Slot", "value": str(trade.get("slot_index")), "inline": True},
                {"name": "Buy", "value": f"{int(trade.get('buy_target') or 0):,} GP", "inline": True},
                {"name": "Sell", "value": f"{int(trade.get('sell_target') or 0):,} GP", "inline": True},
                {"name": "Qty", "value": str(int(trade.get("qty_target") or 0)), "inline": True},
                {"name": "Risk", "value": f"SL {trade.get('stop_loss_pct', 0)}% · max {trade.get('max_duration_minutes', 0)}m", "inline": False},
            ],
            "footer": {"text": "OSRS Flipping AI • Positions"},
            "timestamp": datetime.utcnow().isoformat(),
        }
        requests.post(webhook_url, json={"embeds": [embed]}, timeout=10)
        import logging
        logging.getLogger(__name__).info("NOTIFIER=positions WEBHOOK_SOURCE=%s", source or "unknown")
    except Exception:
        return



@router.post("/accept")
async def accept_trade(body: AcceptTradeRequest):
    def _sync() -> Dict[str, Any]:
        db = get_db()
        try:
            coll = _trades_collection(db)
            ge_slots = int(get_setting(db, "ge_slots", 8) or 8)
            active = list(coll.find({"state": {"$in": list(ACTIVE_STATES)}}))

            # Optional replace flow when full.
            if len(active) >= ge_slots:
                if not body.replace_trade_id:
                    raise HTTPException(status_code=409, detail=f"No free GE slots ({len(active)}/{ge_slots})")

                replaced = coll.find_one({"trade_id": body.replace_trade_id, "state": {"$in": list(ACTIVE_STATES)}})
                if not replaced:
                    raise HTTPException(status_code=404, detail="replace_trade_id is not an active trade")

                coll.update_one(
                    {"_id": replaced["_id"]},
                    {"$set": {"state": "CANCELLED", "last_action": "REPLACED", "updated_at": _now()}},
                )
                active = list(coll.find({"state": {"$in": list(ACTIVE_STATES)}}))

            used = {int(t.get("slot_index")) for t in active if t.get("slot_index") is not None}
            slot_index = next((i for i in range(1, ge_slots + 1) if i not in used), None)
            if slot_index is None:
                raise HTTPException(status_code=409, detail="No free slot available")

            ts = _now()
            trade_id = f"tr_{uuid4().hex[:12]}"

            risk_fields = {}
            if body.type == "dump":
                min_dump_volume = int(get_setting(db, "dump_trade_min_volume", 120) or 120)
                volume_5m = int(body.volume_5m or 0)
                if volume_5m < min_dump_volume:
                    raise HTTPException(status_code=400, detail=f"Dump requires volume_5m >= {min_dump_volume}")
                risk_fields = {
                    "max_duration_minutes": int(get_setting(db, "dump_trade_max_duration_minutes", 20) or 20),
                    "stop_loss_pct": float(get_setting(db, "dump_trade_stop_loss_pct", 1.8) or 1.8),
                    "min_volume_required": min_dump_volume,
                    "volume_5m": volume_5m,
                }

            doc = {
                "trade_id": trade_id,
                "item_id": body.item_id,
                "name": body.name,
                "created_at": ts,
                "updated_at": ts,
                "slot_index": slot_index,
                "state": "BUY_PENDING",
                "buy_target": body.buy_target,
                "sell_target": body.sell_target,
                "qty_target": body.qty_target,
                "max_invest_gp": body.max_invest_gp,
                "type": body.type,
                "last_action": "ACCEPTED",
                "last_alert_ts": None,
                **risk_fields,
            }
            coll.insert_one(doc)
            if body.type == "dump":
                _notify_dump_trade_accepted(db, doc)
            active_count = coll.count_documents({"state": {"$in": list(ACTIVE_STATES)}})
            return {
                "ok": True,
                "trade": _serialize_trade(doc),
                "slots_used": active_count,
                "slots_total": ge_slots,
                "free_slots": max(0, ge_slots - active_count),
            }
        finally:
            db.close()

    return await asyncio.to_thread(_sync)


@router.get("/active")
async def list_active_trades():
    def _sync() -> Dict[str, Any]:
        db = get_db()
        try:
            coll = _trades_collection(db)
            ge_slots = int(get_setting(db, "ge_slots", 8) or 8)
            docs = list(coll.find({"state": {"$in": list(ACTIVE_STATES)}}).sort("slot_index", 1))
            items = [_serialize_trade(d) for d in docs]
            return {
                "slots_used": len(items),
                "slots_total": ge_slots,
                "free_slots": max(0, ge_slots - len(items)),
                "items": items,
            }
        finally:
            db.close()

    return await asyncio.to_thread(_sync)


@router.post("/{trade_id}/event")
async def post_trade_event(trade_id: str, body: TradeEventRequest):
    def _sync() -> Dict[str, Any]:
        db = get_db()
        try:
            coll = _trades_collection(db)
            doc = coll.find_one({"trade_id": trade_id})
            if not doc:
                raise HTTPException(status_code=404, detail="Trade not found")

            next_state = EVENT_TO_STATE[body.event]
            if next_state not in ALL_STATES:
                raise HTTPException(status_code=400, detail="Invalid state transition")

            update = {
                "state": next_state,
                "last_action": body.event if not body.note else f"{body.event}: {body.note}",
                "updated_at": _now(),
            }
            if body.event in {"BUY_PLACED", "SELL_PLACED"}:
                update["last_alert_ts"] = _now()

            coll.update_one({"_id": doc["_id"]}, {"$set": update})
            new_doc = coll.find_one({"_id": doc["_id"]})
            if not new_doc:
                raise HTTPException(status_code=500, detail="Trade update failed")
            return {"ok": True, "trade": _serialize_trade(new_doc)}
        finally:
            db.close()

    return await asyncio.to_thread(_sync)


def _serialize_trade(doc: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(doc)
    out.pop("_id", None)
    for key in ("created_at", "updated_at", "last_alert_ts"):
        if out.get(key) and hasattr(out[key], "isoformat"):
            out[key] = out[key].isoformat()
    return out
