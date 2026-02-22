"""
Settings Endpoints for OSRS Flipping AI
GET  /api/settings  - return all settings
POST /api/settings  - update settings
"""

import asyncio

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Any, Dict

from backend.database import get_db, get_setting, set_setting, get_all_settings

router = APIRouter(prefix="/api/settings", tags=["settings"])


class SettingUpdate(BaseModel):
    """Body for updating one or more settings."""
    settings: Dict[str, Any]


# Default settings applied on first read if the table is empty
DEFAULTS: Dict[str, Any] = {
    "min_profit": 50_000,
    "max_risk": 7,
    "min_volume": 1,
    "scan_interval_seconds": 60,
    "price_poll_seconds": 10,
    "discord_webhook_url": "",
    "discord_alerts_enabled": False,
    "sell_alert_webhook_url": "",   # separate webhook for sell-price-drop alerts
    "dump_alert_webhook_url": "",   # separate webhook for dump-detection alerts
    "tracked_item_ids": [],
    "blacklisted_item_ids": [],
    "allowed_discord_ids": [],      # list of Discord user ID strings allowed to log in
    "max_investment_per_flip": 50_000_000,
    "ge_tax_rate": 0.02,
}


@router.get("")
async def get_settings_endpoint():
    """Return every stored setting, merged with defaults for any missing keys."""
    def _sync():
        db = get_db()
        try:
            stored = get_all_settings(db)
            merged = {**DEFAULTS, **stored}
            return merged
        finally:
            db.close()

    return await asyncio.to_thread(_sync)


@router.post("")
async def update_settings(body: SettingUpdate):
    """Create or update one or more settings."""
    if not body.settings:
        raise HTTPException(status_code=400, detail="No settings provided")

    def _sync():
        db = get_db()
        try:
            for key, value in body.settings.items():
                set_setting(db, key, value)
            return {"status": "ok", "updated": list(body.settings.keys())}
        finally:
            db.close()

    return await asyncio.to_thread(_sync)
