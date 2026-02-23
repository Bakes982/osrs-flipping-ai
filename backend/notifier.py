from __future__ import annotations

import logging
import os
from typing import Optional

import requests

logger = logging.getLogger(__name__)


def send_guidance_alert(
    trade: dict,
    market: dict,
    state: str,
    replacement_item: Optional[str] = None,
) -> None:
    emoji = {
        "HEALTHY": "\U0001F7E2",
        "WATCH": "\U0001F7E1",
        "ADJUST": "\U0001F535",
        "EXIT": "\U0001F534",
    }.get(state, "\u26AA")

    item_name = trade.get("item_name") or f"Item {trade.get('item_id', '')}"
    title = f"{item_name} - {state} {emoji}"
    description = (
        f"Market: {market.get('low')} / {market.get('high')}\n"
        f"Buy Target: {trade.get('buy_target')}\n"
        f"Sell Target: {trade.get('sell_target')}\n"
        f"State: {state}"
    )
    if replacement_item:
        description += f"\nSuggested Replacement: {replacement_item}"

    payload = {
        "embeds": [
            {
                "title": title,
                "description": description,
                "footer": {"text": "OSRS Flipping AI - Position Guidance"},
            }
        ]
    }

    webhook = (os.getenv("DISCORD_WEBHOOK_POSITIONS") or "").strip()
    if not webhook:
        return

    try:
        requests.post(webhook, json=payload, timeout=10)
    except Exception as exc:
        logger.error("Failed to send guidance alert: %s", exc)
