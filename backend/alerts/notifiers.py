"""
backend.alerts.notifiers — Pluggable notification channel implementations.

Design: every channel implements the ``Notifier`` ABC with a single async
``send(alert)`` method.  Add new channels (email, Telegram, PagerDuty …)
without touching the alert monitor.

Current implementations:
    DiscordNotifier    — Discord webhook embed
    CompositeNotifier  — Fan-out to multiple channels

Usage::

    from backend.alerts.notifiers import build_notifier_from_settings
    notifier = build_notifier_from_settings()
    await notifier.send(alert)
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import List, Optional

from backend.metrics import increment_alert_sent_count

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Alert payload (re-exported from monitor for convenience)
# ---------------------------------------------------------------------------

from backend.alerts.monitor import Alert  # noqa: E402


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class Notifier(ABC):
    """Abstract notification channel."""

    @abstractmethod
    async def send(self, alert: Alert) -> bool:
        """Send ``alert``.  Returns True on success, False on failure."""


# ---------------------------------------------------------------------------
# Discord webhook
# ---------------------------------------------------------------------------

class DiscordNotifier(Notifier):
    """Sends rich embeds to a Discord channel via webhook URL."""

    _COLOUR = {
        "INFO":     0x4FC3F7,
        "WARNING":  0xFFA726,
        "CRITICAL": 0xEF5350,
    }

    def __init__(self, webhook_url: str) -> None:
        self._url = webhook_url

    async def send(self, alert: Alert) -> bool:
        if not self._url:
            return False
        try:
            import httpx
            colour = self._COLOUR.get(alert.severity, 0x4FC3F7)
            embed: dict = {
                "title":       f"[{alert.kind.value}] {alert.item_name}",
                "description": alert.message,
                "color":       colour,
                "timestamp":   alert.timestamp.isoformat(),
                "footer":      {"text": "OSRS Flipping AI"},
            }
            if alert.data:
                embed["fields"] = [
                    {"name": k, "value": str(v), "inline": True}
                    for k, v in list(alert.data.items())[:6]
                ]
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.post(self._url, json={"embeds": [embed]})
                resp.raise_for_status()
            increment_alert_sent_count()
            return True
        except Exception as exc:
            logger.error("DiscordNotifier: %s", exc)
            return False


# ---------------------------------------------------------------------------
# Composite fan-out
# ---------------------------------------------------------------------------

class CompositeNotifier(Notifier):
    """Dispatch an alert to all registered channels concurrently."""

    def __init__(self, notifiers: List[Notifier]) -> None:
        self._notifiers = notifiers

    async def send(self, alert: Alert) -> bool:
        import asyncio
        results = await asyncio.gather(
            *[n.send(alert) for n in self._notifiers],
            return_exceptions=True,
        )
        return any(r is True for r in results)


# ---------------------------------------------------------------------------
# No-op (testing / default when nothing is configured)
# ---------------------------------------------------------------------------

class NullNotifier(Notifier):
    """Swallows alerts silently.  Used when no channel is configured."""

    async def send(self, alert: Alert) -> bool:
        logger.debug("NullNotifier: dropped alert %s for %s", alert.kind, alert.item_name)
        return False


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_notifier_from_settings(db=None) -> Notifier:
    """
    Build a notifier from DB settings (or env vars as fallback).

    Falls back to ``NullNotifier`` if nothing is configured.

    Args:
        db: Optional open ``Database`` wrapper.  If None, opens one internally.
    """
    _own_db = False
    try:
        if db is None:
            from backend.database import get_db
            db = get_db()
            _own_db = True

        from backend.database import get_setting
        webhook = get_setting(db, "discord_webhook", default=None)
        if isinstance(webhook, dict):
            enabled = webhook.get("enabled", False)
            url = webhook.get("url", "")
        else:
            enabled = get_setting(db, "discord_alerts_enabled", default=False)
            url = get_setting(db, "discord_webhook_url", default="") or ""

        if enabled and url:
            return DiscordNotifier(str(url))

    except Exception as exc:
        logger.warning("build_notifier_from_settings: %s", exc)
    finally:
        if _own_db and db is not None:
            try:
                db.close()
            except Exception:
                pass

    return NullNotifier()


def build_notifier_for_user(user_id: str, db=None) -> Notifier:
    """
    Build a notifier honouring per-user webhook settings.

    Per-user webhook, if set, overrides the global one.
    Falls back to global, then NullNotifier.

    Args:
        user_id: Discord user ID string.
        db:      Optional open Database wrapper.
    """
    _own_db = False
    try:
        if db is None:
            from backend.database import get_db
            db = get_db()
            _own_db = True

        user_doc = db.db["users"].find_one({"_id": user_id}, {"settings": 1})
        if user_doc:
            s = user_doc.get("settings", {}) or {}
            user_url = s.get("discord_webhook_url", "")
            user_enabled = s.get("discord_alerts_enabled", False)
            if user_enabled and user_url:
                return DiscordNotifier(str(user_url))

    except Exception as exc:
        logger.warning("build_notifier_for_user(%s): %s", user_id, exc)
    finally:
        if _own_db and db is not None:
            try:
                db.close()
            except Exception:
                pass

    return build_notifier_from_settings()
