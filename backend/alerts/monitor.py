"""
OSRS Flipping AI — Alert system (Phase 6).

Implements three alert types:
  • Margin alerts     — when a flip's net profit crosses a configurable threshold
  • Volume spike      — when 5-min volume jumps by ≥ 3× its recent average
  • Trend reversal    — when price crosses a moving average in the opposite direction

Alerts are dispatched through a pluggable ``Notifier`` interface that
currently ships with a ``DiscordNotifier`` implementation.  Additional
channels (email, Telegram, etc.) can be added without touching this module.

The alert worker runs as an async background task::

    monitor = AlertMonitor()
    asyncio.create_task(monitor.run_forever())
"""

from __future__ import annotations

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Alert types
# ---------------------------------------------------------------------------

class AlertKind(str, Enum):
    MARGIN = "MARGIN"
    VOLUME_SPIKE = "VOLUME_SPIKE"
    TREND_REVERSAL = "TREND_REVERSAL"
    PRICE_DROP = "PRICE_DROP"
    OPPORTUNITY = "OPPORTUNITY"


@dataclass
class Alert:
    kind: AlertKind
    item_id: int
    item_name: str
    message: str
    severity: str = "INFO"   # INFO | WARNING | CRITICAL
    data: dict = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# Notifier interface + Discord implementation
# ---------------------------------------------------------------------------

class Notifier(ABC):
    """Abstract base for notification channels."""

    @abstractmethod
    async def send(self, alert: Alert) -> bool:
        """Send ``alert``.  Returns True on success."""


class DiscordNotifier(Notifier):
    """Sends alerts to a Discord channel via webhook."""

    _COLOUR = {
        "INFO": 0x4FC3F7,
        "WARNING": 0xFFA726,
        "CRITICAL": 0xEF5350,
    }

    def __init__(self, webhook_url: str) -> None:
        self._url = webhook_url

    async def send(self, alert: Alert) -> bool:
        if not self._url:
            return False
        try:
            colour = self._COLOUR.get(alert.severity, 0x4FC3F7)
            embed = {
                "title": f"[{alert.kind.value}] {alert.item_name}",
                "description": alert.message,
                "color": colour,
                "timestamp": alert.timestamp.isoformat(),
                "footer": {"text": "OSRS Flipping AI"},
            }
            # Attach extra fields
            if alert.data:
                embed["fields"] = [
                    {"name": k, "value": str(v), "inline": True}
                    for k, v in list(alert.data.items())[:6]
                ]
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.post(self._url, json={"embeds": [embed]})
                resp.raise_for_status()
            return True
        except Exception as exc:
            logger.error("DiscordNotifier: failed to send alert: %s", exc)
            return False


class CompositeNotifier(Notifier):
    """Fan-out to multiple notifiers."""

    def __init__(self, notifiers: List[Notifier]) -> None:
        self._notifiers = notifiers

    async def send(self, alert: Alert) -> bool:
        results = await asyncio.gather(
            *[n.send(alert) for n in self._notifiers], return_exceptions=True
        )
        return any(r is True for r in results)


# ---------------------------------------------------------------------------
# Condition checkers
# ---------------------------------------------------------------------------

def _check_margin(
    item_id: int,
    item_name: str,
    net_profit: int,
    threshold: int,
) -> Optional[Alert]:
    """Fire when net_profit crosses the configured threshold."""
    if net_profit >= threshold:
        from backend.core.utils import format_gp
        return Alert(
            kind=AlertKind.MARGIN,
            item_id=item_id,
            item_name=item_name,
            message=f"Margin opportunity: +{format_gp(net_profit)} GP net profit",
            severity="INFO",
            data={"net_profit": format_gp(net_profit)},
        )
    return None


def _check_volume_spike(
    item_id: int,
    item_name: str,
    current_vol: int,
    avg_vol: float,
    spike_multiplier: float = 3.0,
) -> Optional[Alert]:
    """Fire when current volume is ≥ spike_multiplier × avg_vol."""
    if avg_vol > 0 and current_vol >= avg_vol * spike_multiplier:
        ratio = current_vol / avg_vol
        return Alert(
            kind=AlertKind.VOLUME_SPIKE,
            item_id=item_id,
            item_name=item_name,
            message=f"Volume spike: {current_vol:,} trades (×{ratio:.1f} normal)",
            severity="WARNING",
            data={"current_vol": current_vol, "avg_vol": f"{avg_vol:.0f}", "ratio": f"{ratio:.1f}×"},
        )
    return None


def _check_trend_reversal(
    item_id: int,
    item_name: str,
    prev_trend: str,
    new_trend: str,
    price: int,
) -> Optional[Alert]:
    """Fire on a significant trend direction change."""
    reversals = {
        ("UP", "DOWN"), ("UP", "STRONG_DOWN"),
        ("STRONG_UP", "DOWN"), ("STRONG_UP", "STRONG_DOWN"),
        ("DOWN", "UP"), ("DOWN", "STRONG_UP"),
        ("STRONG_DOWN", "UP"), ("STRONG_DOWN", "STRONG_UP"),
    }
    if (prev_trend, new_trend) in reversals:
        from backend.core.utils import format_gp
        severity = "WARNING" if "STRONG" in new_trend else "INFO"
        return Alert(
            kind=AlertKind.TREND_REVERSAL,
            item_id=item_id,
            item_name=item_name,
            message=f"Trend reversed: {prev_trend} → {new_trend} @ {format_gp(price)} GP",
            severity=severity,
            data={"from": prev_trend, "to": new_trend, "price": format_gp(price)},
        )
    return None


# ---------------------------------------------------------------------------
# Alert monitor (async background worker)
# ---------------------------------------------------------------------------

class AlertMonitor:
    """Async background worker that checks for alert conditions every cycle.

    Maintains per-item state to avoid firing duplicate alerts within the
    configured cooldown window.
    """

    CHECK_INTERVAL: int = 30          # seconds between scans
    COOLDOWN_MINUTES: int = 15        # silence repeated alerts per item

    def __init__(self, notifier: Optional[Notifier] = None) -> None:
        self._notifier = notifier
        # item_id → {"trend": str, "last_alerted": float}
        self._state: Dict[int, dict] = {}

    def set_notifier(self, notifier: Notifier) -> None:
        self._notifier = notifier

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def check_once(self) -> List[Alert]:
        """Run one scan cycle.  Returns list of fired alerts."""
        fired: List[Alert] = []
        try:
            from backend.database import get_db, get_tracked_item_ids, get_price_history, get_item_flips, get_setting
            from backend.prediction.scoring import calculate_flip_metrics

            db = get_db()
            try:
                item_ids = get_tracked_item_ids(db)
                margin_threshold = get_setting(db, "margin_alert_threshold", default=50_000)
                vol_multiplier = float(get_setting(db, "volume_spike_multiplier", default=3.0))
            finally:
                db.close()

            for item_id in item_ids:
                alerts = await self._check_item(
                    item_id, margin_threshold, vol_multiplier
                )
                fired.extend(alerts)

        except Exception as exc:
            logger.error("AlertMonitor.check_once error: %s", exc)

        return fired

    async def run_forever(self) -> None:
        logger.info("AlertMonitor started (interval: %ds)", self.CHECK_INTERVAL)
        await asyncio.sleep(30)  # initial warm-up
        while True:
            try:
                alerts = await self.check_once()
                for alert in alerts:
                    await self._dispatch(alert)
            except Exception as exc:
                logger.error("AlertMonitor tick error: %s", exc)
            await asyncio.sleep(self.CHECK_INTERVAL)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _check_item(
        self,
        item_id: int,
        margin_threshold: int,
        vol_multiplier: float,
    ) -> List[Alert]:
        alerts: List[Alert] = []
        try:
            from backend.database import get_db, get_price_history, get_item_flips, get_item
            from backend.prediction.scoring import calculate_flip_metrics

            db = get_db()
            try:
                snaps = get_price_history(db, item_id, hours=4)
                flips = get_item_flips(db, item_id, days=30)
                item = get_item(db, item_id)
            finally:
                db.close()

            if not snaps:
                return alerts

            latest = snaps[-1]
            item_name = (item.name if item else None) or f"Item {item_id}"
            vol = (latest.buy_volume or 0) + (latest.sell_volume or 0)

            metrics = calculate_flip_metrics({
                "item_id": item_id,
                "item_name": item_name,
                "instant_buy": latest.instant_buy,
                "instant_sell": latest.instant_sell,
                "volume_5m": vol,
                "buy_time": latest.buy_time,
                "sell_time": latest.sell_time,
                "snapshots": snaps,
                "flip_history": flips,
            })

            now = time.time()
            state = self._state.setdefault(item_id, {
                "trend": metrics.get("trend", "NEUTRAL"),
                "last_alerted": 0.0,
                "avg_vol": float(vol),
            })

            cooldown_secs = self.COOLDOWN_MINUTES * 60
            in_cooldown = (now - state["last_alerted"]) < cooldown_secs

            if not metrics["vetoed"]:
                # Margin alert
                if not in_cooldown:
                    a = _check_margin(item_id, item_name, metrics.get("net_profit", 0), margin_threshold)
                    if a:
                        alerts.append(a)

                # Volume spike
                avg_v = state.get("avg_vol", 1.0)
                vs = _check_volume_spike(item_id, item_name, vol, avg_v, vol_multiplier)
                if vs:
                    alerts.append(vs)
                # Exponential moving average for volume baseline
                state["avg_vol"] = 0.8 * avg_v + 0.2 * vol

            # Trend reversal (checked even if vetoed, to warn of crashes)
            prev_trend = state.get("trend", "NEUTRAL")
            new_trend = metrics.get("trend", "NEUTRAL")
            if prev_trend != new_trend:
                tr = _check_trend_reversal(item_id, item_name, prev_trend, new_trend, latest.instant_buy or 0)
                if tr:
                    alerts.append(tr)
            state["trend"] = new_trend

            if alerts:
                state["last_alerted"] = now

        except Exception as exc:
            logger.warning("AlertMonitor._check_item(%d) error: %s", item_id, exc)

        return alerts

    async def _dispatch(self, alert: Alert) -> None:
        """Store alert in DB and send via notifier."""
        # Persist to database
        try:
            from backend.database import get_db, insert_alert, Alert as DbAlert
            db = get_db()
            try:
                insert_alert(db, DbAlert(
                    item_id=alert.item_id,
                    alert_type=alert.kind.value,
                    message=alert.message,
                    severity=alert.severity,
                    timestamp=alert.timestamp,
                ))
            finally:
                db.close()
        except Exception as exc:
            logger.warning("AlertMonitor: failed to persist alert: %s", exc)

        # Notify
        if self._notifier:
            try:
                await self._notifier.send(alert)
            except Exception as exc:
                logger.warning("AlertMonitor: notifier error: %s", exc)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_notifier_from_settings() -> Optional[Notifier]:
    """Build a notifier from the current DB settings.  Returns None if not configured."""
    try:
        from backend.database import get_db, get_setting
        db = get_db()
        try:
            webhook = get_setting(db, "discord_webhook", default=None)
            if isinstance(webhook, dict):
                enabled = webhook.get("enabled", False)
                url = webhook.get("url", "")
            else:
                enabled = get_setting(db, "discord_alerts_enabled", default=False)
                url = get_setting(db, "discord_webhook_url", default="")
        finally:
            db.close()

        if enabled and url:
            return DiscordNotifier(url)
    except Exception as exc:
        logger.warning("build_notifier_from_settings: %s", exc)
    return None
