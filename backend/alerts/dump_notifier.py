"""
backend.alerts.dump_notifier — Dump-specific Discord alert sender (v2).

Responsibilities
----------------
* Build a rich Discord embed that includes all dump-signal fields, trade-plan
  sizing, and a graphical price chart.
* Send via multipart/form-data when a PNG chart path is provided; fall back
  to a plain JSON POST without crashing.
* Resolve item IDs to real names via ``item_name_resolver``.

Usage::

    from backend.alerts.dump_notifier import DumpAlertV2, DumpAlertNotifierV2

    notifier = DumpAlertNotifierV2("https://discord.com/api/webhooks/...")
    notifier.send(alert)          # returns True on success
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import requests as _requests

from backend.alerts.item_name_resolver import resolve_item_name

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Embed colours (keyed by confidence level)
# ---------------------------------------------------------------------------

_COLOUR: dict[str, int] = {
    "HIGH":   0x00FF00,   # green  — high-confidence dump-buy opportunity
    "MEDIUM": 0xFFA726,   # orange
    "LOW":    0xEF5350,   # red    — low confidence, monitor only
}


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------

@dataclass
class DumpAlertV2:
    """Enriched dump alert produced by the v2 scanner."""

    item_id: int
    item_name: str           # Raw name; may be "Item XXXXX" — resolver fixes it
    current_price: int       # Insta-sell price (entry point for buyer)
    reference_price: int     # 4h average used as dump baseline
    drop_pct: float          # Percentage drop from reference_price
    drop_amount: int         # reference_price − current_price  (GP)
    sold_5m: int             # lowPriceVolume from /5m  (panic-sellers)
    bought_5m: int           # highPriceVolume from /5m
    sell_ratio: float        # sold_5m / (sold_5m + bought_5m)
    profit_per_item_net: int # Per-item net profit after GE tax (GP)
    predicted_recovery: int  # Mean-reversion price target (GP)
    confidence: str          # "HIGH" | "MEDIUM" | "LOW"

    # Trade plan (populated by build_trade_plan)
    qty_to_buy: int = 0
    max_invest_gp: int = 0
    estimated_total_profit: int = 0

    # Chart (PNG path written by chart_generator, or None)
    chart_path: Optional[str] = None

    timestamp: datetime = field(default_factory=datetime.utcnow)

    @property
    def resolved_name(self) -> str:
        """Return the real item name, falling back to Wiki lookup."""
        return resolve_item_name(self.item_id, self.item_name)


# ---------------------------------------------------------------------------
# Notifier
# ---------------------------------------------------------------------------

class DumpAlertNotifierV2:
    """Send DumpAlertV2 instances to Discord with chart image and rich embed.

    Uses multipart/form-data when a chart image path is provided.
    Falls back to a plain JSON POST if the chart is unavailable.
    Never raises — errors are logged instead.
    """

    def __init__(self, webhook_url: str) -> None:
        self._url = webhook_url

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def send(self, alert: DumpAlertV2) -> bool:
        """Dispatch the alert.  Returns True if Discord accepted it (200/204)."""
        if not self._url:
            logger.warning("DumpAlertNotifierV2.send: no webhook URL configured")
            return False

        name = alert.resolved_name
        embed = self._build_embed(alert, name)

        # Determine chart attachment
        chart_filename: Optional[str] = None
        if alert.chart_path and os.path.isfile(alert.chart_path):
            chart_filename = os.path.basename(alert.chart_path)
            embed["image"] = {"url": f"attachment://{chart_filename}"}

        payload = {"embeds": [embed]}
        return self._post(payload, alert.chart_path, chart_filename)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _build_embed(alert: DumpAlertV2, name: str) -> dict:
        """Construct the full Discord embed dict."""
        color = _COLOUR.get(alert.confidence, 0xFFA726)
        total_vol = alert.sold_5m + alert.bought_5m
        sell_ratio_str = f"{alert.sell_ratio:.0%}" if total_vol > 0 else "N/A"

        return {
            "title": f"DUMP DETECTED: {name}",
            "color": color,
            "timestamp": alert.timestamp.isoformat(),
            "footer": {"text": "OSRS Flipping AI • Dump Detector"},
            "fields": [
                {"name": "Item ID",              "value": str(alert.item_id),                              "inline": True},
                {"name": "Current Price",        "value": f"{alert.current_price:,} GP",                  "inline": True},
                {"name": "Ref Avg (4h)",         "value": f"{alert.reference_price:,} GP",                "inline": True},
                {"name": "Price Drop",           "value": f"-{alert.drop_pct:.1f}% ({alert.drop_amount:,} GP)", "inline": True},
                {"name": "Sell Ratio",           "value": sell_ratio_str,                                 "inline": True},
                {"name": "Sold 5m",              "value": f"{alert.sold_5m:,}",                           "inline": True},
                {"name": "Bought 5m",            "value": f"{alert.bought_5m:,}",                         "inline": True},
                {"name": "Profit / item (net)",  "value": f"{alert.profit_per_item_net:,} GP",            "inline": True},
                {"name": "Predicted Recovery",   "value": f"{alert.predicted_recovery:,} GP",             "inline": True},
                {"name": "Recommended Qty",      "value": f"{alert.qty_to_buy:,}",                        "inline": True},
                {"name": "Max Invest",           "value": f"{alert.max_invest_gp:,} GP",                  "inline": True},
                {"name": "Est Total Profit",     "value": f"{alert.estimated_total_profit:,} GP",         "inline": True},
            ],
        }

    def _post(
        self,
        payload: dict,
        chart_path: Optional[str],
        chart_filename: Optional[str],
    ) -> bool:
        """POST to the Discord webhook, with or without chart attachment."""
        try:
            if chart_path and chart_filename and os.path.isfile(chart_path):
                with open(chart_path, "rb") as img:
                    resp = _requests.post(
                        self._url,
                        data={"payload_json": json.dumps(payload)},
                        files={"file": (chart_filename, img, "image/png")},
                        timeout=15,
                    )
            else:
                resp = _requests.post(self._url, json=payload, timeout=10)

            if resp.status_code in (200, 204):
                return True

            logger.error(
                "DumpAlertNotifierV2: Discord returned HTTP %d: %s",
                resp.status_code,
                resp.text[:300],
            )
            return False

        except Exception as exc:
            logger.error("DumpAlertNotifierV2._post error: %s", exc)
            return False
