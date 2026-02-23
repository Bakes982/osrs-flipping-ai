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


def _cfg_bool(attr: str, default: bool = False) -> bool:
    """Read a boolean from backend.config, with a safe fallback."""
    try:
        from backend import config as _c
        return bool(getattr(_c, attr, default))
    except Exception:
        return default


# ---------------------------------------------------------------------------
# Embed colours and star ratings (keyed by confidence level)
# ---------------------------------------------------------------------------

_COLOUR: dict[str, int] = {
    "HIGH":   0x00FF00,   # green  — high-confidence dump-buy opportunity
    "MEDIUM": 0xFFA726,   # orange
    "LOW":    0xEF5350,   # red    — low confidence, monitor only
}

_STARS: dict[str, str] = {
    "HIGH":   "⭐⭐⭐  HIGH",
    "MEDIUM": "⭐⭐☆  MEDIUM",
    "LOW":    "⭐☆☆  LOW",
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

        # Suppress ⭐☆☆ LOW alerts when DUMP_SUPPRESS_LOW=True (default).
        if _cfg_bool("DUMP_SUPPRESS_LOW", True) and alert.confidence == "LOW":
            logger.info(
                "DumpAlertNotifierV2: suppressing LOW-confidence alert for item %d (%s)",
                alert.item_id, alert.item_name,
            )
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
        """Copilot-style compact embed: stars title, 4-line trade plan, 3 fields.

        Format
        ------
        Title:       "{name}  ⭐⭐⭐  HIGH"
        Description: 4-line trade plan (buy → sell, qty / max invest,
                     est profit, Wiki + Prices links)
        Fields:      Drop % | Sell Ratio | Volume 5m  (3 inline only)
        Footer:      "OSRS Flipping AI • Dump Detector · X min ago"
        """
        color = _COLOUR.get(alert.confidence, 0xFFA726)
        stars = _STARS.get(alert.confidence, "⭐⭐⭐  HIGH")

        total_vol = alert.sold_5m + alert.bought_5m
        sell_ratio_str = f"{alert.sell_ratio:.0%}" if total_vol > 0 else "N/A"

        # Relative time string for footer
        elapsed_s = (datetime.utcnow() - alert.timestamp).total_seconds()
        if elapsed_s < 90:
            time_str = "just now"
        elif elapsed_s < 3600:
            time_str = f"{int(elapsed_s / 60)} min ago"
        else:
            time_str = f"{elapsed_s / 3600:.1f}h ago"

        # External links
        name_url = name.replace(" ", "_")
        wiki_url   = f"https://oldschool.runescape.wiki/w/{name_url}"
        prices_url = f"https://prices.runescape.wiki/osrs/item/{alert.item_id}"

        # Profit % on invested capital
        profit_pct = (
            (alert.estimated_total_profit / alert.max_invest_gp) * 100
            if alert.max_invest_gp > 0
            else 0.0
        )

        description = "\n".join([
            f"💰 **Buy:** {alert.current_price:,} GP → **Sell:** {alert.predicted_recovery:,} GP",
            f"📦 **Qty:** {alert.qty_to_buy:,}  |  **Max Invest:** {alert.max_invest_gp:,} GP",
            f"📈 **Est Profit:** {alert.estimated_total_profit:,} GP  (+{profit_pct:.1f}%)",
            f"🔗 [Wiki]({wiki_url}) | [Prices]({prices_url})",
        ])

        return {
            "title":       f"{name}  {stars}",
            "description": description,
            "color":       color,
            "timestamp":   alert.timestamp.isoformat(),
            "footer":      {"text": f"OSRS Flipping AI • Dump Detector · {time_str}"},
            "fields": [
                {"name": "Drop",       "value": f"-{alert.drop_pct:.1f}%  ({alert.drop_amount:,} GP)", "inline": True},
                {"name": "Sell Ratio", "value": sell_ratio_str,                                        "inline": True},
                {"name": "Volume 5m",  "value": f"{total_vol:,}",                                      "inline": True},
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
