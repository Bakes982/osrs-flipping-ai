"""
Discord Webhook Notifier for OSRS Flipping AI

Sends rich embed messages for:
- Top 5 flip opportunity alerts (with price charts)
- Dump alerts
- Custom notifications

Each opportunity gets its own embed with an inline price chart image.
"""

import io
import os
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import httpx
import matplotlib
matplotlib.use("Agg")  # headless backend – no GUI needed on server
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

logger = logging.getLogger(__name__)

# ── OSRS-inspired colour palette ──────────────────────────────────────────
COLORS = {
    "background": "#1a1a2e",
    "panel": "#16213e",
    "green": "#00ff00",
    "red": "#ff4444",
    "gold": "#ffd700",
    "blue": "#4fc3f7",
    "orange": "#ff9800",
    "purple": "#9c27b0",
    "text": "#ffffff",
    "text_dim": "#888888",
    "grid": "#333333",
}

WIKI_BASE = "https://prices.runescape.wiki/api/v1/osrs"
WIKI_HEADERS = {"User-Agent": "OSRS-AI-Flipper v2.0 - Discord: bakes982"}


# ── Chart helpers ─────────────────────────────────────────────────────────

def _format_gp(value: float) -> str:
    if value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.1f}B"
    elif value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    elif value >= 1_000:
        return f"{value / 1_000:.1f}K"
    return f"{value:.0f}"


def _fetch_price_history(item_id: int, hours: int = 6) -> Dict:
    """Fetch 5-minute timeseries from OSRS Wiki (sync)."""
    import requests
    try:
        resp = requests.get(
            f"{WIKI_BASE}/timeseries?timestep=5m&id={item_id}",
            headers=WIKI_HEADERS,
            timeout=10,
        )
        data = resp.json().get("data", [])
        cutoff = (datetime.utcnow() - timedelta(hours=hours)).timestamp()
        filtered = [d for d in data if d.get("timestamp", 0) >= cutoff]
        return {
            "timestamps": [datetime.utcfromtimestamp(d["timestamp"]) for d in filtered],
            "high": [d.get("avgHighPrice") for d in filtered],
            "low": [d.get("avgLowPrice") for d in filtered],
            "high_vol": [d.get("highPriceVolume", 0) for d in filtered],
            "low_vol": [d.get("lowPriceVolume", 0) for d in filtered],
        }
    except Exception as e:
        logger.warning("Price history fetch failed for %d: %s", item_id, e)
        return {"timestamps": [], "high": [], "low": [], "high_vol": [], "low_vol": []}


def generate_opportunity_chart(
    item_name: str,
    item_id: int,
    buy_price: int,
    sell_price: int,
    score: float,
    trend: str = "NEUTRAL",
    hours: int = 6,
) -> Optional[bytes]:
    """Generate a PNG price chart for an opportunity and return raw bytes.

    Returns ``None`` if there is not enough price data.
    """
    plt.style.use("dark_background")
    history = _fetch_price_history(item_id, hours)

    if len(history["timestamps"]) < 3:
        return None

    fig, (ax_price, ax_vol) = plt.subplots(
        2, 1,
        figsize=(10, 5.5),
        gridspec_kw={"height_ratios": [3, 1]},
        facecolor=COLORS["background"],
    )

    # ── Price subplot ─────────────────────────────────────────────────
    ax_price.set_facecolor(COLORS["panel"])

    valid_high = [(t, p) for t, p in zip(history["timestamps"], history["high"]) if p]
    valid_low = [(t, p) for t, p in zip(history["timestamps"], history["low"]) if p]

    if valid_high:
        ht, hv = zip(*valid_high)
        ax_price.plot(ht, hv, color=COLORS["green"], linewidth=2, label="Insta-Buy")
    if valid_low:
        lt, lv = zip(*valid_low)
        ax_price.plot(lt, lv, color=COLORS["red"], linewidth=2, label="Insta-Sell")

    # Buy / sell zone lines
    ax_price.axhline(y=buy_price, color=COLORS["green"], ls="--", lw=1.5, alpha=0.7, label=f"Buy {_format_gp(buy_price)}")
    ax_price.axhline(y=sell_price, color=COLORS["gold"], ls="--", lw=1.5, alpha=0.7, label=f"Sell {_format_gp(sell_price)}")
    ax_price.axhspan(buy_price, sell_price, alpha=0.10, color=COLORS["green"])

    # Trend arrow annotation
    trend_symbols = {"RISING": "▲ RISING", "FALLING": "▼ FALLING", "NEUTRAL": "► NEUTRAL"}
    trend_colors = {"RISING": COLORS["green"], "FALLING": COLORS["red"], "NEUTRAL": COLORS["blue"]}
    ax_price.text(
        0.98, 0.05,
        trend_symbols.get(trend, trend),
        transform=ax_price.transAxes,
        fontsize=12, fontweight="bold",
        color=trend_colors.get(trend, COLORS["text"]),
        ha="right", va="bottom",
    )

    # Info box
    profit = sell_price - buy_price
    tax = int(min(sell_price * 0.02, 5_000_000))
    net = profit - tax
    info = (
        f"Buy: {_format_gp(buy_price)}\n"
        f"Sell: {_format_gp(sell_price)}\n"
        f"Net Profit: {_format_gp(net)}\n"
        f"Score: {score:.0f}/100"
    )
    props = dict(boxstyle="round,pad=0.4", facecolor=COLORS["panel"], edgecolor=COLORS["gold"], alpha=0.9)
    ax_price.text(0.02, 0.97, info, transform=ax_price.transAxes, fontsize=10,
                  va="top", bbox=props, color=COLORS["text"], family="monospace")

    ax_price.set_title(item_name, fontsize=14, fontweight="bold", color=COLORS["text"], pad=12)
    ax_price.set_ylabel("Price (GP)", fontsize=10, color=COLORS["text"])
    ax_price.tick_params(colors=COLORS["text"], labelsize=8)
    ax_price.grid(True, alpha=0.25, color=COLORS["grid"])
    ax_price.legend(loc="upper right", fontsize=8, facecolor=COLORS["panel"], edgecolor=COLORS["grid"])
    ax_price.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: _format_gp(x)))

    # ── Volume subplot ────────────────────────────────────────────────
    ax_vol.set_facecolor(COLORS["panel"])
    if valid_high:
        hvols = [history["high_vol"][i] for i, p in enumerate(history["high"]) if p]
        ax_vol.bar(ht, hvols, width=0.002, color=COLORS["green"], alpha=0.6, label="Buy Vol")
    if valid_low:
        lvols = [history["low_vol"][i] for i, p in enumerate(history["low"]) if p]
        ax_vol.bar(lt, lvols, width=0.002, color=COLORS["red"], alpha=0.6, label="Sell Vol")

    ax_vol.set_ylabel("Vol", fontsize=9, color=COLORS["text"])
    ax_vol.tick_params(colors=COLORS["text"], labelsize=8)
    ax_vol.grid(True, alpha=0.25, color=COLORS["grid"])
    ax_vol.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))

    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, facecolor=COLORS["background"], bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


# ── Score colour helper ───────────────────────────────────────────────────

def _score_color(score: float) -> int:
    """Return a Discord embed colour based on flip score."""
    if score >= 70:
        return 0x00FF00  # green
    if score >= 50:
        return 0xFFD700  # gold
    if score >= 35:
        return 0xFF9800  # orange
    return 0xFF4444      # red


def _trend_emoji(trend: str) -> str:
    return {"RISING": "📈", "FALLING": "📉"}.get(trend, "➡️")


# ── Main notifier class ──────────────────────────────────────────────────

class DiscordOpportunityNotifier:
    """Send top‑5 flip opportunities to a Discord webhook with charts."""

    def __init__(self, webhook_url: str):
        self.webhook_url = webhook_url

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def send_top_opportunities(
        self,
        opportunities: List[Dict],
        *,
        max_items: int = 5,
        include_charts: bool = True,
    ) -> bool:
        """Send a header embed followed by one embed per opportunity.

        ``opportunities`` is a list of dicts as returned by the
        ``/api/opportunities`` endpoint (or built from FlipScore).

        Returns True if at least the header was delivered successfully.
        """
        if not opportunities:
            logger.info("No opportunities to send to Discord")
            return False

        top = opportunities[:max_items]

        # ── 1. Header embed ──────────────────────────────────────────
        header_embed = {
            "title": f"🏆 Top {len(top)} Flip Opportunities",
            "description": (
                f"Scanned at <t:{int(datetime.utcnow().timestamp())}:t>\n"
                "Ranked by AI composite score"
            ),
            "color": 0xFFD700,
            "footer": {"text": "OSRS AI Flipper • Auto-scan"},
            "timestamp": datetime.utcnow().isoformat(),
        }
        self._post_json({"embeds": [header_embed]})

        # ── 2. One message per item (embed + chart image) ────────────
        success_count = 0
        for rank, opp in enumerate(top, 1):
            try:
                ok = self._send_single_opportunity(rank, opp, include_charts)
                if ok:
                    success_count += 1
            except Exception as e:
                logger.error("Failed to send opportunity #%d: %s", rank, e)

        logger.info("Discord: sent %d/%d opportunities", success_count, len(top))
        return success_count > 0

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _send_single_opportunity(
        self, rank: int, opp: Dict, include_chart: bool
    ) -> bool:
        """Build and send one embed with an optional chart attachment."""
        name = opp.get("name") or opp.get("item_name") or f"Item {opp.get('item_id')}"
        item_id = opp.get("item_id", 0)
        buy = opp.get("buy_price") or opp.get("recommended_buy") or 0
        sell = opp.get("sell_price") or opp.get("recommended_sell") or 0
        profit = opp.get("potential_profit") or opp.get("expected_profit") or 0
        score = opp.get("flip_score") or opp.get("total_score") or 0
        trend = opp.get("trend", "NEUTRAL")
        volume = opp.get("volume") or opp.get("volume_5m") or 0
        margin_pct = opp.get("margin_pct") or opp.get("roi_pct") or 0
        ml_dir = opp.get("ml_direction")
        ml_conf = opp.get("ml_prediction_confidence") or opp.get("ml_confidence")
        ml_method = opp.get("ml_method")
        win_rate = opp.get("win_rate")
        reason = opp.get("reason", "")

        # Tax
        tax = int(min(sell * 0.02, 5_000_000)) if sell else 0
        net_profit = profit - tax if profit else 0

        # Build embed fields
        fields = [
            {"name": "💰 Buy Price", "value": f"{buy:,} GP", "inline": True},
            {"name": "💸 Sell Price", "value": f"{sell:,} GP", "inline": True},
            {"name": "📊 Net Profit", "value": f"+{net_profit:,} GP ({margin_pct:.1f}%)", "inline": True},
            {"name": "📦 Volume (5m)", "value": f"{volume:,}", "inline": True},
            {"name": f"{_trend_emoji(trend)} Trend", "value": trend, "inline": True},
            {"name": "⭐ Score", "value": f"**{score:.0f}**/100", "inline": True},
        ]

        # ML prediction row (if available)
        if ml_dir:
            ml_text = f"{ml_dir}"
            if ml_conf:
                ml_text += f" ({ml_conf:.0%})"
            if ml_method:
                ml_text += f" • {ml_method}"
            fields.append({"name": "🤖 AI Signal", "value": ml_text, "inline": True})

        # Win rate
        if win_rate is not None:
            fields.append({"name": "🎯 Win Rate", "value": f"{win_rate:.0f}%", "inline": True})

        embed = {
            "title": f"#{rank}  {name}",
            "color": _score_color(score),
            "fields": fields,
            "footer": {"text": reason[:100] if reason else "OSRS AI Flipper"},
            "timestamp": datetime.utcnow().isoformat(),
        }

        # Generate and attach chart
        chart_bytes = None
        if include_chart:
            try:
                chart_bytes = generate_opportunity_chart(
                    item_name=name,
                    item_id=item_id,
                    buy_price=buy,
                    sell_price=sell,
                    score=score,
                    trend=trend,
                    hours=6,
                )
            except Exception as e:
                logger.warning("Chart generation failed for %s: %s", name, e)

        if chart_bytes:
            filename = f"chart_{item_id}.png"
            embed["image"] = {"url": f"attachment://{filename}"}
            return self._post_multipart(embed, filename, chart_bytes)
        else:
            return self._post_json({"embeds": [embed]})

    # ------------------------------------------------------------------
    # HTTP helpers
    # ------------------------------------------------------------------

    def _post_json(self, payload: dict) -> bool:
        import requests
        try:
            r = requests.post(self.webhook_url, json=payload, timeout=15)
            if r.status_code == 429:
                # Rate-limited – wait and retry once
                import time
                retry_after = r.json().get("retry_after", 2)
                logger.warning("Discord rate-limited, waiting %.1fs", retry_after)
                time.sleep(retry_after)
                r = requests.post(self.webhook_url, json=payload, timeout=15)
            return r.status_code in (200, 204)
        except Exception as e:
            logger.error("Discord POST failed: %s", e)
            return False

    def _post_multipart(self, embed: dict, filename: str, image_bytes: bytes) -> bool:
        import requests, json
        try:
            payload_json = json.dumps({"embeds": [embed]})
            files = {"file": (filename, io.BytesIO(image_bytes), "image/png")}
            r = requests.post(
                self.webhook_url,
                data={"payload_json": payload_json},
                files=files,
                timeout=15,
            )
            if r.status_code == 429:
                import time
                retry_after = r.json().get("retry_after", 2)
                logger.warning("Discord rate-limited, waiting %.1fs", retry_after)
                time.sleep(retry_after)
                files = {"file": (filename, io.BytesIO(image_bytes), "image/png")}
                r = requests.post(
                    self.webhook_url,
                    data={"payload_json": payload_json},
                    files=files,
                    timeout=15,
                )
            return r.status_code in (200, 204)
        except Exception as e:
            logger.error("Discord multipart POST failed: %s", e)
            return False
