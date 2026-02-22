#!/usr/bin/env python3
"""
OSRS Dump Detector — v2

Detects sudden price crashes with recovery prediction and sends graphical
Discord alerts with real item names, trade-plan sizing, and quality filters
that reduce spam / junk signals.

v2 improvements
---------------
A) Item names resolved via ``backend.alerts.item_name_resolver`` (6-hour TTL
   Wiki cache) so alerts show "Dragon claws" not "Item 31099".
B) Discord embeds include a PNG chart attached as multipart/form-data.
C) Quality filters applied before alerting:
     • min_price_gp        ≥ 500 000 GP
     • min_drop_pct        ≥ 4.0 %
     • (sold_5m+bought_5m) ≥ 25  (activity confirmation)
     • sell_ratio          ≥ 0.80 (panic selling)
     • profit_per_item_net ≥ 2 000 GP
     • total_profit        ≥ 150 000 GP
     • cooldown            60 min per item
D) Trade plan / position sizing included in every embed.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import numpy as np
import requests

# ---------------------------------------------------------------------------
# Optional backend imports (graceful fallback if running standalone)
# ---------------------------------------------------------------------------

try:
    from backend.alerts.item_name_resolver import resolve_item_name
    _RESOLVER_AVAILABLE = True
except ImportError:
    _RESOLVER_AVAILABLE = False

    def resolve_item_name(item_id: int, fallback: Optional[str] = None) -> str:  # type: ignore[misc]
        if fallback and not fallback.startswith("Item "):
            return fallback
        return f"Item {item_id}"

try:
    from backend.alerts.dump_notifier import DumpAlertV2, DumpAlertNotifierV2
    _V2_NOTIFIER_AVAILABLE = True
except ImportError:
    _V2_NOTIFIER_AVAILABLE = False
    DumpAlertV2 = None  # type: ignore[assignment,misc]
    DumpAlertNotifierV2 = None  # type: ignore[assignment,misc]

try:
    from backend.analytics.trade_plan import build_trade_plan
    from backend.core.constants import GE_TAX_RATE, GE_TAX_CAP, GE_TAX_FREE_BELOW
    _TRADE_PLAN_AVAILABLE = True
except ImportError:
    _TRADE_PLAN_AVAILABLE = False
    GE_TAX_RATE = 0.02
    GE_TAX_CAP = 5_000_000
    GE_TAX_FREE_BELOW = 100
    build_trade_plan = None  # type: ignore[assignment]

try:
    from backend import config as _cfg
    _CONFIG_AVAILABLE = True
except ImportError:
    _CONFIG_AVAILABLE = False

try:
    from chart_generator import get_chart_generator
    _CHARTS_ENABLED = True
except ImportError:
    _CHARTS_ENABLED = False

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_cfg(attr: str, default):
    """Read a value from backend.config if available, else return default."""
    if _CONFIG_AVAILABLE:
        return getattr(_cfg, attr, default)
    return default


def _build_trade_plan(buy_price: int, sell_price: int, item_limit: Optional[int]) -> dict:
    """Compute trade-plan sizing.  Returns zeros if backend libs are absent."""
    capital = _read_cfg("DEFAULT_CAPITAL_GP", 50_000_000)
    pos_cap = _read_cfg("DUMP_V2_POSITION_CAP_PCT", 0.10)

    if _TRADE_PLAN_AVAILABLE and build_trade_plan is not None:
        try:
            return build_trade_plan(
                buy_price=buy_price,
                sell_price=sell_price,
                item_limit=item_limit,
                liquidity_score=None,
                risk_profile_position_cap_pct=pos_cap,
                capital_gp=capital,
                ge_tax_rate=GE_TAX_RATE,
                ge_tax_cap=GE_TAX_CAP,
                ge_tax_free_below=GE_TAX_FREE_BELOW,
            )
        except Exception as exc:
            logger.debug("_build_trade_plan error: %s", exc)

    # Fallback: manual sizing
    max_invest_gp = int(capital * pos_cap)
    qty = max_invest_gp // max(buy_price, 1)
    if item_limit:
        qty = min(qty, item_limit)
    tax = min(int(sell_price * GE_TAX_RATE), GE_TAX_CAP)
    profit_per = sell_price - buy_price - tax
    return {
        "buy_price": buy_price,
        "sell_price": sell_price,
        "qty_to_buy": max(qty, 0),
        "profit_per_item": max(profit_per, 0),
        "total_profit": max(profit_per * qty, 0),
        "max_invest_gp": max_invest_gp,
    }


# ---------------------------------------------------------------------------
# Original DumpAlert / DumpDetector (kept for backward compatibility)
# ---------------------------------------------------------------------------

@dataclass
class DumpAlert:
    """Represents a detected price dump (v1 — kept for back-compat)."""
    item_id: int
    item_name: str
    pre_dump_price: int
    current_price: int
    drop_amount: int
    drop_pct: float
    predicted_recovery: int
    recovery_profit: int
    confidence: str   # HIGH | MEDIUM | LOW
    risk_level: str   # LOW | MEDIUM | HIGH
    timestamp: datetime
    chart_path: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "item_id": self.item_id,
            "item_name": self.item_name,
            "pre_dump_price": self.pre_dump_price,
            "current_price": self.current_price,
            "drop_amount": self.drop_amount,
            "drop_pct": self.drop_pct,
            "predicted_recovery": self.predicted_recovery,
            "recovery_profit": self.recovery_profit,
            "confidence": self.confidence,
            "risk_level": self.risk_level,
            "timestamp": self.timestamp.isoformat(),
            "chart_path": self.chart_path,
        }


class DumpDetector:
    """
    Detects sudden price crashes and predicts recovery potential.

    v2 changes
    ----------
    * Uses ``item_name_resolver`` (6-hour Wiki mapping cache) instead of an
      ad-hoc per-instance cache with no TTL.
    * Adds ``scan_for_dumps_v2()`` with quality filters and trade-plan sizing.
    * ``_fetch_5m_prices()`` fetches the bulk 5-minute volume endpoint once per
      scan, avoiding per-item API calls for volume data.
    """

    API_URL = "https://prices.runescape.wiki/api/v1/osrs"
    HEADERS = {
        "User-Agent": (
            "OSRS-Flipping-AI/2.0 "
            "(github.com/Bakes982/osrs-flipping-ai; "
            "contact: mike.baker982@hotmail.com)"
        )
    }

    def __init__(self) -> None:
        # v1 thresholds (kept for backward compat)
        self.min_drop_pct = 3.0
        self.min_price = 1_000_000
        self.max_price = 2_000_000_000
        # v1 cooldown tracking
        self.recent_alerts: Dict[int, datetime] = {}
        self.alert_cooldown = timedelta(hours=1)
        # v1 internal caches (still used by old methods)
        self._price_cache: dict = {}
        self._mapping_cache: Optional[dict] = None  # legacy; resolver is authoritative now

    # ------------------------------------------------------------------
    # Data fetchers
    # ------------------------------------------------------------------

    def _get(self, path: str, **kwargs) -> dict:
        """GET helper — returns parsed JSON or {}."""
        try:
            resp = requests.get(
                f"{self.API_URL}/{path}",
                headers=self.HEADERS,
                timeout=10,
                **kwargs,
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            logger.warning("DumpDetector._get(%s): %s", path, exc)
            return {}

    def fetch_latest_prices(self) -> Dict:
        """Fetch current instant prices for all items (``/latest``)."""
        return self._get("latest").get("data", {})

    def _fetch_5m_prices(self) -> Dict:
        """Fetch 5-minute averaged prices and volumes for all items (``/5m``)."""
        return self._get("5m").get("data", {})

    def fetch_item_mapping(self) -> Dict:
        """Legacy mapping fetch — now delegates to ``item_name_resolver``."""
        if self._mapping_cache is None:
            try:
                resp = requests.get(
                    f"{self.API_URL}/mapping",
                    headers=self.HEADERS,
                    timeout=10,
                )
                items = resp.json()
                self._mapping_cache = {str(item["id"]): item for item in items}
            except Exception as exc:
                logger.warning("fetch_item_mapping: %s", exc)
                self._mapping_cache = {}
        return self._mapping_cache

    def fetch_price_history(self, item_id: int, hours: int = 24) -> List[Dict]:
        """Fetch 5-minute timeseries for ``item_id`` covering ``hours``."""
        try:
            resp = requests.get(
                f"{self.API_URL}/timeseries",
                headers=self.HEADERS,
                params={"timestep": "5m", "id": item_id},
                timeout=10,
            )
            data = resp.json().get("data", [])
            cutoff_ts = (datetime.now() - timedelta(hours=hours)).timestamp()
            return [d for d in data if d.get("timestamp", 0) >= cutoff_ts]
        except Exception as exc:
            logger.warning("fetch_price_history(%d): %s", item_id, exc)
            return []

    # ------------------------------------------------------------------
    # Statistical helpers
    # ------------------------------------------------------------------

    def calculate_historical_averages(self, history: List[Dict]) -> Dict:
        """Return 1h/4h/24h price statistics from timeseries data."""
        if not history:
            return {}
        now = datetime.now().timestamp()
        p1h, p4h, p24h, v1h = [], [], [], []
        for entry in history:
            ts = entry.get("timestamp", 0)
            high = entry.get("avgHighPrice")
            vol = entry.get("highPriceVolume", 0)
            if high:
                age_h = (now - ts) / 3600
                if age_h <= 1:
                    p1h.append(high)
                    v1h.append(vol)
                if age_h <= 4:
                    p4h.append(high)
                if age_h <= 24:
                    p24h.append(high)
        return {
            "avg_1h": float(np.mean(p1h)) if p1h else None,
            "avg_4h": float(np.mean(p4h)) if p4h else None,
            "avg_24h": float(np.mean(p24h)) if p24h else None,
            "std_24h": float(np.std(p24h)) if len(p24h) > 5 else None,
            "min_24h": min(p24h) if p24h else None,
            "max_24h": max(p24h) if p24h else None,
            "avg_volume_1h": float(np.mean(v1h)) if v1h else None,
            "price_count": len(p24h),
        }

    # ------------------------------------------------------------------
    # v1 detect_dump (kept for backward compat)
    # ------------------------------------------------------------------

    def detect_dump(self, item_id: int, current_price: int, history: List[Dict]) -> Optional[DumpAlert]:
        """v1 dump detection — kept for backward compatibility."""
        averages = self.calculate_historical_averages(history)
        if not averages.get("avg_4h"):
            return None
        ref = averages["avg_4h"]
        drop_amount = ref - current_price
        drop_pct = (drop_amount / ref) * 100
        if drop_pct < self.min_drop_pct:
            return None
        if item_id in self.recent_alerts:
            if datetime.now() - self.recent_alerts[item_id] < self.alert_cooldown:
                return None

        avg_24h = averages.get("avg_24h", ref)
        std_24h = averages.get("std_24h") or (ref * 0.02)
        z = (current_price - avg_24h) / std_24h
        if z < -2:
            rec_pct, confidence = 0.8, "HIGH"
        elif z < -1.5:
            rec_pct, confidence = 0.7, "HIGH"
        elif z < -1:
            rec_pct, confidence = 0.6, "MEDIUM"
        else:
            rec_pct, confidence = 0.5, "LOW"

        predicted_recovery = int(current_price + (drop_amount * rec_pct))
        gross = predicted_recovery - current_price
        tax = min(int(predicted_recovery * GE_TAX_RATE), GE_TAX_CAP)
        recovery_profit = int(gross - tax)
        if recovery_profit < 50_000:
            return None

        risk_level = "HIGH" if drop_pct > 10 else ("MEDIUM" if drop_pct > 5 else "LOW")

        # Use resolver for item name
        item_name = resolve_item_name(item_id)

        self.recent_alerts[item_id] = datetime.now()
        return DumpAlert(
            item_id=item_id,
            item_name=item_name,
            pre_dump_price=int(ref),
            current_price=current_price,
            drop_amount=int(drop_amount),
            drop_pct=drop_pct,
            predicted_recovery=predicted_recovery,
            recovery_profit=recovery_profit,
            confidence=confidence,
            risk_level=risk_level,
            timestamp=datetime.now(),
        )

    # ------------------------------------------------------------------
    # v2 scan — quality filters + 5m volume + trade plan
    # ------------------------------------------------------------------

    def scan_for_dumps_v2(self) -> "List[DumpAlertV2]":
        """
        v2 dump scan: quality filters, 5m volume, trade-plan sizing.

        Filters applied
        ---------------
        1. min_price_gp         ≥ 500 000 GP
        2. min_drop_pct         ≥ 4.0 %
        3. total_5m_volume      ≥ 25
        4. sell_ratio           ≥ 0.80
        5. profit_per_item_net  ≥ 2 000 GP
        6. estimated_total_profit ≥ 150 000 GP
        7. per-item cooldown    60 minutes
        """
        if not _V2_NOTIFIER_AVAILABLE:
            logger.error("scan_for_dumps_v2: backend.alerts.dump_notifier not importable")
            return []

        # Config (read from backend.config if available)
        min_price_gp          = _read_cfg("DUMP_V2_MIN_PRICE_GP",          500_000)
        min_drop_pct          = _read_cfg("DUMP_V2_MIN_DROP_PCT",           4.0)
        min_volume_trades     = _read_cfg("DUMP_V2_MIN_VOLUME_TRADES",      25)
        min_sell_ratio        = _read_cfg("DUMP_V2_MIN_SELL_RATIO",         0.80)
        min_profit_per_item   = _read_cfg("DUMP_V2_MIN_PROFIT_PER_ITEM",    2_000)
        min_total_profit      = _read_cfg("DUMP_V2_MIN_TOTAL_PROFIT",       150_000)
        cooldown_seconds      = _read_cfg("DUMP_V2_COOLDOWN_MINUTES",       60) * 60

        logger.info("scan_for_dumps_v2: starting scan")

        # ── Bulk API calls ────────────────────────────────────────────────
        latest_prices = self.fetch_latest_prices()          # /latest
        prices_5m     = self._fetch_5m_prices()             # /5m (bulk volume)
        mapping_v1    = self.fetch_item_mapping()           # /mapping (legacy dict for buy_limit)

        # ── Build candidate list applying cheap pre-filters ───────────────
        candidates = []
        for item_id_str, latest_data in latest_prices.items():
            try:
                item_id = int(item_id_str)
                low  = latest_data.get("low")   # insta-sell (entry)
                high = latest_data.get("high")  # insta-buy  (exit target)
                if not low or not high:
                    continue

                # Filter 1: minimum price
                if low < min_price_gp:
                    continue

                # 5m volume data
                data_5m   = prices_5m.get(item_id_str, {})
                sold_5m   = int(data_5m.get("lowPriceVolume",  0) or 0)
                bought_5m = int(data_5m.get("highPriceVolume", 0) or 0)
                total_5m  = sold_5m + bought_5m

                # Filter 2: must have some volume
                if total_5m < min_volume_trades:
                    continue

                # Filter 3: panic selling dominance
                sell_ratio = sold_5m / total_5m
                if sell_ratio < min_sell_ratio:
                    continue

                # Per-item cooldown
                if item_id in self.recent_alerts:
                    elapsed = (datetime.now() - self.recent_alerts[item_id]).total_seconds()
                    if elapsed < cooldown_seconds:
                        continue

                # Item name / limit from v1 mapping
                item_info  = mapping_v1.get(item_id_str, {})
                raw_name   = item_info.get("name", f"Item {item_id}")
                buy_limit  = item_info.get("limit") or None

                # Skip noted / unnamed junk
                if not raw_name or "(noted)" in raw_name.lower():
                    continue

                candidates.append({
                    "id":         item_id,
                    "name":       raw_name,
                    "low":        int(low),
                    "high":       int(high),
                    "sold_5m":    sold_5m,
                    "bought_5m":  bought_5m,
                    "sell_ratio": sell_ratio,
                    "buy_limit":  buy_limit,
                })
            except Exception:
                continue

        logger.info(
            "scan_for_dumps_v2: %d candidates after pre-filters (of %d items)",
            len(candidates), len(latest_prices),
        )

        # ── Per-candidate historical analysis ─────────────────────────────
        alerts: List[DumpAlertV2] = []
        # Sort by price desc to hit expensive high-value items first
        for candidate in sorted(candidates, key=lambda x: x["low"], reverse=True)[:300]:
            try:
                item_id   = candidate["id"]
                low       = candidate["low"]
                high      = candidate["high"]
                sold_5m   = candidate["sold_5m"]
                bought_5m = candidate["bought_5m"]
                sell_ratio = candidate["sell_ratio"]
                buy_limit  = candidate["buy_limit"]

                # Fetch history for 4h reference price
                history = self.fetch_price_history(item_id, hours=24)
                if len(history) < 5:
                    continue

                averages = self.calculate_historical_averages(history)
                ref_price = averages.get("avg_4h")
                if not ref_price:
                    continue

                drop_amount = ref_price - low
                drop_pct    = (drop_amount / ref_price) * 100

                # Filter 4: minimum drop
                if drop_pct < min_drop_pct:
                    continue

                # Per-item profit after GE tax
                tax_on_sell = min(int(high * GE_TAX_RATE), GE_TAX_CAP)
                profit_per_item_net = high - low - tax_on_sell

                # Filter 5: minimum per-item net profit
                if profit_per_item_net < min_profit_per_item:
                    continue

                # Trade plan
                plan = _build_trade_plan(
                    buy_price=low,
                    sell_price=high,
                    item_limit=buy_limit,
                )
                qty_to_buy    = plan.get("qty_to_buy", 0)
                max_invest_gp = plan.get("max_invest_gp", 0)
                est_total_profit = profit_per_item_net * qty_to_buy

                # Filter 6: minimum total profit
                if est_total_profit < min_total_profit:
                    continue

                # Confidence from z-score
                avg_24h = averages.get("avg_24h", ref_price)
                std_24h = averages.get("std_24h") or (ref_price * 0.02)
                z = (low - avg_24h) / std_24h
                if z < -2:
                    rec_pct, confidence = 0.8, "HIGH"
                elif z < -1.5:
                    rec_pct, confidence = 0.7, "HIGH"
                elif z < -1:
                    rec_pct, confidence = 0.6, "MEDIUM"
                else:
                    rec_pct, confidence = 0.5, "LOW"

                predicted_recovery = int(low + (drop_amount * rec_pct))

                # ── Chart ──────────────────────────────────────────────────
                chart_path: Optional[str] = None
                if _CHARTS_ENABLED:
                    try:
                        chart_gen = get_chart_generator()
                        chart_path = chart_gen.create_dump_alert_chart(
                            item_name=candidate["name"],
                            item_id=item_id,
                            dump_price=low,
                            pre_dump_price=int(ref_price),
                            predicted_recovery=predicted_recovery,
                            drop_pct=drop_pct,
                            hours=6,
                        )
                    except Exception as chart_exc:
                        logger.warning(
                            "chart generation failed for item %d: %s",
                            item_id, chart_exc,
                        )

                # ── Build v2 alert ─────────────────────────────────────────
                alert = DumpAlertV2(
                    item_id=item_id,
                    item_name=candidate["name"],
                    current_price=low,
                    reference_price=int(ref_price),
                    drop_pct=drop_pct,
                    drop_amount=int(drop_amount),
                    sold_5m=sold_5m,
                    bought_5m=bought_5m,
                    sell_ratio=sell_ratio,
                    profit_per_item_net=profit_per_item_net,
                    predicted_recovery=predicted_recovery,
                    confidence=confidence,
                    qty_to_buy=qty_to_buy,
                    max_invest_gp=max_invest_gp,
                    estimated_total_profit=est_total_profit,
                    chart_path=chart_path,
                )

                self.recent_alerts[item_id] = datetime.now()
                alerts.append(alert)

                logger.info(
                    "DUMP DETECTED (v2): %s  -%.1f%%  net_profit/item=%d GP  total=%d GP",
                    alert.resolved_name, drop_pct, profit_per_item_net, est_total_profit,
                )

            except Exception as exc:
                logger.debug(
                    "scan_for_dumps_v2: error processing item %s: %s",
                    candidate.get("id", "?"), exc,
                )
                continue

        # Sort by estimated total profit
        alerts.sort(key=lambda a: a.estimated_total_profit, reverse=True)
        logger.info("scan_for_dumps_v2: found %d quality dump alerts", len(alerts))
        return alerts

    # ------------------------------------------------------------------
    # v1 scan (kept for back-compat)
    # ------------------------------------------------------------------

    def scan_for_dumps(
        self,
        min_profit: int = 100_000,
        risk_filter: str = "ALL",
    ) -> List[DumpAlert]:
        """v1 scan — kept for backward compatibility.  Use scan_for_dumps_v2."""
        logger.info("scan_for_dumps (v1): scanning...")
        alerts: List[DumpAlert] = []
        prices  = self.fetch_latest_prices()
        mapping = self.fetch_item_mapping()

        candidates = []
        for item_id_str, price_data in prices.items():
            try:
                item_id = int(item_id_str)
                high    = price_data.get("high")
                low     = price_data.get("low")
                if not high or not low:
                    continue
                if low < self.min_price or high > self.max_price:
                    continue
                item_info = mapping.get(item_id_str, {})
                item_name = item_info.get("name", "")
                if not item_name or "(noted)" in item_name.lower():
                    continue
                candidates.append({"id": item_id, "name": item_name, "high": high, "low": low})
            except Exception:
                continue

        candidates.sort(key=lambda x: x["high"], reverse=True)
        for i, c in enumerate(candidates[:500]):
            try:
                history = self.fetch_price_history(c["id"], hours=24)
                if len(history) < 10:
                    continue
                alert = self.detect_dump(c["id"], c["low"], history)
                if alert:
                    if alert.recovery_profit < min_profit:
                        continue
                    if risk_filter != "ALL" and alert.risk_level != risk_filter:
                        continue
                    if _CHARTS_ENABLED:
                        try:
                            chart_gen = get_chart_generator()
                            alert.chart_path = chart_gen.create_dump_alert_chart(
                                item_name=alert.item_name,
                                item_id=alert.item_id,
                                dump_price=alert.current_price,
                                pre_dump_price=alert.pre_dump_price,
                                predicted_recovery=alert.predicted_recovery,
                                drop_pct=alert.drop_pct,
                                hours=6,
                            )
                        except Exception as e:
                            logger.debug("chart error: %s", e)
                    alerts.append(alert)
            except Exception:
                continue
            if (i + 1) % 100 == 0:
                logger.info("v1 scan: checked %d/500 items", i + 1)

        alerts.sort(key=lambda x: x.recovery_profit, reverse=True)
        return alerts

    def get_dump_summary(self, alerts: List[DumpAlert]) -> str:
        if not alerts:
            return "No dump alerts at this time."
        lines = ["# DUMP ALERTS", f"*{len(alerts)} opportunities detected*", ""]
        for i, alert in enumerate(alerts[:10], 1):
            ce = {"HIGH": "[+++]", "MEDIUM": "[++]", "LOW": "[+]"}[alert.confidence]
            re_ = {"LOW": "[Safe]", "MEDIUM": "[Mod]", "HIGH": "[Risk]"}[alert.risk_level]
            lines += [
                f"**{i}. {alert.item_name}**",
                f"   Drop: -{alert.drop_pct:.1f}% ({alert.drop_amount:,} GP)",
                f"   Buy Now: {alert.current_price:,} GP",
                f"   Target: {alert.predicted_recovery:,} GP",
                f"   Profit: {alert.recovery_profit:,} GP",
                f"   {ce} {re_}",
                "",
            ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# v1 Notifier (kept for backward compat)
# ---------------------------------------------------------------------------

class DumpAlertNotifier:
    """v1 Discord notifier — kept for backward compatibility.

    New code should use ``DumpAlertNotifierV2`` from
    ``backend.alerts.dump_notifier`` instead.
    """

    def __init__(self, webhook_url: str) -> None:
        self.webhook_url = webhook_url

    def send_dump_alert(self, alert: DumpAlert) -> bool:
        import json
        colors = {"HIGH": 0x00FF00, "MEDIUM": 0xFFAA00, "LOW": 0xFF4444}
        item_name = resolve_item_name(alert.item_id, alert.item_name)
        embed: dict = {
            "title":       f"DUMP ALERT: {item_name}",
            "description": f"Price dropped **-{alert.drop_pct:.1f}%** from recent average!",
            "color":       colors.get(alert.confidence, 0xFFAA00),
            "fields": [
                {"name": "Pre-Dump Price",      "value": f"{alert.pre_dump_price:,} GP",    "inline": True},
                {"name": "Current Price",        "value": f"{alert.current_price:,} GP",     "inline": True},
                {"name": "Drop",                 "value": f"-{alert.drop_amount:,} GP (-{alert.drop_pct:.1f}%)", "inline": True},
                {"name": "Predicted Recovery",   "value": f"{alert.predicted_recovery:,} GP","inline": True},
                {"name": "Potential Profit",     "value": f"{alert.recovery_profit:,} GP",   "inline": True},
                {"name": "Confidence",           "value": alert.confidence,                  "inline": True},
                {"name": "Risk Level",           "value": alert.risk_level,                  "inline": True},
            ],
            "timestamp": alert.timestamp.isoformat(),
            "footer":    {"text": "OSRS Flipping AI • Dump Detector"},
        }
        payload = {"embeds": [embed]}
        try:
            if alert.chart_path and os.path.exists(alert.chart_path):
                fname = os.path.basename(alert.chart_path)
                embed["image"] = {"url": f"attachment://{fname}"}
                with open(alert.chart_path, "rb") as f:
                    resp = requests.post(
                        self.webhook_url,
                        data={"payload_json": json.dumps(payload)},
                        files={"file": (fname, f, "image/png")},
                        timeout=10,
                    )
            else:
                resp = requests.post(self.webhook_url, json=payload, timeout=10)
            return resp.status_code in (200, 204)
        except Exception as exc:
            logger.error("DumpAlertNotifier error: %s", exc)
            return False

    def send_dump_summary(self, alerts: List[DumpAlert]) -> bool:
        if not alerts:
            return False
        import json
        embed: dict = {
            "title":       f"DUMP ALERT SUMMARY — {len(alerts)} Opportunities!",
            "description": "Items with significant price drops detected",
            "color":       0xFF4444,
            "fields":      [],
            "timestamp":   datetime.now().isoformat(),
            "footer":      {"text": "OSRS Flipping AI • Dump Detector"},
        }
        for alert in alerts[:5]:
            name = resolve_item_name(alert.item_id, alert.item_name)
            embed["fields"].append({
                "name": f"{name} (-{alert.drop_pct:.1f}%)",
                "value": (
                    f"Buy: {alert.current_price:,} GP\n"
                    f"Target: {alert.predicted_recovery:,} GP\n"
                    f"Profit: {alert.recovery_profit:,} GP\n"
                    f"Risk: {alert.risk_level}"
                ),
                "inline": False,
            })
        try:
            resp = requests.post(self.webhook_url, json={"embeds": [embed]}, timeout=10)
            return resp.status_code in (200, 204)
        except Exception as exc:
            logger.error("DumpAlertNotifier.send_dump_summary error: %s", exc)
            return False


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_dump_detector: Optional[DumpDetector] = None


def get_dump_detector() -> DumpDetector:
    global _dump_detector
    if _dump_detector is None:
        _dump_detector = DumpDetector()
    return _dump_detector


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    print("OSRS Dump Detector v2")
    print("=" * 60)

    detector = get_dump_detector()

    # ── v2 scan ──────────────────────────────────────────────────────────
    webhook_url = os.environ.get("DISCORD_WEBHOOK", "")
    alerts_v2 = detector.scan_for_dumps_v2()

    if alerts_v2:
        print(f"\nFound {len(alerts_v2)} v2 dump alerts:")
        for a in alerts_v2:
            print(
                f"  {a.resolved_name}  -{a.drop_pct:.1f}%  "
                f"profit/item={a.profit_per_item_net:,} GP  "
                f"total={a.estimated_total_profit:,} GP  "
                f"[{a.confidence}]"
            )

        if webhook_url:
            notifier = DumpAlertNotifierV2(webhook_url)
            for alert in alerts_v2:
                sent = notifier.send(alert)
                status = "✓" if sent else "✗"
                print(f"  {status} Discord: {alert.resolved_name}")
        else:
            print("\nSet DISCORD_WEBHOOK env var to send alerts to Discord.")
    else:
        print("\nNo quality dump alerts found at this time.")
