"""Price/time-series helpers for item detail endpoints."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import requests

logger = logging.getLogger(__name__)

WIKI_BASE = "https://prices.runescape.wiki/api/v1/osrs"
WIKI_HEADERS = {"User-Agent": "osrs-flipping-ai"}

RANGE_SPECS: Dict[str, Tuple[str, int]] = {
    "30h": ("5m", 360),
    "15d": ("1h", 360),
    "3m": ("6h", 360),
    "1y": ("24h", 365),
}

GRAPH_RANGE_SPECS: Dict[str, Tuple[str, int]] = {
    "1h": ("5m", 12),
    "6h": ("5m", 72),
    "24h": ("1h", 24),
    "7d": ("1h", 168),
}


def normalize_range(range_key: Optional[str]) -> str:
    key = (range_key or "15d").strip().lower()
    return key if key in RANGE_SPECS else "15d"


def normalize_graph_range(range_key: Optional[str]) -> str:
    key = (range_key or "24h").strip().lower()
    return key if key in GRAPH_RANGE_SPECS else "24h"


def get_price_series(item_id: int, range_key: str) -> Dict[str, List[int]]:
    """Return aligned arrays for charting: ts, buy, sell, volume."""
    normalized = normalize_range(range_key)
    timestep, max_points = RANGE_SPECS[normalized]

    try:
        resp = requests.get(
            f"{WIKI_BASE}/timeseries",
            params={"timestep": timestep, "id": item_id},
            headers=WIKI_HEADERS,
            timeout=15,
        )
        resp.raise_for_status()
        raw_points = resp.json().get("data", [])
    except Exception as exc:
        logger.warning("get_price_series failed item_id=%s range=%s: %s", item_id, normalized, exc)
        return {"ts": [], "buy": [], "sell": [], "volume": []}

    ts: List[int] = []
    buy: List[int] = []
    sell: List[int] = []
    volume: List[int] = []
    last_buy: Optional[int] = None
    last_sell: Optional[int] = None

    for pt in raw_points:
        t = pt.get("timestamp")
        high = pt.get("avgHighPrice")
        low = pt.get("avgLowPrice")
        if t is None or (high is None and low is None):
            continue

        buy_price = int(high) if high is not None else last_buy
        sell_price = int(low) if low is not None else last_sell

        if buy_price is None and sell_price is None:
            continue
        if buy_price is None:
            buy_price = sell_price
        if sell_price is None:
            sell_price = buy_price

        total_vol = int(pt.get("highPriceVolume") or 0) + int(pt.get("lowPriceVolume") or 0)
        ts.append(int(t))
        buy.append(int(buy_price))
        sell.append(int(sell_price))
        volume.append(total_vol)
        last_buy = int(buy_price)
        last_sell = int(sell_price)

    if max_points > 0 and len(ts) > max_points:
        ts = ts[-max_points:]
        buy = buy[-max_points:]
        sell = sell[-max_points:]
        volume = volume[-max_points:]

    return {"ts": ts, "buy": buy, "sell": sell, "volume": volume}


def get_latest_quote(item_id: int) -> Optional[Dict[str, int]]:
    """Return latest GE quote: buy/high, sell/low, volume_5m."""
    try:
        resp = requests.get(f"{WIKI_BASE}/latest", headers=WIKI_HEADERS, timeout=10)
        resp.raise_for_status()
        row = resp.json().get("data", {}).get(str(item_id))
        if not row:
            return None
        buy = row.get("high")
        sell = row.get("low")
        if buy is None and sell is None:
            return None
        if buy is None:
            buy = sell
        if sell is None:
            sell = buy
        volume_5m = int(row.get("highPriceVolume") or 0) + int(row.get("lowPriceVolume") or 0)
        return {"buy": int(buy), "sell": int(sell), "volume_5m": volume_5m}
    except Exception as exc:
        logger.warning("get_latest_quote failed item_id=%s: %s", item_id, exc)
        return None


def get_graph_points(item_id: int, range_key: str) -> List[Dict[str, int]]:
    """Return graph points for range keys: 1h, 6h, 24h, 7d."""
    normalized = normalize_graph_range(range_key)
    timestep, max_points = GRAPH_RANGE_SPECS[normalized]

    try:
        resp = requests.get(
            f"{WIKI_BASE}/timeseries",
            params={"timestep": timestep, "id": item_id},
            headers=WIKI_HEADERS,
            timeout=15,
        )
        resp.raise_for_status()
        raw_points = resp.json().get("data", [])
    except Exception as exc:
        logger.warning("get_graph_points failed item_id=%s range=%s: %s", item_id, normalized, exc)
        return []

    points: List[Dict[str, int]] = []
    for pt in raw_points:
        ts = pt.get("timestamp")
        high = pt.get("avgHighPrice")
        low = pt.get("avgLowPrice")
        if ts is None or (high is None and low is None):
            continue
        if high is None:
            high = low
        if low is None:
            low = high
        volume = int(pt.get("highPriceVolume") or 0) + int(pt.get("lowPriceVolume") or 0)
        points.append(
            {
                "ts": int(ts),
                "buy": int(high),
                "sell": int(low),
                "high": int(high),
                "low": int(low),
                "volume": volume,
            }
        )

    if max_points > 0 and len(points) > max_points:
        points = points[-max_points:]
    return points
