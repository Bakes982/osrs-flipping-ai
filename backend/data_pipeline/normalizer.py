"""
OSRS Flipping AI — Price data normalizer and validator.

Converts raw Wiki API dicts into typed PriceSnapshot objects, detects
ghost margins, deduplicates unchanged entries, and validates data quality.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import List, Optional, Tuple

from backend.core.constants import GE_TAX_FREE_BELOW
from backend.core.utils import pct_change, safe_div

logger = logging.getLogger(__name__)

# Tolerance for ghost-margin detection (deviation from 5-min VWAP)
_GHOST_MARGIN_THRESHOLD: float = 0.05  # 5 %


# ---------------------------------------------------------------------------
# Public: raw dict → PriceSnapshot
# ---------------------------------------------------------------------------

def normalize_latest(
    item_id: int,
    raw: dict,
    volume_raw: Optional[dict] = None,
    timestamp: Optional[datetime] = None,
) -> Optional["PriceSnapshot"]:  # noqa: F821 (forward ref to database module)
    """Convert a /latest API entry into a ``PriceSnapshot``.

    Parameters
    ----------
    item_id:
        Integer OSRS item ID.
    raw:
        Single item dict from ``/latest`` (``{"high": ..., "low": ..., ...}``).
    volume_raw:
        Optional matching entry from ``/5m`` for volume data.
    timestamp:
        Snapshot timestamp; defaults to ``datetime.utcnow()``.

    Returns
    -------
    PriceSnapshot, or ``None`` if both ``high`` and ``low`` are missing.
    """
    # Import here to avoid circular dependency at module load time.
    from backend.database import PriceSnapshot

    high = raw.get("high")
    low = raw.get("low")

    if not high and not low:
        return None

    now = timestamp or datetime.utcnow()

    buy_vol: int = 0
    sell_vol: int = 0
    avg_buy: Optional[int] = None
    avg_sell: Optional[int] = None

    if volume_raw:
        buy_vol = volume_raw.get("highPriceVolume") or 0
        sell_vol = volume_raw.get("lowPriceVolume") or 0
        avg_buy = volume_raw.get("avgHighPrice")
        avg_sell = volume_raw.get("avgLowPrice")

    return PriceSnapshot(
        item_id=item_id,
        timestamp=now,
        instant_buy=int(high) if high else None,
        instant_sell=int(low) if low else None,
        buy_time=raw.get("highTime"),
        sell_time=raw.get("lowTime"),
        avg_buy=int(avg_buy) if avg_buy else None,
        avg_sell=int(avg_sell) if avg_sell else None,
        buy_volume=buy_vol,
        sell_volume=sell_vol,
    )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_snapshot(snap: "PriceSnapshot") -> Tuple[bool, List[str]]:  # noqa: F821
    """Validate a ``PriceSnapshot`` for obvious data quality issues.

    Returns
    -------
    (valid, warnings)
        ``valid`` is ``False`` if the snapshot should be discarded entirely.
        ``warnings`` lists non-fatal issues.
    """
    issues: List[str] = []

    if snap.instant_buy is not None and snap.instant_buy <= 0:
        return False, ["instant_buy must be positive"]
    if snap.instant_sell is not None and snap.instant_sell <= 0:
        return False, ["instant_sell must be positive"]

    if snap.instant_buy and snap.instant_sell:
        if snap.instant_sell > snap.instant_buy:
            issues.append("inverted spread (sell > buy)")
        spread_pct = pct_change(snap.instant_sell, snap.instant_buy) * 100
        if spread_pct > 50:
            issues.append(f"extreme spread {spread_pct:.0f}% — likely bad data")

    if snap.buy_volume < 0 or snap.sell_volume < 0:
        return False, ["volume must be non-negative"]

    return True, issues


# ---------------------------------------------------------------------------
# Ghost margin detection
# ---------------------------------------------------------------------------

def detect_ghost_price(
    instant: int,
    vwap_5m: float,
    threshold: float = _GHOST_MARGIN_THRESHOLD,
) -> bool:
    """Return ``True`` if ``instant`` deviates from ``vwap_5m`` by more than
    ``threshold`` (default 5 %), indicating a one-off fat-finger trade.
    """
    if vwap_5m <= 0:
        return False
    dev = abs(pct_change(vwap_5m, instant))
    return dev > threshold


def correct_ghost_prices(
    instant_buy: int,
    instant_sell: int,
    snapshots: List["PriceSnapshot"],  # noqa: F821
    threshold: float = _GHOST_MARGIN_THRESHOLD,
) -> Tuple[int, int, bool]:
    """Validate instant prices against the recent 5-min VWAP.

    If either price deviates beyond ``threshold``, replace it with the VWAP.

    Returns
    -------
    (corrected_buy, corrected_sell, was_ghost)
    """
    from backend.core.utils import vwap as _vwap
    from datetime import timedelta

    cutoff = datetime.utcnow() - timedelta(minutes=5)
    recent = [s for s in snapshots if s.timestamp >= cutoff]

    def _vwap_for(use_buy: bool) -> Optional[float]:
        prices = []
        vols = []
        for s in recent:
            p = s.instant_buy if use_buy else s.instant_sell
            v = s.buy_volume if use_buy else s.sell_volume
            if p and p > 0:
                prices.append(p)
                vols.append(max(v or 0, 1))
        return _vwap(prices, vols) if prices else None

    was_ghost = False
    corrected_buy = instant_buy
    corrected_sell = instant_sell

    vwap_buy = _vwap_for(use_buy=True)
    if vwap_buy and detect_ghost_price(instant_buy, vwap_buy, threshold):
        corrected_buy = int(vwap_buy)
        was_ghost = True

    vwap_sell = _vwap_for(use_buy=False)
    if vwap_sell and detect_ghost_price(instant_sell, vwap_sell, threshold):
        corrected_sell = int(vwap_sell)
        was_ghost = True

    return corrected_buy, corrected_sell, was_ghost


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

def is_duplicate_snapshot(
    new_snap: "PriceSnapshot",  # noqa: F821
    prev_snap: Optional["PriceSnapshot"],  # noqa: F821
    price_tolerance: float = 0.0,
) -> bool:
    """Return ``True`` if ``new_snap`` carries no new information vs ``prev_snap``.

    Two snapshots are considered duplicates when their prices are identical
    (or within ``price_tolerance`` fraction of each other) and both volumes
    are unchanged.
    """
    if prev_snap is None:
        return False
    if new_snap.instant_buy != prev_snap.instant_buy:
        if price_tolerance > 0 and new_snap.instant_buy and prev_snap.instant_buy:
            if abs(pct_change(prev_snap.instant_buy, new_snap.instant_buy)) > price_tolerance:
                return False
        else:
            return False
    if new_snap.instant_sell != prev_snap.instant_sell:
        return False
    if new_snap.buy_volume != prev_snap.buy_volume:
        return False
    if new_snap.sell_volume != prev_snap.sell_volume:
        return False
    return True
