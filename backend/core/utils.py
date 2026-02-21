"""
OSRS Flipping AI — Shared utilities.

Pure functions used across the whole backend. No imports from other backend
modules; only the standard library and backend.core.constants are allowed.
"""

from __future__ import annotations

import math
import statistics
from typing import List, Optional, Sequence

from backend.core.constants import GE_TAX_RATE, GE_TAX_CAP, GE_TAX_FREE_BELOW


# ---------------------------------------------------------------------------
# GE tax
# ---------------------------------------------------------------------------

def ge_tax(sell_price: int) -> int:
    """Return the GE tax on a single item sold at ``sell_price``.

    Items below ``GE_TAX_FREE_BELOW`` are not taxed.  All other items are
    taxed at ``GE_TAX_RATE``, capped at ``GE_TAX_CAP``.
    """
    if sell_price < GE_TAX_FREE_BELOW:
        return 0
    return min(int(sell_price * GE_TAX_RATE), GE_TAX_CAP)


def net_profit(buy: int, sell: int) -> int:
    """Return profit after GE tax: ``sell - buy - tax``."""
    return sell - buy - ge_tax(sell)


def roi_pct(buy: int, sell: int) -> float:
    """Return ROI as a percentage.  Returns 0.0 if buy is zero."""
    if buy <= 0:
        return 0.0
    return net_profit(buy, sell) / buy * 100.0


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def format_gp(amount: int | float) -> str:
    """Human-readable GP amount with K/M/B suffix."""
    amount = int(amount)
    if abs(amount) >= 1_000_000_000:
        return f"{amount / 1_000_000_000:.2f}B"
    if abs(amount) >= 1_000_000:
        return f"{amount / 1_000_000:.2f}M"
    if abs(amount) >= 1_000:
        return f"{amount / 1_000:.1f}K"
    return f"{amount:,}"


def format_pct(value: float, decimals: int = 1) -> str:
    """Format a float as a percentage string, e.g. ``1.23 → '1.2%'``."""
    return f"{value:.{decimals}f}%"


# ---------------------------------------------------------------------------
# Math helpers
# ---------------------------------------------------------------------------

def safe_div(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Divide without raising on zero denominator."""
    if denominator == 0:
        return default
    return numerator / denominator


def clamp(value: float, lo: float, hi: float) -> float:
    """Clamp ``value`` to the range [lo, hi]."""
    return max(lo, min(hi, value))


def pct_change(old: float, new: float) -> float:
    """Return ``(new - old) / old``.  Returns 0.0 when ``old`` is zero."""
    return safe_div(new - old, old, 0.0)


# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------

def coefficient_of_variation(values: Sequence[float]) -> float:
    """Coefficient of variation (stdev / mean).

    Returns 0.0 when fewer than 2 values are provided or the mean is zero.
    """
    if len(values) < 2:
        return 0.0
    mean = statistics.mean(values)
    if mean == 0:
        return 0.0
    return statistics.stdev(values) / mean


def mean_safe(values: Sequence[float], default: float = 0.0) -> float:
    """Return mean of ``values``, or ``default`` when the sequence is empty."""
    if not values:
        return default
    return statistics.mean(values)


def vwap(
    prices: Sequence[float],
    volumes: Sequence[float],
) -> Optional[float]:
    """Volume-Weighted Average Price.

    Falls back to a simple mean when all volumes are zero.
    Returns ``None`` when ``prices`` is empty.
    """
    if not prices:
        return None
    vols = [max(v, 1) for v in volumes]  # treat 0-volume as 1 to avoid div/0
    total = sum(vols)
    return sum(p * v for p, v in zip(prices, vols)) / total


# ---------------------------------------------------------------------------
# Price-bracket helpers
# ---------------------------------------------------------------------------

def price_bracket_label(price: int) -> str:
    """Return a human-readable label for a price bracket."""
    if price >= 1_000_000_000:
        return "billion"
    if price >= 50_000_000:
        return "high_value"
    if price >= 10_000_000:
        return "optimal"
    if price >= 1_000_000:
        return "solid"
    if price >= 100_000:
        return "weak"    # historically worst bracket
    if price >= 10_000:
        return "neutral"
    return "bulk"
