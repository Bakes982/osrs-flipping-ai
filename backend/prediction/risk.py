"""
OSRS Flipping AI — Risk metrics.

Pure-function risk calculations used by both the scoring engine and the
portfolio optimizer.  No DB access; all inputs are passed explicitly.
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence

from backend.core.constants import (
    STOP_LOSS_DEFAULT_PCT, STOP_LOSS_HIGH_VALUE_PCT, STOP_LOSS_LOW_VALUE_PCT,
    MIN_WIN_RATE_FOR_SIZING,
)
from backend.core.utils import clamp, safe_div


# ---------------------------------------------------------------------------
# Kelly Criterion
# ---------------------------------------------------------------------------

def kelly_fraction(
    win_rate: float,
    avg_win: float,
    avg_loss: float,
    buy_price: int = 0,
    sell_price: int = 0,
    stop_loss_pct: float = STOP_LOSS_DEFAULT_PCT,
) -> float:
    """Compute the raw Kelly fraction for a flip.

    Uses historical win/loss statistics when available; falls back to a
    spread-derived estimate.

    Parameters
    ----------
    win_rate:
        Historical probability of a profitable flip (0–1).
    avg_win:
        Average GP profit on winning flips.
    avg_loss:
        Average GP loss on losing flips (pass as a positive number).
    buy_price:
        Buy price per unit (used for the fallback estimate).
    sell_price:
        Expected sell price per unit (used for the fallback estimate).
    stop_loss_pct:
        Stop-loss percentage (used for the fallback estimate).

    Returns
    -------
    float
        Raw Kelly fraction in [0, 1].  Callers should apply half-Kelly.
    """
    p = clamp(win_rate, 0.01, 0.99)
    q = 1.0 - p

    if avg_win > 0 and avg_loss > 0:
        b = avg_win / avg_loss
    elif buy_price > 0 and sell_price > buy_price:
        expected_win = sell_price - buy_price
        expected_loss = buy_price * stop_loss_pct
        b = safe_div(expected_win, max(expected_loss, 1))
    else:
        b = 1.0  # even odds

    raw = (p * b - q) / b if b > 0 else 0.0
    return max(0.0, raw)


# ---------------------------------------------------------------------------
# Stop-loss
# ---------------------------------------------------------------------------

def stop_loss_pct(buy_price: int, volume_5m: int, score: float) -> float:
    """Compute the recommended stop-loss percentage for a flip.

    Wider stop for high-value items (price noise is larger in absolute GP
    terms) and tighter stop for low-conviction trades.
    """
    if buy_price >= 10_000_000:
        base = STOP_LOSS_HIGH_VALUE_PCT
    elif buy_price <= 100_000:
        base = STOP_LOSS_LOW_VALUE_PCT
    else:
        base = STOP_LOSS_DEFAULT_PCT

    # Widen slightly for low-volume (more volatile)
    if volume_5m < 5:
        base *= 1.5
    elif volume_5m > 50:
        base *= 0.8

    # Tighten for low-conviction trades
    if score < 50:
        base *= 0.7

    return round(clamp(base, 0.005, 0.15), 4)


# ---------------------------------------------------------------------------
# Volatility metrics
# ---------------------------------------------------------------------------

def historical_volatility(prices: Sequence[float]) -> float:
    """Return the annualised historical volatility of a price series.

    Uses log returns, scaled to a 24-hour GE trading cycle.
    Returns 0.0 when fewer than 2 prices are provided.
    """
    if len(prices) < 2:
        return 0.0
    log_returns = [
        math.log(prices[i] / prices[i - 1])
        for i in range(1, len(prices))
        if prices[i - 1] > 0 and prices[i] > 0
    ]
    if not log_returns:
        return 0.0
    import statistics
    stdev = statistics.stdev(log_returns) if len(log_returns) > 1 else 0.0
    # Scale to daily: sqrt(144) because OSRS Wiki data is ~10s intervals
    return round(stdev * math.sqrt(144), 6)


def sharpe_ratio(
    avg_profit: float,
    profit_stdev: float,
    risk_free: float = 0.0,
) -> float:
    """GP-based Sharpe ratio for a set of historical flips.

    ``risk_free`` is the "risk-free" return benchmark (usually 0 for GE).
    Returns 0.0 when ``profit_stdev`` is zero.
    """
    if profit_stdev <= 0:
        return 0.0
    return (avg_profit - risk_free) / profit_stdev


def sortino_ratio(
    profits: Sequence[float],
    target: float = 0.0,
) -> float:
    """GP-based Sortino ratio (penalises only downside volatility).

    Returns 0.0 when there are fewer than 2 observations.
    """
    if len(profits) < 2:
        return 0.0
    import statistics
    avg = statistics.mean(profits)
    downside = [min(0.0, p - target) for p in profits]
    downside_var = sum(d ** 2 for d in downside) / len(downside)
    downside_std = math.sqrt(downside_var)
    if downside_std <= 0:
        return 0.0
    return (avg - target) / downside_std


# ---------------------------------------------------------------------------
# Risk classification
# ---------------------------------------------------------------------------

def classify_risk(
    score: float,
    volatility_1h: float,
    volume_5m: int,
    win_rate: Optional[float] = None,
) -> str:
    """Return a risk tier from combined score/volatility/liquidity signals."""
    # Convert the existing "higher is better score" + market quality into
    # a normalized risk index in [0, 1], then tier by fixed thresholds.
    score_component = clamp(1.0 - (score / 100.0), 0.0, 1.0)
    vol_component = clamp(volatility_1h / 0.03, 0.0, 1.0)
    liq_component = clamp(1.0 - (volume_5m / 20.0), 0.0, 1.0)

    idx = 0.40 * vol_component + 0.30 * score_component + 0.30 * liq_component
    if win_rate is not None:
        idx = clamp(idx + (0.55 - clamp(win_rate, 0.0, 1.0)) * 0.15, 0.0, 1.0)

    if idx < 0.33:
        return "LOW"
    if idx <= 0.70:
        return "MEDIUM"
    if idx <= 0.88:
        return "HIGH"
    return "VERY_HIGH"
