"""
backend.analytics.risk — Risk classification and Kelly-based position sizing.

This module supersedes backend/prediction/risk.py.  The old module is kept
for backward-compatibility; new code should import from here.
"""

from __future__ import annotations

import math
from typing import List

from backend.domain.enums import RiskLevel, RiskProfile


# ---------------------------------------------------------------------------
# Risk classification
# ---------------------------------------------------------------------------

def classify_risk(
    risk_score: float,
    volatility_1h: float,
    volume_5m: int,
    win_rate: float = 0.5,
    profile: RiskProfile = RiskProfile.BALANCED,
) -> RiskLevel:
    """
    Map raw risk signals to a RiskLevel label.

    The thresholds shift with ``profile`` so that the same item can be
    rated MEDIUM for a balanced user but HIGH for a conservative user.

    Args:
        risk_score:    Composite risk score [0, 10] (higher = riskier).
        volatility_1h: Hourly price volatility (coefficient of variation).
        volume_5m:     5-minute traded volume.
        win_rate:      Historical win rate [0, 1].
        profile:       User's risk profile.

    Returns RiskLevel enum value.
    """
    # Profile-aware score thresholds [low, medium, high]
    thresholds = {
        RiskProfile.CONSERVATIVE: (2.5, 5.0, 7.5),
        RiskProfile.BALANCED:     (3.5, 6.0, 8.5),
        RiskProfile.AGGRESSIVE:   (5.0, 7.5, 9.5),
    }.get(profile, (3.5, 6.0, 8.5))

    low_t, med_t, high_t = thresholds

    # Boost effective risk for high volatility and thin volume
    effective = risk_score
    if volatility_1h > 0.05:
        effective += (volatility_1h - 0.05) * 20
    if volume_5m < 50:
        effective += 1.5
    if win_rate < 0.4:
        effective += 1.0

    effective = min(10.0, max(0.0, effective))

    if effective < low_t:
        return RiskLevel.LOW
    if effective < med_t:
        return RiskLevel.MEDIUM
    if effective < high_t:
        return RiskLevel.HIGH
    return RiskLevel.VERY_HIGH


# ---------------------------------------------------------------------------
# Kelly criterion
# ---------------------------------------------------------------------------

def kelly_fraction(
    win_rate: float,
    avg_win_gp: float,
    avg_loss_gp: float,
    half_kelly: bool = True,
) -> float:
    """
    Kelly criterion: optimal fraction of bankroll to stake.

    f* = (win_rate / avg_loss) - (loss_rate / avg_win)

    Returns a fraction in [0, 1].  Uses half-Kelly by default for safety.
    """
    if avg_win_gp <= 0 or avg_loss_gp <= 0:
        return 0.0
    loss_rate = 1.0 - win_rate
    f = (win_rate / avg_loss_gp) - (loss_rate / avg_win_gp)
    f = max(0.0, min(1.0, f))
    return f / 2.0 if half_kelly else f


def stop_loss_pct(
    buy_price: int,
    volume_5m: int,
    risk_score: float,
    profile: RiskProfile = RiskProfile.BALANCED,
) -> float:
    """
    Suggested stop-loss distance as a fraction of buy price.

    Conservative profiles tighten the stop-loss; aggressive loosen it.
    Returns a float in roughly [0.005, 0.15].
    """
    base = 0.02  # 2 % default

    # Wider stop for thin-volume / high-value items
    if volume_5m < 100:
        base += 0.03
    if buy_price >= 1_000_000:
        base += 0.02

    # Risk score influence
    base += (risk_score / 10.0) * 0.05

    # Profile modifier
    modifier = {
        RiskProfile.CONSERVATIVE: 0.7,
        RiskProfile.BALANCED:     1.0,
        RiskProfile.AGGRESSIVE:   1.4,
    }.get(profile, 1.0)

    return max(0.005, min(0.15, base * modifier))


# ---------------------------------------------------------------------------
# Volatility helpers
# ---------------------------------------------------------------------------

def historical_volatility(prices: List[float]) -> float:
    """
    Annualised volatility via log-returns (coefficient of variation proxy).

    Returns 0.0 if fewer than two prices are supplied.
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
    n = len(log_returns)
    mean = sum(log_returns) / n
    variance = sum((r - mean) ** 2 for r in log_returns) / max(1, n - 1)
    return math.sqrt(variance)


def sharpe_ratio(avg_profit: float, profit_stdev: float, risk_free: float = 0.0) -> float:
    """Simple Sharpe ratio.  Returns 0 if stdev is zero."""
    if profit_stdev <= 0:
        return 0.0
    return (avg_profit - risk_free) / profit_stdev


def sortino_ratio(profits: List[float], target: float = 0.0) -> float:
    """Sortino ratio using downside deviation below ``target``."""
    if not profits:
        return 0.0
    avg = sum(profits) / len(profits)
    downside = [min(0.0, p - target) ** 2 for p in profits]
    down_dev = math.sqrt(sum(downside) / len(profits))
    if down_dev <= 0:
        return 0.0
    return (avg - target) / down_dev
