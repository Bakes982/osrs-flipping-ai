"""
backend.analytics.gp_per_hour — Expected profit and risk-adjusted GP/hour.

All public functions receive plain numeric arguments so they are trivially
unit-testable without any database or external dependency.
"""

from __future__ import annotations

from backend.core.constants import GE_TAX_RATE, GE_TAX_CAP
from backend.domain.enums import RiskProfile


# ---------------------------------------------------------------------------
# GP/hour calculations
# ---------------------------------------------------------------------------

def raw_gp_per_hour(
    net_profit_gp: int,
    estimated_hold_minutes: float,
    buy_time_minutes: float = 5.0,
) -> float:
    """
    Estimate GP/hour for a flip assuming one round-trip.

    ``estimated_hold_minutes`` is the median sell-side fill time.
    ``buy_time_minutes`` is added as the buy-side latency.

    Returns 0 if total time is non-positive.
    """
    total_minutes = estimated_hold_minutes + buy_time_minutes
    if total_minutes <= 0 or net_profit_gp <= 0:
        return 0.0
    return net_profit_gp * (60.0 / total_minutes)


def risk_adjusted_gph(
    gp_per_hour: float,
    confidence_pct: float,
    risk_score: float,
    profile: RiskProfile = RiskProfile.BALANCED,
) -> float:
    """
    Penalise raw GP/hour by confidence and risk score, scaled by risk profile.

    Formula:
        adj_gph = gph × (confidence/100) × (1 – risk_factor)

    where ``risk_factor`` is shaped by the user's risk profile.

    Args:
        gp_per_hour:    Raw GP/hour (un-adjusted).
        confidence_pct: Confidence in [0, 100].
        risk_score:     Risk in [0, 10] (0 = safe, 10 = maximum risk).
        profile:        User risk profile (affects how hard risk is penalised).

    Returns float ≥ 0.
    """
    if gp_per_hour <= 0:
        return 0.0

    # Accept either 0..1 or 0..100 confidence inputs for compatibility.
    confidence = confidence_pct / 100.0 if confidence_pct > 1.0 else confidence_pct
    confidence = max(0.0, min(1.0, confidence))
    risk_norm = max(0.0, min(1.0, risk_score / 10.0))

    # Chunk formula baseline: raw_gph * confidence * (1 - 0.5*risk_score_raw)
    # where risk_score_raw is 0..1.
    base = gp_per_hour * confidence * (1.0 - 0.5 * risk_norm)

    # Keep mild profile shaping without violating the baseline behavior.
    profile_scale = {
        RiskProfile.CONSERVATIVE: 0.92,
        RiskProfile.BALANCED: 1.00,
        RiskProfile.AGGRESSIVE: 1.08,
    }.get(profile, 1.0)
    return max(0.0, base * profile_scale)


def expected_profit(
    net_profit_gp: int,
    confidence_pct: float,
    qty: int = 1,
) -> int:
    """
    Expected value of a flip: net_profit × confidence × qty.

    Returns integer GP.
    """
    conf = max(0.0, min(100.0, confidence_pct)) / 100.0
    return int(net_profit_gp * conf * max(1, qty))


def apply_user_calibration(
    base_gph: float,
    profit_multiplier: float,
    hold_multiplier: float,
) -> float:
    """
    Personalise a raw GP/hour estimate using the user's historic calibration.

    Args:
        base_gph:          Raw risk-adjusted GP/hour from the model.
        profit_multiplier: median(realized_profit / expected_profit) for this user.
                           1.0 = model is well-calibrated; <1.0 = model over-estimates.
        hold_multiplier:   median(realized_hold / estimated_hold) for this user.
                           1.0 = model is well-calibrated; >1.0 = user's holds run longer.

    Returns personalised GP/hour estimate.
    """
    if base_gph <= 0:
        return 0.0

    # Profit multiplier directly scales the numerator.
    # Hold multiplier > 1 means the denominator grows → GPH shrinks.
    hold_adj = max(0.1, hold_multiplier)   # clamp: never divide by ~0
    return base_gph * profit_multiplier / hold_adj


def compute_fill_time_minutes(
    volume_5m: int,
    quantity: int,
    buy_price: int,
) -> float:
    """
    Estimate order fill time in minutes based on volume and order size.

    Heuristic: assumes the item trades at the 5-min clip rate and our
    order competes with roughly half the market volume.

    Returns a float in the range [1, 1440] (min 1 minute, max 24 hours).
    """
    if volume_5m <= 0 or quantity <= 0:
        return 60.0   # default: 1 hour if no data

    # Effective competition: we see ~50% of volume as available
    effective_vol_per_minute = volume_5m / 5.0 * 0.5

    # Price-bracket adjustment: expensive items have thinner real liquidity
    if buy_price >= 10_000_000:
        effective_vol_per_minute *= 0.2
    elif buy_price >= 1_000_000:
        effective_vol_per_minute *= 0.5
    elif buy_price >= 100_000:
        effective_vol_per_minute *= 0.8

    if effective_vol_per_minute <= 0:
        return 120.0

    minutes = quantity / effective_vol_per_minute
    return max(1.0, min(1440.0, minutes))
