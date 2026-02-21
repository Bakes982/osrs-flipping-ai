"""
backend.analytics.scoring — Risk-profile-aware item scorer.

This module wraps the canonical ``calculate_flip_metrics`` engine and
applies:

1. Risk-profile weight overrides (conservative / balanced / aggressive).
2. Personalisation boosts (item/category affinity, calibration multipliers).
3. Final re-normalised score in [0, 100].

Usage::

    from backend.analytics.scoring import score_item
    from backend.domain.models import UserContext

    ctx = UserContext(risk_profile=RiskProfile.CONSERVATIVE)
    metrics = score_item(item_data, ctx)   # returns ItemMetrics
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence

from backend.analytics.gp_per_hour import risk_adjusted_gph, raw_gp_per_hour
from backend.analytics.risk import classify_risk
from backend.core.constants import GE_TAX_RATE, GE_TAX_CAP
from backend.core.utils import clamp
from backend.domain.enums import RiskProfile, TrendDirection, VolumeRating
from backend.domain.models import ItemMetrics, UserContext

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Default component weights (balanced profile)
# ---------------------------------------------------------------------------

_DEFAULT_WEIGHTS: Dict[str, float] = {
    "margin":          25.0,
    "volume":          25.0,
    "freshness":       12.0,
    "trend":           10.0,
    "history":         10.0,
    "stability":        8.0,
    "fill_probability": 5.0,
    "ml":               5.0,
}


def _effective_weights(profile: RiskProfile) -> Dict[str, float]:
    """Apply profile-specific multipliers to default weights, then renormalise."""
    overrides = profile.score_weight_overrides
    weights = {}
    for k, v in _DEFAULT_WEIGHTS.items():
        mult = overrides.get(k, 1.0)
        # Negative override keys are penalties — we take abs and later subtract
        weights[k] = abs(v * mult)
    # Renormalise so they still sum to 100
    total = sum(weights.values()) or 1.0
    return {k: v / total * 100.0 for k, v in weights.items()}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def score_item(
    item_data: Dict[str, Any],
    user_ctx: Optional[UserContext] = None,
) -> ItemMetrics:
    """
    Score a single item and return an ``ItemMetrics`` object.

    ``item_data`` is the same dict accepted by ``calculate_flip_metrics``:
        Required: item_id, instant_buy, instant_sell
        Optional: item_name, volume_5m, snapshots, flip_history, ...

    Args:
        item_data: Raw item data dict.
        user_ctx:  User context with risk profile and personalisation data.
                   If None, uses anonymous (balanced) context.

    Returns:
        ItemMetrics populated with all scores and flags.
    """
    if user_ctx is None:
        user_ctx = UserContext.anonymous()

    # --- Step 1: run canonical engine -----------------------------------------
    from backend.prediction.scoring import calculate_flip_metrics
    raw = calculate_flip_metrics(item_data)

    # --- Step 2: build base ItemMetrics from raw dict -------------------------
    m = _raw_to_metrics(raw, item_data)

    if m.vetoed:
        return m

    # --- Step 3: apply risk-profile weight overrides -------------------------
    weights = _effective_weights(user_ctx.risk_profile)
    profile_score = _weighted_score(m, weights)
    m.final_score = round(profile_score, 2)
    m.total_score  = m.final_score

    # --- Step 4: apply personalization ----------------------------------------
    m = _apply_personalization(m, user_ctx)

    # --- Step 5: recompute risk-adjusted GP/hour with user calibration --------
    from backend.analytics.gp_per_hour import apply_user_calibration
    base_radgph = risk_adjusted_gph(
        m.gp_per_hour, m.confidence_pct, m.risk_score, user_ctx.risk_profile
    )
    m.risk_adjusted_gp_per_hour = apply_user_calibration(
        base_radgph, user_ctx.profit_multiplier, user_ctx.hold_multiplier
    )

    return m


def score_batch(
    items: Sequence[Dict[str, Any]],
    user_ctx: Optional[UserContext] = None,
) -> List[ItemMetrics]:
    """Score multiple items, skip failures, return sorted by final_score desc."""
    results = []
    for item_data in items:
        try:
            results.append(score_item(item_data, user_ctx))
        except Exception as exc:
            logger.debug("score_batch: skipped item %s: %s", item_data.get("item_id"), exc)
    results.sort(key=lambda m: m.final_score, reverse=True)
    return results


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _raw_to_metrics(raw: Dict[str, Any], item_data: Dict[str, Any]) -> ItemMetrics:
    """Map the flat dict from calculate_flip_metrics to an ItemMetrics."""
    from backend.analytics.risk import classify_risk, historical_volatility

    buy   = raw.get("recommended_buy")  or item_data.get("instant_buy",  0) or 0
    sell  = raw.get("recommended_sell") or item_data.get("instant_sell", 0) or 0
    conf = raw.get("confidence", 0.5)
    if conf is None:
        conf = 0.5
    conf_pct = conf * 100.0 if conf <= 1.0 else float(conf)
    conf_pct = clamp(conf_pct, 0.0, 100.0)

    risk_s = raw.get("risk_score", 5.0)
    if risk_s is None:
        risk_s = 5.0

    vol_5m = item_data.get("volume_5m", 0) or 0

    # Volatility from snapshots
    snaps = item_data.get("snapshots", [])
    sell_prices = [
        float(getattr(s, "instant_sell", None) or (s.get("sell") or s.get("instant_sell") or 0))
        for s in snaps
        if (getattr(s, "instant_sell", None) or s.get("sell") or s.get("instant_sell") or 0) > 0
    ] if snaps else []

    vol_1h  = raw.get("volatility_1h",  0.0) or historical_volatility(sell_prices[-12:])
    vol_24h = raw.get("volatility_24h", 0.0) or historical_volatility(sell_prices)

    trend_val  = raw.get("trend", "NEUTRAL")
    try:
        trend = TrendDirection(trend_val)
    except ValueError:
        trend = TrendDirection.NEUTRAL

    raw_gph = raw.get("gp_per_hour", 0.0) or 0.0
    net_p   = raw.get("net_profit", 0) or 0

    # Volume rating
    sv = raw.get("score_volume", 0) or 0
    if sv >= 75:
        vr = VolumeRating.HIGH
    elif sv >= 40:
        vr = VolumeRating.MEDIUM
    else:
        vr = VolumeRating.LOW

    risk_level = classify_risk(risk_s * 10.0 if risk_s <= 1.0 else risk_s, vol_1h, vol_5m)
    veto_reason = raw.get("veto_reason", "")
    if not veto_reason:
        reasons = raw.get("veto_reasons") or []
        if reasons:
            veto_reason = "; ".join(str(r) for r in reasons)

    m = ItemMetrics(
        item_id=raw.get("item_id", item_data.get("item_id", 0)),
        item_name=raw.get("item_name", item_data.get("item_name", "")),
        buy=buy,
        sell=sell,
        margin_after_tax=net_p,
        roi_pct=raw.get("roi_pct", 0.0) or 0.0,
        volatility_1h=vol_1h,
        volatility_24h=vol_24h,
        volume_score=raw.get("score_volume", 0.0) or 0.0,
        liquidity_score=(raw.get("liquidity_score", 0.0) or 0.0) * 100.0 if (raw.get("liquidity_score", 0.0) or 0.0) <= 1.0 else (raw.get("liquidity_score", 0.0) or 0.0),
        fill_probability=raw.get("fill_probability", 0.5) or 0.5,
        est_fill_time_minutes=raw.get("estimated_hold_time", 60.0) or 60.0,
        trend_score=raw.get("score_trend", 0.0) or 0.0,
        trend=trend,
        decay_score=raw.get("spread_compression", 0.0) or 0.0,
        risk_level=risk_level,
        risk_score=risk_s,
        confidence_pct=conf_pct,
        expected_profit=int(raw.get("expected_profit", net_p * (conf_pct / 100.0))),
        risk_adjusted_gp_per_hour=0.0,   # filled after profile overrides
        final_score=raw.get("total_score", 0.0) or 0.0,
        vetoed=raw.get("vetoed", False),
        veto_reason=veto_reason,
        score_spread=raw.get("score_spread", 0.0) or 0.0,
        score_volume=raw.get("score_volume", 0.0) or 0.0,
        score_trend=raw.get("score_trend", 0.0) or 0.0,
        score_history=raw.get("score_history", 0.0) or 0.0,
        score_stability=raw.get("score_stability", 0.0) or 0.0,
        score_freshness=raw.get("score_freshness", 0.0) or 0.0,
        score_ml=raw.get("score_ml", 0.0) or 0.0,
        volume_rating=vr,
        estimated_hold_minutes=raw.get("estimated_hold_time", 60.0) or 60.0,
        recommended_buy=buy,
        recommended_sell=sell,
        net_profit=net_p,
        total_score=raw.get("total_score", 0.0) or 0.0,
        gp_per_hour=raw_gph,
        ma_signal=raw.get("ma_signal", 0.0) or 0.0,
        volume_delta=raw.get("volume_delta", 0.0) or 0.0,
        spread_compression=raw.get("spread_compression", 0.0) or 0.0,
    )
    return m


def _weighted_score(m: ItemMetrics, weights: Dict[str, float]) -> float:
    """Compute weighted composite score from ItemMetrics component scores."""
    total = weights.get("margin",   0) * (m.score_spread   / 100)
    total += weights.get("volume",   0) * (m.score_volume   / 100)
    total += weights.get("freshness",0) * (m.score_freshness / 100)
    total += weights.get("trend",    0) * (m.score_trend    / 100)
    total += weights.get("history",  0) * (m.score_history  / 100)
    total += weights.get("stability",0) * (m.score_stability / 100)
    total += weights.get("fill_probability", 0) * m.fill_probability
    total += weights.get("ml",       0) * ((m.score_ml + 10) / 20)  # remap [–10,10]→[0,1]
    return max(0.0, min(100.0, total))


def _apply_personalization(m: ItemMetrics, ctx: UserContext) -> ItemMetrics:
    """Apply item/category affinity boosts and flag personalisation."""
    if ctx.user_id is None:
        return m

    boost = ctx.item_affinity.get(m.item_id, 0.0)
    # Category affinity would require category tag lookup — deferred to Chunk 2
    if boost != 0.0:
        m.final_score = max(0.0, min(100.0, m.final_score + boost))
        m.total_score  = m.final_score
        m.affinity_boost = boost
        m.personalization_applied = True

    return m
