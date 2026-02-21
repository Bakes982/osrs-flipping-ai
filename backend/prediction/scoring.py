"""
OSRS Flipping AI — Canonical flip metric calculator.

``calculate_flip_metrics`` is the single authoritative function for every
scoring, ROI, volatility, and risk calculation in the system.  Every other
module that needs these values must call this function rather than implementing
its own logic.

Covers all Phase-2 metrics:
  • Margin after GE tax          • ROI %
  • 1-hour volatility            • 24-hour volatility
  • Volume delta                 • Moving-average signal
  • Spread compression speed     • Fill probability estimate
  • Risk score                   • Confidence score
  • Estimated hold time          • GP/hour potential
  • Weighted composite score
"""

from __future__ import annotations

import logging
import math
import statistics
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

from backend.core.constants import (
    GE_TAX_RATE, GE_TAX_CAP,
    PRICE_MULTIPLIER_OPTIMAL, PRICE_MULTIPLIER_HIGH,
    PRICE_MULTIPLIER_SOLID, PRICE_MULTIPLIER_BULK,
    PRICE_MULTIPLIER_WEAK, PRICE_MULTIPLIER_NEUTRAL,
    PRICE_BRACKET_OPTIMAL_LOW, PRICE_BRACKET_OPTIMAL_HIGH,
    SPREAD_MAX_PCT,
    MARGIN_OPTIMAL_LOW, MARGIN_OPTIMAL_HIGH,
    MARGIN_DECENT_LOW, MARGIN_THIN_LOW, MARGIN_WIDE_HIGH, MARGIN_VERY_WIDE,
    STALE_LIMIT_MINUTES_HIGH_VALUE, STALE_LIMIT_MINUTES_DEFAULT,
    STALE_VOLUME_THRESHOLD_HIGH, STALE_VOLUME_THRESHOLD_DEFAULT,
    WATERFALL_BUCKET_MINUTES, WATERFALL_MIN_BUCKETS,
    WATERFALL_MIN_CONSECUTIVE_DROPS, WATERFALL_TOTAL_DROP_THRESHOLD,
    DEAD_VOLUME_LOOKBACK, DEAD_VOLUME_HISTORICAL_FLOOR, DECLINING_VOLUME_RATIO,
    SCORE_WEIGHT_SPREAD, SCORE_WEIGHT_VOLUME, SCORE_WEIGHT_FRESHNESS,
    SCORE_WEIGHT_TREND, SCORE_WEIGHT_HISTORY, SCORE_WEIGHT_STABILITY,
    SCORE_WEIGHT_ML,
    STABILITY_CV_EXCELLENT, STABILITY_CV_GOOD, STABILITY_CV_FAIR,
    STABILITY_CV_POOR, STABILITY_CV_BAD,
)
from backend.core.utils import (
    ge_tax, safe_div, clamp, pct_change, coefficient_of_variation,
    mean_safe, vwap as _vwap_fn,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def calculate_flip_metrics(item_data: dict) -> dict:
    """Canonical flip metric calculator.

    All scoring, ROI, volatility, and risk calculations in the system pass
    through this function.

    Parameters
    ----------
    item_data : dict
        Required keys:
            item_id   (int)   OSRS item ID
            instant_buy  (int)   Current insta-buy price (what buyers pay)
            instant_sell (int)   Current insta-sell price (what sellers get)
        Optional keys:
            item_name    (str)
            volume_5m    (int)   5-minute combined volume (default 0)
            buy_time     (int)   Unix timestamp of last buy price update
            sell_time    (int)   Unix timestamp of last sell price update
            snapshots    (list)  List[PriceSnapshot] for VWAP/trend analysis
            flip_history (list)  List[FlipHistory] for historical win rate

    Returns
    -------
    dict with the following keys (all Phase-2 metrics):

    Core spread metrics
        spread          (int)
        spread_pct      (float)

    Pricing
        recommended_buy  (int)
        recommended_sell (int)

    Profit / ROI
        gross_profit    (int)
        tax             (int)
        net_profit      (int)
        roi_pct         (float)

    Phase-2 indicators
        gp_per_hour         (float)  Estimated GP/hr at this fill rate & margin
        estimated_hold_time (int)    Minutes to expect the flip to resolve
        fill_probability    (float)  0–1, likelihood of order filling
        spread_compression  (float)  Rate of spread narrowing (pct/min, – = widening)
        volatility_1h       (float)  Coefficient of variation over 1 hour
        volatility_24h      (float)  Coefficient of variation over 24 hours
        volume_delta        (float)  Pct change in volume: recent vs historical
        ma_signal           (float)  –1 to +1 moving-average alignment score

    Trend analysis
        trend           (str)    NEUTRAL|UP|DOWN|STRONG_UP|STRONG_DOWN
        momentum        (float)  GP/min
        bb_position     (float | None)
        vwap_1m         (float | None)
        vwap_5m         (float | None)
        vwap_30m        (float | None)
        vwap_2h         (float | None)

    Historical metrics
        win_rate        (float | None)
        total_flips     (int)
        avg_profit      (float | None)

    Component scores (0–100 each)
        score_spread    (float)
        score_volume    (float)
        score_freshness (float)
        score_trend     (float)
        score_history   (float)
        score_stability (float)
        score_ml        (float)  –1 when ML unavailable

    Composite
        total_score     (float)  Weighted 0–100

    Risk / confidence
        confidence      (float)  0–1 combined confidence
        risk_score      (float)  0–10 (higher = riskier)
        stale_data      (bool)
        anomalous_spread (bool)

    Vetoes
        vetoed          (bool)
        veto_reasons    (list[str])

    Human-readable
        reason          (str)
    """
    # -- Unpack inputs -------------------------------------------------------
    item_id: int = item_data.get("item_id", 0)
    item_name: str = item_data.get("item_name", f"Item {item_id}")
    instant_buy: Optional[int] = item_data.get("instant_buy")
    instant_sell: Optional[int] = item_data.get("instant_sell")
    volume_5m: int = item_data.get("volume_5m", 0)
    buy_time: Optional[int] = item_data.get("buy_time")
    sell_time: Optional[int] = item_data.get("sell_time")
    snapshots: List[Any] = item_data.get("snapshots") or []
    flip_history: List[Any] = item_data.get("flip_history") or []

    # Initialise result with safe defaults
    result: dict = _empty_metrics(item_id, item_name)

    if not instant_buy or not instant_sell or instant_buy <= 0 or instant_sell <= 0:
        result["vetoed"] = True
        result["veto_reasons"].append("Missing or invalid buy/sell price")
        return result

    # -- Basic spread metrics ------------------------------------------------
    spread = instant_buy - instant_sell
    spread_pct = safe_div(spread, instant_sell) * 100
    result["spread"] = spread
    result["spread_pct"] = round(spread_pct, 4)

    # -- Recommended prices (spread-position algorithm) ----------------------
    trend_enum, momentum = _detect_trend(snapshots)
    trend_str = trend_enum
    result["trend"] = trend_str
    result["momentum"] = round(momentum, 2)

    rec_buy, rec_sell = _spread_position(instant_buy, instant_sell, volume_5m, trend_str, snapshots)
    result["recommended_buy"] = rec_buy
    result["recommended_sell"] = rec_sell

    # -- Tax & profit --------------------------------------------------------
    tax = ge_tax(rec_sell)
    gross = rec_sell - rec_buy
    net = gross - tax
    roi = safe_div(net, rec_buy) * 100

    result["gross_profit"] = gross
    result["tax"] = tax
    result["net_profit"] = net
    result["roi_pct"] = round(roi, 4)

    # Realized margin percentage (used by scoring)
    realized_pct = safe_div(net, rec_buy) * 100

    # -- VWAP & Bollinger bands ----------------------------------------------
    result["vwap_1m"] = _calc_vwap(snapshots, 1)
    result["vwap_5m"] = _calc_vwap(snapshots, 5)
    result["vwap_30m"] = _calc_vwap(snapshots, 30)
    result["vwap_2h"] = _calc_vwap(snapshots, 120)
    bb_pos = _bollinger_position(snapshots, instant_buy)
    result["bb_position"] = bb_pos

    # -- Volatility ----------------------------------------------------------
    result["volatility_1h"] = round(_volatility(snapshots, 60), 6)
    result["volatility_24h"] = round(_volatility(snapshots, 1440), 6)

    # -- Volume delta --------------------------------------------------------
    result["volume_delta"] = round(_volume_delta(snapshots), 4)

    # -- Moving-average signal (-1..+1) ----------------------------------------
    result["ma_signal"] = round(_ma_signal(snapshots), 4)

    # -- Spread compression speed (pct/min, negative = spreading) -----------
    result["spread_compression"] = round(_spread_compression(snapshots), 6)

    # -- Fill probability (0–1) -----------------------------------------------
    fill_prob = _fill_probability(volume_5m, spread_pct, instant_buy)
    result["fill_probability"] = round(fill_prob, 4)

    # -- Estimated hold time -------------------------------------------------
    hold_minutes = _estimated_hold_time(volume_5m, spread_pct)
    result["estimated_hold_time"] = hold_minutes

    # -- GP/hour potential ---------------------------------------------------
    if net > 0 and hold_minutes > 0:
        result["gp_per_hour"] = round(net / (hold_minutes / 60), 0)
    else:
        result["gp_per_hour"] = 0.0

    # -- Data quality --------------------------------------------------------
    stale, anomalous, confidence = _check_sanity(
        instant_buy, instant_sell, buy_time, sell_time, volume_5m, flip_history
    )
    result["stale_data"] = stale
    result["anomalous_spread"] = anomalous
    result["confidence"] = round(confidence, 4)

    # -- Historical metrics --------------------------------------------------
    _fill_history_metrics(result, flip_history)

    # -- Hard vetoes ---------------------------------------------------------
    _check_vetoes(result, instant_buy, instant_sell, spread_pct, volume_5m, snapshots, rec_buy, rec_sell)
    if result["vetoed"]:
        return result

    # -- Component scores (0–100 each) ---------------------------------------
    result["score_spread"] = _score_spread(realized_pct, spread_pct, volume_5m)
    result["score_volume"] = _score_volume(volume_5m, instant_buy)
    result["score_freshness"] = _score_freshness(buy_time, sell_time)
    result["score_trend"] = _score_trend(trend_str, bb_pos)
    result["score_history"] = _score_history(flip_history, result)
    result["score_stability"] = _score_stability(snapshots)
    result["score_ml"] = -1.0  # populated externally when ML is available

    # -- Composite weighted score -------------------------------------------
    result["total_score"] = _composite_score(result, instant_buy, instant_sell, confidence)

    # -- Risk score (0–10, higher = riskier) --------------------------------
    result["risk_score"] = _risk_score(result)

    # -- Human-readable summary ---------------------------------------------
    result["reason"] = _build_reason(result, item_name)

    return result


# ---------------------------------------------------------------------------
# Score injector (called by ML layer after prediction is available)
# ---------------------------------------------------------------------------

def apply_ml_score(metrics: dict, ml_score: float) -> dict:
    """Inject the ML score into an already-computed metrics dict and
    recompute the weighted total.  Returns the updated dict in-place.

    ``ml_score`` should be 0–100 (or –1 if ML is unavailable).
    """
    metrics["score_ml"] = ml_score
    instant_buy = metrics.get("recommended_buy") or 0
    instant_sell = metrics.get("recommended_sell") or 0
    confidence = metrics.get("confidence", 1.0)
    metrics["total_score"] = _composite_score(metrics, instant_buy, instant_sell, confidence)
    metrics["reason"] = _build_reason(metrics, metrics.get("item_name", ""))
    return metrics


# ---------------------------------------------------------------------------
# Internals: trend detection
# ---------------------------------------------------------------------------

def _detect_trend(snapshots: list) -> tuple:
    """Returns (trend_str, momentum_gp_per_min)."""
    sma_5 = _sma(snapshots, 5)
    sma_30 = _sma(snapshots, 30)
    sma_2h = _sma(snapshots, 120)

    # Momentum over last 5 minutes
    cutoff = datetime.utcnow() - timedelta(minutes=5)
    recent = [s for s in snapshots if s.timestamp >= cutoff and s.instant_buy]
    momentum = 0.0
    if len(recent) >= 2:
        elapsed = max((recent[-1].timestamp - recent[0].timestamp).total_seconds() / 60, 0.1)
        momentum = (recent[-1].instant_buy - recent[0].instant_buy) / elapsed

    if not sma_5 or not sma_30:
        return "NEUTRAL", momentum

    if sma_2h:
        if sma_5 > sma_30 > sma_2h and momentum > 0:
            return "STRONG_UP", momentum
        if sma_5 < sma_30 < sma_2h and momentum < 0:
            return "STRONG_DOWN", momentum

    if sma_5 > sma_30:
        return "UP", momentum
    if sma_5 < sma_30:
        return "DOWN", momentum
    return "NEUTRAL", momentum


def _sma(snapshots: list, minutes: int, use_buy: bool = True) -> Optional[float]:
    cutoff = datetime.utcnow() - timedelta(minutes=minutes)
    prices = []
    for s in snapshots:
        if s.timestamp < cutoff:
            continue
        p = s.instant_buy if use_buy else s.instant_sell
        if p and p > 0:
            prices.append(p)
    return statistics.mean(prices) if prices else None


def _calc_vwap(snapshots: list, minutes: int, use_buy: bool = True) -> Optional[float]:
    cutoff = datetime.utcnow() - timedelta(minutes=minutes)
    prices, vols = [], []
    for s in snapshots:
        if s.timestamp < cutoff:
            continue
        p = s.instant_buy if use_buy else s.instant_sell
        v = s.buy_volume if use_buy else s.sell_volume
        if p and p > 0:
            prices.append(float(p))
            vols.append(max(v or 0, 1))
    result = _vwap_fn(prices, vols)
    return round(result, 2) if result is not None else None


def _bollinger_position(snapshots: list, instant_buy: int, period_min: int = 120, num_std: float = 2.0) -> Optional[float]:
    cutoff = datetime.utcnow() - timedelta(minutes=period_min)
    prices = [s.instant_buy for s in snapshots if s.timestamp >= cutoff and s.instant_buy]
    if len(prices) < 5:
        return None
    mid = statistics.mean(prices)
    std = statistics.stdev(prices)
    upper = mid + num_std * std
    lower = mid - num_std * std
    if upper <= lower:
        return None
    pos = (instant_buy - lower) / (upper - lower)
    return round(clamp(pos, 0.0, 1.0), 4)


# ---------------------------------------------------------------------------
# Internals: Phase-2 derived metrics
# ---------------------------------------------------------------------------

def _volatility(snapshots: list, minutes: int) -> float:
    """Coefficient of variation of buy prices over a time window."""
    cutoff = datetime.utcnow() - timedelta(minutes=minutes)
    prices = [s.instant_buy for s in snapshots if s.timestamp >= cutoff and s.instant_buy]
    if len(prices) < 2:
        return 0.0
    return coefficient_of_variation(prices)


def _volume_delta(snapshots: list) -> float:
    """Percentage change in volume: recent 3 snapshots vs older snapshots.

    Positive = volume increasing; negative = volume declining.
    """
    if len(snapshots) < 6:
        return 0.0
    recent = snapshots[-3:]
    older = snapshots[:-3]
    recent_vol = sum((s.buy_volume or 0) + (s.sell_volume or 0) for s in recent)
    older_vol = sum((s.buy_volume or 0) + (s.sell_volume or 0) for s in older)
    avg_older_scaled = (older_vol / len(older)) * 3 if older else 0
    return pct_change(avg_older_scaled, recent_vol)


def _ma_signal(snapshots: list) -> float:
    """Moving-average alignment score in [–1, +1].

    +1 = all SMAs strongly aligned upward (strong uptrend)
    –1 = all SMAs strongly aligned downward (strong downtrend)
     0 = no clear direction
    """
    sma_5 = _sma(snapshots, 5)
    sma_30 = _sma(snapshots, 30)
    sma_2h = _sma(snapshots, 120)

    if not sma_5 or not sma_30:
        return 0.0

    score = 0.0
    # 5m vs 30m (weight 0.5)
    if sma_5 > sma_30:
        score += 0.5
    elif sma_5 < sma_30:
        score -= 0.5
    # 30m vs 2h (weight 0.3)
    if sma_2h:
        if sma_30 > sma_2h:
            score += 0.3
        elif sma_30 < sma_2h:
            score -= 0.3
    # 5m vs 2h (weight 0.2) — cross-term
    if sma_2h:
        if sma_5 > sma_2h:
            score += 0.2
        elif sma_5 < sma_2h:
            score -= 0.2

    return clamp(score, -1.0, 1.0)


def _spread_compression(snapshots: list) -> float:
    """Estimate how quickly the spread is narrowing (pct/min).

    Positive = spread narrowing (good for fills);
    Negative = spread widening (price discovery still active).
    """
    if len(snapshots) < 6:
        return 0.0

    def _spread_pct_at(snap) -> float:
        if snap.instant_buy and snap.instant_sell and snap.instant_sell > 0:
            return (snap.instant_buy - snap.instant_sell) / snap.instant_sell
        return 0.0

    old_snap = snapshots[max(0, len(snapshots) - 12)]
    new_snap = snapshots[-1]
    old_pct = _spread_pct_at(old_snap)
    new_pct = _spread_pct_at(new_snap)

    elapsed_min = max(
        (new_snap.timestamp - old_snap.timestamp).total_seconds() / 60, 0.1
    )
    return (old_pct - new_pct) / elapsed_min  # positive = narrowing


def _fill_probability(volume_5m: int, spread_pct: float, price: int) -> float:
    """Estimate probability that a limit order fills within one GE 4-hour window.

    Heuristic: combines volume, spread width, and price bracket.
    Returns a value in [0.05, 0.98].
    """
    if volume_5m <= 0:
        return 0.05

    # Base probability from volume (scaled by price bracket)
    if price >= 50_000_000:
        vol_thresholds = [(3, 0.95), (2, 0.85), (1, 0.70)]
    elif price >= 10_000_000:
        vol_thresholds = [(15, 0.95), (8, 0.85), (4, 0.70), (2, 0.55), (1, 0.40)]
    else:
        vol_thresholds = [(100, 0.98), (50, 0.92), (20, 0.82), (10, 0.70), (5, 0.55), (2, 0.35), (1, 0.20)]

    base = 0.10
    for threshold, prob in vol_thresholds:
        if volume_5m >= threshold:
            base = prob
            break

    # Spread penalty: wide spread → trickier to fill profitably
    if spread_pct > 8:
        base *= 0.60
    elif spread_pct > 4:
        base *= 0.80
    elif spread_pct < 1:
        base *= 0.90  # tight spread = competitive but often liquid

    return clamp(base, 0.05, 0.98)


def _estimated_hold_time(volume_5m: int, spread_pct: float) -> int:
    """Estimate how many minutes until both legs of the flip complete.

    Returns minutes (integer).
    """
    if volume_5m >= 100:
        base = 15
    elif volume_5m >= 50:
        base = 30
    elif volume_5m >= 20:
        base = 60
    elif volume_5m >= 5:
        base = 120
    elif volume_5m >= 1:
        base = 180
    else:
        base = 240  # no volume = very slow

    # Wide spread = item is less competitive, may take longer
    if spread_pct > 6:
        base = int(base * 1.5)
    elif spread_pct > 3:
        base = int(base * 1.2)

    return base


# ---------------------------------------------------------------------------
# Internals: sanity & data quality
# ---------------------------------------------------------------------------

def _check_sanity(
    instant_buy: int,
    instant_sell: int,
    buy_time: Optional[int],
    sell_time: Optional[int],
    volume_5m: int,
    flip_history: list,
) -> tuple:
    """Returns (stale: bool, anomalous: bool, confidence: float)."""
    now = int(time.time())
    stale = False
    anomalous = False
    confidence = 1.0

    # Staleness
    price = instant_buy or 0
    limit_mins = STALE_LIMIT_MINUTES_HIGH_VALUE if price >= 10_000_000 else STALE_LIMIT_MINUTES_DEFAULT
    vol_threshold = STALE_VOLUME_THRESHOLD_HIGH if price >= 10_000_000 else STALE_VOLUME_THRESHOLD_DEFAULT

    if buy_time:
        age_min = (now - buy_time) / 60
        if age_min > limit_mins and volume_5m < vol_threshold:
            stale = True
            confidence *= 0.4
        elif age_min > 30:
            confidence *= 0.7
    if sell_time:
        age_min = (now - sell_time) / 60
        if age_min > limit_mins and volume_5m < vol_threshold:
            stale = True
            confidence *= 0.5

    # Spread anomaly
    if instant_buy and instant_sell and instant_sell > 0:
        current_pct = (instant_buy - instant_sell) / instant_sell * 100
        hist_margins = [f.margin_pct for f in flip_history if getattr(f, "margin_pct", None) and f.margin_pct > 0]
        if hist_margins and len(hist_margins) >= 3:
            import statistics as _stats
            hist_med = _stats.median(hist_margins)
            if current_pct > hist_med * 3:
                anomalous = True
                confidence *= 0.4
        elif current_pct > 5:
            anomalous = True
            confidence *= 0.6

    # Volume confidence
    if volume_5m >= 20:
        pass
    elif volume_5m >= 5:
        confidence *= 0.85
    elif volume_5m >= 1:
        confidence *= 0.6
    else:
        confidence *= 0.3

    return stale, anomalous, clamp(confidence, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Internals: historical
# ---------------------------------------------------------------------------

def _fill_history_metrics(result: dict, flip_history: list) -> None:
    if not flip_history:
        result["win_rate"] = None
        result["total_flips"] = 0
        result["avg_profit"] = None
        return

    wins = [f for f in flip_history if getattr(f, "net_profit", None) and f.net_profit > 0]
    total = len(flip_history)
    profits = [f.net_profit for f in flip_history if getattr(f, "net_profit", None) is not None]

    result["win_rate"] = round(len(wins) / total, 4) if total > 0 else None
    result["total_flips"] = total
    result["avg_profit"] = round(mean_safe(profits), 0) if profits else None


# ---------------------------------------------------------------------------
# Internals: hard vetoes
# ---------------------------------------------------------------------------

def _check_vetoes(
    result: dict,
    instant_buy: int,
    instant_sell: int,
    spread_pct: float,
    volume_5m: int,
    snapshots: list,
    rec_buy: int,
    rec_sell: int,
) -> None:
    reasons = result["veto_reasons"]

    # 1. Unprofitable after tax
    net = rec_sell - rec_buy - ge_tax(rec_sell)
    if net <= 0:
        result["vetoed"] = True
        reasons.append(f"Unprofitable after tax ({net:,} GP)")

    # 2. Inverted or zero spread
    if instant_buy <= instant_sell:
        result["vetoed"] = True
        reasons.append("Inverted spread (buy ≤ sell)")

    # 3. Spread too wide
    if spread_pct > SPREAD_MAX_PCT:
        result["vetoed"] = True
        reasons.append(f"Spread too wide ({spread_pct:.1f}%) — illiquid trap")

    # 4. Zero volume
    if volume_5m == 0:
        result["vetoed"] = True
        reasons.append("Zero 5-min volume — likely illiquid trap")

    # 5. Stale data veto
    if result.get("stale_data") and volume_5m < 2:
        result["vetoed"] = True
        reasons.append("Stale price data with no volume")

    if not snapshots:
        return

    # 6. Waterfall crash
    if _detect_waterfall(snapshots):
        result["vetoed"] = True
        reasons.append("Waterfall crash detected — price dropping >5% in 15 min")

    # 7. Volume velocity trap
    vol_status = _check_volume_liveness(snapshots)
    if vol_status == "DEAD_VOLUME_TRAP":
        result["vetoed"] = True
        reasons.append("Volume velocity trap — trades dried up in last interval")
    elif vol_status == "DECLINING" and volume_5m < 5:
        result["vetoed"] = True
        reasons.append(f"Volume collapsed to {DECLINING_VOLUME_RATIO:.0%} of normal — liquidity freeze")


def _detect_waterfall(snapshots: list) -> bool:
    from collections import defaultdict
    buckets: dict = defaultdict(list)
    for s in snapshots:
        if s.instant_buy and s.instant_buy > 0:
            bucket = s.timestamp.replace(
                minute=(s.timestamp.minute // WATERFALL_BUCKET_MINUTES) * WATERFALL_BUCKET_MINUTES,
                second=0, microsecond=0,
            )
            buckets[bucket].append(s.instant_buy)
    if len(buckets) < WATERFALL_MIN_BUCKETS:
        return False
    sorted_b = sorted(buckets.items())
    avgs = [statistics.mean(v) for _, v in sorted_b]
    if len(avgs) < 4:
        return False
    recent = avgs[-4:]
    changes = [(recent[i] - recent[i - 1]) / recent[i - 1] for i in range(1, len(recent)) if recent[i - 1] > 0]
    return len(changes) >= WATERFALL_MIN_CONSECUTIVE_DROPS and all(c < 0 for c in changes) and sum(changes) < WATERFALL_TOTAL_DROP_THRESHOLD


def _check_volume_liveness(snapshots: list) -> str:
    if len(snapshots) < 6:
        return "HEALTHY"
    recent = snapshots[-DEAD_VOLUME_LOOKBACK:]
    older = snapshots[:-DEAD_VOLUME_LOOKBACK]
    recent_vol = sum((s.buy_volume or 0) + (s.sell_volume or 0) for s in recent)
    older_vol = sum((s.buy_volume or 0) + (s.sell_volume or 0) for s in older)
    avg_older_scaled = (older_vol / len(older)) * DEAD_VOLUME_LOOKBACK if older else 0
    if recent_vol == 0 and avg_older_scaled > DEAD_VOLUME_HISTORICAL_FLOOR:
        return "DEAD_VOLUME_TRAP"
    if avg_older_scaled > 0 and safe_div(recent_vol, avg_older_scaled) < DECLINING_VOLUME_RATIO:
        return "DECLINING"
    return "HEALTHY"


# ---------------------------------------------------------------------------
# Internals: recommended pricing (spread-position algorithm)
# ---------------------------------------------------------------------------

def _spread_position(
    insta_buy: int,
    insta_sell: int,
    volume_5m: int,
    trend: str,
    snapshots: list,
) -> tuple:
    """Returns (recommended_buy, recommended_sell)."""
    spread = insta_buy - insta_sell
    if spread <= 0:
        return insta_sell, insta_buy

    spread_pct = spread / max(insta_sell, 1)

    # Buy fraction base (tighter spread → more aggressive)
    if spread_pct < 0.005:
        buy_base = 0.40
    elif spread_pct < 0.01:
        buy_base = 0.35
    elif spread_pct < 0.02:
        buy_base = 0.30
    elif spread_pct < 0.05:
        buy_base = 0.22
    else:
        buy_base = 0.15

    # Volume scaling
    if volume_5m >= 100:
        buy_base *= 0.75
    elif volume_5m >= 50:
        buy_base *= 0.85
    elif volume_5m >= 20:
        buy_base *= 0.95
    elif volume_5m < 2:
        buy_base *= 0.80

    _buy_nudge = {"STRONG_UP": 0.05, "UP": 0.02, "NEUTRAL": 0.0, "DOWN": -0.03, "STRONG_DOWN": -0.05}
    _sell_nudge = {"STRONG_UP": -0.03, "UP": -0.02, "NEUTRAL": 0.0, "DOWN": 0.03, "STRONG_DOWN": 0.05}

    buy_fraction = clamp(buy_base + _buy_nudge.get(trend, 0.0), 0.03, 0.50)
    sell_fraction = clamp(buy_base + _sell_nudge.get(trend, 0.0), 0.03, 0.50)

    buy_price = int(insta_sell + spread * buy_fraction)
    sell_price = int(insta_buy - spread * sell_fraction)

    # Hard cap: offset ≤ 0.6% of price for items ≥ 1M
    if insta_sell >= 1_000_000:
        cap = int(insta_sell * 0.006)
        buy_price = min(buy_price, insta_sell + cap)
    if insta_buy >= 1_000_000:
        cap = int(insta_buy * 0.006)
        sell_price = max(sell_price, insta_buy - cap)

    # Sanity: buy < sell
    buy_price = max(insta_sell, buy_price)
    sell_price = min(insta_buy, sell_price)
    if buy_price >= sell_price:
        mid = (insta_buy + insta_sell) // 2
        buy_price = mid - 1
        sell_price = mid + 1

    return buy_price, sell_price


# ---------------------------------------------------------------------------
# Internals: component scores (0–100)
# ---------------------------------------------------------------------------

def _score_spread(realized_pct: float, raw_pct: float, volume_5m: int) -> float:
    if realized_pct <= 0:
        return 0.0

    if MARGIN_OPTIMAL_LOW <= realized_pct <= MARGIN_OPTIMAL_HIGH:
        score = 100.0
    elif MARGIN_DECENT_LOW <= realized_pct < MARGIN_OPTIMAL_LOW:
        score = 90.0
    elif MARGIN_THIN_LOW <= realized_pct < MARGIN_DECENT_LOW:
        score = 65.0
    elif MARGIN_OPTIMAL_HIGH < realized_pct <= MARGIN_WIDE_HIGH:
        score = 75.0
    elif MARGIN_WIDE_HIGH < realized_pct <= MARGIN_VERY_WIDE:
        score = 50.0
    elif realized_pct < MARGIN_THIN_LOW:
        score = 30.0
    else:
        score = 15.0

    if realized_pct < 0.5 and volume_5m < 10:
        score = max(10.0, score - 20.0)
    elif realized_pct >= 1.0 and volume_5m >= 30:
        score = min(100.0, score + 5.0)

    if raw_pct > 8:
        score = min(score, 40.0)

    return score


def _score_volume(volume_5m: int, price: int) -> float:
    if price >= 50_000_000:
        table = [(10, 100), (5, 90), (3, 80), (2, 70), (1, 55)]
    elif price >= 10_000_000:
        table = [(30, 100), (15, 90), (8, 75), (4, 60), (2, 40), (1, 25)]
    else:
        table = [(100, 100), (50, 90), (20, 75), (10, 60), (5, 40), (2, 20), (1, 10)]
    for threshold, score in table:
        if volume_5m >= threshold:
            return float(score)
    return 0.0


def _score_freshness(buy_time: Optional[int], sell_time: Optional[int]) -> float:
    if not buy_time and not sell_time:
        return 20.0
    now = int(time.time())
    ages = []
    if buy_time:
        ages.append((now - buy_time) / 60)
    if sell_time:
        ages.append((now - sell_time) / 60)
    max_age = max(ages)
    if max_age < 2:
        return 100.0
    elif max_age < 5:
        return 90.0
    elif max_age < 10:
        return 75.0
    elif max_age < 15:
        return 60.0
    elif max_age < 30:
        return 40.0
    elif max_age < 60:
        return 20.0
    return 5.0


def _score_trend(trend: str, bb_pos: Optional[float]) -> float:
    base = {"NEUTRAL": 90, "UP": 75, "DOWN": 60, "STRONG_UP": 50, "STRONG_DOWN": 25}.get(trend, 50)
    if bb_pos is not None:
        if 0.3 <= bb_pos <= 0.7:
            base = min(100, base + 10)
        elif bb_pos < 0.1 or bb_pos > 0.9:
            base = max(0, base - 15)
    return float(base)


def _score_history(flip_history: list, result: dict) -> float:
    if not flip_history or len(flip_history) < 3:
        return 50.0
    win_rate = result.get("win_rate") or 0.0
    if win_rate >= 0.9:
        return 100.0
    elif win_rate >= 0.8:
        return 85.0
    elif win_rate >= 0.7:
        return 70.0
    elif win_rate >= 0.6:
        return 55.0
    elif win_rate >= 0.5:
        return 40.0
    return 15.0


def _score_stability(snapshots: list) -> float:
    recent = snapshots[-30:]
    prices = [s.instant_buy for s in recent if s.instant_buy and s.instant_buy > 0]
    if len(prices) < 3:
        return 50.0
    cv = coefficient_of_variation(prices)
    if cv < STABILITY_CV_EXCELLENT:
        return 100.0
    elif cv < STABILITY_CV_GOOD:
        return 85.0
    elif cv < STABILITY_CV_FAIR:
        return 70.0
    elif cv < STABILITY_CV_POOR:
        return 50.0
    elif cv < STABILITY_CV_BAD:
        return 30.0
    return 10.0


# ---------------------------------------------------------------------------
# Internals: composite score & risk
# ---------------------------------------------------------------------------

def _composite_score(result: dict, instant_buy: int, instant_sell: int, confidence: float) -> float:
    ml_score = result.get("score_ml", -1.0)
    has_ml = ml_score >= 0

    w = {
        "spread": SCORE_WEIGHT_SPREAD,
        "volume": SCORE_WEIGHT_VOLUME,
        "freshness": SCORE_WEIGHT_FRESHNESS,
        "trend": SCORE_WEIGHT_TREND,
        "history": SCORE_WEIGHT_HISTORY,
        "stability": SCORE_WEIGHT_STABILITY,
    }

    if has_ml:
        score = (
            result.get("score_spread", 0) * w["spread"]
            + result.get("score_volume", 0) * w["volume"]
            + result.get("score_freshness", 0) * w["freshness"]
            + result.get("score_trend", 0) * w["trend"]
            + result.get("score_history", 0) * w["history"]
            + result.get("score_stability", 0) * w["stability"]
            + ml_score * SCORE_WEIGHT_ML
        )
    else:
        total_w = sum(w.values())
        score = (
            result.get("score_spread", 0) * w["spread"] / total_w
            + result.get("score_volume", 0) * w["volume"] / total_w
            + result.get("score_freshness", 0) * w["freshness"] / total_w
            + result.get("score_trend", 0) * w["trend"] / total_w
            + result.get("score_history", 0) * w["history"] / total_w
            + result.get("score_stability", 0) * w["stability"] / total_w
        )

    # Confidence multiplier (floor at 0.3 so bad data can still score)
    score *= max(0.3, confidence)

    # Price-bracket multiplier
    mid = (instant_buy + instant_sell) // 2 if instant_buy and instant_sell else 0
    if mid > 0:
        if PRICE_BRACKET_OPTIMAL_LOW <= mid <= PRICE_BRACKET_OPTIMAL_HIGH:
            score *= PRICE_MULTIPLIER_OPTIMAL
        elif mid > PRICE_BRACKET_OPTIMAL_HIGH:
            score *= PRICE_MULTIPLIER_HIGH
        elif mid >= 1_000_000:
            score *= PRICE_MULTIPLIER_SOLID
        elif mid < 10_000:
            score *= PRICE_MULTIPLIER_BULK
        elif mid >= 100_000:
            score *= PRICE_MULTIPLIER_WEAK
        else:
            score *= PRICE_MULTIPLIER_NEUTRAL

    return round(clamp(score, 0.0, 100.0), 2)


def _risk_score(result: dict) -> float:
    """Compute a risk score 0–10 (higher = riskier)."""
    risk = 5.0  # baseline

    # Penalise downtrend
    trend = result.get("trend", "NEUTRAL")
    risk += {"STRONG_DOWN": 2.5, "DOWN": 1.5, "NEUTRAL": 0.0, "UP": -0.5, "STRONG_UP": -1.0}.get(trend, 0.0)

    # Penalise high volatility
    vol = result.get("volatility_1h", 0.0)
    if vol > 0.1:
        risk += 2.0
    elif vol > 0.05:
        risk += 1.0
    elif vol > 0.02:
        risk += 0.5

    # Penalise low confidence
    conf = result.get("confidence", 1.0)
    risk += (1.0 - conf) * 2.0

    # Penalise stale data
    if result.get("stale_data"):
        risk += 1.0

    # Reward good history
    win_rate = result.get("win_rate") or 0.5
    if win_rate >= 0.8:
        risk -= 1.0
    elif win_rate < 0.5:
        risk += 1.0

    return round(clamp(risk, 0.0, 10.0), 2)


# ---------------------------------------------------------------------------
# Internals: human-readable summary
# ---------------------------------------------------------------------------

def _build_reason(result: dict, item_name: str) -> str:
    score = result.get("total_score", 0)
    if score >= 70:
        label = "STRONG FLIP"
    elif score >= 55:
        label = "GOOD FLIP"
    elif score >= 45:
        label = "MARGINAL"
    else:
        label = "WEAK"

    parts = [label, f"Score: {score:.0f}"]

    net = result.get("net_profit", 0)
    roi = result.get("roi_pct", 0)
    if net > 0:
        from backend.core.utils import format_gp
        parts.append(f"+{format_gp(net)} ({roi:.1f}%)")

    gph = result.get("gp_per_hour", 0)
    if gph > 0:
        from backend.core.utils import format_gp
        parts.append(f"{format_gp(gph)}/hr")

    trend = result.get("trend", "NEUTRAL")
    parts.append(f"Trend: {trend}")

    vol_5m = result.get("score_volume", 0)
    if vol_5m >= 75:
        parts.append("High liquidity")
    elif vol_5m >= 50:
        parts.append("Decent liquidity")
    else:
        parts.append("Low liquidity")

    win_rate = result.get("win_rate")
    total_flips = result.get("total_flips", 0)
    if win_rate is not None and total_flips >= 3:
        parts.append(f"History: {win_rate * 100:.0f}% WR ({total_flips} flips)")

    hold = result.get("estimated_hold_time", 0)
    if hold:
        parts.append(f"~{hold}min hold")

    return " | ".join(parts)


# ---------------------------------------------------------------------------
# Internal: empty result template
# ---------------------------------------------------------------------------

def _empty_metrics(item_id: int, item_name: str) -> dict:
    return {
        "item_id": item_id,
        "item_name": item_name,
        # Spread
        "spread": 0,
        "spread_pct": 0.0,
        # Pricing
        "recommended_buy": 0,
        "recommended_sell": 0,
        # Profit
        "gross_profit": 0,
        "tax": 0,
        "net_profit": 0,
        "roi_pct": 0.0,
        # Phase-2
        "gp_per_hour": 0.0,
        "estimated_hold_time": 0,
        "fill_probability": 0.0,
        "spread_compression": 0.0,
        "volatility_1h": 0.0,
        "volatility_24h": 0.0,
        "volume_delta": 0.0,
        "ma_signal": 0.0,
        # Trend
        "trend": "NEUTRAL",
        "momentum": 0.0,
        "bb_position": None,
        "vwap_1m": None,
        "vwap_5m": None,
        "vwap_30m": None,
        "vwap_2h": None,
        # History
        "win_rate": None,
        "total_flips": 0,
        "avg_profit": None,
        # Scores
        "score_spread": 0.0,
        "score_volume": 0.0,
        "score_freshness": 0.0,
        "score_trend": 0.0,
        "score_history": 0.0,
        "score_stability": 0.0,
        "score_ml": -1.0,
        "total_score": 0.0,
        # Risk
        "confidence": 1.0,
        "risk_score": 5.0,
        "stale_data": False,
        "anomalous_spread": False,
        # Vetoes
        "vetoed": False,
        "veto_reasons": [],
        "reason": "",
    }
