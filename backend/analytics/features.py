"""
backend.analytics.features — Feature computation for the scoring engine.

Converts a list of PricePoints (or compatible dicts) into a flat feature
dict consumed by ``backend.analytics.scoring.score_item``.

All functions are pure (no I/O) so they are easy to unit-test.
"""

from __future__ import annotations

import math
import statistics
from typing import Any, Dict, List, Optional, Sequence

from backend.domain.enums import TrendDirection


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _prices(snapshots: Sequence[Any], field: str = "sell") -> List[float]:
    """Extract a list of float prices from snapshot dicts or dataclass objects."""
    result = []
    for s in snapshots:
        val = s.get(field, 0) if isinstance(s, dict) else getattr(s, field, 0)
        if val and val > 0:
            result.append(float(val))
    return result


def _vols(snapshots: Sequence[Any]) -> List[int]:
    """Extract combined volume from snapshots."""
    result = []
    for s in snapshots:
        if isinstance(s, dict):
            vol = (s.get("buy_vol") or s.get("buy_volume") or 0) + \
                  (s.get("sell_vol") or s.get("sell_volume") or 0)
        else:
            vol = (getattr(s, "buy_vol", 0) or 0) + (getattr(s, "sell_vol", 0) or 0)
        result.append(int(vol))
    return result


def _safe_cv(values: List[float]) -> float:
    """Coefficient of variation (σ/μ).  Returns 0 if mean is zero."""
    if len(values) < 2:
        return 0.0
    mean = statistics.mean(values)
    if mean == 0:
        return 0.0
    return statistics.stdev(values) / mean


def _log_vol(prices: List[float]) -> float:
    """Annualised log-return volatility."""
    if len(prices) < 2:
        return 0.0
    log_rets = [
        math.log(prices[i] / prices[i - 1])
        for i in range(1, len(prices))
        if prices[i - 1] > 0 and prices[i] > 0
    ]
    if not log_rets:
        return 0.0
    n = len(log_rets)
    mean = sum(log_rets) / n
    var = sum((r - mean) ** 2 for r in log_rets) / max(1, n - 1)
    return math.sqrt(var)


# ---------------------------------------------------------------------------
# Individual feature computers
# ---------------------------------------------------------------------------

def compute_volatility(
    snapshots_1h: Sequence[Any],
    snapshots_24h: Optional[Sequence[Any]] = None,
) -> Dict[str, float]:
    """
    Returns:
        volatility_1h:  log-return vol over the last 60 min of sell prices.
        volatility_24h: log-return vol over the last 24 h.
    """
    prices_1h = _prices(snapshots_1h)
    vol_1h = _log_vol(prices_1h)

    if snapshots_24h is not None:
        prices_24h = _prices(snapshots_24h)
    else:
        prices_24h = prices_1h  # fallback: same window
    vol_24h = _log_vol(prices_24h)

    return {"volatility_1h": vol_1h, "volatility_24h": vol_24h}


def compute_volume_features(
    snapshots: Sequence[Any],
    current_vol_5m: int,
    buy_price: int,
) -> Dict[str, float]:
    """
    Returns:
        volume_score:    [0, 100] — higher = more liquid.
        liquidity_score: [0, 100] — adjusted for item price tier.
        volume_delta:    relative change vs. recent rolling average.
    """
    vols = _vols(snapshots)
    avg_vol = sum(vols) / len(vols) if vols else 0

    # Volume delta
    volume_delta = 0.0
    if avg_vol > 0:
        volume_delta = (current_vol_5m - avg_vol) / avg_vol

    # Raw volume score (diminishing returns)
    raw_vol = min(1.0, current_vol_5m / max(1, _vol_target(buy_price)))
    volume_score = raw_vol * 100

    # Liquidity score: volume_score penalised by volatility of volume
    vol_cv = _safe_cv([float(v) for v in vols]) if vols else 0.0
    stability = max(0.0, 1.0 - vol_cv)
    liquidity_score = volume_score * stability

    return {
        "volume_score":   round(volume_score, 2),
        "liquidity_score": round(liquidity_score, 2),
        "volume_delta":   round(volume_delta, 4),
    }


def _vol_target(buy_price: int) -> int:
    """Volume needed for a 'full' score, based on price bracket."""
    if buy_price >= 10_000_000:
        return 20
    if buy_price >= 1_000_000:
        return 100
    if buy_price >= 100_000:
        return 500
    if buy_price >= 10_000:
        return 2_000
    return 10_000


def compute_trend_features(
    snapshots: Sequence[Any],
) -> Dict[str, Any]:
    """
    Returns:
        trend:        TrendDirection enum value.
        trend_score:  [0, 100] — higher = stronger favourable trend.
        ma_signal:    moving-average cross signal in [–1, 1].
        decay_score:  spread compression speed (higher = spreads shrinking faster).
    """
    sell_prices = _prices(snapshots, "sell")
    buy_prices  = _prices(snapshots, "buy")

    trend = _detect_trend(sell_prices)
    ma_signal = _ma_cross(sell_prices)

    # Trend score: NEUTRAL is best for margin flipping (easy to fill both sides)
    _trend_score_map = {
        TrendDirection.NEUTRAL:     90,
        TrendDirection.UP:          70,
        TrendDirection.DOWN:        70,
        TrendDirection.STRONG_UP:   45,
        TrendDirection.STRONG_DOWN: 40,
    }
    trend_score = float(_trend_score_map.get(trend, 60))

    # Decay / spread compression
    decay_score = _spread_compression(buy_prices, sell_prices)

    return {
        "trend":       trend,
        "trend_score": trend_score,
        "ma_signal":   round(ma_signal, 4),
        "decay_score": round(decay_score, 4),
    }


def _detect_trend(prices: List[float]) -> TrendDirection:
    if len(prices) < 4:
        return TrendDirection.NEUTRAL
    half = len(prices) // 2
    first_half_avg = sum(prices[:half]) / half
    second_half_avg = sum(prices[half:]) / (len(prices) - half)
    if first_half_avg <= 0:
        return TrendDirection.NEUTRAL
    change = (second_half_avg - first_half_avg) / first_half_avg
    if change > 0.03:
        return TrendDirection.STRONG_UP
    if change > 0.01:
        return TrendDirection.UP
    if change < -0.03:
        return TrendDirection.STRONG_DOWN
    if change < -0.01:
        return TrendDirection.DOWN
    return TrendDirection.NEUTRAL


def _ma_cross(prices: List[float], short: int = 5, long: int = 20) -> float:
    """Returns a signal in [–1, 1]: positive = short MA above long MA."""
    if len(prices) < long:
        return 0.0
    ma_short = sum(prices[-short:]) / short
    ma_long  = sum(prices[-long:])  / long
    if ma_long == 0:
        return 0.0
    return max(-1.0, min(1.0, (ma_short - ma_long) / ma_long))


def _spread_compression(
    buy_prices: List[float],
    sell_prices: List[float],
) -> float:
    """Rate of spread narrowing (positive = compressing, negative = widening)."""
    if len(buy_prices) < 2 or len(sell_prices) < 2:
        return 0.0
    spreads = [s - b for s, b in zip(sell_prices, buy_prices) if s > 0 and b > 0]
    if len(spreads) < 2:
        return 0.0
    # Linear regression slope over spread sequence (normalised)
    n = len(spreads)
    xs = list(range(n))
    mean_x = (n - 1) / 2.0
    mean_y = sum(spreads) / n
    num = sum((xs[i] - mean_x) * (spreads[i] - mean_y) for i in range(n))
    den = sum((xs[i] - mean_x) ** 2 for i in range(n)) or 1.0
    slope = num / den
    # Normalise by average spread
    avg_spread = mean_y or 1.0
    return -slope / avg_spread   # negative slope = compression = positive decay score


def compute_fill_probability(
    volume_5m: int,
    quantity: int,
    buy_price: int,
    sell_price: int,
    avg_spread_pct: float,
) -> float:
    """
    Estimate order fill probability as a fraction in [0, 1].

    Higher volume and tighter spread → easier fill → higher probability.
    """
    if volume_5m <= 0:
        return 0.0

    vol_score = min(1.0, volume_5m / max(1, _vol_target(buy_price)))
    spread_pct = avg_spread_pct if avg_spread_pct > 0 else 0.02
    spread_score = max(0.0, 1.0 - (spread_pct / 0.12))   # 0 at 12 % spread

    # Qty penalty: large orders are harder to fill
    if buy_price > 0:
        order_value = buy_price * quantity
        # If order represents more than 30 % of 5 min volume × price → penalise
        market_value = volume_5m * buy_price
        size_ratio = order_value / max(1, market_value)
        qty_penalty = max(0.0, 1.0 - size_ratio * 2)
    else:
        qty_penalty = 1.0

    prob = vol_score * spread_score * qty_penalty
    return round(max(0.0, min(1.0, prob)), 3)


# ---------------------------------------------------------------------------
# Master feature bundle
# ---------------------------------------------------------------------------

def compute_all_features(
    item_id: int,
    buy_price: int,
    sell_price: int,
    volume_5m: int,
    snapshots_1h: Sequence[Any],
    snapshots_24h: Optional[Sequence[Any]] = None,
    quantity: int = 1,
) -> Dict[str, Any]:
    """
    Compute the full feature bundle used by ``scoring.score_item``.

    Args:
        item_id:       OSRS item ID.
        buy_price:     Current instant-buy price.
        sell_price:    Current instant-sell price.
        volume_5m:     5-minute traded volume.
        snapshots_1h:  List of recent price snapshots (≤60 min).
        snapshots_24h: Optional extended history (24 h).
        quantity:      Intended trade quantity (default 1 for scoring).

    Returns a flat dict of float/str features ready for the scorer.
    """
    feat: Dict[str, Any] = {
        "item_id":     item_id,
        "buy_price":   buy_price,
        "sell_price":  sell_price,
        "volume_5m":   volume_5m,
    }

    feat.update(compute_volatility(snapshots_1h, snapshots_24h))
    feat.update(compute_volume_features(snapshots_1h, volume_5m, buy_price))
    feat.update(compute_trend_features(snapshots_1h))

    if buy_price > 0 and sell_price > 0:
        spread_pct = (sell_price - buy_price) / sell_price
    else:
        spread_pct = 0.0
    feat["spread_pct"] = round(spread_pct, 4)

    feat["fill_probability"] = compute_fill_probability(
        volume_5m, quantity, buy_price, sell_price, spread_pct
    )

    return feat
