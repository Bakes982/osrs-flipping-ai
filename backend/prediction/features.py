"""
Chunk-2 feature math helpers.

Pure functions so formulas are deterministic and easy to test.
"""

from __future__ import annotations

import math
import statistics
from typing import List, Sequence, Tuple

from backend.core.utils import clamp, safe_div


def log_return_volatility(prices: Sequence[float]) -> float:
    if len(prices) < 2:
        return 0.0
    rets = []
    for i in range(1, len(prices)):
        p0 = prices[i - 1]
        p1 = prices[i]
        if p0 > 0 and p1 > 0:
            rets.append(math.log(p1 / p0))
    if len(rets) < 2:
        return 0.0
    return statistics.stdev(rets) * math.sqrt(len(rets))


def ema(values: Sequence[float], span: int) -> float:
    if not values:
        return 0.0
    alpha = 2.0 / (span + 1.0)
    acc = float(values[0])
    for v in values[1:]:
        acc = alpha * float(v) + (1.0 - alpha) * acc
    return acc


def trend_score(mid_prices: Sequence[float], trend_ref: float = 0.01) -> Tuple[float, float]:
    if not mid_prices:
        return 0.0, 0.0
    fast = ema(mid_prices, span=6)
    slow = ema(mid_prices, span=18)
    raw = safe_div((fast - slow), max(slow, 1.0))
    score = clamp((safe_div(raw, max(trend_ref, 1e-9)) + 1.0) / 2.0, 0.0, 1.0)
    return raw, score


def decay_penalty(spread_now: float, spread_15m_ago: float, decay_ref: float = 0.25) -> Tuple[float, float]:
    if spread_15m_ago <= 0:
        return 0.0, 0.0
    raw = safe_div(spread_now - spread_15m_ago, spread_15m_ago)
    penalty = clamp(safe_div(-raw, max(decay_ref, 1e-9)), 0.0, 1.0)
    return raw, penalty


def spread_stability(spreads_1h: Sequence[float], cv_ref: float = 0.5) -> Tuple[float, float]:
    spreads = [float(x) for x in spreads_1h if x >= 0]
    if len(spreads) < 2:
        return 0.0, 0.0
    mean_s = statistics.mean(spreads)
    if mean_s <= 0:
        return 0.0, 0.0
    cv = statistics.stdev(spreads) / mean_s
    stability = clamp(1.0 - safe_div(cv, max(cv_ref, 1e-9)), 0.0, 1.0)
    return cv, stability


def liquidity_score(
    avg_update_seconds: float,
    spread_pct: float,
    spread_stability_score: float,
    freq_ref: float = 60.0,
    spread_pct_ref: float = 0.05,
) -> Tuple[float, float, float]:
    freq = clamp(safe_div(freq_ref, max(avg_update_seconds, 1.0)), 0.0, 1.0)
    spread_quality = clamp(1.0 - safe_div(spread_pct, max(spread_pct_ref, 1e-9)), 0.0, 1.0)
    liq = clamp(0.4 * freq + 0.3 * spread_quality + 0.3 * spread_stability_score, 0.0, 1.0)
    return freq, spread_quality, liq


def sigmoid_fill_probability(
    liq_score: float,
    spread_stability_score: float,
    vol_norm: float,
    decay_pen: float,
    a0: float = -0.2,
    a1: float = 2.0,
    a2: float = 1.2,
    a3: float = 1.0,
    a4: float = 1.2,
) -> float:
    x = a0 + a1 * liq_score + a2 * spread_stability_score - a3 * vol_norm - a4 * decay_pen
    if x >= 0:
        z = math.exp(-x)
        sig = 1.0 / (1.0 + z)
    else:
        z = math.exp(x)
        sig = z / (1.0 + z)
    return clamp(sig, 0.0, 1.0)


def confidence(fill_probability: float, spread_stability_score: float, vol_norm: float, decay_pen: float) -> float:
    return clamp(
        0.35 * fill_probability
        + 0.25 * spread_stability_score
        + 0.20 * (1.0 - vol_norm)
        + 0.20 * (1.0 - decay_pen),
        0.0,
        1.0,
    )


def risk_score_raw(vol_norm: float, decay_pen: float, fill_probability: float) -> float:
    return clamp(0.45 * vol_norm + 0.35 * decay_pen + 0.20 * (1.0 - fill_probability), 0.0, 1.0)

