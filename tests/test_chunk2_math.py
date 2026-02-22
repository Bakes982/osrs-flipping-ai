from __future__ import annotations

from datetime import datetime, timedelta

from backend.analytics.gp_per_hour import risk_adjusted_gph
from backend.prediction.features import (
    confidence,
    decay_penalty,
    liquidity_score,
    log_return_volatility,
    risk_score_raw,
    sigmoid_fill_probability,
    spread_stability,
    trend_score,
)
from backend.prediction.scoring import calculate_flip_metrics


def _series(n: int = 24):
    now = datetime.utcnow()
    out = []
    low = 1000
    for i in range(n):
        low += 1
        out.append(
            {
                "timestamp": now - timedelta(minutes=n - i),
                "instant_sell": low,
                "instant_buy": low + 40,
                "buy_volume": 50,
                "sell_volume": 45,
            }
        )
    return out


def test_volatility_flat_series_zero():
    vals = [100.0] * 20
    assert log_return_volatility(vals) == 0.0


def test_trend_score_positive_when_fast_above_slow():
    raw, s = trend_score([100, 101, 102, 103, 105, 107, 109, 110])
    assert raw > 0
    assert 0.5 < s <= 1.0


def test_decay_penalty_compression():
    raw, pen = decay_penalty(spread_now=60, spread_15m_ago=100, decay_ref=0.25)
    assert raw < 0
    assert pen > 0


def test_spread_stability_bounds():
    cv, stability = spread_stability([100, 102, 99, 101, 100], cv_ref=0.5)
    assert cv >= 0
    assert 0.0 <= stability <= 1.0


def test_fill_confidence_risk_ranges():
    _, _, liq = liquidity_score(avg_update_seconds=40, spread_pct=0.02, spread_stability_score=0.8)
    fill = sigmoid_fill_probability(liq, spread_stability_score=0.8, vol_norm=0.2, decay_pen=0.1)
    conf = confidence(fill, spread_stability_score=0.8, vol_norm=0.2, decay_pen=0.1)
    risk = risk_score_raw(vol_norm=0.2, decay_pen=0.1, fill_probability=fill)
    assert 0 <= fill <= 1
    assert 0 <= conf <= 1
    assert 0 <= risk <= 1


def test_scoring_output_contract_has_chunk2_fields():
    m = calculate_flip_metrics(
        {
            "item_id": 4151,
            "item_name": "Abyssal whip",
            "instant_buy": 1040,
            "instant_sell": 1000,
            "volume_5m": 95,
            "snapshots": _series(30),
            "risk_profile": "balanced",
            "user_capital": 5_000_000,
            "item_limit": 70,
        }
    )
    for key in [
        "margin_after_tax",
        "liquidity_score",
        "fill_probability",
        "decay_penalty",
        "risk_level",
        "confidence_pct",
        "qty_suggested",
        "expected_profit",
        "risk_adjusted_gp_per_hour",
        "final_score",
    ]:
        assert key in m


def test_negative_margin_caps_score():
    m = calculate_flip_metrics(
        {
            "item_id": 1,
            "instant_buy": 1010,
            "instant_sell": 1000,
            "volume_5m": 100,
            "snapshots": _series(20),
            "risk_profile": "balanced",
        }
    )
    assert m["margin_after_tax"] <= 0
    assert m["total_score"] <= 5


def test_risk_adjusted_gph_accepts_fraction_confidence():
    v = risk_adjusted_gph(1_000_000, confidence_pct=0.8, risk_score=4.0)
    assert v > 0

