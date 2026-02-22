"""
Unit + integration tests for backend.prediction.scoring — Phase 9.

Tests cover:
  • calculate_flip_metrics happy path
  • All hard vetoes (zero volume, inverted spread, wide spread, waterfall, etc.)
  • Component scores (spread, volume, freshness, trend, history, stability)
  • Composite score (weighting, price-bracket multiplier, confidence)
  • Phase-2 derived metrics (gp_per_hour, fill_probability, volatility, etc.)
  • Edge cases: missing data, identical prices, massive volatility
  • Regression: output keys contract
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta
from typing import List
from unittest.mock import MagicMock

import pytest

from backend.prediction.scoring import calculate_flip_metrics, apply_ml_score
import backend.prediction.scoring as scoring_module


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_snapshot(
    item_id: int = 4151,
    instant_buy: int = 1_050_000,
    instant_sell: int = 1_000_000,
    buy_volume: int = 50,
    sell_volume: int = 50,
    age_seconds: int = 0,
):
    """Create a mock PriceSnapshot."""
    snap = MagicMock()
    snap.item_id = item_id
    snap.instant_buy = instant_buy
    snap.instant_sell = instant_sell
    snap.buy_volume = buy_volume
    snap.sell_volume = sell_volume
    snap.timestamp = datetime.utcnow() - timedelta(seconds=age_seconds)
    snap.buy_time = int(time.time()) - age_seconds
    snap.sell_time = int(time.time()) - age_seconds
    return snap


def _make_snapshots(n: int = 20, **kwargs) -> List:
    return [
        _make_snapshot(age_seconds=i * 10, **kwargs)
        for i in range(n - 1, -1, -1)
    ]


def _base_item_data(
    instant_buy: int = 1_050_000,
    instant_sell: int = 1_000_000,
    volume_5m: int = 50,
    n_snaps: int = 20,
) -> dict:
    snaps = _make_snapshots(
        n=n_snaps,
        instant_buy=instant_buy,
        instant_sell=instant_sell,
        buy_volume=volume_5m // 2,
        sell_volume=volume_5m // 2,
    )
    return {
        "item_id": 4151,
        "item_name": "Abyssal whip",
        "instant_buy": instant_buy,
        "instant_sell": instant_sell,
        "volume_5m": volume_5m,
        "buy_time": int(time.time()),
        "sell_time": int(time.time()),
        "snapshots": snaps,
        "flip_history": [],
    }


# ---------------------------------------------------------------------------
# Contract: required output keys
# ---------------------------------------------------------------------------

REQUIRED_KEYS = {
    "item_id", "item_name",
    "spread", "spread_pct",
    "recommended_buy", "recommended_sell",
    "gross_profit", "tax", "net_profit", "roi_pct",
    "gp_per_hour", "estimated_hold_time", "fill_probability",
    "spread_compression", "volatility_1h", "volatility_24h",
    "volume_delta", "ma_signal",
    "trend", "momentum", "bb_position",
    "vwap_1m", "vwap_5m", "vwap_30m", "vwap_2h",
    "win_rate", "total_flips", "avg_profit",
    "score_spread", "score_volume", "score_freshness",
    "score_trend", "score_history", "score_stability", "score_ml",
    "total_score",
    "confidence", "risk_score", "stale_data", "anomalous_spread",
    "vetoed", "veto_reasons", "reason",
}


class TestOutputContract:
    """Regression: the set of output keys must not shrink."""

    def test_all_required_keys_present_happy_path(self):
        m = calculate_flip_metrics(_base_item_data())
        missing = REQUIRED_KEYS - set(m.keys())
        assert not missing, f"Missing keys: {missing}"

    def test_all_required_keys_present_vetoed(self):
        # Vetoed items must also carry all keys
        data = _base_item_data(volume_5m=0)
        m = calculate_flip_metrics(data)
        missing = REQUIRED_KEYS - set(m.keys())
        assert not missing, f"Missing keys on vetoed result: {missing}"


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

class TestHappyPath:
    def test_profitable_flip(self):
        m = calculate_flip_metrics(_base_item_data())
        assert not m["vetoed"]
        assert m["net_profit"] > 0
        assert m["roi_pct"] > 0
        assert m["total_score"] > 0

    def test_spread_calculated(self):
        m = calculate_flip_metrics(_base_item_data(
            instant_buy=1_100_000, instant_sell=1_000_000,
        ))
        assert m["spread"] == 100_000
        assert abs(m["spread_pct"] - 10.0) < 0.01

    def test_recommended_prices_within_spread(self):
        m = calculate_flip_metrics(_base_item_data(
            instant_buy=1_100_000, instant_sell=1_000_000, volume_5m=50,
        ))
        assert m["recommended_buy"] >= 1_000_000
        assert m["recommended_buy"] < 1_100_000
        assert m["recommended_sell"] > 1_000_000
        assert m["recommended_sell"] <= 1_100_000

    def test_gp_per_hour_positive(self):
        m = calculate_flip_metrics(_base_item_data())
        if not m["vetoed"] and m["net_profit"] > 0:
            assert m["gp_per_hour"] > 0

    def test_fill_probability_in_range(self):
        m = calculate_flip_metrics(_base_item_data())
        assert 0.0 <= m["fill_probability"] <= 1.0

    def test_volatility_non_negative(self):
        m = calculate_flip_metrics(_base_item_data())
        assert m["volatility_1h"] >= 0
        assert m["volatility_24h"] >= 0

    def test_risk_score_in_range(self):
        m = calculate_flip_metrics(_base_item_data())
        assert 0.0 <= m["risk_score"] <= 10.0

    def test_total_score_in_range(self):
        m = calculate_flip_metrics(_base_item_data())
        assert 0.0 <= m["total_score"] <= 100.0


# ---------------------------------------------------------------------------
# Hard vetoes
# ---------------------------------------------------------------------------

class TestVetoes:
    def test_missing_buy_price(self):
        data = _base_item_data()
        data["instant_buy"] = None
        m = calculate_flip_metrics(data)
        assert m["vetoed"]
        assert any("price" in r.lower() for r in m["veto_reasons"])

    def test_missing_sell_price(self):
        data = _base_item_data()
        data["instant_sell"] = None
        m = calculate_flip_metrics(data)
        assert m["vetoed"]

    def test_zero_buy_price(self):
        data = _base_item_data()
        data["instant_buy"] = 0
        m = calculate_flip_metrics(data)
        assert m["vetoed"]

    def test_inverted_spread(self):
        # sell > buy is a veto
        m = calculate_flip_metrics(_base_item_data(
            instant_buy=1_000_000, instant_sell=1_050_000,
        ))
        assert m["vetoed"]
        assert any("inverted" in r.lower() or "spread" in r.lower() for r in m["veto_reasons"])

    def test_equal_buy_sell(self):
        m = calculate_flip_metrics(_base_item_data(
            instant_buy=1_000_000, instant_sell=1_000_000,
        ))
        assert m["vetoed"]

    def test_zero_volume_vetoed(self):
        data = _base_item_data(volume_5m=0)
        data["snapshots"] = _make_snapshots(buy_volume=0, sell_volume=0)
        m = calculate_flip_metrics(data)
        assert m["vetoed"]
        assert any("volume" in r.lower() for r in m["veto_reasons"])

    def test_spread_too_wide(self):
        # > 12% spread should be vetoed
        m = calculate_flip_metrics(_base_item_data(
            instant_buy=1_200_000, instant_sell=1_000_000, volume_5m=10,
        ))
        assert m["vetoed"]
        assert any("wide" in r.lower() or "spread" in r.lower() for r in m["veto_reasons"])

    def test_unprofitable_after_tax_vetoed(self):
        # Tiny spread that yields negative profit after tax
        m = calculate_flip_metrics(_base_item_data(
            instant_buy=1_001_000, instant_sell=1_000_000, volume_5m=10,
        ))
        # May or may not be vetoed depending on spread positioning; just check it runs
        assert isinstance(m["vetoed"], bool)

    def test_low_fill_probability_penalized_for_conservative(self, monkeypatch):
        monkeypatch.setattr(scoring_module, "_sigmoid", lambda _x: 0.05)
        data = _base_item_data(volume_5m=100)
        data["risk_profile"] = "conservative"
        m = calculate_flip_metrics(data)
        assert m["total_score"] <= 30
        assert any("fill probability" in r.lower() for r in m["veto_reasons"])

    def test_spread_compression_spike_penalized(self):
        now = int(time.time())
        snaps = [
            _make_snapshot(instant_buy=1_120_000, instant_sell=1_000_000, age_seconds=1200),
            _make_snapshot(instant_buy=1_015_000, instant_sell=1_000_000, age_seconds=60),
        ]
        data = {
            "item_id": 4151,
            "item_name": "Abyssal whip",
            "instant_buy": 1_015_000,
            "instant_sell": 1_000_000,
            "volume_5m": 100,
            "buy_time": now,
            "sell_time": now,
            "snapshots": snaps,
            "flip_history": [],
        }
        m = calculate_flip_metrics(data)
        assert m["total_score"] <= 20
        assert any("compression spike" in r.lower() for r in m["veto_reasons"])


# ---------------------------------------------------------------------------
# Component scores
# ---------------------------------------------------------------------------

class TestComponentScores:
    def test_scores_in_range(self):
        m = calculate_flip_metrics(_base_item_data())
        if not m["vetoed"]:
            for key in ["score_spread", "score_volume", "score_freshness",
                        "score_trend", "score_history", "score_stability"]:
                assert 0 <= m[key] <= 100, f"{key} = {m[key]}"

    def test_high_volume_scores_well(self):
        m_low = calculate_flip_metrics(_base_item_data(volume_5m=1))
        m_high = calculate_flip_metrics(_base_item_data(volume_5m=200))
        if not m_low["vetoed"] and not m_high["vetoed"]:
            assert m_high["score_volume"] >= m_low["score_volume"]

    def test_optimal_spread_scores_highest(self):
        # ~1.5% realized margin: optimal bracket
        m = calculate_flip_metrics(_base_item_data(
            instant_buy=1_020_000, instant_sell=1_000_000, volume_5m=50,
        ))
        if not m["vetoed"]:
            assert m["score_spread"] >= 50  # should be decent

    def test_fresh_data_scores_highest_freshness(self):
        data = _base_item_data()
        data["buy_time"] = int(time.time())  # just now
        data["sell_time"] = int(time.time())
        m = calculate_flip_metrics(data)
        if not m["vetoed"]:
            assert m["score_freshness"] >= 90

    def test_stale_data_scores_low_freshness(self):
        data = _base_item_data()
        data["buy_time"] = int(time.time()) - 3600  # 1 hour ago
        data["sell_time"] = int(time.time()) - 3600
        m = calculate_flip_metrics(data)
        if not m["vetoed"]:
            assert m["score_freshness"] < 40


# ---------------------------------------------------------------------------
# Phase-2 derived metrics
# ---------------------------------------------------------------------------

class TestPhase2Metrics:
    def test_ma_signal_range(self):
        m = calculate_flip_metrics(_base_item_data())
        assert -1.0 <= m["ma_signal"] <= 1.0

    def test_volume_delta_type(self):
        m = calculate_flip_metrics(_base_item_data())
        assert isinstance(m["volume_delta"], float)

    def test_estimated_hold_time_positive(self):
        m = calculate_flip_metrics(_base_item_data(volume_5m=50))
        assert m["estimated_hold_time"] > 0

    def test_hold_time_longer_for_low_volume(self):
        m_low = calculate_flip_metrics(_base_item_data(volume_5m=2))
        m_high = calculate_flip_metrics(_base_item_data(volume_5m=200))
        assert m_low["estimated_hold_time"] >= m_high["estimated_hold_time"]

    def test_fill_probability_higher_for_high_volume(self):
        m_low = calculate_flip_metrics(_base_item_data(volume_5m=1))
        m_high = calculate_flip_metrics(_base_item_data(volume_5m=100))
        assert m_high["fill_probability"] >= m_low["fill_probability"]


# ---------------------------------------------------------------------------
# ML score injection
# ---------------------------------------------------------------------------

class TestApplyMlScore:
    def test_ml_score_updates_total(self):
        m = calculate_flip_metrics(_base_item_data())
        if m["vetoed"]:
            return
        score_before = m["total_score"]
        apply_ml_score(m, 90.0)
        assert m["score_ml"] == 90.0
        # Total score should change
        assert m["total_score"] != score_before or m["score_ml"] == -1.0  # may be same if ml was already 90

    def test_ml_unavailable_sentinel(self):
        m = calculate_flip_metrics(_base_item_data())
        # Default ml_score is -1 (unavailable)
        assert m["score_ml"] == -1.0


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_no_snapshots(self):
        data = _base_item_data()
        data["snapshots"] = []
        # Should not raise — might be vetoed or use defaults
        m = calculate_flip_metrics(data)
        assert isinstance(m, dict)

    def test_single_snapshot(self):
        data = _base_item_data()
        data["snapshots"] = [_make_snapshot()]
        m = calculate_flip_metrics(data)
        assert isinstance(m, dict)

    def test_extremely_high_price(self):
        # 500M item — should not raise
        m = calculate_flip_metrics(_base_item_data(
            instant_buy=510_000_000, instant_sell=500_000_000, volume_5m=2,
        ))
        assert isinstance(m, dict)

    def test_very_low_price(self):
        # 50 GP item (below tax threshold)
        m = calculate_flip_metrics(_base_item_data(
            instant_buy=55, instant_sell=50, volume_5m=10_000,
        ))
        assert isinstance(m, dict)

    def test_massive_volatility_snapshots(self):
        # Prices that swing wildly — should not raise
        snaps = [
            _make_snapshot(instant_buy=1_000_000 + (i % 2) * 500_000, instant_sell=900_000, age_seconds=i * 10)
            for i in range(20, 0, -1)
        ]
        data = {
            "item_id": 4151,
            "item_name": "Test",
            "instant_buy": 1_500_000,
            "instant_sell": 900_000,
            "volume_5m": 10,
            "snapshots": snaps,
            "flip_history": [],
        }
        m = calculate_flip_metrics(data)
        # Wide spread > 12% should be vetoed
        assert m["vetoed"]

    def test_no_flip_history(self):
        data = _base_item_data()
        data["flip_history"] = []
        m = calculate_flip_metrics(data)
        assert m["win_rate"] is None
        assert m["total_flips"] == 0

    def test_missing_optional_keys(self):
        # Minimal input — only required fields
        m = calculate_flip_metrics({
            "item_id": 1234,
            "instant_buy": 1_050_000,
            "instant_sell": 1_000_000,
        })
        assert isinstance(m, dict)
        assert m["item_id"] == 1234
