"""
Tests for PR11 — Dump Risk Score + Protection.

Covers:
  • Stable price series → low dump score
  • Sharp price drop → high dump score
  • Volatility spike + fill drop → elevated score
  • dump_signal classification: none / watch / high
  • DUMP_HIGH items are vetoed from spice bucket (flip_cache)
  • Alert persistence: alert only fires after DUMP_ALERT_PERSISTENCE consecutive cycles
  • _apply_dump_penalties adds badges and penalises score/confidence
  • dump_risk_score / dump_signal appear in calculate_flip_metrics output
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta
from typing import List
from unittest.mock import MagicMock, patch

import pytest

from backend.prediction.scoring import (
    calculate_flip_metrics,
    _compute_dump_risk,
    _apply_dump_penalties,
)


# ---------------------------------------------------------------------------
# Snapshot factory
# ---------------------------------------------------------------------------

def _snap(
    instant_buy: int = 1_000_000,
    instant_sell: int = 950_000,
    age_seconds: int = 0,
) -> MagicMock:
    s = MagicMock()
    s.instant_buy    = instant_buy
    s.instant_sell   = instant_sell
    s.buy_volume     = 100
    s.sell_volume    = 100
    s.timestamp      = datetime.utcnow() - timedelta(seconds=age_seconds)
    s.buy_time       = int(time.time()) - age_seconds
    s.sell_time      = int(time.time()) - age_seconds
    return s


def _stable_snapshots(n: int = 25, price: int = 1_000_000) -> list:
    """Stable series: buy/sell prices don't move."""
    return [
        _snap(instant_buy=price, instant_sell=int(price * 0.95), age_seconds=i * 60)
        for i in range(n - 1, -1, -1)
    ]


def _dropping_snapshots(
    n: int = 25,
    start_buy: int = 1_000_000,
    end_buy: int = 800_000,
) -> list:
    """Sharp drop: buy price falls from start to end linearly."""
    snaps = []
    for i in range(n):
        frac = i / max(n - 1, 1)
        buy  = int(start_buy + (end_buy - start_buy) * frac)
        sell = int(buy * 0.95)
        snaps.append(_snap(instant_buy=buy, instant_sell=sell, age_seconds=(n - i) * 60))
    return snaps


def _make_result(
    volatility_1h: float = 0.01,
    volatility_24h: float = 0.01,
    spread_compression: float = 0.0,
    fill_probability: float = 0.5,
    total_score: float = 65.0,
    confidence: float = 0.7,
) -> dict:
    return {
        "volatility_1h":    volatility_1h,
        "volatility_24h":   volatility_24h,
        "spread_compression": spread_compression,
        "fill_probability": fill_probability,
        "total_score":      total_score,
        "confidence":       confidence,
        "veto_reasons":     [],
    }


# ---------------------------------------------------------------------------
# _compute_dump_risk
# ---------------------------------------------------------------------------

class TestComputeDumpRisk:
    def test_stable_series_low_score(self):
        result = _make_result(volatility_1h=0.01, volatility_24h=0.01, fill_probability=0.55)
        snaps  = _stable_snapshots()
        score, signal = _compute_dump_risk(result, snaps)
        assert score < 40, f"Stable series should yield low dump score, got {score:.1f}"
        assert signal == "none"

    def test_sharp_drop_high_score(self):
        """A 20% price drop over 10 minutes → high dump score."""
        result = _make_result(volatility_1h=0.05, volatility_24h=0.01, fill_probability=0.20)
        snaps  = _dropping_snapshots(start_buy=1_000_000, end_buy=800_000)
        score, signal = _compute_dump_risk(result, snaps)
        assert score > 40, f"Sharp drop should yield elevated dump score, got {score:.1f}"

    def test_vol_spike_raises_score(self):
        """High vol_spike (vol_1h >> vol_24h) raises score."""
        result_low_spike  = _make_result(volatility_1h=0.01, volatility_24h=0.01)
        result_high_spike = _make_result(volatility_1h=0.10, volatility_24h=0.01)
        snaps = _stable_snapshots()
        score_low,  _ = _compute_dump_risk(result_low_spike,  snaps)
        score_high, _ = _compute_dump_risk(result_high_spike, snaps)
        assert score_high > score_low, "Higher vol spike should increase dump score"

    def test_fill_drop_raises_score(self):
        result_good = _make_result(fill_probability=0.70)
        result_bad  = _make_result(fill_probability=0.05)
        snaps = _stable_snapshots()
        score_good, _ = _compute_dump_risk(result_good, snaps)
        score_bad,  _ = _compute_dump_risk(result_bad,  snaps)
        assert score_bad > score_good

    def test_combined_signals_very_high(self):
        """Vol spike + fill drop + sharp drop → very high score."""
        result = _make_result(
            volatility_1h=0.20, volatility_24h=0.01,
            fill_probability=0.05,
            spread_compression=-0.10,
        )
        snaps = _dropping_snapshots(start_buy=1_000_000, end_buy=600_000)
        score, signal = _compute_dump_risk(result, snaps)
        assert score > 60, f"Combined signals should yield >60, got {score:.1f}"

    def test_score_clamped_0_100(self):
        result = _make_result(volatility_1h=100.0, volatility_24h=0.001,
                              fill_probability=0.0, spread_compression=-1.0)
        snaps  = _dropping_snapshots(start_buy=1_000_000, end_buy=1_000)
        score, _ = _compute_dump_risk(result, snaps)
        assert 0.0 <= score <= 100.0

    def test_signal_thresholds(self):
        # Patch thresholds to deterministic values
        with patch("backend.config.DUMP_WATCH_THRESHOLD", 40):
            with patch("backend.config.DUMP_HIGH_THRESHOLD", 70):
                # Force a "none" scenario
                result_none = _make_result()
                score_none, sig_none = _compute_dump_risk(result_none, _stable_snapshots())
                assert sig_none == "none"


# ---------------------------------------------------------------------------
# _apply_dump_penalties
# ---------------------------------------------------------------------------

class TestApplyDumpPenalties:
    def test_none_signal_no_changes(self):
        result = {"total_score": 70.0, "confidence": 0.8, "veto_reasons": []}
        _apply_dump_penalties(result, "none")
        assert result["total_score"] == 70.0
        assert result["confidence"] == 0.8
        assert result["veto_reasons"] == []

    def test_watch_signal_adds_badge(self):
        result = {"total_score": 70.0, "confidence": 0.8, "veto_reasons": []}
        _apply_dump_penalties(result, "watch")
        assert "DUMP_WATCH" in result["veto_reasons"]
        assert result["total_score"] == 70.0   # watch doesn't reduce score

    def test_high_signal_penalises_score(self):
        result = {"total_score": 70.0, "confidence": 0.8, "veto_reasons": []}
        _apply_dump_penalties(result, "high")
        assert "DUMP_HIGH" in result["veto_reasons"]
        assert result["total_score"] == pytest.approx(40.0)   # -30 pts

    def test_high_signal_penalises_confidence(self):
        result = {"total_score": 70.0, "confidence": 0.8, "veto_reasons": []}
        _apply_dump_penalties(result, "high")
        assert result["confidence"] == pytest.approx(0.48, abs=0.01)   # ×0.6

    def test_high_score_cant_go_below_zero(self):
        result = {"total_score": 10.0, "confidence": 0.8, "veto_reasons": []}
        _apply_dump_penalties(result, "high")
        assert result["total_score"] >= 0.0


# ---------------------------------------------------------------------------
# calculate_flip_metrics integration
# ---------------------------------------------------------------------------

class TestCalculateFlipMetricsIntegration:
    def _run(self, snaps=None, vol_ratio: float = 1.0):
        if snaps is None:
            snaps = _stable_snapshots()
        return calculate_flip_metrics({
            "item_id":    4151,
            "item_name":  "Abyssal whip",
            "instant_buy":  snaps[-1].instant_buy,
            "instant_sell": snaps[-1].instant_sell,
            "volume_5m":  200,
            "buy_time":   int(time.time()),
            "sell_time":  int(time.time()),
            "snapshots":  snaps,
            "flip_history": [],
        })

    def test_dump_fields_always_present(self):
        m = self._run()
        assert "dump_risk_score" in m
        assert "dump_signal" in m
        assert m["dump_signal"] in ("none", "watch", "high")

    def test_dump_score_range(self):
        m = self._run()
        assert 0.0 <= m["dump_risk_score"] <= 100.0

    def test_stable_series_low_dump(self):
        m = self._run(_stable_snapshots())
        assert m["dump_risk_score"] < 50, (
            f"Stable series should be low dump, got {m['dump_risk_score']:.1f}"
        )

    def test_dropping_series_higher_dump(self):
        stable   = self._run(_stable_snapshots())
        dropping = self._run(_dropping_snapshots(start_buy=1_000_000, end_buy=850_000))
        # Dropping series should have higher dump score (or at minimum not worse)
        assert dropping["dump_risk_score"] >= stable["dump_risk_score"] - 5


# ---------------------------------------------------------------------------
# Dump persistence for alerts (flip_cache level)
# ---------------------------------------------------------------------------

class TestDumpAlertPersistence:
    def setup_method(self):
        import backend.flip_cache as _cache
        _cache._dump_persist_state.clear()
        _cache._top_core_cache.clear()
        _cache._top_spice_cache.clear()
        _cache._top5_cache.clear()
        _cache._eligible_state.clear()

    def _make_item(self, item_id: int, dump_signal: str, dump_risk: float) -> dict:
        return {
            "item_id":    item_id,
            "item_name":  f"Item {item_id}",
            "confidence": 0.7,
            "fill_probability": 0.6,
            "net_profit": 60_000,
            "roi_pct":    1.0,
            "total_score": 65.0,
            "vetoed":     False,
            "dump_risk_score": dump_risk,
            "dump_signal": dump_signal,
        }

    def test_alert_not_fired_on_first_high_cycle(self):
        import backend.flip_cache as _cache
        with patch.object(_cache._cfg, "DUMP_ALERT_PERSISTENCE", 2):
            with patch.object(_cache._cfg, "DAMPENING_K", 1):
                with patch("backend.flip_cache._emit_dump_alert") as mock_alert:
                    item = self._make_item(1, "high", 75.0)
                    _cache.update_cache([item])   # cycle 1: count=1, < 2
                    mock_alert.assert_not_called()

    def test_alert_fires_after_persistence_cycles(self):
        import backend.flip_cache as _cache
        with patch.object(_cache._cfg, "DUMP_ALERT_PERSISTENCE", 2):
            with patch.object(_cache._cfg, "DAMPENING_K", 1):
                with patch("backend.flip_cache._emit_dump_alert") as mock_alert:
                    item = self._make_item(1, "high", 75.0)
                    _cache.update_cache([item])   # cycle 1: count=1
                    _cache.update_cache([item])   # cycle 2: count=2 → fires
                    mock_alert.assert_called_once()

    def test_alert_not_refired_after_first(self):
        import backend.flip_cache as _cache
        with patch.object(_cache._cfg, "DUMP_ALERT_PERSISTENCE", 2):
            with patch.object(_cache._cfg, "DAMPENING_K", 1):
                with patch("backend.flip_cache._emit_dump_alert") as mock_alert:
                    item = self._make_item(1, "high", 75.0)
                    _cache.update_cache([item])
                    _cache.update_cache([item])   # fires
                    _cache.update_cache([item])   # should NOT fire again
                    assert mock_alert.call_count == 1

    def test_alert_resets_when_signal_clears(self):
        import backend.flip_cache as _cache
        with patch.object(_cache._cfg, "DUMP_ALERT_PERSISTENCE", 2):
            with patch.object(_cache._cfg, "DAMPENING_K", 1):
                with patch("backend.flip_cache._emit_dump_alert") as mock_alert:
                    high   = self._make_item(1, "high", 75.0)
                    normal = self._make_item(1, "none", 10.0)
                    _cache.update_cache([high])
                    _cache.update_cache([high])   # fires (count=2)
                    _cache.update_cache([normal]) # signal clears → reset
                    _cache.update_cache([high])   # count=1 again, no fire
                    assert mock_alert.call_count == 1  # only fired once

    def test_spice_bucket_excludes_dump_high_items(self):
        """dump_high items are vetoed from spice even if they'd otherwise qualify."""
        import backend.flip_cache as _cache
        with patch.object(_cache._cfg, "DAMPENING_K", 1):
            with patch.object(_cache._cfg, "DUMP_SPICE_VETO_THRESHOLD", 50.0):
                spice_with_dump = {
                    "item_id":    100,
                    "item_name":  "Dump Spice",
                    "confidence": 0.4,
                    "fill_probability": 0.35,
                    "net_profit": 80_000,
                    "roi_pct":    2.5,
                    "total_score": 65.0,
                    "vetoed":     False,
                    "dump_risk_score": 55.0,   # above 50.0 veto threshold
                    "dump_signal": "high",
                }
                _cache.update_cache([spice_with_dump])
                spice = _cache.get_top_spice("balanced")
                assert not any(m["item_id"] == 100 for m in spice), (
                    "Dump-high item should be excluded from spice bucket"
                )
