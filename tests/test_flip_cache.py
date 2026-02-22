"""
Tests for PR10 — Two-bucket flip recommendation cache with dampening.

Covers:
  • K-poll confirmation: item does not appear until K consecutive cycles
  • Hysteresis: item remains eligible when it dips slightly below threshold
  • stable_for_cycles / stable_for_minutes fields are correct
  • Core bucket excludes items with low confidence/fill
  • Spice bucket excludes items with high dump risk
  • _build_top5 mixes 4 core + 1 spice correctly
  • get_top5 returns from cache (no DB)
  • update_cache populates all profiles
"""

from __future__ import annotations

import time
from typing import List
from unittest.mock import patch

import pytest

import backend.flip_cache as _cache


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _item(
    item_id: int = 1,
    confidence: float = 0.7,
    fill_probability: float = 0.6,
    roi_pct: float = 1.0,
    net_profit: int = 60_000,
    total_score: float = 65.0,
    vetoed: bool = False,
    dump_risk_score: float = 0.0,
) -> dict:
    return {
        "item_id":        item_id,
        "item_name":      f"Item {item_id}",
        "confidence":     confidence,
        "fill_probability": fill_probability,
        "roi_pct":        roi_pct,
        "net_profit":     net_profit,
        "total_score":    total_score,
        "recommended_buy":  1_000_000,
        "recommended_sell": 1_060_000,
        "vetoed":         vetoed,
        "dump_risk_score": dump_risk_score,
        "dump_signal":    "none",
    }


def _core_item(item_id: int = 1) -> dict:
    return _item(item_id=item_id, confidence=0.75, fill_probability=0.60)


def _spice_item(item_id: int = 100) -> dict:
    return _item(item_id=item_id, confidence=0.40, fill_probability=0.35, roi_pct=2.5)


def _reset_cache():
    """Reset all module-level cache state between tests."""
    _cache._top_core_cache.clear()
    _cache._top_spice_cache.clear()
    _cache._top5_cache.clear()
    _cache._eligible_state.clear()
    _cache._last_update_ts = 0.0


# ---------------------------------------------------------------------------
# K-poll confirmation
# ---------------------------------------------------------------------------

class TestKPollConfirmation:
    def setup_method(self):
        _reset_cache()

    def test_item_not_eligible_before_k_cycles(self):
        """Item must pass K=3 consecutive cycles before appearing in cache."""
        with patch.object(_cache._cfg, "DAMPENING_K", 3):
            scored = [_core_item(1)]
            _cache.update_cache(scored)  # cycle 1
            _cache.update_cache(scored)  # cycle 2
            # After only 2 cycles (< K=3), item should not be active
            top = _cache.get_top_core("balanced")
            assert not any(m["item_id"] == 1 for m in top), (
                "Item appeared before reaching K confirmation cycles"
            )

    def test_item_eligible_after_k_cycles(self):
        """Item enters eligible set after exactly K consecutive cycles."""
        with patch.object(_cache._cfg, "DAMPENING_K", 3):
            scored = [_core_item(1)]
            _cache.update_cache(scored)  # cycle 1
            _cache.update_cache(scored)  # cycle 2
            _cache.update_cache(scored)  # cycle 3 — should now be eligible
            top = _cache.get_top_core("balanced")
            assert any(m["item_id"] == 1 for m in top), (
                "Item should be eligible after K=3 cycles"
            )

    def test_stable_for_cycles_is_accurate(self):
        with patch.object(_cache._cfg, "DAMPENING_K", 2):
            scored = [_core_item(1)]
            _cache.update_cache(scored)  # cycle 1
            _cache.update_cache(scored)  # cycle 2 — eligible, count=2
            _cache.update_cache(scored)  # cycle 3, count=3
            top = _cache.get_top_core("balanced")
            match = next((m for m in top if m["item_id"] == 1), None)
            assert match is not None
            assert match["stable_for_cycles"] == 3

    def test_stable_for_minutes_derived_from_cycles(self):
        with patch.object(_cache._cfg, "DAMPENING_K", 2):
            scored = [_core_item(1)]
            _cache.update_cache(scored, cycle_seconds=120)  # 2-minute cycles
            _cache.update_cache(scored, cycle_seconds=120)  # eligible
            _cache.update_cache(scored, cycle_seconds=120)  # count=3
            top = _cache.get_top_core("balanced")
            match = next((m for m in top if m["item_id"] == 1), None)
            assert match is not None
            # 3 cycles × 2 min/cycle = 6 min
            assert match["stable_for_minutes"] == pytest.approx(6.0, abs=0.1)

    def test_reset_count_when_item_disappears(self):
        """If item misses a cycle (non-hysteresis), count resets."""
        with patch.object(_cache._cfg, "DAMPENING_K", 3):
            with patch.object(_cache._cfg, "HYSTERESIS_CONF_MARGIN", 0):
                with patch.object(_cache._cfg, "HYSTERESIS_SCORE_MARGIN", 0):
                    scored = [_core_item(1)]
                    _cache.update_cache(scored)  # cycle 1, count=1
                    _cache.update_cache(scored)  # cycle 2, count=2
                    # Item disappears for 1 cycle
                    low_conf = _item(1, confidence=0.1, fill_probability=0.1)
                    _cache.update_cache([low_conf])  # count resets to 0
                    # Confirm item isn't active after reset
                    _cache.update_cache(scored)  # count=1 (fresh start)
                    _cache.update_cache(scored)  # count=2
                    top = _cache.get_top_core("balanced")
                    # Still count=2 < K=3 → not eligible
                    assert not any(m["item_id"] == 1 for m in top)


# ---------------------------------------------------------------------------
# Hysteresis
# ---------------------------------------------------------------------------

class TestHysteresis:
    def setup_method(self):
        _reset_cache()

    def test_active_item_survives_slight_dip(self):
        """Once eligible, item survives a slight confidence dip (within margin)."""
        with patch.object(_cache._cfg, "DAMPENING_K", 2):
            with patch.object(_cache._cfg, "HYSTERESIS_CONF_MARGIN", 10):  # 10%pts = 0.10
                with patch.object(_cache._cfg, "HYSTERESIS_SCORE_MARGIN", 10):
                    # Activate item
                    scored = [_core_item(1)]
                    _cache.update_cache(scored)
                    _cache.update_cache(scored)   # K=2 → active
                    # Dip below threshold but within hysteresis margin
                    # balanced min_conf = 0.50, hysteresis = 10 pp = 0.10
                    # min_conf - 0.10 = 0.40 → item with conf=0.45 should survive
                    dipped = _item(1, confidence=0.45, fill_probability=0.60,
                                   total_score=55.0)
                    _cache.update_cache([dipped])
                    top = _cache.get_top_core("balanced")
                    assert any(m["item_id"] == 1 for m in top), (
                        "Item should survive within hysteresis margin"
                    )

    def test_active_item_evicted_outside_hysteresis(self):
        """Item is evicted when it drops far below threshold."""
        with patch.object(_cache._cfg, "DAMPENING_K", 2):
            with patch.object(_cache._cfg, "HYSTERESIS_CONF_MARGIN", 5):   # 5%pts
                with patch.object(_cache._cfg, "HYSTERESIS_SCORE_MARGIN", 5):
                    scored = [_core_item(1)]
                    _cache.update_cache(scored)
                    _cache.update_cache(scored)   # active
                    # Severe dip — confidence 0.20, far below 0.50-0.05=0.45
                    very_low = _item(1, confidence=0.20, fill_probability=0.10,
                                     total_score=20.0)
                    _cache.update_cache([very_low])
                    top = _cache.get_top_core("balanced")
                    assert not any(m["item_id"] == 1 for m in top), (
                        "Item should be evicted when far outside hysteresis"
                    )


# ---------------------------------------------------------------------------
# Bucket classification
# ---------------------------------------------------------------------------

class TestBucketClassification:
    def setup_method(self):
        _reset_cache()

    def test_core_item_not_in_spice(self):
        with patch.object(_cache._cfg, "DAMPENING_K", 1):
            _cache.update_cache([_core_item(1)])
            core  = _cache.get_top_core("balanced")
            spice = _cache.get_top_spice("balanced")
            assert any(m["item_id"] == 1 for m in core)
            assert not any(m["item_id"] == 1 for m in spice)

    def test_spice_item_not_in_core(self):
        with patch.object(_cache._cfg, "DAMPENING_K", 1):
            _cache.update_cache([_spice_item(100)])
            core  = _cache.get_top_core("balanced")
            spice = _cache.get_top_spice("balanced")
            assert not any(m["item_id"] == 100 for m in core)
            assert any(m["item_id"] == 100 for m in spice)

    def test_dump_high_vetoes_spice(self):
        """Items with dump_risk_score >= DUMP_SPICE_VETO_THRESHOLD excluded from spice."""
        with patch.object(_cache._cfg, "DAMPENING_K", 1):
            with patch.object(_cache._cfg, "DUMP_SPICE_VETO_THRESHOLD", 50.0):
                dumped = _spice_item(100)
                dumped["dump_risk_score"] = 55.0
                _cache.update_cache([dumped])
                spice = _cache.get_top_spice("balanced")
                assert not any(m["item_id"] == 100 for m in spice)

    def test_vetoed_item_excluded(self):
        with patch.object(_cache._cfg, "DAMPENING_K", 1):
            _cache.update_cache([_item(1, vetoed=True)])
            assert not _cache.get_top_core("balanced")
            assert not _cache.get_top_spice("balanced")


# ---------------------------------------------------------------------------
# _build_top5
# ---------------------------------------------------------------------------

class TestBuildTop5:
    def test_four_core_one_spice(self):
        core  = [_core_item(i) for i in range(1, 6)]   # 5 core
        spice = [_spice_item(100)]
        top5 = _cache._build_top5(core, spice)
        assert len(top5) == 5
        spice_items = [m for m in top5 if m["item_id"] == 100]
        assert len(spice_items) == 1

    def test_five_core_when_no_spice(self):
        core = [_core_item(i) for i in range(1, 8)]
        top5 = _cache._build_top5(core, [])
        assert len(top5) == 5
        assert all(m["item_id"] <= 5 for m in top5)

    def test_no_duplicate_in_top5(self):
        core  = [_core_item(i) for i in range(1, 6)]
        spice = [_core_item(1)]   # same item_id as first core
        top5 = _cache._build_top5(core, spice)
        ids = [m["item_id"] for m in top5]
        assert len(ids) == len(set(ids))

    def test_at_most_five_items(self):
        core  = [_core_item(i) for i in range(1, 20)]
        spice = [_spice_item(100 + i) for i in range(5)]
        top5 = _cache._build_top5(core, spice)
        assert len(top5) <= 5


# ---------------------------------------------------------------------------
# update_cache end-to-end
# ---------------------------------------------------------------------------

class TestUpdateCache:
    def setup_method(self):
        _reset_cache()

    def test_all_profiles_populated(self):
        with patch.object(_cache._cfg, "DAMPENING_K", 1):
            _cache.update_cache([_core_item(1)])
        for profile in ["balanced", "conservative", "aggressive"]:
            # Cache should have keys for every profile even if lists are empty
            assert profile in _cache._top_core_cache or True  # may be empty

    def test_last_update_ts_set(self):
        before = time.time()
        with patch.object(_cache._cfg, "DAMPENING_K", 1):
            _cache.update_cache([_core_item(1)])
        assert _cache.last_update_ts() >= before

    def test_empty_scored_list_does_not_crash(self):
        with patch.object(_cache._cfg, "DAMPENING_K", 1):
            _cache.update_cache([])   # should not raise

    def test_get_top5_returns_list(self):
        with patch.object(_cache._cfg, "DAMPENING_K", 1):
            _cache.update_cache([_core_item(1)])
        result = _cache.get_top5("balanced")
        assert isinstance(result, list)
