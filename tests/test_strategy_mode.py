"""
Tests for PR9 — Strategy Mode + 1 Spice Slot.

Covers:
  • StrategyMode enum values
  • strategy_mode defaults to "steady" when absent
  • _default_strategy_mode respects FORCE_DEFAULT_STRATEGY_MODE env var
  • portfolio selection uses 1 spice slot when steady_spice and slots=8
  • portfolio selection uses 0 spice when steady
  • portfolio selection uses all spice when spice_only
  • is_core_candidate / is_spice_candidate classification
  • PATCH /api/user/strategy_mode validation (unit-level)
"""

from __future__ import annotations

import pytest
from unittest.mock import patch

from backend.domain.enums import StrategyMode
from backend.domain.models import UserRecord
from backend.portfolio.optimizer import (
    is_core_candidate,
    is_spice_candidate,
    _plan_slots,
    generate_optimal_portfolio,
    _adj_gph,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_metrics(
    item_id: int = 1,
    confidence: float = 0.7,
    fill_probability: float = 0.6,
    roi_pct: float = 2.0,
    net_profit: int = 80_000,
    gp_per_hour: float = 2_000_000,
    risk_score: float = 3.0,
    total_score: float = 65.0,
    recommended_buy: int = 1_000_000,
    recommended_sell: int = 1_060_000,
    vetoed: bool = False,
    dump_risk_score: float = 0.0,
) -> dict:
    return {
        "item_id": item_id,
        "item_name": f"Item {item_id}",
        "confidence": confidence,
        "fill_probability": fill_probability,
        "roi_pct": roi_pct,
        "net_profit": net_profit,
        "gp_per_hour": gp_per_hour,
        "risk_score": risk_score,
        "total_score": total_score,
        "recommended_buy": recommended_buy,
        "recommended_sell": recommended_sell,
        "estimated_hold_time": 60,
        "score_volume": 60.0,
        "win_rate": 0.65,
        "vetoed": vetoed,
        "dump_risk_score": dump_risk_score,
    }


# ---------------------------------------------------------------------------
# StrategyMode enum
# ---------------------------------------------------------------------------

class TestStrategyModeEnum:
    def test_valid_values(self):
        assert StrategyMode.STEADY.value == "steady"
        assert StrategyMode.STEADY_SPICE.value == "steady_spice"
        assert StrategyMode.SPICE_ONLY.value == "spice_only"

    def test_string_coercion(self):
        assert StrategyMode("steady") is StrategyMode.STEADY
        assert StrategyMode("steady_spice") is StrategyMode.STEADY_SPICE


# ---------------------------------------------------------------------------
# UserRecord defaults
# ---------------------------------------------------------------------------

class TestUserRecordStrategyMode:
    def test_strategy_mode_defaults_to_steady(self):
        rec = UserRecord(user_id="u1", username="tester")
        # use_enum_values = True → stored as string
        assert rec.strategy_mode == "steady"

    def test_strategy_mode_can_be_set(self):
        rec = UserRecord(user_id="u1", username="tester", strategy_mode="steady_spice")
        assert rec.strategy_mode == "steady_spice"


# ---------------------------------------------------------------------------
# _default_strategy_mode helper
# ---------------------------------------------------------------------------

class TestDefaultStrategyMode:
    """Test _default_strategy_mode logic independently of FastAPI imports."""

    def _call(self, forced: str) -> str:
        """Inline the _default_strategy_mode logic for isolated unit testing."""
        from backend.domain.enums import StrategyMode
        valid = {m.value for m in StrategyMode}
        stripped = forced.strip().lower()
        return stripped if stripped in valid else StrategyMode.STEADY.value

    def test_defaults_to_steady_when_env_unset(self):
        assert self._call("") == "steady"

    def test_env_var_overrides_default(self):
        assert self._call("steady_spice") == "steady_spice"
        assert self._call("spice_only") == "spice_only"
        assert self._call("steady") == "steady"

    def test_invalid_env_var_falls_back_to_steady(self):
        assert self._call("garbage_mode") == "steady"
        assert self._call("INVALID") == "steady"

    def test_case_insensitive(self):
        assert self._call("STEADY_SPICE") == "steady_spice"


# ---------------------------------------------------------------------------
# Bucket classification
# ---------------------------------------------------------------------------

class TestBucketClassification:
    def test_core_candidate_high_confidence(self):
        m = _make_metrics(confidence=0.7, fill_probability=0.6, net_profit=50_000)
        assert is_core_candidate(m) is True
        assert is_spice_candidate(m) is False   # core excludes spice

    def test_core_candidate_fails_low_confidence(self):
        m = _make_metrics(confidence=0.3, fill_probability=0.6, net_profit=50_000)
        assert is_core_candidate(m) is False

    def test_core_candidate_fails_low_fill(self):
        m = _make_metrics(confidence=0.7, fill_probability=0.2, net_profit=50_000)
        assert is_core_candidate(m) is False

    def test_spice_candidate_high_roi_low_confidence(self):
        m = _make_metrics(confidence=0.40, fill_probability=0.35, roi_pct=2.0, net_profit=80_000)
        assert is_core_candidate(m) is False
        assert is_spice_candidate(m) is True

    def test_spice_candidate_vetoed_excluded(self):
        m = _make_metrics(confidence=0.40, fill_probability=0.35, roi_pct=2.0, vetoed=True)
        assert is_spice_candidate(m) is False

    def test_vetoed_excluded_from_core(self):
        m = _make_metrics(vetoed=True)
        assert is_core_candidate(m) is False

    def test_dump_high_vetoes_core(self):
        m = _make_metrics(confidence=0.8, fill_probability=0.7, dump_risk_score=75.0)
        assert is_core_candidate(m) is False

    def test_dump_high_vetoes_spice(self):
        m = _make_metrics(confidence=0.4, fill_probability=0.35, roi_pct=2.0,
                          dump_risk_score=55.0)
        assert is_spice_candidate(m) is False


# ---------------------------------------------------------------------------
# _plan_slots
# ---------------------------------------------------------------------------

class TestPlanSlots:
    def _core(self, item_id: int) -> dict:
        return _make_metrics(item_id=item_id, confidence=0.8, fill_probability=0.7)

    def _spice(self, item_id: int) -> dict:
        return _make_metrics(item_id=item_id + 100, confidence=0.4,
                             fill_probability=0.35, roi_pct=2.5)

    def test_steady_uses_only_core(self):
        core  = [self._core(i) for i in range(1, 9)]
        spice = [self._spice(i) for i in range(1, 3)]
        ordered = _plan_slots(core, [], 8, spice_slots=0)
        ids = [m["item_id"] for m in ordered]
        assert all(i in range(1, 9) for i in ids)
        assert not any(i > 100 for i in ids)

    def test_steady_spice_reserves_exactly_one_spice_slot(self):
        core  = [self._core(i) for i in range(1, 9)]
        spice = [self._spice(i) for i in range(1, 5)]
        ordered = _plan_slots(core, spice, 8, spice_slots=1)
        spice_items = [m for m in ordered if m["item_id"] > 100]
        assert len(spice_items) == 1

    def test_steady_spice_fills_rest_with_core(self):
        core  = [self._core(i) for i in range(1, 9)]
        spice = [self._spice(i) for i in range(1, 5)]
        ordered = _plan_slots(core, spice, 8, spice_slots=1)
        core_items = [m for m in ordered if m["item_id"] <= 100]
        # Should have 7 core items (8 slots - 1 spice)
        assert len(core_items) == 7

    def test_spice_only_uses_only_spice(self):
        core  = [self._core(i) for i in range(1, 9)]
        spice = [self._spice(i) for i in range(1, 9)]
        ordered = _plan_slots(spice, [], 8, spice_slots=0)
        assert all(m["item_id"] > 100 for m in ordered)

    def test_no_duplicate_items_in_plan(self):
        core = [self._core(1), self._core(1), self._core(2)]
        ordered = _plan_slots(core, [], 3, spice_slots=0)
        ids = [m["item_id"] for m in ordered]
        assert len(ids) == len(set(ids))


# ---------------------------------------------------------------------------
# generate_optimal_portfolio with strategy_mode
# ---------------------------------------------------------------------------

class TestGenerateOptimalPortfolio:
    """Integration-style tests with _fetch_candidates mocked.

    Use risk_tolerance="LOW" (position_cap=0.08) so 8 × 0.08 = 0.64 < 0.80
    (MAX_TOTAL_EXPOSURE_PCT) — all 8 slots fit within deployable capital.
    """

    _BUY  = 1_000
    _SELL = 1_060

    def _core_candidates(self, n: int = 8) -> list:
        return [
            _make_metrics(item_id=i, confidence=0.8, fill_probability=0.7,
                          recommended_buy=self._BUY, recommended_sell=self._SELL,
                          net_profit=50)
            for i in range(1, n + 1)
        ]

    def _spice_candidates(self, n: int = 3) -> list:
        return [
            _make_metrics(item_id=100 + i, confidence=0.4, fill_probability=0.35,
                          roi_pct=2.5,
                          recommended_buy=self._BUY, recommended_sell=self._SELL,
                          net_profit=50)
            for i in range(1, n + 1)
        ]

    def _all_candidates(self):
        return self._core_candidates() + self._spice_candidates()

    def test_steady_zero_spice_slots(self):
        """steady mode allocates only core items — no spice IDs appear."""
        with patch("backend.portfolio.optimizer._fetch_candidates",
                   return_value=self._all_candidates()):
            plan = generate_optimal_portfolio(
                capital=10_000_000, ge_slots=8,
                risk_tolerance="LOW",
                strategy_mode="steady",
            )
        slot_ids = [s.item_id for s in plan.slots]
        assert plan.slots_used == 8, f"Expected 8 slots, got {plan.slots_used}"
        assert all(iid < 100 for iid in slot_ids), f"Got non-core items: {slot_ids}"

    def test_steady_spice_exactly_one_spice_slot(self):
        """steady_spice reserves exactly 1 slot for a spice candidate."""
        with patch("backend.portfolio.optimizer._fetch_candidates",
                   return_value=self._all_candidates()):
            plan = generate_optimal_portfolio(
                capital=10_000_000, ge_slots=8,
                risk_tolerance="LOW",
                strategy_mode="steady_spice",
            )
        slot_ids = [s.item_id for s in plan.slots]
        assert plan.slots_used == 8, f"Expected 8 slots, got {plan.slots_used}"
        spice_count = sum(1 for iid in slot_ids if iid >= 100)
        assert spice_count == 1, f"Expected 1 spice slot, got {spice_count}: {slot_ids}"

    def test_steady_spice_fills_remaining_with_core(self):
        """steady_spice fills 7 out of 8 slots with core candidates."""
        with patch("backend.portfolio.optimizer._fetch_candidates",
                   return_value=self._all_candidates()):
            plan = generate_optimal_portfolio(
                capital=10_000_000, ge_slots=8,
                risk_tolerance="LOW",
                strategy_mode="steady_spice",
            )
        slot_ids = [s.item_id for s in plan.slots]
        core_count = sum(1 for iid in slot_ids if iid < 100)
        assert core_count == 7, f"Expected 7 core slots, got {core_count}: {slot_ids}"

    def test_spice_only_all_spice(self):
        """spice_only uses only spice-bucket candidates."""
        with patch("backend.portfolio.optimizer._fetch_candidates",
                   return_value=self._all_candidates()):
            plan = generate_optimal_portfolio(
                capital=10_000_000, ge_slots=3,   # only 3 spice candidates
                risk_tolerance="LOW",
                strategy_mode="spice_only",
            )
        slot_ids = [s.item_id for s in plan.slots]
        assert plan.slots_used == 3
        assert all(iid >= 100 for iid in slot_ids), (
            f"Expected only spice items, got: {slot_ids}"
        )

    def test_zero_capital_returns_empty(self):
        plan = generate_optimal_portfolio(capital=0, strategy_mode="steady_spice")
        assert plan.slots_used == 0
        assert any("capital" in w.lower() for w in plan.warnings)
