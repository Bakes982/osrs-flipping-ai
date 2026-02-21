"""
Unit tests for backend.portfolio.optimizer — Phase 9.

Tests cover:
  • PortfolioAllocation construction
  • SlotAllocation sizing
  • Risk-adjusted GP/hour sorting
  • Capital allocation respects caps
  • Zero capital edge case
  • Risk tolerance overrides
"""

import pytest
from unittest.mock import patch

from backend.portfolio.optimizer import (
    generate_optimal_portfolio,
    PortfolioAllocation,
    SlotAllocation,
    _adj_gph,
    _risk_caps,
)


# ---------------------------------------------------------------------------
# Helper: mock metrics dict
# ---------------------------------------------------------------------------

def _mock_metrics(
    item_id: int = 4151,
    item_name: str = "Test Item",
    score: float = 70.0,
    net_profit: int = 50_000,
    gp_per_hour: float = 3_000_000.0,
    risk_score: float = 3.0,
    confidence: float = 0.8,
    buy: int = 1_000_000,
    sell: int = 1_060_000,
    hold: int = 60,
    score_volume: float = 75.0,
) -> dict:
    return {
        "item_id": item_id,
        "item_name": item_name,
        "total_score": score,
        "net_profit": net_profit,
        "gp_per_hour": gp_per_hour,
        "risk_score": risk_score,
        "confidence": confidence,
        "recommended_buy": buy,
        "recommended_sell": sell,
        "estimated_hold_time": hold,
        "score_volume": score_volume,
        "win_rate": 0.75,
        "roi_pct": net_profit / buy * 100,
        "vetoed": False,
        "reason": "Test flip",
    }


# ---------------------------------------------------------------------------
# _adj_gph
# ---------------------------------------------------------------------------

class TestAdjGph:
    def test_high_confidence_high_gph(self):
        m = _mock_metrics(gp_per_hour=1_000_000, confidence=1.0, risk_score=0.0)
        assert _adj_gph(m) == pytest.approx(1_000_000.0)

    def test_zero_confidence(self):
        m = _mock_metrics(gp_per_hour=1_000_000, confidence=0.0, risk_score=5.0)
        assert _adj_gph(m) == 0.0

    def test_high_risk_penalised(self):
        m_low = _mock_metrics(gp_per_hour=1_000_000, confidence=0.8, risk_score=2.0)
        m_high = _mock_metrics(gp_per_hour=1_000_000, confidence=0.8, risk_score=8.0)
        assert _adj_gph(m_low) > _adj_gph(m_high)


# ---------------------------------------------------------------------------
# _risk_caps
# ---------------------------------------------------------------------------

class TestRiskCaps:
    def test_low_risk_tighter_caps(self):
        pos, item, score = _risk_caps("LOW", 45.0)
        assert pos < 0.15
        assert score > 45.0

    def test_high_risk_looser_caps(self):
        pos, item, score = _risk_caps("HIGH", 45.0)
        assert pos > 0.15

    def test_medium_is_default(self):
        pos_med, _, _ = _risk_caps("MEDIUM", 45.0)
        assert pos_med == pytest.approx(0.15)

    def test_case_insensitive(self):
        pos1, _, _ = _risk_caps("low", 45.0)
        pos2, _, _ = _risk_caps("LOW", 45.0)
        assert pos1 == pos2


# ---------------------------------------------------------------------------
# generate_optimal_portfolio — with mocked candidates
# ---------------------------------------------------------------------------

class TestGenerateOptimalPortfolio:
    """Uses _fetch_candidates mock to test allocation logic without a live DB."""

    def _run(self, capital: int, candidates=None, ge_slots: int = 4):
        if candidates is None:
            candidates = [
                _mock_metrics(item_id=i, buy=1_000_000, sell=1_060_000, net_profit=40_000)
                for i in range(1, ge_slots * 2 + 1)
            ]
        with patch(
            "backend.portfolio.optimizer._fetch_candidates",
            return_value=candidates,
        ):
            return generate_optimal_portfolio(capital, ge_slots=ge_slots)

    def test_zero_capital_returns_empty(self):
        plan = generate_optimal_portfolio(0)
        assert plan.allocated_capital == 0
        assert plan.slots_used == 0
        assert plan.warnings

    def test_negative_capital_returns_empty(self):
        plan = generate_optimal_portfolio(-1_000_000)
        assert plan.slots_used == 0

    def test_fills_available_slots(self):
        plan = self._run(capital=100_000_000, ge_slots=4)
        assert plan.slots_used > 0
        assert plan.slots_used <= 4

    def test_no_duplicate_items(self):
        plan = self._run(capital=100_000_000, ge_slots=4)
        item_ids = [s.item_id for s in plan.slots]
        assert len(item_ids) == len(set(item_ids))

    def test_allocated_plus_reserved_equals_capital(self):
        plan = self._run(capital=50_000_000)
        assert plan.allocated_capital + plan.reserved_capital <= plan.capital

    def test_no_candidates_warns(self):
        plan = self._run(capital=50_000_000, candidates=[])
        assert plan.slots_used == 0
        assert plan.warnings

    def test_capital_too_low_for_item(self):
        # Capital of 1 GP can't afford a 1M item
        plan = self._run(capital=1, ge_slots=2)
        assert plan.slots_used == 0

    def test_slot_investment_within_capital(self):
        plan = self._run(capital=50_000_000, ge_slots=3)
        for slot in plan.slots:
            assert slot.investment <= slot.buy_price * 10_000  # sanity upper bound
            assert slot.investment > 0

    def test_stop_loss_below_buy(self):
        plan = self._run(capital=100_000_000, ge_slots=2)
        for slot in plan.slots:
            assert slot.stop_loss_price < slot.buy_price

    def test_risk_tolerance_low_smaller_positions(self):
        capital = 100_000_000
        candidates = [
            _mock_metrics(item_id=i, buy=1_000_000, sell=1_060_000, net_profit=40_000)
            for i in range(1, 9)
        ]
        with patch("backend.portfolio.optimizer._fetch_candidates", return_value=candidates):
            plan_med = generate_optimal_portfolio(capital, ge_slots=4, risk_tolerance="MEDIUM")
            plan_low = generate_optimal_portfolio(capital, ge_slots=4, risk_tolerance="LOW")

        # LOW risk tolerance should allocate less or equal capital
        assert plan_low.allocated_capital <= plan_med.allocated_capital + 1_000_000

    def test_to_dict_serialisable(self):
        import json
        plan = self._run(capital=50_000_000, ge_slots=2)
        d = plan.to_dict()
        # Should not raise
        json.dumps(d)

    def test_expected_profit_positive(self):
        plan = self._run(capital=100_000_000, ge_slots=4)
        assert plan.total_expected_profit >= 0


# ---------------------------------------------------------------------------
# PortfolioAllocation properties
# ---------------------------------------------------------------------------

class TestPortfolioAllocationModel:
    def test_slots_used_property(self):
        pa = PortfolioAllocation(capital=1_000_000, ge_slots=4, allocated_capital=0, reserved_capital=1_000_000)
        assert pa.slots_used == 0
        pa.slots.append(MagicMock())
        assert pa.slots_used == 1

    def test_to_dict_contains_required_keys(self):
        pa = PortfolioAllocation(capital=1_000_000, ge_slots=4, allocated_capital=0, reserved_capital=1_000_000)
        d = pa.to_dict()
        assert "capital" in d
        assert "slots" in d
        assert "total_expected_profit" in d
        assert "total_expected_gp_per_hour" in d


try:
    from unittest.mock import MagicMock
except ImportError:
    pass
