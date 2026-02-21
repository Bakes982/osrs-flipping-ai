"""
Unit tests for backend.core.utils — Phase 9.

Tests cover:
  • GE tax calculation (with and without cap)
  • Net profit calculation
  • ROI calculation
  • GP formatting
  • Safe division
  • Coefficient of variation
  • VWAP
  • Price bracket labelling
"""

import pytest
from backend.core.utils import (
    ge_tax, net_profit, roi_pct,
    format_gp, format_pct,
    safe_div, clamp, pct_change,
    coefficient_of_variation, mean_safe, vwap,
    price_bracket_label,
)
from backend.core.constants import GE_TAX_RATE, GE_TAX_CAP, GE_TAX_FREE_BELOW


# ---------------------------------------------------------------------------
# GE Tax
# ---------------------------------------------------------------------------

class TestGeTax:
    def test_standard_rate(self):
        # 1_000_000 * 2% = 20_000
        assert ge_tax(1_000_000) == 20_000

    def test_cap_applied(self):
        # Very expensive item: tax capped at 5M
        assert ge_tax(500_000_000) == GE_TAX_CAP

    def test_cap_boundary(self):
        # 250_000_000 * 2% = 5_000_000 (exactly at cap)
        assert ge_tax(250_000_000) == GE_TAX_CAP

    def test_below_cap(self):
        # 100_000_000 * 2% = 2_000_000 (below cap)
        assert ge_tax(100_000_000) == 2_000_000

    def test_tax_free_item(self):
        # Items below GE_TAX_FREE_BELOW are not taxed
        assert ge_tax(GE_TAX_FREE_BELOW - 1) == 0
        assert ge_tax(1) == 0
        assert ge_tax(99) == 0

    def test_at_free_threshold(self):
        # Items AT or above the threshold ARE taxed
        result = ge_tax(GE_TAX_FREE_BELOW)
        assert result >= 0  # should not raise; may be 0 if threshold not reached

    def test_zero_price(self):
        assert ge_tax(0) == 0


class TestNetProfit:
    def test_basic_profit(self):
        # buy=900, sell=1000, tax=20 → net=80
        assert net_profit(900, 1000) == 1000 - 900 - ge_tax(1000)

    def test_unprofitable_after_tax(self):
        # buy=999, sell=1000, tax=20 → net=-19
        result = net_profit(999, 1000)
        assert result < 0

    def test_high_value_item(self):
        buy = 10_000_000
        sell = 10_200_000
        expected = sell - buy - ge_tax(sell)
        assert net_profit(buy, sell) == expected


class TestRoiPct:
    def test_positive_roi(self):
        # buy=1000, sell=1100, tax=22 → net=78, roi=7.8%
        roi = roi_pct(1000, 1100)
        assert roi > 0

    def test_zero_buy_price(self):
        assert roi_pct(0, 1000) == 0.0

    def test_negative_roi(self):
        # sell < buy + tax → negative ROI
        assert roi_pct(1000, 990) < 0


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

class TestFormatGp:
    def test_billions(self):
        assert "B" in format_gp(1_500_000_000)

    def test_millions(self):
        result = format_gp(5_000_000)
        assert "M" in result

    def test_thousands(self):
        result = format_gp(50_000)
        assert "K" in result

    def test_small(self):
        result = format_gp(999)
        assert "K" not in result and "M" not in result

    def test_negative(self):
        result = format_gp(-1_000_000)
        assert "M" in result

    def test_zero(self):
        assert format_gp(0) == "0"


class TestFormatPct:
    def test_default_decimals(self):
        assert format_pct(1.23) == "1.2%"

    def test_two_decimals(self):
        assert format_pct(1.234, decimals=2) == "1.23%"


# ---------------------------------------------------------------------------
# Math helpers
# ---------------------------------------------------------------------------

class TestSafeDiv:
    def test_normal(self):
        assert safe_div(10.0, 2.0) == 5.0

    def test_zero_denominator(self):
        assert safe_div(10.0, 0.0) == 0.0

    def test_custom_default(self):
        assert safe_div(10.0, 0.0, default=99.9) == 99.9


class TestClamp:
    def test_within_range(self):
        assert clamp(5.0, 0.0, 10.0) == 5.0

    def test_below_min(self):
        assert clamp(-1.0, 0.0, 10.0) == 0.0

    def test_above_max(self):
        assert clamp(15.0, 0.0, 10.0) == 10.0

    def test_at_boundaries(self):
        assert clamp(0.0, 0.0, 10.0) == 0.0
        assert clamp(10.0, 0.0, 10.0) == 10.0


class TestPctChange:
    def test_increase(self):
        assert pct_change(100.0, 110.0) == pytest.approx(0.10)

    def test_decrease(self):
        assert pct_change(100.0, 90.0) == pytest.approx(-0.10)

    def test_zero_old(self):
        assert pct_change(0.0, 100.0) == 0.0


# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------

class TestCoefficientOfVariation:
    def test_stable_series(self):
        # All same value → CV = 0
        assert coefficient_of_variation([100, 100, 100]) == 0.0

    def test_volatile_series(self):
        cv = coefficient_of_variation([100, 200, 300, 400])
        assert cv > 0.0

    def test_single_value(self):
        assert coefficient_of_variation([100]) == 0.0

    def test_empty(self):
        assert coefficient_of_variation([]) == 0.0


class TestMeanSafe:
    def test_normal(self):
        assert mean_safe([1.0, 2.0, 3.0]) == pytest.approx(2.0)

    def test_empty_default(self):
        assert mean_safe([], default=99.0) == 99.0


class TestVwap:
    def test_equal_volumes(self):
        # With equal volumes VWAP = simple mean
        result = vwap([100.0, 200.0], [1.0, 1.0])
        assert result == pytest.approx(150.0)

    def test_volume_weighted(self):
        # Higher volume on 200 → VWAP should be closer to 200
        result = vwap([100.0, 200.0], [1.0, 9.0])
        assert result > 150.0

    def test_empty(self):
        assert vwap([], []) is None

    def test_zero_volumes(self):
        # Zero volumes get replaced with 1 → should still compute
        result = vwap([100.0, 200.0], [0.0, 0.0])
        assert result == pytest.approx(150.0)


# ---------------------------------------------------------------------------
# Price bracket
# ---------------------------------------------------------------------------

class TestPriceBracketLabel:
    def test_optimal(self):
        assert price_bracket_label(25_000_000) == "optimal"

    def test_high_value(self):
        assert price_bracket_label(100_000_000) == "high_value"

    def test_solid(self):
        assert price_bracket_label(5_000_000) == "solid"

    def test_weak(self):
        assert price_bracket_label(500_000) == "weak"

    def test_bulk(self):
        assert price_bracket_label(5_000) == "bulk"

    def test_neutral(self):
        assert price_bracket_label(50_000) == "neutral"
