"""
Tests for backend.analytics.trade_plan.build_trade_plan()
and backend.flip_cache._format_dump_message().

Covers:
  1. Position cap constrains max_invest_gp
  2. item_limit caps qty_to_buy
  3. Liquidity throttle lowers qty when liquidity is low
  4. profit_per_item and total_profit correct (including tax, tax cap, tax-free)
  5. Zero/negative profit guard
  6. Dump alert formatter: contains human-readable fields, NOT item_id
"""

from __future__ import annotations

import math
import pytest

from backend.analytics.trade_plan import build_trade_plan


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TAX_RATE = 0.02
_TAX_CAP  = 5_000_000
_TAX_FREE_BELOW = 100


def _plan(
    buy=1_000,
    sell=1_100,
    item_limit=None,
    liq=50.0,
    cap_pct=0.15,
    capital=50_000_000,
    tax_rate=_TAX_RATE,
    tax_cap=_TAX_CAP,
    tax_free=_TAX_FREE_BELOW,
):
    return build_trade_plan(
        buy_price=buy,
        sell_price=sell,
        item_limit=item_limit,
        liquidity_score=liq,
        risk_profile_position_cap_pct=cap_pct,
        capital_gp=capital,
        ge_tax_rate=tax_rate,
        ge_tax_cap=tax_cap,
        ge_tax_free_below=tax_free,
    )


# ---------------------------------------------------------------------------
# 1. Position cap
# ---------------------------------------------------------------------------

class TestPositionCap:
    def test_max_invest_gp_respects_cap(self):
        result = _plan(buy=1_000, sell=1_100, capital=10_000_000, cap_pct=0.15)
        assert result["max_invest_gp"] == math.floor(10_000_000 * 0.15)

    def test_qty_bounded_by_capital(self):
        # capital=100k, cap=10%, buy=1000 → max_invest=10k → qty_cap=10
        # liq=50 → throttle=0.75 → qty=7
        result = _plan(buy=1_000, sell=1_100, capital=100_000, cap_pct=0.10, liq=50.0)
        max_invest = math.floor(100_000 * 0.10)   # 10_000
        qty_cap = math.floor(max_invest / 1_000)   # 10
        expected_qty = math.floor(qty_cap * (0.5 + 0.5 * 0.5))  # 10 * 0.75 = 7
        assert result["qty_to_buy"] == expected_qty

    def test_low_cap_pct_limits_invest(self):
        r = _plan(capital=1_000_000, cap_pct=0.05)
        assert r["max_invest_gp"] == 50_000


# ---------------------------------------------------------------------------
# 2. Item limit
# ---------------------------------------------------------------------------

class TestItemLimit:
    def test_item_limit_caps_qty(self):
        # With 50M capital and 15% cap, qty_cap would be large;
        # item_limit=100 must override it.
        r = _plan(buy=1_000, sell=1_100, item_limit=100, liq=100.0, capital=50_000_000, cap_pct=0.15)
        assert r["qty_to_buy"] <= 100

    def test_item_limit_zero_ignored(self):
        # item_limit=0 should NOT cap qty
        r_no_limit = _plan(buy=1_000, sell=1_100, item_limit=None, liq=100.0, capital=10_000_000, cap_pct=0.15)
        r_zero_limit = _plan(buy=1_000, sell=1_100, item_limit=0, liq=100.0, capital=10_000_000, cap_pct=0.15)
        assert r_no_limit["qty_to_buy"] == r_zero_limit["qty_to_buy"]

    def test_item_limit_larger_than_capital_cap_no_effect(self):
        # capital=10M, cap=1%, buy=1M → qty_cap=100 < item_limit=10000
        r = _plan(buy=1_000_000, sell=1_050_000, item_limit=10_000, liq=100.0, capital=10_000_000, cap_pct=0.01)
        max_invest = math.floor(10_000_000 * 0.01)
        qty_cap = math.floor(max_invest / 1_000_000)
        assert r["qty_to_buy"] <= qty_cap


# ---------------------------------------------------------------------------
# 3. Liquidity throttle
# ---------------------------------------------------------------------------

class TestLiquidityThrottle:
    def test_zero_liquidity_halves_qty(self):
        r_liq0   = _plan(buy=1_000, sell=1_100, liq=0.0,   item_limit=None, capital=10_000_000, cap_pct=0.10)
        r_liq100 = _plan(buy=1_000, sell=1_100, liq=100.0, item_limit=None, capital=10_000_000, cap_pct=0.10)
        # liq=0 → factor=0.5, liq=100 → factor=1.0; ratio should be ~0.5
        assert r_liq0["qty_to_buy"] == math.floor(r_liq100["qty_to_buy"] * 0.5)

    def test_mid_liquidity_between_extremes(self):
        r_liq50  = _plan(buy=1_000, sell=1_100, liq=50.0,  item_limit=None, capital=10_000_000, cap_pct=0.10)
        r_liq0   = _plan(buy=1_000, sell=1_100, liq=0.0,   item_limit=None, capital=10_000_000, cap_pct=0.10)
        r_liq100 = _plan(buy=1_000, sell=1_100, liq=100.0, item_limit=None, capital=10_000_000, cap_pct=0.10)
        assert r_liq0["qty_to_buy"] <= r_liq50["qty_to_buy"] <= r_liq100["qty_to_buy"]

    def test_none_liquidity_uses_50_default(self):
        r_none = _plan(buy=1_000, sell=1_100, liq=None, item_limit=None, capital=10_000_000, cap_pct=0.10)
        r_50   = _plan(buy=1_000, sell=1_100, liq=50.0, item_limit=None, capital=10_000_000, cap_pct=0.10)
        assert r_none["qty_to_buy"] == r_50["qty_to_buy"]


# ---------------------------------------------------------------------------
# 4. Tax calculation
# ---------------------------------------------------------------------------

class TestTaxCalculation:
    def test_normal_tax(self):
        # sell=10_000, rate=2% → tax=200, profit = 10_000 - 9_000 - 200 = 800
        r = _plan(buy=9_000, sell=10_000, liq=100.0, tax_rate=0.02, tax_cap=5_000_000, tax_free=100)
        assert r["profit_per_item"] == 10_000 - 9_000 - math.floor(10_000 * 0.02)

    def test_tax_cap_applied(self):
        # sell=1_000_000_000 → raw tax = 20M > cap of 5M
        r = _plan(buy=900_000_000, sell=1_000_000_000, liq=100.0,
                  tax_rate=0.02, tax_cap=5_000_000, tax_free=100)
        assert r["profit_per_item"] == 1_000_000_000 - 900_000_000 - 5_000_000

    def test_tax_free_below_threshold(self):
        # sell=50 < tax_free_below=100 → no tax
        r = _plan(buy=40, sell=50, liq=100.0, tax_rate=0.02, tax_cap=5_000_000, tax_free=100)
        assert r["profit_per_item"] == 50 - 40   # no tax

    def test_total_profit_equals_ppi_times_qty(self):
        r = _plan(buy=1_000, sell=1_200, liq=100.0, item_limit=500, capital=50_000_000, cap_pct=0.15)
        if r["qty_to_buy"] > 0:
            assert r["total_profit"] == r["profit_per_item"] * r["qty_to_buy"]


# ---------------------------------------------------------------------------
# 5. Zero / negative profit guard
# ---------------------------------------------------------------------------

class TestProfitGuard:
    def test_zero_profit_zeroes_qty(self):
        # buy == sell → profit_per_item = -tax → negative → guard fires
        r = _plan(buy=1_000, sell=1_000, liq=100.0)
        assert r["qty_to_buy"] == 0
        assert r["total_profit"] == 0
        assert r["profit_per_item"] >= 0   # clamped to 0

    def test_negative_profit_zeroes_qty(self):
        r = _plan(buy=2_000, sell=1_000, liq=100.0)
        assert r["qty_to_buy"] == 0
        assert r["total_profit"] == 0
        assert r["profit_per_item"] == 0

    def test_zero_buy_price_handled(self):
        # buy_price=0: qty_cap uses max(buy_price,1)=1
        r = _plan(buy=0, sell=100, liq=50.0, capital=1_000, cap_pct=0.10)
        # max_invest=100, qty_cap=100/1=100, throttle→75, sell=100 > tax_free=100 → taxed
        assert isinstance(r["qty_to_buy"], int)


# ---------------------------------------------------------------------------
# 6. Dump alert formatter
# ---------------------------------------------------------------------------

class TestDumpAlertFormatter:
    def _make_metrics(self):
        return {
            "item_id":        4151,
            "item_name":      "Abyssal whip",
            "recommended_buy":  2_000_000,
            "recommended_sell": 2_100_000,
            "dump_signal":    "high",
            "dump_risk_score": 85.0,
        }

    def test_format_contains_buy(self):
        from backend.flip_cache import _format_dump_message
        msg = _format_dump_message(self._make_metrics())
        assert "Buy" in msg

    def test_format_contains_sell(self):
        from backend.flip_cache import _format_dump_message
        msg = _format_dump_message(self._make_metrics())
        assert "Sell" in msg

    def test_format_contains_qty(self):
        from backend.flip_cache import _format_dump_message
        msg = _format_dump_message(self._make_metrics())
        assert "Qty" in msg

    def test_format_contains_profit_per_item(self):
        from backend.flip_cache import _format_dump_message
        msg = _format_dump_message(self._make_metrics())
        assert " ea" in msg

    def test_format_contains_total_profit(self):
        from backend.flip_cache import _format_dump_message
        msg = _format_dump_message(self._make_metrics())
        assert "total" in msg

    def test_format_does_not_contain_item_id_raw(self):
        """User-facing message must not leak the numeric item_id."""
        from backend.flip_cache import _format_dump_message
        msg = _format_dump_message(self._make_metrics())
        # The literal numeric id (4151) must not appear in the message
        assert "4151" not in msg

    def test_format_does_not_contain_word_item_id(self):
        from backend.flip_cache import _format_dump_message
        msg = _format_dump_message(self._make_metrics())
        assert "item_id" not in msg.lower()

    def test_format_contains_signal_uppercase(self):
        from backend.flip_cache import _format_dump_message
        msg = _format_dump_message(self._make_metrics())
        assert "HIGH" in msg

    def test_format_contains_item_name(self):
        from backend.flip_cache import _format_dump_message
        msg = _format_dump_message(self._make_metrics())
        assert "Abyssal whip" in msg
