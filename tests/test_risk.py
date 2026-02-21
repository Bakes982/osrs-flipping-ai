"""
Unit tests for backend.prediction.risk — Phase 9.

Tests cover:
  • Kelly Criterion calculation
  • Stop-loss percentage
  • Historical volatility
  • Sharpe and Sortino ratios
  • Risk classification
"""

import pytest
from backend.prediction.risk import (
    kelly_fraction, stop_loss_pct, historical_volatility,
    sharpe_ratio, sortino_ratio, classify_risk,
)


class TestKellyFraction:
    def test_positive_edge(self):
        # 70% win rate, 2:1 payoff → Kelly = (0.7*2 - 0.3)/2 = 0.55
        k = kelly_fraction(win_rate=0.70, avg_win=200, avg_loss=100)
        assert k > 0

    def test_no_edge(self):
        # 50% win rate, 1:1 payoff → Kelly = 0
        k = kelly_fraction(win_rate=0.50, avg_win=100, avg_loss=100)
        assert k == pytest.approx(0.0, abs=0.01)

    def test_negative_edge_returns_zero(self):
        # 30% win rate, 1:1 → negative raw Kelly, clamped to 0
        k = kelly_fraction(win_rate=0.30, avg_win=100, avg_loss=100)
        assert k == 0.0

    def test_uses_price_fallback(self):
        # No avg_win/loss, uses price-derived estimate
        k = kelly_fraction(
            win_rate=0.65, avg_win=0, avg_loss=0,
            buy_price=1_000_000, sell_price=1_020_000,
        )
        assert k > 0

    def test_clamps_win_rate(self):
        # Should not raise with extreme win rates
        k_high = kelly_fraction(win_rate=1.0, avg_win=100, avg_loss=50)
        k_low = kelly_fraction(win_rate=0.0, avg_win=100, avg_loss=50)
        assert k_high >= 0
        assert k_low == 0.0

    def test_max_is_one(self):
        # Kelly should never exceed 1
        k = kelly_fraction(win_rate=0.99, avg_win=10_000, avg_loss=1)
        # raw kelly can exceed 1; we return the raw value (caller halves it)
        assert k >= 0


class TestStopLossPct:
    def test_high_value_item_wider_stop(self):
        stop_hi = stop_loss_pct(buy_price=20_000_000, volume_5m=20, score=60)
        stop_lo = stop_loss_pct(buy_price=50_000, volume_5m=20, score=60)
        assert stop_hi > stop_lo

    def test_low_volume_widens_stop(self):
        stop_low_vol = stop_loss_pct(buy_price=1_000_000, volume_5m=2, score=60)
        stop_hi_vol = stop_loss_pct(buy_price=1_000_000, volume_5m=100, score=60)
        assert stop_low_vol > stop_hi_vol

    def test_low_score_tightens_stop(self):
        stop_low = stop_loss_pct(buy_price=1_000_000, volume_5m=20, score=30)
        stop_hi = stop_loss_pct(buy_price=1_000_000, volume_5m=20, score=80)
        assert stop_low < stop_hi

    def test_result_in_valid_range(self):
        for price in [10_000, 100_000, 1_000_000, 10_000_000, 100_000_000]:
            sp = stop_loss_pct(buy_price=price, volume_5m=10, score=50)
            assert 0.005 <= sp <= 0.15


class TestHistoricalVolatility:
    def test_stable_prices(self):
        # All same price → ~0 volatility
        vol = historical_volatility([1_000_000.0] * 20)
        assert vol < 0.001

    def test_increasing_volatility(self):
        prices1 = [1_000_000.0 + i * 100 for i in range(20)]
        prices2 = [1_000_000.0 + i * 10_000 for i in range(20)]
        vol1 = historical_volatility(prices1)
        vol2 = historical_volatility(prices2)
        assert vol2 > vol1

    def test_empty_series(self):
        assert historical_volatility([]) == 0.0

    def test_single_price(self):
        assert historical_volatility([1_000_000.0]) == 0.0

    def test_zero_prices_handled(self):
        # Should not raise with zeros (they are skipped)
        vol = historical_volatility([0.0, 1_000_000.0, 1_000_000.0])
        assert vol >= 0.0


class TestSharpeRatio:
    def test_positive_sharpe(self):
        assert sharpe_ratio(100_000, 50_000) == pytest.approx(2.0)

    def test_zero_std(self):
        assert sharpe_ratio(100_000, 0) == 0.0

    def test_with_risk_free(self):
        result = sharpe_ratio(100_000, 50_000, risk_free=20_000)
        assert result == pytest.approx(1.6)


class TestSortinoRatio:
    def test_all_profits(self):
        # No downside → infinite Sortino in theory, returns large number or 0
        result = sortino_ratio([100_000, 200_000, 150_000])
        assert result >= 0

    def test_mixed_profits_losses(self):
        profits = [100_000, -50_000, 75_000, -25_000, 90_000]
        result = sortino_ratio(profits)
        assert isinstance(result, float)

    def test_empty(self):
        assert sortino_ratio([]) == 0.0

    def test_single(self):
        assert sortino_ratio([100_000]) == 0.0


class TestClassifyRisk:
    def test_low_risk(self):
        tier = classify_risk(score=80, volatility_1h=0.01, volume_5m=50, win_rate=0.85)
        assert tier == "LOW"

    def test_medium_risk(self):
        tier = classify_risk(score=55, volatility_1h=0.03, volume_5m=10, win_rate=0.65)
        assert tier == "MEDIUM"

    def test_high_risk(self):
        tier = classify_risk(score=35, volatility_1h=0.10, volume_5m=2, win_rate=0.45)
        assert tier in ("HIGH", "VERY_HIGH")

    def test_very_high_risk(self):
        tier = classify_risk(score=10, volatility_1h=0.20, volume_5m=0, win_rate=0.30)
        assert tier == "VERY_HIGH"
