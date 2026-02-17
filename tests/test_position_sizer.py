"""Tests for the PositionSizer (Kelly Criterion position sizing)."""

import pytest
from unittest.mock import patch, MagicMock
from backend.position_sizer import PositionSizer, PositionAdvice


class TestKellyFormula:
    """Test the Kelly Criterion calculation."""

    def setup_method(self):
        self.sizer = PositionSizer(bankroll=100_000_000)

    def test_positive_edge_gives_positive_kelly(self):
        """A flip with >50% win rate and positive EV should have positive Kelly."""
        advice = self.sizer.size_position(
            item_id=1,
            buy_price=1_000_000,
            sell_price=1_050_000,
            score=70,
            win_rate=0.75,
            avg_win=50_000,
            avg_loss=30_000,
            volume_5m=20,
        )
        assert advice.kelly_fraction > 0
        assert advice.half_kelly > 0
        assert advice.quantity > 0
        assert advice.max_investment > 0

    def test_negative_edge_gives_zero_position(self):
        """A flip with poor win rate should result in zero position."""
        advice = self.sizer.size_position(
            item_id=1,
            buy_price=1_000_000,
            sell_price=1_010_000,
            score=30,
            win_rate=0.3,
            avg_win=10_000,
            avg_loss=50_000,
            volume_5m=20,
        )
        assert advice.kelly_fraction == 0
        assert advice.quantity == 0

    def test_even_odds_moderate_win_rate(self):
        """50% win rate with even win/loss ratio gives near-zero Kelly."""
        advice = self.sizer.size_position(
            item_id=1,
            buy_price=1_000_000,
            sell_price=1_050_000,
            score=60,
            win_rate=0.5,
            avg_win=50_000,
            avg_loss=50_000,
            volume_5m=20,
        )
        # Kelly = (0.5 * 1 - 0.5) / 1 = 0
        assert advice.kelly_fraction == 0
        assert advice.quantity == 0

    def test_high_win_rate_high_position(self):
        """90% win rate should give a substantial Kelly fraction."""
        advice = self.sizer.size_position(
            item_id=1,
            buy_price=1_000_000,
            sell_price=1_100_000,
            score=80,
            win_rate=0.9,
            avg_win=100_000,
            avg_loss=30_000,
            volume_5m=50,
        )
        assert advice.kelly_fraction > 0.3
        assert advice.quantity > 0


class TestPositionLimits:
    """Test hard limits and caps."""

    def setup_method(self):
        self.sizer = PositionSizer(bankroll=100_000_000)

    def test_max_single_position_cap(self):
        """Never exceed MAX_SINGLE_POSITION_PCT of bankroll."""
        advice = self.sizer.size_position(
            item_id=1,
            buy_price=100_000,
            sell_price=200_000,
            score=100,
            win_rate=0.99,
            avg_win=100_000,
            avg_loss=1_000,
            volume_5m=1000,
        )
        max_allowed = int(100_000_000 * PositionSizer.MAX_SINGLE_POSITION_PCT)
        assert advice.max_investment <= max_allowed

    def test_buy_limit_respected(self):
        """Quantity should not exceed GE buy limit."""
        advice = self.sizer.size_position(
            item_id=1,
            buy_price=10_000,
            sell_price=15_000,
            score=80,
            win_rate=0.8,
            avg_win=5_000,
            avg_loss=2_000,
            volume_5m=500,
            buy_limit=100,
        )
        assert advice.quantity <= 100

    def test_volume_cap(self):
        """Don't buy more than 2x the 5-minute volume."""
        advice = self.sizer.size_position(
            item_id=1,
            buy_price=10_000,
            sell_price=15_000,
            score=80,
            win_rate=0.8,
            avg_win=5_000,
            avg_loss=2_000,
            volume_5m=5,
            buy_limit=10000,
        )
        assert advice.quantity <= 10  # 2x volume of 5

    def test_existing_exposure_limits_new_position(self):
        """Existing exposure should reduce available position size."""
        advice = self.sizer.size_position(
            item_id=1,
            buy_price=1_000_000,
            sell_price=1_100_000,
            score=80,
            win_rate=0.8,
            avg_win=100_000,
            avg_loss=30_000,
            volume_5m=50,
            existing_exposure=24_000_000,  # 24% of 100M -> near 25% limit
        )
        # Should be very small since already near MAX_ITEM_EXPOSURE_PCT
        assert advice.max_investment < 5_000_000

    def test_over_exposed_blocked(self):
        """Exceeding item exposure limit should result in zero position."""
        advice = self.sizer.size_position(
            item_id=1,
            buy_price=1_000_000,
            sell_price=1_100_000,
            score=80,
            win_rate=0.8,
            avg_win=100_000,
            avg_loss=30_000,
            volume_5m=50,
            existing_exposure=30_000_000,  # 30% > 25% limit
        )
        assert advice.within_limits is False
        assert advice.quantity == 0
        assert len(advice.limit_warnings) > 0


class TestStopLoss:
    """Test stop-loss computation."""

    def setup_method(self):
        self.sizer = PositionSizer(bankroll=100_000_000)

    def test_stop_loss_below_buy_price(self):
        """Stop loss should always be below buy price."""
        advice = self.sizer.size_position(
            item_id=1,
            buy_price=1_000_000,
            sell_price=1_050_000,
            score=70,
            win_rate=0.7,
            avg_win=50_000,
            avg_loss=30_000,
            volume_5m=20,
        )
        assert advice.stop_loss_price < advice.buy_price
        assert advice.stop_loss_pct > 0

    def test_high_value_wider_stop(self):
        """High-value items (>10M) should have wider stop-loss."""
        advice_high = self.sizer.size_position(
            item_id=1, buy_price=50_000_000, sell_price=52_000_000,
            score=70, win_rate=0.7, avg_win=2_000_000, avg_loss=1_000_000, volume_5m=20,
        )
        advice_low = self.sizer.size_position(
            item_id=2, buy_price=50_000, sell_price=55_000,
            score=70, win_rate=0.7, avg_win=5_000, avg_loss=2_000, volume_5m=20,
        )
        assert advice_high.stop_loss_pct > advice_low.stop_loss_pct

    def test_take_profit_is_sell_price(self):
        """Take profit should equal the sell price recommendation."""
        advice = self.sizer.size_position(
            item_id=1,
            buy_price=1_000_000,
            sell_price=1_050_000,
            score=70,
            win_rate=0.7,
            avg_win=50_000,
            avg_loss=30_000,
            volume_5m=20,
        )
        assert advice.take_profit_price == 1_050_000

    def test_max_hold_time_varies_by_volume(self):
        """Higher volume should allow longer hold times."""
        advice_high = self.sizer.size_position(
            item_id=1, buy_price=1_000_000, sell_price=1_050_000,
            score=70, win_rate=0.7, avg_win=50_000, avg_loss=30_000, volume_5m=100,
        )
        advice_low = self.sizer.size_position(
            item_id=2, buy_price=1_000_000, sell_price=1_050_000,
            score=70, win_rate=0.7, avg_win=50_000, avg_loss=30_000, volume_5m=2,
        )
        assert advice_high.max_hold_minutes > advice_low.max_hold_minutes


class TestEdgeCases:
    """Test edge cases and invalid inputs."""

    def setup_method(self):
        self.sizer = PositionSizer(bankroll=100_000_000)

    def test_zero_buy_price(self):
        """Zero buy price should not crash."""
        advice = self.sizer.size_position(
            item_id=1, buy_price=0, sell_price=100,
            score=50, win_rate=0.5, volume_5m=10,
        )
        assert advice.within_limits is False

    def test_zero_bankroll(self):
        """Zero bankroll should not crash."""
        sizer = PositionSizer(bankroll=0)
        advice = sizer.size_position(
            item_id=1, buy_price=1000, sell_price=1100,
            score=50, win_rate=0.5, volume_5m=10,
        )
        assert advice.within_limits is False

    def test_inverted_prices(self):
        """Sell price below buy price should give zero Kelly."""
        advice = self.sizer.size_position(
            item_id=1, buy_price=1_000_000, sell_price=900_000,
            score=50, win_rate=0.5, avg_win=0, avg_loss=100_000, volume_5m=10,
        )
        assert advice.kelly_fraction == 0
        assert advice.quantity == 0

    def test_reason_string_populated(self):
        """Reason field should always be populated."""
        advice = self.sizer.size_position(
            item_id=1, buy_price=1_000_000, sell_price=1_050_000,
            score=70, win_rate=0.7, avg_win=50_000, avg_loss=30_000, volume_5m=20,
        )
        assert len(advice.reason) > 0
