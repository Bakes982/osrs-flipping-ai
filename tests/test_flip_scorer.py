"""Tests for the FlipScorer composite scoring system."""

import time
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

import pytest

from backend.database import PriceSnapshot, FlipHistory
from backend.flip_scorer import FlipScorer, FlipScore, score_opportunities
from backend.smart_pricer import Trend, PriceRecommendation


def _snap(minutes_ago=0, buy=1000000, sell=980000, buy_vol=20, sell_vol=15):
    now_unix = int(time.time())
    return PriceSnapshot(
        item_id=4151,
        timestamp=datetime.utcnow() - timedelta(minutes=minutes_ago),
        instant_buy=buy,
        instant_sell=sell,
        buy_volume=buy_vol,
        sell_volume=sell_vol,
        buy_time=now_unix - int(minutes_ago * 60),
        sell_time=now_unix - int(minutes_ago * 60),
    )


def _flip(profit, minutes_ago=0):
    now = datetime.utcnow()
    return FlipHistory(
        item_id=4151,
        item_name="Abyssal whip",
        buy_price=1000000,
        sell_price=1000000 + profit + 20000,  # + tax estimate
        quantity=1,
        gross_profit=profit + 20000,
        tax=20000,
        net_profit=profit,
        margin_pct=abs(profit) / 1000000 * 100,
        buy_time=now - timedelta(minutes=minutes_ago + 10),
        sell_time=now - timedelta(minutes=minutes_ago),
        duration_seconds=600,
    )


@pytest.fixture
def scorer():
    return FlipScorer()


@pytest.fixture
def good_snapshots():
    """30 snapshots with healthy spread, volume, and freshness."""
    return [_snap(i * 0.17, buy=1000000, sell=980000, buy_vol=25, sell_vol=20) for i in range(30)]


@pytest.fixture
def good_flips():
    """10 profitable flips."""
    return [_flip(15000, minutes_ago=i * 60) for i in range(10)]


class TestFlipScoreDataclass:
    def test_default_values(self):
        fs = FlipScore(item_id=4151)
        assert fs.total_score == 0.0
        assert fs.vetoed is False
        assert fs.veto_reasons == []
        assert fs.item_name == ""


class TestSpreadScoring:
    def test_good_spread_high_volume(self, scorer):
        fs = FlipScore(item_id=1, spread_pct=2.0)
        rec = PriceRecommendation(item_id=1, timestamp=datetime.utcnow(), volume_5m=50)
        score = scorer._score_spread(fs, rec)
        assert score >= 75

    def test_too_wide_spread(self, scorer):
        fs = FlipScore(item_id=1, spread_pct=12.0)
        rec = PriceRecommendation(item_id=1, timestamp=datetime.utcnow(), volume_5m=5)
        score = scorer._score_spread(fs, rec)
        assert score <= 40

    def test_zero_spread(self, scorer):
        fs = FlipScore(item_id=1, spread_pct=0)
        rec = PriceRecommendation(item_id=1, timestamp=datetime.utcnow(), volume_5m=50)
        score = scorer._score_spread(fs, rec)
        assert score == 0


class TestVolumeScoring:
    def test_high_volume(self, scorer):
        fs = FlipScore(item_id=1)
        rec = PriceRecommendation(item_id=1, timestamp=datetime.utcnow(), volume_5m=100)
        snaps = [_snap(0)]
        score = scorer._score_volume(fs, rec, snaps)
        assert score == 100

    def test_zero_volume(self, scorer):
        fs = FlipScore(item_id=1)
        rec = PriceRecommendation(item_id=1, timestamp=datetime.utcnow(), volume_5m=0)
        snaps = [_snap(0)]
        score = scorer._score_volume(fs, rec, snaps)
        assert score == 0


class TestFreshnessScoring:
    def test_fresh_data(self, scorer):
        now_unix = int(time.time())
        snaps = [PriceSnapshot(
            item_id=1,
            timestamp=datetime.utcnow(),
            instant_buy=1000,
            instant_sell=900,
            buy_time=now_unix - 30,
            sell_time=now_unix - 30,
        )]
        rec = PriceRecommendation(item_id=1, timestamp=datetime.utcnow())
        score = scorer._score_freshness(rec, snaps)
        assert score >= 90

    def test_stale_data(self, scorer):
        now_unix = int(time.time())
        snaps = [PriceSnapshot(
            item_id=1,
            timestamp=datetime.utcnow(),
            instant_buy=1000,
            instant_sell=900,
            buy_time=now_unix - 3600,  # 1 hour old
            sell_time=now_unix - 3600,
        )]
        rec = PriceRecommendation(item_id=1, timestamp=datetime.utcnow())
        score = scorer._score_freshness(rec, snaps)
        assert score <= 20


class TestTrendScoring:
    def test_neutral_scores_high(self, scorer):
        rec = PriceRecommendation(item_id=1, timestamp=datetime.utcnow())
        rec.trend = Trend.NEUTRAL
        score = scorer._score_trend(rec)
        assert score >= 80

    def test_strong_down_scores_low(self, scorer):
        rec = PriceRecommendation(item_id=1, timestamp=datetime.utcnow())
        rec.trend = Trend.STRONG_DOWN
        score = scorer._score_trend(rec)
        assert score <= 30


class TestHistoryScoring:
    def test_no_history_neutral(self, scorer):
        fs = FlipScore(item_id=1)
        score = scorer._score_history(fs, [])
        assert score == 50

    def test_good_history(self, scorer, good_flips):
        fs = FlipScore(item_id=1)
        score = scorer._score_history(fs, good_flips)
        assert score >= 85
        assert fs.win_rate == 1.0
        assert fs.total_flips == 10


class TestStabilityScoring:
    def test_stable_prices(self, scorer, good_snapshots):
        score = scorer._score_stability(good_snapshots)
        assert score >= 70

    def test_volatile_prices(self, scorer):
        snaps = []
        for i in range(30):
            # Wildly oscillating prices
            price = 100000 if i % 2 == 0 else 120000
            snaps.append(_snap(i * 0.17, buy=price, sell=price - 5000))
        score = scorer._score_stability(snaps)
        assert score <= 50


class TestVetoes:
    def test_veto_negative_profit(self, scorer):
        fs = FlipScore(item_id=1, spread=100, spread_pct=0.01)
        rec = PriceRecommendation(
            item_id=1, timestamp=datetime.utcnow(),
            expected_profit=-5000, volume_5m=10,
        )
        snaps = [_snap(0)]
        scorer._check_vetoes(fs, rec, snaps)
        assert fs.vetoed
        assert any("Unprofitable" in r for r in fs.veto_reasons)

    def test_veto_zero_volume(self, scorer):
        fs = FlipScore(item_id=1, spread=5000, spread_pct=2.0)
        rec = PriceRecommendation(
            item_id=1, timestamp=datetime.utcnow(),
            expected_profit=5000, volume_5m=0,
            instant_buy=100000, instant_sell=98000,
        )
        snaps = [_snap(0, buy_vol=0, sell_vol=0)]
        scorer._check_vetoes(fs, rec, snaps)
        assert fs.vetoed
        assert any("Zero volume" in r for r in fs.veto_reasons)

    def test_veto_wide_spread(self, scorer):
        fs = FlipScore(item_id=1, spread=20000, spread_pct=20.0)
        rec = PriceRecommendation(
            item_id=1, timestamp=datetime.utcnow(),
            expected_profit=15000, volume_5m=10,
        )
        snaps = [_snap(0)]
        scorer._check_vetoes(fs, rec, snaps)
        assert fs.vetoed
        assert any("Spread too wide" in r for r in fs.veto_reasons)

    def test_veto_inverted_spread(self, scorer):
        fs = FlipScore(item_id=1, spread=-100, spread_pct=-0.1)
        rec = PriceRecommendation(
            item_id=1, timestamp=datetime.utcnow(),
            expected_profit=-100, volume_5m=10,
        )
        snaps = [_snap(0)]
        scorer._check_vetoes(fs, rec, snaps)
        assert fs.vetoed


class TestWeights:
    def test_weights_sum_to_one(self):
        total = sum(FlipScorer.WEIGHTS.values())
        assert abs(total - 1.0) < 0.001


class TestBuildReason:
    def test_strong_flip(self, scorer):
        fs = FlipScore(
            item_id=1, total_score=75,
            expected_profit=50000, expected_profit_pct=2.5,
            win_rate=0.85, total_flips=10,
        )
        rec = PriceRecommendation(
            item_id=1, timestamp=datetime.utcnow(),
            volume_5m=30, trend=Trend.NEUTRAL,
        )
        reason = scorer._build_reason(fs, rec)
        assert "STRONG FLIP" in reason
        assert "High liquidity" in reason
