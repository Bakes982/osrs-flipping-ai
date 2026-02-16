"""Tests for the ML Feature Engine."""

import math
from datetime import datetime, timedelta

import pytest

from backend.database import PriceSnapshot, FlipHistory
from backend.ml.feature_engine import FeatureEngine


def _snap(minutes_ago=0, buy=100000, sell=98000, buy_vol=10, sell_vol=10):
    return PriceSnapshot(
        item_id=4151,
        timestamp=datetime.utcnow() - timedelta(minutes=minutes_ago),
        instant_buy=buy,
        instant_sell=sell,
        buy_volume=buy_vol,
        sell_volume=sell_vol,
    )


def _flip(profit=5000, minutes_ago=0):
    now = datetime.utcnow()
    return FlipHistory(
        item_id=4151,
        item_name="Abyssal whip",
        buy_price=100000,
        sell_price=100000 + profit + 2000,
        quantity=1,
        gross_profit=profit + 2000,
        tax=2000,
        net_profit=profit,
        margin_pct=profit / 100000 * 100,
        buy_time=now - timedelta(minutes=minutes_ago + 10),
        sell_time=now - timedelta(minutes=minutes_ago),
        duration_seconds=600,
    )


@pytest.fixture
def engine():
    return FeatureEngine()


@pytest.fixture
def snapshots():
    """60 snapshots over 10 minutes with slightly increasing prices."""
    snaps = []
    for i in range(60):
        price = 100000 + i * 5
        snaps.append(_snap(10 - i * 0.17, buy=price, sell=price - 2000))
    return snaps


@pytest.fixture
def flips():
    return [_flip(5000, minutes_ago=i * 60) for i in range(5)]


class TestFeatureNames:
    def test_feature_names_returns_list(self):
        names = FeatureEngine.feature_names()
        assert isinstance(names, list)
        assert len(names) == 35

    def test_all_features_computed(self, engine, snapshots, flips):
        features = engine.compute_features(4151, snapshots, flips)
        expected_names = FeatureEngine.feature_names()
        for name in expected_names:
            assert name in features, f"Missing feature: {name}"


class TestPriceFeatures:
    def test_current_price(self, engine, snapshots):
        features = engine._price_features(snapshots)
        assert features["current_price"] > 0

    def test_sma_deviations(self, engine, snapshots):
        features = engine._price_features(snapshots)
        # With slightly increasing prices, current should be above SMA
        assert "price_vs_sma_5m" in features

    def test_bollinger_position(self, engine, snapshots):
        features = engine._price_features(snapshots)
        assert 0 <= features["bollinger_position"] <= 1

    def test_momentum(self, engine, snapshots):
        features = engine._price_features(snapshots)
        assert "momentum_1x" in features
        assert "momentum_2x" in features
        assert "momentum_4x" in features

    def test_spread_pct(self, engine, snapshots):
        features = engine._price_features(snapshots)
        assert features["spread_pct"] != 0  # Non-zero with different buy/sell

    def test_empty_snapshots(self, engine):
        features = engine._price_features([])
        assert features["current_price"] == 0.0


class TestVolumeFeatures:
    def test_volume_ratio(self, engine, snapshots):
        features = engine._volume_features(snapshots)
        assert "volume_ratio" in features
        assert features["volume_ratio"] >= 0

    def test_buy_sell_ratio(self, engine, snapshots):
        features = engine._volume_features(snapshots)
        # Equal buy/sell volumes -> ratio should be ~0.5
        assert 0.4 <= features["buy_sell_ratio"] <= 0.6

    def test_volume_price_divergence(self, engine, snapshots):
        features = engine._volume_features(snapshots)
        assert "volume_price_divergence" in features


class TestTechnicalFeatures:
    def test_rsi(self, engine, snapshots):
        features = engine._technical_features(snapshots)
        assert 0 <= features["rsi_14"] <= 100

    def test_z_score(self, engine, snapshots):
        features = engine._technical_features(snapshots)
        assert "z_score" in features

    def test_macd(self, engine, snapshots):
        features = engine._technical_features(snapshots)
        assert "macd" in features
        assert "macd_signal" in features
        assert "macd_histogram" in features

    def test_stochastic(self, engine, snapshots):
        features = engine._technical_features(snapshots)
        assert 0 <= features["stochastic_k"] <= 100
        assert 0 <= features["stochastic_d"] <= 100


class TestTemporalFeatures:
    def test_cyclical_encoding(self, engine):
        features = engine._temporal_features()
        assert -1 <= features["hour_sin"] <= 1
        assert -1 <= features["hour_cos"] <= 1
        assert -1 <= features["dow_sin"] <= 1
        assert -1 <= features["dow_cos"] <= 1

    def test_is_weekend(self, engine):
        features = engine._temporal_features()
        assert features["is_weekend"] in (0.0, 1.0)

    def test_minutes_to_peak(self, engine):
        features = engine._temporal_features()
        assert features["minutes_to_peak"] >= 0


class TestHistoricalFeatures:
    def test_no_flips(self, engine):
        features = engine._historical_features([])
        assert features["win_rate"] == 0.0
        assert features["avg_profit_per_flip"] == 0.0

    def test_with_flips(self, engine, flips):
        features = engine._historical_features(flips)
        assert features["win_rate"] == 1.0  # All profitable
        assert features["avg_profit_per_flip"] == 5000.0
        assert features["avg_flip_duration"] == 600.0


class TestRSI:
    def test_insufficient_data(self, engine):
        prices = [100.0, 101.0]
        assert engine._rsi(prices, 14) == 50.0

    def test_all_gains(self, engine):
        prices = list(range(100, 120))
        rsi = engine._rsi(prices, 14)
        assert rsi == 100.0

    def test_all_losses(self, engine):
        prices = list(range(120, 100, -1))
        rsi = engine._rsi(prices, 14)
        assert rsi == 0.0


class TestEMA:
    def test_basic_ema(self, engine):
        values = [10.0, 11.0, 12.0, 13.0, 14.0, 15.0]
        ema = engine._ema(values, 3)
        assert len(ema) > 0
        # EMA should be smoothed and lag behind the raw data
        assert ema[-1] < 15.0

    def test_empty_values(self, engine):
        assert engine._ema([], 3) == []


class TestATR:
    def test_atr_computation(self, engine, snapshots):
        atr = engine._atr(snapshots, 14)
        assert atr >= 0  # Should be non-negative

    def test_insufficient_data(self, engine):
        snaps = [_snap(0)]
        assert engine._atr(snaps, 14) == 0.0
