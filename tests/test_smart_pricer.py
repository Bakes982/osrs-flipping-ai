"""Tests for the SmartPricer engine."""

import time
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

import pytest

from backend.smart_pricer import SmartPricer, Trend, PriceRecommendation, GE_TAX_RATE, GE_TAX_CAP
from backend.database import PriceSnapshot


def _snap(minutes_ago=0, buy=1000, sell=900, buy_vol=10, sell_vol=10, buy_time=None, sell_time=None):
    """Helper to create a PriceSnapshot-like object."""
    ts = datetime.utcnow() - timedelta(minutes=minutes_ago)
    now_unix = int(time.time())
    return PriceSnapshot(
        item_id=4151,
        timestamp=ts,
        instant_buy=buy,
        instant_sell=sell,
        buy_volume=buy_vol,
        sell_volume=sell_vol,
        buy_time=buy_time or (now_unix - int(minutes_ago * 60)),
        sell_time=sell_time or (now_unix - int(minutes_ago * 60)),
    )


@pytest.fixture
def pricer():
    return SmartPricer()


@pytest.fixture
def stable_snapshots():
    """30 snapshots over 5 minutes with stable prices."""
    return [_snap(i * 0.17, buy=100000, sell=98000) for i in range(30)]


@pytest.fixture
def uptrend_snapshots():
    """Snapshots showing an uptrend over 2h."""
    snaps = []
    for i in range(120):
        # Price increases by 10 GP per snapshot over 2 hours
        price = 100000 + i * 10
        snaps.append(_snap(120 - i, buy=price, sell=price - 2000))
    return snaps


class TestCalculateVWAP:
    def test_basic_vwap(self, pricer):
        snaps = [
            _snap(0, buy=100, sell=90, buy_vol=10, sell_vol=5),
            _snap(1, buy=110, sell=95, buy_vol=20, sell_vol=8),
            _snap(2, buy=105, sell=92, buy_vol=15, sell_vol=6),
        ]
        vwap = pricer.calculate_vwap(snaps, minutes=5, use_buy=True)
        assert vwap is not None
        # Weighted: (100*10 + 110*20 + 105*15) / (10+20+15) = 4775/45 ≈ 106.1
        assert 105 < vwap < 107

    def test_empty_snapshots(self, pricer):
        assert pricer.calculate_vwap([], minutes=5) is None

    def test_respects_time_window(self, pricer):
        snaps = [
            _snap(0, buy=100, buy_vol=10),
            _snap(10, buy=200, buy_vol=10),  # 10 minutes ago, outside 5m window
        ]
        vwap = pricer.calculate_vwap(snaps, minutes=5)
        assert vwap is not None
        assert vwap == 100.0  # Only the recent snapshot should count


class TestCalculateSMA:
    def test_basic_sma(self, pricer):
        snaps = [
            _snap(0, buy=100),
            _snap(1, buy=110),
            _snap(2, buy=120),
        ]
        sma = pricer.calculate_sma(snaps, minutes=5)
        assert sma == 110.0

    def test_empty_returns_none(self, pricer):
        assert pricer.calculate_sma([], minutes=5) is None


class TestDetectTrend:
    def test_neutral_on_stable(self, pricer, stable_snapshots):
        trend, momentum = pricer.detect_trend(stable_snapshots)
        assert trend == Trend.NEUTRAL
        assert abs(momentum) < 1  # Negligible momentum

    def test_uptrend(self, pricer, uptrend_snapshots):
        trend, momentum = pricer.detect_trend(uptrend_snapshots)
        assert trend in (Trend.UP, Trend.STRONG_UP)
        assert momentum > 0


class TestBollingerBands:
    def test_returns_none_insufficient_data(self, pricer):
        snaps = [_snap(0, buy=100)]
        upper, middle, lower = pricer.calculate_bollinger_bands(snaps)
        assert upper is None
        assert middle is None
        assert lower is None

    def test_basic_bands(self, pricer):
        """Use slightly varying prices so stdev > 0."""
        snaps = [_snap(i * 0.17, buy=100000 + (i % 5) * 100, sell=98000) for i in range(30)]
        upper, middle, lower = pricer.calculate_bollinger_bands(snaps)
        assert upper is not None
        assert middle is not None
        assert lower is not None
        assert lower < middle < upper


class TestGhostMarginValidation:
    def test_no_ghost_on_stable(self, pricer, stable_snapshots):
        v_buy, v_sell, was_ghost = pricer.validate_against_5m(
            100000, 98000, stable_snapshots
        )
        assert not was_ghost
        assert v_buy == 100000
        assert v_sell == 98000

    def test_detects_ghost_spike(self, pricer, stable_snapshots):
        # Instant buy is 20% higher than the stable 100K price
        v_buy, v_sell, was_ghost = pricer.validate_against_5m(
            120000, 98000, stable_snapshots
        )
        assert was_ghost
        assert v_buy < 120000  # Should be clamped down


class TestCalculateUndercut:
    def test_cheap_item(self, pricer):
        assert pricer.calculate_undercut(5000, volume_5m=100, is_buy=True) == 1

    def test_mid_item_high_volume(self, pricer):
        offset = pricer.calculate_undercut(1_000_000, volume_5m=100, is_buy=True)
        assert offset == 1000  # 0.1% of 1M

    def test_expensive_low_volume(self, pricer):
        offset = pricer.calculate_undercut(50_000_000, volume_5m=2, is_buy=True)
        assert offset <= 50000  # Capped


class TestWaterfallDetection:
    def test_no_waterfall_on_stable(self, pricer, stable_snapshots):
        assert not pricer.detect_waterfall(stable_snapshots)

    def test_detects_waterfall(self, pricer):
        snaps = []
        # Create a crash: 4 buckets of 5 minutes each, each dropping >2%
        for bucket in range(4):
            base_price = int(100000 * (0.97 ** bucket))  # 3% drop per bucket
            for i in range(6):  # 6 snapshots per bucket (~1 per 10s over 1 min)
                ts = datetime.utcnow() - timedelta(minutes=(3 - bucket) * 5 + i * 0.17)
                snaps.append(PriceSnapshot(
                    item_id=1,
                    timestamp=ts,
                    instant_buy=base_price,
                    instant_sell=base_price - 2000,
                    buy_volume=10,
                    sell_volume=10,
                ))
        snaps.sort(key=lambda s: s.timestamp)
        assert pricer.detect_waterfall(snaps)


class TestVolumeLiveness:
    def test_healthy_volume(self, pricer, stable_snapshots):
        assert pricer.check_volume_liveness(stable_snapshots) == "HEALTHY"

    def test_dead_volume_trap(self, pricer):
        older = [_snap(i, buy_vol=20, sell_vol=20) for i in range(10, 4, -1)]
        recent = [_snap(i, buy_vol=0, sell_vol=0) for i in range(3, 0, -1)]
        snaps = older + recent
        assert pricer.check_volume_liveness(snaps) == "DEAD_VOLUME_TRAP"


class TestSanityChecks:
    def test_fresh_normal_data(self, pricer):
        now = int(time.time())
        stale, anomalous, conf = pricer.check_sanity(
            instant_buy=100000,
            instant_sell=98000,
            buy_time=now - 30,
            sell_time=now - 30,
            volume_5m=50,
            historical_spread_pct=2.0,
        )
        assert not stale
        assert not anomalous
        assert conf > 0.8

    def test_stale_data(self, pricer):
        now = int(time.time())
        stale, _, conf = pricer.check_sanity(
            instant_buy=100000,
            instant_sell=98000,
            buy_time=now - 2000,  # ~33 minutes old
            sell_time=now - 2000,
            volume_5m=1,
            historical_spread_pct=None,
        )
        assert stale
        assert conf < 0.5


class TestClampPrices:
    def test_neutral_no_change(self, pricer):
        clamped = pricer.clamp_buy_price(
            instant_sell=100000,
            trend=Trend.NEUTRAL,
            vwap_5m=100500.0,
            vwap_30m=100200.0,
            momentum=0.0,
            spread=2000,
        )
        assert clamped == 100000

    def test_uptrend_pays_more(self, pricer):
        clamped = pricer.clamp_buy_price(
            instant_sell=100000,
            trend=Trend.UP,
            vwap_5m=101000.0,
            vwap_30m=100500.0,
            momentum=5.0,
            spread=2000,
        )
        assert clamped >= 100000


class TestGETax:
    def test_tax_rate(self):
        sell_price = 1_000_000
        tax = int(min(sell_price * GE_TAX_RATE, GE_TAX_CAP))
        assert tax == 20000  # 2% of 1M

    def test_tax_cap(self):
        sell_price = 500_000_000
        tax = int(min(sell_price * GE_TAX_RATE, GE_TAX_CAP))
        assert tax == GE_TAX_CAP  # Capped at 5M
