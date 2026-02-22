"""
Unit tests for backend.alerts.monitor — Phase 9.

Tests cover:
  • Margin alert trigger conditions
  • Volume spike detection
  • Trend reversal detection
  • Alert severity
  • Discord notifier (mocked)
  • CompositeNotifier fan-out
  • AlertMonitor state management
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.alerts.monitor import (
    Alert,
    AlertKind,
    AlertMonitor,
    CompositeNotifier,
    DiscordNotifier,
    _check_margin,
    _check_trend_reversal,
    _check_volume_spike,
)


# ---------------------------------------------------------------------------
# _check_margin
# ---------------------------------------------------------------------------

class TestCheckMargin:
    def test_fires_at_threshold(self):
        a = _check_margin(4151, "Abyssal whip", 100_000, threshold=100_000)
        assert a is not None
        assert a.kind == AlertKind.MARGIN

    def test_fires_above_threshold(self):
        a = _check_margin(4151, "Abyssal whip", 200_000, threshold=100_000)
        assert a is not None

    def test_no_fire_below_threshold(self):
        assert _check_margin(4151, "Abyssal whip", 50_000, threshold=100_000) is None

    def test_zero_profit(self):
        assert _check_margin(4151, "Abyssal whip", 0, threshold=1) is None

    def test_negative_profit(self):
        assert _check_margin(4151, "Abyssal whip", -1_000, threshold=1) is None


# ---------------------------------------------------------------------------
# _check_volume_spike
# ---------------------------------------------------------------------------

class TestCheckVolumeSpike:
    def test_fires_at_3x(self):
        a = _check_volume_spike(1, "Item", current_vol=300, avg_vol=100.0, spike_multiplier=3.0)
        assert a is not None
        assert a.kind == AlertKind.VOLUME_SPIKE

    def test_no_fire_below_3x(self):
        assert _check_volume_spike(1, "Item", current_vol=299, avg_vol=100.0, spike_multiplier=3.0) is None

    def test_zero_avg_no_fire(self):
        assert _check_volume_spike(1, "Item", current_vol=9999, avg_vol=0.0) is None

    def test_data_contains_ratio(self):
        a = _check_volume_spike(1, "Item", current_vol=600, avg_vol=100.0, spike_multiplier=3.0)
        assert "ratio" in a.data


# ---------------------------------------------------------------------------
# _check_trend_reversal
# ---------------------------------------------------------------------------

class TestCheckTrendReversal:
    @pytest.mark.parametrize("prev,new", [
        ("UP", "DOWN"),
        ("UP", "STRONG_DOWN"),
        ("STRONG_UP", "DOWN"),
        ("DOWN", "UP"),
        ("STRONG_DOWN", "STRONG_UP"),
    ])
    def test_fires_on_reversal(self, prev, new):
        a = _check_trend_reversal(1, "Item", prev, new, price=1_000_000)
        assert a is not None
        assert a.kind == AlertKind.TREND_REVERSAL

    @pytest.mark.parametrize("prev,new", [
        ("UP", "UP"),
        ("NEUTRAL", "NEUTRAL"),
        ("UP", "STRONG_UP"),
        ("DOWN", "STRONG_DOWN"),
    ])
    def test_no_fire_same_direction(self, prev, new):
        assert _check_trend_reversal(1, "Item", prev, new, price=1_000_000) is None

    def test_strong_reversal_is_warning(self):
        a = _check_trend_reversal(1, "Item", "STRONG_UP", "STRONG_DOWN", price=1_000_000)
        assert a is not None
        assert a.severity == "WARNING"


# ---------------------------------------------------------------------------
# Alert dataclass
# ---------------------------------------------------------------------------

class TestAlertModel:
    def test_default_timestamp(self):
        a = Alert(kind=AlertKind.MARGIN, item_id=1, item_name="x", message="y")
        assert isinstance(a.timestamp, datetime)

    def test_default_severity(self):
        a = Alert(kind=AlertKind.MARGIN, item_id=1, item_name="x", message="y")
        assert a.severity == "INFO"


# ---------------------------------------------------------------------------
# DiscordNotifier
# ---------------------------------------------------------------------------

class TestDiscordNotifier:
    @pytest.mark.asyncio
    async def test_empty_url_returns_false(self):
        n = DiscordNotifier("")
        a = Alert(kind=AlertKind.MARGIN, item_id=1, item_name="x", message="y")
        assert await n.send(a) is False

    @pytest.mark.asyncio
    @pytest.mark.skipif(not __import__("importlib").util.find_spec("httpx"), reason="httpx not installed")
    async def test_successful_send(self):
        n = DiscordNotifier("https://example.com/hook")
        a = Alert(kind=AlertKind.MARGIN, item_id=1, item_name="x", message="y")
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        ctx = AsyncMock()
        ctx.__aenter__ = AsyncMock(return_value=ctx)
        ctx.__aexit__ = AsyncMock(return_value=False)
        ctx.post = AsyncMock(return_value=mock_resp)
        with patch("backend.alerts.monitor.httpx.AsyncClient", return_value=ctx):
            result = await n.send(a)
        assert result is True


# ---------------------------------------------------------------------------
# CompositeNotifier
# ---------------------------------------------------------------------------

class TestCompositeNotifier:
    @pytest.mark.asyncio
    async def test_fans_out_to_all(self):
        n1 = MagicMock(spec=DiscordNotifier)
        n1.send = AsyncMock(return_value=True)
        n2 = MagicMock(spec=DiscordNotifier)
        n2.send = AsyncMock(return_value=False)
        composite = CompositeNotifier([n1, n2])
        a = Alert(kind=AlertKind.MARGIN, item_id=1, item_name="x", message="y")
        result = await composite.send(a)
        assert result is True
        n1.send.assert_awaited_once()
        n2.send.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_returns_false_if_all_fail(self):
        n1 = MagicMock(spec=DiscordNotifier)
        n1.send = AsyncMock(return_value=False)
        composite = CompositeNotifier([n1])
        a = Alert(kind=AlertKind.MARGIN, item_id=1, item_name="x", message="y")
        result = await composite.send(a)
        assert result is False


# ---------------------------------------------------------------------------
# AlertMonitor state
# ---------------------------------------------------------------------------

class TestAlertMonitorState:
    def test_empty_state_on_init(self):
        monitor = AlertMonitor()
        assert monitor._state == {}

    def test_state_set_manually(self):
        monitor = AlertMonitor()
        monitor._state[1] = {"trend": "UP", "last_alerted": 0.0, "avg_vol": 10.0}
        assert 1 in monitor._state

    def test_set_notifier(self):
        monitor = AlertMonitor()
        n = MagicMock(spec=DiscordNotifier)
        monitor.set_notifier(n)
        assert monitor._notifier is n
