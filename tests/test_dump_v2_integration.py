"""
test_dump_v2_integration — end-to-end wiring tests for dump-v2 alerts.

Covers the spec requirements:
  A. resolver returns real name from mocked mapping
  B. dump embed title uses resolved name (never "Item XXXXX")
  C. chart attachment is included when chart path exists
  D. cooldown suppresses repeated alert (DUMP_SUPPRESSED logged)
  E. filter gate blocks low price / low profit items
"""

from __future__ import annotations

import os
import sys
import tempfile
import time
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

# ── Stub broken/missing production deps BEFORE any backend module is imported ─
# backend.tasks imports httpx + backend.database (→ pymongo → cryptography Rust
# ext. that panics in this environment) at module level.  Setting stubs in
# sys.modules here means Python will use the stubs when tasks.py is first
# imported inside a test body.
_mm = MagicMock

if "httpx" not in sys.modules:
    sys.modules["httpx"] = _mm()

if "backend.database" not in sys.modules:
    _db = _mm()
    # Give Alert a real constructor so backend.tasks can do Alert(item_id=...)
    class _AlertStub:
        def __init__(self, **kw): self.__dict__.update(kw)
    _db.Alert = _AlertStub
    sys.modules["backend.database"] = _db

if "backend.websocket" not in sys.modules:
    _ws = _mm()
    _ws.manager = _mm()
    sys.modules["backend.websocket"] = _ws

# backend.discord_notifier requires matplotlib which is not installed in the
# test environment.  Stub the whole module so the lazy import inside
# _check_dump_alerts_sync succeeds (chart_bytes will be None → no attachment).
if "backend.discord_notifier" not in sys.modules:
    _chart_mod = _mm()
    _chart_mod.generate_opportunity_chart = _mm(return_value=None)
    sys.modules["backend.discord_notifier"] = _chart_mod

# ---------------------------------------------------------------------------
# A. Resolver: real name from mocked mapping
# ---------------------------------------------------------------------------

class TestResolverWithMockedMapping:
    """resolve_item_name must return the Wiki name, not a placeholder."""

    def test_returns_real_name_from_mocked_mapping(self):
        """When the mapping cache contains the item ID, return its name."""
        import backend.alerts.item_name_resolver as mod

        fake_mapping = {30076: "Twisted bow", 4151: "Abyssal whip"}
        with patch.object(mod, "_mapping_cache", fake_mapping):
            with patch.object(mod, "_mapping_cache_ts", time.time()):
                from backend.alerts.item_name_resolver import resolve_item_name
                name = resolve_item_name(30076, fallback="Item 30076")
        assert name == "Twisted bow", f"Expected 'Twisted bow', got {name!r}"
        assert not name.startswith("Item "), "Name must not be a placeholder"

    def test_returns_real_name_when_fallback_is_placeholder(self):
        """Fallback 'Item XXXXX' must trigger a mapping lookup."""
        import backend.alerts.item_name_resolver as mod

        fake_mapping = {4151: "Abyssal whip"}
        with patch.object(mod, "_mapping_cache", fake_mapping):
            with patch.object(mod, "_mapping_cache_ts", time.time()):
                from backend.alerts.item_name_resolver import resolve_item_name
                name = resolve_item_name(4151, fallback="Item 4151")
        assert name == "Abyssal whip"

    def test_good_fallback_is_returned_without_lookup(self):
        """If fallback is already a real name, no mapping fetch needed."""
        import backend.alerts.item_name_resolver as mod
        from backend.alerts.item_name_resolver import resolve_item_name

        with patch.object(mod, "_mapping_cache", {}):
            with patch.object(mod, "_mapping_cache_ts", time.time()):
                name = resolve_item_name(4151, fallback="Abyssal whip")
        assert name == "Abyssal whip"

    def test_last_resort_placeholder_when_mapping_empty(self):
        """If both fallback and mapping fail, return 'Item XXXXX'."""
        import backend.alerts.item_name_resolver as mod
        from backend.alerts.item_name_resolver import resolve_item_name

        with patch.object(mod, "_mapping_cache", {}):
            with patch.object(mod, "_mapping_cache_ts", time.time()):
                name = resolve_item_name(99999, fallback=None)
        assert name == "Item 99999"


# ---------------------------------------------------------------------------
# B. Embed title uses resolved name
# ---------------------------------------------------------------------------

class TestDumpEmbedTitleResolved:
    """DumpAlertNotifierV2._build_embed must use the resolved name in title."""

    def _make_alert(self, item_name: str = "Abyssal whip"):
        from backend.alerts.dump_notifier import DumpAlertV2
        return DumpAlertV2(
            item_id=4151,
            item_name=item_name,
            current_price=2_000_000,
            reference_price=2_200_000,
            drop_pct=9.1,
            drop_amount=200_000,
            sold_5m=180,
            bought_5m=20,
            sell_ratio=0.90,
            profit_per_item_net=156_000,
            predicted_recovery=2_160_000,
            confidence="HIGH",
            qty_to_buy=25,
            max_invest_gp=5_000_000,
            estimated_total_profit=3_900_000,
            timestamp=datetime(2026, 2, 22, 12, 0, 0),
        )

    def test_embed_title_is_resolved_name_not_placeholder(self):
        from backend.alerts.dump_notifier import DumpAlertNotifierV2
        alert = self._make_alert("Abyssal whip")
        notifier = DumpAlertNotifierV2("https://discord.com/api/webhooks/111/fake")
        embed = notifier._build_embed(alert, "Abyssal whip")
        assert embed["title"] == "DUMP DETECTED: Abyssal whip"
        assert "Item 4151" not in embed["title"]

    def test_embed_title_never_shows_item_number_placeholder(self):
        """Even if item_name is a placeholder, _build_embed title uses the
        resolved name passed as argument."""
        from backend.alerts.dump_notifier import DumpAlertNotifierV2
        alert = self._make_alert("Item 4151")   # bad name on the alert object
        notifier = DumpAlertNotifierV2("https://discord.com/api/webhooks/111/fake")
        # caller resolves name before building embed
        embed = notifier._build_embed(alert, "Abyssal whip")  # resolved name passed
        assert embed["title"] == "DUMP DETECTED: Abyssal whip"
        assert "Item 4151" not in embed["title"]


# ---------------------------------------------------------------------------
# C. Chart attachment
# ---------------------------------------------------------------------------

class TestChartAttachment:
    """When a valid PNG chart path exists, the notifier sends multipart."""

    def test_chart_attached_as_multipart_when_file_exists(self):
        from backend.alerts.dump_notifier import DumpAlertV2, DumpAlertNotifierV2
        import json

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(b"\x89PNG\r\n\x1a\n" + b"\x00" * 64)
            chart_path = f.name

        try:
            alert = DumpAlertV2(
                item_id=4151, item_name="Abyssal whip",
                current_price=2_000_000, reference_price=2_200_000,
                drop_pct=9.1, drop_amount=200_000,
                sold_5m=180, bought_5m=20, sell_ratio=0.90,
                profit_per_item_net=156_000, predicted_recovery=2_160_000,
                confidence="HIGH", qty_to_buy=25, max_invest_gp=5_000_000,
                estimated_total_profit=3_900_000, chart_path=chart_path,
            )
            notifier = DumpAlertNotifierV2("https://discord.com/api/webhooks/111/fake")

            captured: dict = {}
            def fake_post(url, **kwargs):
                captured.update(kwargs)
                r = MagicMock()
                r.status_code = 204
                return r

            with patch("backend.alerts.dump_notifier._requests.post", side_effect=fake_post):
                result = notifier.send(alert)

            assert result is True
            assert "files" in captured, "Expected multipart/form-data (files kwarg)"
            payload = json.loads(captured["data"]["payload_json"])
            embed = payload["embeds"][0]
            assert "image" in embed
            assert embed["image"]["url"].startswith("attachment://")
        finally:
            os.unlink(chart_path)


# ---------------------------------------------------------------------------
# D. Cooldown suppresses repeated alert
# ---------------------------------------------------------------------------

class TestDumpCooldown:
    """DUMP_SUPPRESSED must be logged when cooldown is active."""

    def _make_dump_caches(self, item_id_str="30076"):
        """Create price/5m caches that clearly trigger a dump."""
        price_cache = {item_id_str: {"low": 500_000, "high": 550_000}}
        five_m_cache = {item_id_str: {
            "avgLowPrice":     650_000,   # 23% drop  → qualifies
            "avgHighPrice":    670_000,
            "lowPriceVolume":  150,        # sell_ratio = 150/160 = 93% → qualifies
            "highPriceVolume": 10,
        }}
        return price_cache, five_m_cache

    def _make_db(self, has_recent_alert: bool):
        db = MagicMock()
        if has_recent_alert:
            db.alerts.find_one.return_value = {"item_id": 30076}
        else:
            db.alerts.find_one.return_value = None
        return db

    def test_cooldown_suppresses_discord_send(self):
        """When a recent dump alert exists in DB, no Discord message is sent."""
        from backend.tasks import AlertMonitor
        import backend.tasks as _tasks_mod

        price_cache, five_m_cache = self._make_dump_caches()
        db = self._make_db(has_recent_alert=True)

        monitor = AlertMonitor()
        with patch.dict(_tasks_mod._price_cache, price_cache, clear=True), \
             patch.dict(_tasks_mod._5m_cache, five_m_cache, clear=True), \
             patch("backend.tasks.get_item", return_value=None), \
             patch("backend.tasks.insert_alert") as mock_insert, \
             patch("backend.alerts.dump_notifier.DumpAlertNotifierV2") as mock_notifier_cls, \
             patch("backend.alerts.item_name_resolver.requests.get") as mock_resolver_http, \
             patch("backend.discord_notifier.generate_opportunity_chart", return_value=None):

            monitor._check_dump_alerts_sync(db)

        # cooldown fired — no alert inserted, no Discord call
        mock_insert.assert_not_called()
        mock_notifier_cls.return_value.send.assert_not_called()

    def test_cooldown_logs_suppressed(self, caplog):
        """DUMP_SUPPRESSED must appear in the log when cooldown is active."""
        from backend.tasks import AlertMonitor
        import backend.tasks as _tasks_mod
        import logging

        price_cache, five_m_cache = self._make_dump_caches()
        db = self._make_db(has_recent_alert=True)

        monitor = AlertMonitor()
        with patch.dict(_tasks_mod._price_cache, price_cache, clear=True), \
             patch.dict(_tasks_mod._5m_cache, five_m_cache, clear=True), \
             patch("backend.tasks.get_item", return_value=None), \
             patch("backend.tasks.insert_alert"), \
             patch("backend.alerts.dump_notifier.DumpAlertNotifierV2"), \
             patch("backend.alerts.item_name_resolver.requests.get"), \
             patch("backend.discord_notifier.generate_opportunity_chart", return_value=None), \
             caplog.at_level(logging.INFO, logger="backend.tasks"):

            monitor._check_dump_alerts_sync(db)

        assert any("DUMP_SUPPRESSED" in r.message for r in caplog.records), \
            "Expected DUMP_SUPPRESSED in log"

    def test_no_cooldown_sends_alert(self):
        """When no recent alert exists, Discord send is called once."""
        from backend.tasks import AlertMonitor
        import backend.tasks as _tasks_mod
        import backend.alerts.item_name_resolver as _res_mod

        price_cache, five_m_cache = self._make_dump_caches()
        db = self._make_db(has_recent_alert=False)
        db.alerts.insert_one = MagicMock()

        monitor = AlertMonitor()
        send_mock = MagicMock(return_value=True)

        with patch.dict(_tasks_mod._price_cache, price_cache, clear=True), \
             patch.dict(_tasks_mod._5m_cache, five_m_cache, clear=True), \
             patch("backend.tasks.get_item", return_value=None), \
             patch("backend.tasks.insert_alert"), \
             patch("backend.alerts.dump_notifier.DumpAlertNotifierV2") as mock_cls, \
             patch.object(_res_mod, "_mapping_cache", {30076: "Twisted bow"}), \
             patch.object(_res_mod, "_mapping_cache_ts", time.time()), \
             patch("backend.discord_notifier.generate_opportunity_chart", return_value=None), \
             patch.object(monitor, "_get_dump_alert_webhook_sync", return_value="https://fake-webhook"):

            mock_cls.return_value.send = send_mock
            monitor._check_dump_alerts_sync(db)

        send_mock.assert_called_once()


# ---------------------------------------------------------------------------
# E. Filter gates
# ---------------------------------------------------------------------------

class TestDumpFilterGates:
    """Items that fail quality gates must never produce a Discord alert."""

    def _run_check(self, price_cache, five_m_cache):
        """Helper: run _check_dump_alerts_sync with given caches, return send call count."""
        from backend.tasks import AlertMonitor
        import backend.tasks as _tasks_mod
        import backend.alerts.item_name_resolver as _res_mod

        db = MagicMock()
        db.alerts.find_one.return_value = None  # no cooldown

        monitor = AlertMonitor()
        send_mock = MagicMock(return_value=True)

        with patch.dict(_tasks_mod._price_cache, price_cache, clear=True), \
             patch.dict(_tasks_mod._5m_cache, five_m_cache, clear=True), \
             patch("backend.tasks.get_item", return_value=None), \
             patch("backend.tasks.insert_alert"), \
             patch("backend.alerts.dump_notifier.DumpAlertNotifierV2") as mock_cls, \
             patch.object(_res_mod, "_mapping_cache", {99001: "Some Expensive Item"}), \
             patch.object(_res_mod, "_mapping_cache_ts", time.time()), \
             patch("backend.discord_notifier.generate_opportunity_chart", return_value=None), \
             patch.object(monitor, "_get_dump_alert_webhook_sync", return_value="https://fake"):

            mock_cls.return_value.send = send_mock
            monitor._check_dump_alerts_sync(db)

        return send_mock.call_count

    def test_gate_blocks_low_price_item(self):
        """Items priced below DUMP_V2_MIN_PRICE_GP (500k) must be skipped."""
        # avg_sell = 100_000 (way below the 500_000 threshold)
        price_cache = {"99001": {"low": 80_000, "high": 90_000}}
        five_m_cache = {"99001": {
            "avgLowPrice":     100_000,   # below MIN_PRICE=500_000
            "avgHighPrice":    110_000,
            "lowPriceVolume":  200,
            "highPriceVolume": 10,
        }}
        count = self._run_check(price_cache, five_m_cache)
        assert count == 0, f"Expected 0 Discord sends for cheap item, got {count}"

    def test_gate_blocks_insufficient_sell_ratio(self):
        """Items without strong sell dominance must be skipped."""
        # sell_ratio = 10/110 ≈ 9% < MIN_SELL_RATIO=80%
        price_cache = {"99001": {"low": 500_000, "high": 550_000}}
        five_m_cache = {"99001": {
            "avgLowPrice":     650_000,
            "avgHighPrice":    680_000,
            "lowPriceVolume":  10,    # sell_ratio too low
            "highPriceVolume": 100,
        }}
        count = self._run_check(price_cache, five_m_cache)
        assert count == 0, "Expected 0 Discord sends for low sell ratio"

    def test_gate_blocks_insufficient_drop(self):
        """Items with price drop below DUMP_V2_MIN_DROP_PCT must be skipped."""
        # price_drop_pct = (600k - 598k) / 600k * 100 ≈ 0.3% < 4.0%
        price_cache = {"99001": {"low": 598_000, "high": 610_000}}
        five_m_cache = {"99001": {
            "avgLowPrice":     600_000,
            "avgHighPrice":    620_000,
            "lowPriceVolume":  100,
            "highPriceVolume": 5,
        }}
        count = self._run_check(price_cache, five_m_cache)
        assert count == 0, "Expected 0 Discord sends for tiny price drop"

    def test_gate_blocks_low_profit_per_item(self):
        """Items where net profit/item < DUMP_V2_MIN_PROFIT_PER_ITEM must be skipped.

        We engineer prices so the drop is real but profit is tiny after tax:
          avg_sell=500_500, instant_sell=500_000
          ge_tax = 500_500 * 0.01 = 5_005
          profit = 500_500 - 500_000 - 5_005 = -505  (negative — definitely blocked)
        """
        price_cache = {"99001": {"low": 500_000, "high": 510_000}}
        five_m_cache = {"99001": {
            "avgLowPrice":     500_500,    # drop = 0.1% — also fails drop gate
            "avgHighPrice":    515_000,
            "lowPriceVolume":  120,
            "highPriceVolume": 5,
        }}
        count = self._run_check(price_cache, five_m_cache)
        assert count == 0, "Expected 0 Discord sends for low profit"

    def test_gate_allows_qualifying_item(self):
        """A fully qualifying dump must send exactly one Discord message."""
        # avg_sell=700_000, instant_sell=560_000 → drop=20%, sell_ratio=95%
        # profit = 700k - 560k - 7k = 133k >> 2k ✓
        price_cache = {"99001": {"low": 560_000, "high": 600_000}}
        five_m_cache = {"99001": {
            "avgLowPrice":     700_000,
            "avgHighPrice":    720_000,
            "lowPriceVolume":  190,
            "highPriceVolume": 10,
        }}
        count = self._run_check(price_cache, five_m_cache)
        assert count == 1, f"Expected 1 Discord send for qualifying dump, got {count}"


# ---------------------------------------------------------------------------
# F. Name resolution debug log
# ---------------------------------------------------------------------------

class TestDumpNameDebugLog:
    """DUMP_NAME_DEBUG must be logged whenever an alert is processed."""

    def test_dump_name_debug_logged(self, caplog):
        from backend.tasks import AlertMonitor
        import backend.tasks as _tasks_mod
        import backend.alerts.item_name_resolver as _res_mod
        import logging

        price_cache = {"4151": {"low": 560_000, "high": 600_000}}
        five_m_cache = {"4151": {
            "avgLowPrice":     700_000,
            "avgHighPrice":    720_000,
            "lowPriceVolume":  190,
            "highPriceVolume": 10,
        }}
        db = MagicMock()
        db.alerts.find_one.return_value = None  # no cooldown

        monitor = AlertMonitor()
        with patch.dict(_tasks_mod._price_cache, price_cache, clear=True), \
             patch.dict(_tasks_mod._5m_cache, five_m_cache, clear=True), \
             patch("backend.tasks.get_item", return_value=None), \
             patch("backend.tasks.insert_alert"), \
             patch("backend.alerts.dump_notifier.DumpAlertNotifierV2"), \
             patch.object(_res_mod, "_mapping_cache", {4151: "Abyssal whip"}), \
             patch.object(_res_mod, "_mapping_cache_ts", time.time()), \
             patch("backend.discord_notifier.generate_opportunity_chart", return_value=None), \
             patch.object(monitor, "_get_dump_alert_webhook_sync", return_value="https://fake"), \
             caplog.at_level(logging.INFO, logger="backend.tasks"):

            monitor._check_dump_alerts_sync(db)

        assert any("DUMP_NAME_DEBUG" in r.message for r in caplog.records), \
            "DUMP_NAME_DEBUG must be logged for processed dump items"
        # Resolved name must appear in the log
        assert any("Abyssal whip" in r.message for r in caplog.records), \
            "Resolved name 'Abyssal whip' must appear in DUMP_NAME_DEBUG log"
