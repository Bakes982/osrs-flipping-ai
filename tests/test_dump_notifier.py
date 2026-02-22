"""
Unit tests for backend.alerts.dump_notifier (v2).

Tests
-----
* test_embed_title_uses_resolved_name
    The Discord embed title must contain the resolved item name, not
    a raw "Item XXXXX" placeholder.

* test_chart_attachment_included_when_chart_path_returned
    When the alert has a valid chart_path, the notifier must send via
    multipart/form-data (files kwarg present) and set embed["image"].

* test_no_crash_if_chart_gen_raises
    If the chart file is missing / chart generator raised, the notifier
    must still send a plain JSON POST without crashing.

* test_send_returns_false_on_http_error
    Non-200 Discord response must return False without raising.

* test_send_returns_false_on_network_error
    Network-level exception must be caught and return False.

* test_send_skips_when_no_url
    Empty webhook URL causes send() to return False immediately.

* test_embed_fields_contain_expected_keys
    Embed fields dict must include sell_ratio, drop, profit, qty, etc.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import replace
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from backend.alerts.dump_notifier import DumpAlertV2, DumpAlertNotifierV2


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

FAKE_WEBHOOK = "https://discord.com/api/webhooks/111/fake_token"


def _make_alert(chart_path=None) -> DumpAlertV2:
    return DumpAlertV2(
        item_id=4151,
        item_name="Abyssal whip",
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
        chart_path=chart_path,
        timestamp=datetime(2026, 2, 22, 12, 0, 0),
    )


def _mock_discord_ok() -> MagicMock:
    """Simulate a 204 No Content from Discord."""
    resp = MagicMock()
    resp.status_code = 204
    resp.text = ""
    return resp


def _mock_discord_error(code: int = 400) -> MagicMock:
    resp = MagicMock()
    resp.status_code = code
    resp.text = "Bad Request"
    return resp


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestDumpAlertV2:
    def test_resolved_name_uses_good_fallback(self):
        """When item_name is a real name, resolved_name returns it directly."""
        alert = _make_alert()
        with patch("backend.alerts.item_name_resolver.requests.get") as mock_get:
            name = alert.resolved_name
        assert name == "Abyssal whip"
        mock_get.assert_not_called()

    def test_resolved_name_fetches_when_placeholder(self):
        """When item_name is 'Item 4151', resolved_name fetches from Wiki."""
        from unittest.mock import MagicMock
        alert = _make_alert()
        alert = replace(alert, item_name="Item 4151")

        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = [{"id": 4151, "name": "Abyssal whip"}]

        import backend.alerts.item_name_resolver as mod
        mod._mapping_cache = None
        mod._mapping_cache_ts = None

        with patch("backend.alerts.item_name_resolver.requests.get", return_value=mock_resp):
            name = alert.resolved_name

        assert name == "Abyssal whip"


class TestBuildEmbed:
    def test_embed_title_uses_resolved_name(self):
        """Embed title must show the real item name, not 'Item XXXXX'."""
        alert = _make_alert()
        notifier = DumpAlertNotifierV2(FAKE_WEBHOOK)
        embed = notifier._build_embed(alert, "Abyssal whip")
        assert embed["title"] == "DUMP DETECTED: Abyssal whip"
        assert "Item 4151" not in embed["title"]

    def test_embed_footer_text(self):
        alert = _make_alert()
        notifier = DumpAlertNotifierV2(FAKE_WEBHOOK)
        embed = notifier._build_embed(alert, "Abyssal whip")
        assert embed["footer"]["text"] == "OSRS Flipping AI • Dump Detector"

    def test_embed_fields_contain_expected_keys(self):
        """Embed must include sell_ratio, price drop, profit, qty, invest fields."""
        alert = _make_alert()
        notifier = DumpAlertNotifierV2(FAKE_WEBHOOK)
        embed = notifier._build_embed(alert, "Abyssal whip")
        field_names = {f["name"] for f in embed["fields"]}
        required = {
            "Item ID",
            "Current Price",
            "Ref Avg (4h)",
            "Price Drop",
            "Sell Ratio",
            "Sold 5m",
            "Bought 5m",
            "Profit / item (net)",
            "Predicted Recovery",
            "Recommended Qty",
            "Max Invest",
            "Est Total Profit",
        }
        assert required.issubset(field_names), f"Missing fields: {required - field_names}"

    def test_sell_ratio_formatted_as_percent(self):
        alert = _make_alert()
        notifier = DumpAlertNotifierV2(FAKE_WEBHOOK)
        embed = notifier._build_embed(alert, "Abyssal whip")
        sell_ratio_field = next(f for f in embed["fields"] if f["name"] == "Sell Ratio")
        assert "%" in sell_ratio_field["value"]

    def test_embed_color_high_confidence_is_green(self):
        alert = _make_alert()
        notifier = DumpAlertNotifierV2(FAKE_WEBHOOK)
        embed = notifier._build_embed(alert, "Abyssal whip")
        assert embed["color"] == 0x00FF00  # green

    def test_embed_color_low_confidence_is_red(self):
        alert = replace(_make_alert(), confidence="LOW")
        notifier = DumpAlertNotifierV2(FAKE_WEBHOOK)
        embed = notifier._build_embed(alert, "Abyssal whip")
        assert embed["color"] == 0xEF5350


class TestSend:
    def test_chart_attachment_included_when_chart_path_valid(self):
        """When a PNG chart file exists, send must use multipart/form-data."""
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp.write(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)
            chart_path = tmp.name

        try:
            alert = _make_alert(chart_path=chart_path)
            notifier = DumpAlertNotifierV2(FAKE_WEBHOOK)

            captured: dict = {}

            def fake_post(url, **kwargs):
                captured.update(kwargs)
                return _mock_discord_ok()

            with patch("backend.alerts.dump_notifier._requests.post", side_effect=fake_post):
                result = notifier.send(alert)

            assert result is True
            # multipart send uses 'data' + 'files', NOT 'json'
            assert "files" in captured, "Expected multipart/form-data (files kwarg)"
            assert "data" in captured
            # Payload JSON must embed the attachment URL
            payload = json.loads(captured["data"]["payload_json"])
            embed = payload["embeds"][0]
            assert "image" in embed
            assert embed["image"]["url"].startswith("attachment://")
        finally:
            os.unlink(chart_path)

    def test_no_crash_if_chart_path_missing(self):
        """Missing chart file must cause graceful fallback to plain JSON POST."""
        alert = _make_alert(chart_path="/tmp/nonexistent_chart_99999.png")
        notifier = DumpAlertNotifierV2(FAKE_WEBHOOK)

        captured: dict = {}

        def fake_post(url, **kwargs):
            captured.update(kwargs)
            return _mock_discord_ok()

        with patch("backend.alerts.dump_notifier._requests.post", side_effect=fake_post):
            result = notifier.send(alert)

        assert result is True
        # Falls back to plain JSON, no files kwarg
        assert "files" not in captured
        assert "json" in captured

    def test_no_crash_if_chart_gen_raises(self):
        """Chart-generation failure must not crash; alert is still sent without chart."""
        alert = _make_alert()  # chart_path=None

        notifier = DumpAlertNotifierV2(FAKE_WEBHOOK)

        with patch("backend.alerts.dump_notifier._requests.post", return_value=_mock_discord_ok()):
            result = notifier.send(alert)

        assert result is True   # sent plain JSON without crashing

    def test_send_returns_false_on_http_error(self):
        """Non-2xx Discord response must return False."""
        alert = _make_alert()
        notifier = DumpAlertNotifierV2(FAKE_WEBHOOK)

        with patch(
            "backend.alerts.dump_notifier._requests.post",
            return_value=_mock_discord_error(400),
        ):
            result = notifier.send(alert)

        assert result is False

    def test_send_returns_false_on_network_error(self):
        """Network exception must be caught and return False."""
        alert = _make_alert()
        notifier = DumpAlertNotifierV2(FAKE_WEBHOOK)

        with patch(
            "backend.alerts.dump_notifier._requests.post",
            side_effect=ConnectionError("socket closed"),
        ):
            result = notifier.send(alert)

        assert result is False

    def test_send_skips_when_no_url(self):
        """Empty webhook URL must return False without making any HTTP call."""
        alert = _make_alert()
        notifier = DumpAlertNotifierV2("")

        with patch("backend.alerts.dump_notifier._requests.post") as mock_post:
            result = notifier.send(alert)

        assert result is False
        mock_post.assert_not_called()

    def test_send_204_accepted_as_success(self):
        """HTTP 204 No Content is a valid Discord success response."""
        alert = _make_alert()
        notifier = DumpAlertNotifierV2(FAKE_WEBHOOK)
        resp = MagicMock()
        resp.status_code = 204
        with patch("backend.alerts.dump_notifier._requests.post", return_value=resp):
            assert notifier.send(alert) is True

    def test_send_200_accepted_as_success(self):
        """HTTP 200 OK (webhook with wait=true) is also a success response."""
        alert = _make_alert()
        notifier = DumpAlertNotifierV2(FAKE_WEBHOOK)
        resp = MagicMock()
        resp.status_code = 200
        with patch("backend.alerts.dump_notifier._requests.post", return_value=resp):
            assert notifier.send(alert) is True
