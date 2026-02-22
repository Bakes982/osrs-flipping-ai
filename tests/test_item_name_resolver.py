"""
Unit tests for backend.alerts.item_name_resolver.

Tests
-----
* test_resolve_uses_fallback_when_valid
    When a non-empty fallback that does NOT start with "Item " is supplied,
    resolve_item_name must return it immediately without hitting the network.

* test_resolve_fetches_mapping_when_fallback_is_Item_x
    When fallback starts with "Item " (placeholder), the resolver must
    call requests.get and return the real name from the mapping.

* test_resolve_returns_Item_x_on_fetch_failure
    When requests.get raises an exception the resolver must silently
    degrade and return "Item {item_id}".

* test_cache_ttl_not_expired_avoids_second_fetch
    A second call within the TTL must NOT trigger another HTTP request.

* test_cache_ttl_expired_triggers_refetch
    Simulating an expired timestamp must trigger a fresh HTTP request.
"""

from __future__ import annotations

import time
from typing import Optional
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mapping_response(items: list[dict]) -> MagicMock:
    """Build a mock requests.Response that returns ``items`` as JSON."""
    mock_resp = MagicMock()
    mock_resp.raise_for_status.return_value = None
    mock_resp.json.return_value = items
    return mock_resp


# Reset the module-level cache before/after each test so tests are isolated.
@pytest.fixture(autouse=True)
def _reset_cache():
    """Force the resolver to start each test with a cold cache."""
    import backend.alerts.item_name_resolver as mod
    mod._mapping_cache = None
    mod._mapping_cache_ts = None
    yield
    mod._mapping_cache = None
    mod._mapping_cache_ts = None


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestResolveItemName:
    def test_resolve_uses_fallback_when_valid(self):
        """Non-placeholder fallback is returned without any HTTP call."""
        from backend.alerts.item_name_resolver import resolve_item_name

        with patch("backend.alerts.item_name_resolver.requests.get") as mock_get:
            result = resolve_item_name(4151, fallback="Dragon claws")

        assert result == "Dragon claws"
        mock_get.assert_not_called()

    def test_resolve_uses_fallback_strips_whitespace(self):
        """Fallback with surrounding whitespace is stripped and accepted."""
        from backend.alerts.item_name_resolver import resolve_item_name

        with patch("backend.alerts.item_name_resolver.requests.get") as mock_get:
            result = resolve_item_name(4151, fallback="  Twisted bow  ")

        assert result == "Twisted bow"
        mock_get.assert_not_called()

    def test_resolve_fetches_mapping_when_fallback_is_Item_x(self):
        """Placeholder fallback 'Item 4151' triggers a Wiki mapping fetch."""
        from backend.alerts.item_name_resolver import resolve_item_name

        mapping_data = [
            {"id": 4151, "name": "Abyssal whip"},
            {"id": 20997, "name": "Twisted bow"},
        ]
        mock_resp = _make_mapping_response(mapping_data)

        with patch("backend.alerts.item_name_resolver.requests.get", return_value=mock_resp) as mock_get:
            result = resolve_item_name(4151, fallback="Item 4151")

        mock_get.assert_called_once()
        assert result == "Abyssal whip"

    def test_resolve_fetches_mapping_when_no_fallback(self):
        """With no fallback, the resolver must query the mapping and return the name."""
        from backend.alerts.item_name_resolver import resolve_item_name

        mapping_data = [{"id": 31099, "name": "Spectral spirit shield"}]
        mock_resp = _make_mapping_response(mapping_data)

        with patch("backend.alerts.item_name_resolver.requests.get", return_value=mock_resp):
            result = resolve_item_name(31099)

        assert result == "Spectral spirit shield"

    def test_resolve_returns_Item_x_on_fetch_failure(self):
        """Network failure must cause silent fallback to 'Item {item_id}'."""
        from backend.alerts.item_name_resolver import resolve_item_name

        with patch(
            "backend.alerts.item_name_resolver.requests.get",
            side_effect=ConnectionError("timeout"),
        ):
            result = resolve_item_name(31099, fallback="Item 31099")

        assert result == "Item 31099"

    def test_resolve_returns_Item_x_on_fetch_failure_no_fallback(self):
        """Network failure with no fallback returns 'Item {item_id}'."""
        from backend.alerts.item_name_resolver import resolve_item_name

        with patch(
            "backend.alerts.item_name_resolver.requests.get",
            side_effect=OSError("DNS failure"),
        ):
            result = resolve_item_name(99999)

        assert result == "Item 99999"

    def test_cache_ttl_not_expired_avoids_second_fetch(self):
        """Two calls within TTL must only fire one HTTP request."""
        from backend.alerts.item_name_resolver import resolve_item_name

        mapping_data = [{"id": 4151, "name": "Abyssal whip"}]
        mock_resp = _make_mapping_response(mapping_data)

        with patch("backend.alerts.item_name_resolver.requests.get", return_value=mock_resp) as mock_get:
            resolve_item_name(4151)          # first call — cold cache
            resolve_item_name(4151)          # second call — hot cache

        mock_get.assert_called_once()

    def test_cache_ttl_expired_triggers_refetch(self):
        """Simulating an expired timestamp must trigger a new HTTP request."""
        import backend.alerts.item_name_resolver as mod
        from backend.alerts.item_name_resolver import resolve_item_name

        mapping_data = [{"id": 4151, "name": "Abyssal whip"}]
        mock_resp = _make_mapping_response(mapping_data)

        with patch("backend.alerts.item_name_resolver.requests.get", return_value=mock_resp) as mock_get:
            # Warm cache
            resolve_item_name(4151)
            assert mock_get.call_count == 1

            # Expire the cache manually
            mod._mapping_cache_ts = time.time() - (mod._TTL_SECONDS + 1)

            # Should trigger a new fetch
            resolve_item_name(4151)
            assert mock_get.call_count == 2

    def test_empty_fallback_triggers_mapping_lookup(self):
        """Empty string fallback should NOT short-circuit; resolver fetches mapping."""
        from backend.alerts.item_name_resolver import resolve_item_name

        mapping_data = [{"id": 4151, "name": "Abyssal whip"}]
        mock_resp = _make_mapping_response(mapping_data)

        with patch("backend.alerts.item_name_resolver.requests.get", return_value=mock_resp) as mock_get:
            result = resolve_item_name(4151, fallback="")

        mock_get.assert_called_once()
        assert result == "Abyssal whip"
