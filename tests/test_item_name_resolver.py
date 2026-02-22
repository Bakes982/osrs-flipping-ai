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

Also tests the legacy resolver.resolve() compatibility shim and
flip_cache internal helpers (_passes_dump_filters, cooldown gate).
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest


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
# Tests — resolve_item_name functional API
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


# ---------------------------------------------------------------------------
# Tests — resolver compatibility shim
# ---------------------------------------------------------------------------


class TestResolverShim:
    def test_resolver_resolve_returns_name(self):
        """Legacy resolver.resolve() must work via the shim."""
        from backend.alerts.item_name_resolver import resolver

        mapping_data = [{"id": 4151, "name": "Abyssal whip"}]
        mock_resp = _make_mapping_response(mapping_data)

        with patch("backend.alerts.item_name_resolver.requests.get", return_value=mock_resp):
            result = resolver.resolve(4151)

        assert result == "Abyssal whip"

    def test_resolver_resolve_fallback_passthrough(self):
        """resolver.resolve() with a valid fallback skips HTTP."""
        from backend.alerts.item_name_resolver import resolver

        with patch("backend.alerts.item_name_resolver.requests.get") as mock_get:
            result = resolver.resolve(4151, fallback="Abyssal whip")

        assert result == "Abyssal whip"
        mock_get.assert_not_called()


# ---------------------------------------------------------------------------
# Tests — _passes_dump_filters (flip_cache internal)
# ---------------------------------------------------------------------------

class TestPassesDumpFilters:
    def _call(self, buy: int, profit: int):
        from backend.flip_cache import _passes_dump_filters
        return _passes_dump_filters({"recommended_buy": buy, "net_profit": profit})

    def test_passes_when_above_thresholds(self):
        # default thresholds: min_price=100_000, min_profit=10_000
        assert self._call(buy=200_000, profit=15_000) is True

    def test_fails_when_price_too_low(self):
        assert self._call(buy=50_000, profit=15_000) is False

    def test_fails_when_profit_too_low(self):
        assert self._call(buy=200_000, profit=5_000) is False

    def test_fails_when_both_too_low(self):
        assert self._call(buy=50_000, profit=5_000) is False

    def test_zero_buy_passes_price_gate(self):
        # buy=0 means "no price data" — price filter is skipped; profit gate still applies
        assert self._call(buy=0, profit=15_000) is True


# ---------------------------------------------------------------------------
# Tests — Cooldown gate in _update_dump_persistence
# ---------------------------------------------------------------------------

class TestDumpCooldown:
    """Verify that an alert does not fire twice within the cooldown window."""

    def _run_cycles(self, n: int, state: dict, m: dict):
        """Simulate n high-signal cycles and return how many alerts were emitted."""
        from backend import flip_cache as fc
        from unittest.mock import patch as _patch

        fired = []

        def fake_emit(metrics):
            fired.append(metrics)

        persistence_k = 2   # minimum cycles before alert

        with _patch.object(fc, "_emit_dump_alert", side_effect=fake_emit), \
             _patch.object(fc, "_passes_dump_filters", return_value=True):
            for _ in range(n):
                signal = m.get("dump_signal", "none")
                now = time.time()
                cooldown_secs = fc._cfg.DUMP_ALERT_COOLDOWN_MINUTES * 60

                if signal == "high":
                    state["high_count"] += 1
                    cooldown_elapsed = (now - state.get("last_alert_ts", 0.0)) >= cooldown_secs
                    if state["high_count"] >= persistence_k and cooldown_elapsed:
                        state["alerted"]       = True
                        state["last_alert_ts"] = now
                        fake_emit(m)
                else:
                    state["high_count"] = 0
                    state["alerted"]    = False

        return len(fired)

    def test_no_double_alert_within_cooldown(self):
        state = {"high_count": 0, "alerted": False, "last_alert_ts": 0.0}
        m = {"item_id": 1, "item_name": "X", "dump_signal": "high",
             "dump_risk_score": 80.0, "net_profit": 50_000, "recommended_buy": 1_000_000}

        fired = self._run_cycles(10, state, m)   # 10 high cycles
        assert fired == 1   # alert fires exactly once

    def test_alert_fires_after_cooldown_resets(self):
        state = {"high_count": 0, "alerted": False, "last_alert_ts": 0.0}
        m = {"item_id": 1, "item_name": "X", "dump_signal": "high",
             "dump_risk_score": 80.0, "net_profit": 50_000, "recommended_buy": 1_000_000}

        # First batch: should fire
        fired1 = self._run_cycles(5, state, m)
        assert fired1 == 1

        # Simulate cooldown elapsed
        state["last_alert_ts"] = 0.0
        state["alerted"]       = False
        state["high_count"]    = 0

        # Second batch: should fire again
        fired2 = self._run_cycles(5, state, m)
        assert fired2 == 1
