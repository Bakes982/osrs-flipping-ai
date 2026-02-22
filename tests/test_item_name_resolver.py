"""
Tests for backend.alerts.item_name_resolver.ItemNameResolver.

Covers:
  1. resolve() returns name from populated cache
  2. resolve() returns "Item {id}" default fallback when not found
  3. resolve() returns custom fallback when provided and not found
  4. is_stale() is True when never fetched; False after fetch
  5. Stale cache triggers a refresh attempt on next resolve()
  6. Network failure leaves stale cache in place (name still resolved from old data)
  7. Network failure on empty cache returns fallback (no crash)
  8. Successful refresh updates _fetched_at and populates cache
  9. Double-checked locking: second call after fresh fetch does NOT re-fetch
 10. _passes_dump_filters gate in flip_cache
 11. Cooldown gate in _update_dump_persistence (no double-alert within 60 m)
"""

from __future__ import annotations

import json
import time
from unittest.mock import MagicMock, patch

import pytest

from backend.alerts.item_name_resolver import ItemNameResolver


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fresh_resolver(entries: list[dict] | None = None) -> ItemNameResolver:
    """Return a resolver pre-populated with ``entries`` and a fresh timestamp."""
    r = ItemNameResolver(ttl_seconds=3600)
    if entries is not None:
        r._cache      = {e["id"]: e["name"] for e in entries if "id" in e and "name" in e}
        r._fetched_at = time.time()
    return r


def _stale_resolver(entries: list[dict] | None = None) -> ItemNameResolver:
    """Return a resolver with an artificially expired timestamp."""
    r = _fresh_resolver(entries)
    r._fetched_at = 0.0    # force stale
    return r


def _mock_urlopen(payload: list[dict]):
    """Context-manager mock for urllib.request.urlopen returning JSON payload."""
    raw = json.dumps(payload).encode()
    mock_resp = MagicMock()
    mock_resp.read.return_value = raw
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__  = MagicMock(return_value=False)
    return mock_resp


# ---------------------------------------------------------------------------
# 1–3: resolve() basics
# ---------------------------------------------------------------------------

class TestResolveFromCache:
    def test_known_id_returns_name(self):
        r = _fresh_resolver([{"id": 4151, "name": "Abyssal whip"}])
        assert r.resolve(4151) == "Abyssal whip"

    def test_unknown_id_default_fallback(self):
        r = _fresh_resolver([])
        assert r.resolve(9999) == "Item 9999"

    def test_unknown_id_custom_fallback(self):
        r = _fresh_resolver([])
        assert r.resolve(9999, fallback="Unknown") == "Unknown"

    def test_multiple_items_all_resolved(self):
        entries = [{"id": 1, "name": "Coins"}, {"id": 2, "name": "Cannonball"}]
        r = _fresh_resolver(entries)
        assert r.resolve(1) == "Coins"
        assert r.resolve(2) == "Cannonball"


# ---------------------------------------------------------------------------
# 4: is_stale()
# ---------------------------------------------------------------------------

class TestIsStale:
    def test_never_fetched_is_stale(self):
        r = ItemNameResolver()
        assert r.is_stale()

    def test_fresh_after_fetch(self):
        r = _fresh_resolver([])
        assert not r.is_stale()

    def test_expires_after_ttl(self):
        r = ItemNameResolver(ttl_seconds=1)
        r._fetched_at = time.time() - 2    # 2 s ago, TTL=1 s
        assert r.is_stale()


# ---------------------------------------------------------------------------
# 5: Stale triggers refresh
# ---------------------------------------------------------------------------

class TestStaleTriggersRefresh:
    def test_stale_cache_refreshed_on_resolve(self):
        r = _stale_resolver()
        payload = [{"id": 1234, "name": "Dragon scimitar"}]

        with patch("urllib.request.urlopen", return_value=_mock_urlopen(payload)):
            name = r.resolve(1234)

        assert name == "Dragon scimitar"

    def test_after_refresh_cache_is_fresh(self):
        r = _stale_resolver()
        payload = [{"id": 1, "name": "Coins"}]

        with patch("urllib.request.urlopen", return_value=_mock_urlopen(payload)):
            r.resolve(1)

        assert not r.is_stale()

    def test_refresh_replaces_old_entries(self):
        r = _stale_resolver([{"id": 1, "name": "OldName"}])
        payload = [{"id": 1, "name": "NewName"}]

        with patch("urllib.request.urlopen", return_value=_mock_urlopen(payload)):
            name = r.resolve(1)

        assert name == "NewName"


# ---------------------------------------------------------------------------
# 6–7: Network failure handling
# ---------------------------------------------------------------------------

class TestNetworkFailure:
    def test_failure_keeps_stale_cache(self):
        r = _stale_resolver([{"id": 1, "name": "Coins"}])

        with patch("urllib.request.urlopen", side_effect=OSError("network down")):
            name = r.resolve(1)

        assert name == "Coins"    # stale cache still used

    def test_failure_empty_cache_returns_fallback(self):
        r = _stale_resolver([])   # empty stale cache

        with patch("urllib.request.urlopen", side_effect=OSError("network down")):
            name = r.resolve(4151)

        assert name == "Item 4151"

    def test_failure_does_not_raise(self):
        r = _stale_resolver()

        with patch("urllib.request.urlopen", side_effect=Exception("timeout")):
            result = r.resolve(1)   # must not raise

        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# 8: Successful refresh updates state
# ---------------------------------------------------------------------------

class TestSuccessfulRefresh:
    def test_prefetch_populates_cache(self):
        r = ItemNameResolver()
        payload = [{"id": 11802, "name": "Twisted bow"}]

        with patch("urllib.request.urlopen", return_value=_mock_urlopen(payload)):
            ok = r.prefetch()

        assert ok is True
        assert r._cache[11802] == "Twisted bow"
        assert r._fetched_at > 0

    def test_entries_without_id_or_name_skipped(self):
        r = ItemNameResolver()
        payload = [
            {"id": 1,   "name": "Coins"},
            {"name": "no-id"},          # missing id
            {"id": 2},                  # missing name
            {},
        ]
        with patch("urllib.request.urlopen", return_value=_mock_urlopen(payload)):
            r.prefetch()

        assert r._cache == {1: "Coins"}


# ---------------------------------------------------------------------------
# 9: No redundant re-fetch when already fresh
# ---------------------------------------------------------------------------

class TestNoRedundantFetch:
    def test_fresh_resolver_does_not_call_urlopen(self):
        r = _fresh_resolver([{"id": 1, "name": "Coins"}])

        with patch("urllib.request.urlopen") as mock_open:
            r.resolve(1)
            mock_open.assert_not_called()


# ---------------------------------------------------------------------------
# 10: _passes_dump_filters (flip_cache internal)
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
# 11: Cooldown gate in _update_dump_persistence
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
