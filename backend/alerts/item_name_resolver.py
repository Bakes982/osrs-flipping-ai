"""
backend.alerts.item_name_resolver — OSRS item ID → name lookup with TTL cache.

Uses the OSRS Wiki mapping endpoint (no auth required).  The full mapping is
fetched at most once per TTL window (default 6 h) and held in memory.  All
public methods are safe to call from any thread (a threading.Lock protects
the lazy-refresh path).

Typical usage::

    from backend.alerts.item_name_resolver import resolver
    name = resolver.resolve(4151)   # → "Abyssal whip"
"""

from __future__ import annotations

import json
import logging
import threading
import time
import urllib.request
from typing import Dict, Optional

logger = logging.getLogger(__name__)

_MAPPING_URL = "https://prices.runescape.wiki/api/v1/osrs/mapping"
_USER_AGENT  = "OSRS-AI-Flipper v2.0 - Discord: bakes982"


class ItemNameResolver:
    """Thread-safe item-ID-to-name resolver backed by OSRS Wiki mapping API.

    Parameters
    ----------
    ttl_seconds:
        How long the cached mapping is considered fresh.  After this period
        the next ``resolve()`` call will attempt a background refresh.
        Default: 6 hours.
    """

    def __init__(self, ttl_seconds: float = 6 * 3600) -> None:
        self._ttl      = ttl_seconds
        self._cache:   Dict[int, str] = {}
        self._fetched_at: float       = 0.0
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def resolve(self, item_id: int, fallback: str = "") -> str:
        """Return the display name for ``item_id``.

        If the cache is stale a refresh is attempted; failures leave the
        stale cache in place so callers always get *something*.

        Parameters
        ----------
        item_id:
            Numeric OSRS item ID.
        fallback:
            Returned when the item is not found in the mapping.
            Defaults to ``"Item {item_id}"`` when empty.
        """
        if self.is_stale():
            self._try_refresh()

        name = self._cache.get(item_id)
        if name:
            return name
        return fallback if fallback else f"Item {item_id}"

    def is_stale(self) -> bool:
        """True if the mapping has never been fetched or has expired."""
        return (time.time() - self._fetched_at) > self._ttl

    def prefetch(self) -> bool:
        """Eagerly fetch the mapping.  Returns True on success."""
        return self._try_refresh()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _try_refresh(self) -> bool:
        """Attempt to refresh the mapping; silently swallows errors."""
        with self._lock:
            # Double-checked locking: another thread may have refreshed
            # while we were waiting.
            if not self.is_stale():
                return True
            try:
                self._fetch()
                return True
            except Exception as exc:
                logger.warning(
                    "ItemNameResolver: mapping refresh failed — %s "
                    "(stale cache has %d entries)",
                    exc, len(self._cache),
                )
                return False

    def _fetch(self) -> None:
        """Fetch the OSRS Wiki mapping and update the cache (not thread-safe; hold lock)."""
        req = urllib.request.Request(
            _MAPPING_URL,
            headers={"User-Agent": _USER_AGENT},
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            raw = resp.read()

        items = json.loads(raw)
        new_cache: Dict[int, str] = {}
        for entry in items:
            iid  = entry.get("id")
            name = entry.get("name")
            if isinstance(iid, int) and name:
                new_cache[iid] = name

        self._cache      = new_cache
        self._fetched_at = time.time()
        logger.debug("ItemNameResolver: refreshed mapping (%d items)", len(new_cache))


# Module-level singleton shared across the process
resolver = ItemNameResolver()
