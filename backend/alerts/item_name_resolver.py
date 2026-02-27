"""
backend.alerts.item_name_resolver — Item ID → human-readable name (v2).

Fetches the OSRS Wiki /mapping endpoint once and caches the result in
module-level globals for 6 hours.  Avoids "Item 31099" appearing in Discord
alerts by ensuring the mapping is always fresh and correctly indexed.

Public API
----------
    resolve_item_name(item_id, fallback=None) -> str
    invalidate_cache() -> None

Resolution order:
    1. If ``fallback`` is non-empty and does NOT start with "Item " → return it.
    2. Look up the OSRS Wiki mapping (6-hour TTL cache) → return name.
    3. Return f"Item {item_id}" as a last resort.

Compatibility
-------------
A ``resolver`` singleton is also provided for legacy callers that use
``resolver.resolve(item_id, fallback="")``.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import requests

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MAPPING_URL = "https://prices.runescape.wiki/api/v1/osrs/mapping"
_HEADERS = {
    "User-Agent": (
        "OSRS-Flipping-AI/2.0 "
        "(github.com/Bakes982/osrs-flipping-ai; "
        "contact: mike.baker982@hotmail.com)"
    )
}
_TTL_SECONDS: int = 6 * 3600  # 6 hours

# ---------------------------------------------------------------------------
# Module-level cache (survives the process lifetime between refreshes)
# ---------------------------------------------------------------------------

_mapping_cache: Optional[dict[int, str]] = None
_limit_cache: Optional[dict[int, int]] = None
_mapping_cache_ts: Optional[float] = None

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _fetch_mapping() -> tuple[dict[int, str], dict[int, int]]:
    """Download the full item mapping from the OSRS Wiki.

    Returns (name_map, limit_map) where limit_map is {id: ge_buy_limit}.
    """
    resp = requests.get(_MAPPING_URL, headers=_HEADERS, timeout=10)
    resp.raise_for_status()
    entries = resp.json()  # list of dicts
    names: dict[int, str] = {}
    limits: dict[int, int] = {}
    for entry in entries:
        try:
            item_id = int(entry["id"])
            names[item_id] = str(entry["name"])
            if entry.get("limit"):
                limits[item_id] = int(entry["limit"])
        except (KeyError, ValueError, TypeError):
            continue
    return names, limits


def _ensure_cache_fresh() -> dict[int, str]:
    """Return the in-memory name mapping, refreshing it when stale or absent."""
    global _mapping_cache, _limit_cache, _mapping_cache_ts
    now = time.time()
    needs_refresh = (
        _mapping_cache is None
        or _mapping_cache_ts is None
        or (now - _mapping_cache_ts) >= _TTL_SECONDS
    )
    if needs_refresh:
        try:
            new_names, new_limits = _fetch_mapping()
            _mapping_cache = new_names
            _limit_cache = new_limits
            _mapping_cache_ts = now
            logger.info(
                "item_name_resolver: mapping refreshed (%d items, %d limits)",
                len(_mapping_cache), len(_limit_cache),
            )
        except Exception as exc:
            logger.warning("item_name_resolver: mapping fetch failed: %s", exc)
            if _mapping_cache is None:
                # First-ever fetch failed — use empty dicts so the module stays usable
                _mapping_cache = {}
                _limit_cache = {}
            # Do NOT update _mapping_cache_ts so the next call will retry
    return _mapping_cache  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def resolve_item_name(item_id: int, fallback: Optional[str] = None) -> str:
    """Resolve an OSRS item ID to its human-readable name.

    Parameters
    ----------
    item_id:
        Integer item ID (e.g. 4151).
    fallback:
        Optional pre-existing name string (e.g. from a DB record or a
        previous scan result).

    Returns
    -------
    str
        Human-readable name such as ``"Abyssal whip"``, or
        ``"Item 4151"`` if resolution fails entirely.
    """
    # Fast path: caller already has a real name that isn't a fallback placeholder
    if fallback is not None:
        stripped = fallback.strip()
        if stripped and not stripped.startswith("Item "):
            return stripped

    # Fetch / use cached mapping
    mapping = _ensure_cache_fresh()
    name = mapping.get(int(item_id))
    if name:
        return name

    return f"Item {item_id}"


def resolve_item_limit(item_id: int) -> Optional[int]:
    """Return the GE 4-hour buy limit for *item_id*, or None if unknown.

    Sourced from the same OSRS Wiki /mapping endpoint as resolve_item_name,
    so it costs no extra network requests after the first call.
    """
    _ensure_cache_fresh()
    if _limit_cache is None:
        return None
    return _limit_cache.get(int(item_id))


def invalidate_cache() -> None:
    """Force the next call to re-fetch the mapping (useful in tests)."""
    global _mapping_cache, _limit_cache, _mapping_cache_ts
    _mapping_cache = None
    _limit_cache = None
    _mapping_cache_ts = None


# ---------------------------------------------------------------------------
# Compatibility shim for legacy callers using resolver.resolve(item_id)
# ---------------------------------------------------------------------------

class _ResolverCompat:
    """Thin wrapper around resolve_item_name() for legacy callers."""

    def resolve(self, item_id: int, fallback: str = "") -> str:
        return resolve_item_name(item_id, fallback if fallback else None)


# Module-level singleton shared across the process
resolver = _ResolverCompat()
