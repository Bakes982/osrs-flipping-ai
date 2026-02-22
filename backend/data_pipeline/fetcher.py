"""
OSRS Flipping AI — Wiki Price API fetcher.

Wraps all HTTP calls to prices.runescape.wiki into a single, reusable class.
Handles retries, rate-limiting, and exposes the raw data the rest of the
pipeline needs.

Usage::

    fetcher = WikiFetcher()
    latest  = await fetcher.fetch_latest()
    volume  = await fetcher.fetch_5m()
    mapping = await fetcher.fetch_mapping()
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Dict, List, Optional

import httpx

from backend.core.constants import WIKI_BASE_URL, WIKI_USER_AGENT

logger = logging.getLogger(__name__)

_HEADERS = {"User-Agent": WIKI_USER_AGENT}

# ---------------------------------------------------------------------------
# Shared module-level caches
# Populated by WikiFetcher; consumed by PositionMonitor, AlertMonitor, etc.
# ---------------------------------------------------------------------------

# item_id_str → {"high": int, "low": int, "highTime": int, "lowTime": int}
price_cache: Dict[str, dict] = {}

# item_id_str → {"avgHighPrice": int, "avgLowPrice": int,
#                "highPriceVolume": int, "lowPriceVolume": int}
volume_cache: Dict[str, dict] = {}

# item_id_int → {"id": int, "name": str, "members": bool, "limit": int, ...}
mapping_cache: Dict[int, dict] = {}


class WikiFetcher:
    """Async HTTP client for the OSRS Wiki price API.

    Instantiate once per process; the internal httpx.AsyncClient is
    lazily created and reused across calls.
    """

    # Retry configuration
    MAX_RETRIES: int = 4
    RETRY_BACKOFF_BASE: float = 2.0   # 2 s, 4 s, 8 s, 16 s

    def __init__(self) -> None:
        self._client: Optional[httpx.AsyncClient] = None
        self._last_mapping_fetch: float = 0.0
        # Cache mapping for 1 hour (it almost never changes)
        self._mapping_ttl: float = 3600.0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _client_get(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                headers=_HEADERS,
                timeout=15.0,
                follow_redirects=True,
            )
        return self._client

    async def _get(self, url: str) -> Optional[dict]:
        """GET ``url`` with retries and exponential backoff.

        Returns the parsed JSON body on success, or ``None`` on all failures.
        """
        client = await self._client_get()
        delay = self.RETRY_BACKOFF_BASE
        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                resp = await client.get(url)
                resp.raise_for_status()
                return resp.json()
            except httpx.HTTPStatusError as exc:
                # 429 = rate-limited; back off and retry
                if exc.response.status_code == 429:
                    logger.warning("Wiki API rate-limited; retrying in %.0fs", delay)
                    await asyncio.sleep(delay)
                    delay *= 2
                    continue
                logger.error("Wiki API HTTP error %s for %s", exc.response.status_code, url)
                return None
            except Exception as exc:
                if attempt < self.MAX_RETRIES:
                    logger.warning(
                        "Wiki API request failed (%s); retry %d/%d in %.0fs",
                        exc, attempt, self.MAX_RETRIES, delay,
                    )
                    await asyncio.sleep(delay)
                    delay *= 2
                else:
                    logger.error("Wiki API request failed after %d attempts: %s", self.MAX_RETRIES, exc)
        return None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def fetch_latest(self) -> Dict[str, dict]:
        """Fetch instantaneous prices (/latest).

        Returns a dict keyed by item-ID string::

            {"2": {"high": 153, "highTime": 1700000000, "low": 149, "lowTime": ...}, ...}

        Also updates the module-level ``price_cache``.
        """
        data = await self._get(f"{WIKI_BASE_URL}/latest")
        if data is None:
            return price_cache  # return stale data rather than empty

        items: Dict[str, dict] = data.get("data", {})
        price_cache.clear()
        price_cache.update(items)
        return items

    async def fetch_5m(self, timestamp: Optional[int] = None) -> Dict[str, dict]:
        """Fetch 5-minute averaged prices (/5m).

        Returns a dict keyed by item-ID string::

            {"2": {"avgHighPrice": 152, "highPriceVolume": 427,
                   "avgLowPrice": 148, "lowPriceVolume": 389}, ...}

        Also updates the module-level ``volume_cache``.
        """
        url = f"{WIKI_BASE_URL}/5m"
        if timestamp:
            url += f"?timestamp={timestamp}"

        data = await self._get(url)
        if data is None:
            return volume_cache

        items: Dict[str, dict] = data.get("data", {})
        volume_cache.clear()
        volume_cache.update(items)
        return items

    async def fetch_1h(self, timestamp: Optional[int] = None) -> Dict[str, dict]:
        """Fetch 1-hour averaged prices (/1h)."""
        url = f"{WIKI_BASE_URL}/1h"
        if timestamp:
            url += f"?timestamp={timestamp}"
        data = await self._get(url)
        return data.get("data", {}) if data else {}

    async def fetch_mapping(self, force: bool = False) -> Dict[int, dict]:
        """Fetch item ID → metadata mapping (/mapping).

        The result is cached for ``self._mapping_ttl`` seconds.
        Pass ``force=True`` to bypass the cache.

        Returns a dict keyed by integer item ID::

            {2: {"id": 2, "name": "Cannonball", "limit": 10000, ...}, ...}
        """
        age = time.time() - self._last_mapping_fetch
        if not force and mapping_cache and age < self._mapping_ttl:
            return mapping_cache

        data = await self._get(f"{WIKI_BASE_URL}/mapping")
        if data is None:
            return mapping_cache  # return stale cache rather than empty

        mapping_cache.clear()
        for entry in data:
            item_id = entry.get("id")
            if item_id is not None:
                mapping_cache[int(item_id)] = entry

        self._last_mapping_fetch = time.time()
        return mapping_cache

    async def fetch_item_price(self, item_id: int) -> Optional[dict]:
        """Fetch the current price for a single item.

        Returns a dict with ``high``, ``low``, ``highTime``, ``lowTime``,
        or ``None`` if the item has no current trade data.
        """
        latest = await self.fetch_latest()
        return latest.get(str(item_id))

    async def close(self) -> None:
        """Cleanly close the underlying HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    # ------------------------------------------------------------------
    # Convenience: top-N items by volume
    # ------------------------------------------------------------------

    @staticmethod
    def top_items_by_volume(n: int = 50) -> List[int]:
        """Return the ``n`` item IDs with the highest combined 5-min volume.

        Uses the in-memory ``volume_cache``; returns an empty list if the
        cache has not been populated yet.
        """
        if not volume_cache:
            return []
        scored = []
        for id_str, data in volume_cache.items():
            vol = (data.get("highPriceVolume") or 0) + (data.get("lowPriceVolume") or 0)
            if vol > 0:
                scored.append((int(id_str), vol))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [item_id for item_id, _ in scored[:n]]
