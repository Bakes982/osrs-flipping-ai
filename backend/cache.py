"""Redis-style cache accessor used by workers and routers.

Provides ``get_redis()`` with a minimal ``get``/``set`` interface. When Redis
is unavailable, it transparently falls back to the shared in-memory cache
backend so callers can keep using the same API.
"""

from __future__ import annotations

from typing import Optional

from backend.cache_backend import get_cache_backend


class _RedisCompatAdapter:
    """Small adapter exposing a Redis-like get/set API.

    Supported methods:
      - get(key)
      - set(key, value, ex=seconds)
      - exists(key)
    """

    def __init__(self, cache_backend) -> None:
        self._cache = cache_backend

    def get(self, key: str) -> Optional[str]:
        return self._cache.get(key)

    def set(self, key: str, value: str, ex: Optional[int] = None) -> None:
        ttl_seconds = int(ex) if ex is not None else None
        self._cache.set(key, value, ttl_seconds=ttl_seconds)

    def exists(self, key: str) -> int:
        return 1 if self._cache.get(key) is not None else 0


def get_redis():
    """Return a Redis-compatible client.

    If the active cache backend is Redis, return the underlying Redis client.
    Otherwise, return an in-process compatibility adapter.
    """
    backend = get_cache_backend()
    redis_client = getattr(backend, "_client", None)
    if redis_client is not None:
        return redis_client
    return _RedisCompatAdapter(backend)

