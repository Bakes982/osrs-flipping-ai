"""
Shared cache backend (Redis preferred, in-memory fallback).
"""

from __future__ import annotations

import json
import threading
import time
from typing import Any, Dict, Optional

from backend import config

try:
    import redis  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    redis = None


class CacheBackend:
    backend: str = "none"

    def get(self, key: str) -> Optional[str]:
        raise NotImplementedError

    def set(self, key: str, value: str, ttl_seconds: Optional[int] = None) -> None:
        raise NotImplementedError

    def get_json(self, key: str) -> Optional[Any]:
        raw = self.get(key)
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except Exception:
            return None

    def set_json(self, key: str, value: Any, ttl_seconds: Optional[int] = None) -> None:
        self.set(key, json.dumps(value), ttl_seconds=ttl_seconds)


class MemoryCacheBackend(CacheBackend):
    backend = "memory"

    def __init__(self) -> None:
        self._store: Dict[str, tuple[str, Optional[float]]] = {}
        self._lock = threading.Lock()

    def get(self, key: str) -> Optional[str]:
        now = time.time()
        with self._lock:
            hit = self._store.get(key)
            if not hit:
                return None
            value, expires_at = hit
            if expires_at is not None and now > expires_at:
                self._store.pop(key, None)
                return None
            return value

    def set(self, key: str, value: str, ttl_seconds: Optional[int] = None) -> None:
        expires_at = None
        if ttl_seconds is not None:
            expires_at = time.time() + max(1, ttl_seconds)
        with self._lock:
            self._store[key] = (value, expires_at)


class RedisCacheBackend(CacheBackend):
    backend = "redis"

    def __init__(self, url: str) -> None:
        if redis is None:
            raise RuntimeError("redis package not installed")
        self._client = redis.from_url(url, decode_responses=True)

    def get(self, key: str) -> Optional[str]:
        return self._client.get(key)

    def set(self, key: str, value: str, ttl_seconds: Optional[int] = None) -> None:
        if ttl_seconds:
            self._client.setex(key, max(1, int(ttl_seconds)), value)
        else:
            self._client.set(key, value)


_backend_singleton: Optional[CacheBackend] = None
_backend_lock = threading.Lock()


def get_cache_backend() -> CacheBackend:
    global _backend_singleton
    if _backend_singleton is not None:
        return _backend_singleton

    with _backend_lock:
        if _backend_singleton is not None:
            return _backend_singleton

        if config.REDIS_URL:
            try:
                _backend_singleton = RedisCacheBackend(config.REDIS_URL)
                return _backend_singleton
            except Exception:
                # Safe fallback for local/dev or transient redis errors.
                _backend_singleton = MemoryCacheBackend()
                return _backend_singleton

        _backend_singleton = MemoryCacheBackend()
        return _backend_singleton


def reset_cache_backend_for_tests() -> None:
    """Test helper to clear singleton cache backend."""
    global _backend_singleton
    with _backend_lock:
        _backend_singleton = None

