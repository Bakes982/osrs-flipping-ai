from __future__ import annotations

from types import SimpleNamespace

from backend import cache_backend


def test_memory_cache_backend_round_trip_and_ttl():
    backend = cache_backend.MemoryCacheBackend()
    backend.set("k", "v", ttl_seconds=1)
    assert backend.get("k") == "v"


def test_get_cache_backend_falls_back_to_memory(monkeypatch):
    monkeypatch.setattr(cache_backend.config, "REDIS_URL", "")
    cache_backend.reset_cache_backend_for_tests()
    backend = cache_backend.get_cache_backend()
    assert backend.backend == "memory"


def test_get_cache_backend_uses_redis_when_available(monkeypatch):
    class FakeRedisClient:
        def __init__(self):
            self.store = {}

        def get(self, key):
            return self.store.get(key)

        def set(self, key, value):
            self.store[key] = value

        def setex(self, key, _ttl, value):
            self.store[key] = value

    fake_client = FakeRedisClient()
    fake_redis_module = SimpleNamespace(from_url=lambda *_args, **_kwargs: fake_client)

    monkeypatch.setattr(cache_backend.config, "REDIS_URL", "redis://example")
    monkeypatch.setattr(cache_backend, "redis", fake_redis_module)
    cache_backend.reset_cache_backend_for_tests()

    backend = cache_backend.get_cache_backend()
    assert backend.backend == "redis"
    backend.set_json("sample", {"ok": True}, ttl_seconds=30)
    assert backend.get_json("sample") == {"ok": True}

