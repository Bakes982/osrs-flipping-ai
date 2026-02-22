from __future__ import annotations

from backend.cache_backend import MemoryCacheBackend
from backend import rate_limiter


def test_check_rate_limit_blocks_after_threshold(monkeypatch):
    cache = MemoryCacheBackend()
    monkeypatch.setattr(rate_limiter, "get_cache_backend", lambda: cache)

    ok1, c1 = rate_limiter.check_rate_limit("top5", "key:abc", 2)
    ok2, c2 = rate_limiter.check_rate_limit("top5", "key:abc", 2)
    ok3, c3 = rate_limiter.check_rate_limit("top5", "key:abc", 2)

    assert ok1 is True and c1 == 1
    assert ok2 is True and c2 == 2
    assert ok3 is False and c3 == 3


def test_check_rate_limit_fails_open_when_cache_errors(monkeypatch):
    class BrokenCache:
        def incr(self, *_args, **_kwargs):
            raise RuntimeError("cache down")

    monkeypatch.setattr(rate_limiter, "get_cache_backend", lambda: BrokenCache())
    ok, count = rate_limiter.check_rate_limit("top5", "key:abc", 2)
    assert ok is True
    assert count == 0
