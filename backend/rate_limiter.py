"""
Small rate-limiter utility backed by shared cache.
"""

from __future__ import annotations

import time

from backend.cache_backend import get_cache_backend


def check_rate_limit(
    bucket: str,
    identity: str,
    limit_per_minute: int,
) -> tuple[bool, int]:
    if limit_per_minute <= 0:
        return True, 0

    minute_bucket = int(time.time() // 60)
    key = f"rl:{bucket}:{identity}:{minute_bucket}"
    cache = get_cache_backend()
    try:
        count = cache.incr(key, ttl_seconds=70)
        return count <= int(limit_per_minute), count
    except Exception:
        # Fail-open if cache backend is unavailable.
        return True, 0
