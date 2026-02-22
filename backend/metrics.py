"""
Lightweight runtime metrics for health/observability.

Uses in-process counters so it works without extra dependencies.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import Deque, Dict


class _RuntimeMetrics:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._cache_hits = 0
        self._cache_misses = 0
        self._alert_sent_count = 0
        self._error_timestamps: Deque[float] = deque()

    def record_cache_access(self, hit: bool) -> None:
        with self._lock:
            if hit:
                self._cache_hits += 1
            else:
                self._cache_misses += 1

    def increment_alert_sent_count(self, amount: int = 1) -> None:
        with self._lock:
            self._alert_sent_count += max(0, int(amount))

    def record_error(self, ts: float | None = None) -> None:
        now = ts if ts is not None else time.time()
        with self._lock:
            self._error_timestamps.append(now)
            self._prune_locked(now)

    def snapshot(self) -> Dict[str, float | int]:
        now = time.time()
        with self._lock:
            self._prune_locked(now)
            total = self._cache_hits + self._cache_misses
            cache_hit_rate = (self._cache_hits / total) if total > 0 else 0.0
            return {
                "cache_hit_rate": round(cache_hit_rate, 4),
                "alert_sent_count": self._alert_sent_count,
                "errors_last_hour": len(self._error_timestamps),
            }

    def reset_for_tests(self) -> None:
        with self._lock:
            self._cache_hits = 0
            self._cache_misses = 0
            self._alert_sent_count = 0
            self._error_timestamps.clear()

    def _prune_locked(self, now: float) -> None:
        cutoff = now - 3600.0
        while self._error_timestamps and self._error_timestamps[0] < cutoff:
            self._error_timestamps.popleft()


_METRICS = _RuntimeMetrics()


def record_cache_access(hit: bool) -> None:
    _METRICS.record_cache_access(hit)


def increment_alert_sent_count(amount: int = 1) -> None:
    _METRICS.increment_alert_sent_count(amount)


def record_error(ts: float | None = None) -> None:
    _METRICS.record_error(ts)


def metrics_snapshot() -> Dict[str, float | int]:
    return _METRICS.snapshot()


def reset_metrics_for_tests() -> None:
    _METRICS.reset_for_tests()

