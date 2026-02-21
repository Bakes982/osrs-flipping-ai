from __future__ import annotations

import time

from backend.metrics import (
    increment_alert_sent_count,
    metrics_snapshot,
    record_cache_access,
    record_error,
    reset_metrics_for_tests,
)


def test_metrics_snapshot_counts_and_rates():
    reset_metrics_for_tests()
    record_cache_access(True)
    record_cache_access(True)
    record_cache_access(False)
    increment_alert_sent_count(2)
    record_error(time.time() - 4000)  # pruned from 1h window
    record_error(time.time())

    snap = metrics_snapshot()
    assert snap["cache_hit_rate"] == 0.6667
    assert snap["alert_sent_count"] == 2
    assert snap["errors_last_hour"] == 1
