from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from backend import flips_cache
from backend.cache_backend import get_cache_backend, reset_cache_backend_for_tests


def test_warm_flip_caches_writes_expected_keys(monkeypatch):
    reset_cache_backend_for_tests()

    def _fake_compute(*_args, **_kwargs):
        return [
            {"item_id": 1, "item_name": "Item 1", "total_score": 60.0},
            {"item_id": 2, "item_name": "Item 2", "total_score": 55.0},
        ]

    monkeypatch.setattr(flips_cache, "compute_scored_opportunities_sync", _fake_compute)

    warmed = flips_cache.warm_flip_caches_sync(profiles=("balanced",))
    assert warmed["balanced"] == 2

    cache = get_cache_backend()
    top5 = cache.get_json("flips:top5:balanced")
    top100 = cache.get_json("flips:top100:balanced")
    last_updated = cache.get("flips:last_updated_ts")

    assert isinstance(top5, dict)
    assert isinstance(top100, dict)
    assert isinstance(last_updated, str)
    assert len(top5["flips"]) == 2
    assert len(top100["flips"]) == 2


def test_compute_scored_opportunities_skips_stale_snapshots(monkeypatch):
    class FakeDB:
        def close(self):
            return None

    stale_snapshot = SimpleNamespace(
        instant_buy=1050,
        instant_sell=1000,
        buy_volume=10,
        sell_volume=10,
        buy_time=0,
        sell_time=0,
        timestamp=datetime.now(timezone.utc) - timedelta(minutes=120),
    )

    monkeypatch.setattr(flips_cache, "get_db", lambda: FakeDB())
    monkeypatch.setattr(flips_cache, "get_tracked_item_ids", lambda _db: [4151])
    monkeypatch.setattr(flips_cache, "get_price_history", lambda _db, _item_id, hours=4: [stale_snapshot])
    monkeypatch.setattr(flips_cache, "get_item_flips", lambda _db, _item_id, days=30: [])
    monkeypatch.setattr(flips_cache, "get_item", lambda _db, _item_id: SimpleNamespace(name="Whip", buy_limit=70))
    monkeypatch.setattr(flips_cache, "calculate_flip_metrics", lambda _data: {"vetoed": False, "total_score": 90.0, "confidence": 0.9})
    monkeypatch.setattr(flips_cache.config, "SCORE_STALE_MAX_MINUTES", 45)

    results = flips_cache.compute_scored_opportunities_sync(limit=10, min_score=0, profile="balanced")
    assert results == []
