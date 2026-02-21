from __future__ import annotations

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

