from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from starlette.requests import Request

from backend.api import routes
from backend.cache_backend import MemoryCacheBackend


def _request(path: str, client_host: str = "127.0.0.1") -> Request:
    scope = {
        "type": "http",
        "method": "GET",
        "path": path,
        "headers": [],
        "query_string": b"",
        "client": (client_host, 50000),
    }
    request = Request(scope)
    request.state.user_ctx = SimpleNamespace(risk_profile=None)
    return request


def _metric(item_id: int = 4151) -> dict:
    return {
        "item_id": item_id,
        "item_name": "Abyssal whip",
        "recommended_buy": 1_000_000,
        "recommended_sell": 1_050_000,
        "net_profit": 29_000,
        "margin_after_tax": 29_000,
        "roi_pct": 2.9,
        "total_score": 61.0,
        "final_score": 61.0,
        "confidence": 0.91,
        "risk_score": 2.1,
        "risk_level": "LOW",
        "score_volume": 75.0,
        "fill_probability": 0.88,
        "estimated_hold_time": 12,
        "est_fill_time_minutes": 12.0,
        "trend_score": 0.62,
        "decay_penalty": 0.07,
        "volatility_1h": 0.01,
        "volatility_24h": 0.03,
        "gp_per_hour": 145000.0,
        "expected_profit": 290000,
        "expected_profit_personal": 290000,
        "risk_adjusted_gp_per_hour": 120000.0,
        "risk_adjusted_gph_personal": 120000.0,
        "qty_suggested": 10,
        "trend": "UP",
        "vetoed": False,
    }


@pytest.mark.asyncio
async def test_top5_cache_only_does_not_trigger_live_compute(monkeypatch):
    memory = MemoryCacheBackend()
    memory.set_json(
        "flips:top5:balanced",
        {"ts": datetime.now(timezone.utc).isoformat(), "flips": [_metric()]},
        ttl_seconds=60,
    )
    monkeypatch.setattr(routes, "get_cache_backend", lambda: memory)
    monkeypatch.setattr(routes, "_authenticate_plugin_access", lambda *_args, **_kwargs: "anon:test")
    monkeypatch.setattr(routes, "_enforce_plugin_rate_limit", lambda *_args, **_kwargs: None)

    async def _should_not_run(*_args, **_kwargs):
        raise AssertionError("live compute should not run for /flips/top5")

    monkeypatch.setattr(routes, "compute_scored_opportunities", _should_not_run)

    response = await routes.get_top5_runelite(
        _request("/flips/top5"),
        profile="balanced",
        min_score=45.0,
        min_confidence=0.0,
    )
    assert len(response.flips) == 1
    assert response.profile_used == "balanced"
    assert response.cache_ts is not None
    assert response.flips[0].c == 91.0
    assert len(response.flips[0].reasons) >= 1
    assert len(response.flips[0].badges) >= 1


@pytest.mark.asyncio
async def test_top_uses_cache_when_not_fresh(monkeypatch):
    memory = MemoryCacheBackend()
    memory.set_json(
        "flips:top100:balanced",
        {"ts": datetime.now(timezone.utc).isoformat(), "flips": [_metric()]},
        ttl_seconds=60,
    )
    monkeypatch.setattr(routes, "get_cache_backend", lambda: memory)
    monkeypatch.setattr(routes, "_authenticate_plugin_access", lambda *_args, **_kwargs: "anon:test")
    monkeypatch.setattr(routes, "_enforce_plugin_rate_limit", lambda *_args, **_kwargs: None)

    async def _should_not_run(*_args, **_kwargs):
        raise AssertionError("live compute should not run when fresh=0")

    monkeypatch.setattr(routes, "compute_scored_opportunities", _should_not_run)

    response = await routes.get_top_flips(
        _request("/flips/top"),
        limit=20,
        profile="balanced",
        fresh=0,
        min_score=45.0,
        min_roi=0.0,
        min_confidence=0.0,
        max_risk=10.0,
        min_volume="LOW",
        min_price=0,
        max_price=0,
        sort_by="score",
    )
    assert response.count == 1
    assert response.profile_used == "balanced"
    assert response.cache_age_seconds is not None
    assert response.flips[0].confidence_pct == 91.0
    assert len(response.flips[0].reasons) >= 1
    assert len(response.flips[0].badges) >= 1


@pytest.mark.asyncio
async def test_top_fresh_rate_limit(monkeypatch):
    routes._reset_fresh_rate_limit_for_tests()
    monkeypatch.setattr(routes.config, "FLIPS_FRESH_MAX_PER_MINUTE", 1)
    memory = MemoryCacheBackend()
    monkeypatch.setattr(routes, "get_cache_backend", lambda: memory)
    monkeypatch.setattr(routes, "_authenticate_plugin_access", lambda *_args, **_kwargs: "anon:test")
    monkeypatch.setattr(routes, "_enforce_plugin_rate_limit", lambda *_args, **_kwargs: None)

    async def _live_compute(*_args, **_kwargs):
        return [_metric()]

    monkeypatch.setattr(routes, "compute_scored_opportunities", _live_compute)

    req = _request("/flips/top", client_host="1.2.3.4")
    first = await routes.get_top_flips(
        req,
        limit=20,
        profile="balanced",
        fresh=1,
        min_score=45.0,
        min_roi=0.0,
        min_confidence=0.0,
        max_risk=10.0,
        min_volume="LOW",
        min_price=0,
        max_price=0,
        sort_by="score",
    )
    assert first.count == 1

    with pytest.raises(HTTPException) as exc:
        await routes.get_top_flips(
            req,
            limit=20,
            profile="balanced",
            fresh=1,
            min_score=45.0,
            min_roi=0.0,
            min_confidence=0.0,
            max_risk=10.0,
            min_volume="LOW",
            min_price=0,
            max_price=0,
            sort_by="score",
        )
    assert exc.value.status_code == 429


def test_confidence_pct_helper_normalizes_to_100_scale():
    assert routes._confidence_pct({"confidence": 0.87}) == 87.0
    assert routes._confidence_pct({"confidence_pct": 92}) == 92.0
    assert routes._confidence_pct({"confidence_pct": 0.92}) == 92.0
