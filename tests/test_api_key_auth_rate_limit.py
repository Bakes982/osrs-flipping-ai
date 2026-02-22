from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from starlette.requests import Request

from backend.api import routes
from backend.cache_backend import MemoryCacheBackend


def _request(path: str, api_key: str | None = None) -> Request:
    headers = []
    if api_key is not None:
        headers.append((b"x-api-key", api_key.encode("utf-8")))
    scope = {
        "type": "http",
        "method": "GET",
        "path": path,
        "headers": headers,
        "query_string": b"",
        "client": ("127.0.0.1", 50000),
    }
    request = Request(scope)
    request.state.user_ctx = SimpleNamespace(risk_profile=None)
    return request


def _cached_metric() -> dict:
    return {
        "item_id": 4151,
        "item_name": "Abyssal whip",
        "recommended_buy": 1_000_000,
        "recommended_sell": 1_050_000,
        "net_profit": 29_000,
        "roi_pct": 2.9,
        "total_score": 61.0,
        "confidence": 0.91,
        "risk_level": "LOW",
    }


@pytest.mark.asyncio
async def test_top5_invalid_key_returns_401(monkeypatch):
    cache = MemoryCacheBackend()
    cache.set_json(
        "flips:top5:balanced",
        {"ts": datetime.now(timezone.utc).isoformat(), "flips": [_cached_metric()]},
        ttl_seconds=60,
    )
    monkeypatch.setattr(routes, "get_cache_backend", lambda: cache)
    monkeypatch.setattr(routes.config, "ALLOW_ANON", False)
    monkeypatch.setattr(routes, "resolve_api_key_owner", lambda _k: None)
    monkeypatch.setattr(routes, "check_rate_limit", lambda **_kwargs: (True, 1))

    with pytest.raises(HTTPException) as exc:
        await routes.get_top5_runelite(_request("/flips/top5", api_key="bad"), profile="balanced")
    assert exc.value.status_code == 401


@pytest.mark.asyncio
async def test_top5_valid_key_returns_200(monkeypatch):
    cache = MemoryCacheBackend()
    cache.set_json(
        "flips:top5:balanced",
        {"ts": datetime.now(timezone.utc).isoformat(), "flips": [_cached_metric()]},
        ttl_seconds=60,
    )
    monkeypatch.setattr(routes, "get_cache_backend", lambda: cache)
    monkeypatch.setattr(routes.config, "ALLOW_ANON", False)
    monkeypatch.setattr(routes, "resolve_api_key_owner", lambda _k: ("hash123", "user1"))
    monkeypatch.setattr(routes, "check_rate_limit", lambda **_kwargs: (True, 1))

    response = await routes.get_top5_runelite(_request("/flips/top5", api_key="valid"), profile="balanced")
    assert len(response.flips) == 1


@pytest.mark.asyncio
async def test_top5_rate_limit_exceeded_returns_429(monkeypatch):
    cache = MemoryCacheBackend()
    cache.set_json(
        "flips:top5:balanced",
        {"ts": datetime.now(timezone.utc).isoformat(), "flips": [_cached_metric()]},
        ttl_seconds=60,
    )
    monkeypatch.setattr(routes, "get_cache_backend", lambda: cache)
    monkeypatch.setattr(routes.config, "ALLOW_ANON", False)
    monkeypatch.setattr(routes, "resolve_api_key_owner", lambda _k: ("hash123", "user1"))
    monkeypatch.setattr(routes, "check_rate_limit", lambda **_kwargs: (False, 61))

    with pytest.raises(HTTPException) as exc:
        await routes.get_top5_runelite(_request("/flips/top5", api_key="valid"), profile="balanced")
    assert exc.value.status_code == 429

