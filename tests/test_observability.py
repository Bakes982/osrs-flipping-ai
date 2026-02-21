from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest
from fastapi.responses import Response
from starlette.requests import Request

from backend.api import routes
from backend.app import request_logging_middleware
from backend.cache_backend import MemoryCacheBackend
from backend.metrics import metrics_snapshot, reset_metrics_for_tests


class _FakeDBWrapper:
    def __init__(self):
        self._db = self

    def command(self, _name: str):
        return {"ok": 1}

    def close(self):
        return None


@pytest.mark.asyncio
async def test_health_check_returns_observability_fields(monkeypatch):
    cache = MemoryCacheBackend()
    now_iso = datetime.now(timezone.utc).isoformat()
    cache.set("flips:last_updated_ts", now_iso, ttl_seconds=60)
    cache.set_json("flips:stats:conservative", {"count": 3}, ttl_seconds=60)
    cache.set_json("flips:stats:balanced", {"count": 4}, ttl_seconds=60)
    cache.set_json("flips:stats:aggressive", {"count": 5}, ttl_seconds=60)

    monkeypatch.setattr(routes, "get_cache_backend", lambda: cache)
    monkeypatch.setattr(routes, "metrics_snapshot", lambda: {"cache_hit_rate": 0.5, "alert_sent_count": 2, "errors_last_hour": 1})
    monkeypatch.setattr("backend.database.get_db", lambda: _FakeDBWrapper())
    monkeypatch.setattr("backend.tasks._tasks", [1, 2])

    health = await routes.health_check()
    assert health.db_connected is True
    assert health.cache_backend == "memory"
    assert health.items_scored_count_last_run == 12
    assert health.last_poll_ts is not None
    assert health.cache_hit_rate == 0.5
    assert health.alert_sent_count == 2
    assert health.errors_last_hour == 1


@pytest.mark.asyncio
async def test_request_logging_middleware_logs_structured_payload(caplog):
    reset_metrics_for_tests()
    scope = {
        "type": "http",
        "method": "GET",
        "path": "/flips/top5",
        "headers": [],
        "query_string": b"",
        "client": ("127.0.0.1", 50000),
        "scheme": "http",
        "server": ("testserver", 80),
    }
    request = Request(scope)
    request.state.cache_hit = True
    request.state.profile_used = "balanced"

    async def _ok(_request: Request) -> Response:
        return Response(status_code=200)

    with caplog.at_level("INFO"):
        response = await request_logging_middleware(request, _ok)

    assert response.headers.get("X-Request-ID")
    line = next(msg for msg in caplog.messages if msg.startswith("request_log "))
    payload = json.loads(line.split(" ", 1)[1])
    assert payload["path"] == "/flips/top5"
    assert payload["cache_hit"] is True
    assert payload["profile"] == "balanced"


@pytest.mark.asyncio
async def test_request_logging_middleware_counts_server_errors():
    reset_metrics_for_tests()
    scope = {
        "type": "http",
        "method": "GET",
        "path": "/boom",
        "headers": [],
        "query_string": b"",
        "client": ("127.0.0.1", 50000),
        "scheme": "http",
        "server": ("testserver", 80),
    }
    request = Request(scope)

    async def _fail(_request: Request) -> Response:
        return Response(status_code=500)

    await request_logging_middleware(request, _fail)
    assert metrics_snapshot()["errors_last_hour"] == 1

