from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from starlette.requests import Request

from backend.routers import opportunities


class _FakeRedis:
    def __init__(self, payload: dict):
        self._raw = json.dumps(payload)

    def get(self, _key):
        return self._raw

    def ttl(self, _key):
        return 60


def _request(path: str, query: str = "") -> Request:
    scope = {
        "type": "http",
        "method": "GET",
        "path": path,
        "headers": [],
        "query_string": query.encode("utf-8"),
        "client": ("127.0.0.1", 50000),
    }
    request = Request(scope)
    request.state.user_ctx = SimpleNamespace(risk_profile=None)
    return request


def _cached_payload() -> dict:
    return {
        "generated_at": 1_700_000_000,
        "items": [
            {
                "item_id": 100,
                "name": "Low value item",
                "buy_price": 259,
                "sell_price": 300,
                "margin_gp": 41,
                "qty_suggested": 50,
                "potential_profit": 500,
                "volume_5m": 5000,
                "roi_pct": 15.8,
                "flip_score": 80,
            },
            {
                "item_id": 200,
                "name": "One mil item",
                "buy_price": 1_500_000,
                "sell_price": 1_490_000,
                "margin_gp": 10_000,
                "qty_suggested": 5,
                "potential_profit": 50_000,
                "volume_5m": 100,
                "roi_pct": 0.67,
                "flip_score": 60,
            },
            {
                "item_id": 300,
                "name": "Ten mil by sell item",
                "buy_price": 9_900_000,
                "sell_price": 10_100_000,
                "margin_gp": 200_000,
                "qty_suggested": 1,
                "potential_profit": 200_000,
                "volume_5m": 10,
                "roi_pct": 2.0,
                "flip_score": 70,
            },
        ],
    }


@pytest.mark.asyncio
async def test_value_mode_1m_filters_on_buy_or_sell(monkeypatch, caplog):
    monkeypatch.setattr(opportunities, "get_redis", lambda: _FakeRedis(_cached_payload()))
    req = _request("/api/opportunities", "profile=balanced&value_mode=1m")

    with caplog.at_level("INFO"):
        result = await opportunities.list_opportunities(
            req,
            profile="balanced",
            limit=100,
            value_mode="1m",
            min_price=0,
            min_price_gp=0,
            min_volume=0,
            min_roi_pct=0,
            min_profit_gp=0,
            min_profit_per_item_gp=0,
            min_total_profit_gp=0,
            ignore_low_value=False,
        )

    ids = [row["item_id"] for row in result["items"]]
    assert 100 not in ids
    assert 200 in ids
    assert 300 in ids
    assert result["value_mode"] == "1m"
    assert any("OPP_VALUE_FILTER" in rec.message for rec in caplog.records)


@pytest.mark.asyncio
async def test_value_mode_10m_filters_on_max_buy_sell(monkeypatch):
    monkeypatch.setattr(opportunities, "get_redis", lambda: _FakeRedis(_cached_payload()))
    req = _request("/api/opportunities", "profile=balanced&value_mode=10m")

    result = await opportunities.list_opportunities(
        req,
        profile="balanced",
        limit=100,
        value_mode="10m",
        min_price=0,
        min_price_gp=0,
        min_volume=0,
        min_roi_pct=0,
        min_profit_gp=0,
        min_profit_per_item_gp=0,
        min_total_profit_gp=0,
        ignore_low_value=False,
    )
    ids = [row["item_id"] for row in result["items"]]
    assert ids == [300]
    assert result["value_mode"] == "10m"
