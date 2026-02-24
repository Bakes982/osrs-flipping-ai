from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from backend.routers import items


class _FakeCursor:
    def __init__(self, docs):
        self._docs = list(docs)
        self._limit = None

    def limit(self, n: int):
        self._limit = n
        return self

    def __iter__(self):
        docs = self._docs if self._limit is None else self._docs[: self._limit]
        return iter(docs)


class _FakeCollection:
    def __init__(self, docs):
        self._docs = list(docs)

    def find(self, *_args, **_kwargs):
        return _FakeCursor(self._docs)


class _FakeDb:
    def __init__(self, item_docs=None, trade_docs=None):
        self.items = _FakeCollection(item_docs or [])
        self.trades = _FakeCollection(trade_docs or [])

    def close(self):
        return None


class _FakeRedis:
    def __init__(self, seed=None):
        self._store = dict(seed or {})

    def get(self, key):
        return self._store.get(key)

    def set(self, key, value, ex=None):
        self._store[key] = value


def test_search_items_supports_numeric_exact_and_name(monkeypatch):
    mapping = {4151: "Abyssal whip", 4153: "Granite maul"}
    fake_db = _FakeDb(
        item_docs=[{"_id": 4151, "name": "Item 4151"}],
        trade_docs=[{"item_id": 4151, "item_name": "Abyssal whip"}],
    )

    monkeypatch.setattr(items, "_load_wiki_mapping", lambda: mapping)
    monkeypatch.setattr(items, "get_db", lambda: fake_db)
    monkeypatch.setattr(
        items,
        "get_item",
        lambda _db, item_id: SimpleNamespace(id=item_id, name="Item 4151"),
    )
    monkeypatch.setattr(
        items,
        "_clean_name",
        lambda item_id, fallback=None: mapping.get(item_id, fallback or f"Item {item_id}"),
    )

    by_id = items._search_items_sync("4151", 20)
    assert len(by_id) >= 1
    assert by_id[0]["item_id"] == 4151
    assert by_id[0]["name"] == "Abyssal whip"

    by_name = items._search_items_sync("whip", 20)
    ids = [row["item_id"] for row in by_name]
    assert 4151 in ids


def test_search_items_empty_query_returns_empty_list():
    assert items._search_items_sync("", 20) == []


@pytest.mark.asyncio
async def test_item_graph_cache_hit(monkeypatch):
    key = "item_graph:4151:range:24h"
    cached_payload = {
        "item_id": 4151,
        "name": "Abyssal whip",
        "latest": {
            "buy_price": 2,
            "sell_price": 1,
            "margin_gp": 1,
            "volume_5m": 123,
            "trend": "up",
            "updated_at": 1,
        },
        "points": [{"ts": 1, "buy": 2, "sell": 1, "high": 2, "low": 1, "volume": 123}],
    }
    redis = _FakeRedis({key: json.dumps(cached_payload)})
    monkeypatch.setattr(items, "get_redis", lambda: redis)

    result = await items.get_item_graph(4151, range="24h")
    assert result["item_id"] == 4151
    assert result["latest"]["margin_gp"] == 1


@pytest.mark.asyncio
async def test_item_graph_cache_miss_writes_cache(monkeypatch):
    redis = _FakeRedis()
    monkeypatch.setattr(items, "get_redis", lambda: redis)
    monkeypatch.setattr(
        items,
        "_build_item_graph",
        lambda _item_id, _range: {
            "item_id": 4151,
            "name": "Abyssal whip",
            "latest": {
                "buy_price": 100,
                "sell_price": 90,
                "margin_gp": 10,
                "volume_5m": 200,
                "trend": "up",
                "updated_at": 123,
            },
            "points": [{"ts": 1, "buy": 100, "sell": 90, "high": 100, "low": 90, "volume": 200}],
        },
    )

    result = await items.get_item_graph(4151, range="24h")
    assert len(result["points"]) == 1
    assert redis.get("item_graph:4151:range:24h") is not None


def test_build_item_graph_404_when_missing(monkeypatch):
    monkeypatch.setattr(items, "_load_wiki_mapping", lambda: {})
    monkeypatch.setattr(items, "get_db", lambda: _FakeDb(item_docs=[], trade_docs=[]))
    monkeypatch.setattr(items, "get_item", lambda _db, _item_id: None)
    monkeypatch.setattr(items, "get_graph_points", lambda _item_id, _range: [])
    monkeypatch.setattr(items, "get_latest_quote", lambda _item_id: None)
    with pytest.raises(HTTPException):
        items._build_item_graph(999999, "24h")
