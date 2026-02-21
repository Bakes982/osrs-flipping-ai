from __future__ import annotations

import pytest

from backend.tasks import PriceCollector


class _FailingClient:
    def __init__(self):
        self.calls = 0

    async def get(self, _url: str):
        self.calls += 1
        raise RuntimeError("upstream unavailable")


@pytest.mark.asyncio
async def test_price_collector_opens_circuit_after_repeated_failures(monkeypatch):
    collector = PriceCollector()
    collector._failure_threshold = 2
    collector._circuit_open_seconds = 60
    client = _FailingClient()

    async def _fake_client():
        return client

    monkeypatch.setattr(collector, "_get_client", _fake_client)

    await collector.fetch_latest()
    assert collector._is_circuit_open() is False
    await collector.fetch_latest()
    assert collector._is_circuit_open() is True

    calls_before = client.calls
    await collector.fetch_latest()
    assert client.calls == calls_before

