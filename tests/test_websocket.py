"""Tests for the WebSocket connection manager."""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.websocket import ConnectionManager


@pytest.fixture
def manager():
    return ConnectionManager()


@pytest.fixture
def mock_ws():
    ws = AsyncMock()
    ws.accept = AsyncMock()
    ws.send_text = AsyncMock()
    return ws


@pytest.mark.asyncio
async def test_connect(manager, mock_ws):
    await manager.connect(mock_ws)
    assert manager.client_count == 1
    mock_ws.accept.assert_called_once()


@pytest.mark.asyncio
async def test_disconnect(manager, mock_ws):
    await manager.connect(mock_ws)
    assert manager.client_count == 1
    await manager.disconnect(mock_ws)
    assert manager.client_count == 0


@pytest.mark.asyncio
async def test_disconnect_nonexistent(manager, mock_ws):
    # Should not raise
    await manager.disconnect(mock_ws)
    assert manager.client_count == 0


@pytest.mark.asyncio
async def test_broadcast_prices(manager, mock_ws):
    await manager.connect(mock_ws)
    data = {"type": "price_update", "data": {"4151": {"high": 100}}}
    await manager.broadcast_prices(data)
    mock_ws.send_text.assert_called_once_with(json.dumps(data))


@pytest.mark.asyncio
async def test_broadcast_removes_stale(manager):
    good_ws = AsyncMock()
    good_ws.accept = AsyncMock()
    good_ws.send_text = AsyncMock()

    bad_ws = AsyncMock()
    bad_ws.accept = AsyncMock()
    bad_ws.send_text = AsyncMock(side_effect=Exception("disconnected"))

    await manager.connect(good_ws)
    await manager.connect(bad_ws)
    assert manager.client_count == 2

    await manager.broadcast_prices({"type": "test"})
    assert manager.client_count == 1  # bad_ws removed


@pytest.mark.asyncio
async def test_multiple_clients(manager):
    clients = []
    for _ in range(5):
        ws = AsyncMock()
        ws.accept = AsyncMock()
        ws.send_text = AsyncMock()
        await manager.connect(ws)
        clients.append(ws)

    assert manager.client_count == 5

    await manager.broadcast_prices({"type": "test"})
    for ws in clients:
        ws.send_text.assert_called_once()
