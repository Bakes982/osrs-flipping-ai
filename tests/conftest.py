"""
Pytest configuration and shared fixtures — Phase 9.

Fixtures available to all tests:
  • mock_snapshot(...)     — create a single PriceSnapshot mock
  • make_snapshots(n, ...) — create a list of N snapshots
  • profitable_item_data   — standard profitable flip item_data dict
"""

from __future__ import annotations

import os
import sys
import time
from datetime import datetime, timedelta
from typing import List
from unittest.mock import MagicMock

import pytest

# Ensure the project root is on the path so all backend imports resolve.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ---------------------------------------------------------------------------
# Snapshot factory
# ---------------------------------------------------------------------------

def _snap(
    item_id: int = 4151,
    instant_buy: int = 1_050_000,
    instant_sell: int = 1_000_000,
    buy_volume: int = 50,
    sell_volume: int = 50,
    age_seconds: int = 0,
) -> MagicMock:
    s = MagicMock()
    s.item_id = item_id
    s.instant_buy = instant_buy
    s.instant_sell = instant_sell
    s.buy_volume = buy_volume
    s.sell_volume = sell_volume
    s.timestamp = datetime.utcnow() - timedelta(seconds=age_seconds)
    s.buy_time = int(time.time()) - age_seconds
    s.sell_time = int(time.time()) - age_seconds
    return s


@pytest.fixture
def mock_snapshot():
    return _snap


@pytest.fixture
def make_snapshots():
    def _factory(n: int = 20, **kwargs) -> List[MagicMock]:
        return [
            _snap(age_seconds=i * 10, **kwargs)
            for i in range(n - 1, -1, -1)
        ]
    return _factory


@pytest.fixture
def profitable_item_data(make_snapshots):
    snaps = make_snapshots(
        n=20, instant_buy=1_050_000, instant_sell=1_000_000,
        buy_volume=25, sell_volume=25,
    )
    return {
        "item_id": 4151,
        "item_name": "Abyssal whip",
        "instant_buy": 1_050_000,
        "instant_sell": 1_000_000,
        "volume_5m": 50,
        "buy_time": int(time.time()),
        "sell_time": int(time.time()),
        "snapshots": snaps,
        "flip_history": [],
    }


# ---------------------------------------------------------------------------
# asyncio — required for pytest-asyncio
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def event_loop():
    import asyncio
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()
