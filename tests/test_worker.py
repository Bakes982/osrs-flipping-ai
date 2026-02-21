from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from backend import worker


@pytest.mark.asyncio
async def test_worker_iteration_runs_start_and_stop():
    start = AsyncMock()
    stop = AsyncMock()

    await worker.run_worker_iteration(start_fn=start, stop_fn=stop, runtime_seconds=0.0)

    start.assert_awaited_once()
    stop.assert_awaited_once()


@pytest.mark.asyncio
async def test_worker_main_exits_when_not_worker_mode(monkeypatch):
    monkeypatch.setattr(worker.config, "RUN_MODE", "api")
    monkeypatch.setattr(worker, "init_db", AsyncMock())
    start = AsyncMock()
    monkeypatch.setattr(worker, "start_background_tasks", start)

    await worker.main_async()

    start.assert_not_called()
