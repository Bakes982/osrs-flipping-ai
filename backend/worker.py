"""
Dedicated worker process entrypoint.

Run with:
    python -m backend.worker

This process is responsible for background loops only (polling, scoring,
alerts, pruning). It should run as a separate Railway worker service.
"""

from __future__ import annotations

import asyncio
import logging
import signal
from typing import Awaitable, Callable

from backend import config
from backend.core.logging import configure_logging
from backend.database import init_db
from backend.tasks import start_background_tasks, stop_background_tasks

logger = logging.getLogger(__name__)

StartFn = Callable[[], Awaitable[None]]
StopFn = Callable[[], Awaitable[None]]


def _install_signal_handlers(stop_event: asyncio.Event) -> None:
    """Install SIGINT/SIGTERM handlers that request a graceful shutdown."""
    loop = asyncio.get_running_loop()

    def _request_stop() -> None:
        if not stop_event.is_set():
            logger.info("Shutdown signal received, stopping worker...")
            stop_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _request_stop)
        except NotImplementedError:
            # Windows fallback
            signal.signal(sig, lambda _s, _f: _request_stop())


async def run_worker_iteration(
    start_fn: StartFn = start_background_tasks,
    stop_fn: StopFn = stop_background_tasks,
    runtime_seconds: float = 0.05,
) -> None:
    """
    Run one worker cycle.

    Useful for tests to verify worker start/stop wiring.
    """
    await start_fn()
    try:
        await asyncio.sleep(max(0.0, runtime_seconds))
    finally:
        await stop_fn()


async def _run_worker_session(
    stop_event: asyncio.Event,
    start_fn: StartFn,
    stop_fn: StopFn,
) -> None:
    """
    Start worker tasks and wait until shutdown is requested.
    """
    init_db()
    await start_fn()
    logger.info("Worker session started.")
    try:
        await stop_event.wait()
    finally:
        logger.info("Stopping worker background tasks...")
        await stop_fn()


async def run_worker_forever(
    stop_event: asyncio.Event,
    start_fn: StartFn = start_background_tasks,
    stop_fn: StopFn = stop_background_tasks,
) -> None:
    """
    Keep the worker alive with retry/backoff around session startup failures.
    """
    initial_backoff = max(0.5, config.WORKER_RETRY_INITIAL_SECONDS)
    max_backoff = max(initial_backoff, config.WORKER_RETRY_MAX_SECONDS)
    backoff = initial_backoff

    while not stop_event.is_set():
        try:
            await _run_worker_session(stop_event, start_fn, stop_fn)
            # Clean shutdown
            return
        except asyncio.CancelledError:
            raise
        except Exception:
            if stop_event.is_set():
                return
            logger.exception("Worker session crashed; retrying in %.1fs", backoff)
            await asyncio.sleep(backoff)
            backoff = min(max_backoff, backoff * 2.0)


async def main_async() -> None:
    if config.RUN_MODE != "worker":
        logger.warning(
            "backend.worker invoked with RUN_MODE=%s. Exiting without starting worker loops.",
            config.RUN_MODE,
        )
        return

    stop_event = asyncio.Event()
    _install_signal_handlers(stop_event)

    if config.WORKER_RUN_ONCE:
        logger.info(
            "Worker run-once mode enabled (WORKER_RUN_ONCE_SECONDS=%.2f).",
            config.WORKER_RUN_ONCE_SECONDS,
        )
        init_db()
        await run_worker_iteration(runtime_seconds=config.WORKER_RUN_ONCE_SECONDS)
        return

    logger.info("Starting worker in continuous mode.")
    await run_worker_forever(stop_event)


def main() -> None:
    configure_logging()
    asyncio.run(main_async())


if __name__ == "__main__":
    main()

