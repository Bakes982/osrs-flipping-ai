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
import os
import signal
from typing import Awaitable, Callable

from backend import config
from backend.core.logging import configure_logging
from backend.database import init_db
from backend.flips_cache import warm_flip_caches
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
    cache_warm_task = asyncio.create_task(_cache_warm_loop(stop_event))
    logger.info("Worker session started.")
    try:
        await stop_event.wait()
    finally:
        cache_warm_task.cancel()
        await asyncio.gather(cache_warm_task, return_exceptions=True)
        logger.info("Stopping worker background tasks...")
        await stop_fn()


async def _cache_warm_loop(stop_event: asyncio.Event) -> None:
    """Warm top-list caches periodically so API endpoints remain cache-only fast."""
    interval_seconds = max(5, int(config.FLIPS_CACHE_WARM_INTERVAL_SECONDS))
    while not stop_event.is_set():
        try:
            warmed = await warm_flip_caches()
            logger.info("Cache warm complete: %s", warmed)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Cache warm failed")

        try:
            await asyncio.wait_for(stop_event.wait(), timeout=interval_seconds)
        except asyncio.TimeoutError:
            continue


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


async def _run_dump_smoke_test() -> None:
    """Resolve item 4151 and post a smoke-test message to the DUMPS webhook.

    Activated by setting DUMP_SMOKE_TEST=1 on the worker.  Used to verify
    that dump-v2 name resolution and the Discord webhook are wired correctly.
    """
    import requests as _requests

    from backend.alerts.item_name_resolver import resolve_item_name
    from backend.database import get_db, get_setting

    SMOKE_ITEM_ID = 4151  # Abyssal whip — well-known, always in the mapping

    # --- 1. Resolve the item name (same resolver used in real dump alerts) ---
    try:
        name = resolve_item_name(SMOKE_ITEM_ID)
    except Exception as exc:
        logger.warning("DUMP_SMOKE_TEST: resolver raised %s — using fallback", exc)
        name = f"Item {SMOKE_ITEM_ID}"

    logger.info("DUMP_SMOKE_TEST resolved %d -> %s", SMOKE_ITEM_ID, name)

    # --- 2. Fetch the dump webhook URL (same priority as AlertMonitor) ---
    def _get_dump_webhook() -> str | None:
        # env override is highest priority — same as _get_dump_alert_webhook_sync
        env_url = os.environ.get("DISCORD_WEBHOOK_DUMPS", "").strip()
        if env_url:
            return env_url
        init_db()
        db = get_db()
        try:
            url = get_setting(db, "dump_alert_webhook_url")
            if url:
                return str(url).strip()
            wh = get_setting(db, "discord_webhook")
            if isinstance(wh, dict):
                if wh.get("enabled", False) and wh.get("url"):
                    return str(wh["url"]).strip()
            url = get_setting(db, "discord_webhook_url")
            enabled = get_setting(db, "discord_alerts_enabled", False)
            if url and enabled:
                return str(url).strip()
        except Exception as exc:
            logger.warning("DUMP_SMOKE_TEST: webhook lookup failed: %s", exc)
        finally:
            db.close()
        return None

    webhook_url = await asyncio.to_thread(_get_dump_webhook)

    if not webhook_url:
        logger.warning("DUMP_SMOKE_TEST: no dump webhook URL found — skipping Discord send")
        return

    # --- 3. Send the smoke-test Discord message ---
    body = f"Resolved {SMOKE_ITEM_ID} -> {name}"
    embed = {
        "title": "DUMP SMOKE TEST",
        "description": body,
        "color": 0x00BFFF,
        "fields": [
            {"name": "Item ID", "value": str(SMOKE_ITEM_ID), "inline": True},
            {"name": "Resolved Name", "value": name, "inline": True},
        ],
        "footer": {"text": "OSRS Flipping AI • Smoke Test"},
    }

    def _send() -> None:
        try:
            resp = _requests.post(webhook_url, json={"embeds": [embed]}, timeout=10)
            if resp.status_code in (200, 204):
                logger.info("DUMP_SMOKE_TEST: Discord message sent OK (HTTP %d)", resp.status_code)
            else:
                logger.error(
                    "DUMP_SMOKE_TEST: Discord returned HTTP %d: %s",
                    resp.status_code,
                    resp.text[:300],
                )
        except Exception as exc:
            logger.error("DUMP_SMOKE_TEST: Discord send failed: %s", exc)

    await asyncio.to_thread(_send)


async def main_async() -> None:
    if config.RUN_MODE != "worker":
        logger.warning(
            "backend.worker invoked with RUN_MODE=%s. Exiting without starting worker loops.",
            config.RUN_MODE,
        )
        return

    # Version stamp — visible in Railway logs on every deploy
    try:
        import subprocess as _sp
        _git_hash = _sp.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=_sp.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        _git_hash = "unknown"
    logger.info("APP_VERSION dump-v2 commit=%s", _git_hash)

    if os.environ.get("DUMP_SMOKE_TEST"):
        await _run_dump_smoke_test()

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
