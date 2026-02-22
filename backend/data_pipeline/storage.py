"""
OSRS Flipping AI — Price snapshot writer.

Batches PriceSnapshot objects and writes them to MongoDB, with support for
write throttling, deduplication, and time-series schema semantics.

This is a thin, focused write layer on top of database.insert_price_snapshots.
It is deliberately separate from the fetch pipeline so that each concern can
be tested and scaled independently.
"""

from __future__ import annotations

import logging
import time
from typing import Dict, List, Optional

from backend.core.constants import SNAPSHOT_STORE_INTERVAL, SNAPSHOT_TOP_N_ITEMS

logger = logging.getLogger(__name__)


class SnapshotWriter:
    """Collects PriceSnapshot objects and flushes them to MongoDB in batches.

    Usage::

        writer = SnapshotWriter()
        writer.queue(snap)
        ...
        written = writer.flush()   # returns number of rows written

    The writer silently skips the flush when the configured ``store_interval``
    has not elapsed since the last successful write (write throttling).  Call
    ``flush(force=True)`` to bypass throttling.
    """

    def __init__(
        self,
        store_interval: int = SNAPSHOT_STORE_INTERVAL,
        top_n: int = SNAPSHOT_TOP_N_ITEMS,
    ) -> None:
        self._queue: List["PriceSnapshot"] = []  # noqa: F821
        self._store_interval = store_interval
        self._top_n = top_n
        self._last_store: float = 0.0
        # Track last snapshot per item for deduplication
        self._last_snap: Dict[int, "PriceSnapshot"] = {}  # noqa: F821

    # ------------------------------------------------------------------
    # Queueing
    # ------------------------------------------------------------------

    def queue(self, snap: "PriceSnapshot") -> None:  # noqa: F821
        """Add a snapshot to the write queue."""
        self._queue.append(snap)

    def queue_batch(self, snaps: List["PriceSnapshot"]) -> None:  # noqa: F821
        """Add a list of snapshots to the write queue."""
        self._queue.extend(snaps)

    def pending_count(self) -> int:
        """Number of snapshots waiting to be written."""
        return len(self._queue)

    # ------------------------------------------------------------------
    # Flushing
    # ------------------------------------------------------------------

    def flush(self, force: bool = False) -> int:
        """Write all queued snapshots to MongoDB.

        Parameters
        ----------
        force:
            Bypass the write-throttle interval.

        Returns
        -------
        int
            Number of rows written (0 if throttled or queue is empty).
        """
        if not self._queue:
            return 0

        if not force and (time.time() - self._last_store) < self._store_interval:
            return 0

        # Take a local snapshot of the queue and reset it
        batch = self._queue[:]
        self._queue.clear()

        # Deduplicate: keep only the latest snapshot per item
        per_item: Dict[int, "PriceSnapshot"] = {}  # noqa: F821
        for snap in batch:
            prev = per_item.get(snap.item_id)
            if prev is None or snap.timestamp > prev.timestamp:
                per_item[snap.item_id] = snap

        # Keep top-N by combined volume
        ranked = sorted(
            per_item.values(),
            key=lambda s: (s.buy_volume or 0) + (s.sell_volume or 0),
            reverse=True,
        )
        to_write = ranked[: self._top_n]

        if not to_write:
            return 0

        written = self._write_to_db(to_write)
        if written > 0:
            self._last_store = time.time()
            # Update last-snap tracking for deduplication on next cycle
            for snap in to_write:
                self._last_snap[snap.item_id] = snap

        return written

    # ------------------------------------------------------------------
    # Internal DB write
    # ------------------------------------------------------------------

    @staticmethod
    def _write_to_db(snaps: List["PriceSnapshot"]) -> int:  # noqa: F821
        """Synchronous DB write.  Must be called from a thread pool when in
        an async context (use ``asyncio.to_thread(writer.flush)``).
        """
        from backend.database import get_db, insert_price_snapshots

        db = get_db()
        try:
            insert_price_snapshots(db, snaps)
            return len(snaps)
        except Exception as exc:
            logger.error("SnapshotWriter: DB write failed: %s", exc)
            return 0
        finally:
            db.close()
