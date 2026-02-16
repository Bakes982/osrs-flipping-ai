"""
Background Tasks for OSRS Flipping AI
- PriceCollector: fetches prices every 10s, 5m data every 60s
- FeatureComputer: recomputes ML features every 60s (stub)
- MLScorer: runs predictions every 60s (stub)
- DataPruner: aggregates old data once per day
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, Optional

import httpx
from sqlalchemy import func

from backend.database import (
    SessionLocal,
    PriceSnapshot,
    PriceAggregate,
    Item,
    get_tracked_item_ids,
)
from backend.websocket import manager

logger = logging.getLogger(__name__)

WIKI_BASE = "https://prices.runescape.wiki/api/v1/osrs"
USER_AGENT = "OSRS-AI-Flipper v2.0 - Discord: bakes982"
HEADERS = {"User-Agent": USER_AGENT}


# ---------------------------------------------------------------------------
# PriceCollector
# ---------------------------------------------------------------------------

class PriceCollector:
    """Fetches /latest every 10 seconds and /5m every 60 seconds.

    Stores PriceSnapshot rows and broadcasts new data to WebSocket clients.
    """

    def __init__(self):
        self._latest_data: Dict = {}
        self._5m_data: Dict = {}
        self._last_5m_fetch: float = 0.0
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(headers=HEADERS, timeout=15.0)
        return self._client

    async def fetch_latest(self) -> Dict:
        """Fetch instant prices from /latest."""
        client = await self._get_client()
        try:
            resp = await client.get(f"{WIKI_BASE}/latest")
            resp.raise_for_status()
            self._latest_data = resp.json().get("data", {})
            return self._latest_data
        except Exception as e:
            logger.error("Failed to fetch /latest: %s", e)
            return self._latest_data

    async def fetch_5m(self) -> Dict:
        """Fetch 5-minute averaged prices from /5m."""
        client = await self._get_client()
        try:
            resp = await client.get(f"{WIKI_BASE}/5m")
            resp.raise_for_status()
            self._5m_data = resp.json().get("data", {})
            self._last_5m_fetch = time.time()
            return self._5m_data
        except Exception as e:
            logger.error("Failed to fetch /5m: %s", e)
            return self._5m_data

    async def store_snapshots(self) -> int:
        """Store current price data as PriceSnapshot rows.

        Returns the number of rows inserted.
        """
        if not self._latest_data:
            return 0

        now = datetime.utcnow()
        db = SessionLocal()
        count = 0
        try:
            for item_id_str, instant in self._latest_data.items():
                item_id = int(item_id_str)
                avg = self._5m_data.get(item_id_str, {})

                snap = PriceSnapshot(
                    item_id=item_id,
                    timestamp=now,
                    instant_buy=instant.get("high"),
                    instant_sell=instant.get("low"),
                    buy_time=instant.get("highTime"),
                    sell_time=instant.get("lowTime"),
                    avg_buy=avg.get("avgHighPrice"),
                    avg_sell=avg.get("avgLowPrice"),
                    buy_volume=avg.get("highPriceVolume", 0),
                    sell_volume=avg.get("lowPriceVolume", 0),
                )
                db.add(snap)
                count += 1

            db.commit()
        except Exception as e:
            db.rollback()
            logger.error("Error storing snapshots: %s", e)
        finally:
            db.close()

        return count

    async def broadcast(self):
        """Push latest prices to all connected WebSocket clients."""
        if not self._latest_data:
            return

        # Build a lightweight payload for the frontend
        payload = {
            "type": "price_update",
            "timestamp": datetime.utcnow().isoformat(),
            "data": {},
        }
        for item_id_str, instant in self._latest_data.items():
            avg = self._5m_data.get(item_id_str, {})
            payload["data"][item_id_str] = {
                "high": instant.get("high"),
                "low": instant.get("low"),
                "highTime": instant.get("highTime"),
                "lowTime": instant.get("lowTime"),
                "avgHigh": avg.get("avgHighPrice"),
                "avgLow": avg.get("avgLowPrice"),
                "buyVol": avg.get("highPriceVolume", 0),
                "sellVol": avg.get("lowPriceVolume", 0),
            }

        await manager.broadcast_prices(payload)

    async def run_forever(self):
        """Main loop: /latest every 10s, /5m every 60s."""
        logger.info("PriceCollector started")
        while True:
            try:
                await self.fetch_latest()

                # Fetch 5m data every 60 seconds
                if time.time() - self._last_5m_fetch >= 60:
                    await self.fetch_5m()

                await self.store_snapshots()
                await self.broadcast()
            except Exception as e:
                logger.error("PriceCollector tick error: %s", e)

            await asyncio.sleep(10)


# ---------------------------------------------------------------------------
# FeatureComputer (stub)
# ---------------------------------------------------------------------------

class FeatureComputer:
    """Recomputes ML feature vectors for tracked items every 60 seconds.

    This is a stub -- the actual feature engineering will be added by the
    ML pipeline later.
    """

    async def compute_features(self):
        """Placeholder: compute and cache features for all tracked items."""
        db = SessionLocal()
        try:
            item_ids = get_tracked_item_ids(db)
            # TODO: for each item_id, compute feature vector and store in ItemFeature table
            logger.debug("FeatureComputer: %d items tracked (stub)", len(item_ids))
        finally:
            db.close()

    async def run_forever(self):
        logger.info("FeatureComputer started (stub)")
        while True:
            try:
                await self.compute_features()
            except Exception as e:
                logger.error("FeatureComputer tick error: %s", e)
            await asyncio.sleep(60)


# ---------------------------------------------------------------------------
# MLScorer (stub)
# ---------------------------------------------------------------------------

class MLScorer:
    """Runs ML model predictions for all tracked items every 60 seconds.

    This is a stub -- the real inference code will be plugged in once
    models are trained.
    """

    async def score_items(self):
        """Placeholder: run predictions for tracked items."""
        db = SessionLocal()
        try:
            item_ids = get_tracked_item_ids(db)
            # TODO: load model, run inference, store Prediction rows
            logger.debug("MLScorer: %d items to score (stub)", len(item_ids))
        finally:
            db.close()

    async def run_forever(self):
        logger.info("MLScorer started (stub)")
        while True:
            try:
                await self.score_items()
            except Exception as e:
                logger.error("MLScorer tick error: %s", e)
            await asyncio.sleep(60)


# ---------------------------------------------------------------------------
# DataPruner
# ---------------------------------------------------------------------------

class DataPruner:
    """Aggregates and prunes old price data once per day.

    - Snapshots older than 7 days -> 5m candles in PriceAggregate
    - PriceAggregate 5m rows older than 30 days -> 1h candles
    - Deletes raw snapshots older than 7 days after aggregation
    """

    async def _aggregate_snapshots_to_5m(self, db):
        """Aggregate raw snapshots older than 7 days into 5-minute candles."""
        cutoff = datetime.utcnow() - timedelta(days=7)

        # Get distinct item_ids with old data
        item_ids = (
            db.query(PriceSnapshot.item_id)
            .filter(PriceSnapshot.timestamp < cutoff)
            .distinct()
            .all()
        )

        total_created = 0
        for (item_id,) in item_ids:
            snapshots = (
                db.query(PriceSnapshot)
                .filter(
                    PriceSnapshot.item_id == item_id,
                    PriceSnapshot.timestamp < cutoff,
                )
                .order_by(PriceSnapshot.timestamp.asc())
                .all()
            )

            if not snapshots:
                continue

            # Group into 5-minute buckets
            buckets: Dict[datetime, list] = {}
            for snap in snapshots:
                bucket_ts = snap.timestamp.replace(
                    minute=(snap.timestamp.minute // 5) * 5,
                    second=0,
                    microsecond=0,
                )
                buckets.setdefault(bucket_ts, []).append(snap)

            for bucket_ts, group in buckets.items():
                buys = [s.instant_buy for s in group if s.instant_buy]
                sells = [s.instant_sell for s in group if s.instant_sell]
                buy_vols = [s.buy_volume or 0 for s in group]
                sell_vols = [s.sell_volume or 0 for s in group]

                agg = PriceAggregate(
                    item_id=item_id,
                    timestamp=bucket_ts,
                    interval="5m",
                    open_buy=buys[0] if buys else None,
                    close_buy=buys[-1] if buys else None,
                    high_buy=max(buys) if buys else None,
                    low_buy=min(buys) if buys else None,
                    open_sell=sells[0] if sells else None,
                    close_sell=sells[-1] if sells else None,
                    high_sell=max(sells) if sells else None,
                    low_sell=min(sells) if sells else None,
                    total_buy_volume=sum(buy_vols),
                    total_sell_volume=sum(sell_vols),
                    snapshot_count=len(group),
                )
                db.add(agg)
                total_created += 1

        db.commit()
        logger.info("Created %d 5m aggregate candles", total_created)
        return total_created

    async def _aggregate_5m_to_1h(self, db):
        """Roll up 5m candles older than 30 days into 1h candles."""
        cutoff = datetime.utcnow() - timedelta(days=30)

        rows = (
            db.query(PriceAggregate)
            .filter(
                PriceAggregate.interval == "5m",
                PriceAggregate.timestamp < cutoff,
            )
            .order_by(PriceAggregate.item_id, PriceAggregate.timestamp)
            .all()
        )

        # Group by (item_id, hour)
        buckets: Dict[tuple, list] = {}
        for row in rows:
            hour_ts = row.timestamp.replace(minute=0, second=0, microsecond=0)
            key = (row.item_id, hour_ts)
            buckets.setdefault(key, []).append(row)

        total_created = 0
        for (item_id, hour_ts), group in buckets.items():
            buys = [r.high_buy for r in group if r.high_buy is not None]
            sells = [r.high_sell for r in group if r.high_sell is not None]

            agg = PriceAggregate(
                item_id=item_id,
                timestamp=hour_ts,
                interval="1h",
                open_buy=group[0].open_buy,
                close_buy=group[-1].close_buy,
                high_buy=max(buys) if buys else None,
                low_buy=min([r.low_buy for r in group if r.low_buy is not None]) if any(r.low_buy for r in group) else None,
                open_sell=group[0].open_sell,
                close_sell=group[-1].close_sell,
                high_sell=max(sells) if sells else None,
                low_sell=min([r.low_sell for r in group if r.low_sell is not None]) if any(r.low_sell for r in group) else None,
                total_buy_volume=sum(r.total_buy_volume or 0 for r in group),
                total_sell_volume=sum(r.total_sell_volume or 0 for r in group),
                snapshot_count=sum(r.snapshot_count or 0 for r in group),
            )
            db.add(agg)
            total_created += 1

        # Delete old 5m rows
        if rows:
            for row in rows:
                db.delete(row)

        db.commit()
        logger.info("Created %d 1h aggregate candles, deleted %d old 5m candles", total_created, len(rows))

    async def _delete_old_snapshots(self, db):
        """Delete raw snapshots older than 7 days (already aggregated)."""
        cutoff = datetime.utcnow() - timedelta(days=7)
        deleted = (
            db.query(PriceSnapshot)
            .filter(PriceSnapshot.timestamp < cutoff)
            .delete()
        )
        db.commit()
        logger.info("Deleted %d old raw snapshots", deleted)

    async def prune(self):
        """Run the full pruning cycle."""
        db = SessionLocal()
        try:
            await self._aggregate_snapshots_to_5m(db)
            await self._aggregate_5m_to_1h(db)
            await self._delete_old_snapshots(db)
        except Exception as e:
            db.rollback()
            logger.error("DataPruner error: %s", e)
        finally:
            db.close()

    async def run_forever(self):
        logger.info("DataPruner started (runs once per day)")
        while True:
            try:
                await self.prune()
            except Exception as e:
                logger.error("DataPruner tick error: %s", e)
            # Sleep 24 hours
            await asyncio.sleep(86400)


# ---------------------------------------------------------------------------
# Task launcher (called from app.py on startup)
# ---------------------------------------------------------------------------

_tasks = []


async def start_background_tasks():
    """Create and start all background asyncio tasks."""
    global _tasks

    collector = PriceCollector()
    feature_computer = FeatureComputer()
    scorer = MLScorer()
    pruner = DataPruner()

    _tasks = [
        asyncio.create_task(collector.run_forever()),
        asyncio.create_task(feature_computer.run_forever()),
        asyncio.create_task(scorer.run_forever()),
        asyncio.create_task(pruner.run_forever()),
    ]

    logger.info("All background tasks started (%d tasks)", len(_tasks))


async def stop_background_tasks():
    """Cancel all running background tasks."""
    for task in _tasks:
        task.cancel()
    if _tasks:
        await asyncio.gather(*_tasks, return_exceptions=True)
    logger.info("All background tasks stopped")
