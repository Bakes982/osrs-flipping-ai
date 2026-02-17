"""
Background Tasks for OSRS Flipping AI
- PriceCollector: fetches prices every 10s, 5m data every 60s
- FeatureComputer: recomputes ML features every 60s
- MLScorer: runs predictions every 60s
- ModelRetrainer: retrains stale ML models every 6 hours
- AlertMonitor: checks price alerts and generates notifications
- DataPruner: aggregates old data once per day
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import httpx
from sqlalchemy import func

from backend.database import (
    SessionLocal,
    PriceSnapshot,
    PriceAggregate,
    Item,
    Alert,
    get_tracked_item_ids,
    get_db,
    get_setting,
    get_latest_price,
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
# FeatureComputer
# ---------------------------------------------------------------------------

class FeatureComputer:
    """Recomputes ML feature vectors for tracked items every 60 seconds.

    Uses the ML pipeline's FeatureEngine to compute and cache features
    in the ItemFeature table for fast inference.
    """

    def __init__(self):
        self._engine = None

    def _get_engine(self):
        if self._engine is None:
            try:
                from backend.ml.feature_engine import FeatureEngine
                self._engine = FeatureEngine()
            except Exception as e:
                logger.error("Failed to initialize FeatureEngine: %s", e)
        return self._engine

    async def compute_features(self):
        """Compute and cache features for all tracked items."""
        engine = self._get_engine()
        if engine is None:
            return

        db = SessionLocal()
        try:
            from backend.database import ItemFeature, get_price_history, get_item_flips
            item_ids = get_tracked_item_ids(db)
            computed = 0

            for item_id in item_ids:
                try:
                    snapshots = get_price_history(db, item_id, hours=4)
                    flips = get_item_flips(db, item_id, days=30)
                    features = engine.compute_features(item_id, snapshots, flips)

                    if features:
                        import json
                        now = datetime.utcnow()
                        # Upsert feature cache
                        existing = db.query(ItemFeature).filter(
                            ItemFeature.item_id == item_id
                        ).first()
                        if existing:
                            existing.features = json.dumps(features)
                            existing.timestamp = now
                        else:
                            db.add(ItemFeature(
                                item_id=item_id,
                                timestamp=now,
                                features=json.dumps(features),
                            ))
                        computed += 1
                except Exception as e:
                    logger.debug("Feature compute failed for %d: %s", item_id, e)

            db.commit()
            if computed > 0:
                logger.info("FeatureComputer: computed features for %d/%d items", computed, len(item_ids))
        finally:
            db.close()

    async def run_forever(self):
        logger.info("FeatureComputer started")
        while True:
            try:
                await self.compute_features()
            except Exception as e:
                logger.error("FeatureComputer tick error: %s", e)
            await asyncio.sleep(60)


# ---------------------------------------------------------------------------
# MLScorer
# ---------------------------------------------------------------------------

class MLScorer:
    """Runs ML model predictions for all tracked items every 60 seconds.

    Uses the Predictor from the ML pipeline. Falls back to statistical
    methods if no trained models exist yet.
    """

    def __init__(self):
        self._predictor = None

    def _get_predictor(self):
        if self._predictor is None:
            try:
                from backend.ml.predictor import Predictor
                self._predictor = Predictor()
                self._predictor.load_models()
            except Exception as e:
                logger.error("Failed to initialize Predictor: %s", e)
        return self._predictor

    async def score_items(self):
        """Run predictions for tracked items and store in DB."""
        predictor = self._get_predictor()
        if predictor is None:
            return

        db = SessionLocal()
        try:
            item_ids = get_tracked_item_ids(db)
            if not item_ids:
                return

            # Batch predict (saves to DB internally)
            results = predictor.predict_batch(item_ids[:100], save_to_db=True)
            logger.info("MLScorer: scored %d items (%s method)",
                len(results),
                "ml" if predictor._models_loaded else "statistical"
            )

            # Also record outcomes for past predictions (accuracy tracking)
            for item_id in item_ids[:50]:
                try:
                    predictor.record_outcomes(item_id)
                except Exception:
                    pass
        finally:
            db.close()

    async def run_forever(self):
        logger.info("MLScorer started")
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
# ModelRetrainer
# ---------------------------------------------------------------------------

class ModelRetrainer:
    """Checks if ML models are stale and retrains them automatically.

    Runs every 6 hours. Uses the ModelTrainer to rebuild models when
    the latest training metrics are older than max_age_hours.
    """

    RETRAIN_INTERVAL = 6 * 3600  # 6 hours between checks
    MAX_MODEL_AGE_HOURS = 24     # Retrain if model older than this

    def __init__(self):
        self._trainer = None

    def _get_trainer(self):
        if self._trainer is None:
            try:
                from backend.ml.model_trainer import ModelTrainer
                self._trainer = ModelTrainer()
            except Exception as e:
                logger.error("Failed to initialize ModelTrainer: %s", e)
        return self._trainer

    async def retrain_if_stale(self):
        """Check each horizon and retrain if stale."""
        trainer = self._get_trainer()
        if trainer is None:
            return

        try:
            from backend.ml.feature_engine import HORIZONS
        except ImportError:
            logger.error("Cannot import HORIZONS from feature_engine")
            return

        retrained = []
        for horizon in HORIZONS:
            try:
                if trainer.should_retrain(horizon, max_age_hours=self.MAX_MODEL_AGE_HOURS):
                    logger.info("ModelRetrainer: retraining horizon %s", horizon)
                    # Run training in a thread to avoid blocking the event loop
                    loop = asyncio.get_event_loop()
                    metrics = await loop.run_in_executor(
                        None, trainer.train_horizon, horizon, 168,
                    )
                    if metrics:
                        retrained.append(horizon)
                        logger.info(
                            "ModelRetrainer: %s retrained - accuracy=%.1f%%",
                            horizon,
                            metrics.get("val_direction_accuracy", 0) * 100,
                        )
            except Exception as e:
                logger.error("ModelRetrainer: failed to retrain %s: %s", horizon, e)

        if retrained:
            logger.info("ModelRetrainer: retrained %d horizons: %s", len(retrained), retrained)

    async def run_forever(self):
        logger.info("ModelRetrainer started (runs every %d hours)", self.RETRAIN_INTERVAL // 3600)
        # Wait 5 minutes before first run to let price data accumulate
        await asyncio.sleep(300)
        while True:
            try:
                await self.retrain_if_stale()
            except Exception as e:
                logger.error("ModelRetrainer tick error: %s", e)
            await asyncio.sleep(self.RETRAIN_INTERVAL)


# ---------------------------------------------------------------------------
# AlertMonitor
# ---------------------------------------------------------------------------

class AlertMonitor:
    """Monitors prices and generates alerts for significant events.

    Checks every 30 seconds for:
    - Price targets hit (user-configured)
    - Dump detection (rapid price drops)
    - High-score opportunity alerts
    - ML signal alerts (strong directional predictions)
    """

    CHECK_INTERVAL = 30  # seconds

    async def check_alerts(self):
        """Run all alert checks."""
        db = SessionLocal()
        try:
            await self._check_price_targets(db)
            await self._check_dump_alerts(db)
            await self._check_opportunity_alerts(db)
        except Exception as e:
            logger.error("AlertMonitor check error: %s", e)
        finally:
            db.close()

    async def _check_price_targets(self, db):
        """Check if any user-configured price targets have been hit."""
        targets = get_setting(db, "price_alerts", default=[])
        if not targets:
            return

        new_targets = []
        for target in targets:
            item_id = target.get("item_id")
            target_price = target.get("target_price")
            direction = target.get("direction", "below")  # "below" or "above"
            item_name = target.get("item_name", f"Item {item_id}")

            if not item_id or not target_price:
                new_targets.append(target)
                continue

            snap = get_latest_price(db, item_id)
            if not snap:
                new_targets.append(target)
                continue

            current = snap.instant_buy or snap.instant_sell
            if not current:
                new_targets.append(target)
                continue

            triggered = False
            if direction == "below" and current <= target_price:
                triggered = True
            elif direction == "above" and current >= target_price:
                triggered = True

            if triggered:
                alert = Alert(
                    item_id=item_id,
                    item_name=item_name,
                    alert_type="price_target",
                    message=f"{item_name} hit {direction} target: {current:,} GP (target: {target_price:,} GP)",
                    data={"current_price": current, "target_price": target_price, "direction": direction},
                )
                db.add(alert)
                logger.info("Price alert triggered: %s", alert.message)
                # Broadcast alert via WebSocket
                await manager.broadcast_json({
                    "type": "alert",
                    "alert_type": "price_target",
                    "item_id": item_id,
                    "item_name": item_name,
                    "message": alert.message,
                })
            else:
                new_targets.append(target)

        # Update remaining targets (remove triggered ones)
        if len(new_targets) != len(targets):
            from backend.database import set_setting
            set_setting(db, "price_alerts", new_targets)

    async def _check_dump_alerts(self, db):
        """Detect items experiencing rapid price drops (>5% in 15 min)."""
        item_ids = get_tracked_item_ids(db)
        now = datetime.utcnow()
        cutoff_15m = now - timedelta(minutes=15)

        # Only check top-volume items to limit DB queries
        for item_id in item_ids[:200]:
            try:
                snaps = (
                    db.query(PriceSnapshot)
                    .filter(
                        PriceSnapshot.item_id == item_id,
                        PriceSnapshot.timestamp >= cutoff_15m,
                    )
                    .order_by(PriceSnapshot.timestamp.asc())
                    .all()
                )
                if len(snaps) < 5:
                    continue

                prices = [s.instant_buy for s in snaps if s.instant_buy and s.instant_buy > 0]
                if len(prices) < 5:
                    continue

                first_price = prices[0]
                last_price = prices[-1]
                change_pct = (last_price - first_price) / first_price * 100

                if change_pct < -5:
                    # Check we haven't alerted for this item in last 30 min
                    recent_alert = (
                        db.query(Alert)
                        .filter(
                            Alert.item_id == item_id,
                            Alert.alert_type == "dump",
                            Alert.timestamp >= now - timedelta(minutes=30),
                        )
                        .first()
                    )
                    if recent_alert:
                        continue

                    item_row = db.query(Item).filter(Item.id == item_id).first()
                    item_name = item_row.name if item_row else f"Item {item_id}"

                    alert = Alert(
                        item_id=item_id,
                        item_name=item_name,
                        alert_type="dump",
                        message=f"{item_name} dropping {change_pct:.1f}% in 15min ({first_price:,} -> {last_price:,})",
                        data={"change_pct": round(change_pct, 1), "from_price": first_price, "to_price": last_price},
                    )
                    db.add(alert)
                    await manager.broadcast_json({
                        "type": "alert",
                        "alert_type": "dump",
                        "item_id": item_id,
                        "item_name": item_name,
                        "message": alert.message,
                    })
            except Exception:
                pass

        db.commit()

    async def _check_opportunity_alerts(self, db):
        """Alert when a very high-score opportunity appears (score 75+)."""
        # This runs less frequently - check last scored items
        try:
            from backend.flip_scorer import FlipScorer
            scorer = FlipScorer()
        except ImportError:
            return

        item_ids = get_tracked_item_ids(db)
        now = datetime.utcnow()

        for item_id in item_ids[:50]:
            try:
                from backend.database import get_price_history, get_item_flips
                snapshots = get_price_history(db, item_id, hours=1)
                if len(snapshots) < 10:
                    continue

                flips = get_item_flips(db, item_id, days=30)
                fs = scorer.score_item(item_id, snapshots=snapshots, flips=flips)

                if fs.vetoed or fs.total_score < 75:
                    continue

                # Check we haven't alerted recently
                recent_alert = (
                    db.query(Alert)
                    .filter(
                        Alert.item_id == item_id,
                        Alert.alert_type == "opportunity",
                        Alert.timestamp >= now - timedelta(minutes=15),
                    )
                    .first()
                )
                if recent_alert:
                    continue

                alert = Alert(
                    item_id=item_id,
                    item_name=fs.item_name or f"Item {item_id}",
                    alert_type="opportunity",
                    message=f"High-score opportunity: {fs.item_name} (score {fs.total_score:.0f}, +{fs.expected_profit:,} GP)" if fs.expected_profit else f"High-score opportunity: {fs.item_name} (score {fs.total_score:.0f})",
                    data={
                        "score": round(fs.total_score, 1),
                        "profit": fs.expected_profit,
                        "margin_pct": round(fs.spread_pct, 1) if fs.spread_pct else 0,
                        "volume": fs.volume_5m,
                    },
                )
                db.add(alert)
                await manager.broadcast_json({
                    "type": "alert",
                    "alert_type": "opportunity",
                    "item_id": item_id,
                    "item_name": fs.item_name,
                    "message": alert.message,
                    "score": round(fs.total_score, 1),
                })
            except Exception:
                pass

        db.commit()

    async def run_forever(self):
        logger.info("AlertMonitor started (checks every %ds)", self.CHECK_INTERVAL)
        # Wait 30 seconds for initial data
        await asyncio.sleep(30)
        while True:
            try:
                await self.check_alerts()
            except Exception as e:
                logger.error("AlertMonitor tick error: %s", e)
            await asyncio.sleep(self.CHECK_INTERVAL)


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
    retrainer = ModelRetrainer()
    alert_monitor = AlertMonitor()
    pruner = DataPruner()

    _tasks = [
        asyncio.create_task(collector.run_forever()),
        asyncio.create_task(feature_computer.run_forever()),
        asyncio.create_task(scorer.run_forever()),
        asyncio.create_task(retrainer.run_forever()),
        asyncio.create_task(alert_monitor.run_forever()),
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
