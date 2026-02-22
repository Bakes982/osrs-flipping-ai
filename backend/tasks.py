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

from backend.database import (
    get_db,
    PriceSnapshot,
    PriceAggregate,
    Item,
    Alert,
    get_tracked_item_ids,
    get_setting,
    set_setting,
    get_latest_price,
    insert_price_snapshots,
    insert_alert,
    upsert_item_feature,
    get_item,
    find_active_positions,
    dismiss_position,
    find_pending_sells,
    get_price_history,
)
from backend.websocket import manager

logger = logging.getLogger(__name__)

WIKI_BASE = "https://prices.runescape.wiki/api/v1/osrs"
USER_AGENT = "OSRS-AI-Flipper v2.0 - Discord: bakes982"
HEADERS = {"User-Agent": USER_AGENT}

# In-memory price cache: item_id_str → {high, low, highTime, lowTime}
# Populated by PriceCollector every 10s; used as fallback when DB has no snapshots.
_price_cache: Dict = {}

# In-memory 5-minute cache: item_id_str → {avgHighPrice, avgLowPrice, highPriceVolume, lowPriceVolume}
# Populated by PriceCollector every 60s alongside the /5m fetch.
_5m_cache: Dict = {}


# ---------------------------------------------------------------------------
# PriceCollector
# ---------------------------------------------------------------------------

class PriceCollector:
    """Fetches /latest every 10 seconds and /5m every 60 seconds.

    Stores PriceSnapshot rows and broadcasts new data to WebSocket clients.
    """

    # Write to MongoDB every 5 minutes (not every 10 seconds).
    # Broadcast + _price_cache still refresh every 10s.
    STORE_INTERVAL = 300  # seconds

    def __init__(self):
        self._latest_data: Dict = {}
        self._5m_data: Dict = {}
        self._last_5m_fetch: float = 0.0
        self._last_store: float = 0.0
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
            # Update global in-memory cache for all items (used by PositionMonitor)
            _price_cache.update(self._latest_data)
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
            # Expose globally so AlertMonitor can access volume data for dump detection
            _5m_cache.update(self._5m_data)
            return self._5m_data
        except Exception as e:
            logger.error("Failed to fetch /5m: %s", e)
            return self._5m_data

    async def store_snapshots(self) -> int:
        """Store current price data as PriceSnapshot rows.

        Stores top 50 items by volume, every 5 minutes (not every 10 seconds).
        Broadcast + in-memory cache still update every 10s — this only controls
        MongoDB writes to stay well under the 512 MB Atlas quota.
        Returns the number of rows inserted (0 if skipped due to throttle).
        """
        if not self._latest_data:
            return 0

        # Throttle: only write to MongoDB every STORE_INTERVAL seconds
        if time.time() - self._last_store < self.STORE_INTERVAL:
            return 0

        def _sync_store():
            now = datetime.utcnow()
            db = get_db()
            count = 0
            try:
                # Filter to items with meaningful volume, keep top 50
                items_with_volume = []
                for item_id_str, instant in self._latest_data.items():
                    avg = self._5m_data.get(item_id_str, {})
                    vol = (avg.get("highPriceVolume", 0) or 0) + (avg.get("lowPriceVolume", 0) or 0)
                    if vol > 0 and instant.get("high") and instant.get("low"):
                        items_with_volume.append((item_id_str, instant, avg, vol))

                items_with_volume.sort(key=lambda x: x[3], reverse=True)
                items_with_volume = items_with_volume[:100]  # was 200; top-100 covers all meaningful flip targets

                snapshots_list = []
                for item_id_str, instant, avg, vol in items_with_volume:
                    item_id = int(item_id_str)
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
                    snapshots_list.append(snap)

                count = insert_price_snapshots(db, snapshots_list)
            except Exception as e:
                logger.error("Error storing snapshots: %s", e)
            finally:
                db.close()
            return count

        count = await asyncio.to_thread(_sync_store)
        self._last_store = time.time()
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

        def _sync_compute():
            db = get_db()
            try:
                from backend.database import get_price_history, get_item_flips
                item_ids = get_tracked_item_ids(db)
                computed = 0

                for item_id in item_ids[:200]:  # Cap to avoid excessive processing
                    try:
                        snapshots = get_price_history(db, item_id, hours=4)
                        flips = get_item_flips(db, item_id, days=30)
                        features = engine.compute_features(item_id, snapshots, flips)

                        if features:
                            import json
                            now = datetime.utcnow()
                            upsert_item_feature(db, item_id, json.dumps(features), now)
                            computed += 1
                    except Exception as e:
                        logger.debug("Feature compute failed for %d: %s", item_id, e)

                if computed > 0:
                    logger.info("FeatureComputer: computed features for %d/%d items", computed, len(item_ids))
            finally:
                db.close()

        await asyncio.to_thread(_sync_compute)

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

        def _sync_score():
            db = get_db()
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

        await asyncio.to_thread(_sync_score)

    async def run_forever(self):
        logger.info("MLScorer started")
        while True:
            try:
                await self.score_items()
            except Exception as e:
                logger.error("MLScorer tick error: %s", e)
            await asyncio.sleep(600)  # Every 10 min: 100 items x 6 horizons = 600 inserts/run; was 60s (8,640/day) -> now 144/day predictions


# ---------------------------------------------------------------------------
# DataPruner
# ---------------------------------------------------------------------------

class DataPruner:
    """Aggregates and prunes old price data once per day.

    - Snapshots older than 3 days -> 5m candles in PriceAggregate
    - PriceAggregate 5m rows older than 14 days -> 1h candles
    - Deletes raw snapshots older than 3 days after aggregation
    - Deletes old predictions (> 3 days) and alerts (> 7 days)
    """

    def _aggregate_snapshots_to_5m(self, db):
        """Aggregate raw snapshots older than 6 hours into 5-minute candles.

        Cutoff is 6h (not 3 days) so snapshots are aggregated before the
        24h deletion window in _delete_old_snapshots. Previously the 3-day
        cutoff meant snapshots were deleted at 24h with no candles ever created.
        """
        cutoff = datetime.utcnow() - timedelta(hours=6)

        # Get distinct item_ids with old data
        item_ids = db.price_snapshots.distinct("item_id", {"timestamp": {"$lt": cutoff}})

        total_created = 0
        for item_id in item_ids:
            # Process in batches of 5000 to limit memory
            docs = list(db.price_snapshots.find(
                {"item_id": item_id, "timestamp": {"$lt": cutoff}}
            ).sort("timestamp", 1).limit(5000))
            snapshots = [PriceSnapshot.from_doc(d) for d in docs]

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
                db.price_aggregates.insert_one(agg.to_doc())
                total_created += 1

        logger.info("Created %d 5m aggregate candles", total_created)
        return total_created

    def _aggregate_5m_to_1h(self, db):
        """Roll up 5m candles older than 14 days into 1h candles."""
        cutoff = datetime.utcnow() - timedelta(days=14)

        # Get distinct item_ids with old 5m data
        item_ids = db.price_aggregates.distinct(
            "item_id", {"interval": "5m", "timestamp": {"$lt": cutoff}}
        )

        total_created = 0
        total_deleted = 0
        for item_id in item_ids:
            docs = list(db.price_aggregates.find(
                {"item_id": item_id, "interval": "5m", "timestamp": {"$lt": cutoff}}
            ).sort("timestamp", 1).limit(5000))  # process in batches

            # Group by hour
            buckets: Dict[datetime, list] = {}
            for row in docs:
                hour_ts = row["timestamp"].replace(minute=0, second=0, microsecond=0)
                buckets.setdefault(hour_ts, []).append(row)

            for hour_ts, group in buckets.items():
                buys = [r["high_buy"] for r in group if r.get("high_buy") is not None]
                sells = [r["high_sell"] for r in group if r.get("high_sell") is not None]

                agg = PriceAggregate(
                    item_id=item_id,
                    timestamp=hour_ts,
                    interval="1h",
                    open_buy=group[0].get("open_buy"),
                    close_buy=group[-1].get("close_buy"),
                    high_buy=max(buys) if buys else None,
                    low_buy=min([r["low_buy"] for r in group if r.get("low_buy") is not None]) if any(r.get("low_buy") for r in group) else None,
                    open_sell=group[0].get("open_sell"),
                    close_sell=group[-1].get("close_sell"),
                    high_sell=max(sells) if sells else None,
                    low_sell=min([r["low_sell"] for r in group if r.get("low_sell") is not None]) if any(r.get("low_sell") for r in group) else None,
                    total_buy_volume=sum(r.get("total_buy_volume", 0) or 0 for r in group),
                    total_sell_volume=sum(r.get("total_sell_volume", 0) or 0 for r in group),
                    snapshot_count=sum(r.get("snapshot_count", 0) or 0 for r in group),
                )
                db.price_aggregates.insert_one(agg.to_doc())
                total_created += 1

            # Delete old 5m rows for this item
            if docs:
                old_ids = [r["_id"] for r in docs]
                db.price_aggregates.delete_many({"_id": {"$in": old_ids}})
                total_deleted += len(old_ids)

        logger.info("Created %d 1h aggregate candles, deleted %d old 5m candles", total_created, total_deleted)

    def _delete_old_snapshots(self, db):
        """Delete raw snapshots older than 24 hours (already aggregated)."""
        cutoff = datetime.utcnow() - timedelta(hours=24)
        result = db.price_snapshots.delete_many({"timestamp": {"$lt": cutoff}})
        logger.info("Deleted %d old raw snapshots", result.deleted_count)

        # Also prune predictions and model metrics to stay under storage quota
        cutoff_pred = datetime.utcnow() - timedelta(days=1)  # was 3 days; 1 day is enough for outcome recording
        try:
            r2 = db.predictions.delete_many({"timestamp": {"$lt": cutoff_pred}})
            logger.info("Deleted %d old predictions", r2.deleted_count)
        except Exception:
            pass

        cutoff_alerts = datetime.utcnow() - timedelta(days=7)
        try:
            r3 = db.alerts.delete_many({"timestamp": {"$lt": cutoff_alerts}})
            logger.info("Deleted %d old alerts", r3.deleted_count)
        except Exception:
            pass

        cutoff_metrics = datetime.utcnow() - timedelta(days=7)
        try:
            r4 = db.model_metrics.delete_many({"timestamp": {"$lt": cutoff_metrics}})
            logger.info("Deleted %d old model_metrics", r4.deleted_count)
        except Exception:
            pass

    async def prune(self):
        """Run the full pruning cycle in a background thread."""
        def _sync_prune():
            db = get_db()
            try:
                self._aggregate_snapshots_to_5m(db)
                self._aggregate_5m_to_1h(db)
                self._delete_old_snapshots(db)
            except Exception as e:
                logger.error("DataPruner error: %s", e)
            finally:
                db.close()

        await asyncio.to_thread(_sync_prune)

    async def run_forever(self):
        logger.info("DataPruner started (runs every 2 hours)")
        while True:
            try:
                await self.prune()
            except Exception as e:
                logger.error("DataPruner tick error: %s", e)
            await asyncio.sleep(2 * 3600)  # every 2 hours (was 6h; shorter window limits peak accumulation)


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
        db = get_db()
        try:
            await self._check_price_targets(db)
            await asyncio.to_thread(self._check_dump_alerts_sync, db)
            await asyncio.to_thread(self._check_opportunity_alerts_sync, db)
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
                insert_alert(db, alert)
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

    def _check_dump_alerts_sync(self, db):
        """Detect bot dump events across all ~6000 tradeable items.

        A dump is identified by two simultaneous signals in the /5m data:
          1. Sell volume >> buy volume  — someone is mass-selling (bots dumping)
          2. Current price < 5m average — the selling has already pushed the price down

        Both signals together give high confidence that the drop is temporary and
        will recover once the dump is absorbed, creating a buy opportunity.

        Does NOT rely on MongoDB snapshots — uses the in-memory _price_cache (latest,
        updated every 10s) and _5m_cache (5m averages + volumes, updated every 60s).
        This means we can check every item on the GE, not just tracked ones.
        """
        if not _price_cache or not _5m_cache:
            return  # Caches not populated yet

        now = datetime.utcnow()

        # Thresholds (tuned for OSRS GE dump catching)
        MIN_ITEM_VALUE       = 50_000    # GP — ignore cheap junk (too noisy)
        MIN_SELL_VOLUME      = 50        # items — minimum meaningful dump size
        SELL_RATIO_THRESHOLD = 0.65      # 65%+ of 5m volume must be selling
        PRICE_DROP_THRESHOLD = 3.0       # % below 5m average to qualify
        COOLDOWN_MINUTES     = 30        # suppress repeat alerts for same item

        for item_id_str, instant in _price_cache.items():
            try:
                five_m = _5m_cache.get(item_id_str, {})

                # Current instant prices
                instant_sell = instant.get("low")   # what you get selling right now
                instant_buy  = instant.get("high")  # what you pay buying right now

                # 5-minute averaged prices and volumes
                avg_sell  = five_m.get("avgLowPrice")
                avg_buy   = five_m.get("avgHighPrice")
                sell_vol  = five_m.get("lowPriceVolume", 0) or 0   # items sold last 5m
                buy_vol   = five_m.get("highPriceVolume", 0) or 0  # items bought last 5m

                # Must have valid data
                if not instant_sell or not avg_sell or avg_sell <= 0:
                    continue

                # Skip cheap items (junk / noise)
                if avg_sell < MIN_ITEM_VALUE:
                    continue

                # Must have enough volume to be meaningful
                total_vol = sell_vol + buy_vol
                if total_vol < MIN_SELL_VOLUME or sell_vol < MIN_SELL_VOLUME:
                    continue

                # === DUMP SIGNAL 1: volume imbalance ===
                # Most of the 5m volume is sell-side (bots dumping)
                sell_ratio = sell_vol / total_vol
                if sell_ratio < SELL_RATIO_THRESHOLD:
                    continue

                # === DUMP SIGNAL 2: price below 5m average ===
                # Selling pressure has already pushed price below the recent average
                price_drop_pct = (avg_sell - instant_sell) / avg_sell * 100
                if price_drop_pct < PRICE_DROP_THRESHOLD:
                    continue

                # Both signals confirmed — this looks like a bot dump
                item_id = int(item_id_str)

                # Cooldown: don't spam alerts for the same item
                recent_alert = db.alerts.find_one({
                    "item_id": item_id,
                    "alert_type": "dump",
                    "timestamp": {"$gte": now - timedelta(minutes=COOLDOWN_MINUTES)},
                })
                if recent_alert:
                    continue

                # Item name (may not be in DB if it's not a top-100 item)
                item_row = get_item(db, item_id)
                item_name = item_row.name if item_row else f"Item {item_id}"

                # Recovery metrics
                # If price snaps back to 5m average after you buy at instant_sell:
                #   profit = avg_sell - instant_sell - GE_tax
                ge_tax = int(min(avg_sell * 0.01, 5_000_000))  # 1% buyer's side
                profit_per_item = int(avg_sell - instant_sell - ge_tax)

                # Total GP being dumped in this 5m window
                dump_gp_total = int(sell_vol * instant_sell)

                # Severity score 1-10: combines drop depth and volume imbalance
                severity = round(min(10.0, price_drop_pct * sell_ratio * 3), 1)

                alert = Alert(
                    item_id=item_id,
                    item_name=item_name,
                    alert_type="dump",
                    message=(
                        f"{item_name}: {price_drop_pct:.1f}% below 5m avg "
                        f"({instant_sell:,} vs avg {avg_sell:,} GP) — "
                        f"{sell_vol:,} sold vs {buy_vol:,} bought ({sell_ratio:.0%} sell ratio) — "
                        f"~{profit_per_item:+,} GP/item recovery potential"
                    ),
                    data={
                        "dump_type":         "volume_dump",
                        "instant_sell":      instant_sell,
                        "instant_buy":       instant_buy,
                        "avg_sell":          avg_sell,
                        "avg_buy":           avg_buy,
                        "price_drop_pct":    round(price_drop_pct, 1),
                        "sell_volume":       sell_vol,
                        "buy_volume":        buy_vol,
                        "sell_ratio":        round(sell_ratio, 2),
                        "recovery_gp":       int(avg_sell - instant_sell),
                        "profit_per_item":   profit_per_item,
                        "dump_gp_total":     dump_gp_total,
                        "severity":          severity,
                    },
                )
                insert_alert(db, alert)
                logger.info(
                    "Dump alert [severity %.1f]: %s — %.1f%% drop, %d sold vs %d bought",
                    severity, item_name, price_drop_pct, sell_vol, buy_vol,
                )
            except Exception as e:
                logger.debug("Dump check error for item %s: %s", item_id_str, e)

    def _check_opportunity_alerts_sync(self, db):
        """Alert when a very high-score opportunity appears (score 75+). Sync version."""
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
                recent_alert = db.alerts.find_one({
                    "item_id": item_id,
                    "alert_type": "opportunity",
                    "timestamp": {"$gte": now - timedelta(minutes=15)},
                })
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
                insert_alert(db, alert)
                logger.info("Opportunity alert: %s", alert.message)
            except Exception:
                pass

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
# PositionMonitor
# ---------------------------------------------------------------------------

class PositionMonitor:
    """Monitors active positions (unmatched BUY trades) for price changes.

    Every 60 seconds:
    1. Finds all open positions (BOUGHT but not yet matched to a flip)
    2. Looks up the current market price for each held item
    3. Re-calculates the recommended sell price via SmartPricer
    4. If the price has moved significantly since the last alert, fires:
       - A WebSocket broadcast so the dashboard updates live
       - A Discord notification (if enabled) for large changes
       - An in-DB Alert record
    """

    CHECK_INTERVAL = 30  # seconds between checks
    SMALL_CHANGE_PCT = 2.0   # % change to flag on dashboard
    LARGE_CHANGE_PCT = 5.0   # % change to send Discord notification
    COOLDOWN_MINUTES = 15    # minimum minutes between alerts for the same item
    SELL_DROP_THRESHOLD = 2.0  # % below listed sell price to trigger sell alert

    def __init__(self):
        # item_id → {last_alerted_price, last_alerted_at, last_rec_sell}
        self._state: Dict[int, dict] = {}
        self._pricer = None

    def _get_pricer(self):
        if self._pricer is None:
            try:
                from backend.smart_pricer import SmartPricer
                self._pricer = SmartPricer()
            except Exception as e:
                logger.error("PositionMonitor: failed to init SmartPricer: %s", e)
        return self._pricer

    async def check_positions(self):
        """Main tick: scan open positions and check for price movement."""
        positions = await asyncio.to_thread(self._get_positions)
        if not positions:
            return

        pricer = self._get_pricer()
        if pricer is None:
            return

        now = datetime.utcnow()
        updates = []

        for pos in positions:
            item_id = pos["item_id"]
            buy_price = pos["buy_price"]
            quantity = pos["quantity"]

            try:
                result = await asyncio.to_thread(
                    self._evaluate_position, pricer, pos
                )
                if result is None:
                    continue

                current_price = result["current_price"]
                rec_sell = result["recommended_sell"]
                pnl_pct = result["pnl_pct"]
                rec_profit = result["recommended_profit"]
                rec_profit_pct = result["recommended_profit_pct"]

                # Build the position update payload
                update = {
                    "item_id": item_id,
                    "item_name": pos["item_name"],
                    "quantity": quantity,
                    "buy_price": buy_price,
                    "current_price": current_price,
                    "recommended_sell": rec_sell,
                    "pnl_pct": round(pnl_pct, 2),
                    "recommended_profit": rec_profit,
                    "recommended_profit_pct": round(rec_profit_pct, 2),
                    "bought_at": pos.get("bought_at"),
                    "trade_id": pos.get("trade_id"),
                }

                updates.append(update)

                # Check if we should fire an alert
                state = self._state.get(item_id, {})
                last_alerted = state.get("last_alerted_price", buy_price)
                last_alerted_at = state.get("last_alerted_at")
                change_from_alert = ((current_price - last_alerted) / last_alerted * 100) if last_alerted else 0

                # Cooldown check
                on_cooldown = (
                    last_alerted_at
                    and (now - last_alerted_at).total_seconds() < self.COOLDOWN_MINUTES * 60
                )

                if abs(change_from_alert) >= self.SMALL_CHANGE_PCT and not on_cooldown:
                    direction = "dropped" if change_from_alert < 0 else "risen"
                    alert_msg = (
                        f"{pos['item_name']} has {direction} {abs(change_from_alert):.1f}% "
                        f"since your buy ({buy_price:,} → {current_price:,} GP). "
                        f"Recommended sell: {rec_sell:,} GP "
                        f"(est. profit: {'+' if rec_profit >= 0 else ''}{rec_profit:,} GP)"
                    )

                    # Save alert to DB
                    await asyncio.to_thread(
                        self._save_alert, item_id, pos["item_name"],
                        alert_msg, update
                    )

                    # Update state
                    self._state[item_id] = {
                        "last_alerted_price": current_price,
                        "last_alerted_at": now,
                        "last_rec_sell": rec_sell,
                    }

                    # Discord for large moves:
                    # - PRICE DROP: always notify (losing more is always actionable)
                    # - PRICE RISE: only notify if the position has crossed into profit.
                    #   A bounce from -74% to -73% is noise, not a signal.
                    is_now_profitable = update.get("pnl_pct", -1) >= 0
                    should_discord = abs(change_from_alert) >= self.LARGE_CHANGE_PCT and (
                        change_from_alert < 0 or is_now_profitable
                    )
                    if should_discord:
                        asyncio.create_task(
                            self._send_discord_position_alert(pos, update, change_from_alert)
                        )

            except Exception as e:
                logger.debug("PositionMonitor: error checking %s: %s", pos["item_name"], e)

        # Broadcast all position updates to connected frontends
        if updates:
            await manager.broadcast_json({
                "type": "position_update",
                "timestamp": now.isoformat(),
                "positions": updates,
            })

    def _get_positions(self, source: Optional[str] = None, player: Optional[str] = None) -> List[Dict]:
        """Sync: find active positions from DB."""
        db = get_db()
        try:
            return find_active_positions(db, source=source, player=player)
        finally:
            db.close()

    def _evaluate_position(self, pricer, pos: dict) -> Optional[dict]:
        """Sync: compute current price + recommended sell for one position."""
        item_id = pos["item_id"]
        buy_price = pos["buy_price"]
        quantity = pos["quantity"]

        db = get_db()
        try:
            snapshots = get_price_history(db, item_id, hours=4)

            if snapshots:
                latest = snapshots[-1]
                current_price = latest.instant_buy or latest.instant_sell
                if not current_price:
                    return None
                rec = pricer.price_item(item_id, snapshots=snapshots)
                rec_sell = rec.recommended_sell or current_price
            else:
                # No DB snapshots (item not in top-200 volume) — use in-memory cache
                cached = _price_cache.get(str(item_id))
                if not cached:
                    return None
                current_price = cached.get("high") or cached.get("low")
                if not current_price:
                    return None
                # Without history, SmartPricer can't run — use current market price as target
                rec_sell = current_price

            # P&L from current instant price vs buy price
            pnl_pct = (current_price - buy_price) / buy_price * 100 if buy_price else 0

            # Estimated profit if sold at recommended sell
            tax = int(min(rec_sell * 0.02, 5_000_000))
            rec_profit = (rec_sell - buy_price) * quantity - tax * quantity

            rec_profit_pct = (
                (rec_sell - buy_price - tax) / buy_price * 100
            ) if buy_price else 0

            return {
                "current_price": current_price,
                "recommended_sell": rec_sell,
                "pnl_pct": pnl_pct,
                "recommended_profit": rec_profit,
                "recommended_profit_pct": rec_profit_pct,
            }
        finally:
            db.close()

    def _save_alert(self, item_id: int, item_name: str, message: str, data: dict):
        """Sync: insert a position_change alert into the DB."""
        db = get_db()
        try:
            alert = Alert(
                item_id=item_id,
                item_name=item_name,
                alert_type="position_change",
                message=message,
                data=data,
            )
            insert_alert(db, alert)
            logger.info("Position alert: %s", message)
        finally:
            db.close()

    async def _send_discord_position_alert(self, pos: dict, update: dict, change_pct: float):
        """Send a Discord embed for a large price move on an active position."""
        try:
            def _sync():
                db = get_db()
                try:
                    wh = get_setting(db, "discord_webhook")
                    if isinstance(wh, dict):
                        if not wh.get("enabled", False):
                            return None
                        return wh.get("url")
                    url = get_setting(db, "discord_webhook_url")
                    enabled = get_setting(db, "discord_alerts_enabled", False)
                    return url if url and enabled else None
                finally:
                    db.close()

            webhook_url = await asyncio.to_thread(_sync)
            if not webhook_url:
                return

            import requests as _requests

            pnl_pct = update.get("pnl_pct", 0)
            if change_pct < 0:
                direction = "\U0001f4c9 PRICE DROP"
                color = 0xFF4444
            else:
                # Only reaches here when pnl_pct >= 0 (gated in check_positions)
                direction = "\U0001f4b0 NOW PROFITABLE \u2014 CONSIDER SELLING"
                color = 0x00FF88

            embed = {
                "title": f"{direction}: {pos['item_name']}",
                "color": color,
                "fields": [
                    {"name": "\U0001f6d2 Buy Price", "value": f"{pos['buy_price']:,} GP", "inline": True},
                    {"name": "\U0001f4b2 Current Price", "value": f"{update['current_price']:,} GP", "inline": True},
                    {"name": "\U0001f4c9 Change", "value": f"{change_pct:+.1f}%", "inline": True},
                    {"name": "\U0001f3af Recommended Sell", "value": f"{update['recommended_sell']:,} GP", "inline": True},
                    {"name": "\U0001f4b0 Est. Profit", "value": f"{update['recommended_profit']:+,} GP ({update['recommended_profit_pct']:+.1f}%)", "inline": True},
                    {"name": "\U0001f4e6 Quantity", "value": f"{pos['quantity']:,}", "inline": True},
                ],
                "footer": {"text": "OSRS Flipping AI \u2022 Position Monitor"},
                "timestamp": datetime.utcnow().isoformat(),
            }

            _requests.post(webhook_url, json={"embeds": [embed]}, timeout=10)
            logger.info("Discord position alert sent for %s", pos["item_name"])
        except Exception as e:
            logger.error("Failed to send Discord position alert: %s", e)

    def get_positions_with_prices(self, source: Optional[str] = None, player: Optional[str] = None) -> List[Dict]:
        """Get all active positions with live pricing (for API endpoint)."""
        pricer = self._get_pricer()
        positions = self._get_positions(source=source, player=player)
        if not positions or pricer is None:
            return positions or []

        enriched = []
        for pos in positions:
            result = self._evaluate_position(pricer, pos)
            if result:
                pos.update(result)
            enriched.append(pos)
        return enriched

    async def _get_sell_alert_webhook(self) -> Optional[str]:
        """Return the dedicated sell-alert webhook URL (or general webhook as fallback)."""
        def _sync():
            db = get_db()
            try:
                url = get_setting(db, "sell_alert_webhook_url")
                if url:
                    return url
                # Fallback to general webhook if enabled
                wh = get_setting(db, "discord_webhook")
                if isinstance(wh, dict):
                    return wh.get("url") if wh.get("enabled") else None
                url = get_setting(db, "discord_webhook_url")
                enabled = get_setting(db, "discord_alerts_enabled", False)
                return url if url and enabled else None
            finally:
                db.close()
        return await asyncio.to_thread(_sync)

    async def check_selling_offers(self):
        """Check active SELLING offers against current market price.

        If the current instant-buy price (what buyers pay) has dropped more than
        SELL_DROP_THRESHOLD% below the player's listed sell price, fire:
        - WS broadcast so the Portfolio page highlights the row
        - Discord alert via the dedicated sell-alert webhook
        - DB Alert record
        """
        def _get_sells():
            db = get_db()
            try:
                return find_pending_sells(db)
            finally:
                db.close()

        sells = await asyncio.to_thread(_get_sells)
        if not sells:
            return

        pricer = self._get_pricer()
        now = datetime.utcnow()
        sell_alerts = []

        for sell in sells:
            item_id = sell["item_id"]
            listed_price = sell["listed_sell_price"]
            if not listed_price:
                continue

            try:
                def _eval(iid=item_id):
                    db = get_db()
                    try:
                        snaps = get_price_history(db, iid, hours=1)
                        if not snaps:
                            return None
                        latest = snaps[-1]
                        # instant_buy = what buyers are currently paying (the relevant price for sell offers)
                        return latest.instant_buy or latest.instant_sell
                    finally:
                        db.close()

                current_market = await asyncio.to_thread(_eval)
                if not current_market:
                    continue

                drop_pct = (listed_price - current_market) / listed_price * 100
                if drop_pct < self.SELL_DROP_THRESHOLD:
                    continue

                # Cooldown check (use a sell-specific state key)
                state_key = f"sell_{item_id}"
                state = self._state.get(state_key, {})
                last_alerted_at = state.get("last_alerted_at")
                on_cooldown = (
                    last_alerted_at
                    and (now - last_alerted_at).total_seconds() < self.COOLDOWN_MINUTES * 60
                )
                if on_cooldown:
                    continue

                alert_msg = (
                    f"{sell['item_name']} market price dropped {drop_pct:.1f}% below your "
                    f"listed sell price ({listed_price:,} GP → market {current_market:,} GP). "
                    f"Consider re-listing at {int(current_market * 0.99):,} GP."
                )

                sell_alert = {
                    "item_id": item_id,
                    "item_name": sell["item_name"],
                    "listed_sell_price": listed_price,
                    "current_market_price": current_market,
                    "drop_pct": round(drop_pct, 1),
                    "suggested_relist": int(current_market * 0.99),
                    "quantity": sell["quantity"],
                    "listed_at": sell["listed_at"],
                }
                sell_alerts.append(sell_alert)

                self._state[state_key] = {"last_alerted_at": now}

                # Save DB alert
                await asyncio.to_thread(
                    self._save_alert, item_id, sell["item_name"], alert_msg, sell_alert
                )

                # Discord for drops >= LARGE_CHANGE_PCT
                if drop_pct >= self.LARGE_CHANGE_PCT:
                    asyncio.create_task(self._send_discord_sell_alert(sell_alert))

            except Exception as e:
                logger.debug("PositionMonitor: sell-check error for %s: %s", sell.get("item_name"), e)

        # Broadcast all sell alerts to connected frontends
        if sell_alerts:
            await manager.broadcast_json({
                "type": "selling_price_alert",
                "timestamp": now.isoformat(),
                "alerts": sell_alerts,
            })

    async def _send_discord_sell_alert(self, alert: dict):
        """Send a Discord embed for a sell offer that has been undercut by the market."""
        try:
            webhook_url = await self._get_sell_alert_webhook()
            if not webhook_url:
                return

            import requests as _requests

            embed = {
                "title": f"📉 SELL PRICE DROP: {alert['item_name']}",
                "color": 0xFF4444,
                "fields": [
                    {"name": "🏷️ Listed Sell Price", "value": f"{alert['listed_sell_price']:,} GP", "inline": True},
                    {"name": "📊 Current Market", "value": f"{alert['current_market_price']:,} GP", "inline": True},
                    {"name": "📉 Drop", "value": f"{alert['drop_pct']:+.1f}%", "inline": True},
                    {"name": "💡 Suggested Re-list", "value": f"{alert['suggested_relist']:,} GP", "inline": True},
                    {"name": "📦 Quantity", "value": f"{alert['quantity']:,}", "inline": True},
                ],
                "description": "Market price has dropped below your listed sell price — you may be stuck selling for hours. Consider cancelling and re-listing.",
                "footer": {"text": "OSRS Flipping AI • Sell Price Watcher"},
                "timestamp": datetime.utcnow().isoformat(),
            }
            _requests.post(webhook_url, json={"embeds": [embed]}, timeout=10)
            logger.info("Discord sell-price alert sent for %s", alert["item_name"])
        except Exception as e:
            logger.error("Failed to send Discord sell alert: %s", e)

    def _auto_archive_stale_positions(self):
        """Auto-dismiss positions that have been open longer than the configured
        archive threshold.  Runs once on startup and every 6 hours.

        A position is considered stale when:
          - Its timestamp is older than `position_auto_archive_days` days
          - It still has no matching sell (i.e. it is still "active")

        This cleans up ghost positions caused by:
          - Dink missing the SELL webhook while you were offline
          - CSV imports of old trades that were already completed
          - Items sold for a loss before the plugin was installed

        The threshold is stored in the ``position_auto_archive_days`` setting.
        A value of 0 disables auto-archiving.
        """
        db = get_db()
        try:
            days = get_setting(db, "position_auto_archive_days", default=7)
            if not days or int(days) <= 0:
                return

            cutoff = datetime.utcnow() - timedelta(days=int(days))
            positions = find_active_positions(db)
            archived = 0

            for pos in positions:
                bought_at_raw = pos.get("bought_at")
                if not bought_at_raw:
                    continue
                # bought_at is stored as an ISO string by find_active_positions
                try:
                    ts = datetime.fromisoformat(bought_at_raw.replace("Z", "+00:00"))
                    # Normalise to naive UTC for comparison
                    if ts.tzinfo is not None:
                        ts = ts.replace(tzinfo=None)
                except Exception:
                    continue
                if ts < cutoff:
                    dismiss_position(db, pos["trade_id"])
                    archived += 1
                    age_days = (datetime.utcnow() - ts).days
                    logger.info(
                        "Auto-archived stale position: %s (opened %s, %d days old)",
                        pos.get("item_name", pos.get("item_id")),
                        ts.strftime("%Y-%m-%d"),
                        age_days,
                    )

            if archived:
                logger.info("Auto-archive complete: dismissed %d stale position(s)", archived)
        except Exception as e:
            logger.error("Auto-archive error: %s", e)
        finally:
            db.close()

    async def run_forever(self):
        logger.info("PositionMonitor started (checks every %ds)", self.CHECK_INTERVAL)
        # Wait for initial price data
        await asyncio.sleep(45)

        # Run auto-archive on startup, then every 6 hours
        archive_interval = 6 * 3600
        last_archive = 0.0

        while True:
            try:
                await self.check_positions()
                await self.check_selling_offers()
            except Exception as e:
                logger.error("PositionMonitor tick error: %s", e)

            # Periodic auto-archive (non-blocking, runs in thread pool)
            if time.time() - last_archive >= archive_interval:
                try:
                    await asyncio.to_thread(self._auto_archive_stale_positions)
                    last_archive = time.time()
                except Exception as e:
                    logger.error("PositionMonitor auto-archive error: %s", e)

            await asyncio.sleep(self.CHECK_INTERVAL)


# Singleton
_position_monitor: Optional[PositionMonitor] = None


def get_position_monitor() -> PositionMonitor:
    global _position_monitor
    if _position_monitor is None:
        _position_monitor = PositionMonitor()
    return _position_monitor


# ---------------------------------------------------------------------------
# FlipCacheWorker (PR10) — maintains the stable in-memory top lists
# ---------------------------------------------------------------------------

class FlipCacheWorker:
    """Scores all tracked items every ~60 seconds and updates the in-memory
    flip cache with dampening / hysteresis applied.

    This is the only writer of ``backend.flip_cache``.  The /flips/top5
    endpoint reads from that cache without touching the database.
    """

    CYCLE_SECONDS = 60

    async def run_cycle(self) -> None:
        """Score items and update the flip cache."""
        from backend.database import (
            get_db, get_tracked_item_ids, get_price_history,
            get_item_flips, get_item,
        )
        from backend.prediction.scoring import calculate_flip_metrics
        import backend.flip_cache as _cache

        def _sync() -> List[dict]:
            db = get_db()
            results = []
            try:
                item_ids = get_tracked_item_ids(db)
                for item_id in item_ids[:200]:
                    try:
                        snaps = get_price_history(db, item_id, hours=4)
                        if not snaps:
                            continue
                        flips = get_item_flips(db, item_id, days=30)
                        item  = get_item(db, item_id)
                        latest = snaps[-1]
                        metrics = calculate_flip_metrics({
                            "item_id":    item_id,
                            "item_name":  item.name if item else f"Item {item_id}",
                            "instant_buy":  latest.instant_buy,
                            "instant_sell": latest.instant_sell,
                            "volume_5m": (latest.buy_volume or 0) + (latest.sell_volume or 0),
                            "buy_time":  latest.buy_time,
                            "sell_time": latest.sell_time,
                            "snapshots": snaps,
                            "flip_history": flips,
                        })
                        results.append(metrics)
                    except Exception:
                        continue
            finally:
                db.close()
            return results

        try:
            scored = await asyncio.to_thread(_sync)
            _cache.update_cache(scored, cycle_seconds=self.CYCLE_SECONDS)
            logger.debug("FlipCacheWorker: updated cache with %d scored items", len(scored))
        except Exception as e:
            logger.error("FlipCacheWorker cycle error: %s", e)

    async def run_forever(self) -> None:
        logger.info("FlipCacheWorker started (cycle=%ds)", self.CYCLE_SECONDS)
        # Initial short delay to let PriceCollector gather some data
        await asyncio.sleep(30)
        while True:
            try:
                await self.run_cycle()
            except Exception as e:
                logger.error("FlipCacheWorker tick error: %s", e)
            await asyncio.sleep(self.CYCLE_SECONDS)


# ---------------------------------------------------------------------------
# OpportunityNotifier
# ---------------------------------------------------------------------------

class OpportunityNotifier:
    """Periodically sends a top-5 opportunity digest to Discord.

    Reads the webhook URL and interval from the settings collection.
    Settings keys:
        discord_webhook  → { url: "...", enabled: true/false }
        discord_top5_interval_minutes → int  (default 30)
    """

    DEFAULT_INTERVAL_MIN = 30  # minutes between digests
    _last_sent: Optional[datetime] = None

    async def _get_webhook_url(self) -> Optional[str]:
        """Read webhook URL from DB settings. Returns None if disabled."""
        def _sync():
            db = get_db()
            try:
                wh = get_setting(db, "discord_webhook")
                if not wh:
                    # Also try the flat key used by settings page
                    url = get_setting(db, "discord_webhook_url")
                    enabled = get_setting(db, "discord_alerts_enabled", False)
                    if url and enabled:
                        return url
                    return None
                if isinstance(wh, dict):
                    if not wh.get("enabled", False):
                        return None
                    return wh.get("url") or None
                # Plain string
                return wh if wh else None
            finally:
                db.close()
        return await asyncio.to_thread(_sync)

    async def _get_interval_minutes(self) -> int:
        def _sync():
            db = get_db()
            try:
                val = get_setting(db, "discord_top5_interval_minutes")
                return int(val) if val else self.DEFAULT_INTERVAL_MIN
            finally:
                db.close()
        return await asyncio.to_thread(_sync)

    async def _get_top_opportunities(self, limit: int = 5):
        """Score tracked items and return the top N dicts."""
        def _sync():
            from backend.flip_scorer import FlipScorer, score_opportunities
            from ai_strategist import scan_all_items_for_flips

            try:
                raw = scan_all_items_for_flips(
                    min_price=10_000,
                    max_price=500_000_000,
                    min_margin_pct=0.3,
                    max_risk=7,
                    limit=200,
                )
            except Exception as e:
                logger.error("OpportunityNotifier: scan failed: %s", e)
                return []

            scored = score_opportunities(raw, min_score=30, limit=limit * 3)

            results = []
            for fs in scored:
                if fs.expected_profit is not None and fs.expected_profit < 10_000:
                    continue
                results.append({
                    "item_id": fs.item_id,
                    "name": fs.item_name,
                    "buy_price": fs.recommended_buy,
                    "sell_price": fs.recommended_sell,
                    "potential_profit": fs.expected_profit,
                    "flip_score": round(fs.total_score, 1),
                    "margin_pct": round(fs.spread_pct, 2) if fs.spread_pct else 0,
                    "volume": fs.volume_5m,
                    "trend": fs.trend,
                    "ml_direction": fs.ml_direction,
                    "ml_prediction_confidence": round(fs.ml_confidence, 3) if fs.ml_confidence else None,
                    "ml_method": fs.ml_method,
                    "win_rate": round(fs.win_rate * 100, 1) if fs.win_rate is not None else None,
                    "reason": fs.reason,
                })
            results.sort(key=lambda x: x.get("flip_score", 0), reverse=True)
            return results[:limit]

        return await asyncio.to_thread(_sync)

    async def maybe_send(self):
        """Check if it's time to send, and send if so."""
        webhook_url = await self._get_webhook_url()
        if not webhook_url:
            return  # webhook not configured or disabled

        interval = await self._get_interval_minutes()
        now = datetime.utcnow()

        # Skip if we sent recently
        if self._last_sent and (now - self._last_sent).total_seconds() < interval * 60:
            return

        logger.info("OpportunityNotifier: building top-5 digest…")
        top = await self._get_top_opportunities(limit=5)
        if not top:
            logger.info("OpportunityNotifier: no qualifying opportunities right now")
            return

        from backend.discord_notifier import DiscordOpportunityNotifier
        notifier = DiscordOpportunityNotifier(webhook_url)

        ok = await asyncio.to_thread(
            notifier.send_top_opportunities, top, max_items=5, include_charts=True
        )
        if ok:
            self._last_sent = now
            logger.info("OpportunityNotifier: digest sent (%d items)", len(top))
        else:
            logger.warning("OpportunityNotifier: sending failed")

    # Status tracking for the frontend
    _send_status: str = "idle"  # idle | scanning | sending | done | error
    _send_result: Optional[dict] = None

    async def send_now(self) -> dict:
        """Kick off a top-5 digest in the background (non-blocking).

        Returns immediately with {"ok": True, "status": "scanning"}.
        The frontend can poll /api/alerts/send-top5/status for progress.
        """
        if self._send_status in ("scanning", "sending"):
            return {"ok": True, "status": self._send_status, "message": "Already in progress"}

        webhook_url = await self._get_webhook_url()
        if not webhook_url:
            return {"ok": False, "error": "Discord webhook not configured or disabled"}

        # Launch background work
        self._send_status = "scanning"
        self._send_result = None
        asyncio.create_task(self._do_send_background(webhook_url))
        return {"ok": True, "status": "scanning", "message": "Scanning for opportunities…"}

    async def _do_send_background(self, webhook_url: str):
        """Heavy lifting: scan → score → chart → Discord. Runs as a task."""
        try:
            top = await self._get_top_opportunities(limit=5)
            if not top:
                self._send_status = "error"
                self._send_result = {"ok": False, "error": "No qualifying opportunities found"}
                return

            self._send_status = "sending"
            from backend.discord_notifier import DiscordOpportunityNotifier
            notifier = DiscordOpportunityNotifier(webhook_url)
            ok = await asyncio.to_thread(
                notifier.send_top_opportunities, top, max_items=5, include_charts=True
            )
            if ok:
                self._last_sent = datetime.utcnow()
            self._send_status = "done" if ok else "error"
            self._send_result = {
                "ok": ok,
                "items_sent": len(top) if ok else 0,
                "items": [
                    {"name": o["name"], "score": o["flip_score"], "profit": o["potential_profit"]}
                    for o in top
                ],
            }
        except Exception as e:
            logger.error("OpportunityNotifier background send error: %s", e)
            self._send_status = "error"
            self._send_result = {"ok": False, "error": str(e)}

    def get_send_status(self) -> dict:
        """Return current send status for polling."""
        result = {
            "status": self._send_status,
            "last_sent": self._last_sent.isoformat() if self._last_sent else None,
        }
        if self._send_result:
            result.update(self._send_result)
        # Auto-reset to idle after frontend reads a terminal state
        if self._send_status in ("done", "error"):
            self._send_status = "idle"
        return result

    async def run_forever(self):
        logger.info("OpportunityNotifier started (default interval: %dm)", self.DEFAULT_INTERVAL_MIN)
        # Wait 60s for initial data collection
        await asyncio.sleep(60)
        while True:
            try:
                await self.maybe_send()
            except Exception as e:
                logger.error("OpportunityNotifier tick error: %s", e)
            # Check every 60 seconds whether it's time to send
            await asyncio.sleep(60)


# Singleton for manual-trigger access
_opportunity_notifier: Optional[OpportunityNotifier] = None


def get_opportunity_notifier() -> OpportunityNotifier:
    global _opportunity_notifier
    if _opportunity_notifier is None:
        _opportunity_notifier = OpportunityNotifier()
    return _opportunity_notifier


# ---------------------------------------------------------------------------
# Task launcher (called from app.py on startup)
# ---------------------------------------------------------------------------

_tasks = []


async def _emergency_compact_db():
    """One-time startup cleanup: nuke the accumulated price_snapshots bloat.

    Keeps only the last 24 hours of snapshots. With the new 5-minute write
    interval that's at most 50 items × 288 writes = 14,400 docs, vs the
    millions that may be sitting in Atlas right now.
    """
    def _sync():
        db = get_db()
        try:
            cutoff = datetime.utcnow() - timedelta(hours=24)
            result = db.price_snapshots.delete_many({"timestamp": {"$lt": cutoff}})
            if result.deleted_count:
                logger.info("Startup compact: deleted %d stale price_snapshots", result.deleted_count)
            # Also prune old predictions and metrics that accumulate silently
            cutoff3 = datetime.utcnow() - timedelta(days=1)  # match 1-day TTL in DataPruner
            db.predictions.delete_many({"timestamp": {"$lt": cutoff3}})
            db.model_metrics.delete_many({"timestamp": {"$lt": cutoff3}})
        except Exception as e:
            logger.warning("Startup compact failed (non-fatal): %s", e)
        finally:
            db.close()

    await asyncio.to_thread(_sync)


async def _seed_sell_alert_webhook():
    """One-time seed: if sell_alert_webhook_url is unset in DB, write the default."""
    _SELL_ALERT_DEFAULT = "https://discord.com/api/webhooks/1474337747446796351/EjVW1qDmMl3th8gAQ-ra8idgw3qHVSlQbpt_GqPeZ3bG3-teskzT2NKICLJGdEKsgirJ"

    def _sync():
        db = get_db()
        try:
            existing = get_setting(db, "sell_alert_webhook_url")
            if not existing:
                set_setting(db, "sell_alert_webhook_url", _SELL_ALERT_DEFAULT)
                logger.info("Seeded sell_alert_webhook_url in settings")
        finally:
            db.close()

    await asyncio.to_thread(_sync)


async def start_background_tasks():
    """Create and start all background asyncio tasks, staggered to avoid
    overloading MongoDB with simultaneous queries."""
    global _tasks

    # Immediately compact the DB to reclaim space before any new writes
    await _emergency_compact_db()
    await _seed_sell_alert_webhook()

    collector = PriceCollector()
    feature_computer = FeatureComputer()
    scorer = MLScorer()
    retrainer = ModelRetrainer()
    alert_monitor = AlertMonitor()
    pruner = DataPruner()
    opp_notifier = get_opportunity_notifier()
    pos_monitor = get_position_monitor()
    flip_cache_worker = FlipCacheWorker()   # PR10

    # Start PriceCollector first (it feeds data to everything else)
    _tasks.append(asyncio.create_task(collector.run_forever()))
    logger.info("PriceCollector task created")

    # Stagger the remaining tasks so they don't all hit MongoDB at once
    await asyncio.sleep(15)  # Let collector gather some data first
    _tasks.append(asyncio.create_task(feature_computer.run_forever()))
    logger.info("FeatureComputer task created")

    await asyncio.sleep(10)
    _tasks.append(asyncio.create_task(scorer.run_forever()))
    logger.info("MLScorer task created")

    await asyncio.sleep(10)
    _tasks.append(asyncio.create_task(alert_monitor.run_forever()))
    logger.info("AlertMonitor task created")

    # FlipCacheWorker: maintains stable in-memory top lists (PR10)
    _tasks.append(asyncio.create_task(flip_cache_worker.run_forever()))
    logger.info("FlipCacheWorker task created")

    # These run infrequently, start them last
    _tasks.append(asyncio.create_task(retrainer.run_forever()))
    _tasks.append(asyncio.create_task(pruner.run_forever()))
    _tasks.append(asyncio.create_task(opp_notifier.run_forever()))
    _tasks.append(asyncio.create_task(pos_monitor.run_forever()))

    logger.info("All %d background tasks started", len(_tasks))


async def stop_background_tasks():
    """Cancel all running background tasks."""
    for task in _tasks:
        task.cancel()
    if _tasks:
        await asyncio.gather(*_tasks, return_exceptions=True)
    logger.info("All background tasks stopped")
