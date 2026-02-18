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
    get_latest_price,
    insert_price_snapshots,
    insert_alert,
    upsert_item_feature,
    get_item,
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

        Only stores the top ~200 items by volume to avoid filling MongoDB.
        Returns the number of rows inserted.
        """
        if not self._latest_data:
            return 0

        def _sync_store():
            now = datetime.utcnow()
            db = get_db()
            count = 0
            try:
                # Filter to items with meaningful volume (top ~200)
                items_with_volume = []
                for item_id_str, instant in self._latest_data.items():
                    avg = self._5m_data.get(item_id_str, {})
                    vol = (avg.get("highPriceVolume", 0) or 0) + (avg.get("lowPriceVolume", 0) or 0)
                    if vol > 0 and instant.get("high") and instant.get("low"):
                        items_with_volume.append((item_id_str, instant, avg, vol))

                # Sort by volume descending, keep top 200
                items_with_volume.sort(key=lambda x: x[3], reverse=True)
                items_with_volume = items_with_volume[:200]

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

        return await asyncio.to_thread(_sync_store)

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

    def _aggregate_snapshots_to_5m(self, db):
        """Aggregate raw snapshots older than 7 days into 5-minute candles."""
        cutoff = datetime.utcnow() - timedelta(days=7)

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
        """Roll up 5m candles older than 30 days into 1h candles."""
        cutoff = datetime.utcnow() - timedelta(days=30)

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
        """Delete raw snapshots older than 7 days (already aggregated)."""
        cutoff = datetime.utcnow() - timedelta(days=7)
        result = db.price_snapshots.delete_many({"timestamp": {"$lt": cutoff}})
        logger.info("Deleted %d old raw snapshots", result.deleted_count)

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
        """Detect items experiencing rapid price drops (>5% in 15 min). Sync version."""
        item_ids = get_tracked_item_ids(db)
        now = datetime.utcnow()
        cutoff_15m = now - timedelta(minutes=15)

        # Only check top 50 items to limit DB load
        for item_id in item_ids[:50]:
            try:
                docs = db.price_snapshots.find(
                    {"item_id": item_id, "timestamp": {"$gte": cutoff_15m}}
                ).sort("timestamp", 1)
                snaps = [PriceSnapshot.from_doc(d) for d in docs]

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
                    recent_alert = db.alerts.find_one({
                        "item_id": item_id,
                        "alert_type": "dump",
                        "timestamp": {"$gte": now - timedelta(minutes=30)},
                    })
                    if recent_alert:
                        continue

                    item_row = get_item(db, item_id)
                    item_name = item_row.name if item_row else f"Item {item_id}"

                    alert = Alert(
                        item_id=item_id,
                        item_name=item_name,
                        alert_type="dump",
                        message=f"{item_name} dropping {change_pct:.1f}% in 15min ({first_price:,} -> {last_price:,})",
                        data={"change_pct": round(change_pct, 1), "from_price": first_price, "to_price": last_price},
                    )
                    insert_alert(db, alert)
                    logger.info("Dump alert: %s", alert.message)
            except Exception:
                pass

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

    async def send_now(self) -> dict:
        """Force-send immediately (called by the manual trigger endpoint).

        Returns a summary dict.
        """
        webhook_url = await self._get_webhook_url()
        if not webhook_url:
            return {"ok": False, "error": "Discord webhook not configured or disabled"}

        top = await self._get_top_opportunities(limit=5)
        if not top:
            return {"ok": False, "error": "No qualifying opportunities found"}

        from backend.discord_notifier import DiscordOpportunityNotifier
        notifier = DiscordOpportunityNotifier(webhook_url)
        ok = await asyncio.to_thread(
            notifier.send_top_opportunities, top, max_items=5, include_charts=True
        )
        if ok:
            self._last_sent = datetime.utcnow()
        return {
            "ok": ok,
            "items_sent": len(top) if ok else 0,
            "items": [
                {"name": o["name"], "score": o["flip_score"], "profit": o["potential_profit"]}
                for o in top
            ],
        }

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


async def start_background_tasks():
    """Create and start all background asyncio tasks, staggered to avoid
    overloading MongoDB with simultaneous queries."""
    global _tasks

    collector = PriceCollector()
    feature_computer = FeatureComputer()
    scorer = MLScorer()
    retrainer = ModelRetrainer()
    alert_monitor = AlertMonitor()
    pruner = DataPruner()
    opp_notifier = get_opportunity_notifier()

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

    # These run infrequently, start them last
    _tasks.append(asyncio.create_task(retrainer.run_forever()))
    _tasks.append(asyncio.create_task(pruner.run_forever()))
    _tasks.append(asyncio.create_task(opp_notifier.run_forever()))

    logger.info("All %d background tasks started", len(_tasks))


async def stop_background_tasks():
    """Cancel all running background tasks."""
    for task in _tasks:
        task.cancel()
    if _tasks:
        await asyncio.gather(*_tasks, return_exceptions=True)
    logger.info("All background tasks stopped")
