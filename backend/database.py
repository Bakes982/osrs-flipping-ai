"""
MongoDB Database Layer for OSRS Flipping AI
Stores price history, trades, predictions, and model metrics.

Replaces the previous SQLite/SQLAlchemy implementation with pymongo
while keeping the same external interface (dataclasses, helper functions).
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any

from pymongo import MongoClient, ASCENDING, DESCENDING
from bson import ObjectId

from backend.config import MONGODB_URL, DATABASE_NAME

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level connection state
# ---------------------------------------------------------------------------

_client: Optional[MongoClient] = None
_wrapper: Optional["Database"] = None


# ---------------------------------------------------------------------------
# Dataclass models (attribute-compatible with old SQLAlchemy models)
# ---------------------------------------------------------------------------

@dataclass
class Item:
    id: int
    name: str
    members: bool = True
    buy_limit: int = 10000
    high_alch: int = 0
    category: Optional[str] = None
    last_updated: Optional[datetime] = None

    @classmethod
    def from_doc(cls, doc):
        if doc is None:
            return None
        return cls(
            id=doc.get("_id", doc.get("id", 0)),
            name=doc.get("name", ""),
            members=doc.get("members", True),
            buy_limit=doc.get("buy_limit", 10000),
            high_alch=doc.get("high_alch", 0),
            category=doc.get("category"),
            last_updated=doc.get("last_updated"),
        )

    def to_doc(self):
        return {
            "_id": self.id,
            "name": self.name,
            "members": self.members,
            "buy_limit": self.buy_limit,
            "high_alch": self.high_alch,
            "category": self.category,
            "last_updated": self.last_updated or datetime.utcnow(),
        }


@dataclass
class PriceSnapshot:
    item_id: int
    timestamp: datetime
    instant_buy: Optional[int] = None
    instant_sell: Optional[int] = None
    buy_time: Optional[int] = None
    sell_time: Optional[int] = None
    avg_buy: Optional[int] = None
    avg_sell: Optional[int] = None
    buy_volume: int = 0
    sell_volume: int = 0

    @classmethod
    def from_doc(cls, doc):
        if doc is None:
            return None
        return cls(
            item_id=doc.get("item_id", 0),
            timestamp=doc.get("timestamp", datetime.utcnow()),
            instant_buy=doc.get("instant_buy"),
            instant_sell=doc.get("instant_sell"),
            buy_time=doc.get("buy_time"),
            sell_time=doc.get("sell_time"),
            avg_buy=doc.get("avg_buy"),
            avg_sell=doc.get("avg_sell"),
            buy_volume=doc.get("buy_volume", 0),
            sell_volume=doc.get("sell_volume", 0),
        )

    def to_doc(self):
        return {
            "item_id": self.item_id,
            "timestamp": self.timestamp,
            "instant_buy": self.instant_buy,
            "instant_sell": self.instant_sell,
            "buy_time": self.buy_time,
            "sell_time": self.sell_time,
            "avg_buy": self.avg_buy,
            "avg_sell": self.avg_sell,
            "buy_volume": self.buy_volume,
            "sell_volume": self.sell_volume,
        }


@dataclass
class PriceAggregate:
    item_id: int
    timestamp: datetime
    interval: str
    open_buy: Optional[int] = None
    close_buy: Optional[int] = None
    high_buy: Optional[int] = None
    low_buy: Optional[int] = None
    open_sell: Optional[int] = None
    close_sell: Optional[int] = None
    high_sell: Optional[int] = None
    low_sell: Optional[int] = None
    total_buy_volume: int = 0
    total_sell_volume: int = 0
    snapshot_count: int = 0

    def to_doc(self):
        return {
            "item_id": self.item_id,
            "timestamp": self.timestamp,
            "interval": self.interval,
            "open_buy": self.open_buy,
            "close_buy": self.close_buy,
            "high_buy": self.high_buy,
            "low_buy": self.low_buy,
            "open_sell": self.open_sell,
            "close_sell": self.close_sell,
            "high_sell": self.high_sell,
            "low_sell": self.low_sell,
            "total_buy_volume": self.total_buy_volume,
            "total_sell_volume": self.total_sell_volume,
            "snapshot_count": self.snapshot_count,
        }


@dataclass
class Trade:
    item_id: int
    item_name: str
    trade_type: str
    status: str
    quantity: int
    price: int
    total_value: int
    id: Optional[str] = None
    timestamp: Optional[datetime] = None
    player: Optional[str] = None
    slot: Optional[int] = None
    market_price: Optional[int] = None
    seller_tax: Optional[int] = None
    market_high: Optional[int] = None
    market_low: Optional[int] = None
    source: str = "dink"

    @classmethod
    def from_doc(cls, doc):
        if doc is None:
            return None
        return cls(
            id=str(doc["_id"]) if "_id" in doc else None,
            item_id=doc.get("item_id", 0),
            item_name=doc.get("item_name", ""),
            trade_type=doc.get("trade_type", ""),
            status=doc.get("status", ""),
            quantity=doc.get("quantity", 0),
            price=doc.get("price", 0),
            total_value=doc.get("total_value", 0),
            timestamp=doc.get("timestamp"),
            player=doc.get("player"),
            slot=doc.get("slot"),
            market_price=doc.get("market_price"),
            seller_tax=doc.get("seller_tax"),
            market_high=doc.get("market_high"),
            market_low=doc.get("market_low"),
            source=doc.get("source", "dink"),
        )

    def to_doc(self):
        return {
            "item_id": self.item_id,
            "item_name": self.item_name,
            "trade_type": self.trade_type,
            "status": self.status,
            "quantity": self.quantity,
            "price": self.price,
            "total_value": self.total_value,
            "timestamp": self.timestamp or datetime.utcnow(),
            "player": self.player,
            "slot": self.slot,
            "market_price": self.market_price,
            "seller_tax": self.seller_tax,
            "market_high": self.market_high,
            "market_low": self.market_low,
            "source": self.source,
        }


@dataclass
class FlipHistory:
    item_id: int
    item_name: str
    buy_price: int
    sell_price: int
    quantity: int
    gross_profit: int
    tax: int
    net_profit: int
    margin_pct: float
    buy_time: datetime
    sell_time: datetime
    duration_seconds: int
    id: Optional[str] = None
    player: Optional[str] = None
    buy_trade_id: Optional[str] = None
    sell_trade_id: Optional[str] = None

    @classmethod
    def from_doc(cls, doc):
        if doc is None:
            return None
        return cls(
            id=str(doc["_id"]) if "_id" in doc else None,
            item_id=doc.get("item_id", 0),
            item_name=doc.get("item_name", ""),
            player=doc.get("player"),
            buy_trade_id=str(doc["buy_trade_id"]) if doc.get("buy_trade_id") else None,
            sell_trade_id=str(doc["sell_trade_id"]) if doc.get("sell_trade_id") else None,
            buy_price=doc.get("buy_price", 0),
            sell_price=doc.get("sell_price", 0),
            quantity=doc.get("quantity", 0),
            gross_profit=doc.get("gross_profit", 0),
            tax=doc.get("tax", 0),
            net_profit=doc.get("net_profit", 0),
            margin_pct=doc.get("margin_pct", 0.0),
            buy_time=doc.get("buy_time", datetime.utcnow()),
            sell_time=doc.get("sell_time", datetime.utcnow()),
            duration_seconds=doc.get("duration_seconds", 0),
        )

    def to_doc(self):
        return {
            "item_id": self.item_id,
            "item_name": self.item_name,
            "player": self.player,
            "buy_trade_id": ObjectId(self.buy_trade_id) if self.buy_trade_id else None,
            "sell_trade_id": ObjectId(self.sell_trade_id) if self.sell_trade_id else None,
            "buy_price": self.buy_price,
            "sell_price": self.sell_price,
            "quantity": self.quantity,
            "gross_profit": self.gross_profit,
            "tax": self.tax,
            "net_profit": self.net_profit,
            "margin_pct": self.margin_pct,
            "buy_time": self.buy_time,
            "sell_time": self.sell_time,
            "duration_seconds": self.duration_seconds,
        }


@dataclass
class Prediction:
    item_id: int
    horizon: str
    timestamp: Optional[datetime] = None
    predicted_buy: Optional[int] = None
    predicted_sell: Optional[int] = None
    predicted_direction: Optional[str] = None
    confidence: Optional[float] = None
    actual_buy: Optional[int] = None
    actual_sell: Optional[int] = None
    actual_direction: Optional[str] = None
    outcome_recorded: bool = False
    model_version: Optional[str] = None
    id: Optional[str] = None

    @classmethod
    def from_doc(cls, doc):
        if doc is None:
            return None
        return cls(
            id=str(doc["_id"]) if "_id" in doc else None,
            item_id=doc.get("item_id", 0),
            horizon=doc.get("horizon", ""),
            timestamp=doc.get("timestamp"),
            predicted_buy=doc.get("predicted_buy"),
            predicted_sell=doc.get("predicted_sell"),
            predicted_direction=doc.get("predicted_direction"),
            confidence=doc.get("confidence"),
            actual_buy=doc.get("actual_buy"),
            actual_sell=doc.get("actual_sell"),
            actual_direction=doc.get("actual_direction"),
            outcome_recorded=doc.get("outcome_recorded", False),
            model_version=doc.get("model_version"),
        )

    def to_doc(self):
        return {
            "item_id": self.item_id,
            "horizon": self.horizon,
            "timestamp": self.timestamp or datetime.utcnow(),
            "predicted_buy": self.predicted_buy,
            "predicted_sell": self.predicted_sell,
            "predicted_direction": self.predicted_direction,
            "confidence": self.confidence,
            "actual_buy": self.actual_buy,
            "actual_sell": self.actual_sell,
            "actual_direction": self.actual_direction,
            "outcome_recorded": self.outcome_recorded,
            "model_version": self.model_version,
        }


@dataclass
class ModelMetrics:
    horizon: str
    model_version: str
    timestamp: Optional[datetime] = None
    direction_accuracy: Optional[float] = None
    price_mae: Optional[float] = None
    price_mape: Optional[float] = None
    profit_accuracy: Optional[float] = None
    sample_count: int = 0

    def to_doc(self):
        return {
            "horizon": self.horizon,
            "model_version": self.model_version,
            "timestamp": self.timestamp or datetime.utcnow(),
            "direction_accuracy": self.direction_accuracy,
            "price_mae": self.price_mae,
            "price_mape": self.price_mape,
            "profit_accuracy": self.profit_accuracy,
            "sample_count": self.sample_count,
        }

    @classmethod
    def from_doc(cls, doc):
        if doc is None:
            return None
        return cls(
            horizon=doc.get("horizon", ""),
            model_version=doc.get("model_version", ""),
            timestamp=doc.get("timestamp"),
            direction_accuracy=doc.get("direction_accuracy"),
            price_mae=doc.get("price_mae"),
            price_mape=doc.get("price_mape"),
            profit_accuracy=doc.get("profit_accuracy"),
            sample_count=doc.get("sample_count", 0),
        )


@dataclass
class ItemFeature:
    item_id: int
    timestamp: datetime
    features: Any  # JSON dict or str

    def to_doc(self):
        return {
            "item_id": self.item_id,
            "timestamp": self.timestamp,
            "features": self.features,
        }


@dataclass
class Alert:
    item_id: int
    item_name: str
    alert_type: str
    id: Optional[str] = None
    timestamp: Optional[datetime] = None
    message: Optional[str] = None
    data: Optional[dict] = None
    sent_discord: bool = False
    acknowledged: bool = False

    @classmethod
    def from_doc(cls, doc):
        if doc is None:
            return None
        return cls(
            id=str(doc["_id"]) if "_id" in doc else None,
            item_id=doc.get("item_id", 0),
            item_name=doc.get("item_name", ""),
            alert_type=doc.get("alert_type", ""),
            timestamp=doc.get("timestamp"),
            message=doc.get("message"),
            data=doc.get("data"),
            sent_discord=doc.get("sent_discord", False),
            acknowledged=doc.get("acknowledged", False),
        )

    def to_doc(self):
        return {
            "item_id": self.item_id,
            "item_name": self.item_name,
            "alert_type": self.alert_type,
            "timestamp": self.timestamp or datetime.utcnow(),
            "message": self.message,
            "data": self.data,
            "sent_discord": self.sent_discord,
            "acknowledged": self.acknowledged,
        }


# Setting is now just a key-value pair stored with _id = key
# No dedicated dataclass needed, but keep a placeholder for import compatibility
Setting = None  # Not used as a class; settings are plain dicts


# ---------------------------------------------------------------------------
# Database wrapper (backward-compatible close() method)
# ---------------------------------------------------------------------------

class Database:
    """Wraps a pymongo database and exposes collections as attributes.

    Provides a no-op ``close()`` for backward compatibility with code
    written for SQLAlchemy's session pattern.
    """

    def __init__(self, db):
        self._db = db
        self.items = db["items"]
        self.price_snapshots = db["price_snapshots"]
        self.price_aggregates = db["price_aggregates"]
        self.trades = db["trades"]
        self.flip_history = db["flip_history"]
        self.predictions = db["predictions"]
        self.model_metrics = db["model_metrics"]
        self.item_features = db["item_features"]
        self.alerts = db["alerts"]
        self.settings = db["settings"]

    def close(self):
        """No-op. Pymongo connection pooling handles cleanup."""
        pass

    def commit(self):
        """No-op. MongoDB auto-commits writes."""
        pass

    def rollback(self):
        """No-op. No transaction in default pymongo usage."""
        pass


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

def init_db():
    """Initialize MongoDB connection, create indexes."""
    global _client, _wrapper

    try:
        _client = MongoClient(
            MONGODB_URL,
            serverSelectionTimeoutMS=10_000,  # fail fast if unreachable
            maxPoolSize=10,        # limit connection pool
            minPoolSize=1,
            connectTimeoutMS=5_000,
            socketTimeoutMS=30_000,
        )
        # Force a connection test so errors surface at startup
        _client.admin.command("ping")
    except Exception as exc:
        logger.error("MongoDB connection FAILED (%s). Is MONGODB_URL set correctly?", exc)
        raise

    db = _client[DATABASE_NAME]
    _wrapper = Database(db)

    # Create indexes
    _wrapper.price_snapshots.create_index(
        [("item_id", ASCENDING), ("timestamp", ASCENDING)],
        background=True,
    )
    _wrapper.price_snapshots.create_index(
        [("timestamp", ASCENDING)],
        background=True,
    )
    _wrapper.price_aggregates.create_index(
        [("item_id", ASCENDING), ("timestamp", ASCENDING), ("interval", ASCENDING)],
        background=True,
    )
    _wrapper.trades.create_index(
        [("item_id", ASCENDING), ("timestamp", DESCENDING)],
        background=True,
    )
    _wrapper.trades.create_index(
        [("timestamp", DESCENDING)],
        background=True,
    )
    _wrapper.flip_history.create_index(
        [("item_id", ASCENDING), ("sell_time", DESCENDING)],
        background=True,
    )
    _wrapper.flip_history.create_index(
        [("buy_trade_id", ASCENDING)],
        background=True,
    )
    _wrapper.predictions.create_index(
        [("item_id", ASCENDING), ("horizon", ASCENDING), ("timestamp", DESCENDING)],
        background=True,
    )
    _wrapper.model_metrics.create_index(
        [("horizon", ASCENDING), ("timestamp", DESCENDING)],
        background=True,
    )
    _wrapper.item_features.create_index(
        [("item_id", ASCENDING)],
        unique=True,
        background=True,
    )
    _wrapper.alerts.create_index(
        [("item_id", ASCENDING), ("timestamp", DESCENDING)],
        background=True,
    )
    _wrapper.alerts.create_index(
        [("timestamp", DESCENDING)],
        background=True,
    )
    # Player (RSN) indexes for multi-account filtering
    _wrapper.trades.create_index(
        [("player", ASCENDING), ("timestamp", DESCENDING)],
        background=True,
    )
    _wrapper.flip_history.create_index(
        [("player", ASCENDING), ("sell_time", DESCENDING)],
        background=True,
    )

    logger.info("MongoDB initialized: %s / %s", MONGODB_URL.split("@")[-1], DATABASE_NAME)


def get_db() -> Database:
    """Get the Database wrapper. Caller may call .close() (no-op)."""
    if _wrapper is None:
        init_db()
    return _wrapper


# Backward compatibility alias
SessionLocal = get_db


# ---------------------------------------------------------------------------
# Helper queries (same signatures as old SQLAlchemy version)
# ---------------------------------------------------------------------------

def get_latest_price(db: Database, item_id: int) -> Optional[PriceSnapshot]:
    """Get the most recent price snapshot for an item."""
    doc = db.price_snapshots.find_one(
        {"item_id": item_id},
        sort=[("timestamp", DESCENDING)],
    )
    return PriceSnapshot.from_doc(doc) if doc else None


def get_price_history(
    db: Database,
    item_id: int,
    hours: int = 24,
    limit: int = 10000,
) -> List[PriceSnapshot]:
    """Get price snapshots for an item over the last N hours."""
    cutoff = datetime.utcnow() - timedelta(hours=hours)
    docs = (
        db.price_snapshots.find(
            {"item_id": item_id, "timestamp": {"$gte": cutoff}},
        )
        .sort("timestamp", ASCENDING)
        .limit(limit)
    )
    return [PriceSnapshot.from_doc(d) for d in docs]


def get_item_flips(
    db: Database,
    item_id: int,
    days: int = 30,
) -> List[FlipHistory]:
    """Get completed flips for an item."""
    cutoff = datetime.utcnow() - timedelta(days=days)
    docs = (
        db.flip_history.find(
            {"item_id": item_id, "sell_time": {"$gte": cutoff}},
        )
        .sort("sell_time", DESCENDING)
    )
    return [FlipHistory.from_doc(d) for d in docs]


def get_recent_predictions(
    db: Database,
    item_id: int,
    horizon: str,
    hours: int = 24,
) -> List[Prediction]:
    """Get recent predictions for backtesting."""
    cutoff = datetime.utcnow() - timedelta(hours=hours)
    docs = (
        db.predictions.find({
            "item_id": item_id,
            "horizon": horizon,
            "timestamp": {"$gte": cutoff},
        })
        .sort("timestamp", DESCENDING)
    )
    return [Prediction.from_doc(d) for d in docs]


def get_setting(db: Database, key: str, default: Any = None) -> Any:
    """Get a setting value."""
    doc = db.settings.find_one({"_id": key})
    return doc["value"] if doc else default


def set_setting(db: Database, key: str, value: Any):
    """Set a setting value (upsert)."""
    db.settings.update_one(
        {"_id": key},
        {"$set": {"value": value, "updated_at": datetime.utcnow()}},
        upsert=True,
    )


def get_tracked_item_ids(db: Database, hours: int = 1) -> List[int]:
    """Get all item IDs that have recent price data (active items)."""
    cutoff = datetime.utcnow() - timedelta(hours=hours)
    return db.price_snapshots.distinct(
        "item_id",
        {"timestamp": {"$gte": cutoff}},
    )


# ---------------------------------------------------------------------------
# Additional helpers for callers that previously used raw SQLAlchemy queries
# ---------------------------------------------------------------------------

def get_item(db: Database, item_id: int) -> Optional[Item]:
    """Get an item by ID."""
    doc = db.items.find_one({"_id": item_id})
    return Item.from_doc(doc) if doc else None


def upsert_item(db: Database, item: Item):
    """Insert or update an item."""
    db.items.update_one(
        {"_id": item.id},
        {"$set": item.to_doc()},
        upsert=True,
    )


def insert_price_snapshots(db: Database, snapshots: List[PriceSnapshot]) -> int:
    """Bulk insert price snapshots. Returns count inserted."""
    if not snapshots:
        return 0
    docs = [s.to_doc() for s in snapshots]
    result = db.price_snapshots.insert_many(docs)
    return len(result.inserted_ids)


def insert_trade(db: Database, trade: Trade) -> str:
    """Insert a trade, de-duplicating by (item_id, player, slot, status, quantity, price).

    DINK can send the same webhook multiple times — this prevents
    phantom duplicate positions from appearing in the portfolio.
    Returns the existing document's ID if a duplicate is found.
    """
    # Build a dedup key from the fields that uniquely identify a GE event
    dedup_query: Dict = {
        "item_id": trade.item_id,
        "player": trade.player,
        "status": trade.status,
        "quantity": trade.quantity,
        "price": trade.price,
    }
    if trade.slot is not None:
        dedup_query["slot"] = trade.slot
    # If a matching doc was inserted within the last 60 seconds, skip
    if trade.timestamp:
        dedup_query["timestamp"] = {
            "$gte": trade.timestamp - timedelta(seconds=60),
            "$lte": trade.timestamp + timedelta(seconds=60),
        }
    existing = db.trades.find_one(dedup_query)
    if existing:
        trade.id = str(existing["_id"])
        return trade.id

    doc = trade.to_doc()
    result = db.trades.insert_one(doc)
    trade.id = str(result.inserted_id)
    return trade.id


def insert_flip(db: Database, flip: FlipHistory) -> str:
    """Insert a flip history record."""
    doc = flip.to_doc()
    result = db.flip_history.insert_one(doc)
    flip.id = str(result.inserted_id)
    return flip.id


def insert_alert(db: Database, alert: Alert) -> str:
    """Insert an alert and return the new ObjectId as string."""
    doc = alert.to_doc()
    result = db.alerts.insert_one(doc)
    alert.id = str(result.inserted_id)
    return alert.id


def insert_prediction(db: Database, pred: Prediction) -> str:
    """Insert a prediction record."""
    doc = pred.to_doc()
    result = db.predictions.insert_one(doc)
    pred.id = str(result.inserted_id)
    return pred.id


def insert_model_metrics(db: Database, metrics: ModelMetrics):
    """Insert model metrics record."""
    db.model_metrics.insert_one(metrics.to_doc())


def upsert_item_feature(db: Database, item_id: int, features: Any, timestamp: datetime):
    """Upsert cached feature vector for an item."""
    db.item_features.update_one(
        {"item_id": item_id},
        {"$set": {"features": features, "timestamp": timestamp}},
        upsert=True,
    )


def get_all_settings(db: Database) -> Dict[str, Any]:
    """Get all settings as a dict."""
    docs = db.settings.find()
    return {doc["_id"]: doc["value"] for doc in docs}


def find_alerts(
    db: Database,
    hours: int = 24,
    alert_type: Optional[str] = None,
    unacknowledged_only: bool = False,
    limit: int = 50,
) -> List[Alert]:
    """Find alerts with optional filters."""
    cutoff = datetime.utcnow() - timedelta(hours=hours)
    query: Dict = {"timestamp": {"$gte": cutoff}}
    if alert_type:
        query["alert_type"] = alert_type
    if unacknowledged_only:
        query["acknowledged"] = False

    docs = (
        db.alerts.find(query)
        .sort("timestamp", DESCENDING)
        .limit(limit)
    )
    return [Alert.from_doc(d) for d in docs]


def acknowledge_alerts(db: Database, alert_ids: Optional[List[str]] = None, all_unack: bool = False) -> int:
    """Mark alerts as acknowledged. Returns count modified."""
    if all_unack:
        result = db.alerts.update_many(
            {"acknowledged": False},
            {"$set": {"acknowledged": True}},
        )
        return result.modified_count

    if alert_ids:
        oids = [ObjectId(aid) for aid in alert_ids if ObjectId.is_valid(aid)]
        if not oids:
            return 0
        result = db.alerts.update_many(
            {"_id": {"$in": oids}},
            {"$set": {"acknowledged": True}},
        )
        return result.modified_count

    return 0


def find_trades(
    db: Database,
    item_id: Optional[int] = None,
    limit: int = 100,
    player: Optional[str] = None,
    status: Optional[str] = None,
    completed_only: bool = False,
) -> List[Trade]:
    """Find trades, newest first.

    Args:
        item_id: Filter by item.
        limit: Max rows.
        player: Filter by player RSN.
        status: Exact status filter (e.g. 'BOUGHT', 'SOLD').
        completed_only: If True, only return trades with
            status BOUGHT or SOLD (i.e. actually filled).
    """
    query: Dict = {}
    if item_id is not None:
        query["item_id"] = item_id
    if player:
        query["player"] = player
    if status:
        query["status"] = status
    elif completed_only:
        query["status"] = {"$in": ["BOUGHT", "SOLD"]}
    docs = (
        db.trades.find(query)
        .sort("timestamp", DESCENDING)
        .limit(limit)
    )
    return [Trade.from_doc(d) for d in docs]


def find_all_flips(
    db: Database, limit: int = 5000, player: Optional[str] = None,
) -> List[FlipHistory]:
    """Find all flips, newest first.  Optionally filter by player (RSN)."""
    query: Dict = {}
    if player:
        query["player"] = player
    docs = (
        db.flip_history.find(query)
        .sort("sell_time", DESCENDING)
        .limit(limit)
    )
    return [FlipHistory.from_doc(d) for d in docs]


def get_matched_buy_trade_ids(
    db: Database, player: Optional[str] = None,
) -> set:
    """Return set of buy_trade_id strings that have been matched to a flip."""
    query: Dict = {"buy_trade_id": {"$ne": None}}
    if player:
        query["player"] = player
    docs = db.flip_history.find(query, {"buy_trade_id": 1})
    return {str(d["buy_trade_id"]) for d in docs}


def find_unmatched_buy_trades(db: Database, item_id: int, player: Optional[str] = None) -> List[Trade]:
    """Find BUY trades not yet matched to a flip (for flip matching)."""
    matched_ids = get_matched_buy_trade_ids(db)
    query: Dict = {
        "item_id": item_id,
        "trade_type": "BUY",
        "status": "BOUGHT",
    }
    if player:
        query["player"] = player
    docs = db.trades.find(query).sort("timestamp", ASCENDING)
    trades = []
    for d in docs:
        trade = Trade.from_doc(d)
        if trade.id not in matched_ids:
            trades.append(trade)
    return trades


def get_model_metrics_latest(db: Database, horizon: str) -> Optional[ModelMetrics]:
    """Get the most recent model metrics for a horizon."""
    doc = db.model_metrics.find_one(
        {"horizon": horizon},
        sort=[("timestamp", DESCENDING)],
    )
    return ModelMetrics.from_doc(doc) if doc else None


def find_pending_predictions(
    db: Database,
    item_id: int,
    horizon: str,
    cutoff: datetime,
    limit: int = 100,
) -> List[Prediction]:
    """Find predictions whose outcome hasn't been recorded yet."""
    docs = (
        db.predictions.find({
            "item_id": item_id,
            "horizon": horizon,
            "outcome_recorded": False,
            "timestamp": {"$lte": cutoff},
        })
        .limit(limit)
    )
    return [Prediction.from_doc(d) for d in docs]


def update_prediction_outcome(
    db: Database,
    pred_id: str,
    actual_buy: Optional[int],
    actual_sell: Optional[int],
    actual_direction: Optional[str],
):
    """Record the actual outcome for a prediction."""
    db.predictions.update_one(
        {"_id": ObjectId(pred_id)},
        {"$set": {
            "actual_buy": actual_buy,
            "actual_sell": actual_sell,
            "actual_direction": actual_direction,
            "outcome_recorded": True,
        }},
    )


def find_snapshot_near_time(
    db: Database,
    item_id: int,
    target_time: datetime,
    window_seconds: int = 30,
) -> Optional[PriceSnapshot]:
    """Find the price snapshot closest to a target time."""
    window = timedelta(seconds=window_seconds)
    doc = db.price_snapshots.find_one(
        {
            "item_id": item_id,
            "timestamp": {
                "$gte": target_time - window,
                "$lte": target_time + window,
            },
        },
        sort=[("timestamp", ASCENDING)],
    )
    return PriceSnapshot.from_doc(doc) if doc else None


# ---------------------------------------------------------------------------
# Active-position tracking helpers
# ---------------------------------------------------------------------------

def find_active_positions(
    db: Database,
    source: Optional[str] = None,
    player: Optional[str] = None,
) -> List[Dict]:
    """Return active (open) positions — BUY trades not yet matched to a flip.

    Args:
        source: Filter by trade source ('dink', 'csv_import', or None for all).
        player: Filter by player RSN.  None returns all players.
                Dismissed positions are always excluded.
    """
    matched_ids = get_matched_buy_trade_ids(db, player=player)
    dismissed = set(get_setting(db, "dismissed_positions", default=[]))

    query: Dict = {
        "trade_type": "BUY",
        "status": "BOUGHT",
    }
    if source:
        query["source"] = source
    if player:
        query["player"] = player

    docs = db.trades.find(query).sort("timestamp", DESCENDING)

    positions = []
    for d in docs:
        trade = Trade.from_doc(d)
        if trade.id not in matched_ids and trade.id not in dismissed:
            positions.append({
                "trade_id": trade.id,
                "item_id": trade.item_id,
                "item_name": trade.item_name,
                "quantity": trade.quantity,
                "buy_price": trade.price,
                "total_cost": trade.total_value,
                "bought_at": trade.timestamp.isoformat() if trade.timestamp else None,
                "player": trade.player,
                "market_price_at_buy": trade.market_price,
                "source": trade.source,
            })
    return positions


def dismiss_position(db: Database, trade_id: str):
    """Mark a position as dismissed so it no longer shows in active positions."""
    dismissed = get_setting(db, "dismissed_positions", default=[])
    if trade_id not in dismissed:
        dismissed.append(trade_id)
        set_setting(db, "dismissed_positions", dismissed)


def dismiss_positions_by_source(db: Database, source: str) -> int:
    """Dismiss all active positions from a given source (e.g. 'csv_import')."""
    positions = find_active_positions(db, source=source)
    dismissed = get_setting(db, "dismissed_positions", default=[])
    count = 0
    for p in positions:
        tid = p["trade_id"]
        if tid not in dismissed:
            dismissed.append(tid)
            count += 1
    if count:
        set_setting(db, "dismissed_positions", dismissed)
    return count


def get_position_monitoring_state(db: Database) -> Dict:
    """Get the saved position monitoring state (last-alerted prices etc.)."""
    return get_setting(db, "position_monitor_state", default={})


def get_distinct_players(db: Database) -> List[str]:
    """Return sorted list of distinct player RSNs from the trades collection."""
    players = db.trades.distinct("player")
    # Filter out None/empty strings
    return sorted([p for p in players if p])
