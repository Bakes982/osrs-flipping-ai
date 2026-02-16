"""
SQLite Database Layer for OSRS Flipping AI
Stores price history, trades, predictions, and model metrics.
"""

import os
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any

from sqlalchemy import (
    create_engine, Column, Integer, Float, String, Boolean,
    DateTime, Text, Index, ForeignKey, JSON, event
)
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from sqlalchemy.pool import StaticPool

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "flipping_ai.db")

engine = create_engine(
    f"sqlite:///{DB_PATH}",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)

# Enable WAL mode for concurrent reads during writes
@event.listens_for(engine, "connect")
def _set_sqlite_pragma(dbapi_connection, connection_record):
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA synchronous=NORMAL")
    cursor.execute("PRAGMA cache_size=-64000")  # 64MB cache
    cursor.execute("PRAGMA busy_timeout=5000")
    cursor.close()

SessionLocal = sessionmaker(bind=engine, autoflush=False)
Base = declarative_base()


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class Item(Base):
    """Item metadata from Wiki mapping endpoint."""
    __tablename__ = "items"

    id = Column(Integer, primary_key=True)  # OSRS item ID
    name = Column(String(255), nullable=False, index=True)
    members = Column(Boolean, default=True)
    buy_limit = Column(Integer, default=10000)
    high_alch = Column(Integer, default=0)
    category = Column(String(64), nullable=True)  # e.g. "high_pvm_gear"
    last_updated = Column(DateTime, default=datetime.utcnow)


class PriceSnapshot(Base):
    """
    10-second price snapshots from Wiki API /latest.
    This is the core data for all analysis.
    """
    __tablename__ = "price_snapshots"

    id = Column(Integer, primary_key=True, autoincrement=True)
    item_id = Column(Integer, nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)

    # Instant prices from /latest
    instant_buy = Column(Integer, nullable=True)   # high (what buyers pay)
    instant_sell = Column(Integer, nullable=True)   # low (what sellers get)
    buy_time = Column(Integer, nullable=True)       # Unix timestamp of last buy
    sell_time = Column(Integer, nullable=True)       # Unix timestamp of last sell

    # 5-minute averaged prices from /5m (fetched less often)
    avg_buy = Column(Integer, nullable=True)
    avg_sell = Column(Integer, nullable=True)
    buy_volume = Column(Integer, default=0)
    sell_volume = Column(Integer, default=0)

    __table_args__ = (
        Index("ix_price_item_ts", "item_id", "timestamp"),
    )


class PriceAggregate(Base):
    """
    Aggregated price data (5-min or 1-hour candles).
    Created by the data pruner from raw snapshots.
    """
    __tablename__ = "price_aggregates"

    id = Column(Integer, primary_key=True, autoincrement=True)
    item_id = Column(Integer, nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False)
    interval = Column(String(8), nullable=False)  # "5m" or "1h"

    open_buy = Column(Integer, nullable=True)
    close_buy = Column(Integer, nullable=True)
    high_buy = Column(Integer, nullable=True)
    low_buy = Column(Integer, nullable=True)

    open_sell = Column(Integer, nullable=True)
    close_sell = Column(Integer, nullable=True)
    high_sell = Column(Integer, nullable=True)
    low_sell = Column(Integer, nullable=True)

    total_buy_volume = Column(Integer, default=0)
    total_sell_volume = Column(Integer, default=0)
    snapshot_count = Column(Integer, default=0)

    __table_args__ = (
        Index("ix_agg_item_ts_int", "item_id", "timestamp", "interval"),
    )


class Trade(Base):
    """Individual GE trades captured via DINK or manual entry."""
    __tablename__ = "trades"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow)
    player = Column(String(64), nullable=True)
    item_id = Column(Integer, nullable=False, index=True)
    item_name = Column(String(255), nullable=False)
    trade_type = Column(String(8), nullable=False)  # BUY or SELL
    status = Column(String(16), nullable=False)      # BOUGHT, SOLD, BUYING, SELLING, CANCELLED
    quantity = Column(Integer, nullable=False)
    price = Column(Integer, nullable=False)
    total_value = Column(Integer, nullable=False)
    slot = Column(Integer, nullable=True)
    market_price = Column(Integer, nullable=True)
    seller_tax = Column(Integer, nullable=True)
    market_high = Column(Integer, nullable=True)
    market_low = Column(Integer, nullable=True)
    source = Column(String(16), default="dink")  # dink, manual, csv_import


class FlipHistory(Base):
    """Matched buy/sell pairs representing completed flips."""
    __tablename__ = "flip_history"

    id = Column(Integer, primary_key=True, autoincrement=True)
    item_id = Column(Integer, nullable=False, index=True)
    item_name = Column(String(255), nullable=False)
    player = Column(String(64), nullable=True)

    buy_trade_id = Column(Integer, ForeignKey("trades.id"), nullable=True)
    sell_trade_id = Column(Integer, ForeignKey("trades.id"), nullable=True)

    buy_price = Column(Integer, nullable=False)
    sell_price = Column(Integer, nullable=False)
    quantity = Column(Integer, nullable=False)
    gross_profit = Column(Integer, nullable=False)
    tax = Column(Integer, nullable=False)
    net_profit = Column(Integer, nullable=False)
    margin_pct = Column(Float, nullable=False)

    buy_time = Column(DateTime, nullable=False)
    sell_time = Column(DateTime, nullable=False)
    duration_seconds = Column(Integer, nullable=False)

    __table_args__ = (
        Index("ix_flip_item_time", "item_id", "sell_time"),
    )


class Prediction(Base):
    """Model predictions logged for backtesting and accuracy tracking."""
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    item_id = Column(Integer, nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow)
    horizon = Column(String(8), nullable=False)  # 1m, 5m, 30m, 2h, 8h, 24h

    predicted_buy = Column(Integer, nullable=True)
    predicted_sell = Column(Integer, nullable=True)
    predicted_direction = Column(String(8), nullable=True)  # up, down, flat
    confidence = Column(Float, nullable=True)

    # Filled in later when we have the actual outcome
    actual_buy = Column(Integer, nullable=True)
    actual_sell = Column(Integer, nullable=True)
    actual_direction = Column(String(8), nullable=True)
    outcome_recorded = Column(Boolean, default=False)

    model_version = Column(String(32), nullable=True)

    __table_args__ = (
        Index("ix_pred_item_horizon_ts", "item_id", "horizon", "timestamp"),
    )


class ModelMetrics(Base):
    """Track model accuracy per horizon over time."""
    __tablename__ = "model_metrics"

    id = Column(Integer, primary_key=True, autoincrement=True)
    horizon = Column(String(8), nullable=False)
    model_version = Column(String(32), nullable=False)
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow)

    direction_accuracy = Column(Float, nullable=True)   # % correct direction
    price_mae = Column(Float, nullable=True)             # Mean Absolute Error
    price_mape = Column(Float, nullable=True)            # Mean Abs % Error
    profit_accuracy = Column(Float, nullable=True)       # % of profitable predictions that were actually profitable
    sample_count = Column(Integer, default=0)


class ItemFeature(Base):
    """Cached feature vectors for ML models, updated every 60 seconds."""
    __tablename__ = "item_features"

    id = Column(Integer, primary_key=True, autoincrement=True)
    item_id = Column(Integer, nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow)

    features = Column(JSON, nullable=False)  # Full feature dict


class Alert(Base):
    """Price alerts, dump alerts, opportunity alerts."""
    __tablename__ = "alerts"

    id = Column(Integer, primary_key=True, autoincrement=True)
    item_id = Column(Integer, nullable=False, index=True)
    item_name = Column(String(255), nullable=False)
    alert_type = Column(String(32), nullable=False)  # price_target, dump, opportunity, ml_signal
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow)
    message = Column(Text, nullable=True)
    data = Column(JSON, nullable=True)
    sent_discord = Column(Boolean, default=False)
    acknowledged = Column(Boolean, default=False)


class Setting(Base):
    """Key-value settings store (replaces user_config.json)."""
    __tablename__ = "settings"

    key = Column(String(128), primary_key=True)
    value = Column(JSON, nullable=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


# ---------------------------------------------------------------------------
# Database initialization
# ---------------------------------------------------------------------------

def init_db():
    """Create all tables if they don't exist."""
    Base.metadata.create_all(bind=engine)


def get_db() -> Session:
    """Get a database session. Caller must close it."""
    return SessionLocal()


# ---------------------------------------------------------------------------
# Helper queries
# ---------------------------------------------------------------------------

def get_latest_price(db: Session, item_id: int) -> Optional[PriceSnapshot]:
    """Get the most recent price snapshot for an item."""
    return (
        db.query(PriceSnapshot)
        .filter(PriceSnapshot.item_id == item_id)
        .order_by(PriceSnapshot.timestamp.desc())
        .first()
    )


def get_price_history(
    db: Session,
    item_id: int,
    hours: int = 24,
    limit: int = 10000,
) -> List[PriceSnapshot]:
    """Get price snapshots for an item over the last N hours."""
    cutoff = datetime.utcnow() - timedelta(hours=hours)
    return (
        db.query(PriceSnapshot)
        .filter(PriceSnapshot.item_id == item_id, PriceSnapshot.timestamp >= cutoff)
        .order_by(PriceSnapshot.timestamp.asc())
        .limit(limit)
        .all()
    )


def get_item_flips(
    db: Session,
    item_id: int,
    days: int = 30,
) -> List[FlipHistory]:
    """Get completed flips for an item."""
    cutoff = datetime.utcnow() - timedelta(days=days)
    return (
        db.query(FlipHistory)
        .filter(FlipHistory.item_id == item_id, FlipHistory.sell_time >= cutoff)
        .order_by(FlipHistory.sell_time.desc())
        .all()
    )


def get_recent_predictions(
    db: Session,
    item_id: int,
    horizon: str,
    hours: int = 24,
) -> List[Prediction]:
    """Get recent predictions for backtesting."""
    cutoff = datetime.utcnow() - timedelta(hours=hours)
    return (
        db.query(Prediction)
        .filter(
            Prediction.item_id == item_id,
            Prediction.horizon == horizon,
            Prediction.timestamp >= cutoff,
        )
        .order_by(Prediction.timestamp.desc())
        .all()
    )


def get_setting(db: Session, key: str, default: Any = None) -> Any:
    """Get a setting value."""
    row = db.query(Setting).filter(Setting.key == key).first()
    return row.value if row else default


def set_setting(db: Session, key: str, value: Any):
    """Set a setting value."""
    row = db.query(Setting).filter(Setting.key == key).first()
    if row:
        row.value = value
        row.updated_at = datetime.utcnow()
    else:
        row = Setting(key=key, value=value)
        db.add(row)
    db.commit()


def get_tracked_item_ids(db: Session) -> List[int]:
    """Get all item IDs that have recent price data (active items)."""
    cutoff = datetime.utcnow() - timedelta(hours=1)
    rows = (
        db.query(PriceSnapshot.item_id)
        .filter(PriceSnapshot.timestamp >= cutoff)
        .distinct()
        .all()
    )
    return [r[0] for r in rows]
