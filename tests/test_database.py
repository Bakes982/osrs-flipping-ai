"""Tests for the database layer."""

import os
import tempfile
from datetime import datetime, timedelta

import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from backend.database import (
    Base,
    Item,
    PriceSnapshot,
    FlipHistory,
    Prediction,
    Setting,
    ItemFeature,
    Alert,
    Trade,
)


@pytest.fixture
def db_session():
    """Create an in-memory SQLite database for testing."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )

    @event.listens_for(engine, "connect")
    def _set_pragma(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.close()

    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine, autoflush=False)
    session = Session()
    yield session
    session.close()


def _make_snapshot(item_id, ts, buy=1000, sell=900, buy_vol=10, sell_vol=10):
    return PriceSnapshot(
        item_id=item_id,
        timestamp=ts,
        instant_buy=buy,
        instant_sell=sell,
        buy_volume=buy_vol,
        sell_volume=sell_vol,
    )


class TestDatabaseModels:
    def test_create_item(self, db_session):
        item = Item(id=4151, name="Abyssal whip", members=True, buy_limit=70)
        db_session.add(item)
        db_session.commit()

        fetched = db_session.query(Item).filter(Item.id == 4151).first()
        assert fetched is not None
        assert fetched.name == "Abyssal whip"
        assert fetched.buy_limit == 70

    def test_create_price_snapshot(self, db_session):
        now = datetime.utcnow()
        snap = _make_snapshot(4151, now, buy=2500000, sell=2480000)
        db_session.add(snap)
        db_session.commit()

        fetched = db_session.query(PriceSnapshot).first()
        assert fetched.item_id == 4151
        assert fetched.instant_buy == 2500000
        assert fetched.instant_sell == 2480000

    def test_create_trade(self, db_session):
        trade = Trade(
            timestamp=datetime.utcnow(),
            item_id=4151,
            item_name="Abyssal whip",
            trade_type="BUY",
            status="BOUGHT",
            quantity=1,
            price=2500000,
            total_value=2500000,
        )
        db_session.add(trade)
        db_session.commit()

        fetched = db_session.query(Trade).first()
        assert fetched.trade_type == "BUY"
        assert fetched.total_value == 2500000

    def test_create_flip_history(self, db_session):
        now = datetime.utcnow()
        flip = FlipHistory(
            item_id=4151,
            item_name="Abyssal whip",
            buy_price=2480000,
            sell_price=2520000,
            quantity=1,
            gross_profit=40000,
            tax=50400,
            net_profit=-10400,
            margin_pct=1.61,
            buy_time=now - timedelta(minutes=10),
            sell_time=now,
            duration_seconds=600,
        )
        db_session.add(flip)
        db_session.commit()

        fetched = db_session.query(FlipHistory).first()
        assert fetched.item_id == 4151
        assert fetched.duration_seconds == 600

    def test_create_prediction(self, db_session):
        pred = Prediction(
            item_id=4151,
            timestamp=datetime.utcnow(),
            horizon="5m",
            predicted_buy=2510000,
            predicted_sell=2490000,
            predicted_direction="up",
            confidence=0.75,
            model_version="ml_v1",
        )
        db_session.add(pred)
        db_session.commit()

        fetched = db_session.query(Prediction).first()
        assert fetched.horizon == "5m"
        assert fetched.confidence == 0.75
        assert fetched.outcome_recorded is False

    def test_settings_crud(self, db_session):
        setting = Setting(key="min_profit", value=100000)
        db_session.add(setting)
        db_session.commit()

        fetched = db_session.query(Setting).filter(Setting.key == "min_profit").first()
        assert fetched.value == 100000

        # Update
        fetched.value = 200000
        db_session.commit()
        refetched = db_session.query(Setting).filter(Setting.key == "min_profit").first()
        assert refetched.value == 200000

    def test_item_feature(self, db_session):
        feat = ItemFeature(
            item_id=4151,
            timestamp=datetime.utcnow(),
            features={"rsi_14": 55.0, "spread_pct": 1.5},
        )
        db_session.add(feat)
        db_session.commit()

        fetched = db_session.query(ItemFeature).first()
        assert fetched.features["rsi_14"] == 55.0

    def test_price_history_query(self, db_session):
        now = datetime.utcnow()
        for i in range(10):
            snap = _make_snapshot(
                4151, now - timedelta(minutes=i), buy=1000 + i, sell=900 + i
            )
            db_session.add(snap)
        db_session.commit()

        cutoff = now - timedelta(hours=1)
        results = (
            db_session.query(PriceSnapshot)
            .filter(
                PriceSnapshot.item_id == 4151,
                PriceSnapshot.timestamp >= cutoff,
            )
            .order_by(PriceSnapshot.timestamp.asc())
            .all()
        )
        assert len(results) == 10


class TestAlertModel:
    def test_create_alert(self, db_session):
        alert = Alert(
            item_id=4151,
            item_name="Abyssal whip",
            alert_type="price_target",
            timestamp=datetime.utcnow(),
            message="Price dropped below 2M",
            data={"target": 2000000, "current": 1950000},
        )
        db_session.add(alert)
        db_session.commit()

        fetched = db_session.query(Alert).first()
        assert fetched.alert_type == "price_target"
        assert fetched.sent_discord is False
