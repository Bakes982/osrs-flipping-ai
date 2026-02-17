"""Tests for the Alerts API endpoints."""

import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime

from backend.database import Base, engine, SessionLocal, Alert, Setting


@pytest.fixture(autouse=True)
def setup_db():
    """Create tables and clean up for each test."""
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        # Clean alerts and settings
        db.query(Alert).delete()
        db.query(Setting).filter(Setting.key == "price_alerts").delete()
        db.commit()
    finally:
        db.close()
    yield
    db = SessionLocal()
    try:
        db.query(Alert).delete()
        db.query(Setting).filter(Setting.key == "price_alerts").delete()
        db.commit()
    finally:
        db.close()


class TestAlertModel:
    """Test Alert DB operations."""

    def test_create_alert(self):
        db = SessionLocal()
        try:
            alert = Alert(
                item_id=13652,
                item_name="Dragon claws",
                alert_type="opportunity",
                message="High-score opportunity: Dragon claws (score 82)",
                data={"score": 82.0, "profit": 500000},
            )
            db.add(alert)
            db.commit()

            result = db.query(Alert).filter(Alert.item_id == 13652).first()
            assert result is not None
            assert result.alert_type == "opportunity"
            assert result.acknowledged is False
            assert result.data["score"] == 82.0
        finally:
            db.close()

    def test_acknowledge_alert(self):
        db = SessionLocal()
        try:
            alert = Alert(
                item_id=1, item_name="Test", alert_type="dump",
                message="Test dump alert",
            )
            db.add(alert)
            db.commit()
            alert_id = alert.id

            # Acknowledge
            db.query(Alert).filter(Alert.id == alert_id).update({"acknowledged": True})
            db.commit()

            result = db.query(Alert).filter(Alert.id == alert_id).first()
            assert result.acknowledged is True
        finally:
            db.close()

    def test_multiple_alert_types(self):
        db = SessionLocal()
        try:
            for atype in ["price_target", "dump", "opportunity", "ml_signal"]:
                db.add(Alert(
                    item_id=1, item_name="Test", alert_type=atype,
                    message=f"Test {atype}",
                ))
            db.commit()

            all_alerts = db.query(Alert).all()
            assert len(all_alerts) == 4
            types = {a.alert_type for a in all_alerts}
            assert types == {"price_target", "dump", "opportunity", "ml_signal"}
        finally:
            db.close()


class TestPriceTargets:
    """Test price target storage in settings."""

    def test_store_price_targets(self):
        from backend.database import get_setting, set_setting
        db = SessionLocal()
        try:
            targets = [
                {"item_id": 13652, "item_name": "Dragon claws", "target_price": 50_000_000, "direction": "below"},
                {"item_id": 11802, "item_name": "Armadyl godsword", "target_price": 20_000_000, "direction": "above"},
            ]
            set_setting(db, "price_alerts", targets)

            loaded = get_setting(db, "price_alerts", default=[])
            assert len(loaded) == 2
            assert loaded[0]["item_id"] == 13652
            assert loaded[1]["direction"] == "above"
        finally:
            db.close()

    def test_remove_triggered_target(self):
        from backend.database import get_setting, set_setting
        db = SessionLocal()
        try:
            targets = [
                {"item_id": 1, "target_price": 100, "direction": "below"},
                {"item_id": 2, "target_price": 200, "direction": "above"},
            ]
            set_setting(db, "price_alerts", targets)

            # Simulate removing triggered target
            remaining = [t for t in targets if t["item_id"] != 1]
            set_setting(db, "price_alerts", remaining)

            loaded = get_setting(db, "price_alerts", default=[])
            assert len(loaded) == 1
            assert loaded[0]["item_id"] == 2
        finally:
            db.close()

    def test_empty_targets_default(self):
        from backend.database import get_setting
        db = SessionLocal()
        try:
            loaded = get_setting(db, "price_alerts", default=[])
            assert loaded == []
        finally:
            db.close()
