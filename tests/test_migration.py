"""Tests for the database migration (CSV/JSON -> SQLite)."""

import csv
import json
import os
import tempfile
from datetime import datetime
from unittest.mock import patch

import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from backend.database import Base, Trade, FlipHistory, Item, PriceSnapshot, Setting
from backend.migrate import safe_int, calculate_ge_tax


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def db_session():
    """In-memory SQLite DB for migration tests."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )

    @event.listens_for(engine, "connect")
    def _pragma(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.close()

    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine, autoflush=False)
    session = Session()
    yield session
    session.close()


@pytest.fixture
def sample_dink_csv(tmp_path):
    """Create a sample dink_trades.csv with realistic data.

    Simulates what pandas writes when market_high/market_low contain None
    (pandas casts the column to float64, so ints become '2500000.0').
    """
    csv_path = tmp_path / "dink_trades.csv"
    rows = [
        {
            "timestamp": "2026-02-10T14:30:00",
            "player": "TestPlayer",
            "item_name": "Abyssal whip",
            "item_id": "4151",
            "type": "BUY",
            "status": "BOUGHT",
            "quantity": "10",
            "price": "2500000",
            "total_value": "25000000",
            "slot": "0",
            "market_price": "2510000",
            "seller_tax": "0",
            "market_high": "2520000",
            "market_low": "2490000",
        },
        {
            # Simulates pandas float64 columns (market_high/low had None values)
            "timestamp": "2026-02-10T14:35:00",
            "player": "TestPlayer",
            "item_name": "Bandos chestplate",
            "item_id": "11832.0",  # pandas float cast
            "type": "SELL",
            "status": "SOLD",
            "quantity": "1.0",     # pandas float cast
            "price": "15800000.0", # pandas float cast
            "total_value": "15800000.0",
            "slot": "1.0",
            "market_price": "15750000.0",
            "seller_tax": "316000.0",
            "market_high": "15900000.0",  # float from pandas
            "market_low": "",             # None -> empty string from pandas
        },
        {
            # Row with missing/empty fields
            "timestamp": "",
            "player": "",
            "item_name": "Dragon bones",
            "item_id": "536",
            "type": "BUY",
            "status": "BOUGHT",
            "quantity": "100",
            "price": "2200",
            "total_value": "220000",
            "slot": "",
            "market_price": "",
            "seller_tax": "",
            "market_high": "",
            "market_low": "",
        },
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    return str(csv_path)


@pytest.fixture
def sample_flips_csv(tmp_path):
    """Create a sample flips.csv (Flipping Copilot format)."""
    csv_path = tmp_path / "flips.csv"
    rows = [
        {
            "item_id": "4151",
            "item_name": "Abyssal whip",
            "buy_price": "2480000",
            "sell_price": "2520000",
            "quantity": "5",
            "buy_time": "2026-02-09T10:00:00",
            "sell_time": "2026-02-09T10:15:00",
        },
        {
            # Float-formatted values
            "item_id": "11832.0",
            "item_name": "Bandos chestplate",
            "buy_price": "15500000.0",
            "sell_price": "15800000.0",
            "quantity": "1.0",
            "buy_time": "2026-02-09T12:00:00",
            "sell_time": "2026-02-09T12:30:00",
        },
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    return str(csv_path)


# ---------------------------------------------------------------------------
# safe_int tests
# ---------------------------------------------------------------------------

class TestSafeInt:
    def test_normal_int_string(self):
        assert safe_int("2500000") == 2500000

    def test_float_string(self):
        """The core bug: pandas writes ints as '2500000.0' when column has None."""
        assert safe_int("2500000.0") == 2500000

    def test_none(self):
        assert safe_int(None) == 0

    def test_empty_string(self):
        assert safe_int("") == 0

    def test_actual_int(self):
        assert safe_int(42) == 42

    def test_actual_float(self):
        assert safe_int(42.7) == 42

    def test_garbage(self):
        assert safe_int("not_a_number") == 0

    def test_custom_default(self):
        assert safe_int("", default=1) == 1
        assert safe_int(None, default=99) == 99

    def test_negative(self):
        assert safe_int("-5000") == -5000
        assert safe_int("-5000.0") == -5000


class TestCalculateGETax:
    def test_normal_tax(self):
        # 2% of 1M = 20K per item, 5 items = 100K
        assert calculate_ge_tax(1_000_000, 5) == 100_000

    def test_tax_cap(self):
        # 2% of 300M = 6M, but cap is 5M per item
        assert calculate_ge_tax(300_000_000, 1) == 5_000_000

    def test_multiple_items_at_cap(self):
        # Each item capped at 5M tax, 3 items = 15M
        assert calculate_ge_tax(300_000_000, 3) == 15_000_000


# ---------------------------------------------------------------------------
# CSV migration tests
# ---------------------------------------------------------------------------

class TestDinkTradesMigration:
    def test_parses_normal_values(self, db_session, sample_dink_csv):
        """Verify normal integer CSV values parse correctly."""
        with open(sample_dink_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        row = rows[0]  # Normal integer row
        item_id = safe_int(row.get("item_id"))
        assert item_id == 4151
        assert safe_int(row.get("quantity")) == 10
        assert safe_int(row.get("price")) == 2500000
        assert safe_int(row.get("market_high")) == 2520000
        assert safe_int(row.get("market_low")) == 2490000

    def test_parses_float_formatted_values(self, db_session, sample_dink_csv):
        """Verify pandas float64 CSV values parse correctly (the main bug)."""
        with open(sample_dink_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        row = rows[1]  # Float-formatted row
        assert safe_int(row.get("item_id")) == 11832
        assert safe_int(row.get("quantity")) == 1
        assert safe_int(row.get("price")) == 15800000
        assert safe_int(row.get("market_high")) == 15900000
        assert safe_int(row.get("market_low")) == 0  # Was empty string (None)

    def test_parses_empty_values(self, db_session, sample_dink_csv):
        """Verify empty/missing CSV fields handled gracefully."""
        with open(sample_dink_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        row = rows[2]  # Row with empty fields
        assert safe_int(row.get("item_id")) == 536
        assert safe_int(row.get("slot")) == 0
        assert safe_int(row.get("market_price")) == 0
        assert safe_int(row.get("market_high")) == 0

    def test_full_migration_creates_trades(self, db_session, sample_dink_csv):
        """End-to-end: parse CSV and create Trade objects."""
        with open(sample_dink_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            count = 0
            for row in reader:
                item_id = safe_int(row.get("item_id"))
                if item_id == 0:
                    continue

                quantity = safe_int(row.get("quantity"))
                price = safe_int(row.get("price"))

                trade = Trade(
                    timestamp=datetime.utcnow(),
                    player=row.get("player", ""),
                    item_id=item_id,
                    item_name=row.get("item_name", f"Item {item_id}"),
                    trade_type=row.get("type", "UNKNOWN"),
                    status=row.get("status", "UNKNOWN"),
                    quantity=quantity,
                    price=price,
                    total_value=safe_int(row.get("total_value"), quantity * price),
                    slot=safe_int(row.get("slot")),
                    market_price=safe_int(row.get("market_price")),
                    seller_tax=safe_int(row.get("seller_tax")),
                    market_high=safe_int(row.get("market_high")),
                    market_low=safe_int(row.get("market_low")),
                    source="csv_import",
                )
                db_session.add(trade)
                count += 1

            db_session.commit()

        assert count == 3  # All 3 rows should parse successfully

        trades = db_session.query(Trade).all()
        assert len(trades) == 3

        # Verify the float-formatted row parsed correctly
        bandos = db_session.query(Trade).filter(Trade.item_id == 11832).first()
        assert bandos is not None
        assert bandos.price == 15800000
        assert bandos.seller_tax == 316000
        assert bandos.market_high == 15900000
        assert bandos.market_low == 0


class TestFlipsCsvMigration:
    def test_parses_flips_with_float_values(self, sample_flips_csv):
        """Verify flips.csv parsing handles float-formatted values."""
        with open(sample_flips_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        # Float-formatted row
        row = rows[1]
        assert safe_int(row.get("item_id")) == 11832
        assert safe_int(row.get("buy_price")) == 15500000
        assert safe_int(row.get("sell_price")) == 15800000
        assert safe_int(row.get("quantity"), default=1) == 1

    def test_full_flip_migration(self, db_session, sample_flips_csv):
        """End-to-end: parse flips.csv and create FlipHistory objects."""
        with open(sample_flips_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            count = 0
            for row in reader:
                buy_price = safe_int(row.get("buy_price"))
                sell_price = safe_int(row.get("sell_price"))
                quantity = safe_int(row.get("quantity"), default=1)

                if buy_price == 0 or sell_price == 0:
                    continue

                gross = (sell_price - buy_price) * quantity
                tax = calculate_ge_tax(sell_price, quantity)
                net = gross - tax
                margin_pct = (net / (buy_price * quantity) * 100) if buy_price > 0 else 0

                buy_time_str = row.get("buy_time", "")
                sell_time_str = row.get("sell_time", "")
                buy_time = datetime.fromisoformat(buy_time_str) if buy_time_str else datetime.utcnow()
                sell_time = datetime.fromisoformat(sell_time_str) if sell_time_str else datetime.utcnow()
                duration = int((sell_time - buy_time).total_seconds()) if sell_time > buy_time else 0

                flip = FlipHistory(
                    item_id=safe_int(row.get("item_id")),
                    item_name=row.get("item_name", "Unknown"),
                    buy_price=buy_price,
                    sell_price=sell_price,
                    quantity=quantity,
                    gross_profit=gross,
                    tax=tax,
                    net_profit=net,
                    margin_pct=round(margin_pct, 2),
                    buy_time=buy_time,
                    sell_time=sell_time,
                    duration_seconds=duration,
                )
                db_session.add(flip)
                count += 1

            db_session.commit()

        assert count == 2

        flips = db_session.query(FlipHistory).all()
        assert len(flips) == 2

        # Verify Bandos flip (float-formatted input)
        bandos = db_session.query(FlipHistory).filter(FlipHistory.item_id == 11832).first()
        assert bandos is not None
        assert bandos.buy_price == 15500000
        assert bandos.sell_price == 15800000
        assert bandos.quantity == 1
        assert bandos.gross_profit == 300000  # 15.8M - 15.5M
        assert bandos.tax == calculate_ge_tax(15800000, 1)  # 2% of 15.8M = 316K
        assert bandos.net_profit == 300000 - 316000  # -16K (loss after tax)
        assert bandos.duration_seconds == 1800  # 30 minutes


class TestAutoMatchFlips:
    def test_fifo_matching(self, db_session):
        """Verify FIFO buy/sell matching produces correct FlipHistory."""
        # Create BUY then SELL trades
        buy = Trade(
            timestamp=datetime(2026, 2, 10, 10, 0),
            player="Test",
            item_id=4151,
            item_name="Abyssal whip",
            trade_type="BUY",
            status="BOUGHT",
            quantity=1,
            price=2480000,
            total_value=2480000,
        )
        sell = Trade(
            timestamp=datetime(2026, 2, 10, 10, 15),
            player="Test",
            item_id=4151,
            item_name="Abyssal whip",
            trade_type="SELL",
            status="SOLD",
            quantity=1,
            price=2530000,
            total_value=2530000,
        )
        db_session.add_all([buy, sell])
        db_session.commit()

        # Simulate auto_match_flips logic
        from collections import defaultdict
        trades = db_session.query(Trade).order_by(Trade.timestamp.asc()).all()
        buys = defaultdict(list)
        sells = defaultdict(list)
        for t in trades:
            if t.trade_type == "BUY" and t.status in ("BOUGHT", "COMPLETED"):
                buys[t.item_id].append(t)
            elif t.trade_type == "SELL" and t.status in ("SOLD", "COMPLETED"):
                sells[t.item_id].append(t)

        count = 0
        for item_id in sells:
            buy_queue = list(buys.get(item_id, []))
            for sell_t in sells[item_id]:
                if not buy_queue:
                    break
                buy_t = buy_queue.pop(0)
                qty = min(buy_t.quantity, sell_t.quantity)
                gross = (sell_t.price - buy_t.price) * qty
                tax = calculate_ge_tax(sell_t.price, qty)
                net = gross - tax

                flip = FlipHistory(
                    item_id=item_id,
                    item_name=sell_t.item_name,
                    buy_price=buy_t.price,
                    sell_price=sell_t.price,
                    quantity=qty,
                    gross_profit=gross,
                    tax=tax,
                    net_profit=net,
                    margin_pct=round(net / (buy_t.price * qty) * 100, 2),
                    buy_time=buy_t.timestamp,
                    sell_time=sell_t.timestamp,
                    duration_seconds=int((sell_t.timestamp - buy_t.timestamp).total_seconds()),
                )
                db_session.add(flip)
                count += 1

        db_session.commit()

        assert count == 1
        flip = db_session.query(FlipHistory).first()
        assert flip.item_id == 4151
        assert flip.buy_price == 2480000
        assert flip.sell_price == 2530000
        assert flip.gross_profit == 50000
        assert flip.tax == calculate_ge_tax(2530000, 1)  # 50600
        assert flip.net_profit == 50000 - 50600  # -600 (tiny loss after tax)
        assert flip.duration_seconds == 900  # 15 minutes
