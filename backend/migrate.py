"""
Migrate existing CSV/JSON data into SQLite database.
Run once on first setup, safe to re-run (skips duplicates).
"""

import os
import sys
import json
import csv
from datetime import datetime

# Add parent dir to path so we can import from the project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.database import (
    init_db, get_db, Trade, Setting, Item,
    FlipHistory,
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def migrate_dink_trades():
    """Import dink_trades.csv into the trades table."""
    csv_path = os.path.join(BASE_DIR, "dink_trades.csv")
    if not os.path.exists(csv_path):
        print("No dink_trades.csv found, skipping.")
        return 0

    db = get_db()
    count = 0
    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    ts_str = row.get("timestamp", "")
                    try:
                        ts = datetime.fromisoformat(ts_str)
                    except (ValueError, TypeError):
                        ts = datetime.utcnow()

                    item_id = int(row.get("item_id", 0) or 0)
                    if item_id == 0:
                        continue

                    quantity = int(row.get("quantity", 0) or 0)
                    price = int(row.get("price", 0) or 0)

                    trade = Trade(
                        timestamp=ts,
                        player=row.get("player", ""),
                        item_id=item_id,
                        item_name=row.get("item_name", f"Item {item_id}"),
                        trade_type=row.get("type", "UNKNOWN"),
                        status=row.get("status", "UNKNOWN"),
                        quantity=quantity,
                        price=price,
                        total_value=int(row.get("total_value", quantity * price) or quantity * price),
                        slot=int(row.get("slot", 0) or 0),
                        market_price=int(row.get("market_price", 0) or 0),
                        seller_tax=int(row.get("seller_tax", 0) or 0),
                        market_high=int(row.get("market_high", 0) or 0),
                        market_low=int(row.get("market_low", 0) or 0),
                        source="csv_import",
                    )
                    db.add(trade)
                    count += 1
                except Exception as e:
                    print(f"  Skipping row: {e}")
                    continue

        db.commit()
        print(f"Imported {count} trades from dink_trades.csv")
    except Exception as e:
        db.rollback()
        print(f"Error importing dink trades: {e}")
    finally:
        db.close()
    return count


def migrate_flips_csv():
    """Import flips.csv (Flipping Copilot export) into flip_history."""
    csv_path = r"C:\Users\Mikeb\OneDrive\Desktop\flips.csv"
    if not os.path.exists(csv_path):
        print("No flips.csv found, skipping.")
        return 0

    db = get_db()
    count = 0
    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    item_name = row.get("item_name", row.get("name", "Unknown"))
                    buy_price = int(float(row.get("buy_price", row.get("avg_buy", 0)) or 0))
                    sell_price = int(float(row.get("sell_price", row.get("avg_sell", 0)) or 0))
                    quantity = int(float(row.get("quantity", row.get("qty", 1)) or 1))

                    if buy_price == 0 or sell_price == 0:
                        continue

                    gross = (sell_price - buy_price) * quantity
                    tax = int(min(sell_price * 0.02, 5_000_000) * quantity)
                    net = gross - tax
                    margin_pct = (net / (buy_price * quantity) * 100) if buy_price > 0 else 0

                    # Try to parse timestamps
                    buy_time_str = row.get("buy_time", row.get("bought_at", ""))
                    sell_time_str = row.get("sell_time", row.get("sold_at", ""))
                    try:
                        buy_time = datetime.fromisoformat(buy_time_str) if buy_time_str else datetime.utcnow()
                    except (ValueError, TypeError):
                        buy_time = datetime.utcnow()
                    try:
                        sell_time = datetime.fromisoformat(sell_time_str) if sell_time_str else datetime.utcnow()
                    except (ValueError, TypeError):
                        sell_time = datetime.utcnow()

                    duration = int((sell_time - buy_time).total_seconds()) if sell_time > buy_time else 0

                    flip = FlipHistory(
                        item_id=int(row.get("item_id", row.get("id", 0)) or 0),
                        item_name=item_name,
                        player=row.get("player", ""),
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
                    db.add(flip)
                    count += 1
                except Exception as e:
                    print(f"  Skipping flip row: {e}")
                    continue

        db.commit()
        print(f"Imported {count} flips from flips.csv")
    except Exception as e:
        db.rollback()
        print(f"Error importing flips: {e}")
    finally:
        db.close()
    return count


def migrate_user_config():
    """Import user_config.json settings into the settings table."""
    config_path = os.path.join(BASE_DIR, "user_config.json")
    if not os.path.exists(config_path):
        print("No user_config.json found, skipping.")
        return

    db = get_db()
    try:
        with open(config_path, "r") as f:
            config = json.load(f)

        from backend.database import set_setting

        for key, value in config.items():
            set_setting(db, key, value)

        print(f"Imported {len(config)} settings from user_config.json")
    except Exception as e:
        print(f"Error importing config: {e}")
    finally:
        db.close()


def main():
    print("=" * 60)
    print("FLIPPING AI - Database Migration")
    print("=" * 60)
    print()

    print("Initializing database...")
    init_db()
    print(f"Database created at: {os.path.join(BASE_DIR, 'flipping_ai.db')}")
    print()

    print("--- Migrating DINK trades ---")
    migrate_dink_trades()
    print()

    print("--- Migrating flips.csv ---")
    migrate_flips_csv()
    print()

    print("--- Migrating user config ---")
    migrate_user_config()
    print()

    print("=" * 60)
    print("Migration complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
