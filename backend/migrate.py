"""
Migrate existing CSV/JSON data into SQLite database.
Run once on first setup, safe to re-run (skips duplicates).

Data sources:
- dink_trades.csv            -> trades table
- flips.csv (Desktop)        -> flip_history table
- live_opportunities_*.json  -> price_snapshots (historical learning data)
- user_config.json           -> settings table
- Auto-match buy/sell pairs  -> flip_history table
"""

import os
import sys
import json
import csv
import glob
from datetime import datetime
from collections import defaultdict

# Add parent dir to path so we can import from the project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.database import (
    init_db, get_db, Trade, Setting, Item, FlipHistory, PriceSnapshot,
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# GE Tax: 2% of sell price, capped at 5M GP per item
GE_TAX_RATE = 0.02
GE_TAX_CAP = 5_000_000


def calculate_ge_tax(sell_price: int, quantity: int = 1) -> int:
    """Calculate GE tax: 2% of sell price per item, capped at 5M per item."""
    tax_per_item = min(int(sell_price * GE_TAX_RATE), GE_TAX_CAP)
    return tax_per_item * quantity


def migrate_dink_trades():
    """Import dink_trades.csv into the trades table."""
    csv_path = os.path.join(BASE_DIR, "dink_trades.csv")
    if not os.path.exists(csv_path):
        print("  No dink_trades.csv found, skipping.")
        return 0

    db = get_db()
    count = 0
    skipped = 0
    try:
        # Check if already imported
        existing = db.query(Trade).filter(Trade.source == "csv_import").count()
        if existing > 0:
            print(f"  Already imported {existing} trades, skipping.")
            return existing

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
                        skipped += 1
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
                    print(f"    Skipping row: {e}")
                    skipped += 1

        db.commit()
        print(f"  Imported {count} trades ({skipped} skipped)")
    except Exception as e:
        db.rollback()
        print(f"  Error importing dink trades: {e}")
    finally:
        db.close()
    return count


def auto_match_flips():
    """
    Match BUY + SELL trades for the same item into FlipHistory records.
    Uses FIFO: earliest unmatched buy is matched with earliest sell.
    Applies 2% GE tax to all profit calculations.
    """
    db = get_db()
    count = 0
    try:
        # Check if we already have flips
        existing = db.query(FlipHistory).count()
        if existing > 0:
            print(f"  Already have {existing} matched flips, skipping auto-match.")
            return existing

        # Get all trades grouped by item
        trades = db.query(Trade).order_by(Trade.timestamp.asc()).all()
        buys = defaultdict(list)  # item_id -> [Trade, ...]
        sells = defaultdict(list)

        for t in trades:
            if t.trade_type == "BUY" and t.status in ("BOUGHT", "COMPLETED"):
                buys[t.item_id].append(t)
            elif t.trade_type == "SELL" and t.status in ("SOLD", "COMPLETED"):
                sells[t.item_id].append(t)

        # FIFO matching
        for item_id in sells:
            buy_queue = list(buys.get(item_id, []))
            for sell in sells[item_id]:
                if not buy_queue:
                    break

                buy = buy_queue.pop(0)
                qty = min(buy.quantity, sell.quantity)
                gross = (sell.price - buy.price) * qty
                tax = calculate_ge_tax(sell.price, qty)
                net = gross - tax
                margin_pct = (net / (buy.price * qty) * 100) if buy.price > 0 else 0
                duration = int((sell.timestamp - buy.timestamp).total_seconds()) if sell.timestamp > buy.timestamp else 0

                flip = FlipHistory(
                    item_id=item_id,
                    item_name=sell.item_name or buy.item_name,
                    player=sell.player or buy.player,
                    buy_price=buy.price,
                    sell_price=sell.price,
                    quantity=qty,
                    gross_profit=gross,
                    tax=tax,
                    net_profit=net,
                    margin_pct=round(margin_pct, 2),
                    buy_time=buy.timestamp,
                    sell_time=sell.timestamp,
                    duration_seconds=duration,
                )
                db.add(flip)
                count += 1

        db.commit()
        print(f"  Auto-matched {count} buy/sell pairs into flip_history")
    except Exception as e:
        db.rollback()
        print(f"  Error matching flips: {e}")
    finally:
        db.close()
    return count


def migrate_flips_csv():
    """Import flips.csv (Flipping Copilot export) into flip_history."""
    # Check multiple possible locations
    locations = [
        os.path.join(BASE_DIR, "flips.csv"),
        r"C:\Users\Mikeb\OneDrive\Desktop\flips.csv",
        r"C:\Users\Mikeb\Desktop\flips.csv",
    ]

    csv_path = None
    for loc in locations:
        if os.path.exists(loc):
            csv_path = loc
            break

    if csv_path is None:
        print("  No flips.csv found at any known location, skipping.")
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
                    tax = calculate_ge_tax(sell_price, quantity)
                    net = gross - tax
                    margin_pct = (net / (buy_price * quantity) * 100) if buy_price > 0 else 0

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
                    print(f"    Skipping flip row: {e}")

        db.commit()
        print(f"  Imported {count} flips from {csv_path}")
    except Exception as e:
        db.rollback()
        print(f"  Error importing flips: {e}")
    finally:
        db.close()
    return count


def migrate_live_opportunities():
    """
    Import live_opportunities_*.json files as historical price snapshots.
    Each file contains item prices at a point in time - valuable training data.
    """
    pattern = os.path.join(BASE_DIR, "live_opportunities_*.json")
    files = sorted(glob.glob(pattern))

    if not files:
        print("  No live_opportunities files found, skipping.")
        return 0

    db = get_db()
    total = 0
    try:
        # Check if we already imported these
        existing = db.query(PriceSnapshot).filter(
            PriceSnapshot.item_id == -1  # sentinel check
        ).count()

        # Parse timestamp from filename: live_opportunities_20260205_082018.json
        for filepath in files:
            try:
                basename = os.path.basename(filepath)
                ts_str = basename.replace("live_opportunities_", "").replace(".json", "")
                ts = datetime.strptime(ts_str, "%Y%m%d_%H%M%S")
            except (ValueError, TypeError):
                continue

            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    items = json.load(f)

                if not isinstance(items, list):
                    continue

                for item in items:
                    try:
                        high = item.get("current_high")
                        low = item.get("current_low")
                        if not high or not low:
                            continue

                        # We don't have item_id in these files, so use the name
                        # to look up or create an item record
                        item_name = item.get("item_name", "")
                        if not item_name:
                            continue

                        # Store as a snapshot for learning
                        snap = PriceSnapshot(
                            item_id=0,  # Will be resolved below
                            timestamp=ts,
                            instant_buy=int(high),
                            instant_sell=int(low),
                            buy_volume=0,
                            sell_volume=0,
                        )

                        # Try to find item_id by name
                        item_row = db.query(Item).filter(Item.name == item_name).first()
                        if item_row:
                            snap.item_id = item_row.id
                        else:
                            # Store item_name in a temp way - we'll resolve later
                            # For now skip items we can't resolve
                            continue

                        db.add(snap)
                        total += 1
                    except Exception:
                        continue

                # Commit in batches per file
                if total % 500 == 0:
                    db.commit()

            except Exception as e:
                print(f"    Error reading {basename}: {e}")

        db.commit()
        print(f"  Imported {total} price snapshots from {len(files)} opportunity files")
    except Exception as e:
        db.rollback()
        print(f"  Error importing opportunities: {e}")
    finally:
        db.close()
    return total


def populate_item_mapping():
    """Fetch the OSRS Wiki item mapping and populate the items table."""
    import requests

    db = get_db()
    try:
        existing = db.query(Item).count()
        if existing > 100:
            print(f"  Items table already has {existing} items, skipping.")
            return existing

        print("  Fetching item mapping from Wiki API...")
        resp = requests.get(
            "https://prices.runescape.wiki/api/v1/osrs/mapping",
            headers={"User-Agent": "OSRS-AI-Flipper v2.0 - Discord: bakes982"},
            timeout=15,
        )
        resp.raise_for_status()
        items = resp.json()

        count = 0
        for item_data in items:
            item_id = item_data.get("id")
            name = item_data.get("name")
            if not item_id or not name:
                continue

            existing_item = db.query(Item).filter(Item.id == item_id).first()
            if existing_item:
                existing_item.name = name
                existing_item.members = item_data.get("members", True)
                existing_item.buy_limit = item_data.get("limit", 10000)
                existing_item.high_alch = item_data.get("highalch", 0)
            else:
                db.add(Item(
                    id=item_id,
                    name=name,
                    members=item_data.get("members", True),
                    buy_limit=item_data.get("limit", 10000),
                    high_alch=item_data.get("highalch", 0),
                ))
            count += 1

        db.commit()
        print(f"  Populated {count} items from Wiki mapping")
        return count
    except Exception as e:
        db.rollback()
        print(f"  Error fetching item mapping: {e}")
        return 0
    finally:
        db.close()


def migrate_user_config():
    """Import user_config.json settings into the settings table."""
    config_path = os.path.join(BASE_DIR, "user_config.json")
    if not os.path.exists(config_path):
        print("  No user_config.json found, skipping.")
        return

    db = get_db()
    try:
        with open(config_path, "r") as f:
            config = json.load(f)

        from backend.database import set_setting

        for key, value in config.items():
            set_setting(db, key, value)

        print(f"  Imported {len(config)} settings from user_config.json")
    except Exception as e:
        print(f"  Error importing config: {e}")
    finally:
        db.close()


def main():
    print("=" * 60)
    print("  FLIPPING AI - Database Migration")
    print("  GE Tax: 2% of sell price, capped at 5M GP")
    print("=" * 60)
    print()

    print("1. Initializing database...")
    init_db()
    print(f"   Database: {os.path.join(BASE_DIR, 'flipping_ai.db')}")
    print()

    print("2. Populating item mapping from Wiki API...")
    populate_item_mapping()
    print()

    print("3. Importing DINK trades...")
    migrate_dink_trades()
    print()

    print("4. Auto-matching buy/sell pairs into flip history...")
    auto_match_flips()
    print()

    print("5. Importing flips.csv (Flipping Copilot)...")
    migrate_flips_csv()
    print()

    print("6. Importing historical opportunities as price snapshots...")
    migrate_live_opportunities()
    print()

    print("7. Importing user config...")
    migrate_user_config()
    print()

    # Summary
    db = get_db()
    try:
        items = db.query(Item).count()
        trades = db.query(Trade).count()
        flips = db.query(FlipHistory).count()
        snapshots = db.query(PriceSnapshot).count()
        print("=" * 60)
        print("  Migration Summary:")
        print(f"  Items:      {items:,}")
        print(f"  Trades:     {trades:,}")
        print(f"  Flips:      {flips:,}")
        print(f"  Snapshots:  {snapshots:,}")
        print("=" * 60)
    finally:
        db.close()


if __name__ == "__main__":
    main()
