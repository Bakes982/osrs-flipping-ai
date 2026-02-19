"""
Portfolio & Trade Endpoints for OSRS Flipping AI
GET  /api/portfolio         - current holdings
GET  /api/trades            - trade history
GET  /api/performance       - performance metrics
POST /api/trades/import     - import trades from CSV
POST /api/dink              - DINK webhook receiver
"""

import asyncio
import csv
import io
import sys
import os
import json
import logging
import time
from datetime import datetime, timezone
from typing import Optional
from collections import defaultdict

import httpx
from fastapi import APIRouter, Request, HTTPException, Query, UploadFile, File

# Ensure project root is on sys.path
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from backend.database import (
    get_db,
    Trade,
    FlipHistory,
    find_trades,
    find_all_flips,
    find_unmatched_buy_trades,
    find_active_positions,
    dismiss_position,
    dismiss_positions_by_source,
    insert_trade,
    insert_flip,
    get_matched_buy_trade_ids,
)
from backend.websocket import manager

logger = logging.getLogger(__name__)

router = APIRouter(tags=["portfolio"])

PORTFOLIO_FILE = os.path.join(_PROJECT_ROOT, "portfolio.json")
WIKI_LATEST_URL = "https://prices.runescape.wiki/api/v1/osrs/latest"
USER_AGENT = "OSRS-AI-Flipper v2.0 - Discord: bakes982"


# ---------------------------------------------------------------------------
# Wiki mapping cache (item name → item_id)
# ---------------------------------------------------------------------------

_wiki_name_to_id: dict = {}
_wiki_id_to_name: dict = {}


def _ensure_wiki_mapping():
    """Load Wiki item mapping into cache if not already loaded."""
    if _wiki_name_to_id:
        return
    import requests
    try:
        resp = requests.get(
            "https://prices.runescape.wiki/api/v1/osrs/mapping",
            headers={"User-Agent": USER_AGENT}, timeout=15,
        )
        resp.raise_for_status()
        for item in resp.json():
            name = item.get("name", "").strip()
            item_id = item.get("id")
            if name and item_id is not None:
                _wiki_name_to_id[name.lower()] = item_id
                _wiki_id_to_name[item_id] = name
        logger.info("Wiki mapping loaded: %d items", len(_wiki_name_to_id))
    except Exception as e:
        logger.error("Failed to load Wiki mapping: %s", e)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_portfolio_file() -> dict:
    """Load portfolio.json if it exists, otherwise return empty structure."""
    if os.path.exists(PORTFOLIO_FILE):
        try:
            with open(PORTFOLIO_FILE, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {"holdings": [], "cash": 0}


async def _fetch_live_price(item_id: int) -> dict:
    """Fetch current GE price for a single item."""
    try:
        async with httpx.AsyncClient(headers={"User-Agent": USER_AGENT}, timeout=10) as client:
            resp = await client.get(WIKI_LATEST_URL)
            data = resp.json().get("data", {})
            item = data.get(str(item_id), {})
            return {"high": item.get("high"), "low": item.get("low")}
    except Exception:
        return {"high": None, "low": None}


# ---------------------------------------------------------------------------
# GET /api/portfolio
# ---------------------------------------------------------------------------

@router.get("/api/portfolio")
async def get_portfolio():
    """Return current holdings from portfolio.json or the database."""
    def _sync():
        portfolio = _load_portfolio_file()

        # Fallback: derive holdings from unmatched BUY trades in DB
        if not portfolio.get("holdings"):
            db = get_db()
            try:
                matched_buy_ids = get_matched_buy_trade_ids(db)
                all_trades = find_trades(db, limit=500)
                buys = [t for t in all_trades if t.trade_type == "BUY" and t.status == "BOUGHT"]
                holdings = []
                for t in buys:
                    if t.id not in matched_buy_ids:
                        holdings.append({
                            "item_id": t.item_id,
                            "item_name": t.item_name,
                            "quantity": t.quantity,
                            "buy_price": t.price,
                            "total_cost": t.total_value,
                            "player": t.player,
                            "bought_at": t.timestamp.isoformat() if t.timestamp else None,
                        })
                portfolio["holdings"] = holdings
            finally:
                db.close()

        return portfolio

    return await asyncio.to_thread(_sync)


# ---------------------------------------------------------------------------
# GET /api/trades
# ---------------------------------------------------------------------------

@router.get("/api/trades")
async def get_trades(
    limit: int = Query(100, ge=1, le=1000),
    item_id: Optional[int] = Query(None),
):
    """Return trade history from the trades table."""
    def _sync():
        db = get_db()
        try:
            rows = find_trades(db, item_id=item_id, limit=limit)
            return [
                {
                    "id": t.id,
                    "timestamp": t.timestamp.isoformat() if t.timestamp else None,
                    "player": t.player,
                    "item_id": t.item_id,
                    "item_name": t.item_name,
                    "trade_type": t.trade_type,
                    "status": t.status,
                    "quantity": t.quantity,
                    "price": t.price,
                    "total_value": t.total_value,
                    "slot": t.slot,
                    "market_price": t.market_price,
                    "seller_tax": t.seller_tax,
                    "source": t.source,
                }
                for t in rows
            ]
        finally:
            db.close()

    return await asyncio.to_thread(_sync)


# ---------------------------------------------------------------------------
# GET /api/performance
# ---------------------------------------------------------------------------

@router.get("/api/performance")
async def get_performance():
    """Return overall performance metrics computed from flip_history."""
    def _sync():
        db = get_db()
        try:
            flips = find_all_flips(db)
            if not flips:
                return {
                    "total_flips": 0,
                    "total_profit": 0,
                    "total_tax_paid": 0,
                    "win_rate": 0.0,
                    "avg_profit_per_flip": 0,
                    "gp_per_hour": 0,
                    "best_flip": None,
                    "worst_flip": None,
                }

            total_profit = sum(f.net_profit for f in flips)
            total_tax = sum(f.tax for f in flips)
            wins = sum(1 for f in flips if f.net_profit > 0)
            win_rate = wins / len(flips) * 100 if flips else 0

            # GP/hour calculation
            total_seconds = sum(f.duration_seconds for f in flips if f.duration_seconds > 0)
            gp_per_hour = int(total_profit / (total_seconds / 3600)) if total_seconds > 0 else 0

            best = max(flips, key=lambda f: f.net_profit)
            worst = min(flips, key=lambda f: f.net_profit)

            def _flip_summary(f):
                return {
                    "item_id": f.item_id,
                    "item_name": f.item_name,
                    "net_profit": f.net_profit,
                    "margin_pct": round(f.margin_pct, 2),
                    "quantity": f.quantity,
                    "sell_time": f.sell_time.isoformat() if f.sell_time else None,
                }

            return {
                "total_flips": len(flips),
                "total_profit": total_profit,
                "total_tax_paid": total_tax,
                "win_rate": round(win_rate, 1),
                "avg_profit_per_flip": int(total_profit / len(flips)),
                "gp_per_hour": gp_per_hour,
                "best_flip": _flip_summary(best),
                "worst_flip": _flip_summary(worst),
                # Per-item breakdown for Performance page
                "item_performance": _item_performance(flips),
                # Profit history for chart
                "profit_history": _profit_history(flips),
            }
        finally:
            db.close()

    return await asyncio.to_thread(_sync)


def _item_performance(flips):
    """Build per-item performance breakdown from flip history."""
    items = defaultdict(lambda: {"flips": [], "name": ""})
    for f in flips:
        items[f.item_id]["flips"].append(f)
        items[f.item_id]["name"] = f.item_name

    result = []
    for item_id, data in items.items():
        flip_list = data["flips"]
        total = sum(f.net_profit for f in flip_list)
        wins = sum(1 for f in flip_list if f.net_profit > 0)
        avg_dur = sum(f.duration_seconds for f in flip_list) / len(flip_list) if flip_list else 0
        result.append({
            "item_id": item_id,
            "item_name": data["name"],
            "flip_count": len(flip_list),
            "total_profit": total,
            "avg_profit": int(total / len(flip_list)),
            "win_rate": round(wins / len(flip_list) * 100, 1) if flip_list else 0,
            "avg_duration_min": round(avg_dur / 60, 1),
        })
    result.sort(key=lambda x: x["total_profit"], reverse=True)
    return result


def _profit_history(flips):
    """Build cumulative profit time-series from flip history (oldest→newest)."""
    sorted_flips = sorted(flips, key=lambda f: f.sell_time or datetime.min)
    cumulative = 0
    history = []
    for f in sorted_flips:
        cumulative += f.net_profit
        history.append({
            "time": f.sell_time.isoformat() if f.sell_time else None,
            "profit": cumulative,
            "item": f.item_name,
            "flip_profit": f.net_profit,
        })
    return history


# ---------------------------------------------------------------------------
# POST /api/trades/import  -  CSV Import
# ---------------------------------------------------------------------------

@router.post("/api/trades/import")
async def import_trades_csv(file: UploadFile = File(...)):
    """Import trade history from a CSV file (e.g. from RuneLite flipping plugins).

    Expected CSV columns:
      First buy time, Last sell time, Account, Item, Status,
      Bought, Sold, Avg. buy price, Avg. sell price, Tax, Profit, Profit ea.

    Status values: BUYING, SELLING, FINISHED
    - FINISHED rows with both buy and sell data create FlipHistory records
    - BUYING rows create active BUY trade records (portfolio holdings)
    - SELLING rows create active SELL trade records
    """
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="File must be a .csv")

    content = await file.read()
    text = content.decode("utf-8-sig")  # handle BOM

    def _sync_import():
        _ensure_wiki_mapping()
        db = get_db()

        reader = csv.DictReader(io.StringIO(text))
        stats = {
            "total_rows": 0,
            "flips_imported": 0,
            "active_trades_imported": 0,
            "skipped": 0,
            "errors": [],
            "items_not_found": [],
        }

        try:
            for row in reader:
                stats["total_rows"] += 1
                try:
                    _import_csv_row(db, row, stats)
                except Exception as e:
                    stats["errors"].append(f"Row {stats['total_rows']}: {e}")
                    if len(stats["errors"]) > 20:
                        stats["errors"].append("... (truncated)")
                        break
        finally:
            db.close()

        # Deduplicate items_not_found
        stats["items_not_found"] = list(set(stats["items_not_found"]))
        return stats

    result = await asyncio.to_thread(_sync_import)
    return result


def _parse_csv_datetime(s: str) -> Optional[datetime]:
    """Parse ISO datetime string from CSV, return None if empty."""
    if not s or not s.strip():
        return None
    s = s.strip()
    try:
        # Handle ISO format: 2026-02-18T16:14:58Z
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        return datetime.fromisoformat(s)
    except Exception:
        return None


def _import_csv_row(db, row: dict, stats: dict):
    """Process a single CSV row and insert into the database."""
    item_name = row.get("Item", "").strip()
    if not item_name:
        stats["skipped"] += 1
        return

    # Resolve item_id from Wiki mapping
    item_id = _wiki_name_to_id.get(item_name.lower())
    if item_id is None:
        stats["items_not_found"].append(item_name)
        # Still import with id=0 so data isn't lost
        item_id = 0

    status = row.get("Status", "").strip().upper()
    account = row.get("Account", "").strip() or "Unknown"
    buy_time = _parse_csv_datetime(row.get("First buy time", ""))
    sell_time = _parse_csv_datetime(row.get("Last sell time", ""))

    bought = int(row.get("Bought", 0) or 0)
    sold = int(row.get("Sold", 0) or 0)
    avg_buy = int(row.get("Avg. buy price", 0) or 0)
    avg_sell = int(row.get("Avg. sell price", 0) or 0)
    tax = int(row.get("Tax", 0) or 0)
    profit = int(row.get("Profit", 0) or 0)

    if status == "FINISHED" and bought > 0 and sold > 0 and avg_buy > 0 and avg_sell > 0:
        # Create a completed flip record
        quantity = min(bought, sold)
        gross = (avg_sell - avg_buy) * quantity
        margin_pct = (profit / (avg_buy * quantity) * 100) if (avg_buy * quantity) > 0 else 0.0
        duration = int((sell_time - buy_time).total_seconds()) if (buy_time and sell_time) else 0

        flip = FlipHistory(
            item_id=item_id,
            item_name=item_name,
            player=account,
            buy_price=avg_buy,
            sell_price=avg_sell,
            quantity=quantity,
            gross_profit=gross,
            tax=tax,
            net_profit=profit,
            margin_pct=round(margin_pct, 2),
            buy_time=buy_time or datetime.now(timezone.utc),
            sell_time=sell_time or datetime.now(timezone.utc),
            duration_seconds=duration,
        )
        insert_flip(db, flip)
        stats["flips_imported"] += 1

    elif status in ("BUYING", "SELLING"):
        # Active trade — goes into holdings/portfolio
        trade_type = "BUY" if status == "BUYING" else "SELL"
        quantity = bought if status == "BUYING" else sold
        price = avg_buy if status == "BUYING" else avg_sell
        if quantity <= 0:
            stats["skipped"] += 1
            return

        trade = Trade(
            item_id=item_id,
            item_name=item_name,
            trade_type=trade_type,
            status="BOUGHT" if status == "BUYING" else "SOLD",
            quantity=quantity,
            price=price,
            total_value=quantity * price,
            timestamp=buy_time or sell_time or datetime.now(timezone.utc),
            player=account,
            source="csv_import",
        )
        insert_trade(db, trade)
        stats["active_trades_imported"] += 1

    else:
        stats["skipped"] += 1


# ---------------------------------------------------------------------------
# POST /api/trades/clear  -  Clear imported trade data
# ---------------------------------------------------------------------------

@router.post("/api/trades/clear")
async def clear_trade_history():
    """Delete all trades and flip history from the database.

    Used before re-importing CSV data to avoid duplicates.
    """
    def _sync():
        db = get_db()
        try:
            trades_result = db.trades.delete_many({})
            flips_result = db.flip_history.delete_many({})
            return {
                "trades_deleted": trades_result.deleted_count,
                "flips_deleted": flips_result.deleted_count,
            }
        finally:
            db.close()

    return await asyncio.to_thread(_sync)


# ---------------------------------------------------------------------------
# GET /api/positions  -  Active positions with live pricing
# ---------------------------------------------------------------------------

@router.get("/api/positions")
async def get_active_positions(
    source: Optional[str] = Query(None, description="Filter by source: 'dink' or 'csv_import'"),
):
    """Return all active (open) positions with live pricing data.

    Each position includes the original buy price, current market price,
    recommended sell price, and estimated profit/loss.

    Use ?source=dink to show only DINK-tracked positions (recommended).
    """
    def _sync():
        from backend.tasks import get_position_monitor
        monitor = get_position_monitor()
        return monitor.get_positions_with_prices(source=source)

    positions = await asyncio.to_thread(_sync)
    return {"positions": positions, "total": len(positions)}


@router.post("/api/positions/dismiss")
async def dismiss_active_position(trade_id: str = Query(..., description="Trade ID to dismiss")):
    """Dismiss a single position so it no longer appears in active positions."""
    def _sync():
        db = get_db()
        try:
            dismiss_position(db, trade_id)
            return {"status": "ok", "dismissed": trade_id}
        finally:
            db.close()

    return await asyncio.to_thread(_sync)


@router.post("/api/positions/clear-csv")
async def clear_csv_positions():
    """Dismiss all positions that came from CSV import (stale data)."""
    def _sync():
        db = get_db()
        try:
            count = dismiss_positions_by_source(db, "csv_import")
            return {"status": "ok", "dismissed_count": count}
        finally:
            db.close()

    return await asyncio.to_thread(_sync)


# ---------------------------------------------------------------------------
# POST /api/dink  -  DINK Webhook Receiver
# ---------------------------------------------------------------------------

@router.post("/api/dink")
async def receive_dink_webhook(request: Request):
    """Receive DINK webhook notifications from the RuneLite plugin.

    Supports JSON and multipart/form-data (when DINK sends screenshots).
    Stores trades in the database and attempts to match buy/sell pairs
    into FlipHistory records.
    """
    content_type = request.headers.get("content-type", "")

    # Parse the incoming payload
    data = None
    if "multipart/form-data" in content_type:
        form = await request.form()
        for field_name in ("payload_json", "payload"):
            if field_name in form:
                try:
                    data = json.loads(form[field_name])
                except Exception:
                    pass
                break
        if data is None:
            # Try every field
            for key in form:
                try:
                    data = json.loads(form[key])
                    break
                except Exception:
                    continue
    else:
        try:
            data = await request.json()
        except Exception:
            body = await request.body()
            try:
                data = json.loads(body)
            except Exception:
                data = {"raw": body.decode("utf-8", errors="ignore")}

    if not data:
        return {"status": "received", "detail": "no parseable data"}

    logger.info("DINK webhook received: type=%s", data.get("type", "unknown"))

    webhook_type = data.get("type", "")
    if webhook_type == "GRAND_EXCHANGE" or "grandExchange" in str(data).lower():
        await _handle_ge_trade(data)
    else:
        logger.debug("Non-GE webhook type: %s", webhook_type)

    return {"status": "received"}


async def _handle_ge_trade(data: dict):
    """Process a DINK Grand Exchange trade and store it in the DB."""
    player_name = data.get("playerName", "Unknown")
    extra = data.get("extra", {})
    item_data = extra.get("item", {})

    item_name = item_data.get("name", "Unknown")
    item_id = item_data.get("id", 0)
    quantity = item_data.get("quantity", 0)
    price = item_data.get("priceEach", 0)
    total_value = quantity * price
    status = extra.get("status", "UNKNOWN")
    slot = extra.get("slot")
    market_price = extra.get("marketPrice")
    seller_tax = extra.get("sellerTax")

    if status in ("BOUGHT", "BUYING"):
        trade_type = "BUY"
    elif status in ("SOLD", "SELLING"):
        trade_type = "SELL"
    else:
        trade_type = "UNKNOWN"

    # Fetch current market prices for context
    live = await _fetch_live_price(item_id)

    db = get_db()
    try:
        trade = Trade(
            timestamp=datetime.now(timezone.utc),
            player=player_name,
            item_id=item_id,
            item_name=item_name,
            trade_type=trade_type,
            status=status,
            quantity=quantity,
            price=price,
            total_value=total_value,
            slot=slot,
            market_price=market_price,
            seller_tax=seller_tax,
            market_high=live.get("high"),
            market_low=live.get("low"),
            source="dink",
        )
        trade_id = insert_trade(db, trade)

        # Attempt flip matching on completed sells
        if status == "SOLD":
            _try_match_flip_mongo(db, trade)

        logger.info(
            "DINK trade stored: %s %s x%d @ %s GP (%s)",
            trade_type, item_name, quantity, f"{price:,}", status,
        )

        # Broadcast position events via WebSocket
        if status == "BOUGHT":
            # New position opened — tell the frontend and trigger monitoring
            await manager.broadcast_json({
                "type": "position_opened",
                "item_id": item_id,
                "item_name": item_name,
                "quantity": quantity,
                "buy_price": price,
                "player": player_name,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
        elif status == "SOLD":
            # Position closed
            await manager.broadcast_json({
                "type": "position_closed",
                "item_id": item_id,
                "item_name": item_name,
                "quantity": quantity,
                "sell_price": price,
                "player": player_name,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
    except Exception as e:
        logger.error("Error storing DINK trade: %s", e)
    finally:
        db.close()


def _try_match_flip_mongo(db, sell_trade: Trade):
    """Try to match a SELL trade to an earlier BUY for the same item/player (MongoDB).

    Uses FIFO matching. Creates a FlipHistory record on success.
    """
    buys = find_unmatched_buy_trades(db, sell_trade.item_id, sell_trade.player)
    if not buys:
        return

    buy = buys[0]  # FIFO — oldest first
    quantity = min(buy.quantity, sell_trade.quantity)
    gross = (sell_trade.price - buy.price) * quantity
    tax_per_item = min(int(sell_trade.price * 0.02), 5_000_000)
    total_tax = tax_per_item * quantity
    net = gross - total_tax
    margin_pct = (net / (buy.price * quantity) * 100) if buy.price else 0.0
    duration = int((sell_trade.timestamp - buy.timestamp).total_seconds()) if (sell_trade.timestamp and buy.timestamp) else 0

    flip = FlipHistory(
        item_id=sell_trade.item_id,
        item_name=sell_trade.item_name,
        player=sell_trade.player,
        buy_trade_id=buy.id,
        sell_trade_id=sell_trade.id,
        buy_price=buy.price,
        sell_price=sell_trade.price,
        quantity=quantity,
        gross_profit=gross,
        tax=total_tax,
        net_profit=net,
        margin_pct=round(margin_pct, 2),
        buy_time=buy.timestamp,
        sell_time=sell_trade.timestamp,
        duration_seconds=duration,
    )
    insert_flip(db, flip)
    logger.info(
        "Flip matched: %s x%d  profit=%s GP (%.1f%%)",
        sell_trade.item_name, quantity, f"{net:,}", margin_pct,
    )
