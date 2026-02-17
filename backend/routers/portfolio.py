"""
Portfolio & Trade Endpoints for OSRS Flipping AI
GET  /api/portfolio   - current holdings
GET  /api/trades      - trade history
GET  /api/performance - performance metrics
POST /api/dink        - DINK webhook receiver
"""

import sys
import os
import json
import logging
import time
from datetime import datetime
from typing import Optional

import httpx
from fastapi import APIRouter, Request, HTTPException, Query

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
    insert_trade,
    insert_flip,
    get_matched_buy_trade_ids,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["portfolio"])

PORTFOLIO_FILE = os.path.join(_PROJECT_ROOT, "portfolio.json")
WIKI_LATEST_URL = "https://prices.runescape.wiki/api/v1/osrs/latest"
USER_AGENT = "OSRS-AI-Flipper v2.0 - Discord: bakes982"


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
    """Return current holdings from portfolio.json or the database.

    If portfolio.json exists it is treated as the source of truth.
    Otherwise, open buy trades that have not been matched to a sell are
    treated as current holdings.
    """
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
                        "bought_at": t.timestamp.isoformat() if t.timestamp else None,
                    })
            portfolio["holdings"] = holdings
        finally:
            db.close()

    return portfolio


# ---------------------------------------------------------------------------
# GET /api/trades
# ---------------------------------------------------------------------------

@router.get("/api/trades")
async def get_trades(
    limit: int = Query(100, ge=1, le=1000),
    item_id: Optional[int] = Query(None),
):
    """Return trade history from the trades table."""
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


# ---------------------------------------------------------------------------
# GET /api/performance
# ---------------------------------------------------------------------------

@router.get("/api/performance")
async def get_performance():
    """Return overall performance metrics computed from flip_history."""
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
        }
    finally:
        db.close()


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

    db = SessionLocal()
    try:
        trade = Trade(
            timestamp=datetime.utcnow(),
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
        db.add(trade)
        db.flush()  # get trade.id

        # Attempt flip matching on completed sells
        if status == "SOLD":
            _try_match_flip(db, trade)

        db.commit()
        logger.info(
            "DINK trade stored: %s %s x%d @ %s GP (%s)",
            trade_type, item_name, quantity, f"{price:,}", status,
        )
    except Exception as e:
        db.rollback()
        logger.error("Error storing DINK trade: %s", e)
    finally:
        db.close()


def _try_match_flip(db, sell_trade: Trade):
    """Try to match a SELL trade to an earlier BUY for the same item/player.

    Uses FIFO matching. Creates a FlipHistory row on success.
    """
    # Find unmatched BUY trades for this item + player
    matched_buy_ids = {
        fh.buy_trade_id
        for fh in db.query(FlipHistory.buy_trade_id).all()
        if fh.buy_trade_id is not None
    }

    buy = (
        db.query(Trade)
        .filter(
            Trade.item_id == sell_trade.item_id,
            Trade.player == sell_trade.player,
            Trade.trade_type == "BUY",
            Trade.status == "BOUGHT",
            ~Trade.id.in_(matched_buy_ids) if matched_buy_ids else True,
        )
        .order_by(Trade.timestamp.asc())
        .first()
    )

    if not buy:
        return

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
        margin_pct=margin_pct,
        buy_time=buy.timestamp,
        sell_time=sell_trade.timestamp,
        duration_seconds=duration,
    )
    db.add(flip)
    logger.info(
        "Flip matched: %s x%d  profit=%s GP (%.1f%%)",
        sell_trade.item_name, quantity, f"{net:,}", margin_pct,
    )
