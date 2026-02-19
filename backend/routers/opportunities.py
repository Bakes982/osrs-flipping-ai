"""
Opportunity Endpoints for OSRS Flipping AI
GET /api/opportunities       - list flip opportunities (scored and filtered)
GET /api/opportunities/{id}  - detailed analysis for a single item
"""

import asyncio
import sys
import os
from typing import Optional

from fastapi import APIRouter, Query, HTTPException

# Ensure project root is on sys.path so we can import top-level modules
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from backend.database import get_db, get_price_history, get_latest_price, get_item_flips, get_item, PriceSnapshot
from backend.smart_pricer import SmartPricer
from backend.flip_scorer import FlipScorer, FlipScore, score_opportunities
from backend.arbitrage_finder import ArbitrageFinder
from backend.position_sizer import PositionSizer
from ai_strategist import scan_all_items_for_flips, analyze_single_item

router = APIRouter(prefix="/api/opportunities", tags=["opportunities"])

_pricer = SmartPricer()
_scorer = FlipScorer()
_arb_finder = ArbitrageFinder()
_sizer = PositionSizer()

# ------- Wiki API helpers (fallback when DB has no data) -------

def _fetch_wiki_snapshot(item_id: int):
    """Fetch current prices from OSRS Wiki API → PriceSnapshot."""
    import requests
    from datetime import datetime, timezone
    try:
        resp = requests.get(
            "https://prices.runescape.wiki/api/v1/osrs/latest",
            headers={"User-Agent": "osrs-flipping-ai"}, timeout=10,
        )
        resp.raise_for_status()
        d = resp.json().get("data", {}).get(str(item_id))
        if not d or not d.get("high") or not d.get("low"):
            return None
        return PriceSnapshot(
            item_id=item_id,
            instant_buy=d["high"],
            instant_sell=d["low"],
            timestamp=datetime.now(timezone.utc),
            buy_volume=d.get("highPriceVolume", 0),
            sell_volume=d.get("lowPriceVolume", 0),
        )
    except Exception:
        return None


_wiki_name_cache: dict = {}


def _fetch_wiki_item_name(item_id: int) -> str:
    """Fetch item name from OSRS Wiki mapping API (cached)."""
    if _wiki_name_cache:
        return _wiki_name_cache.get(item_id, f"Item {item_id}")
    import requests
    try:
        resp = requests.get(
            "https://prices.runescape.wiki/api/v1/osrs/mapping",
            headers={"User-Agent": "osrs-flipping-ai"}, timeout=15,
        )
        resp.raise_for_status()
        for item in resp.json():
            _wiki_name_cache[item.get("id", -1)] = item.get("name", "Unknown")
    except Exception:
        pass
    return _wiki_name_cache.get(item_id, f"Item {item_id}")


@router.get("")
async def list_opportunities(
    min_profit: int = Query(0, description="Minimum net profit in GP"),
    min_price: int = Query(0, description="Minimum item buy price in GP"),
    min_score: float = Query(15, ge=0, le=100, description="Minimum flip score (0-100)"),
    max_risk: int = Query(8, description="Maximum risk score (1-10) for initial scan"),
    min_volume: int = Query(1, description="Minimum 5-minute volume"),
    sort_by: str = Query("total_score", description="Sort field: total_score, expected_profit, spread_pct, volume_5m"),
    limit: int = Query(50, ge=1, le=200, description="Max results"),
):
    """Return a ranked list of current flip opportunities.

    Pipeline:
    1. Scan all items via ai_strategist (Wiki API)
    2. Score each through FlipScorer (veto + 0-100 composite)
    3. Filter by minimum score, profit, volume
    4. Return sorted results
    """
    try:
        raw = await asyncio.to_thread(
            scan_all_items_for_flips,
            min_price=max(10_000, min_price),
            max_price=500_000_000,
            min_margin_pct=0.2,
            max_risk=max_risk,
            limit=limit * 4,  # over-fetch for scoring
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Failed to fetch market data: {e}")

    # Score all items through the composite scorer (has sync DB calls)
    scored = await asyncio.to_thread(score_opportunities, raw, min_score, limit * 2)

    # Post-filter
    results = []
    for fs in scored:
        if fs.expected_profit is not None and fs.expected_profit < min_profit:
            continue
        if fs.volume_5m < min_volume:
            continue

        results.append({
            "item_id": fs.item_id,
            "name": fs.item_name,
            "buy_price": fs.recommended_buy,
            "sell_price": fs.recommended_sell,
            "instant_buy": fs.instant_buy,
            "instant_sell": fs.instant_sell,
            "margin": fs.spread,
            "margin_pct": round(fs.spread_pct, 2) if fs.spread_pct else 0,
            "potential_profit": fs.expected_profit,
            "roi_pct": round(fs.expected_profit_pct, 2) if fs.expected_profit_pct else 0,
            "tax": fs.tax,
            "volume": fs.volume_5m,
            "trend": fs.trend,
            "ml_confidence": round(fs.confidence, 3),

            # Composite score
            "flip_score": round(fs.total_score, 1),
            "spread_score": round(fs.spread_score, 1),
            "volume_score": round(fs.volume_score, 1),
            "freshness_score": round(fs.freshness_score, 1),
            "trend_score": round(fs.trend_score, 1),
            "history_score": round(fs.history_score, 1),
            "stability_score": round(fs.stability_score, 1),
            "ml_signal_score": round(fs.ml_score, 1),

            # ML prediction details
            "ml_direction": fs.ml_direction,
            "ml_prediction_confidence": round(fs.ml_confidence, 3) if fs.ml_confidence else None,
            "ml_method": fs.ml_method,

            # Historical
            "win_rate": round(fs.win_rate * 100, 1) if fs.win_rate is not None else None,
            "total_flips": fs.total_flips,
            "avg_profit": int(fs.avg_profit) if fs.avg_profit else None,

            "reason": fs.reason,

            # Position sizing (lightweight for list view)
            "position_sizing": _quick_sizing(fs),
        })

    # Sort
    valid_sort_keys = {
        "total_score": "flip_score",
        "expected_profit": "potential_profit",
        "spread_pct": "margin_pct",
        "volume_5m": "volume",
        "flip_score": "flip_score",
    }
    sort_key = valid_sort_keys.get(sort_by, "flip_score")
    results.sort(key=lambda x: x.get(sort_key) or 0, reverse=True)

    return {"items": results[:limit], "total": len(results)}


@router.get("/arbitrage")
async def list_arbitrage():
    """Return current set unpacking & decanting arbitrage opportunities.

    These are near-zero-risk trades:
    - Buy set -> Unpack at GE -> Sell pieces
    - Buy 3-dose potions -> Decant at Bob Barter -> Sell 4-dose
    """
    try:
        results = await asyncio.to_thread(_arb_finder.find_all_arbitrage)
        return {"items": results, "total": len(results)}
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Arbitrage scan failed: {e}")


@router.get("/{item_id}")
async def get_opportunity_detail(item_id: int):
    """Return a full analysis for a single item.

    Combines FlipScorer, SmartPricer, quant analysis, and flip history.
    """
    def _sync_detail():
        db = get_db()
        try:
            snapshots = get_price_history(db, item_id, hours=4)
            flips = get_item_flips(db, item_id, days=30)

            # Fall back to Wiki API if no DB snapshots
            if not snapshots:
                wiki_snap = _fetch_wiki_snapshot(item_id)
                if wiki_snap:
                    snapshots = [wiki_snap]

            # Score the item
            fs = _scorer.score_item(item_id, snapshots=snapshots, flips=flips)

            # SmartPricer recommendation
            rec = _pricer.price_item(item_id, snapshots=snapshots)

            # Quant analysis (text report)
            try:
                quant_report = analyze_single_item(item_id)
            except Exception:
                quant_report = None

            # Recent flips
            flip_data = [
                {
                    "buy_price": f.buy_price,
                    "sell_price": f.sell_price,
                    "quantity": f.quantity,
                    "net_profit": f.net_profit,
                    "margin_pct": round(f.margin_pct, 2),
                    "duration_seconds": f.duration_seconds,
                    "sell_time": f.sell_time.isoformat() if f.sell_time else None,
                }
                for f in flips[:20]
            ]

            # Item metadata — prefer DB, fallback to Wiki
            item_row = get_item(db, item_id)
            if item_row and item_row.name and not item_row.name.startswith("Item "):
                item_name = item_row.name
            else:
                item_name = _fetch_wiki_item_name(item_id)

            return {
                "item_id": item_id,
                "item_name": item_name,
                "current_buy": rec.instant_buy,
                "current_sell": rec.instant_sell,

                # Flip score
                "flip_score": {
                    "total": round(fs.total_score, 1),
                    "spread": round(fs.spread_score, 1),
                    "volume": round(fs.volume_score, 1),
                    "freshness": round(fs.freshness_score, 1),
                    "trend": round(fs.trend_score, 1),
                    "history": round(fs.history_score, 1),
                    "stability": round(fs.stability_score, 1),
                    "vetoed": fs.vetoed,
                    "veto_reasons": fs.veto_reasons,
                },

                "recommendation": {
                    "recommended_buy": rec.recommended_buy,
                    "recommended_sell": rec.recommended_sell,
                    "expected_profit": rec.expected_profit,
                    "expected_profit_pct": round(rec.expected_profit_pct, 2) if rec.expected_profit_pct else None,
                    "tax": rec.tax,
                    "trend": rec.trend.value,
                    "momentum": round(rec.momentum, 2),
                    "confidence": round(rec.confidence, 3),
                    "reason": rec.reason,
                    "stale_data": rec.stale_data,
                    "anomalous_spread": rec.anomalous_spread,
                },

                "vwap": {
                    "1m": round(rec.vwap_1m, 2) if rec.vwap_1m else None,
                    "5m": round(rec.vwap_5m, 2) if rec.vwap_5m else None,
                    "30m": round(rec.vwap_30m, 2) if rec.vwap_30m else None,
                    "2h": round(rec.vwap_2h, 2) if rec.vwap_2h else None,
                },
                "bollinger": {
                    "upper": round(rec.bb_upper, 2) if rec.bb_upper else None,
                    "middle": round(rec.bb_middle, 2) if rec.bb_middle else None,
                    "lower": round(rec.bb_lower, 2) if rec.bb_lower else None,
                    "position": round(rec.bb_position, 3) if rec.bb_position is not None else None,
                },
                "volume_5m": rec.volume_5m,

                # Historical performance
                "history": {
                    "win_rate": round(fs.win_rate * 100, 1) if fs.win_rate is not None else None,
                    "total_flips": fs.total_flips,
                    "avg_profit": int(fs.avg_profit) if fs.avg_profit else None,
                },

                "quant_report": quant_report,
                "recent_flips": flip_data,

                # Position sizing advice
                "position_sizing": _get_position_sizing(
                    item_id, item_name, rec, fs,
                ),
            }
        finally:
            db.close()

    return await asyncio.to_thread(_sync_detail)


def _get_position_sizing(item_id, item_name, rec, fs):
    """Compute position sizing advice using Kelly Criterion."""
    try:
        if not rec.recommended_buy or not rec.recommended_sell:
            return None

        item_row_db = get_db()
        try:
            item_row = get_item(item_row_db, item_id)
            buy_limit = item_row.buy_limit if item_row and item_row.buy_limit else 10000
        finally:
            item_row_db.close()

        advice = _sizer.size_position(
            item_id=item_id,
            buy_price=rec.recommended_buy,
            sell_price=rec.recommended_sell,
            score=fs.total_score,
            win_rate=fs.win_rate,
            volume_5m=rec.volume_5m,
            item_name=item_name,
            buy_limit=buy_limit,
        )
        return {
            "kelly_fraction": round(advice.kelly_fraction, 4),
            "half_kelly": round(advice.half_kelly, 4),
            "recommended_fraction": round(advice.recommended_fraction, 4),
            "max_investment": advice.max_investment,
            "quantity": advice.quantity,
            "stop_loss_price": advice.stop_loss_price,
            "stop_loss_pct": round(advice.stop_loss_pct * 100, 1),
            "take_profit_price": advice.take_profit_price,
            "max_hold_minutes": advice.max_hold_minutes,
            "portfolio_exposure_pct": round(advice.portfolio_exposure_pct, 1),
            "within_limits": advice.within_limits,
            "warnings": advice.limit_warnings,
            "reason": advice.reason,
        }
    except Exception as e:
        return {"error": str(e)}


def _quick_sizing(fs: FlipScore) -> dict:
    """Lightweight position sizing for list view (no DB calls)."""
    try:
        if not fs.recommended_buy or not fs.recommended_sell or fs.recommended_buy <= 0:
            return None

        sizer = _sizer
        bankroll = sizer.bankroll

        # Quick Kelly estimate from available data
        win_rate = fs.win_rate if fs.win_rate is not None else 0.55
        spread_profit = fs.expected_profit or (fs.recommended_sell - fs.recommended_buy)
        est_loss = fs.recommended_buy * 0.03  # 3% stop loss estimate

        if est_loss > 0 and spread_profit > 0:
            b = spread_profit / est_loss
            p = max(0.01, min(0.99, win_rate))
            q = 1 - p
            kelly = max(0, (p * b - q) / b)
        else:
            kelly = 0

        half_kelly = kelly / 2
        fraction = min(half_kelly * max(0.3, fs.total_score / 80), sizer.MAX_SINGLE_POSITION_PCT)
        max_invest = int(bankroll * fraction)
        qty = max_invest // fs.recommended_buy if fs.recommended_buy > 0 else 0

        return {
            "kelly": round(kelly, 3),
            "max_investment": max_invest,
            "quantity": qty,
            "stop_loss_pct": 3.0,
        }
    except Exception:
        return None
