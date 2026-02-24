"""
Opportunity Endpoints for OSRS Flipping AI
GET /api/opportunities       - list flip opportunities (scored and filtered)
GET /api/opportunities/{id}  - detailed analysis for a single item
"""

import asyncio
import json
import logging
import sys
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query, Request

# Ensure project root is on sys.path so we can import top-level modules
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from backend.cache import get_redis
from backend.database import get_db, get_price_history, get_item_flips, get_item, PriceSnapshot
from backend.smart_pricer import SmartPricer
from backend.flip_scorer import FlipScorer, FlipScore
from backend.arbitrage_finder import ArbitrageFinder
from ai_strategist import analyze_single_item

router = APIRouter(prefix="/api/opportunities", tags=["opportunities"])
logger = logging.getLogger(__name__)

_pricer = SmartPricer()
_scorer = FlipScorer()
_arb_finder = ArbitrageFinder()

# ------- Wiki API helpers (fallback when DB has no data) -------

def _fetch_wiki_snapshot(item_id: int):
    """Fetch current prices from OSRS Wiki API → PriceSnapshot."""
    import requests
    from datetime import datetime
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
            timestamp=datetime.utcnow(),
            buy_volume=d.get("highPriceVolume", 0),
            sell_volume=d.get("lowPriceVolume", 0),
        )
    except Exception:
        return None


_wiki_name_cache: dict = {}


def _score_to_reason(item: Dict[str, Any]) -> str:
    bullets: List[str] = []
    volume = int(item.get("volume_5m") or item.get("volume") or 0)
    margin_pct = float(item.get("spread_pct") or item.get("margin_pct") or 0)
    stability = float(item.get("score_stability") or item.get("stability_score") or 0)

    if volume >= 2000:
        bullets.append(f"High liquidity ({volume:,}/5m)")
    elif volume >= 500:
        bullets.append(f"Healthy liquidity ({volume:,}/5m)")

    if margin_pct >= 2.0:
        bullets.append(f"Strong margin ({margin_pct:.1f}% after spread)")
    elif margin_pct >= 1.0:
        bullets.append(f"Usable margin ({margin_pct:.1f}%)")

    if stability >= 70:
        bullets.append("Stable pricing profile")

    if not bullets:
        bullets.append("Balanced score across spread, volume, and risk")

    return "\n".join(f"• {b}" for b in bullets[:2])


def _map_cached_flip(item: Dict[str, Any]) -> Dict[str, Any]:
    item_id = int(item.get("item_id") or 0)
    buy_price = int(item.get("recommended_buy") or item.get("buy_price") or 0)
    sell_price = int(item.get("recommended_sell") or item.get("sell_price") or 0)
    volume_5m = int(item.get("volume_5m") or item.get("volume") or 0)
    potential_profit = int(item.get("expected_profit") or item.get("potential_profit") or item.get("net_profit") or 0)
    confidence = float(item.get("confidence") or item.get("ml_confidence") or 0)
    if confidence > 1.0:
        confidence = confidence / 100.0

    score_breakdown = {
        "spread": float(item.get("score_spread") or item.get("spread_score") or 0),
        "volume": float(item.get("score_volume") or item.get("volume_score") or 0),
        "freshness": float(item.get("score_freshness") or item.get("freshness_score") or 0),
        "trend": float(item.get("score_trend") or item.get("trend_score") or 0),
        "history": float(item.get("score_history") or item.get("history_score") or 0),
        "stability": float(item.get("score_stability") or item.get("stability_score") or 0),
        "ml_signal": float(item.get("score_ml") or item.get("ml_signal_score") or 0),
    }

    margin_gp = int(
        item.get("margin_gp")
        or item.get("spread")
        or max(0, sell_price - buy_price)
    )
    qty_suggested = int(
        item.get("qty_suggested")
        or (item.get("position_sizing") or {}).get("quantity")
        or 0
    )

    raw_name = item.get("item_name") or item.get("name") or ""
    name = str(raw_name).strip()
    if not name or name.isdigit() or name.lower().startswith("item "):
        name = _fetch_wiki_item_name(item_id) if item_id else "Unknown"

    mapped = {
        # Required normalized fields
        "item_id": item_id,
        "name": name,
        "buy_price": buy_price,
        "sell_price": sell_price,
        "margin_gp": margin_gp,
        "roi_pct": float(item.get("roi_pct") or 0),
        "instant_buy": int(item.get("instant_buy") or buy_price),
        "instant_sell": int(item.get("instant_sell") or sell_price),
        "margin": margin_gp,
        "margin_pct": float(item.get("spread_pct") or item.get("margin_pct") or 0),
        "potential_profit": potential_profit,
        "expected_profit": potential_profit,
        "tax": int(item.get("tax") or 0),
        "volume_5m": volume_5m,
        "volume": volume_5m,
        "trend": item.get("trend"),
        "confidence": confidence,
        "ml_confidence": confidence,
        "flip_score": float(item.get("total_score") or item.get("flip_score") or 0),
        "score_breakdown": score_breakdown,
        "spread_score": score_breakdown["spread"],
        "volume_score": score_breakdown["volume"],
        "freshness_score": score_breakdown["freshness"],
        "trend_score": score_breakdown["trend"],
        "history_score": score_breakdown["history"],
        "stability_score": score_breakdown["stability"],
        "ml_signal_score": score_breakdown["ml_signal"],
        "qty_suggested": qty_suggested,
        "ml_direction": item.get("ml_direction"),
        "ml_prediction_confidence": item.get("ml_prediction_confidence"),
        "ml_method": item.get("ml_method"),
        "win_rate": item.get("win_rate"),
        "total_flips": int(item.get("total_flips") or 0),
        "avg_profit": item.get("avg_profit"),
        "position_sizing": item.get("position_sizing") or {},
    }
    mapped["reason"] = _score_to_reason(item | mapped)
    return mapped


def _parse_generated_at(value: Optional[str]) -> Optional[float]:
    if not value:
        return None
    try:
        iso = value.replace("Z", "+00:00")
        dt = datetime.fromisoformat(iso)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    except Exception:
        return None


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


def _normalize_item(m: dict) -> dict:
    """Add frontend-facing aliases onto a raw ``calculate_flip_metrics`` dict.

    The scoring layer uses terse internal names; the Opportunities page expects
    a stable, human-readable contract.  We add aliases without removing the
    originals so nothing downstream breaks.
    """
    out = dict(m)

    # Primary score
    out.setdefault("flip_score",      m.get("total_score", 0.0))

    # Display name  (frontend searches on ``name``)
    out.setdefault("name",            m.get("item_name", f"Item {m.get('item_id', '?')}"))

    # Prices
    out.setdefault("buy_price",       m.get("recommended_buy", 0))
    out.setdefault("sell_price",      m.get("recommended_sell", 0))

    # Margin / profit
    out.setdefault("margin_pct",      m.get("spread_pct", 0.0))
    out.setdefault("potential_profit",m.get("net_profit",  m.get("margin_after_tax", 0)))
    # ``margin`` is used in the expanded-row pricing panel
    out.setdefault("margin",          m.get("gross_profit", 0))

    # Volume — use score_volume (0-100 liquidity score) scaled to a small
    # integer so the "Total Volume" summary card is meaningful.
    raw_vol = m.get("score_volume", m.get("liquidity_score", 0.0))
    out.setdefault("volume", int(raw_vol))

    # ML confidence — backend emits 0-1; frontend renders as 0-100% bar
    out.setdefault("ml_confidence", m.get("confidence", 0.0))

    # Per-score-component aliases (used in expanded ExpandedDetail panel)
    out.setdefault("spread_score",    m.get("score_spread",    0.0))
    out.setdefault("volume_score",    m.get("score_volume",    0.0))
    out.setdefault("freshness_score", m.get("score_freshness", 0.0))
    out.setdefault("trend_score",     m.get("score_trend",     0.0))
    out.setdefault("history_score",   m.get("score_history",   0.0))
    out.setdefault("stability_score", m.get("score_stability", 0.0))
    out.setdefault("ml_signal_score", m.get("score_ml",        0.0))

    return out


@router.get("")
async def list_opportunities(
    request: Request,
    profile: str = Query("balanced", pattern="^(conservative|balanced|aggressive)$"),
    limit: int = Query(100, ge=1, le=200),
    min_price: int = Query(0, ge=0),
    min_volume: int = Query(0, ge=0),
    min_roi_pct: float = Query(0, ge=0),
    min_profit_gp: int = Query(0, ge=0),
    min_price_gp: int = Query(0, ge=0),
    min_profit_per_item_gp: int = Query(0, ge=0),
    min_total_profit_gp: int = Query(0, ge=0),
    value_mode: str = Query("all", pattern="^(all|1m|10m)$"),
    ignore_low_value: bool = Query(False),
):
    """Return cached opportunities from profile cache with server-side filtering."""
    active_profile = request.query_params.get("profile", profile) or "balanced"
    if active_profile not in {"conservative", "balanced", "aggressive"}:
        active_profile = "balanced"

    effective_value_mode = (request.query_params.get("value_mode") or value_mode or "all").strip().lower()
    if effective_value_mode not in {"all", "1m", "10m"}:
        effective_value_mode = "all"

    effective_min_price = max(int(min_price or 0), int(min_price_gp or 0))
    if ignore_low_value and effective_min_price == 0:
        effective_min_price = 50_000

    effective_min_ppi = int(min_profit_per_item_gp or 0)
    filters_applied = {
        "value_mode": effective_value_mode,
        "min_price_gp": effective_min_price,
        "min_profit_per_item_gp": effective_min_ppi,
        "min_total_profit_gp": int(min_total_profit_gp or 0),
    }

    redis = get_redis()
    key = f"flips:top100:{active_profile}"
    raw = redis.get(key)
    ttl = -1
    try:
        ttl_fn = getattr(redis, "ttl", None)
        if callable(ttl_fn):
            ttl_val = ttl_fn(key)
            ttl = int(ttl_val) if ttl_val is not None else -1
    except Exception:
        ttl = -1

    if not raw:
        logger.info("OPP_API_READ key=%s hit=%s count=%d ttl=%s", key, False, 0, ttl)
        return {
            "generated_at": None,
            "count": 0,
            "items": [],
            "profile": active_profile,
            "value_mode": effective_value_mode,
            "filters_applied": filters_applied,
        }

    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", errors="ignore")

    try:
        parsed = json.loads(raw)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Invalid cached opportunities payload: {exc}")

    source_items = parsed.get("flips") if isinstance(parsed, dict) else None
    if not isinstance(source_items, list):
        source_items = parsed.get("items") if isinstance(parsed, dict) else []

    mapped_items = [_map_cached_flip(item) for item in source_items if isinstance(item, dict)]

    before_count = len(mapped_items)
    value_threshold = 0
    if effective_value_mode == "1m":
        value_threshold = 1_000_000
    elif effective_value_mode == "10m":
        value_threshold = 10_000_000

    value_filtered_items: List[Dict[str, Any]] = []
    for item in mapped_items:
        buy_price = int(item.get("buy_price") or item.get("recommended_buy") or 0)
        sell_price = int(item.get("sell_price") or item.get("recommended_sell") or 0)
        if value_threshold > 0 and max(buy_price, sell_price) < value_threshold:
            continue
        value_filtered_items.append(item)

    logger.info(
        "OPP_VALUE_FILTER profile=%s value_mode=%s before=%d after=%d",
        active_profile,
        effective_value_mode,
        before_count,
        len(value_filtered_items),
    )

    filtered_items: List[Dict[str, Any]] = []
    for item in value_filtered_items:
        buy_price = int(item.get("buy_price") or item.get("recommended_buy") or 0)
        volume_5m = int(item.get("volume_5m") or item.get("volume") or 0)
        roi_pct = float(item.get("roi_pct") or 0)
        margin_gp = int(item.get("margin_gp") or item.get("margin") or item.get("net_profit") or 0)
        qty_suggested = int(item.get("qty_suggested") or 0)
        existing_potential_profit = int(item.get("expected_profit") or item.get("potential_profit") or 0)
        potential_profit_gp = max(existing_potential_profit, int(margin_gp * max(0, qty_suggested)))
        item["potential_profit"] = potential_profit_gp
        item["expected_profit"] = potential_profit_gp
        profit_gp = potential_profit_gp

        if buy_price < effective_min_price:
            continue
        if volume_5m < min_volume:
            continue
        if roi_pct < min_roi_pct:
            continue
        if profit_gp < min_profit_gp:
            continue
        if margin_gp < effective_min_ppi:
            continue
        if potential_profit_gp < min_total_profit_gp:
            continue

        filtered_items.append(item)

    filtered_items.sort(key=lambda x: float(x.get("flip_score") or 0), reverse=True)
    filtered_items = filtered_items[:limit]

    generated_at = _parse_generated_at(parsed.get("ts") if isinstance(parsed, dict) else None)
    if generated_at is None and isinstance(parsed, dict):
        generated_at = parsed.get("generated_at")

    logger.info(
        "OPP_API_READ key=%s hit=%s count=%d ttl=%s",
        key,
        True,
        len(filtered_items),
        ttl,
    )
    logger.info(
        "OPP_API_FILTER profile=%s before=%d after=%d min_price=%d min_ppi=%d min_total=%d value_mode=%s",
        active_profile,
        len(value_filtered_items),
        len(filtered_items),
        effective_min_price,
        effective_min_ppi,
        int(min_total_profit_gp or 0),
        effective_value_mode,
    )
    return {
        "generated_at": generated_at,
        "count": len(filtered_items),
        "items": filtered_items,
        "profile": active_profile,
        "value_mode": effective_value_mode,
        "filters_applied": filters_applied,
    }


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
