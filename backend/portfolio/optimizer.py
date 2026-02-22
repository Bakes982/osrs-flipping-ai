"""
OSRS Flipping AI — Portfolio optimizer (Phase 5).

``generate_optimal_portfolio`` is the main entry point.  It:

1. Fetches the current top flip opportunities (scored by calculate_flip_metrics)
2. Allocates capital across GE slots using risk-adjusted GP/hour maximization
3. Respects buy limits, item exposure caps, and the Kelly Criterion
4. Returns a concrete capital allocation plan

This module is deliberately free of FastAPI dependencies so it can be called
from background tasks, API handlers, or tests without modification.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from backend.core.constants import (
    MAX_SINGLE_POSITION_PCT,
    MAX_ITEM_EXPOSURE_PCT,
    MAX_TOTAL_EXPOSURE_PCT,
    MIN_WIN_RATE_FOR_SIZING,
    STOP_LOSS_DEFAULT_PCT,
    MIN_SUGGEST_SCORE,
)
from backend.core.utils import ge_tax, safe_div, clamp, format_gp
from backend.prediction.risk import kelly_fraction, stop_loss_pct as compute_stop_loss

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Bucket classification thresholds (PR9)
# ---------------------------------------------------------------------------

# Core bucket: high-certainty flips
_CORE_MIN_CONFIDENCE    = 0.55   # 0-1 scale (== 55 / 100)
_CORE_MIN_FILL_PROB     = 0.45
_CORE_MAX_DECAY         = 0.0    # stub; 0 = no decay filter yet

# Spice bucket: higher upside but lower certainty
_SPICE_MIN_CONFIDENCE   = 0.35
_SPICE_MIN_FILL_PROB    = 0.30
_SPICE_MIN_ROI          = 1.5    # %
_SPICE_MAX_DECAY        = 0.0    # stub


def is_core_candidate(metrics: dict) -> bool:
    """Return True if the metrics qualify as a *core* flip candidate."""
    if metrics.get("vetoed"):
        return False
    confidence   = metrics.get("confidence", 0.0)
    fill_prob    = metrics.get("fill_probability", 0.0)
    net_profit   = metrics.get("net_profit", 0)
    dump_risk    = metrics.get("dump_risk_score", 0.0)     # PR11 stub (0 if absent)

    from backend import config as _cfg
    if dump_risk >= _cfg.DUMP_CORE_VETO_THRESHOLD:
        return False

    return (
        net_profit > 0
        and confidence >= _CORE_MIN_CONFIDENCE
        and fill_prob  >= _CORE_MIN_FILL_PROB
    )


def is_spice_candidate(metrics: dict) -> bool:
    """Return True if the metrics qualify as a *spice* flip candidate.

    Spice items are NOT core items that have above-average upside.
    """
    if metrics.get("vetoed"):
        return False
    if is_core_candidate(metrics):
        return False   # core takes priority; spice is the "non-core" bucket

    confidence  = metrics.get("confidence", 0.0)
    fill_prob   = metrics.get("fill_probability", 0.0)
    roi_pct     = metrics.get("roi_pct", 0.0)
    net_profit  = metrics.get("net_profit", 0)
    dump_risk   = metrics.get("dump_risk_score", 0.0)     # PR11 stub

    from backend import config as _cfg
    if dump_risk >= _cfg.DUMP_SPICE_VETO_THRESHOLD:
        return False

    high_upside = (roi_pct >= _SPICE_MIN_ROI) or (net_profit >= 50_000)
    return (
        net_profit > 0
        and high_upside
        and confidence >= _SPICE_MIN_CONFIDENCE
        and fill_prob  >= _SPICE_MIN_FILL_PROB
    )


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class SlotAllocation:
    """Recommended allocation for a single GE slot."""
    slot: int                        # GE slot index (1-based)
    item_id: int
    item_name: str
    buy_price: int
    sell_price: int
    quantity: int
    investment: int                  # buy_price * quantity
    expected_net_profit: int
    expected_roi_pct: float
    estimated_hold_minutes: int
    gp_per_hour: float
    risk_score: float                # 0–10
    confidence: float                # 0–1
    stop_loss_price: int
    kelly_fraction: float
    score: float
    reason: str = ""


@dataclass
class PortfolioAllocation:
    """Complete portfolio allocation plan."""
    capital: int
    ge_slots: int
    allocated_capital: int
    reserved_capital: int            # kept as cash buffer
    slots: List[SlotAllocation] = field(default_factory=list)
    total_expected_profit: int = 0
    total_expected_gp_per_hour: float = 0.0
    portfolio_roi_pct: float = 0.0
    warnings: List[str] = field(default_factory=list)

    @property
    def slots_used(self) -> int:
        return len(self.slots)

    def to_dict(self) -> dict:
        return {
            "capital": self.capital,
            "ge_slots": self.ge_slots,
            "allocated_capital": self.allocated_capital,
            "reserved_capital": self.reserved_capital,
            "slots_used": self.slots_used,
            "total_expected_profit": self.total_expected_profit,
            "total_expected_gp_per_hour": self.total_expected_gp_per_hour,
            "portfolio_roi_pct": round(self.portfolio_roi_pct, 4),
            "warnings": self.warnings,
            "slots": [
                {
                    "slot": s.slot,
                    "item_id": s.item_id,
                    "item_name": s.item_name,
                    "buy_price": s.buy_price,
                    "sell_price": s.sell_price,
                    "quantity": s.quantity,
                    "investment": s.investment,
                    "expected_net_profit": s.expected_net_profit,
                    "expected_roi_pct": round(s.expected_roi_pct, 4),
                    "estimated_hold_minutes": s.estimated_hold_minutes,
                    "gp_per_hour": round(s.gp_per_hour, 0),
                    "risk_score": s.risk_score,
                    "confidence": s.confidence,
                    "stop_loss_price": s.stop_loss_price,
                    "kelly_fraction": round(s.kelly_fraction, 4),
                    "score": round(s.score, 2),
                    "reason": s.reason,
                }
                for s in self.slots
            ],
        }


# ---------------------------------------------------------------------------
# Main entry point (Phase 5)
# ---------------------------------------------------------------------------

def generate_optimal_portfolio(
    capital: int,
    ge_slots: int = 8,
    min_score: float = MIN_SUGGEST_SCORE,
    risk_tolerance: str = "MEDIUM",
    strategy_mode: str = "steady",
) -> PortfolioAllocation:
    """Generate an optimal GE portfolio allocation.

    Parameters
    ----------
    capital:
        Total available GP.
    ge_slots:
        Number of GE slots available (default 8 for members).
    min_score:
        Minimum flip score to consider (filters out weak opportunities).
    risk_tolerance:
        ``"LOW"`` | ``"MEDIUM"`` | ``"HIGH"`` — adjusts position-size caps
        and minimum score threshold.
    strategy_mode:
        ``"steady"`` | ``"steady_spice"`` | ``"spice_only"`` (PR9).
        Controls how many slots are allocated to core vs spice candidates.
        - steady:       all slots from core bucket
        - steady_spice: exactly 1 slot reserved for the best spice candidate;
                        remaining slots filled from core
        - spice_only:   all slots from spice bucket

    Returns
    -------
    PortfolioAllocation
        A complete allocation plan with one ``SlotAllocation`` per filled slot.
    """
    plan = PortfolioAllocation(
        capital=capital,
        ge_slots=ge_slots,
        allocated_capital=0,
        reserved_capital=0,
    )

    if capital <= 0:
        plan.warnings.append("Capital must be positive")
        return plan

    # Apply risk tolerance overrides
    position_cap, item_cap, min_score = _risk_caps(risk_tolerance, min_score)

    # Maximum capital we're willing to deploy
    max_deployable = int(capital * MAX_TOTAL_EXPOSURE_PCT)
    remaining = max_deployable

    # Normalise strategy_mode
    mode = (strategy_mode or "steady").lower()

    # Fetch candidate opportunities from the scoring pipeline
    candidates = _fetch_candidates(min_score, limit=ge_slots * 6)
    if not candidates:
        plan.warnings.append("No opportunities scored above threshold — try lowering min_score")
        return plan

    # Sort by risk-adjusted GP/hour (primary) then by score (secondary)
    candidates.sort(key=lambda c: (_adj_gph(c), c.get("total_score", 0)), reverse=True)

    # PR9 — Partition candidates into core and spice pools.
    # Candidates without fill_probability (legacy / test data) default to a
    # pass-through: treat the whole list as core when both pools are empty.
    core_pool  = [c for c in candidates if is_core_candidate(c)]
    spice_pool = [c for c in candidates if is_spice_candidate(c)]

    # Fallback: if strict gating leaves both pools empty, treat all non-vetoed
    # candidates as core so legacy behaviour and existing tests aren't broken.
    if not core_pool and not spice_pool:
        core_pool = [c for c in candidates if not c.get("vetoed")]

    # Build the ordered slot list according to strategy_mode
    if mode == "spice_only":
        pool = spice_pool or core_pool   # graceful fallback when no spice
        ordered = _plan_slots(pool, [], ge_slots, 0)
    elif mode == "steady_spice":
        # Reserve exactly 1 slot for the best spice item
        ordered = _plan_slots(core_pool, spice_pool, ge_slots, spice_slots=1)
    else:  # "steady" (default)
        ordered = _plan_slots(core_pool, [], ge_slots, 0)

    # Greedy allocation over the ordered candidate list
    item_exposure: Dict[int, int] = {}
    used_item_ids: set = set()
    slot_idx = 1

    for chosen in ordered:
        if remaining <= 0 or slot_idx > ge_slots:
            break

        iid = chosen.get("item_id", 0)
        if iid in used_item_ids:
            continue
        existing = item_exposure.get(iid, 0)
        if existing >= capital * item_cap:
            continue

        alloc = _build_slot_allocation(
            slot=slot_idx,
            metrics=chosen,
            capital=capital,
            remaining=remaining,
            position_cap=position_cap,
            item_cap=item_cap,
            existing_exposure=existing,
        )
        if alloc is None:
            continue

        used_item_ids.add(alloc.item_id)
        item_exposure[alloc.item_id] = existing + alloc.investment
        remaining -= alloc.investment

        plan.slots.append(alloc)
        plan.allocated_capital += alloc.investment
        plan.total_expected_profit += alloc.expected_net_profit
        plan.total_expected_gp_per_hour += alloc.gp_per_hour
        slot_idx += 1

    plan.reserved_capital = capital - plan.allocated_capital
    plan.portfolio_roi_pct = safe_div(plan.total_expected_profit, plan.allocated_capital) * 100

    if plan.slots_used == 0:
        plan.warnings.append("No viable allocations — capital too low or all opportunities filtered out")
    elif plan.slots_used < ge_slots:
        plan.warnings.append(
            f"Only {plan.slots_used}/{ge_slots} slots filled — insufficient opportunities at this score threshold"
        )

    return plan


# ---------------------------------------------------------------------------
# PR9 — Slot ordering helper
# ---------------------------------------------------------------------------

def _plan_slots(
    core_pool: List[dict],
    spice_pool: List[dict],
    total_slots: int,
    spice_slots: int,
) -> List[dict]:
    """Return an ordered list of candidates for slot filling.

    core_pool  : list of core-bucket candidates (sorted best-first)
    spice_pool : list of spice-bucket candidates (sorted best-first)
    total_slots: total GE slots to fill
    spice_slots: number of slots reserved for spice (0 or 1 for now)
    """
    core_slots = total_slots - spice_slots
    ordered: List[dict] = []

    # Fill core slots first
    core_seen: set = set()
    for c in core_pool:
        if len(ordered) >= core_slots:
            break
        iid = c.get("item_id", 0)
        if iid not in core_seen:
            ordered.append(c)
            core_seen.add(iid)

    # Then append spice candidates at the end
    spice_seen: set = set()
    spice_added = 0
    for s in spice_pool:
        if spice_added >= spice_slots:
            break
        iid = s.get("item_id", 0)
        if iid not in core_seen and iid not in spice_seen:
            ordered.append(s)
            spice_seen.add(iid)
            spice_added += 1

    return ordered


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

def _risk_caps(risk_tolerance: str, min_score: float):
    """Return (position_cap, item_cap, min_score) adjusted for risk tolerance."""
    tol = risk_tolerance.upper()
    if tol == "LOW":
        return 0.08, 0.15, max(min_score, 55.0)
    if tol == "HIGH":
        return 0.20, 0.35, max(min_score, 35.0)
    # MEDIUM (default)
    return MAX_SINGLE_POSITION_PCT, MAX_ITEM_EXPOSURE_PCT, min_score


def _adj_gph(metrics: dict) -> float:
    """Risk-adjusted GP/hour: raw GPH × confidence × (1 - risk/10)."""
    gph = metrics.get("gp_per_hour") or 0.0
    # Explicit None guards so 0.0 confidence / 0.0 risk are honoured correctly.
    conf = metrics.get("confidence")
    if conf is None:
        conf = 1.0
    risk = metrics.get("risk_score")
    if risk is None:
        risk = 5.0
    risk_factor = clamp(1.0 - risk / 10.0, 0.1, 1.0)
    return gph * conf * risk_factor


def _fetch_candidates(min_score: float, limit: int) -> List[dict]:
    """Pull scored opportunity metrics from the database."""
    try:
        from backend.database import get_db, get_tracked_item_ids, get_price_history, get_item_flips
        from backend.prediction.scoring import calculate_flip_metrics

        db = get_db()
        try:
            item_ids = get_tracked_item_ids(db)
        finally:
            db.close()

        if not item_ids:
            return []

        results = []
        db = get_db()
        try:
            for item_id in item_ids[:100]:  # cap to avoid timeout
                try:
                    snaps = get_price_history(db, item_id, hours=4)
                    if not snaps:
                        continue
                    flips = get_item_flips(db, item_id, days=30)
                    latest = snaps[-1]
                    metrics = calculate_flip_metrics({
                        "item_id": item_id,
                        "instant_buy": latest.instant_buy,
                        "instant_sell": latest.instant_sell,
                        "volume_5m": (latest.buy_volume or 0) + (latest.sell_volume or 0),
                        "buy_time": latest.buy_time,
                        "sell_time": latest.sell_time,
                        "snapshots": snaps,
                        "flip_history": flips,
                    })
                    if not metrics["vetoed"] and metrics["total_score"] >= min_score:
                        results.append(metrics)
                except Exception:
                    continue
        finally:
            db.close()

        return sorted(results, key=lambda m: _adj_gph(m), reverse=True)[:limit]

    except Exception as exc:
        logger.error("Portfolio optimizer: failed to fetch candidates: %s", exc)
        return []


def _build_slot_allocation(
    slot: int,
    metrics: dict,
    capital: int,
    remaining: int,
    position_cap: float,
    item_cap: float,
    existing_exposure: int,
) -> Optional[SlotAllocation]:
    """Convert a metrics dict into a concrete SlotAllocation."""
    item_id = metrics.get("item_id", 0)
    item_name = metrics.get("item_name", f"Item {item_id}")
    buy_price = metrics.get("recommended_buy", 0)
    sell_price = metrics.get("recommended_sell", 0)
    net_per_unit = metrics.get("net_profit", 0)
    hold = metrics.get("estimated_hold_time", 60)
    gph_per_unit = (net_per_unit / (hold / 60)) if hold > 0 else 0
    score = metrics.get("total_score", 0.0)
    confidence = metrics.get("confidence", 1.0)
    risk = metrics.get("risk_score", 5.0)
    win_rate = metrics.get("win_rate") or MIN_WIN_RATE_FOR_SIZING
    volume_5m = int((metrics.get("score_volume", 0) / 100) * 20)  # rough reverse

    if buy_price <= 0 or sell_price <= 0 or net_per_unit <= 0:
        return None

    # Kelly sizing
    k = kelly_fraction(
        win_rate=win_rate,
        avg_win=float(net_per_unit),
        avg_loss=buy_price * STOP_LOSS_DEFAULT_PCT,
        buy_price=buy_price,
        sell_price=sell_price,
    )
    half_k = k / 2
    score_multiplier = clamp(score / 80.0, 0.3, 1.0)
    fraction = clamp(half_k * score_multiplier, 0.0, position_cap)

    # Respect item-level cap
    item_room = max(0, int(capital * item_cap) - existing_exposure)
    max_investment = min(int(capital * fraction), remaining, item_room)

    if max_investment < buy_price:
        return None

    quantity = max_investment // buy_price
    if quantity < 1:
        return None

    actual_investment = quantity * buy_price
    total_net = quantity * net_per_unit
    roi = safe_div(total_net, actual_investment) * 100
    total_gph = gph_per_unit * quantity
    stop = int(buy_price * (1 - compute_stop_loss(buy_price, volume_5m, score)))

    return SlotAllocation(
        slot=slot,
        item_id=item_id,
        item_name=item_name,
        buy_price=buy_price,
        sell_price=sell_price,
        quantity=quantity,
        investment=actual_investment,
        expected_net_profit=total_net,
        expected_roi_pct=roi,
        estimated_hold_minutes=hold,
        gp_per_hour=total_gph,
        risk_score=risk,
        confidence=confidence,
        stop_loss_price=stop,
        kelly_fraction=k,
        score=score,
        reason=metrics.get("reason", ""),
    )
