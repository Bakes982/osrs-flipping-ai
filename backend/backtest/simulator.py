"""
backend.backtest.simulator — Minimum-viable backtest engine (PR12).

Simulates a strategy_mode + risk_profile over historical price snapshots.

At each 5-minute time step the simulator:
  1. Computes flip metrics for each tracked item using the scoring engine.
  2. Partitions items into core / spice buckets (same logic as live system).
  3. Selects a portfolio according to strategy_mode (steady / steady_spice / spice_only).
  4. Estimates fill success using fill_probability.
  5. Estimates realised profit using expected_profit × profit_multiplier,
     penalised when dump_risk is elevated.

Output metrics:
  avg_gp_per_hour        — mean GP/hr across simulated steps
  std_gp_per_hour        — standard deviation (stability)
  fail_to_fill_rate      — fraction of slot allocations that failed to fill
  avg_hold_time          — average hold_minutes per slot
  spice_contribution_pct — fraction of total profit coming from spice slots
  top_picks              — top 10 simulated items by avg_gp_per_hour
"""

from __future__ import annotations

import logging
import math
import random
import statistics
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from backend.portfolio.optimizer import is_core_candidate, is_spice_candidate, _plan_slots
from backend.prediction.scoring import calculate_flip_metrics, _compute_dump_risk

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration defaults
# ---------------------------------------------------------------------------

DEFAULT_GE_SLOTS = 8
DEFAULT_STEP_MINUTES = 5

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class SimStep:
    """Results from one 5-minute simulation step."""
    ts: float                        # Unix timestamp of the step
    gp_this_step: float = 0.0        # total GP earned this step (all slots)
    spice_gp: float     = 0.0        # GP from the spice slot (if any)
    fills: int          = 0          # number of orders that filled
    misses: int         = 0          # number of orders that failed to fill
    hold_minutes: float = 0.0        # average hold time of filled orders


@dataclass
class SimResult:
    """Aggregated results from a full backtest run."""
    days:            int
    profile:         str
    strategy:        str
    steps_simulated: int   = 0
    total_gp:        float = 0.0
    spice_gp:        float = 0.0

    step_gp: List[float] = field(default_factory=list)       # gp per step
    hold_times: List[float] = field(default_factory=list)
    fills:  int = 0
    misses: int = 0

    top_picks: List[Dict[str, Any]] = field(default_factory=list)

    # Derived (computed on export)
    avg_gp_per_hour:        float = 0.0
    std_gp_per_hour:        float = 0.0
    fail_to_fill_rate:      float = 0.0
    avg_hold_time:          float = 0.0
    spice_contribution_pct: float = 0.0

    def compute_derived(self) -> None:
        """Compute all derived metrics from raw step data."""
        total_orders = self.fills + self.misses

        # GP/hour: each step is DEFAULT_STEP_MINUTES minutes
        gph_steps = [
            g / (DEFAULT_STEP_MINUTES / 60)
            for g in self.step_gp
        ]
        self.avg_gp_per_hour = statistics.mean(gph_steps) if gph_steps else 0.0
        self.std_gp_per_hour = statistics.stdev(gph_steps) if len(gph_steps) > 1 else 0.0
        self.fail_to_fill_rate = (
            self.misses / total_orders if total_orders > 0 else 0.0
        )
        self.avg_hold_time = (
            statistics.mean(self.hold_times) if self.hold_times else 0.0
        )
        self.spice_contribution_pct = (
            self.spice_gp / max(self.total_gp, 1) * 100 if self.total_gp > 0 else 0.0
        )

    def to_dict(self) -> Dict[str, Any]:
        self.compute_derived()
        return {
            "days":               self.days,
            "profile":            self.profile,
            "strategy":           self.strategy,
            "steps_simulated":    self.steps_simulated,
            "avg_gp_per_hour":    round(self.avg_gp_per_hour),
            "std_gp_per_hour":    round(self.std_gp_per_hour),
            "fail_to_fill_rate":  round(self.fail_to_fill_rate, 4),
            "avg_hold_time":      round(self.avg_hold_time, 1),
            "spice_contribution_pct": round(self.spice_contribution_pct, 2),
            "total_gp":           round(self.total_gp),
            "spice_gp":           round(self.spice_gp),
            "top_picks":          self.top_picks[:10],
        }


# ---------------------------------------------------------------------------
# Simulation helpers
# ---------------------------------------------------------------------------

def _simulate_fill(fill_probability: float, rng: random.Random) -> bool:
    """Return True if an order fills given fill_probability."""
    return rng.random() < fill_probability


def _estimate_profit(
    metrics: dict,
    profit_multiplier: float = 1.0,
    dump_penalty: bool = False,
) -> float:
    """Estimate realised profit for one slot in one step."""
    base = metrics.get("net_profit", 0) * profit_multiplier
    if dump_penalty:
        base *= 0.5   # penalise dump-risk items by 50%
    return max(0.0, base)


def _classify_slot(metrics: dict, profile: str) -> str:
    """Return 'core', 'spice', or 'unknown'."""
    if is_core_candidate(metrics):
        return "core"
    if is_spice_candidate(metrics):
        return "spice"
    return "unknown"


# ---------------------------------------------------------------------------
# Main simulator
# ---------------------------------------------------------------------------

def run_backtest(
    snapshots_by_item: Dict[int, List[Any]],
    item_names: Optional[Dict[int, str]] = None,
    days: int = 7,
    profile: str = "balanced",
    strategy: str = "steady_spice",
    ge_slots: int = DEFAULT_GE_SLOTS,
    profit_multiplier: float = 1.0,
    seed: int = 42,
    step_minutes: int = DEFAULT_STEP_MINUTES,
) -> SimResult:
    """Simulate a strategy over historical snapshots.

    Parameters
    ----------
    snapshots_by_item:
        Dict mapping item_id → list of PriceSnapshot objects (or mock-compatible
        objects with instant_buy, instant_sell, buy_volume, sell_volume attrs).
        Snapshots should be sorted oldest-first.
    item_names:
        Optional dict of item_id → item_name strings.
    days:
        Number of days to simulate.
    profile:
        Risk profile name (balanced / conservative / aggressive).
    strategy:
        Strategy mode (steady / steady_spice / spice_only).
    ge_slots:
        Number of GE slots to fill per step.
    profit_multiplier:
        Personal execution calibration scalar (1.0 = no adjustment).
    seed:
        Random seed for deterministic fill simulation.
    step_minutes:
        Time resolution of each step in minutes.

    Returns
    -------
    SimResult
    """
    rng    = random.Random(seed)
    result = SimResult(days=days, profile=profile, strategy=strategy)
    item_names = item_names or {}

    # Total steps
    total_minutes = days * 24 * 60
    n_steps       = total_minutes // step_minutes

    # Build a time-indexed lookup: for each step, slide a window through snapshots
    # For simplicity, treat each step as using the snapshot at index `step_idx`
    # (real backtest would use exact timestamps; this is MVP).
    item_ids = list(snapshots_by_item.keys())
    if not item_ids:
        logger.warning("run_backtest: no items to simulate")
        return result

    # Per-item GP accumulators for top_picks
    item_gp: Dict[int, float] = {iid: 0.0 for iid in item_ids}
    item_gph: Dict[int, List[float]] = {iid: [] for iid in item_ids}

    for step_idx in range(n_steps):
        step_gp    = 0.0
        step_spice = 0.0
        step_fills = 0
        step_miss  = 0
        step_holds: List[float] = []

        # Score items at this time step using a sliding window of snapshots
        scored: List[dict] = []
        for iid, snaps in snapshots_by_item.items():
            if not snaps:
                continue
            # Use a window ending at snap index proportional to step_idx
            window_end = max(1, min(step_idx + 1, len(snaps)))
            window     = snaps[:window_end]
            latest     = window[-1]
            try:
                m = calculate_flip_metrics({
                    "item_id":    iid,
                    "item_name":  item_names.get(iid, f"Item {iid}"),
                    "instant_buy":  latest.instant_buy,
                    "instant_sell": latest.instant_sell,
                    "volume_5m": (getattr(latest, "buy_volume", 0) or 0)
                                + (getattr(latest, "sell_volume", 0) or 0),
                    "buy_time":  getattr(latest, "buy_time", 0) or 0,
                    "sell_time": getattr(latest, "sell_time", 0) or 0,
                    "snapshots": window,
                    "flip_history": [],
                })
            except Exception:
                continue
            scored.append(m)

        if not scored:
            result.steps_simulated += 1
            result.step_gp.append(0.0)
            continue

        # Sort by total_score
        scored.sort(key=lambda x: x.get("total_score", 0), reverse=True)

        # Partition into buckets (with dump veto stub)
        core_pool  = [m for m in scored if is_core_candidate(m)]
        spice_pool = [m for m in scored if is_spice_candidate(m)]

        # Build ordered slot list
        mode = strategy.lower()
        if mode == "spice_only":
            pool = spice_pool or core_pool
            ordered = _plan_slots(pool, [], ge_slots, spice_slots=0)
        elif mode == "steady_spice":
            ordered = _plan_slots(core_pool, spice_pool, ge_slots, spice_slots=1)
        else:
            ordered = _plan_slots(core_pool, [], ge_slots, spice_slots=0)

        # Simulate each selected slot
        seen_ids: set = set()
        for metrics in ordered[:ge_slots]:
            iid = metrics.get("item_id", 0)
            if iid in seen_ids:
                continue
            seen_ids.add(iid)

            fill_prob   = metrics.get("fill_probability", 0.5)
            hold_mins   = metrics.get("estimated_hold_time", 60)
            slot_class  = _classify_slot(metrics, profile)
            dump_signal = metrics.get("dump_signal", "none")

            if _simulate_fill(fill_prob, rng):
                profit = _estimate_profit(
                    metrics,
                    profit_multiplier=profit_multiplier,
                    dump_penalty=(dump_signal == "high"),
                )
                step_gp += profit
                step_fills += 1
                step_holds.append(float(hold_mins))
                if slot_class == "spice":
                    step_spice += profit
                item_gp[iid]  = item_gp.get(iid, 0.0) + profit
                item_gph[iid].append(
                    profit / max(hold_mins / 60, 0.01)
                )
            else:
                step_miss += 1

        result.step_gp.append(step_gp)
        result.total_gp  += step_gp
        result.spice_gp  += step_spice
        result.fills     += step_fills
        result.misses    += step_miss
        result.hold_times.extend(step_holds)
        result.steps_simulated += 1

    # Build top_picks
    top_items = sorted(item_gph.keys(), key=lambda i: statistics.mean(item_gph[i]) if item_gph[i] else 0, reverse=True)
    for iid in top_items[:10]:
        gph_vals = item_gph[iid]
        result.top_picks.append({
            "item_id":      iid,
            "item_name":    item_names.get(iid, f"Item {iid}"),
            "avg_gp_per_hour": round(statistics.mean(gph_vals)) if gph_vals else 0,
            "total_gp":    round(item_gp.get(iid, 0.0)),
        })

    return result
