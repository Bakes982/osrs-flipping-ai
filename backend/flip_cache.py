"""
backend.flip_cache — In-memory two-bucket flip recommendation cache (PR10).

Architecture
------------
A background worker (FlipCacheWorker in tasks.py) calls ``update_cache()``
every ~60 seconds.  It:

1. Scores all tracked items via ``calculate_flip_metrics``.
2. Classifies each scored item as *core* or *spice* per RiskProfile.
3. Applies K-poll dampening: an item must qualify for DAMPENING_K
   consecutive worker cycles before entering the *eligible* set.
4. Applies hysteresis: once eligible, an item stays eligible even if its
   confidence/score dips slightly below the minimum threshold.
5. Writes the stable top lists to module-level cache dicts:
     _top_core_cache[profile]   → list[dict]
     _top_spice_cache[profile]  → list[dict]
     _top5_cache[profile]       → list[dict]   (≤ 5 items, mixed)

The /flips/top5 endpoint reads ONLY from _top5_cache (no DB I/O).

Cache key convention mirrors the Redis naming from the spec:
  flips:top_core:{profile}   →  _top_core_cache[profile]
  flips:top_spice:{profile}  →  _top_spice_cache[profile]
  flips:top5:{profile}       →  _top5_cache[profile]
"""

from __future__ import annotations

import logging
import time
from datetime import datetime
from typing import Dict, List, Optional

from backend import config as _cfg
from backend.domain.enums import RiskProfile
from backend.portfolio.optimizer import is_core_candidate, is_spice_candidate

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-profile thresholds (mirrors RiskProfile.min_confidence but in 0-1 scale)
# ---------------------------------------------------------------------------

def _min_conf(profile: str) -> float:
    """Minimum confidence (0-1) for the *core* bucket of a given profile."""
    return {
        "conservative": 0.70,
        "balanced":      0.50,
        "aggressive":    0.30,
    }.get(profile, 0.50)


def _min_fill(profile: str) -> float:
    """Minimum fill_probability for the *core* bucket."""
    return {
        "conservative": 0.55,
        "balanced":      0.45,
        "aggressive":    0.30,
    }.get(profile, 0.45)


def _spice_min_conf(profile: str) -> float:
    """Minimum confidence for the *spice* bucket (never < 0.25)."""
    return max(0.25, _min_conf(profile) - 0.10)   # 10-point relaxation


def _spice_min_fill(profile: str) -> float:
    """Minimum fill_probability for the *spice* bucket."""
    return max(0.20, _min_fill(profile) - 0.15)


# ---------------------------------------------------------------------------
# Cache state (module-level, updated by worker)
# ---------------------------------------------------------------------------

# top N items that have been eligible for K consecutive cycles
_top_core_cache:  Dict[str, List[dict]] = {}   # profile → items
_top_spice_cache: Dict[str, List[dict]] = {}   # profile → items
_top5_cache:      Dict[str, List[dict]] = {}   # profile → ≤5 items

# Dampening state per profile per bucket per item
# _eligible_state[profile][bucket][item_id] = {
#   "count": int,            consecutive qualifying cycles
#   "last_ts": float,        time.time() of last qualifying cycle
#   "is_active": bool,       currently in the eligible set
#   "first_active_at": Optional[float],
# }
_eligible_state: Dict[str, Dict[str, Dict[int, dict]]] = {}

# Timestamp of the last successful worker cycle
_last_update_ts: float = 0.0

# Cycle duration used to derive stable_for_minutes
_cycle_seconds: float = 60.0

# PR11 — Dump alert persistence state
# _dump_persist_state[item_id] = {"high_count": int, "alerted": bool}
_dump_persist_state: Dict[int, dict] = {}


def get_top5(profile: str = "balanced") -> List[dict]:
    """Return the cached top-5 list for a given risk profile.

    This is the only data-access function that /flips/top5 may call.
    It reads only from the in-memory cache — no DB I/O.
    """
    return list(_top5_cache.get(profile, []))


def get_top_core(profile: str = "balanced") -> List[dict]:
    return list(_top_core_cache.get(profile, []))


def get_top_spice(profile: str = "balanced") -> List[dict]:
    return list(_top_spice_cache.get(profile, []))


def last_update_ts() -> float:
    return _last_update_ts


# ---------------------------------------------------------------------------
# Core classification (profile-aware)
# ---------------------------------------------------------------------------

def _is_core_for_profile(metrics: dict, profile: str, dump_veto_threshold: float = 70.0) -> bool:
    """Core bucket test that respects the per-profile confidence/fill floors."""
    if metrics.get("vetoed"):
        return False
    if metrics.get("net_profit", 0) <= 0:
        return False
    if metrics.get("dump_risk_score", 0.0) >= dump_veto_threshold:
        return False
    return (
        metrics.get("confidence", 0.0) >= _min_conf(profile)
        and metrics.get("fill_probability", 0.0) >= _min_fill(profile)
    )


def _is_spice_for_profile(metrics: dict, profile: str, dump_veto_threshold: float = 50.0) -> bool:
    """Spice bucket test that respects the per-profile relaxed floors."""
    if metrics.get("vetoed"):
        return False
    if _is_core_for_profile(metrics, profile):
        return False   # core takes priority
    if metrics.get("net_profit", 0) <= 0:
        return False
    if metrics.get("dump_risk_score", 0.0) >= dump_veto_threshold:
        return False

    roi_pct    = metrics.get("roi_pct", 0.0)
    net_profit = metrics.get("net_profit", 0)
    high_upside = (roi_pct >= 1.5) or (net_profit >= 50_000)
    return (
        high_upside
        and metrics.get("confidence", 0.0) >= _spice_min_conf(profile)
        and metrics.get("fill_probability", 0.0) >= _spice_min_fill(profile)
    )


# ---------------------------------------------------------------------------
# Dampening logic
# ---------------------------------------------------------------------------

def _process_bucket(
    profile: str,
    bucket: str,
    qualifying_ids: set,
    all_metrics: Dict[int, dict],
    k: int,
    hysteresis_conf: float,
    hysteresis_score: float,
    min_conf: float,
    min_score: float,
) -> List[dict]:
    """Apply K-poll confirmation and hysteresis for one profile/bucket pair.

    Returns a list of *active* (eligible) item dicts with ``stable_for_cycles``
    and ``stable_for_minutes`` fields added.
    """
    global _eligible_state

    state_root = _eligible_state.setdefault(profile, {})
    state = state_root.setdefault(bucket, {})
    now = time.time()

    # --- Update counts for currently-qualifying items ---
    for iid in qualifying_ids:
        s = state.setdefault(iid, {
            "count": 0, "last_ts": now, "is_active": False,
            "first_active_at": None,
        })
        s["count"] += 1
        s["last_ts"] = now
        if s["count"] >= k and not s["is_active"]:
            s["is_active"] = True
            s["first_active_at"] = now

    # --- Hysteresis: decide whether active items that DIDN'T qualify this
    #     cycle should remain eligible ---
    for iid, s in list(state.items()):
        if iid in qualifying_ids:
            continue   # just updated above

        if not s["is_active"]:
            # Not active → just reset count (it was already not qualifying)
            s["count"] = 0
            continue

        # Item was active but didn't qualify this cycle.
        m = all_metrics.get(iid, {})
        conf  = m.get("confidence", 0.0)
        score = m.get("total_score", 0.0)

        within_hysteresis = (
            conf  >= (min_conf  - hysteresis_conf / 100.0)  # conf is 0-1, margin is %pts
            and score >= (min_score - hysteresis_score)
        )
        if within_hysteresis:
            # Keep active; reset count so it won't be promoted again
            pass
        else:
            s["is_active"] = False
            s["count"] = 0

    # --- Collect active items ---
    active: List[dict] = []
    for iid, s in state.items():
        if not s["is_active"]:
            continue
        m = all_metrics.get(iid)
        if m is None:
            continue
        enriched = dict(m)
        cycles = s["count"]
        enriched["stable_for_cycles"] = cycles
        enriched["stable_for_minutes"] = round(cycles * (_cycle_seconds / 60), 1)
        active.append(enriched)

    # Sort by total_score descending
    active.sort(key=lambda x: x.get("total_score", 0), reverse=True)
    return active


# ---------------------------------------------------------------------------
# Cache update (called by FlipCacheWorker)
# ---------------------------------------------------------------------------

_PROFILES = [p.value for p in RiskProfile]


def update_cache(scored_items: List[dict], cycle_seconds: float = 60.0) -> None:
    """Update all in-memory top-list caches from a freshly-scored item list.

    Parameters
    ----------
    scored_items:
        List of metric dicts from ``calculate_flip_metrics``.  Items that are
        vetoed or have non-positive ``net_profit`` are automatically excluded.
    cycle_seconds:
        Worker cycle duration (used to compute ``stable_for_minutes``).
    """
    global _top_core_cache, _top_spice_cache, _top5_cache
    global _last_update_ts, _cycle_seconds

    _cycle_seconds = cycle_seconds

    k           = _cfg.DAMPENING_K
    hysteresis_conf  = _cfg.HYSTERESIS_CONF_MARGIN
    hysteresis_score = _cfg.HYSTERESIS_SCORE_MARGIN

    # Build a fast lookup dict
    all_metrics: Dict[int, dict] = {
        m["item_id"]: m for m in scored_items if m.get("item_id")
    }

    for profile in _PROFILES:
        min_conf_val  = _min_conf(profile)
        # Use a minimum score proxy: confidence threshold × 100
        min_score_val = min_conf_val * 100

        # Classify this cycle's qualifying items
        core_ids  = {
            m["item_id"]
            for m in scored_items
            if _is_core_for_profile(m, profile,
                                    dump_veto_threshold=_cfg.DUMP_CORE_VETO_THRESHOLD)
        }
        spice_ids = {
            m["item_id"]
            for m in scored_items
            if _is_spice_for_profile(m, profile,
                                     dump_veto_threshold=_cfg.DUMP_SPICE_VETO_THRESHOLD)
        }

        core_active = _process_bucket(
            profile=profile, bucket="core",
            qualifying_ids=core_ids,
            all_metrics=all_metrics,
            k=k,
            hysteresis_conf=hysteresis_conf,
            hysteresis_score=hysteresis_score,
            min_conf=min_conf_val,
            min_score=min_score_val,
        )
        spice_active = _process_bucket(
            profile=profile, bucket="spice",
            qualifying_ids=spice_ids,
            all_metrics=all_metrics,
            k=k,
            hysteresis_conf=hysteresis_conf,
            hysteresis_score=hysteresis_score,
            min_conf=_spice_min_conf(profile),
            min_score=min_score_val * 0.8,
        )

        _top_core_cache[profile]  = core_active
        _top_spice_cache[profile] = spice_active

        # Derive top5 per strategy semantics (default: steady_spice style —
        # 4 core + 1 spice, fall back to 5 core if no spice available)
        top5 = _build_top5(core_active, spice_active)
        _top5_cache[profile] = top5

    # PR11 — Track dump persistence and emit alerts after N consecutive cycles
    _update_dump_persistence(all_metrics)

    _last_update_ts = time.time()
    logger.debug(
        "flip_cache updated: %d profiles, %d total scored items",
        len(_PROFILES), len(scored_items),
    )


def _passes_dump_filters(m: dict) -> bool:
    """Return True if the item clears the v2 quality filters.

    Filters (all configurable via env vars):
      - recommended_buy  >= DUMP_ALERT_MIN_PRICE_GP  (only checked when > 0)
      - net_profit       >= DUMP_ALERT_MIN_PROFIT_GP

    The price filter is intentionally skipped when ``recommended_buy`` is
    absent or zero — the field may legitimately be missing in certain
    scoring contexts (e.g. test fixtures, early pipeline stages).
    """
    buy    = m.get("recommended_buy") or m.get("instant_buy") or 0
    profit = m.get("net_profit", 0)
    if buy > 0 and buy < _cfg.DUMP_ALERT_MIN_PRICE_GP:
        return False
    return profit >= _cfg.DUMP_ALERT_MIN_PROFIT_GP


def _update_dump_persistence(all_metrics: Dict[int, dict]) -> None:
    """PR11 / v2: Track consecutive high-dump cycles and fire alerts when threshold met.

    Improvements over v1:
      - Quality filters (min price, min profit) gate alert eligibility.
      - 60-minute time-based cooldown per item prevents re-alert spam even if
        the signal bounces — only resets after the cooldown window elapses.
    """
    global _dump_persist_state

    persistence_k  = _cfg.DUMP_ALERT_PERSISTENCE
    cooldown_secs  = _cfg.DUMP_ALERT_COOLDOWN_MINUTES * 60
    now            = time.time()

    for iid, m in all_metrics.items():
        signal = m.get("dump_signal", "none")
        state  = _dump_persist_state.setdefault(
            iid,
            {"high_count": 0, "alerted": False, "last_alert_ts": 0.0},
        )

        if signal == "high":
            state["high_count"] += 1

            cooldown_elapsed = (now - state["last_alert_ts"]) >= cooldown_secs
            if (
                state["high_count"] >= persistence_k
                and cooldown_elapsed
                and _passes_dump_filters(m)
            ):
                state["alerted"]      = True
                state["last_alert_ts"] = now
                _emit_dump_alert(m)
        else:
            # Signal cleared — reset count and alerted flag.
            # last_alert_ts is intentionally kept so the cooldown window still
            # applies if the signal bounces back within 60 minutes.
            state["high_count"] = 0
            state["alerted"]    = False


def _format_dump_message(metrics: dict) -> str:
    """Return a clean, human-readable dump alert string.

    Format: DUMP {SIGNAL} {name} — Buy {buy} | Sell {sell} | Qty {qty} | +{ppi} ea | +{total} total

    item_id is intentionally omitted from the user-facing message and is only
    emitted via structured logging.
    """
    from backend.analytics.trade_plan import build_trade_plan
    from backend.core.constants import GE_TAX_RATE, GE_TAX_CAP, GE_TAX_FREE_BELOW
    from backend import config as _cfg

    name   = metrics.get("item_name", "Unknown")
    signal = (metrics.get("dump_signal") or "HIGH").upper()

    tp = build_trade_plan(
        buy_price=int(metrics.get("recommended_buy") or 0),
        sell_price=int(metrics.get("recommended_sell") or 0),
        item_limit=None,
        liquidity_score=None,   # use 50 % default — conservative for alerts
        risk_profile_position_cap_pct=0.15,
        capital_gp=_cfg.DEFAULT_CAPITAL_GP,
        ge_tax_rate=GE_TAX_RATE,
        ge_tax_cap=GE_TAX_CAP,
        ge_tax_free_below=GE_TAX_FREE_BELOW,
    )
    return (
        f"DUMP {signal} {name} — "
        f"Buy {tp['buy_price']:,} | Sell {tp['sell_price']:,} | "
        f"Qty {tp['qty_to_buy']:,} | "
        f"+{tp['profit_per_item']:,} ea | "
        f"+{tp['total_profit']:,} total"
    )


def _emit_dump_alert(metrics: dict) -> None:
    """Fire a rich dump alert via Discord webhook (v2).

    Improvements over v1:
      - Resolves item name via ItemNameResolver (6h TTL cache) in case the
        metrics dict has a missing or stale name.
      - Sends a Discord embed with colour-coded severity and trade plan fields.
      - Attaches a 6h price chart image when chart generation succeeds.
      - Logs item_id in structured logger.warning only (not in user message).
    """
    import io
    import json
    import os
    from datetime import datetime

    webhook_url = os.environ.get("DISCORD_WEBHOOK_URL", "")

    item_id = metrics.get("item_id")

    # Resolve name: prefer metrics dict, fallback to Wiki mapping cache
    from backend.alerts.item_name_resolver import resolver as _name_resolver
    name = metrics.get("item_name") or _name_resolver.resolve(item_id)

    # Structured log for debugging (item_id is safe here)
    logger.warning(
        "DUMP_HIGH alert: item_id=%s name=%s dump_risk=%.1f",
        item_id, name, metrics.get("dump_risk_score", 0),
    )

    if not webhook_url:
        logger.warning("Set DISCORD_WEBHOOK_URL to receive dump alert notifications")
        return

    signal = (metrics.get("dump_signal") or "HIGH").upper()
    buy    = int(metrics.get("recommended_buy") or 0)
    sell   = int(metrics.get("recommended_sell") or 0)
    score  = float(metrics.get("total_score") or 0)
    trend  = metrics.get("trend", "NEUTRAL")

    # Human-readable trade plan line (from existing formatter)
    plan_line = _format_dump_message(metrics)

    embed = {
        "title":       f"\u26a0\ufe0f DUMP {signal} \u2014 {name}",
        "description": plan_line,
        "color":       0xEF5350,   # red
        "timestamp":   datetime.utcnow().isoformat(),
        "footer":      {"text": "OSRS Flipping AI \u2022 Dump Detection v2"},
        "fields": [
            {"name": "Dump Risk",  "value": f"{metrics.get('dump_risk_score', 0):.0f}/100", "inline": True},
            {"name": "Score",      "value": f"{score:.0f}/100",                              "inline": True},
            {"name": "Trend",      "value": trend,                                           "inline": True},
        ],
    }

    # Try to generate a 6h price chart
    chart_bytes = None
    if item_id:
        try:
            from backend.discord_notifier import generate_opportunity_chart
            chart_bytes = generate_opportunity_chart(
                item_name=name,
                item_id=item_id,
                buy_price=buy,
                sell_price=sell,
                score=score,
                trend=trend,
                hours=6,
            )
        except Exception as exc:
            logger.warning("Dump alert chart generation failed for item %s: %s", item_id, exc)

    try:
        import requests as _requests

        if chart_bytes:
            filename = f"dump_{item_id}.png"
            embed["image"] = {"url": f"attachment://{filename}"}
            payload_json = json.dumps({"embeds": [embed]})
            resp = _requests.post(
                webhook_url,
                data={"payload_json": payload_json},
                files={"file": (filename, io.BytesIO(chart_bytes), "image/png")},
                timeout=15,
            )
        else:
            resp = _requests.post(
                webhook_url,
                json={"embeds": [embed]},
                timeout=10,
            )

        if resp.status_code == 429:
            import time as _time
            retry_after = resp.json().get("retry_after", 2)
            logger.warning("Discord rate-limited, retrying in %.1fs", retry_after)
            _time.sleep(retry_after)
            resp = _requests.post(webhook_url, json={"embeds": [embed]}, timeout=10)

        if resp.status_code not in (200, 204):
            logger.error("Dump alert webhook returned HTTP %s", resp.status_code)

    except Exception as exc:
        logger.error("Failed to send dump alert: %s", exc)


def get_dump_high_count() -> int:
    """Return the number of items currently showing a high dump signal.

    Used by /status endpoint (PR11).
    """
    return sum(
        1 for s in _dump_persist_state.values() if s.get("high_count", 0) >= 1
    )


def _build_top5(core: List[dict], spice: List[dict]) -> List[dict]:
    """Build the top-5 display list.

    Defaults to steady_spice semantics: 4 core + 1 spice.
    Falls back to 5 core if no spice qualifies.
    """
    result: List[dict] = []
    max_core = 4 if spice else 5
    result.extend(core[:max_core])
    if spice:
        # Pick best spice item not already in core list
        core_ids = {m["item_id"] for m in result}
        for s in spice:
            if s["item_id"] not in core_ids:
                result.append(s)
                break
    return result[:5]


# ---------------------------------------------------------------------------
# Status for /health or /status endpoint
# ---------------------------------------------------------------------------

def cache_status() -> dict:
    """Return a summary dict suitable for inclusion in /status or /health."""
    return {
        "last_update": datetime.utcfromtimestamp(_last_update_ts).isoformat()
            if _last_update_ts else None,
        "profiles_cached": list(_top5_cache.keys()),
        "top5_sizes": {p: len(v) for p, v in _top5_cache.items()},
        "top_core_sizes": {p: len(v) for p, v in _top_core_cache.items()},
        "top_spice_sizes": {p: len(v) for p, v in _top_spice_cache.items()},
        "dump_high_count": get_dump_high_count(),   # PR11
    }
