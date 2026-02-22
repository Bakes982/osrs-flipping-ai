# Changelog

All notable changes to this project will be documented in this file.

---

## [Unreleased] — PR9–PR12 (2026-02-22)

### PR9 — Strategy Mode + 1 Spice Slot

**Goal:** Maximise consistent risk-adjusted GP/hr with a configurable slot allocation between core (safe) and spice (high-upside) flip candidates.

#### Added
- `StrategyMode` enum: `steady` | `steady_spice` | `spice_only` (`backend/domain/enums.py`)
- `strategy_mode` field on `UserRecord` — defaults to `"steady"` for all new accounts (`backend/domain/models.py`)
- `PATCH /api/user/strategy_mode` — update the authenticated user's strategy mode
- `GET /api/user/profile` now includes `strategy_mode` in the response
- `FORCE_DEFAULT_STRATEGY_MODE` env var — when set to a valid strategy mode, new accounts default to that mode instead of `"steady"`
- `is_core_candidate()` / `is_spice_candidate()` classification functions in the portfolio optimizer
- `_plan_slots()` helper: builds the ordered slot list for each strategy
  - `steady` → all slots from core bucket
  - `steady_spice` → 7 core + 1 spice (for 8 slots)
  - `spice_only` → all slots from spice bucket
- 26 new unit tests

#### Changed
- `generate_optimal_portfolio()` accepts `strategy_mode` parameter (default `"steady"`)
- Graceful fallback: when no items pass strict core/spice gates (e.g., legacy test data without `fill_probability`), all non-vetoed candidates are treated as core

---

### PR10 — Two-Bucket Ranking + Recommendation Dampening

**Goal:** Eliminate recommendation flicker; keep `/flips/top5` stable and fast (<200 ms).

#### Added
- `backend/flip_cache.py` — in-memory core/spice recommendation cache
  - `update_cache(scored_items)` — classifies and dampens each cycle
  - K-poll confirmation (`DAMPENING_K=3`): item must qualify for K consecutive cycles before appearing in the eligible set
  - Hysteresis (`HYSTERESIS_CONF_MARGIN=5`, `HYSTERESIS_SCORE_MARGIN=5`): once eligible, item survives small dips below threshold
  - Separate `_top_core_cache`, `_top_spice_cache`, `_top5_cache` per `RiskProfile`
  - `stable_for_cycles` and `stable_for_minutes` fields added to every cached item
  - `get_top5(profile)`, `get_top_core(profile)`, `get_top_spice(profile)` read-only accessors
  - `cache_status()` for `/status` / `/health` exposure
- `FlipCacheWorker` background task — runs every 60 seconds, writes to `flip_cache`
- 19 new unit tests

#### Changed
- `GET /flips/top5` is now **cache-only** (no DB I/O) — reads exclusively from `backend.flip_cache`
  - Changed query param from `min_score` to `profile` (balanced | conservative | aggressive)
  - Response includes `cached: true`, `stable_for_cycles`, `stable_for_minutes`
- `RuneLiteFlip` schema gains `stable_cycles`, `stable_min`, `dump_risk`, `dump_sig` fields
- `RuneLiteTop5Response` gains `cached: bool = True`
- `FlipSummary` schema gains `stable_for_cycles`, `stable_for_minutes`, `dump_risk_score`, `dump_signal`

---

### PR11 — Dump Risk Score + Protection + Alerts Integration

**Goal:** Detect and protect users from getting trapped in a pump-and-dump.

#### Added
- `_compute_dump_risk(result, snapshots)` in `backend/prediction/scoring.py`
  - Weighted score 0–100 from four signals:
    - 35% `norm_neg_return` — short-term price drop over configurable window (default 10 min)
    - 25% `norm_vol_spike` — volatility_1h / max(volatility_24h, ε)
    - 25% `norm_compression` — spread widening signal
    - 15% `norm_fill_drop` — fill_probability below baseline
  - `dump_signal`: `"none"` (<40), `"watch"` (40–70), `"high"` (>70)
- `_apply_dump_penalties(result, dump_signal)`:
  - `"watch"` → adds `DUMP_WATCH` badge + reason string
  - `"high"` → adds `DUMP_HIGH` badge, reduces `total_score` by 30 pts, reduces `confidence` by 40%
- `dump_risk_score` and `dump_signal` fields added to all scored item dicts (default 0.0 / "none")
- Dump persistence in `flip_cache`:
  - `_update_dump_persistence()` — tracks consecutive high-dump cycles per item
  - `_emit_dump_alert()` — fires Discord webhook after `DUMP_ALERT_PERSISTENCE` consecutive high cycles
  - Alert is suppressed once fired until signal clears (no spam)
  - `get_dump_high_count()` exposed in `cache_status()`
- Spice bucket gate: items with `dump_risk_score >= DUMP_SPICE_VETO_THRESHOLD (50)` excluded from spice
- Core bucket gate: items with `dump_risk_score >= DUMP_CORE_VETO_THRESHOLD (70)` excluded from core
- 21 new unit tests

#### Changed
- `_empty_metrics()` returns `dump_risk_score: 0.0, dump_signal: "none"` by default
- `/flips/top` and `/flips/top5` responses include `dump_risk_score` and `dump_signal`

---

### PR12 — Backtest Simulator (Minimum Viable, Valuable)

**Goal:** Provide objective "consistency" metrics to prove the strategy beats Copilot.

#### Added
- `backend/backtest/simulator.py`:
  - `run_backtest(snapshots_by_item, ...)` — deterministic simulation over historical snapshots
  - Per-step: score items → classify core/spice → simulate fills (using `fill_probability` + RNG) → estimate profit (with `profit_multiplier` and dump penalty)
  - Uses the exact same `is_core_candidate` / `is_spice_candidate` / `_plan_slots` as the live system
  - `SimResult.to_dict()` output:
    - `avg_gp_per_hour`, `std_gp_per_hour` (stability)
    - `fail_to_fill_rate`, `avg_hold_time`
    - `spice_contribution_pct` — fraction of GP from the spice slot
    - `top_picks` — top 10 items by avg GP/hr
- `backend/routers/backtest.py`:
  - `GET /backtest/run?days=7&profile=balanced&strategy=steady_spice`
  - In-memory rate limiter: 1 request per 60 seconds per IP
  - `X-Backtest-Key` header auth when `BACKTEST_ADMIN_KEY` env var is set
- Registered in `register_routes()`
- 16 new unit tests

---

## Configuration reference (new env vars)

| Variable | Default | Description |
|---|---|---|
| `FORCE_DEFAULT_STRATEGY_MODE` | _(empty)_ | Override default strategy mode for new accounts |
| `DAMPENING_K` | `3` | K-poll consecutive cycles before item enters eligible set |
| `HYSTERESIS_CONF_MARGIN` | `5` | Confidence hysteresis margin (percentage points, 0-1 scale × 100) |
| `HYSTERESIS_SCORE_MARGIN` | `5` | Score hysteresis margin (0-100 scale) |
| `DUMP_SHORT_RETURN_WINDOW_MIN` | `10` | Minutes for short-return price signal |
| `DUMP_WATCH_THRESHOLD` | `40` | Dump score threshold for "watch" signal |
| `DUMP_HIGH_THRESHOLD` | `70` | Dump score threshold for "high" signal |
| `DUMP_ALERT_PERSISTENCE` | `2` | Cycles before dump alert fires |
| `DUMP_CORE_VETO_THRESHOLD` | `70` | Dump score above which item is vetoed from core bucket |
| `DUMP_SPICE_VETO_THRESHOLD` | `50` | Dump score above which item is vetoed from spice bucket |
| `BACKTEST_ADMIN_KEY` | _(empty)_ | Required in `X-Backtest-Key` header for `/backtest/run` |
| `BACKTEST_MAX_DAYS` | `30` | Maximum days per backtest request |
