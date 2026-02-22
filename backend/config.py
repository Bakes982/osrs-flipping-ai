"""
Centralized configuration for OSRS Flipping AI.
All settings come from environment variables for 12-factor deployment.
"""

import os
import secrets


def _env_bool(name: str, default: bool = False) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "on"}


# ---------------------------------------------------------------------------
# MongoDB
# ---------------------------------------------------------------------------
MONGODB_URL = os.environ.get("MONGODB_URL", "mongodb://localhost:27017")
DATABASE_NAME = os.environ.get("DATABASE_NAME", "osrs_flipping_ai")
REDIS_URL = os.environ.get("REDIS_URL", "").strip()

# ---------------------------------------------------------------------------
# URLs
# ---------------------------------------------------------------------------
FRONTEND_URL = os.environ.get("FRONTEND_URL", "http://localhost:5173")
BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8001")

# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------
AUTH_SECRET = os.environ.get("AUTH_SECRET", "") or secrets.token_hex(32)
DISCORD_CLIENT_ID = os.environ.get("DISCORD_CLIENT_ID", "")
DISCORD_CLIENT_SECRET = os.environ.get("DISCORD_CLIENT_SECRET", "")
DISCORD_REDIRECT_URI = os.environ.get(
    "DISCORD_REDIRECT_URI",
    f"{BACKEND_URL}/api/auth/callback",
)
ALLOWED_DISCORD_IDS = {
    s.strip()
    for s in os.environ.get("ALLOWED_DISCORD_IDS", "").split(",")
    if s.strip()
}

# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------
PORT = int(os.environ.get("PORT", "8001"))
MODELS_DIR = os.environ.get("MODELS_DIR", "models")
RUN_MODE = os.environ.get("RUN_MODE", "api").strip().lower()

# Worker runtime options
WORKER_RETRY_INITIAL_SECONDS = float(os.environ.get("WORKER_RETRY_INITIAL_SECONDS", "2"))
WORKER_RETRY_MAX_SECONDS = float(os.environ.get("WORKER_RETRY_MAX_SECONDS", "60"))
WORKER_RUN_ONCE = _env_bool("WORKER_RUN_ONCE", False)
WORKER_RUN_ONCE_SECONDS = float(os.environ.get("WORKER_RUN_ONCE_SECONDS", "0.05"))
FLIPS_CACHE_TTL_SECONDS = int(os.environ.get("FLIPS_CACHE_TTL_SECONDS", "180"))
FLIPS_CACHE_WARM_INTERVAL_SECONDS = int(os.environ.get("FLIPS_CACHE_WARM_INTERVAL_SECONDS", "30"))
FLIPS_FRESH_MAX_PER_MINUTE = int(os.environ.get("FLIPS_FRESH_MAX_PER_MINUTE", "6"))
ALLOW_ANON = _env_bool("ALLOW_ANON", False)
TOP5_RATE_LIMIT_PER_MINUTE = int(os.environ.get("TOP5_RATE_LIMIT_PER_MINUTE", "60"))
TOP_RATE_LIMIT_PER_MINUTE = int(os.environ.get("TOP_RATE_LIMIT_PER_MINUTE", "20"))
TOP_REQUIRE_API_KEY = _env_bool("TOP_REQUIRE_API_KEY", False)
WORKER_CIRCUIT_FAILURE_THRESHOLD = int(os.environ.get("WORKER_CIRCUIT_FAILURE_THRESHOLD", "5"))
WORKER_CIRCUIT_OPEN_SECONDS = int(os.environ.get("WORKER_CIRCUIT_OPEN_SECONDS", "300"))
SCORE_STALE_MAX_MINUTES = int(os.environ.get("SCORE_STALE_MAX_MINUTES", "45"))
WORKER_OK_MAX_AGE_SECONDS = int(os.environ.get("WORKER_OK_MAX_AGE_SECONDS", "180"))

# ---------------------------------------------------------------------------
# CORS — Starlette mirrors the request Origin when credentials=True + "*",
# so this effectively allows any origin while still supporting Bearer tokens
# in cross-domain preflight requests.
# ---------------------------------------------------------------------------
CORS_ORIGINS = ["*"]

# ---------------------------------------------------------------------------
# PR9 — Strategy mode default override
# ---------------------------------------------------------------------------
# When set to a valid StrategyMode value (e.g. "steady_spice"), new user
# accounts will default to that strategy instead of "steady".
FORCE_DEFAULT_STRATEGY_MODE = os.environ.get("FORCE_DEFAULT_STRATEGY_MODE", "")

# ---------------------------------------------------------------------------
# PR10 — Flip recommendation dampening / hysteresis
# ---------------------------------------------------------------------------
# An item must qualify for DAMPENING_K consecutive worker cycles before it
# enters the eligible (visible) set.
DAMPENING_K = int(os.environ.get("DAMPENING_K", "3"))
# Once active, confidence can dip this many points below min before eviction.
HYSTERESIS_CONF_MARGIN = float(os.environ.get("HYSTERESIS_CONF_MARGIN", "5"))
# Same margin applied to total_score for score-based hysteresis.
HYSTERESIS_SCORE_MARGIN = float(os.environ.get("HYSTERESIS_SCORE_MARGIN", "5"))

# ---------------------------------------------------------------------------
# PR11 — Dump detector knobs
# ---------------------------------------------------------------------------
# Window (minutes) for short-return price-drop signal.
DUMP_SHORT_RETURN_WINDOW_MIN = int(os.environ.get("DUMP_SHORT_RETURN_WINDOW_MIN", "10"))
# Thresholds for dump_signal classification (score 0-100).
DUMP_WATCH_THRESHOLD  = float(os.environ.get("DUMP_WATCH_THRESHOLD", "40"))
DUMP_HIGH_THRESHOLD   = float(os.environ.get("DUMP_HIGH_THRESHOLD", "70"))
# Consecutive cycles with dump_signal=="high" before alert fires.
DUMP_ALERT_PERSISTENCE = int(os.environ.get("DUMP_ALERT_PERSISTENCE", "2"))
# Score thresholds above which an item is vetoed from each bucket.
DUMP_CORE_VETO_THRESHOLD  = float(os.environ.get("DUMP_CORE_VETO_THRESHOLD", "70"))
DUMP_SPICE_VETO_THRESHOLD = float(os.environ.get("DUMP_SPICE_VETO_THRESHOLD", "50"))

# ---------------------------------------------------------------------------
# PR12 — Backtest endpoint
# ---------------------------------------------------------------------------
# Admin key required to call GET /backtest/run (empty = open access in dev).
BACKTEST_ADMIN_KEY = os.environ.get("BACKTEST_ADMIN_KEY", "")
# Maximum days of history a single backtest request may span.
BACKTEST_MAX_DAYS = int(os.environ.get("BACKTEST_MAX_DAYS", "30"))

# ---------------------------------------------------------------------------
# Trade plan — capital and position sizing defaults
# ---------------------------------------------------------------------------
# Default capital used by build_trade_plan() when no user-level capital is set.
DEFAULT_CAPITAL_GP = int(os.environ.get("DEFAULT_CAPITAL_GP", "50000000"))

# ---------------------------------------------------------------------------
# Dump Detector v2 — quality filters for actionable alerts
# ---------------------------------------------------------------------------
# Minimum insta-sell price to consider (skip cheap junk).
DUMP_V2_MIN_PRICE_GP = int(os.environ.get("DUMP_V2_MIN_PRICE_GP", "500000"))
# Minimum % drop from 4h reference average before flagging as a dump.
DUMP_V2_MIN_DROP_PCT = float(os.environ.get("DUMP_V2_MIN_DROP_PCT", "4.0"))
# Minimum combined 5m volume (sold_5m + bought_5m) to confirm trading activity.
DUMP_V2_MIN_VOLUME_TRADES = int(os.environ.get("DUMP_V2_MIN_VOLUME_TRADES", "25"))
# Minimum sell-side dominance — sold_5m / total_5m must meet this threshold.
DUMP_V2_MIN_SELL_RATIO = float(os.environ.get("DUMP_V2_MIN_SELL_RATIO", "0.80"))
# Minimum net profit per item (after GE tax) required to alert.
DUMP_V2_MIN_PROFIT_PER_ITEM = int(os.environ.get("DUMP_V2_MIN_PROFIT_PER_ITEM", "2000"))
# Minimum estimated total profit (profit_per_item × qty) required to alert.
DUMP_V2_MIN_TOTAL_PROFIT = int(os.environ.get("DUMP_V2_MIN_TOTAL_PROFIT", "150000"))
# Per-item alert cooldown in minutes — prevents spam for the same item.
DUMP_V2_COOLDOWN_MINUTES = int(os.environ.get("DUMP_V2_COOLDOWN_MINUTES", "60"))
# Position cap used when computing trade plan quantity (fraction of capital).
DUMP_V2_POSITION_CAP_PCT = float(os.environ.get("DUMP_V2_POSITION_CAP_PCT", "0.10"))
