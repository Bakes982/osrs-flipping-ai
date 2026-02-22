"""
Centralized configuration for OSRS Flipping AI.
All settings come from environment variables for 12-factor deployment.
"""

import os
import secrets

# ---------------------------------------------------------------------------
# MongoDB
# ---------------------------------------------------------------------------
MONGODB_URL = os.environ.get("MONGODB_URL", "mongodb://localhost:27017")
DATABASE_NAME = os.environ.get("DATABASE_NAME", "osrs_flipping_ai")

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
