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

# ---------------------------------------------------------------------------
# CORS — Starlette mirrors the request Origin when credentials=True + "*",
# so this effectively allows any origin while still supporting Bearer tokens
# in cross-domain preflight requests.
# ---------------------------------------------------------------------------
CORS_ORIGINS = ["*"]
