"""
Discord OAuth2 Authentication for OSRS Flipping AI.

Flow:
  1. User visits /api/auth/login  -> redirected to Discord OAuth
  2. Discord redirects back to /api/auth/callback  -> token issued
  3. All /api/* routes (except /api/auth/* and /api/health) require valid token
  4. Frontend checks /api/auth/me to see if logged in

Authentication supports two modes:
  - Bearer token via Authorization header (cross-domain SPA deployment)
  - Signed session cookie (same-origin dev / backward compatibility)

Configuration is read from backend.config (see config.py).
"""

import time
import json
import hmac
import hashlib
import base64
import logging
from typing import Optional

import httpx
from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import RedirectResponse, JSONResponse

from backend import config

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/auth", tags=["auth"])

# ---------------------------------------------------------------------------
# Configuration (from backend.config)
# ---------------------------------------------------------------------------

SESSION_EXPIRY = 7 * 24 * 3600  # 1 week
COOKIE_NAME = "flipping_ai_session"

DISCORD_AUTH_URL = "https://discord.com/api/oauth2/authorize"
DISCORD_TOKEN_URL = "https://discord.com/api/oauth2/token"
DISCORD_USER_URL = "https://discord.com/api/users/@me"


def is_configured() -> bool:
    """Return True if Discord OAuth credentials are set."""
    return bool(config.DISCORD_CLIENT_ID and config.DISCORD_CLIENT_SECRET)


# ---------------------------------------------------------------------------
# HMAC-signed session tokens (no external JWT dependency)
# ---------------------------------------------------------------------------

def _sign(payload_bytes: bytes) -> str:
    """Create HMAC-SHA256 signature."""
    return hmac.new(config.AUTH_SECRET.encode(), payload_bytes, hashlib.sha256).hexdigest()


def create_token(user: dict) -> str:
    """Create a signed session token from user data."""
    payload = {
        "sub": user["id"],
        "username": user["username"],
        "avatar": user.get("avatar"),
        "exp": int(time.time()) + SESSION_EXPIRY,
    }
    payload_b64 = base64.urlsafe_b64encode(json.dumps(payload).encode()).decode()
    sig = _sign(payload_b64.encode())
    return f"{payload_b64}.{sig}"


def decode_token(token: str) -> Optional[dict]:
    """Decode and verify a signed session token."""
    try:
        parts = token.split(".", 1)
        if len(parts) != 2:
            return None
        payload_b64, sig = parts
        # Verify signature
        expected_sig = _sign(payload_b64.encode())
        if not hmac.compare_digest(sig, expected_sig):
            return None
        # Decode payload
        payload = json.loads(base64.urlsafe_b64decode(payload_b64))
        # Check expiry
        if payload.get("exp", 0) < time.time():
            return None
        return payload
    except Exception:
        return None


def get_current_user(request: Request) -> Optional[dict]:
    """Extract the current user from Bearer token or session cookie."""
    # Check Authorization: Bearer header first (cross-domain)
    auth_header = request.headers.get("authorization", "")
    if auth_header.startswith("Bearer "):
        token = auth_header[7:]
        user = decode_token(token)
        if user:
            return user
    # Fall back to cookie (same-origin)
    token = request.cookies.get(COOKIE_NAME)
    if token:
        return decode_token(token)
    return None


# ---------------------------------------------------------------------------
# Auth middleware helpers
# ---------------------------------------------------------------------------

PUBLIC_PREFIXES = ("/api/auth/", "/api/health", "/api/dink", "/docs", "/openapi.json", "/ws/")


def requires_auth(request: Request) -> bool:
    """Return True if this request path needs authentication."""
    path = request.url.path
    for prefix in PUBLIC_PREFIXES:
        if path.startswith(prefix):
            return False
    return path.startswith("/api/")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.get("/login")
async def login():
    """Redirect user to Discord OAuth2 consent screen."""
    if not is_configured():
        raise HTTPException(
            status_code=503,
            detail="Discord OAuth not configured. Set DISCORD_CLIENT_ID and DISCORD_CLIENT_SECRET.",
        )
    params = {
        "client_id": config.DISCORD_CLIENT_ID,
        "redirect_uri": config.DISCORD_REDIRECT_URI,
        "response_type": "code",
        "scope": "identify",
    }
    qs = "&".join(f"{k}={v}" for k, v in params.items())
    return RedirectResponse(f"{DISCORD_AUTH_URL}?{qs}")


@router.get("/callback")
async def callback(code: str):
    """Handle Discord OAuth2 callback. Exchange code for token, redirect to frontend."""
    if not is_configured():
        raise HTTPException(status_code=503, detail="Discord OAuth not configured.")

    async with httpx.AsyncClient() as client:
        # Exchange authorization code for access token
        token_resp = await client.post(
            DISCORD_TOKEN_URL,
            data={
                "client_id": config.DISCORD_CLIENT_ID,
                "client_secret": config.DISCORD_CLIENT_SECRET,
                "grant_type": "authorization_code",
                "code": code,
                "redirect_uri": config.DISCORD_REDIRECT_URI,
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        if token_resp.status_code != 200:
            logger.error("Discord token exchange failed: %s", token_resp.text)
            raise HTTPException(status_code=401, detail="Discord authentication failed.")

        access_token = token_resp.json()["access_token"]

        # Fetch user info from Discord
        user_resp = await client.get(
            DISCORD_USER_URL,
            headers={"Authorization": f"Bearer {access_token}"},
        )
        if user_resp.status_code != 200:
            raise HTTPException(status_code=401, detail="Failed to fetch Discord user.")

        user = user_resp.json()

    # Check allowlist
    if config.ALLOWED_DISCORD_IDS and user["id"] not in config.ALLOWED_DISCORD_IDS:
        logger.warning("Denied login for Discord user %s (%s)", user["username"], user["id"])
        raise HTTPException(status_code=403, detail="You are not authorised to access this dashboard.")

    logger.info("User logged in: %s (%s)", user["username"], user["id"])

    # Create signed session token
    session_token = create_token(user)

    # Redirect to frontend with token in URL (for cross-domain SPA flow)
    redirect_url = f"{config.FRONTEND_URL}?token={session_token}"
    response = RedirectResponse(redirect_url, status_code=302)

    # Also set cookie for same-origin fallback
    response.set_cookie(
        COOKIE_NAME,
        session_token,
        max_age=SESSION_EXPIRY,
        httponly=True,
        samesite="lax",
    )
    return response


@router.get("/me")
async def me(request: Request):
    """Return the current logged-in user, or 401 if not authenticated."""
    user = get_current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated.")
    return {
        "id": user["sub"],
        "username": user["username"],
        "avatar": user.get("avatar"),
    }


@router.post("/logout")
async def logout():
    """Clear the session cookie."""
    response = JSONResponse({"status": "logged_out"})
    response.delete_cookie(COOKIE_NAME)
    return response
