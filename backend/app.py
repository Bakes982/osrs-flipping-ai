"""
OSRS Flipping AI - FastAPI Application
Main entry point for the backend server.

Run with:
    uvicorn backend.app:app --reload --host 0.0.0.0 --port 8001
"""

import sys
import os
import logging
from contextlib import asynccontextmanager

import traceback

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# ---------------------------------------------------------------------------
# Ensure the project root is importable so that both ``backend.*`` and
# top-level modules like ``ai_strategist`` can be resolved.
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from backend import config
from backend.core.logging import configure_logging
from backend.database import init_db
from backend.websocket import manager
from backend.auth import (
    router as auth_router, is_configured as auth_configured,
    requires_auth, get_current_user,
)
from backend.domain.models import UserContext
from backend.domain.enums import RiskProfile

# ---------------------------------------------------------------------------
# Logging — use the centralised configurator (Phase 8)
# ---------------------------------------------------------------------------
configure_logging()
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lifespan (startup / shutdown)
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Runs once on startup, yields for the lifetime of the app, then cleans up."""
    if config.RUN_MODE != "api":
        logger.warning(
            "backend.app started with RUN_MODE=%s. API mode is expected for this process.",
            config.RUN_MODE,
        )

    logger.info("Initialising database...")
    init_db()
    logger.info("Database ready.")

    yield  # Application is running


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="OSRS Flipping AI",
    version="2.0.0",
    description="AI-powered Old School RuneScape Grand Exchange flipping assistant",
    lifespan=lifespan,
)

# CORS -- allow the configured frontend origin + dev localhost.
# allow_credentials=True is needed so the browser's preflight (OPTIONS)
# permits the Authorization header on requests from different origins.
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# ---------------------------------------------------------------------------
# Global exception handler -- surfaces unhandled errors as structured JSON
# so the frontend can display useful diagnostic info instead of a generic
# "something went wrong" message.
# ---------------------------------------------------------------------------

@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    tb = traceback.format_exc()
    logger.error(
        "Unhandled exception on %s %s: %s\n%s",
        request.method, request.url.path, exc, tb,
    )
    return JSONResponse(
        status_code=500,
        content={
            "detail": str(exc),
            "type": type(exc).__name__,
            "path": request.url.path,
            "traceback": tb,
        },
    )


# ---------------------------------------------------------------------------
# Auth middleware -- blocks unauthenticated access to /api/* when configured
# ---------------------------------------------------------------------------

@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    """Protect API routes with Discord OAuth when configured."""
    if auth_configured() and requires_auth(request):
        user = get_current_user(request)
        if user is None:
            from fastapi.responses import JSONResponse
            return JSONResponse(
                status_code=401,
                content={"detail": "Not authenticated. Visit /api/auth/login to sign in."},
            )
    return await call_next(request)


@app.middleware("http")
async def user_context_middleware(request: Request, call_next):
    """
    Resolve the authenticated user's risk profile and personalisation data,
    then attach a ``UserContext`` to ``request.state.user_ctx``.

    Downstream route handlers can access it via::

        ctx: UserContext = request.state.user_ctx

    Falls back to ``UserContext.anonymous()`` (balanced, no personalisation)
    for unauthenticated requests or when the users collection has no record.
    """
    ctx = UserContext.anonymous()
    try:
        user = get_current_user(request)
        if user and user.get("id"):
            uid = user["id"]
            ctx.user_id  = uid
            ctx.username = user.get("username")
            # Load user document from DB for risk profile + calibration
            try:
                from backend.database import get_db
                db = get_db()
                doc = db.db["users"].find_one(
                    {"_id": uid},
                    {"risk_profile": 1, "profit_multiplier": 1,
                     "hold_multiplier": 1, "item_affinity": 1,
                     "category_affinity": 1},
                )
                if doc:
                    try:
                        ctx.risk_profile = RiskProfile(doc.get("risk_profile", "balanced"))
                    except ValueError:
                        ctx.risk_profile = RiskProfile.BALANCED
                    ctx.profit_multiplier = float(doc.get("profit_multiplier", 1.0) or 1.0)
                    ctx.hold_multiplier   = float(doc.get("hold_multiplier",   1.0) or 1.0)
                    raw_aff = doc.get("item_affinity", {}) or {}
                    ctx.item_affinity = {
                        int(k): float(v) for k, v in raw_aff.items()
                        if k.isdigit()
                    }
                    raw_cat = doc.get("category_affinity", {}) or {}
                    ctx.category_affinity = {str(k): float(v) for k, v in raw_cat.items()}
            except Exception as db_exc:
                logger.debug("user_context_middleware: DB lookup failed: %s", db_exc)
    except Exception as exc:
        logger.debug("user_context_middleware: %s", exc)

    request.state.user_ctx = ctx
    return await call_next(request)


# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------

app.include_router(auth_router)

# Register all routers (existing + new Phase 4/5/7/8 endpoints)
from backend.api.routes import register_routes
register_routes(app)


# ---------------------------------------------------------------------------
# WebSocket endpoint
# ---------------------------------------------------------------------------

@app.websocket("/ws/prices")
async def websocket_prices(websocket: WebSocket):
    """Stream real-time price updates to connected frontends."""
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            await websocket.send_text('{"type":"ack"}')
    except WebSocketDisconnect:
        await manager.disconnect(websocket)
    except Exception:
        await manager.disconnect(websocket)


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@app.get("/api/health")
async def health():
    """Simple health/status endpoint."""
    return {
        "status": "ok",
        "version": "2.0.0",
        "websocket_clients": manager.client_count,
        "frontend_url": config.FRONTEND_URL,
        "cors_origins": config.CORS_ORIGINS,
    }


# ---------------------------------------------------------------------------
# Development entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "backend.app:app",
        host="0.0.0.0",
        port=config.PORT,
        reload=True,
        reload_dirs=[_PROJECT_ROOT],
    )
