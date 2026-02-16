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
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# ---------------------------------------------------------------------------
# Ensure the project root is importable so that both ``backend.*`` and
# top-level modules like ``ai_strategist`` can be resolved.
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from backend.database import init_db
from backend.tasks import start_background_tasks, stop_background_tasks
from backend.websocket import manager
from backend.routers import opportunities, portfolio, analysis, settings

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lifespan (startup / shutdown)
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Runs once on startup, yields for the lifetime of the app, then cleans up."""
    logger.info("Initialising database...")
    init_db()

    logger.info("Starting background tasks...")
    await start_background_tasks()

    yield  # Application is running

    logger.info("Shutting down background tasks...")
    await stop_background_tasks()


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="OSRS Flipping AI",
    version="2.0.0",
    description="AI-powered Old School RuneScape Grand Exchange flipping assistant",
    lifespan=lifespan,
)

# CORS -- allow everything for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------

app.include_router(opportunities.router)
app.include_router(portfolio.router)
app.include_router(analysis.router)
app.include_router(settings.router)


# ---------------------------------------------------------------------------
# WebSocket endpoint
# ---------------------------------------------------------------------------

@app.websocket("/ws/prices")
async def websocket_prices(websocket: WebSocket):
    """Stream real-time price updates to connected frontends."""
    await manager.connect(websocket)
    try:
        while True:
            # Keep the connection alive; the client can send ping/pong or
            # subscribe messages here if needed in the future.
            data = await websocket.receive_text()
            # For now we just acknowledge; subscriptions can be added later
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
    }


# ---------------------------------------------------------------------------
# Serve static frontend files (if built)
# ---------------------------------------------------------------------------

_frontend_dist = Path(_PROJECT_ROOT) / "frontend" / "dist"
if _frontend_dist.is_dir():
    app.mount("/", StaticFiles(directory=str(_frontend_dist), html=True), name="frontend")
    logger.info("Serving static frontend from %s", _frontend_dist)
else:
    logger.info("No frontend/dist directory found -- skipping static file mount")


# ---------------------------------------------------------------------------
# Development entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "backend.app:app",
        host="0.0.0.0",
        port=8001,
        reload=True,
        reload_dirs=[_PROJECT_ROOT],
    )
