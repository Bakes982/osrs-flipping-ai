"""
WebSocket Manager for OSRS Flipping AI
Manages connected clients and broadcasts price updates in real-time.
"""

import asyncio
import json
import logging
from typing import List

from fastapi import WebSocket

logger = logging.getLogger(__name__)


class ConnectionManager:
    """Tracks connected WebSocket clients and broadcasts messages."""

    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket):
        """Accept a new WebSocket connection and register it."""
        await websocket.accept()
        async with self._lock:
            self.active_connections.append(websocket)
        logger.info(
            "WebSocket client connected. Total clients: %d",
            len(self.active_connections),
        )

    async def disconnect(self, websocket: WebSocket):
        """Remove a disconnected client."""
        async with self._lock:
            if websocket in self.active_connections:
                self.active_connections.remove(websocket)
        logger.info(
            "WebSocket client disconnected. Total clients: %d",
            len(self.active_connections),
        )

    async def broadcast_prices(self, data: dict):
        """Send price data to every connected client.

        Silently removes clients that have gone away.
        """
        payload = json.dumps(data)
        stale: List[WebSocket] = []

        async with self._lock:
            connections = list(self.active_connections)

        for ws in connections:
            try:
                await ws.send_text(payload)
            except Exception:
                stale.append(ws)

        # Clean up dead connections
        if stale:
            async with self._lock:
                for ws in stale:
                    if ws in self.active_connections:
                        self.active_connections.remove(ws)
            logger.info("Removed %d stale WebSocket connections", len(stale))

    async def broadcast_json(self, data: dict):
        """Alias kept for convenience -- sends JSON-encoded dict."""
        await self.broadcast_prices(data)

    @property
    def client_count(self) -> int:
        return len(self.active_connections)


# Singleton used across the application
manager = ConnectionManager()
