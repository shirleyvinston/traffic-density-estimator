"""
api.py — FastAPI server.

Serves:
  GET  /           → dashboard HTML (dashboard/static/index.html)
  GET  /health     → JSON health check
  GET  /stats      → latest snapshot (REST)
  WS   /ws         → live streaming JSON frames

push_frame_data() is called from the inference loop to broadcast
the latest payload to all connected WebSocket clients.
"""

from __future__ import annotations
import asyncio
import json
import time
from pathlib import Path
from typing import Dict, Any, Set

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

# Shared state — written by inference thread, read by WS handler
_latest_payload: Dict[str, Any] = {}
_clients: Set[WebSocket] = set()
_loop: asyncio.AbstractEventLoop | None = None


def push_frame_data(payload: Dict[str, Any]):
    """
    Called from the inference thread to broadcast data.
    Uses thread-safe call_soon_threadsafe to schedule the coroutine.
    """
    global _latest_payload, _loop
    _latest_payload = payload
    if _loop and not _loop.is_closed():
        asyncio.run_coroutine_threadsafe(_broadcast(json.dumps(payload)), _loop)


async def _broadcast(message: str):
    dead = set()
    for ws in list(_clients):
        try:
            await ws.send_text(message)
        except Exception:
            dead.add(ws)
    _clients.difference_update(dead)


def create_app() -> FastAPI:
    app = FastAPI(title="Traffic Density Estimator", version="1.0.0")

    # Mount static files
    static_dir = Path(__file__).parent.parent / "dashboard" / "static"
    static_dir.mkdir(parents=True, exist_ok=True)
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    @app.on_event("startup")
    async def on_startup():
        global _loop
        _loop = asyncio.get_running_loop()

    @app.get("/", response_class=HTMLResponse)
    async def serve_dashboard():
        html_path = static_dir / "index.html"
        if html_path.exists():
            return HTMLResponse(content=html_path.read_text())
        return HTMLResponse("<h2>Dashboard not found. Make sure dashboard/static/index.html exists.</h2>")

    @app.get("/health")
    async def health():
        return JSONResponse({"status": "ok", "timestamp": time.time()})

    @app.get("/stats")
    async def stats():
        return JSONResponse(_latest_payload)

    @app.websocket("/ws")
    async def websocket_endpoint(ws: WebSocket):
        await ws.accept()
        _clients.add(ws)
        try:
            # Send latest snapshot immediately on connect
            if _latest_payload:
                await ws.send_text(json.dumps(_latest_payload))
            while True:
                # Keep connection alive — client sends pings
                await ws.receive_text()
        except WebSocketDisconnect:
            pass
        finally:
            _clients.discard(ws)

    return app
