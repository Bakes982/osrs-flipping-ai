"""Startup script for Railway deployment."""
import os
import uvicorn

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8001))
    print(f"Starting uvicorn on port {port}", flush=True)
    uvicorn.run(
        "backend.app:app",
        host="0.0.0.0",
        port=port,
        log_level="info",
    )
