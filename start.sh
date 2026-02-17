#!/usr/bin/env bash
set -e

echo "=== OSRS Flipping AI v2.0 ==="
echo ""

# Check Python
if ! command -v python3 &>/dev/null; then
    echo "ERROR: python3 is required but not found."
    exit 1
fi

# Check Node.js (optional, for frontend dev mode)
HAS_NODE=false
if command -v node &>/dev/null; then
    HAS_NODE=true
fi

# Install Python dependencies
echo "[1/4] Installing Python dependencies..."
pip install -r requirements.txt --quiet

# Initialize database and run migrations
echo "[2/4] Initializing database..."
python3 -m backend.migrate 2>/dev/null || echo "  (No migration data found, starting fresh)"

# Build frontend if Node.js is available and dist doesn't exist
if [ "$HAS_NODE" = true ] && [ ! -d "frontend/dist" ]; then
    echo "[3/4] Building frontend..."
    cd frontend
    npm install --silent
    npm run build
    cd ..
else
    echo "[3/4] Skipping frontend build (use 'cd frontend && npm run build' to build manually)"
fi

# Start the backend
echo "[4/4] Starting backend on http://localhost:8001"
echo "  API docs: http://localhost:8001/docs"
echo "  Health:   http://localhost:8001/api/health"
echo ""

if [ "$HAS_NODE" = true ] && [ ! -d "frontend/dist" ]; then
    echo "  Frontend dev: cd frontend && npm run dev"
fi

echo "Press Ctrl+C to stop."
echo ""

python3 -m uvicorn backend.app:app --host 0.0.0.0 --port 8001 --reload
