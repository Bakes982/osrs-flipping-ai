#!/usr/bin/env bash
# Always-on server runner with auto-restart
# Logs to server.log, restarts on crash after 3 seconds

set -e

cd /home/user/osrs-flipping-ai

# Load .env if it exists
if [ -f .env ]; then
    set -a
    source .env
    set +a
fi

LOG_FILE="/home/user/osrs-flipping-ai/server.log"

echo "$(date): Starting OSRS Flipping AI server..." >> "$LOG_FILE"

while true; do
    echo "$(date): Server starting on port 8001" >> "$LOG_FILE"
    /usr/local/bin/uvicorn backend.app:app --host 0.0.0.0 --port 8001 >> "$LOG_FILE" 2>&1 || true
    echo "$(date): Server stopped, restarting in 3s..." >> "$LOG_FILE"
    sleep 3
done
