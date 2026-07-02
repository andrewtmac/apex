#!/bin/bash
# APEX V2 - Persistent background runner
# Restarts automatically on crash, logs to file

BOT_DIR="/Users/odin-mini/CODING/apex"
LOG_FILE="$BOT_DIR/logs/apex_v2.log"
PID_FILE="$BOT_DIR/logs/apex_v2.pid"

mkdir -p "$BOT_DIR/logs"

# Write PID for management
echo $$ > "$PID_FILE"

cd "$BOT_DIR"

# Rotate log if > 10MB
if [ -f "$LOG_FILE" ] && [ "$(stat -f%z "$LOG_FILE" 2>/dev/null || echo 0)" -gt 10485760 ]; then
    mv "$LOG_FILE" "${LOG_FILE}.old"
fi

exec uv run python scripts/apex_v2.py >> "$LOG_FILE" 2>&1
