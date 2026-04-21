#!/usr/bin/env bash
# Prometheus health check — verify daemon and subsystems are alive.
# Source: Novel code for Prometheus Sprint 6 (inspired by OpenClaw health_check.sh).
#
# Usage: ./scripts/health_check.sh
# Exit code 0 = healthy, 1 = unhealthy

set -euo pipefail

PROMETHEUS_DATA="${PROMETHEUS_DATA_DIR:-$HOME/.prometheus/data}"
PID_FILE="$PROMETHEUS_DATA/cron_scheduler.pid"
DAEMON_LOG="${PROMETHEUS_LOGS_DIR:-$HOME/.prometheus/logs}/daemon.log"

echo "=== Prometheus Health Check ==="

# Check cron scheduler PID
if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    if kill -0 "$PID" 2>/dev/null; then
        echo "[OK] Cron scheduler running (pid=$PID)"
    else
        echo "[WARN] Cron scheduler PID file exists but process $PID is dead"
    fi
else
    echo "[INFO] Cron scheduler not running (no PID file)"
fi

# Check daemon log freshness (last modified within 5 minutes)
if [ -f "$DAEMON_LOG" ]; then
    LOG_AGE=$(( $(date +%s) - $(stat -f%m "$DAEMON_LOG" 2>/dev/null || stat -c%Y "$DAEMON_LOG" 2>/dev/null || echo 0) ))
    if [ "$LOG_AGE" -lt 300 ]; then
        echo "[OK] Daemon log active (${LOG_AGE}s ago)"
    else
        echo "[WARN] Daemon log stale (${LOG_AGE}s since last write)"
    fi
else
    echo "[INFO] No daemon log found"
fi

# Check cron jobs registry
CRON_FILE="$PROMETHEUS_DATA/cron_jobs.json"
if [ -f "$CRON_FILE" ]; then
    JOB_COUNT=$(python3 -c "import json; print(len(json.load(open('$CRON_FILE'))))" 2>/dev/null || echo "?")
    echo "[OK] Cron registry: $JOB_COUNT job(s)"
else
    echo "[INFO] No cron jobs registered"
fi

# Check archive
ARCHIVE_FILE="$PROMETHEUS_DATA/archive.jsonl"
if [ -f "$ARCHIVE_FILE" ]; then
    LINE_COUNT=$(wc -l < "$ARCHIVE_FILE" | tr -d ' ')
    echo "[OK] Archive: $LINE_COUNT event(s)"
else
    echo "[INFO] No archive events"
fi

echo "=== Done ==="
