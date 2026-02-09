#!/bin/bash
# ============================================================
# Kids Story Channel - Health Check Script
# ============================================================
# Checks if all services are running and restarts if needed.
# Called by systemd watchdog timer every 15 minutes.
# ============================================================

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="$PROJECT_DIR/data/logs"
mkdir -p "$LOG_DIR"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [HEALTH] $1" | tee -a "$LOG_DIR/healthcheck.log"
}

HEALTHY=true

# Check 1: Is Ollama running?
if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    log "WARNING: Ollama is not responding. Restarting..."
    ollama serve &
    sleep 10
    if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        log "ERROR: Ollama failed to restart!"
        HEALTHY=false
    else
        log "Ollama recovered."
    fi
fi

# Check 2: Is the scheduler running?
if ! systemctl is-active --quiet kids-story-pipeline.service 2>/dev/null; then
    log "WARNING: Pipeline service is not running. Restarting..."
    sudo systemctl restart kids-story-pipeline.service
    sleep 5
    if systemctl is-active --quiet kids-story-pipeline.service 2>/dev/null; then
        log "Pipeline service recovered."
    else
        log "ERROR: Pipeline service failed to restart!"
        HEALTHY=false
    fi
fi

# Check 3: Disk space (warn if below 2GB)
AVAIL_KB=$(df "$PROJECT_DIR" --output=avail | tail -1 | tr -d ' ')
AVAIL_GB=$((AVAIL_KB / 1024 / 1024))
if [ "$AVAIL_GB" -lt 2 ]; then
    log "WARNING: Low disk space: ${AVAIL_GB}GB remaining"
    # Clean up old output files (keep last 7 days)
    find "$PROJECT_DIR/output" -type f -mtime +7 -delete 2>/dev/null || true
    log "Cleaned up output files older than 7 days."
fi

# Check 4: Clean up temp files
find /tmp -name "edge-tts-*" -mtime +1 -delete 2>/dev/null || true
find /tmp -name "story_*" -mtime +1 -delete 2>/dev/null || true

if $HEALTHY; then
    log "All services healthy."
else
    log "Some services have issues - check logs."
fi
