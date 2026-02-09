#!/bin/bash
# ============================================================
# Kids Story Channel - Pipeline Startup Script
# ============================================================
# This script ensures all services are running and starts
# the scheduler. Designed to be called by systemd on boot.
# ============================================================

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

LOG_DIR="$PROJECT_DIR/data/logs"
mkdir -p "$LOG_DIR"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_DIR/startup.log"
}

log "============================================"
log "  Pipeline Startup Script"
log "============================================"

# --- Step 1: Ensure Ollama is running ---
log "Checking Ollama..."
if ! pgrep -x "ollama" > /dev/null 2>&1; then
    log "Starting Ollama..."
    ollama serve &
    sleep 5
fi

# Wait for Ollama to be ready (max 60 seconds)
for i in $(seq 1 12); do
    if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        log "Ollama is ready."
        break
    fi
    log "Waiting for Ollama... ($i/12)"
    sleep 5
done

# Verify model is available
if ! ollama list 2>/dev/null | grep -q "llama3.2:3b"; then
    log "Pulling llama3.2:3b model..."
    ollama pull llama3.2:3b
fi

# --- Step 2: Activate Python venv ---
log "Activating Python environment..."
if [ -d "$PROJECT_DIR/venv" ]; then
    source "$PROJECT_DIR/venv/bin/activate"
else
    log "Creating Python venv..."
    python3 -m venv "$PROJECT_DIR/venv"
    source "$PROJECT_DIR/venv/bin/activate"
    pip install --quiet -r "$PROJECT_DIR/requirements.txt"
fi

# --- Step 3: Create required directories ---
mkdir -p output/{videos,audio,images,thumbnails} data/logs assets/music

# --- Step 4: Start the scheduler ---
log "Starting scheduler..."
cd "$PROJECT_DIR"
exec python3 -m scripts.scheduler
