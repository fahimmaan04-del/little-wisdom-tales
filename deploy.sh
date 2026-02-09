#!/bin/bash
set -euo pipefail

# ============================================================
# Kids Story Channel - Full Deployment Script
# ============================================================
# This script sets up everything on a fresh Ubuntu server.
# Designed for Oracle Cloud Always Free Tier (ARM, 4 cores, 24GB RAM)
# but works on any Ubuntu 22.04+ system.
# ============================================================

echo "============================================"
echo "  Kids Story Channel - Deployment"
echo "============================================"
echo ""

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

# --- Step 1: System Dependencies ---
echo "[1/7] Installing system dependencies..."
sudo apt-get update -qq
sudo apt-get install -y -qq \
    docker.io \
    docker-compose \
    python3 \
    python3-pip \
    python3-venv \
    ffmpeg \
    fonts-dejavu-core \
    fonts-noto \
    curl \
    git \
    jq

# Enable Docker for current user
sudo usermod -aG docker "$USER" || true
sudo systemctl enable docker
sudo systemctl start docker

echo "  System dependencies installed."

# --- Step 2: Python Environment ---
echo "[2/7] Setting up Python environment..."
python3 -m venv venv
source venv/bin/activate
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt
echo "  Python environment ready."

# --- Step 3: Configuration ---
echo "[3/7] Setting up configuration..."
if [ ! -f .env ]; then
    cp .env.example .env
    echo ""
    echo "  ================================================"
    echo "  IMPORTANT: You need to configure .env file!"
    echo "  ================================================"
    echo "  Required API keys:"
    echo "    1. YouTube API (Google Cloud Console)"
    echo "    2. Pixabay API (free at pixabay.com/api/docs)"
    echo "  ================================================"
    echo ""
fi

# Create required directories
mkdir -p output/{videos,audio,images,thumbnails} data/logs assets/music

echo "  Configuration ready."

# --- Step 4: Pull Ollama Model ---
echo "[4/7] Setting up Ollama LLM..."
sudo docker-compose up -d ollama
echo "  Waiting for Ollama to start..."
sleep 10

# Pull a small model suitable for story generation
sudo docker-compose exec -T ollama ollama pull llama3.2:3b 2>/dev/null || \
    echo "  Note: Model pull may take a few minutes on first run."

echo "  Ollama ready."

# --- Step 5: YouTube Authentication ---
echo "[5/7] YouTube authentication setup..."
if [ ! -f data/youtube_token.json ]; then
    echo ""
    echo "  ================================================"
    echo "  YouTube Authentication Required (ONE TIME ONLY)"
    echo "  ================================================"
    echo "  Before running this, you need:"
    echo ""
    echo "  1. Go to https://console.cloud.google.com"
    echo "  2. Create a new project (or select existing)"
    echo "  3. Enable 'YouTube Data API v3'"
    echo "  4. Create OAuth 2.0 credentials (Desktop app)"
    echo "  5. Download credentials and set in .env:"
    echo "     YOUTUBE_CLIENT_ID=..."
    echo "     YOUTUBE_CLIENT_SECRET=..."
    echo ""
    echo "  Then run: python scripts/youtube_manager.py auth"
    echo "  (This opens a browser for one-time Google login)"
    echo "  ================================================"
    echo ""
fi

# --- Step 6: Build & Start Services ---
echo "[6/7] Building and starting all services..."
sudo docker-compose build
sudo docker-compose up -d

echo "  All services started."

# --- Step 7: Setup Systemd Auto-Start ---
echo "[7/7] Setting up auto-start on boot..."

COMPOSE_PATH=$(which docker-compose || echo "/usr/bin/docker-compose")

sudo tee /etc/systemd/system/kids-story-channel.service > /dev/null << SERVICEEOF
[Unit]
Description=Kids Story Channel Pipeline
After=docker.service
Requires=docker.service

[Service]
Type=simple
WorkingDirectory=$PROJECT_DIR
ExecStart=$COMPOSE_PATH up
ExecStop=$COMPOSE_PATH down
Restart=always
RestartSec=30
User=$USER

[Install]
WantedBy=multi-user.target
SERVICEEOF

sudo systemctl daemon-reload
sudo systemctl enable kids-story-channel.service

echo ""
echo "============================================"
echo "  Deployment Complete!"
echo "============================================"
echo ""
echo "  Services running:"
echo "    - n8n:       http://localhost:5678"
echo "    - Ollama:    http://localhost:11434"
echo "    - Scheduler: Running (creates 3 stories/day)"
echo ""
echo "  NEXT STEPS:"
echo "    1. Edit .env with your API keys"
echo "    2. Run: python scripts/youtube_manager.py auth"
echo "    3. The pipeline will start automatically!"
echo ""
echo "  Monitoring:"
echo "    - Logs: data/logs/orchestrator.log"
echo "    - n8n UI: http://YOUR_IP:5678"
echo "    - Status: sudo docker-compose ps"
echo ""
echo "============================================"
