#!/bin/bash
# ============================================================
# Kids Story Channel - Install Systemd Services
# ============================================================
# Installs the pipeline as a system service that auto-starts
# on boot and recovers from crashes/reboots automatically.
# ============================================================

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "Installing Kids Story Channel systemd services..."

# Copy service files
sudo cp "$PROJECT_DIR/systemd/kids-story-pipeline.service" /etc/systemd/system/
sudo cp "$PROJECT_DIR/systemd/kids-story-healthcheck.service" /etc/systemd/system/
sudo cp "$PROJECT_DIR/systemd/kids-story-healthcheck.timer" /etc/systemd/system/

# Reload systemd
sudo systemctl daemon-reload

# Enable services (start on boot)
sudo systemctl enable kids-story-pipeline.service
sudo systemctl enable kids-story-healthcheck.timer

echo ""
echo "Services installed successfully!"
echo ""
echo "Commands:"
echo "  Start pipeline:    sudo systemctl start kids-story-pipeline"
echo "  Stop pipeline:     sudo systemctl stop kids-story-pipeline"
echo "  View status:       sudo systemctl status kids-story-pipeline"
echo "  View logs:         journalctl -u kids-story-pipeline -f"
echo "  View scheduler:    tail -f data/logs/scheduler.log"
echo ""
echo "The pipeline will auto-start on boot and restart on failures."
echo "Health checks run every 15 minutes."
