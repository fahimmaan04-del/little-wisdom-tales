#!/bin/bash
# Little Wisdom Tales - One-Time Channel Setup
# Creates Brand Account channels and authenticates OAuth tokens.
#
# Prerequisites:
#   1. One Google account with YouTube access
#   2. Google Cloud project with YouTube Data API v3 enabled
#   3. OAuth 2.0 Client ID (Desktop app) - set YOUTUBE_CLIENT_ID and YOUTUBE_CLIENT_SECRET in .env
#
# Run this script ONCE to set up all 21 channels.
# After setup, the pipeline runs fully autonomously.

set -euo pipefail

cd /mnt/projects/youtube
source venv/bin/activate

if [ -f .env ]; then
    set -a
    source .env
    set +a
fi

echo "=============================================="
echo "  Little Wisdom Tales - Channel Setup"
echo "  Total channels to configure: 21"
echo "=============================================="
echo ""

# Check prerequisites
if [ -z "${YOUTUBE_CLIENT_ID:-}" ] || [ -z "${YOUTUBE_CLIENT_SECRET:-}" ]; then
    echo "ERROR: YOUTUBE_CLIENT_ID and YOUTUBE_CLIENT_SECRET must be set in .env"
    echo ""
    echo "Steps to get these:"
    echo "  1. Go to https://console.cloud.google.com"
    echo "  2. Create a project (or use existing)"
    echo "  3. Enable 'YouTube Data API v3'"
    echo "  4. Create OAuth 2.0 Client ID (Desktop application)"
    echo "  5. Copy Client ID and Client Secret to .env file"
    echo ""
    exit 1
fi

echo "OAuth credentials found."
echo ""

# Get list of all channel keys from channels.json
CHANNEL_KEYS=$(python -c "
import json
with open('config/channels.json') as f:
    data = json.load(f)
for key in data['channels']:
    print(key)
")

TOTAL=$(echo "$CHANNEL_KEYS" | wc -l)
DONE=0
SKIPPED=0

echo "Found $TOTAL channels in config/channels.json"
echo ""
echo "IMPORTANT: Before running this script, create all Brand Account channels:"
echo "  1. Go to https://youtube.com/channel_switcher"
echo "  2. Click 'Create a channel' for each channel name below"
echo "  3. Then run this script to authenticate each one"
echo ""
echo "Channel names to create:"
python -c "
import json
with open('config/channels.json') as f:
    data = json.load(f)
for i, (key, ch) in enumerate(data['channels'].items(), 1):
    print(f'  {i:2d}. {ch[\"name\"]}')
"
echo ""
read -p "Have you created all Brand Account channels? (y/n) " -r
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Please create the channels first, then re-run this script."
    exit 0
fi

echo ""
echo "Starting OAuth authentication for each channel..."
echo "For each channel, a browser window will open. Sign in and authorize."
echo "Make sure to select the correct Brand Account channel each time."
echo ""

for CHANNEL_KEY in $CHANNEL_KEYS; do
    CHANNEL_NAME=$(python -c "
import json
with open('config/channels.json') as f:
    data = json.load(f)
print(data['channels']['$CHANNEL_KEY']['name'])
")
    TOKEN_FILE=$(python -c "
import json
with open('config/channels.json') as f:
    data = json.load(f)
print(data['channels']['$CHANNEL_KEY']['token_file'])
")

    # Check if token already exists and is valid
    if [ -f "data/$TOKEN_FILE" ]; then
        VALID=$(python -c "
import json
from google.oauth2.credentials import Credentials
try:
    creds = Credentials.from_authorized_user_file('data/$TOKEN_FILE')
    print('valid' if creds.valid or creds.refresh_token else 'invalid')
except:
    print('invalid')
" 2>/dev/null || echo "invalid")

        if [ "$VALID" = "valid" ]; then
            DONE=$((DONE + 1))
            SKIPPED=$((SKIPPED + 1))
            echo "[$DONE/$TOTAL] $CHANNEL_NAME - Already authenticated (skipping)"
            continue
        fi
    fi

    echo ""
    echo "=============================================="
    echo "[$((DONE + 1))/$TOTAL] Authenticating: $CHANNEL_NAME"
    echo "  Channel key: $CHANNEL_KEY"
    echo "  Token file: data/$TOKEN_FILE"
    echo "=============================================="
    echo ""
    echo "  >> SELECT the Brand Account '$CHANNEL_NAME' when prompted <<"
    echo ""

    python scripts/channel_manager.py auth "$CHANNEL_KEY"

    DONE=$((DONE + 1))
    echo ""
    echo "  Channel $DONE/$TOTAL authenticated."
    echo ""

    if [ "$DONE" -lt "$TOTAL" ]; then
        read -p "Press Enter to continue to next channel (or Ctrl+C to stop)..."
    fi
done

echo ""
echo "=============================================="
echo "  SETUP COMPLETE!"
echo "  Authenticated: $DONE channels"
echo "  Already set up: $SKIPPED channels"
echo "=============================================="
echo ""
echo "Next steps:"
echo "  1. Verify all channels: python scripts/channel_manager.py stats"
echo "  2. Start the pipeline:  sudo systemctl start kids-story-pipeline"
echo "  3. Enable auto-start:   sudo systemctl enable kids-story-pipeline"
echo "  4. Monitor logs:        journalctl -u kids-story-pipeline -f"
echo ""
echo "The pipeline will now run autonomously 24/7!"
