"""
Quick Setup - Interactive channel authentication helper.

Walks through all 21 channels one-by-one, launching the OAuth flow for each.
You just need to:
1. Select the correct Brand Account in the browser
2. Click "Allow"
3. Come back to this terminal

Usage:
    python scripts/quick_setup.py           # Auth all unauthenticated channels
    python scripts/quick_setup.py --list    # List all channels and auth status
    python scripts/quick_setup.py --test    # Test all authenticated channels
"""

import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

DATA_DIR = Path(os.getenv("DATA_DIR", "./data"))
CONFIG_DIR = Path("./config")


def load_channels():
    with open(CONFIG_DIR / "channels.json") as f:
        return json.load(f)


def check_token(token_file: str) -> dict:
    """Check if a token file exists and has a refresh token."""
    token_path = DATA_DIR / token_file
    if not token_path.exists():
        return {"exists": False, "valid": False, "path": str(token_path)}

    try:
        with open(token_path) as f:
            data = json.load(f)
        has_refresh = bool(data.get("refresh_token"))
        return {"exists": True, "valid": has_refresh, "path": str(token_path)}
    except Exception:
        return {"exists": True, "valid": False, "path": str(token_path)}


def list_channels():
    """List all channels with their authentication status."""
    config = load_channels()
    channels = config["channels"]

    print(f"\n{'='*70}")
    print(f"  Little Wisdom Tales - Channel Authentication Status")
    print(f"  Total channels: {len(channels)}")
    print(f"{'='*70}\n")

    authenticated = 0
    unauthenticated = 0

    for i, (key, ch) in enumerate(channels.items(), 1):
        token_info = check_token(ch["token_file"])
        if token_info["valid"]:
            status = "AUTHENTICATED"
            authenticated += 1
        elif token_info["exists"]:
            status = "TOKEN INVALID"
            unauthenticated += 1
        else:
            status = "NOT SET UP"
            unauthenticated += 1

        icon = "+" if token_info["valid"] else "-"
        print(f"  [{icon}] {i:2d}. {key}")
        print(f"       Name: {ch['name']}")
        print(f"       Type: {ch['type']} | Lang: {ch.get('language', 'en')} | Target: {ch['daily_target']}/day")
        print(f"       Auth: {status}")
        print()

    print(f"{'='*70}")
    print(f"  Authenticated: {authenticated}/{len(channels)}")
    print(f"  Need setup:    {unauthenticated}/{len(channels)}")
    print(f"{'='*70}\n")

    return authenticated, unauthenticated


def test_channels():
    """Test all authenticated channels by making an API call."""
    from scripts.channel_manager import get_youtube_service

    config = load_channels()
    channels = config["channels"]

    print(f"\n{'='*70}")
    print(f"  Testing YouTube API access for all channels...")
    print(f"{'='*70}\n")

    for key, ch in channels.items():
        token_info = check_token(ch["token_file"])
        if not token_info["valid"]:
            print(f"  [-] {key}: Skipped (not authenticated)")
            continue

        try:
            youtube = get_youtube_service(key)
            response = youtube.channels().list(part="snippet,statistics", mine=True).execute()
            if response.get("items"):
                item = response["items"][0]
                subs = item["statistics"].get("subscriberCount", "0")
                videos = item["statistics"].get("videoCount", "0")
                print(f"  [+] {key}: {item['snippet']['title']} "
                      f"({subs} subs, {videos} videos)")
            else:
                print(f"  [?] {key}: Authenticated but no channel data returned")
        except Exception as e:
            print(f"  [!] {key}: API error - {e}")

    print(f"\n{'='*70}\n")


def setup_channels():
    """Interactive setup - authenticate all unauthenticated channels."""
    # Check prerequisites
    client_id = os.getenv("YOUTUBE_CLIENT_ID")
    client_secret = os.getenv("YOUTUBE_CLIENT_SECRET")

    if not client_id or not client_secret:
        print("\nERROR: YouTube OAuth credentials not found!")
        print()
        print("You need to set up a Google Cloud project first:")
        print("  1. Go to https://console.cloud.google.com")
        print("  2. Create a project (or use existing)")
        print("  3. Go to 'APIs & Services' > 'Library'")
        print("  4. Search for 'YouTube Data API v3' and enable it")
        print("  5. Go to 'APIs & Services' > 'Credentials'")
        print("  6. Click 'Create Credentials' > 'OAuth client ID'")
        print("  7. Choose 'Desktop app' as application type")
        print("  8. Copy the Client ID and Client Secret")
        print("  9. Add to your .env file:")
        print("     YOUTUBE_CLIENT_ID=your_client_id_here")
        print("     YOUTUBE_CLIENT_SECRET=your_client_secret_here")
        print()
        return

    config = load_channels()
    channels = config["channels"]

    # Find unauthenticated channels
    to_setup = []
    for key, ch in channels.items():
        token_info = check_token(ch["token_file"])
        if not token_info["valid"]:
            to_setup.append((key, ch))

    if not to_setup:
        print("\nAll channels are already authenticated!")
        print("Run with --test to verify API access.\n")
        return

    print(f"\n{'='*70}")
    print(f"  Channel Authentication Setup")
    print(f"  {len(to_setup)} channels need authentication")
    print(f"{'='*70}")
    print()
    print("BEFORE starting, make sure you have created all Brand Account channels:")
    print("  1. Go to https://youtube.com/channel_switcher")
    print("  2. Click 'Create a channel' for each name listed below:")
    print()

    for i, (key, ch) in enumerate(to_setup, 1):
        print(f"    {i}. {ch['name']}")
    print()

    input("Press Enter when all Brand Account channels are created...")
    print()

    from scripts.channel_manager import setup_channel_auth

    for i, (key, ch) in enumerate(to_setup, 1):
        print(f"\n{'='*70}")
        print(f"  [{i}/{len(to_setup)}] {ch['name']}")
        print(f"  Channel key: {key}")
        print(f"  Type: {ch['type']} | Language: {ch.get('language', 'en')}")
        print(f"{'='*70}")
        print()
        print(f"  A browser window will open.")
        print(f"  >> Select the Brand Account: '{ch['name']}' <<")
        print(f"  >> Then click 'Allow' <<")
        print()

        try:
            setup_channel_auth(key)
            print(f"\n  Channel '{key}' authenticated successfully!")
        except Exception as e:
            print(f"\n  ERROR: {e}")
            print(f"  You can retry later: python scripts/channel_manager.py auth {key}")

        if i < len(to_setup):
            print()
            input("  Press Enter to continue to the next channel...")

    print(f"\n{'='*70}")
    print(f"  SETUP COMPLETE!")
    print(f"{'='*70}")
    print()
    print("  Next steps:")
    print("    python scripts/quick_setup.py --test    # Verify all channels")
    print("    sudo systemctl start kids-story-pipeline # Start autonomous pipeline")
    print()


if __name__ == "__main__":
    if "--list" in sys.argv:
        list_channels()
    elif "--test" in sys.argv:
        test_channels()
    else:
        setup_channels()
