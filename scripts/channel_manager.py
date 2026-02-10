"""
Channel Manager - Multi-channel YouTube management for story + education content.

Manages multiple YouTube channels (stories, Oxford education, Cambridge education),
each with their own OAuth credentials, playlists, and upload quotas.
Routes content to the correct channel based on content type and curriculum.
"""

import json
import logging
import os
import sqlite3
from datetime import datetime, date
from pathlib import Path

from dotenv import load_dotenv
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

load_dotenv()

DATA_DIR = Path(os.getenv("DATA_DIR", "./data"))
CONFIG_DIR = Path("./config")

logger = logging.getLogger(__name__)

SCOPES = [
    "https://www.googleapis.com/auth/youtube.upload",
    "https://www.googleapis.com/auth/youtube",
    "https://www.googleapis.com/auth/youtube.readonly",
    "https://www.googleapis.com/auth/yt-analytics.readonly",
]

# Quota tracking: each upload = 1600 units, daily limit = 10,000
UPLOAD_QUOTA_COST = 1600
DAILY_QUOTA_LIMIT = 10000


def load_channel_config() -> dict:
    """Load multi-channel configuration."""
    config_path = CONFIG_DIR / "channels.json"
    with open(config_path) as f:
        return json.load(f)


def get_db():
    """Get or create the channel management database."""
    db_path = DATA_DIR / "channels.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    conn.execute("""
        CREATE TABLE IF NOT EXISTS channel_uploads (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            channel_key TEXT NOT NULL,
            video_id TEXT,
            title TEXT,
            content_type TEXT,
            upload_date TEXT,
            quota_used INTEGER DEFAULT 1600,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS playlist_cache (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            channel_key TEXT NOT NULL,
            playlist_key TEXT NOT NULL,
            playlist_id TEXT NOT NULL,
            playlist_title TEXT,
            UNIQUE(channel_key, playlist_key)
        )
    """)

    conn.commit()
    return conn


def get_channel_credentials(channel_key: str) -> Credentials:
    """Get OAuth credentials for a specific channel."""
    config = load_channel_config()
    channel = config["channels"].get(channel_key)
    if not channel:
        raise ValueError(f"Unknown channel: {channel_key}")

    token_file = channel["token_file"]
    token_path = DATA_DIR / token_file

    creds = None
    if token_path.exists():
        creds = Credentials.from_authorized_user_file(str(token_path), SCOPES)

    if creds and creds.expired and creds.refresh_token:
        creds.refresh(Request())
        _save_channel_credentials(channel_key, creds)
        return creds

    if creds and creds.valid:
        return creds

    # Need interactive auth - guide user
    client_config = {
        "installed": {
            "client_id": os.getenv("YOUTUBE_CLIENT_ID"),
            "client_secret": os.getenv("YOUTUBE_CLIENT_SECRET"),
            "redirect_uris": ["urn:ietf:wg:oauth:2.0:oob", "http://localhost:8080"],
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
        }
    }

    logger.info(f"Authentication required for channel: {channel['name']}")
    logger.info(f"Please sign in with the Google account for: {channel['name']}")

    flow = InstalledAppFlow.from_client_config(client_config, SCOPES)
    creds = flow.run_local_server(port=8080, open_browser=False)
    _save_channel_credentials(channel_key, creds)

    return creds


def _save_channel_credentials(channel_key: str, creds: Credentials):
    """Save channel credentials to its token file."""
    config = load_channel_config()
    channel = config["channels"][channel_key]
    token_path = DATA_DIR / channel["token_file"]
    token_path.parent.mkdir(parents=True, exist_ok=True)

    token_data = {
        "token": creds.token,
        "refresh_token": creds.refresh_token,
        "token_uri": creds.token_uri,
        "client_id": creds.client_id,
        "client_secret": creds.client_secret,
        "scopes": list(creds.scopes) if creds.scopes else SCOPES,
    }
    with open(token_path, "w") as f:
        json.dump(token_data, f, indent=2)


def get_youtube_service(channel_key: str):
    """Get authenticated YouTube API service for a channel."""
    creds = get_channel_credentials(channel_key)
    return build("youtube", "v3", credentials=creds)


def get_analytics_service(channel_key: str):
    """Get authenticated YouTube Analytics API service for a channel."""
    creds = get_channel_credentials(channel_key)
    return build("youtubeAnalytics", "v2", credentials=creds)


def get_daily_upload_count(channel_key: str) -> int:
    """Get how many videos have been uploaded today for a channel."""
    db = get_db()
    today = date.today().isoformat()
    row = db.execute(
        "SELECT COUNT(*) as cnt FROM channel_uploads WHERE channel_key = ? AND upload_date = ?",
        (channel_key, today),
    ).fetchone()
    db.close()
    return row["cnt"] if row else 0


def get_remaining_quota(channel_key: str) -> int:
    """Get remaining upload quota for a channel today."""
    db = get_db()
    today = date.today().isoformat()
    row = db.execute(
        "SELECT COALESCE(SUM(quota_used), 0) as total FROM channel_uploads WHERE channel_key = ? AND upload_date = ?",
        (channel_key, today),
    ).fetchone()
    db.close()
    used = row["total"] if row else 0
    return max(0, DAILY_QUOTA_LIMIT - used)


def can_upload(channel_key: str) -> bool:
    """Check if a channel has enough quota for another upload."""
    return get_remaining_quota(channel_key) >= UPLOAD_QUOTA_COST


def upload_to_channel(
    channel_key: str,
    video_path: str,
    title: str,
    description: str,
    tags: list,
    content_type: str = "story",
    thumbnail_path: str = None,
    made_for_kids: bool = True,
) -> dict:
    """Upload a video to a specific channel."""
    if not can_upload(channel_key):
        remaining = get_remaining_quota(channel_key)
        raise ValueError(
            f"Channel '{channel_key}' quota exceeded. "
            f"Remaining: {remaining} units, need {UPLOAD_QUOTA_COST}"
        )

    youtube = get_youtube_service(channel_key)

    # Determine language and category from channel config
    config = load_channel_config()
    channel_cfg = config["channels"].get(channel_key, {})
    channel_lang = channel_cfg.get("language", "en")

    # YouTube language codes mapping
    yt_lang_map = {
        "en": "en", "hi": "hi", "es": "es",
        "fr": "fr", "pt": "pt", "ar": "ar",
    }
    yt_lang = yt_lang_map.get(channel_lang, "en")

    # Category: 27=Education, 24=Entertainment, 1=Film & Animation
    if content_type in ("education", "ai_education", "crafts_skills"):
        category_id = "27"  # Education category
    else:
        category_id = "24"  # Entertainment category

    body = {
        "snippet": {
            "title": title[:100],
            "description": description[:5000],
            "tags": list(set(tags))[:30],
            "categoryId": category_id,
            "defaultLanguage": yt_lang,
            "defaultAudioLanguage": yt_lang,
        },
        "status": {
            "privacyStatus": "public",
            "selfDeclaredMadeForKids": made_for_kids,
            "license": "youtube",
            "embeddable": True,
            "publicStatsViewable": True,
        },
    }

    media = MediaFileUpload(
        video_path,
        mimetype="video/mp4",
        resumable=True,
        chunksize=1024 * 1024 * 10,
    )

    request = youtube.videos().insert(
        part="snippet,status",
        body=body,
        media_body=media,
    )

    response = None
    retry_count = 0
    while response is None:
        try:
            status, response = request.next_chunk()
            if status:
                logger.info(f"  Upload progress: {int(status.progress() * 100)}%")
        except Exception as e:
            retry_count += 1
            if retry_count > 5:
                raise
            logger.warning(f"  Upload chunk failed (attempt {retry_count}): {e}")
            import time
            time.sleep(5 * retry_count)

    video_id = response["id"]

    # Upload thumbnail
    if thumbnail_path and os.path.exists(thumbnail_path):
        try:
            youtube.thumbnails().set(
                videoId=video_id,
                media_body=MediaFileUpload(thumbnail_path, mimetype="image/jpeg"),
            ).execute()
        except Exception as e:
            logger.warning(f"  Thumbnail upload failed: {e}")

    # Track upload in database
    db = get_db()
    db.execute(
        """INSERT INTO channel_uploads (channel_key, video_id, title, content_type, upload_date, quota_used)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (channel_key, video_id, title, content_type, date.today().isoformat(), UPLOAD_QUOTA_COST),
    )
    db.commit()
    db.close()

    logger.info(f"  Uploaded to {channel_key}: https://youtube.com/watch?v={video_id}")

    return {
        "video_id": video_id,
        "url": f"https://youtube.com/watch?v={video_id}",
        "title": title,
        "channel_key": channel_key,
        "uploaded_at": datetime.now().isoformat(),
    }


def get_or_create_playlist(channel_key: str, playlist_key: str) -> str:
    """Get or create a playlist on a specific channel. Caches playlist IDs."""
    db = get_db()
    cached = db.execute(
        "SELECT playlist_id FROM playlist_cache WHERE channel_key = ? AND playlist_key = ?",
        (channel_key, playlist_key),
    ).fetchone()

    if cached:
        db.close()
        return cached["playlist_id"]

    # Look up playlist title from config
    config = load_channel_config()
    channel = config["channels"].get(channel_key, {})
    playlist_title = channel.get("playlists", {}).get(playlist_key, playlist_key.replace("_", " ").title())

    youtube = get_youtube_service(channel_key)

    # Check if playlist already exists on YouTube
    response = youtube.playlists().list(
        part="snippet",
        mine=True,
        maxResults=50,
    ).execute()

    for item in response.get("items", []):
        if item["snippet"]["title"] == playlist_title:
            playlist_id = item["id"]
            db.execute(
                "INSERT OR REPLACE INTO playlist_cache (channel_key, playlist_key, playlist_id, playlist_title) VALUES (?, ?, ?, ?)",
                (channel_key, playlist_key, playlist_id, playlist_title),
            )
            db.commit()
            db.close()
            return playlist_id

    # Create new playlist
    response = youtube.playlists().insert(
        part="snippet,status",
        body={
            "snippet": {
                "title": playlist_title,
                "description": f"Educational content - {playlist_title}",
            },
            "status": {"privacyStatus": "public"},
        },
    ).execute()

    playlist_id = response["id"]
    db.execute(
        "INSERT OR REPLACE INTO playlist_cache (channel_key, playlist_key, playlist_id, playlist_title) VALUES (?, ?, ?, ?)",
        (channel_key, playlist_key, playlist_id, playlist_title),
    )
    db.commit()
    db.close()

    logger.info(f"  Created playlist '{playlist_title}' on {channel_key}")
    return playlist_id


def add_video_to_playlist(channel_key: str, playlist_id: str, video_id: str):
    """Add a video to a playlist on a specific channel."""
    youtube = get_youtube_service(channel_key)
    youtube.playlistItems().insert(
        part="snippet",
        body={
            "snippet": {
                "playlistId": playlist_id,
                "resourceId": {
                    "kind": "youtube#video",
                    "videoId": video_id,
                },
            }
        },
    ).execute()


def get_all_channel_stats() -> dict:
    """Get upload stats across all channels for today."""
    config = load_channel_config()
    stats = {}
    for key, channel in config["channels"].items():
        count = get_daily_upload_count(key)
        remaining = get_remaining_quota(key)
        stats[key] = {
            "name": channel["name"],
            "uploads_today": count,
            "quota_remaining": remaining,
            "max_more_uploads": remaining // UPLOAD_QUOTA_COST,
            "daily_target": channel["daily_target"],
        }
    return stats


def pick_best_channel(
    content_type: str,
    curriculum: str = None,
    language: str = None,
) -> str:
    """Pick the channel with the most remaining quota for the content type."""
    config = load_channel_config()

    candidates = []
    for key, channel in config["channels"].items():
        # Match content type
        if content_type == "story" and channel["type"] != "stories":
            continue
        elif content_type == "education" and channel["type"] != "education":
            continue
        elif content_type == "ai_education" and channel["type"] != "ai_education":
            continue
        elif content_type == "crafts_skills" and channel["type"] != "crafts_skills":
            continue

        # Match curriculum for education
        if content_type == "education" and curriculum:
            if channel.get("curriculum") != curriculum:
                continue

        # Match language if specified
        if language and channel.get("language", "en") != language:
            continue

        candidates.append(key)

    if not candidates:
        raise ValueError(f"No channel found for {content_type}/{curriculum}/{language}")

    # Pick channel with most remaining quota
    best = max(candidates, key=lambda k: get_remaining_quota(k))
    return best


def setup_channel_auth(channel_key: str):
    """Interactive setup for channel authentication."""
    config = load_channel_config()
    channel = config["channels"].get(channel_key)
    if not channel:
        print(f"Unknown channel: {channel_key}")
        print(f"Available: {', '.join(config['channels'].keys())}")
        return

    print(f"\n=== Setting up: {channel['name']} ===")
    print(f"Sign in with the Google account that owns this channel.")
    print(f"Token will be saved to: {DATA_DIR / channel['token_file']}")

    creds = get_channel_credentials(channel_key)
    youtube = build("youtube", "v3", credentials=creds)

    response = youtube.channels().list(part="snippet,statistics", mine=True).execute()
    if response.get("items"):
        ch = response["items"][0]
        print(f"\nAuthenticated successfully!")
        print(f"  Channel: {ch['snippet']['title']}")
        print(f"  ID: {ch['id']}")
        print(f"  Subscribers: {ch['statistics'].get('subscriberCount', 0)}")

        # Update config with channel ID if missing
        if not channel.get("channel_id"):
            config["channels"][channel_key]["channel_id"] = ch["id"]
            with open(CONFIG_DIR / "channels.json", "w") as f:
                json.dump(config, f, indent=2)
            print(f"  Channel ID saved to config")
    else:
        print("Warning: Could not fetch channel info")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        cmd = sys.argv[1]
        if cmd == "auth":
            channel_key = sys.argv[2] if len(sys.argv) > 2 else "stories"
            setup_channel_auth(channel_key)
        elif cmd == "stats":
            stats = get_all_channel_stats()
            print(json.dumps(stats, indent=2))
        elif cmd == "list":
            config = load_channel_config()
            for key, ch in config["channels"].items():
                print(f"  {key}: {ch['name']} (target: {ch['daily_target']}/day)")
    else:
        print("Usage: python channel_manager.py [auth <channel>|stats|list]")
