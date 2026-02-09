"""
YouTube Manager - Handles upload, analytics, and channel management.

Uses YouTube Data API v3 for:
- Video uploads with metadata (Made for Kids compliance)
- Analytics tracking for content optimization
- Thumbnail upload
- Playlist management
"""

import json
import os
import sqlite3
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

load_dotenv()

DATA_DIR = Path(os.getenv("DATA_DIR", "./data"))
SCOPES = [
    "https://www.googleapis.com/auth/youtube.upload",
    "https://www.googleapis.com/auth/youtube",
    "https://www.googleapis.com/auth/youtube.readonly",
    "https://www.googleapis.com/auth/yt-analytics.readonly",
]

TOKEN_PATH = DATA_DIR / "youtube_token.json"

# YouTube API quota costs (per request):
# video.insert = 1600 units
# video.list = 1 unit
# channel.list = 1 unit
# Daily quota = 10,000 units
# That means ~6 uploads/day max


def get_credentials() -> Credentials:
    """Get or refresh YouTube API credentials using JSON token storage."""
    creds = None

    if TOKEN_PATH.exists():
        creds = Credentials.from_authorized_user_file(str(TOKEN_PATH), SCOPES)

    if creds and creds.expired and creds.refresh_token:
        creds.refresh(Request())
        _save_credentials(creds)
        return creds

    if creds and creds.valid:
        return creds

    # Need to authenticate - this requires user interaction ONCE
    client_config = {
        "installed": {
            "client_id": os.getenv("YOUTUBE_CLIENT_ID"),
            "client_secret": os.getenv("YOUTUBE_CLIENT_SECRET"),
            "redirect_uris": ["urn:ietf:wg:oauth:2.0:oob", "http://localhost:8080"],
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
        }
    }

    flow = InstalledAppFlow.from_client_config(client_config, SCOPES)
    creds = flow.run_local_server(port=8080, open_browser=False)
    _save_credentials(creds)

    return creds


def _save_credentials(creds: Credentials):
    """Save credentials to JSON file."""
    TOKEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    token_data = {
        "token": creds.token,
        "refresh_token": creds.refresh_token,
        "token_uri": creds.token_uri,
        "client_id": creds.client_id,
        "client_secret": creds.client_secret,
        "scopes": list(creds.scopes) if creds.scopes else SCOPES,
    }
    with open(TOKEN_PATH, "w") as f:
        json.dump(token_data, f, indent=2)


def get_youtube_service():
    """Get authenticated YouTube API service."""
    creds = get_credentials()
    return build("youtube", "v3", credentials=creds)


def upload_video(
    video_path: str,
    title: str,
    description: str,
    tags: list,
    category_id: str = "24",  # Entertainment; use "1" for Film, "22" for People
    is_shorts: bool = False,
    thumbnail_path: str = None,
    made_for_kids: bool = True,
) -> dict:
    """Upload a video to YouTube with Made for Kids compliance."""
    youtube = get_youtube_service()

    # For Shorts, prepend #Shorts to title
    if is_shorts and "#Shorts" not in title:
        title = f"{title} #Shorts"

    # Ensure tags include required ones
    base_tags = [
        "kids stories", "moral stories", "bedtime stories",
        "stories for children", "kids", "children",
    ]
    all_tags = list(set(tags + base_tags))

    # Video metadata
    body = {
        "snippet": {
            "title": title[:100],  # YouTube max title length
            "description": description[:5000],
            "tags": all_tags[:30],  # YouTube max 30 tags
            "categoryId": category_id,
            "defaultLanguage": "en",
        },
        "status": {
            "privacyStatus": "public",
            "selfDeclaredMadeForKids": made_for_kids,
            "license": "youtube",
            "embeddable": True,
            "publicStatsViewable": True,
        },
    }

    # Upload the video
    media = MediaFileUpload(
        video_path,
        mimetype="video/mp4",
        resumable=True,
        chunksize=1024 * 1024 * 10,  # 10MB chunks
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
                print(f"  Upload progress: {int(status.progress() * 100)}%")
        except Exception as e:
            retry_count += 1
            if retry_count > 5:
                raise
            print(f"  Upload chunk failed (attempt {retry_count}): {e}, retrying...")
            import time as _time
            _time.sleep(5 * retry_count)

    video_id = response["id"]
    print(f"  Video uploaded: https://youtube.com/watch?v={video_id}")

    # Upload thumbnail if provided
    if thumbnail_path and os.path.exists(thumbnail_path):
        try:
            youtube.thumbnails().set(
                videoId=video_id,
                media_body=MediaFileUpload(thumbnail_path, mimetype="image/jpeg"),
            ).execute()
            print(f"  Thumbnail set for video {video_id}")
        except Exception as e:
            # Thumbnail upload may fail for unverified channels
            print(f"  Thumbnail upload failed (channel may need verification): {e}")

    return {
        "video_id": video_id,
        "url": f"https://youtube.com/watch?v={video_id}",
        "title": title,
        "uploaded_at": datetime.now().isoformat(),
    }


def get_channel_stats() -> dict:
    """Get current channel statistics."""
    youtube = get_youtube_service()

    response = youtube.channels().list(
        part="statistics,snippet",
        mine=True,
    ).execute()

    if not response.get("items"):
        return {"error": "No channel found"}

    channel = response["items"][0]
    return {
        "channel_id": channel["id"],
        "title": channel["snippet"]["title"],
        "subscribers": int(channel["statistics"].get("subscriberCount", 0)),
        "total_views": int(channel["statistics"].get("viewCount", 0)),
        "video_count": int(channel["statistics"].get("videoCount", 0)),
    }


def get_video_analytics(video_id: str) -> dict:
    """Get analytics for a specific video."""
    youtube = get_youtube_service()

    response = youtube.videos().list(
        part="statistics",
        id=video_id,
    ).execute()

    if not response.get("items"):
        return {"error": "Video not found"}

    stats = response["items"][0]["statistics"]
    return {
        "video_id": video_id,
        "views": int(stats.get("viewCount", 0)),
        "likes": int(stats.get("likeCount", 0)),
        "comments": int(stats.get("commentCount", 0)),
        "favorites": int(stats.get("favoriteCount", 0)),
    }


def get_recent_videos(max_results: int = 10) -> list:
    """Get list of recently uploaded videos."""
    youtube = get_youtube_service()

    # Get uploads playlist
    channel_response = youtube.channels().list(
        part="contentDetails",
        mine=True,
    ).execute()

    if not channel_response.get("items"):
        return []

    uploads_id = channel_response["items"][0]["contentDetails"]["relatedPlaylists"]["uploads"]

    # Get videos from uploads playlist
    response = youtube.playlistItems().list(
        part="snippet,contentDetails",
        playlistId=uploads_id,
        maxResults=max_results,
    ).execute()

    videos = []
    for item in response.get("items", []):
        videos.append({
            "video_id": item["contentDetails"]["videoId"],
            "title": item["snippet"]["title"],
            "published_at": item["snippet"]["publishedAt"],
            "thumbnail": item["snippet"]["thumbnails"].get("medium", {}).get("url"),
        })

    return videos


def update_analytics_db(story_db_path: str = None):
    """Update the analytics database with latest video performance data."""
    if story_db_path is None:
        story_db_path = str(DATA_DIR / "stories.db")

    conn = sqlite3.connect(story_db_path)
    conn.row_factory = sqlite3.Row

    # Get all published stories with video IDs
    published = conn.execute(
        "SELECT id, video_id FROM stories WHERE status = 'published' AND video_id IS NOT NULL"
    ).fetchall()

    today = datetime.now().strftime("%Y-%m-%d")

    for story in published:
        try:
            analytics = get_video_analytics(story["video_id"])
            if "error" not in analytics:
                conn.execute("""
                    INSERT INTO analytics (story_id, date, views, likes)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT DO UPDATE SET
                        views = excluded.views,
                        likes = excluded.likes
                """, (
                    story["id"],
                    today,
                    analytics.get("views", 0),
                    analytics.get("likes", 0),
                ))

                conn.execute(
                    "UPDATE stories SET views = ? WHERE id = ?",
                    (analytics.get("views", 0), story["id"]),
                )
        except Exception as e:
            print(f"  Failed to get analytics for story {story['id']}: {e}")

    conn.commit()
    conn.close()
    print(f"  Updated analytics for {len(published)} videos")


def mark_story_published(story_id: int, video_id: str, db_path: str = None):
    """Mark a story as published in the database."""
    if db_path is None:
        db_path = str(DATA_DIR / "stories.db")

    conn = sqlite3.connect(db_path)
    conn.execute(
        "UPDATE stories SET status = 'published', video_id = ?, published_at = ? WHERE id = ?",
        (video_id, datetime.now().isoformat(), story_id),
    )
    conn.commit()
    conn.close()


def create_or_get_playlist(title: str, description: str = "") -> str:
    """Create a playlist or get existing one by title."""
    youtube = get_youtube_service()

    response = youtube.playlists().list(
        part="snippet",
        mine=True,
        maxResults=50,
    ).execute()

    for item in response.get("items", []):
        if item["snippet"]["title"] == title:
            return item["id"]

    response = youtube.playlists().insert(
        part="snippet,status",
        body={
            "snippet": {
                "title": title,
                "description": description,
            },
            "status": {
                "privacyStatus": "public",
            },
        },
    ).execute()

    return response["id"]


def add_to_playlist(playlist_id: str, video_id: str):
    """Add a video to a playlist."""
    youtube = get_youtube_service()

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


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        cmd = sys.argv[1]
        if cmd == "auth":
            print("Starting YouTube authentication...")
            creds = get_credentials()
            print("Authentication successful!")
            stats = get_channel_stats()
            print(json.dumps(stats, indent=2))
        elif cmd == "stats":
            stats = get_channel_stats()
            print(json.dumps(stats, indent=2))
        elif cmd == "analytics":
            update_analytics_db()
        elif cmd == "recent":
            videos = get_recent_videos()
            print(json.dumps(videos, indent=2))
    else:
        print("Usage: python youtube_manager.py [auth|stats|analytics|recent]")
