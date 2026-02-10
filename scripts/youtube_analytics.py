"""
YouTube Analytics - Real analytics data from YouTube Analytics API v2.

Fetches performance metrics (views, watchTime, CTR, subscribersGained) and
search term data for all channels. Stores results in SQLite for use by
the keyword optimizer and content strategy engine.

Metrics collected per video:
  - views, estimatedMinutesWatched, averageViewDuration
  - impressions, impressionClickThroughRate (CTR)
  - subscribersGained, subscribersLost
  - likes, dislikes, shares, comments

Search terms:
  - Actual YouTube search queries that led viewers to our videos
  - Extracted via insightTrafficSourceDetail dimension with YT_SEARCH filter

Database tables:
  - video_analytics: Per-video daily metrics snapshot
  - search_terms: Search queries driving traffic to our content
  - channel_analytics: Per-channel daily aggregate metrics
"""

import json
import logging
import os
import sqlite3
from datetime import datetime, date, timedelta
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

DATA_DIR = Path(os.getenv("DATA_DIR", "./data"))

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Database setup
# ---------------------------------------------------------------------------

def _get_db():
    """Get analytics database connection with tables created."""
    db_path = DATA_DIR / "stories.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    conn.execute("""
        CREATE TABLE IF NOT EXISTS video_analytics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            channel_key TEXT NOT NULL,
            video_id TEXT NOT NULL,
            date TEXT NOT NULL,
            views INTEGER DEFAULT 0,
            estimated_minutes_watched REAL DEFAULT 0,
            average_view_duration REAL DEFAULT 0,
            impressions INTEGER DEFAULT 0,
            impression_ctr REAL DEFAULT 0,
            subscribers_gained INTEGER DEFAULT 0,
            subscribers_lost INTEGER DEFAULT 0,
            likes INTEGER DEFAULT 0,
            shares INTEGER DEFAULT 0,
            fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(channel_key, video_id, date)
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS search_terms (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            channel_key TEXT NOT NULL,
            search_term TEXT NOT NULL,
            views INTEGER DEFAULT 0,
            estimated_minutes_watched REAL DEFAULT 0,
            date TEXT NOT NULL,
            fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(channel_key, search_term, date)
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS channel_analytics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            channel_key TEXT NOT NULL,
            date TEXT NOT NULL,
            views INTEGER DEFAULT 0,
            estimated_minutes_watched REAL DEFAULT 0,
            subscribers_gained INTEGER DEFAULT 0,
            subscribers_lost INTEGER DEFAULT 0,
            impressions INTEGER DEFAULT 0,
            impression_ctr REAL DEFAULT 0,
            fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(channel_key, date)
        )
    """)

    conn.commit()
    return conn


# ---------------------------------------------------------------------------
# Analytics API calls
# ---------------------------------------------------------------------------

def fetch_video_analytics(
    channel_key: str,
    start_date: str = None,
    end_date: str = None,
) -> list[dict]:
    """Fetch per-video analytics from YouTube Analytics API v2.

    Args:
        channel_key: Key from channels.json (e.g. 'stories_en').
        start_date: YYYY-MM-DD start date. Defaults to 28 days ago.
        end_date: YYYY-MM-DD end date. Defaults to yesterday.

    Returns:
        List of dicts with video_id and metric values.
    """
    from scripts.channel_manager import get_analytics_service

    if start_date is None:
        start_date = (date.today() - timedelta(days=28)).isoformat()
    if end_date is None:
        end_date = (date.today() - timedelta(days=1)).isoformat()

    try:
        analytics = get_analytics_service(channel_key)

        response = analytics.reports().query(
            ids="channel==MINE",
            startDate=start_date,
            endDate=end_date,
            metrics=(
                "views,estimatedMinutesWatched,averageViewDuration,"
                "impressions,impressionClickThroughRate,"
                "subscribersGained,subscribersLost,likes,shares"
            ),
            dimensions="video",
            sort="-views",
            maxResults=200,
        ).execute()

        results = []
        headers = [col["name"] for col in response.get("columnHeaders", [])]
        for row in response.get("rows", []):
            record = dict(zip(headers, row))
            results.append({
                "video_id": record.get("video", ""),
                "views": int(record.get("views", 0)),
                "estimated_minutes_watched": float(record.get("estimatedMinutesWatched", 0)),
                "average_view_duration": float(record.get("averageViewDuration", 0)),
                "impressions": int(record.get("impressions", 0)),
                "impression_ctr": float(record.get("impressionClickThroughRate", 0)),
                "subscribers_gained": int(record.get("subscribersGained", 0)),
                "subscribers_lost": int(record.get("subscribersLost", 0)),
                "likes": int(record.get("likes", 0)),
                "shares": int(record.get("shares", 0)),
            })

        logger.info(f"Fetched analytics for {len(results)} videos from {channel_key}")
        return results

    except Exception as e:
        logger.warning(f"Failed to fetch video analytics for {channel_key}: {e}")
        return []


def fetch_channel_analytics(
    channel_key: str,
    start_date: str = None,
    end_date: str = None,
) -> list[dict]:
    """Fetch daily channel-level aggregate analytics.

    Returns list of dicts with date and aggregate metrics.
    """
    from scripts.channel_manager import get_analytics_service

    if start_date is None:
        start_date = (date.today() - timedelta(days=28)).isoformat()
    if end_date is None:
        end_date = (date.today() - timedelta(days=1)).isoformat()

    try:
        analytics = get_analytics_service(channel_key)

        response = analytics.reports().query(
            ids="channel==MINE",
            startDate=start_date,
            endDate=end_date,
            metrics=(
                "views,estimatedMinutesWatched,"
                "subscribersGained,subscribersLost,"
                "impressions,impressionClickThroughRate"
            ),
            dimensions="day",
            sort="day",
        ).execute()

        results = []
        headers = [col["name"] for col in response.get("columnHeaders", [])]
        for row in response.get("rows", []):
            record = dict(zip(headers, row))
            results.append({
                "date": record.get("day", ""),
                "views": int(record.get("views", 0)),
                "estimated_minutes_watched": float(record.get("estimatedMinutesWatched", 0)),
                "subscribers_gained": int(record.get("subscribersGained", 0)),
                "subscribers_lost": int(record.get("subscribersLost", 0)),
                "impressions": int(record.get("impressions", 0)),
                "impression_ctr": float(record.get("impressionClickThroughRate", 0)),
            })

        logger.info(f"Fetched {len(results)} days of channel analytics for {channel_key}")
        return results

    except Exception as e:
        logger.warning(f"Failed to fetch channel analytics for {channel_key}: {e}")
        return []


def fetch_search_terms(
    channel_key: str,
    start_date: str = None,
    end_date: str = None,
) -> list[dict]:
    """Fetch search terms that drove traffic to the channel.

    Uses insightTrafficSourceDetail dimension with insightTrafficSourceType=YT_SEARCH
    filter to extract actual YouTube search queries viewers used.

    Returns list of dicts with search_term, views, estimated_minutes_watched.
    """
    from scripts.channel_manager import get_analytics_service

    if start_date is None:
        start_date = (date.today() - timedelta(days=28)).isoformat()
    if end_date is None:
        end_date = (date.today() - timedelta(days=1)).isoformat()

    try:
        analytics = get_analytics_service(channel_key)

        response = analytics.reports().query(
            ids="channel==MINE",
            startDate=start_date,
            endDate=end_date,
            metrics="views,estimatedMinutesWatched",
            dimensions="insightTrafficSourceDetail",
            filters="insightTrafficSourceType==YT_SEARCH",
            sort="-views",
            maxResults=200,
        ).execute()

        results = []
        headers = [col["name"] for col in response.get("columnHeaders", [])]
        for row in response.get("rows", []):
            record = dict(zip(headers, row))
            results.append({
                "search_term": record.get("insightTrafficSourceDetail", ""),
                "views": int(record.get("views", 0)),
                "estimated_minutes_watched": float(record.get("estimatedMinutesWatched", 0)),
            })

        logger.info(f"Fetched {len(results)} search terms for {channel_key}")
        return results

    except Exception as e:
        logger.warning(f"Failed to fetch search terms for {channel_key}: {e}")
        return []


# ---------------------------------------------------------------------------
# Database storage
# ---------------------------------------------------------------------------

def store_video_analytics(channel_key: str, analytics_data: list[dict]):
    """Store per-video analytics in the database."""
    if not analytics_data:
        return

    conn = _get_db()
    today = date.today().isoformat()
    stored = 0

    for record in analytics_data:
        try:
            conn.execute("""
                INSERT INTO video_analytics
                    (channel_key, video_id, date, views, estimated_minutes_watched,
                     average_view_duration, impressions, impression_ctr,
                     subscribers_gained, subscribers_lost, likes, shares)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(channel_key, video_id, date) DO UPDATE SET
                    views = excluded.views,
                    estimated_minutes_watched = excluded.estimated_minutes_watched,
                    average_view_duration = excluded.average_view_duration,
                    impressions = excluded.impressions,
                    impression_ctr = excluded.impression_ctr,
                    subscribers_gained = excluded.subscribers_gained,
                    subscribers_lost = excluded.subscribers_lost,
                    likes = excluded.likes,
                    shares = excluded.shares,
                    fetched_at = CURRENT_TIMESTAMP
            """, (
                channel_key, record["video_id"], today,
                record["views"], record["estimated_minutes_watched"],
                record["average_view_duration"], record["impressions"],
                record["impression_ctr"], record["subscribers_gained"],
                record["subscribers_lost"], record["likes"], record["shares"],
            ))
            stored += 1
        except Exception as e:
            logger.warning(f"Failed to store analytics for {record.get('video_id')}: {e}")

    conn.commit()
    conn.close()
    logger.info(f"Stored analytics for {stored} videos ({channel_key})")


def store_channel_analytics(channel_key: str, analytics_data: list[dict]):
    """Store daily channel-level analytics in the database."""
    if not analytics_data:
        return

    conn = _get_db()
    stored = 0

    for record in analytics_data:
        try:
            conn.execute("""
                INSERT INTO channel_analytics
                    (channel_key, date, views, estimated_minutes_watched,
                     subscribers_gained, subscribers_lost, impressions, impression_ctr)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(channel_key, date) DO UPDATE SET
                    views = excluded.views,
                    estimated_minutes_watched = excluded.estimated_minutes_watched,
                    subscribers_gained = excluded.subscribers_gained,
                    subscribers_lost = excluded.subscribers_lost,
                    impressions = excluded.impressions,
                    impression_ctr = excluded.impression_ctr,
                    fetched_at = CURRENT_TIMESTAMP
            """, (
                channel_key, record["date"],
                record["views"], record["estimated_minutes_watched"],
                record["subscribers_gained"], record["subscribers_lost"],
                record["impressions"], record["impression_ctr"],
            ))
            stored += 1
        except Exception as e:
            logger.warning(f"Failed to store channel analytics for {record.get('date')}: {e}")

    conn.commit()
    conn.close()
    logger.info(f"Stored {stored} days of channel analytics ({channel_key})")


def store_search_terms(channel_key: str, search_data: list[dict]):
    """Store search terms in the database."""
    if not search_data:
        return

    conn = _get_db()
    today = date.today().isoformat()
    stored = 0

    for record in search_data:
        try:
            conn.execute("""
                INSERT INTO search_terms
                    (channel_key, search_term, views, estimated_minutes_watched, date)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(channel_key, search_term, date) DO UPDATE SET
                    views = excluded.views,
                    estimated_minutes_watched = excluded.estimated_minutes_watched,
                    fetched_at = CURRENT_TIMESTAMP
            """, (
                channel_key, record["search_term"],
                record["views"], record["estimated_minutes_watched"], today,
            ))
            stored += 1
        except Exception as e:
            logger.warning(f"Failed to store search term: {e}")

    conn.commit()
    conn.close()
    logger.info(f"Stored {stored} search terms ({channel_key})")


# ---------------------------------------------------------------------------
# Query helpers
# ---------------------------------------------------------------------------

def get_video_performance(video_id: str) -> dict:
    """Get the latest analytics snapshot for a specific video."""
    conn = _get_db()
    row = conn.execute("""
        SELECT * FROM video_analytics
        WHERE video_id = ?
        ORDER BY date DESC
        LIMIT 1
    """, (video_id,)).fetchone()
    conn.close()
    return dict(row) if row else {}


def get_top_search_terms(
    channel_key: str = None,
    days: int = 28,
    limit: int = 50,
) -> list[dict]:
    """Get top search terms by views across channels.

    Args:
        channel_key: Filter by channel. None = all channels.
        days: Look back period in days.
        limit: Max results.
    """
    conn = _get_db()
    cutoff = (date.today() - timedelta(days=days)).isoformat()

    query = """
        SELECT search_term,
               SUM(views) as total_views,
               SUM(estimated_minutes_watched) as total_watch_time
        FROM search_terms
        WHERE date >= ?
    """
    params = [cutoff]

    if channel_key:
        query += " AND channel_key = ?"
        params.append(channel_key)

    query += " GROUP BY search_term ORDER BY total_views DESC LIMIT ?"
    params.append(limit)

    results = conn.execute(query, params).fetchall()
    conn.close()
    return [dict(r) for r in results]


def get_channel_performance_summary(channel_key: str, days: int = 7) -> dict:
    """Get a summary of channel performance over the last N days."""
    conn = _get_db()
    cutoff = (date.today() - timedelta(days=days)).isoformat()

    row = conn.execute("""
        SELECT
            SUM(views) as total_views,
            SUM(estimated_minutes_watched) as total_watch_time,
            SUM(subscribers_gained) as total_subs_gained,
            SUM(subscribers_lost) as total_subs_lost,
            AVG(impression_ctr) as avg_ctr,
            SUM(impressions) as total_impressions
        FROM channel_analytics
        WHERE channel_key = ? AND date >= ?
    """, (channel_key, cutoff)).fetchone()
    conn.close()

    if row and row["total_views"] is not None:
        return {
            "channel_key": channel_key,
            "period_days": days,
            "total_views": row["total_views"],
            "total_watch_time_minutes": row["total_watch_time"] or 0,
            "subscribers_gained": row["total_subs_gained"] or 0,
            "subscribers_lost": row["total_subs_lost"] or 0,
            "net_subscribers": (row["total_subs_gained"] or 0) - (row["total_subs_lost"] or 0),
            "avg_ctr": row["avg_ctr"] or 0,
            "total_impressions": row["total_impressions"] or 0,
        }
    return {"channel_key": channel_key, "period_days": days, "total_views": 0}


# ---------------------------------------------------------------------------
# Master update function
# ---------------------------------------------------------------------------

def update_analytics_database():
    """Fetch and store analytics for ALL channels.

    Called by the scheduler every 6 hours to keep analytics data fresh.
    Iterates over all channels in channels.json, fetches video analytics,
    channel analytics, and search terms, then stores everything in SQLite.
    """
    from scripts.channel_manager import load_channel_config

    config = load_channel_config()
    channels = config.get("channels", {})

    total_videos = 0
    total_terms = 0
    channels_updated = 0

    for key, channel in channels.items():
        # Skip channels without auth tokens
        token_path = DATA_DIR / channel.get("token_file", "")
        if not token_path.exists():
            logger.debug(f"Skipping {key}: no auth token")
            continue

        logger.info(f"Updating analytics for {key} ({channel['name']})...")

        try:
            # Video-level analytics
            video_data = fetch_video_analytics(key)
            store_video_analytics(key, video_data)
            total_videos += len(video_data)

            # Channel-level daily analytics
            channel_data = fetch_channel_analytics(key)
            store_channel_analytics(key, channel_data)

            # Search terms
            search_data = fetch_search_terms(key)
            store_search_terms(key, search_data)
            total_terms += len(search_data)

            channels_updated += 1

        except Exception as e:
            logger.warning(f"Analytics update failed for {key}: {e}")

    logger.info(
        f"Analytics update complete: {channels_updated} channels, "
        f"{total_videos} video records, {total_terms} search terms"
    )
    print(
        f"  Analytics update: {channels_updated} channels, "
        f"{total_videos} videos, {total_terms} search terms"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    if len(sys.argv) > 1:
        cmd = sys.argv[1]

        if cmd == "update":
            update_analytics_database()

        elif cmd == "test":
            # Test with the first authenticated channel
            from scripts.channel_manager import load_channel_config
            config = load_channel_config()
            for key, ch in config["channels"].items():
                token_path = DATA_DIR / ch.get("token_file", "")
                if token_path.exists():
                    print(f"\nTesting analytics for: {key} ({ch['name']})")
                    print("-" * 60)

                    videos = fetch_video_analytics(key)
                    print(f"  Videos with data: {len(videos)}")
                    for v in videos[:5]:
                        print(f"    {v['video_id']}: {v['views']} views, "
                              f"CTR={v['impression_ctr']:.1%}, "
                              f"avg_duration={v['average_view_duration']:.0f}s")

                    terms = fetch_search_terms(key)
                    print(f"  Search terms: {len(terms)}")
                    for t in terms[:10]:
                        print(f"    '{t['search_term']}': {t['views']} views")

                    summary = get_channel_performance_summary(key, days=7)
                    print(f"  7-day summary: {summary}")
                    break
            else:
                print("No authenticated channels found. Run channel auth first.")

        elif cmd == "search-terms":
            channel_key = sys.argv[2] if len(sys.argv) > 2 else None
            terms = get_top_search_terms(channel_key=channel_key, limit=30)
            print(f"\nTop Search Terms (last 28 days):")
            print("-" * 60)
            for i, t in enumerate(terms, 1):
                print(f"  {i:2d}. '{t['search_term']}' - "
                      f"{t['total_views']} views, "
                      f"{t['total_watch_time']:.0f} min watched")

        elif cmd == "summary":
            from scripts.channel_manager import load_channel_config
            config = load_channel_config()
            print(f"\nChannel Performance Summary (7 days):")
            print("=" * 80)
            for key, ch in config["channels"].items():
                summary = get_channel_performance_summary(key, days=7)
                if summary.get("total_views", 0) > 0:
                    print(f"  {key}: {summary['total_views']} views, "
                          f"+{summary.get('net_subscribers', 0)} subs, "
                          f"CTR={summary.get('avg_ctr', 0):.1%}")
            print("=" * 80)

        else:
            print(f"Unknown command: {cmd}")
            print("Usage: python youtube_analytics.py [update|test|search-terms|summary]")
    else:
        print("YouTube Analytics Module")
        print("Usage:")
        print("  python youtube_analytics.py update           # Fetch all channel analytics")
        print("  python youtube_analytics.py test             # Test with first auth channel")
        print("  python youtube_analytics.py search-terms     # Show top search terms")
        print("  python youtube_analytics.py summary          # Channel performance summary")
