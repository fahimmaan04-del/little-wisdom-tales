"""
Scheduler - Fully autonomous story pipeline on a fixed schedule.

Creates stories throughout the day, runs analytics, refreshes
trending keywords, and optimizes content strategy automatically.

Schedule (configurable via .env):
- 10 stories per day spread across waking hours
- Every 6 hours: Analytics update + keyword refresh
- Daily: Disk cleanup of old files
"""

import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import schedule
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))
load_dotenv()

from scripts.orchestrator import create_single_story, run_analytics_update

DATA_DIR = Path(os.getenv("DATA_DIR", "./data"))
LOG_DIR = DATA_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "scheduler.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# Maximum video duration in seconds (5 minutes)
MAX_VIDEO_DURATION = int(os.getenv("MAX_VIDEO_DURATION", "300"))


def story_job(job_name: str):
    """Create one story video (max 5 minutes)."""
    if not check_disk_space():
        logger.warning(f"[{job_name}] Skipping - low disk space")
        return

    logger.info(f"[{job_name}] Starting story creation...")

    try:
        result = create_single_story(for_shorts=False, smart_pick=True)
        if result["status"] == "completed":
            duration = result.get("video_duration", 0)
            logger.info(f"[{job_name}] Published: {result.get('title')} ({duration:.0f}s)")
        else:
            logger.error(f"[{job_name}] Failed: {result.get('error')}")
    except Exception as e:
        logger.error(f"[{job_name}] Error: {e}")

    logger.info(f"[{job_name}] Job complete.")


def analytics_job():
    """Update analytics and refresh trending keywords."""
    logger.info("[Analytics] Running analytics + keyword update...")
    try:
        run_analytics_update()

        # Refresh trending keywords from YouTube search data
        from scripts.keyword_optimizer import run_keyword_refresh
        run_keyword_refresh()

        # Log content strategy insights
        from scripts.story_generator import get_top_performing_themes
        top = get_top_performing_themes(limit=3)
        if top:
            logger.info("[Analytics] Top performing themes:")
            for t in top:
                logger.info(f"  {t['collection']}/{t['moral']}: "
                           f"avg_views={t['avg_views']:.0f}, count={t['story_count']}")
        logger.info("[Analytics] Update complete.")
    except Exception as e:
        logger.error(f"[Analytics] Update failed: {e}")


def setup_schedule():
    """Configure the daily schedule based on STORIES_PER_DAY."""
    stories_per_day = int(os.getenv("STORIES_PER_DAY", "10"))

    # Schedule story jobs evenly throughout the day
    schedule_times = {
        3: ["06:00", "14:00", "22:00"],
        5: ["04:00", "08:00", "12:00", "16:00", "20:00"],
        6: ["03:00", "07:00", "11:00", "15:00", "19:00", "23:00"],
        8: ["02:00", "05:00", "08:00", "11:00", "14:00", "17:00", "20:00", "23:00"],
        10: ["01:00", "03:30", "06:00", "08:30", "11:00",
             "13:30", "16:00", "18:30", "21:00", "23:30"],
        12: ["00:00", "02:00", "04:00", "06:00", "08:00", "10:00",
             "12:00", "14:00", "16:00", "18:00", "20:00", "22:00"],
    }

    times = schedule_times.get(stories_per_day, schedule_times[10])

    for i, t in enumerate(times):
        name = f"Story-{i+1}"
        schedule.every().day.at(t).do(story_job, job_name=name)

    # Analytics + keyword refresh every 6 hours
    schedule.every().day.at("00:00").do(analytics_job)
    schedule.every().day.at("06:00").do(analytics_job)
    schedule.every().day.at("12:00").do(analytics_job)
    schedule.every().day.at("18:00").do(analytics_job)

    logger.info(f"Schedule configured ({stories_per_day} stories/day):")
    for t in times:
        logger.info(f"  {t} UTC - Story video (max {MAX_VIDEO_DURATION}s)")
    logger.info(f"  Every 6h - Analytics + keyword refresh")
    logger.info(f"  = {stories_per_day} videos per day")

    # Quota warning
    quota_needed = stories_per_day * 1600
    if quota_needed > 10000:
        logger.warning(f"  WARNING: {stories_per_day} uploads need {quota_needed} quota units/day "
                       f"(standard limit: 10,000). Apply for higher quota at "
                       f"https://support.google.com/youtube/contact/yt_api_form")


def check_disk_space():
    """Check if there's enough disk space to create a video."""
    import shutil
    total, used, free = shutil.disk_usage("/mnt")
    free_gb = free / (1024**3)
    if free_gb < 2.0:
        logger.warning(f"Low disk space: {free_gb:.1f}GB free. Cleaning old output files...")
        output_dir = Path(os.getenv("OUTPUT_DIR", "./output"))
        import glob
        for subdir in ["videos", "audio", "images"]:
            for vdir in sorted(glob.glob(str(output_dir / subdir / "story_*"))):
                vpath = Path(vdir)
                if vpath.is_dir():
                    age_days = (time.time() - vpath.stat().st_mtime) / 86400
                    if age_days > 3:
                        import shutil as _shutil
                        _shutil.rmtree(vdir)
                        logger.info(f"  Cleaned old dir: {vdir}")
        return free_gb >= 1.0
    return True


def main():
    """Main scheduler loop - fully autonomous."""
    logger.info("=" * 60)
    logger.info("Little Wisdom Tales - Autonomous Pipeline")
    logger.info(f"Max video duration: {MAX_VIDEO_DURATION}s")
    logger.info("=" * 60)

    setup_schedule()

    # Run initial keyword refresh on startup
    logger.info("Running initial keyword refresh...")
    try:
        from scripts.keyword_optimizer import run_keyword_refresh
        run_keyword_refresh()
    except Exception as e:
        logger.warning(f"Initial keyword refresh failed: {e}")

    # Run first story immediately on startup
    if check_disk_space():
        logger.info("Running initial story on startup...")
        story_job("Startup")
    else:
        logger.error("Not enough disk space to start.")

    # Enter the schedule loop
    logger.info("Entering autonomous schedule loop...")
    while True:
        try:
            schedule.run_pending()
        except Exception as e:
            logger.error(f"Schedule loop error: {e}")
        time.sleep(60)


if __name__ == "__main__":
    main()
