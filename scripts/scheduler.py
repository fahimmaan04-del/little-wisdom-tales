"""
Scheduler - Fully autonomous story pipeline on a fixed schedule.

Creates stories throughout the day, runs analytics, and optimizes
content strategy automatically. No human intervention required.

Schedule (configurable via .env STORIES_PER_DAY, default 6):
- Every 2-4 hours: 1 video + 1 short
- Every 6 hours: Analytics update + content strategy optimization
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


def story_job(job_name: str):
    """Create one story (video + shorts version)."""
    if not check_disk_space():
        logger.warning(f"[{job_name}] Skipping - low disk space")
        return

    logger.info(f"[{job_name}] Starting story creation...")

    try:
        logger.info(f"[{job_name}] Creating regular video...")
        result_video = create_single_story(for_shorts=False, smart_pick=True)
        if result_video["status"] == "completed":
            logger.info(f"[{job_name}] Video published: {result_video.get('title')}")
        else:
            logger.error(f"[{job_name}] Video failed: {result_video.get('error')}")
    except Exception as e:
        logger.error(f"[{job_name}] Video creation error: {e}")

    time.sleep(60)

    try:
        logger.info(f"[{job_name}] Creating shorts version...")
        result_shorts = create_single_story(for_shorts=True, smart_pick=True)
        if result_shorts["status"] == "completed":
            logger.info(f"[{job_name}] Shorts published: {result_shorts.get('title')}")
        else:
            logger.error(f"[{job_name}] Shorts failed: {result_shorts.get('error')}")
    except Exception as e:
        logger.error(f"[{job_name}] Shorts creation error: {e}")

    logger.info(f"[{job_name}] Job complete.")


def analytics_job():
    """Update analytics and generate content optimization insights."""
    logger.info("[Analytics] Running analytics update...")
    try:
        run_analytics_update()

        # Log content strategy insights
        from scripts.story_generator import get_top_performing_themes
        top = get_top_performing_themes(limit=3)
        if top:
            logger.info("[Analytics] Top performing themes:")
            for t in top:
                logger.info(f"  {t['collection']}/{t['moral']}: "
                           f"avg_views={t['avg_views']:.0f}, count={t['story_count']}")
        logger.info("[Analytics] Update complete - smart_pick will use these insights.")
    except Exception as e:
        logger.error(f"[Analytics] Update failed: {e}")


def setup_schedule():
    """Configure the daily schedule based on STORIES_PER_DAY."""
    stories_per_day = int(os.getenv("STORIES_PER_DAY", "6"))

    # Schedule story jobs evenly throughout the day
    # Each job creates 1 video + 1 short = 2 uploads per job
    schedule_times = {
        3: ["06:00", "14:00", "22:00"],
        4: ["05:00", "11:00", "17:00", "23:00"],
        5: ["04:00", "08:00", "12:00", "16:00", "20:00"],
        6: ["03:00", "07:00", "11:00", "15:00", "19:00", "23:00"],
        8: ["02:00", "05:00", "08:00", "11:00", "14:00", "17:00", "20:00", "23:00"],
    }

    times = schedule_times.get(stories_per_day, schedule_times[6])
    job_names = ["Batch-1", "Batch-2", "Batch-3", "Batch-4", "Batch-5",
                 "Batch-6", "Batch-7", "Batch-8"]

    for i, t in enumerate(times):
        name = job_names[i] if i < len(job_names) else f"Batch-{i+1}"
        schedule.every().day.at(t).do(story_job, job_name=name)

    # Analytics updates every 6 hours
    schedule.every().day.at("00:00").do(analytics_job)
    schedule.every().day.at("06:00").do(analytics_job)
    schedule.every().day.at("12:00").do(analytics_job)
    schedule.every().day.at("18:00").do(analytics_job)

    logger.info(f"Schedule configured ({stories_per_day} stories/day):")
    for t in times:
        logger.info(f"  {t} UTC - Story batch (1 video + 1 short)")
    logger.info(f"  Every 6h - Analytics update")
    logger.info(f"  = {stories_per_day} videos + {stories_per_day} shorts per day")


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
    logger.info("=" * 60)

    setup_schedule()

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
