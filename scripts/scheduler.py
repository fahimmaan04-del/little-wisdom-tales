"""
Scheduler - Fully autonomous multi-channel pipeline for 100+ videos per day.

Manages 21 YouTube channels across 4 content types (stories, education, AI education, crafts/skills)
and 6 languages (EN, HI, ES, FR, PT, AR). Runs 24/7 without human intervention.

Content Types:
  - Stories: Moral stories from world folklore (region-specific channels)
  - Education: Oxford/Cambridge Class 1-10 curricula (cartoon-based)
  - AI Education: AI/coding/tech lessons for kids (Byte the Robot host)

Supports two operating modes:
  - "batch" (default): Phased generation then distributed uploads
  - "incremental": Generate and upload one video at a time

Batch Schedule (UTC):
  Phase 1 (01:00): Generate all scripts (LLM) - stories + education + AI lessons
  Phase 2 (02:00): Generate all audio (TTS) - multi-language voices
  Phase 3 (03:00): Generate all images (GPU) - longest phase (~5h for 100+ videos)
  Phase 4 (08:00): Assemble all videos (FFmpeg) - crossfade animations
  Phase 5 (10:00-22:00): Upload in batches across 18 channels
  Every 6h: Analytics + keyword refresh (all languages)
  Every 1h: Health check (GPU, disk, upload progress, pipeline state)

Self-Healing:
  - Auto-retries failed phases
  - Catches up on missed schedules on restart
  - Cleans old output files when disk is low
  - Tracks progress in SQLite for crash recovery

Configuration via .env:
  STORIES_PER_DAY=36 (6 channels x 6/day)
  EDUCATION_PER_DAY=42 (7 channels x 6/day)
  AI_EDUCATION_PER_DAY=20 (4 channels x 5/day)
  CRAFTS_SKILLS_PER_DAY=12 (3 channels x 4/day)
  PIPELINE_MODE=batch|incremental
"""

import json
import logging
import os
import shutil
import sys
import time
from datetime import datetime, date
from pathlib import Path

import schedule
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))
load_dotenv()

from scripts.orchestrator import create_single_story, run_analytics_update
from scripts.batch_pipeline import (
    run_daily_batch,
    run_incremental,
    generate_story_batch,
    generate_education_batch,
    generate_ai_education_batch,
    generate_regional_story_batch,
    generate_crafts_skills_batch,
    generate_audio_batch,
    generate_images_batch,
    assemble_videos_batch,
    upload_batch,
)
from scripts.channel_manager import load_channel_config, get_all_channel_stats, can_upload

DATA_DIR = Path(os.getenv("DATA_DIR", "./data"))
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "./output"))
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

# ----- Configuration from .env -----
STORIES_PER_DAY = int(os.getenv("STORIES_PER_DAY", "36"))
EDUCATION_PER_DAY = int(os.getenv("EDUCATION_PER_DAY", "48"))
AI_EDUCATION_PER_DAY = int(os.getenv("AI_EDUCATION_PER_DAY", "20"))
CRAFTS_SKILLS_PER_DAY = int(os.getenv("CRAFTS_SKILLS_PER_DAY", "12"))
MAX_VIDEO_DURATION = int(os.getenv("MAX_VIDEO_DURATION", "300"))
MIN_VIDEO_DURATION = int(os.getenv("MIN_VIDEO_DURATION", "180"))
PIPELINE_MODE = os.getenv("PIPELINE_MODE", "batch").lower()  # "batch" or "incremental"

# ----- Shared state for phased batch pipeline -----
# Holds generated scripts between phases so each phase can be scheduled independently.
_batch_state = {
    "date": None,
    "all_scripts": [],
    "phase_completed": set(),
    "upload_batches": [],
    "upload_index": 0,
}


def _reset_batch_state():
    """Reset batch state for a new day."""
    _batch_state["date"] = date.today().isoformat()
    _batch_state["all_scripts"] = []
    _batch_state["phase_completed"] = set()
    _batch_state["upload_batches"] = []
    _batch_state["upload_index"] = 0


def _ensure_today():
    """Reset state if the date has changed since last run."""
    if _batch_state["date"] != date.today().isoformat():
        _reset_batch_state()


# =====================================================================
#  Phase Jobs (Batch Mode)
# =====================================================================

def phase1_scripts_job():
    """Phase 1 (01:00 UTC): Generate all scripts via LLM."""
    _ensure_today()

    if "phase1" in _batch_state["phase_completed"]:
        logger.info("[Phase 1] Already completed today, skipping")
        return

    if not check_disk_space():
        logger.warning("[Phase 1] Skipping - low disk space")
        return

    total_target = STORIES_PER_DAY + EDUCATION_PER_DAY + AI_EDUCATION_PER_DAY + CRAFTS_SKILLS_PER_DAY

    logger.info("=" * 70)
    logger.info("[Phase 1] SCRIPT GENERATION - All Channels")
    logger.info(f"  Stories: {STORIES_PER_DAY}, Education: {EDUCATION_PER_DAY}, "
                f"AI Education: {AI_EDUCATION_PER_DAY}, Crafts/Skills: {CRAFTS_SKILLS_PER_DAY}")
    logger.info(f"  Total target: {total_target}")
    logger.info("=" * 70)

    phase_start = time.time()

    try:
        # Use the batch pipeline's run_daily_batch which reads channels.json
        # and generates scripts for ALL channels automatically
        config = load_channel_config()

        # Build per-channel generation from config
        all_scripts = []

        # Stories (English + regional)
        for key, ch in config["channels"].items():
            if ch["type"] != "stories":
                continue
            target = ch["daily_target"]
            lang = ch.get("language", "en")
            if lang == "en":
                all_scripts.extend(generate_story_batch(target))
            else:
                all_scripts.extend(generate_regional_story_batch(target, lang, key))

        # Education (all curricula)
        for key, ch in config["channels"].items():
            if ch["type"] != "education":
                continue
            target = ch["daily_target"]
            curriculum = ch.get("curriculum", "oxford")
            all_scripts.extend(generate_education_batch(target, curriculum=curriculum))

        # AI Education (all languages)
        for key, ch in config["channels"].items():
            if ch["type"] != "ai_education":
                continue
            target = ch["daily_target"]
            lang = ch.get("language", "en")
            all_scripts.extend(generate_ai_education_batch(target, language=lang))

        # Crafts/Skills (all categories)
        for key, ch in config["channels"].items():
            if ch["type"] != "crafts_skills":
                continue
            target = ch["daily_target"]
            skill_cat = ch.get("skill_category", "carpentry_building")
            all_scripts.extend(generate_crafts_skills_batch(target, skill_category=skill_cat))

        _batch_state["all_scripts"] = all_scripts
        _batch_state["phase_completed"].add("phase1")

        # Count by type for logging
        type_counts = {}
        for s in all_scripts:
            k = f"{s.get('content_type', 'story')}_{s.get('language', 'en')}"
            type_counts[k] = type_counts.get(k, 0) + 1

        elapsed = time.time() - phase_start
        logger.info(f"[Phase 1] Complete: {len(all_scripts)} scripts in {elapsed:.0f}s")
        for k, v in sorted(type_counts.items()):
            logger.info(f"  {k}: {v}")
    except Exception as e:
        logger.error(f"[Phase 1] Failed: {e}")
        import traceback
        logger.error(traceback.format_exc())


def phase2_audio_job():
    """Phase 2 (02:00 UTC): Generate TTS audio for all scripts."""
    _ensure_today()

    if "phase2" in _batch_state["phase_completed"]:
        logger.info("[Phase 2] Already completed today, skipping")
        return
    if "phase1" not in _batch_state["phase_completed"]:
        logger.warning("[Phase 2] Phase 1 not complete yet, running it now")
        phase1_scripts_job()
        if "phase1" not in _batch_state["phase_completed"]:
            logger.error("[Phase 2] Cannot proceed - Phase 1 failed")
            return

    logger.info("=" * 70)
    logger.info(f"[Phase 2] AUDIO GENERATION - {len(_batch_state['all_scripts'])} scripts")
    logger.info("=" * 70)

    phase_start = time.time()
    try:
        _batch_state["all_scripts"] = generate_audio_batch(_batch_state["all_scripts"])
        _batch_state["phase_completed"].add("phase2")
        elapsed = time.time() - phase_start
        logger.info(f"[Phase 2] Complete: {len(_batch_state['all_scripts'])} audio tracks in {elapsed:.0f}s")
    except Exception as e:
        logger.error(f"[Phase 2] Failed: {e}")


def phase3_images_job():
    """Phase 3 (03:00 UTC): Generate AI images for all scripts (GPU intensive)."""
    _ensure_today()

    if "phase3" in _batch_state["phase_completed"]:
        logger.info("[Phase 3] Already completed today, skipping")
        return
    if "phase2" not in _batch_state["phase_completed"]:
        logger.warning("[Phase 3] Phase 2 not complete yet, running it now")
        phase2_audio_job()
        if "phase2" not in _batch_state["phase_completed"]:
            logger.error("[Phase 3] Cannot proceed - Phase 2 failed")
            return

    if not check_disk_space():
        logger.warning("[Phase 3] Skipping - low disk space")
        return

    config = load_channel_config()
    images_per_scene = config["global_settings"].get("images_per_scene", 20)

    logger.info("=" * 70)
    logger.info(f"[Phase 3] IMAGE GENERATION (GPU) - {len(_batch_state['all_scripts'])} scripts")
    logger.info(f"  Images per scene: {images_per_scene}")
    logger.info("=" * 70)

    phase_start = time.time()
    try:
        _batch_state["all_scripts"] = generate_images_batch(
            _batch_state["all_scripts"],
            images_per_scene=images_per_scene,
        )
        _batch_state["phase_completed"].add("phase3")
        elapsed = time.time() - phase_start
        logger.info(f"[Phase 3] Complete: {len(_batch_state['all_scripts'])} image sets in {elapsed:.0f}s")
    except Exception as e:
        logger.error(f"[Phase 3] Failed: {e}")


def phase4_assembly_job():
    """Phase 4 (08:00 UTC): Assemble all videos with FFmpeg."""
    _ensure_today()

    if "phase4" in _batch_state["phase_completed"]:
        logger.info("[Phase 4] Already completed today, skipping")
        return
    if "phase3" not in _batch_state["phase_completed"]:
        logger.warning("[Phase 4] Phase 3 not complete yet, running it now")
        phase3_images_job()
        if "phase3" not in _batch_state["phase_completed"]:
            logger.error("[Phase 4] Cannot proceed - Phase 3 failed")
            return

    if not check_disk_space():
        logger.warning("[Phase 4] Skipping - low disk space")
        return

    logger.info("=" * 70)
    logger.info(f"[Phase 4] VIDEO ASSEMBLY - {len(_batch_state['all_scripts'])} scripts")
    logger.info("=" * 70)

    phase_start = time.time()
    try:
        _batch_state["all_scripts"] = assemble_videos_batch(_batch_state["all_scripts"])
        _batch_state["phase_completed"].add("phase4")

        # Split into upload batches
        config = load_channel_config()
        batch_size = config["global_settings"].get("batch_size", 10)
        scripts = _batch_state["all_scripts"]
        _batch_state["upload_batches"] = [
            scripts[i:i + batch_size] for i in range(0, len(scripts), batch_size)
        ]
        _batch_state["upload_index"] = 0

        elapsed = time.time() - phase_start
        logger.info(f"[Phase 4] Complete: {len(scripts)} videos in {elapsed:.0f}s")
        logger.info(f"  Split into {len(_batch_state['upload_batches'])} upload batches of ~{batch_size}")
    except Exception as e:
        logger.error(f"[Phase 4] Failed: {e}")


def phase5_upload_job():
    """Phase 5 (10:00-22:00 UTC): Upload next batch of videos to YouTube."""
    _ensure_today()

    if "phase4" not in _batch_state["phase_completed"]:
        logger.warning("[Phase 5] Phase 4 not complete, skipping upload")
        return

    batches = _batch_state["upload_batches"]
    idx = _batch_state["upload_index"]

    if idx >= len(batches):
        logger.info("[Phase 5] All batches uploaded for today")
        return

    batch = batches[idx]
    logger.info(f"[Phase 5] Uploading batch {idx + 1}/{len(batches)} ({len(batch)} videos)")

    try:
        results = upload_batch(batch)
        uploaded = sum(1 for s in results if s.get("_upload_status") == "uploaded")
        quota_exceeded = sum(1 for s in results if s.get("_upload_status") == "quota_exceeded")
        errors = sum(1 for s in results if s.get("_upload_status") == "error")

        _batch_state["upload_index"] = idx + 1
        logger.info(f"[Phase 5] Batch {idx + 1} done: uploaded={uploaded}, "
                     f"quota_exceeded={quota_exceeded}, errors={errors}")

        if _batch_state["upload_index"] >= len(batches):
            _batch_state["phase_completed"].add("phase5")
            logger.info("[Phase 5] ALL uploads complete for today")
    except Exception as e:
        logger.error(f"[Phase 5] Batch {idx + 1} upload failed: {e}")


# =====================================================================
#  Incremental Mode Job
# =====================================================================

def incremental_story_job(job_name: str):
    """Create one story video end-to-end (incremental mode)."""
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


# =====================================================================
#  Shared Jobs (Both Modes)
# =====================================================================

def analytics_job():
    """Update analytics and refresh trending keywords for all channels and languages."""
    logger.info("[Analytics] Running analytics + keyword update (all channels)...")
    try:
        run_analytics_update()

        # Fetch real YouTube Analytics data (views, CTR, search terms)
        try:
            from scripts.youtube_analytics import update_analytics_database
            update_analytics_database()
        except Exception as e:
            logger.warning(f"[Analytics] YouTube Analytics API update failed: {e}")

        # Refresh trending keywords across all languages and content types
        from scripts.keyword_optimizer import run_keyword_refresh, integrate_analytics_data
        run_keyword_refresh()  # This now iterates all content_type/language combos

        # Integrate real search terms from Analytics API into keyword database
        try:
            integrate_analytics_data()
        except Exception as e:
            logger.warning(f"[Analytics] Search term integration failed: {e}")

        # Log content strategy insights
        from scripts.story_generator import get_top_performing_themes
        top = get_top_performing_themes(limit=5)
        if top:
            logger.info("[Analytics] Top performing themes:")
            for t in top:
                logger.info(f"  {t['collection']}/{t['moral']}: "
                           f"avg_views={t['avg_views']:.0f}, count={t['story_count']}")

        # Log AI education progress
        try:
            from scripts.ai_education_generator import get_ai_lesson_progress
            progress = get_ai_lesson_progress()
            logger.info(f"[Analytics] AI Education: {progress['generated']}/{progress['total_lessons']} "
                       f"lessons generated ({progress['remaining']} remaining)")
        except ImportError:
            pass

        logger.info("[Analytics] Update complete.")
    except Exception as e:
        logger.error(f"[Analytics] Update failed: {e}")


def health_check_job():
    """Hourly health check: GPU memory, disk space, upload progress."""
    logger.info("[Health] Running health check...")

    # --- Disk space ---
    try:
        total, used, free = shutil.disk_usage("/mnt")
        free_gb = free / (1024 ** 3)
        total_gb = total / (1024 ** 3)
        used_pct = (used / total) * 100
        logger.info(f"[Health] Disk: {free_gb:.1f}GB free / {total_gb:.1f}GB total ({used_pct:.0f}% used)")
        if free_gb < 5.0:
            logger.warning(f"[Health] LOW DISK SPACE: {free_gb:.1f}GB free")
    except Exception as e:
        logger.error(f"[Health] Disk check failed: {e}")

    # --- GPU memory ---
    try:
        import torch
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                total_mem = torch.cuda.get_device_properties(i).total_mem / (1024 ** 3)
                allocated = torch.cuda.memory_allocated(i) / (1024 ** 3)
                reserved = torch.cuda.memory_reserved(i) / (1024 ** 3)
                free_mem = total_mem - reserved
                logger.info(f"[Health] GPU {i} ({torch.cuda.get_device_properties(i).name}): "
                           f"allocated={allocated:.1f}GB, reserved={reserved:.1f}GB, "
                           f"free={free_mem:.1f}GB / {total_mem:.1f}GB")
                if free_mem < 2.0:
                    logger.warning(f"[Health] LOW GPU MEMORY on GPU {i}: {free_mem:.1f}GB free")
        else:
            logger.info("[Health] GPU: No CUDA devices available")
    except ImportError:
        logger.info("[Health] GPU: torch not available for GPU check")
    except Exception as e:
        logger.error(f"[Health] GPU check failed: {e}")

    # --- Upload progress ---
    try:
        stats = get_all_channel_stats()
        total_uploaded_today = 0
        for key, stat in stats.items():
            uploads = stat.get("uploads_today", 0)
            remaining = stat.get("max_more_uploads", 0)
            total_uploaded_today += uploads
            logger.info(f"[Health] Channel '{stat['name']}': "
                       f"{uploads} uploaded today, {remaining} more possible")
        total_target = STORIES_PER_DAY + EDUCATION_PER_DAY + AI_EDUCATION_PER_DAY + CRAFTS_SKILLS_PER_DAY
        logger.info(f"[Health] Upload progress: {total_uploaded_today}/{total_target} total today")
    except Exception as e:
        logger.error(f"[Health] Upload stats check failed: {e}")

    # --- Batch pipeline state (batch mode) ---
    if PIPELINE_MODE == "batch":
        phases_done = sorted(_batch_state.get("phase_completed", set()))
        scripts_count = len(_batch_state.get("all_scripts", []))
        upload_idx = _batch_state.get("upload_index", 0)
        total_batches = len(_batch_state.get("upload_batches", []))
        logger.info(f"[Health] Batch state: phases_completed={phases_done}, "
                   f"scripts={scripts_count}, uploads={upload_idx}/{total_batches} batches")

    # --- Output directory sizes ---
    try:
        for subdir in ["videos", "audio", "images", "thumbnails"]:
            dirpath = OUTPUT_DIR / subdir
            if dirpath.exists():
                size = sum(f.stat().st_size for f in dirpath.rglob("*") if f.is_file())
                size_gb = size / (1024 ** 3)
                count = sum(1 for f in dirpath.rglob("*") if f.is_file())
                logger.info(f"[Health] Output/{subdir}: {count} files, {size_gb:.2f}GB")
    except Exception as e:
        logger.error(f"[Health] Output size check failed: {e}")

    logger.info("[Health] Health check complete.")


def check_disk_space() -> bool:
    """Check if there's enough disk space to create a video."""
    total, used, free = shutil.disk_usage("/mnt")
    free_gb = free / (1024 ** 3)
    if free_gb < 2.0:
        logger.warning(f"Low disk space: {free_gb:.1f}GB free. Cleaning old output files...")
        import glob
        for subdir in ["videos", "audio", "images"]:
            for vdir in sorted(glob.glob(str(OUTPUT_DIR / subdir / "story_*"))):
                vpath = Path(vdir)
                if vpath.is_dir():
                    age_days = (time.time() - vpath.stat().st_mtime) / 86400
                    if age_days > 3:
                        shutil.rmtree(vdir)
                        logger.info(f"  Cleaned old dir: {vdir}")
        return free_gb >= 1.0
    return True


# =====================================================================
#  Re-edit Job (Weekly)
# =====================================================================

def reedit_job():
    """Re-edit the 3 lowest-performing published videos with enhanced quality."""
    logger.info("[Re-edit] Starting weekly re-edit of lowest-performing videos...")
    try:
        from scripts.reupload_videos import batch_reedit
        batch_reedit(count=3, upload=True, update_existing=False)
        logger.info("[Re-edit] Weekly re-edit complete.")
    except ImportError:
        logger.warning("[Re-edit] reupload_videos.py not found - skipping")
    except Exception as e:
        logger.error(f"[Re-edit] Failed: {e}")


# =====================================================================
#  Schedule Setup
# =====================================================================

def setup_batch_schedule():
    """Configure schedule for batch mode: phased generation + distributed uploads."""
    total_target = STORIES_PER_DAY + EDUCATION_PER_DAY + AI_EDUCATION_PER_DAY + CRAFTS_SKILLS_PER_DAY

    # Phase 1: Generate all scripts at 01:00
    schedule.every().day.at("01:00").do(phase1_scripts_job)

    # Phase 2: Generate all audio at 02:00
    schedule.every().day.at("02:00").do(phase2_audio_job)

    # Phase 3: Generate all images at 03:00 (longest phase, ~5h for 100 videos)
    schedule.every().day.at("03:00").do(phase3_images_job)

    # Phase 4: Assemble all videos at 08:00
    schedule.every().day.at("08:00").do(phase4_assembly_job)

    # Phase 5: Upload in batches every hour from 10:00-22:00 UTC
    # 13 upload windows for distributing uploads throughout the day
    upload_times = [
        "10:00", "11:00", "12:00", "13:00", "14:00", "15:00",
        "16:00", "17:00", "18:00", "19:00", "20:00", "21:00", "22:00",
    ]
    for t in upload_times:
        schedule.every().day.at(t).do(phase5_upload_job)

    logger.info(f"Batch schedule configured ({total_target} videos/day):")
    logger.info(f"  01:00 UTC - Phase 1: Generate {total_target} scripts (LLM)")
    logger.info(f"  02:00 UTC - Phase 2: Generate audio (TTS)")
    logger.info(f"  03:00 UTC - Phase 3: Generate images (GPU)")
    logger.info(f"  08:00 UTC - Phase 4: Assemble videos (FFmpeg)")
    logger.info(f"  10:00-22:00 UTC - Phase 5: Upload batches (every hour)")

    # Quota info
    config = load_channel_config()
    max_per_channel = config["global_settings"].get("max_uploads_per_channel", 6)
    channels = len(config["channels"])
    max_total = max_per_channel * channels
    logger.info(f"  Upload capacity: {max_per_channel} per channel x {channels} channels = {max_total}/day")
    if total_target > max_total:
        logger.warning(f"  WARNING: Target {total_target} exceeds max upload capacity {max_total}. "
                       f"Some videos will be queued.")


def setup_incremental_schedule():
    """Configure schedule for incremental mode: generate+upload one at a time."""
    total_target = STORIES_PER_DAY + EDUCATION_PER_DAY + AI_EDUCATION_PER_DAY + CRAFTS_SKILLS_PER_DAY

    # Spread individual jobs throughout the day
    # For 100 videos over ~20 hours (02:00-22:00), that is ~5 per hour
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

    if total_target <= 12:
        # Use predefined time slots for small counts
        times = schedule_times.get(total_target, schedule_times[10])
    else:
        # Generate evenly spaced times across 02:00-22:00 (20 hours)
        times = []
        interval_minutes = (20 * 60) // total_target
        for i in range(total_target):
            offset_minutes = 2 * 60 + i * interval_minutes  # Start at 02:00
            hour = offset_minutes // 60
            minute = offset_minutes % 60
            if hour < 24:
                times.append(f"{hour:02d}:{minute:02d}")

    for i, t in enumerate(times):
        name = f"Video-{i + 1}"
        schedule.every().day.at(t).do(incremental_story_job, job_name=name)

    logger.info(f"Incremental schedule configured ({total_target} videos/day):")
    for t in times[:5]:
        logger.info(f"  {t} UTC - Generate + upload video")
    if len(times) > 5:
        logger.info(f"  ... and {len(times) - 5} more time slots")
    logger.info(f"  = {len(times)} videos per day (incremental mode)")

    # Quota warning
    quota_needed = total_target * 1600
    if quota_needed > 10000:
        logger.warning(f"  WARNING: {total_target} uploads need {quota_needed} quota units/day "
                       f"(standard limit: 10,000 per channel). Ensure quota increase is approved.")


def setup_schedule():
    """Configure the daily schedule based on PIPELINE_MODE."""

    # ----- Common jobs for both modes -----

    # Analytics + keyword refresh every 6 hours
    schedule.every().day.at("00:00").do(analytics_job)
    schedule.every().day.at("06:00").do(analytics_job)
    schedule.every().day.at("12:00").do(analytics_job)
    schedule.every().day.at("18:00").do(analytics_job)

    # Health check every hour
    schedule.every(1).hours.do(health_check_job)

    # Re-edit low-performing videos weekly (Sundays at 23:00)
    schedule.every().sunday.at("23:00").do(reedit_job)

    logger.info("Common jobs:")
    logger.info("  Every 6h - Analytics + keyword refresh (all languages)")
    logger.info("  Every 1h - Health check (GPU, disk, uploads)")
    logger.info("  Weekly (Sun 23:00) - Re-edit 3 lowest-performing videos")

    # ----- Mode-specific schedule -----
    if PIPELINE_MODE == "batch":
        setup_batch_schedule()
    elif PIPELINE_MODE == "incremental":
        setup_incremental_schedule()
    else:
        logger.error(f"Unknown PIPELINE_MODE: {PIPELINE_MODE}. Defaulting to batch.")
        setup_batch_schedule()


# =====================================================================
#  Main Entry Point
# =====================================================================

def main():
    """Main scheduler loop - fully autonomous multi-channel pipeline.

    Runs 24/7 generating and uploading videos across 18 YouTube channels
    in 6 languages with 3 content types. No human intervention needed.
    """
    total_target = STORIES_PER_DAY + EDUCATION_PER_DAY + AI_EDUCATION_PER_DAY + CRAFTS_SKILLS_PER_DAY

    config = load_channel_config()
    num_channels = len(config.get("channels", {}))

    logger.info("=" * 70)
    logger.info("Little Wisdom Tales - AUTONOMOUS Multi-Channel Pipeline")
    logger.info(f"  Mode: {PIPELINE_MODE}")
    logger.info(f"  Channels: {num_channels}")
    logger.info(f"  Stories/day: {STORIES_PER_DAY} (6 languages)")
    logger.info(f"  Education/day: {EDUCATION_PER_DAY} (Oxford+Cambridge, EN/HI/ES/AR)")
    logger.info(f"  AI Education/day: {AI_EDUCATION_PER_DAY} (EN/HI/ES/AR)")
    logger.info(f"  Crafts/Skills/day: {CRAFTS_SKILLS_PER_DAY} (3 channels)")
    logger.info(f"  Total target: {total_target} videos/day")
    logger.info(f"  Video duration: {MIN_VIDEO_DURATION}-{MAX_VIDEO_DURATION}s")
    logger.info("=" * 70)

    setup_schedule()

    # Run initial keyword refresh on startup
    logger.info("Running initial keyword refresh...")
    try:
        from scripts.keyword_optimizer import run_keyword_refresh
        run_keyword_refresh()
    except Exception as e:
        logger.warning(f"Initial keyword refresh failed: {e}")

    # Run initial health check
    logger.info("Running initial health check...")
    health_check_job()

    # On startup: decide whether to run immediately
    if PIPELINE_MODE == "batch":
        # In batch mode, check if today's phases have been missed and catch up
        current_hour = datetime.utcnow().hour

        if current_hour < 1:
            logger.info("Before Phase 1 start time (01:00). Waiting for scheduled run.")
        elif current_hour < 10:
            logger.info("Catching up on missed batch phases...")
            if current_hour >= 1 and "phase1" not in _batch_state.get("phase_completed", set()):
                phase1_scripts_job()
            if current_hour >= 2 and "phase2" not in _batch_state.get("phase_completed", set()):
                phase2_audio_job()
            if current_hour >= 3 and "phase3" not in _batch_state.get("phase_completed", set()):
                phase3_images_job()
            if current_hour >= 8 and "phase4" not in _batch_state.get("phase_completed", set()):
                phase4_assembly_job()
        else:
            logger.info("Late start - running full batch pipeline now...")
            phase1_scripts_job()
            phase2_audio_job()
            phase3_images_job()
            phase4_assembly_job()
            phase5_upload_job()
    else:
        # In incremental mode, run one story on startup as a quick test
        if check_disk_space():
            logger.info("Running initial story on startup (incremental mode)...")
            incremental_story_job("Startup")
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
    # Support command-line mode override: python scheduler.py [batch|incremental]
    if len(sys.argv) > 1 and sys.argv[1] in ("batch", "incremental"):
        PIPELINE_MODE = sys.argv[1]
    main()
