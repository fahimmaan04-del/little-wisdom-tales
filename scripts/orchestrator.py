"""
Master Orchestrator - End-to-end pipeline for automated story video creation.

This is the main entry point that coordinates all components:
1. Generate story script (LLM)
2. Generate voiceover audio (Edge TTS)
3. Fetch scene images (Pixabay/Pexels)
4. Assemble video with effects (FFmpeg)
5. Upload to YouTube (Data API)
6. Track analytics and optimize

Can be triggered by n8n cron or run standalone via cron/systemd.
"""

import json
import logging
import os
import shutil
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

load_dotenv()

from scripts.story_generator import generate_story_script, get_db, pick_smart_story_params
from scripts.tts_generator import generate_audio_sync
from scripts.image_fetcher import fetch_scene_images
from scripts.video_assembler import assemble_story_video
from scripts.engagement_hooks import (
    inject_engagement_hooks,
    generate_engaging_title,
    generate_description,
)
from scripts.youtube_manager import (
    upload_video,
    mark_story_published,
    update_analytics_db,
    create_or_get_playlist,
    add_to_playlist,
)

OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "./output"))
DATA_DIR = Path(os.getenv("DATA_DIR", "./data"))
LOG_DIR = DATA_DIR / "logs"

# Setup logging
LOG_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "orchestrator.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def create_single_story(
    collection: str = None,
    moral: str = None,
    for_shorts: bool = False,
    upload: bool = True,
    smart_pick: bool = True,
) -> dict:
    """
    Complete pipeline: generate story -> create video -> upload.

    Returns dict with story details and upload results.
    """
    result = {"status": "started", "timestamp": datetime.now().isoformat()}

    try:
        # Step 1: Generate story script
        logger.info("=" * 60)
        logger.info("STEP 1: Generating story script...")

        if smart_pick and not collection and not moral:
            params = pick_smart_story_params()
            collection = params.get("collection")
            moral = params.get("moral")

        story_script = generate_story_script(
            collection=collection,
            moral=moral,
            for_shorts=for_shorts,
        )
        story_id = story_script["story_id"]

        # Inject engagement hooks (intro, midpoint, outro with subscribe CTA)
        story_script = inject_engagement_hooks(story_script, is_shorts=for_shorts)

        # Generate optimized title
        title = generate_engaging_title(
            story_title=story_script.get("title", "Untitled Story"),
            moral=story_script.get("moral", ""),
            origin=story_script.get("origin", ""),
            is_shorts=for_shorts,
        )
        story_script["display_title"] = title

        logger.info(f"  Story: '{title}' (ID: {story_id})")
        logger.info(f"  Collection: {story_script.get('collection')}")
        logger.info(f"  Moral: {story_script.get('moral')}")
        logger.info(f"  Scenes (with hooks): {len(story_script.get('scenes', []))}")

        result["story_id"] = story_id
        result["title"] = title

        # Step 2: Generate voiceover audio
        logger.info("STEP 2: Generating voiceover audio...")
        audio_data = generate_audio_sync(story_script, story_id)
        logger.info(f"  Total audio duration: {audio_data['total_duration']:.1f}s")

        result["audio_duration"] = audio_data["total_duration"]

        # Step 3: Fetch scene images
        logger.info("STEP 3: Fetching scene images...")
        image_data = fetch_scene_images(story_script, story_id, for_shorts=for_shorts)
        logger.info(f"  Fetched {len(image_data)} images")

        result["images_fetched"] = len(image_data)

        # Step 4: Assemble video
        logger.info("STEP 4: Assembling video...")
        video_data = assemble_story_video(
            story_script=story_script,
            audio_data=audio_data,
            image_data=image_data,
            story_id=story_id,
            for_shorts=for_shorts,
        )
        logger.info(f"  Video created: {video_data['video_path']}")
        logger.info(f"  Duration: {video_data['duration_seconds']:.1f}s")

        result["video_path"] = video_data["video_path"]
        result["video_duration"] = video_data["duration_seconds"]

        # Step 5: Upload to YouTube
        if upload:
            logger.info("STEP 5: Uploading to YouTube...")
            description = generate_description(story_script, is_shorts=for_shorts)
            tags = story_script.get("tags", [])

            upload_result = upload_video(
                video_path=video_data["video_path"],
                title=title,
                description=description,
                tags=tags,
                is_shorts=for_shorts,
                thumbnail_path=video_data.get("thumbnail_path"),
                made_for_kids=True,
            )

            mark_story_published(story_id, upload_result["video_id"])
            logger.info(f"  Published: {upload_result['url']}")

            # Add to appropriate playlist
            try:
                collection_name = story_script.get("collection", "stories")
                playlist_title = f"Kids Stories - {collection_name.replace('_', ' ').title()}"
                playlist_id = create_or_get_playlist(playlist_title)
                add_to_playlist(playlist_id, upload_result["video_id"])
                logger.info(f"  Added to playlist: {playlist_title}")
            except Exception as e:
                logger.warning(f"  Playlist operation failed: {e}")

            result["upload"] = upload_result
        else:
            logger.info("STEP 5: Upload skipped (upload=False)")

        # Cleanup temporary files (keep final video + thumbnail)
        cleanup_temp_files(story_id)

        result["status"] = "completed"
        logger.info(f"Pipeline completed for story '{title}'")
        logger.info("=" * 60)

    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)
        result["traceback"] = traceback.format_exc()
        logger.error(f"Pipeline failed: {e}")
        logger.error(traceback.format_exc())

    # Save result to log
    result_path = DATA_DIR / "logs" / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2, default=str)

    return result


def build_video_description(story_script: dict) -> str:
    """Build YouTube video description from story script."""
    title = story_script.get("title", "")
    moral = story_script.get("moral", "")
    collection = story_script.get("collection", "").replace("_", " ").title()
    origin = story_script.get("origin", "")
    description = story_script.get("description", "")

    return f"""{description}

{moral}

This story comes from the {collection} collection, originating from {origin}.

---
Little Wisdom Tales brings you beautiful moral stories from around the world!
New stories every day for kids aged 4-10.

#KidsStories #MoralStories #BedtimeStories #ChildrenStories #LittleWisdomTales
#StoriesForKids #{collection.replace(' ', '')}
"""


def cleanup_temp_files(story_id: int):
    """Remove intermediate files, keep only final video + thumbnail."""
    audio_dir = OUTPUT_DIR / "audio" / f"story_{story_id}"
    video_dir = OUTPUT_DIR / "videos" / f"story_{story_id}"
    image_dir = OUTPUT_DIR / "images" / f"story_{story_id}"

    # Remove scene-level audio files (keep combined)
    if audio_dir.exists():
        for f in audio_dir.glob("scene_*"):
            f.unlink()

    # Remove scene-level video clips (keep final)
    if video_dir.exists():
        for f in video_dir.iterdir():
            if "_final" not in f.name and "_raw" not in f.name:
                f.unlink()

    # Remove raw images (keep resized)
    if image_dir.exists():
        for f in image_dir.glob("*_raw.*"):
            f.unlink()


def run_daily_batch():
    """Run the daily batch of story creation."""
    videos_per_day = int(os.getenv("VIDEOS_PER_DAY", "3"))
    publish_shorts = os.getenv("PUBLISH_AS_SHORTS", "true").lower() == "true"
    publish_video = os.getenv("PUBLISH_AS_VIDEO", "true").lower() == "true"

    logger.info(f"Starting daily batch: {videos_per_day} stories")

    results = []
    for i in range(videos_per_day):
        logger.info(f"\n--- Story {i+1}/{videos_per_day} ---")

        # Create regular video version
        if publish_video:
            result = create_single_story(for_shorts=False, smart_pick=True)
            results.append(result)

            if result["status"] == "error":
                logger.error(f"Story {i+1} (video) failed, continuing...")
                time.sleep(10)
                continue

        # Create shorts version of the same story (or a new one)
        if publish_shorts:
            result = create_single_story(for_shorts=True, smart_pick=True)
            results.append(result)

            if result["status"] == "error":
                logger.error(f"Story {i+1} (shorts) failed, continuing...")

        # Wait between stories to respect API rate limits
        if i < videos_per_day - 1:
            wait_time = 300  # 5 minutes between stories
            logger.info(f"Waiting {wait_time}s before next story...")
            time.sleep(wait_time)

    # Update analytics for all published videos
    logger.info("\nUpdating analytics for all published videos...")
    try:
        update_analytics_db()
    except Exception as e:
        logger.warning(f"Analytics update failed: {e}")

    # Summary
    successful = sum(1 for r in results if r["status"] == "completed")
    failed = sum(1 for r in results if r["status"] == "error")
    logger.info(f"\nDaily batch complete: {successful} succeeded, {failed} failed")

    return results


def run_analytics_update():
    """Update analytics and generate content strategy insights."""
    logger.info("Running analytics update...")
    try:
        update_analytics_db()
        logger.info("Analytics updated successfully")
    except Exception as e:
        logger.error(f"Analytics update failed: {e}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Kids Story Channel Orchestrator")
    parser.add_argument(
        "command",
        choices=["single", "batch", "analytics", "test"],
        help="Command to run",
    )
    parser.add_argument("--collection", help="Story collection to use")
    parser.add_argument("--moral", help="Moral theme to use")
    parser.add_argument("--shorts", action="store_true", help="Create as YouTube Shorts")
    parser.add_argument("--no-upload", action="store_true", help="Skip YouTube upload")
    parser.add_argument("--count", type=int, default=1, help="Number of stories to create")

    args = parser.parse_args()

    if args.command == "single":
        results = []
        for i in range(args.count):
            if args.count > 1:
                print(f"\n=== Story {i+1}/{args.count} ===")
            result = create_single_story(
                collection=args.collection,
                moral=args.moral,
                for_shorts=args.shorts,
                upload=not args.no_upload,
            )
            results.append(result)
            if args.count > 1 and i < args.count - 1:
                time.sleep(30)  # Brief pause between stories
        if args.count == 1:
            print(json.dumps(results[0], indent=2, default=str))
        else:
            ok = sum(1 for r in results if r["status"] == "completed")
            print(f"\n=== Batch complete: {ok}/{args.count} succeeded ===")

    elif args.command == "batch":
        results = run_daily_batch()
        print(json.dumps(results, indent=2, default=str))

    elif args.command == "analytics":
        run_analytics_update()

    elif args.command == "test":
        # Test run without uploading
        result = create_single_story(
            for_shorts=False,
            upload=False,
            smart_pick=False,
        )
        print(json.dumps(result, indent=2, default=str))
