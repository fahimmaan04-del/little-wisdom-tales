"""
Re-upload Script - Regenerate existing videos with AI-generated images.

Fetches story scripts from the database, regenerates images using SDXL
Turbo, reassembles videos with properly aligned audio and motion effects,
generates multi-language subtitles, and re-uploads to YouTube.

This replaces the original stock-photo videos with AI-illustrated versions.
"""

import json
import logging
import os
import sqlite3
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))
load_dotenv()

from scripts.tts_generator import generate_audio_sync
from scripts.ai_image_generator import generate_story_images, unload_model
from scripts.video_assembler import assemble_story_video
from scripts.engagement_hooks import (
    inject_engagement_hooks,
    generate_engaging_title,
    generate_description,
)
from scripts.youtube_manager import upload_video, mark_story_published
from scripts.keyword_optimizer import optimize_title, optimize_tags, optimize_description
from scripts.subtitle_generator import generate_subtitles_for_story

DATA_DIR = Path(os.getenv("DATA_DIR", "./data"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(DATA_DIR / "logs" / "reupload.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def get_published_stories() -> list[dict]:
    """Get all published stories from the database."""
    db_path = DATA_DIR / "stories.db"
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    stories = conn.execute("""
        SELECT id, title, collection, region, moral, script, video_id
        FROM stories
        WHERE status = 'published' AND video_id IS NOT NULL
        ORDER BY id ASC
    """).fetchall()

    conn.close()
    return [dict(s) for s in stories]


def reprocess_story(story: dict, upload: bool = True) -> dict:
    """Regenerate a single story with AI images and re-upload.

    Args:
        story: Story record from database with script JSON.
        upload: Whether to upload to YouTube (False for testing).

    Returns:
        Dict with reprocessing results.
    """
    story_id = story["id"]
    old_video_id = story["video_id"]

    logger.info(f"=" * 60)
    logger.info(f"Reprocessing story {story_id}: {story['title']}")
    logger.info(f"  Old video: https://youtube.com/watch?v={old_video_id}")

    result = {"story_id": story_id, "old_video_id": old_video_id}

    try:
        # Parse the stored script
        script = json.loads(story["script"]) if isinstance(story["script"], str) else story["script"]

        # Re-inject engagement hooks
        script = inject_engagement_hooks(script, is_shorts=False)

        # Generate optimized title with trending keywords
        title = generate_engaging_title(
            story_title=script.get("title", story["title"]),
            moral=script.get("moral", story.get("moral", "")),
            origin=script.get("origin", ""),
        )
        title = optimize_title(
            title,
            moral=script.get("moral", ""),
            collection=script.get("collection", ""),
        )

        logger.info(f"  New title: {title}")

        # Step 1: Regenerate voiceover audio
        logger.info(f"  Step 1: Generating voiceover...")
        audio_data = generate_audio_sync(script, story_id)
        logger.info(f"  Audio duration: {audio_data['total_duration']:.1f}s")

        # Step 2: Generate AI images
        logger.info(f"  Step 2: Generating AI images...")
        image_data = generate_story_images(
            script, story_id,
            for_shorts=False,
            images_per_scene=30,
        )
        total_imgs = sum(len(d.get("all_images", [1])) for d in image_data)
        logger.info(f"  Generated {total_imgs} images for {len(image_data)} scenes")

        # Step 3: Assemble video
        logger.info(f"  Step 3: Assembling video...")
        video_data = assemble_story_video(
            story_script=script,
            audio_data=audio_data,
            image_data=image_data,
            story_id=story_id,
            for_shorts=False,
        )
        logger.info(f"  Video: {video_data['video_path']} ({video_data['duration_seconds']:.1f}s)")

        # Step 4: Generate subtitles
        logger.info(f"  Step 4: Generating subtitles...")
        try:
            subtitle_data = generate_subtitles_for_story(script, audio_data, story_id)
            logger.info(f"  Subtitles: {', '.join(subtitle_data.get('languages', ['en']))}")
        except Exception as e:
            subtitle_data = None
            logger.warning(f"  Subtitle generation failed: {e}")

        result["video_path"] = video_data["video_path"]
        result["duration"] = video_data["duration_seconds"]

        # Step 5: Upload to YouTube
        if upload:
            logger.info(f"  Step 5: Uploading to YouTube...")
            description = generate_description(script, is_shorts=False)
            tags = optimize_tags(script.get("tags", []), story_script=script)
            description = optimize_description(description, story_script=script)

            upload_result = upload_video(
                video_path=video_data["video_path"],
                title=title,
                description=description,
                tags=tags,
                is_shorts=False,
                thumbnail_path=video_data.get("thumbnail_path"),
                made_for_kids=True,
            )

            new_video_id = upload_result["video_id"]
            mark_story_published(story_id, new_video_id)
            logger.info(f"  New video: {upload_result['url']}")

            # Upload subtitles
            if subtitle_data and subtitle_data.get("caption_files"):
                try:
                    from scripts.subtitle_generator import upload_captions_to_youtube
                    upload_captions_to_youtube(new_video_id, subtitle_data["caption_files"])
                    logger.info("  Subtitles uploaded")
                except Exception as e:
                    logger.warning(f"  Caption upload failed: {e}")

            result["new_video_id"] = new_video_id
            result["new_url"] = upload_result["url"]
            result["status"] = "uploaded"
        else:
            logger.info("  Step 5: Upload skipped (test mode)")
            result["status"] = "generated"

    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)
        logger.error(f"  Reprocessing failed: {e}")
        import traceback
        logger.error(traceback.format_exc())

    return result


def main():
    """Reprocess and re-upload all published stories."""
    import argparse

    parser = argparse.ArgumentParser(description="Re-upload videos with AI images")
    parser.add_argument("--no-upload", action="store_true", help="Skip YouTube upload (test mode)")
    parser.add_argument("--story-id", type=int, help="Reprocess only this story ID")
    parser.add_argument("--limit", type=int, default=0, help="Max stories to process (0 = all)")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Little Wisdom Tales - Video Re-upload with AI Images")
    logger.info("=" * 60)

    stories = get_published_stories()
    logger.info(f"Found {len(stories)} published stories")

    if args.story_id:
        stories = [s for s in stories if s["id"] == args.story_id]
        if not stories:
            logger.error(f"Story ID {args.story_id} not found")
            return

    if args.limit > 0:
        stories = stories[:args.limit]

    results = []
    for i, story in enumerate(stories):
        logger.info(f"\n--- Story {i+1}/{len(stories)} ---")
        result = reprocess_story(story, upload=not args.no_upload)
        results.append(result)

        # Wait between uploads to respect API rate limits
        if not args.no_upload and i < len(stories) - 1:
            logger.info("Waiting 60s before next story...")
            time.sleep(60)

    # Free GPU memory
    try:
        unload_model()
    except Exception:
        pass

    # Summary
    successful = sum(1 for r in results if r["status"] in ("uploaded", "generated"))
    failed = sum(1 for r in results if r["status"] == "error")
    logger.info(f"\n{'=' * 60}")
    logger.info(f"Re-upload complete: {successful} succeeded, {failed} failed")
    logger.info(f"{'=' * 60}")

    # Save results
    results_path = DATA_DIR / "logs" / "reupload_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()
