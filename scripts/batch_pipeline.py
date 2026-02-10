"""
Batch Pipeline - High-throughput video production for 100+ videos per day.

Coordinates parallel generation across story and education content types,
manages GPU resources for image generation, and distributes uploads across
multiple YouTube channels to respect API quotas.

Architecture:
  Phase 1: Generate all scripts (LLM) - fast, ~10s each
  Phase 2: Generate all audio (TTS) - async, ~10s each
  Phase 3: Generate all images (GPU) - sequential, ~2min each
  Phase 4: Assemble all videos (FFmpeg) - ~30s each
  Phase 5: Upload in batches (YouTube API) - quota-limited

100 videos at ~3min/video = ~5 hours total pipeline time.
"""

import json
import logging
import os
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))
load_dotenv()

from scripts.story_generator import generate_story_script, pick_smart_story_params
from scripts.tts_generator import generate_audio_sync
from scripts.ai_image_generator import generate_story_images, unload_model
from scripts.video_assembler import assemble_story_video
from scripts.engagement_hooks import (
    inject_engagement_hooks,
    generate_engaging_title,
    generate_description,
)
from scripts.keyword_optimizer import optimize_title, optimize_tags, optimize_description
from scripts.subtitle_generator import generate_subtitles_for_story
from scripts.regional_content import create_regional_version, get_regional_tts_voice
from scripts.channel_manager import (
    load_channel_config,
    upload_to_channel,
    can_upload,
    get_or_create_playlist,
    add_video_to_playlist,
    get_all_channel_stats,
)

DATA_DIR = Path(os.getenv("DATA_DIR", "./data"))
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "./output"))
LOG_DIR = DATA_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "batch_pipeline.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def generate_story_batch(count: int) -> list:
    """Phase 1a: Generate story scripts via LLM."""
    logger.info(f"Generating {count} story scripts...")
    scripts = []
    for i in range(count):
        try:
            params = pick_smart_story_params()
            script = generate_story_script(
                collection=params.get("collection"),
                moral=params.get("moral"),
                for_shorts=False,
            )
            script = inject_engagement_hooks(script, is_shorts=False)

            title = generate_engaging_title(
                story_title=script.get("title", "Untitled"),
                moral=script.get("moral", ""),
                origin=script.get("origin", ""),
            )
            title = optimize_title(
                title,
                moral=script.get("moral", ""),
                collection=script.get("collection", ""),
                language="en",
                content_type="story",
            )
            script["display_title"] = title
            script["content_type"] = "story"
            script["channel_key"] = "stories"

            scripts.append(script)
            logger.info(f"  Story {i+1}/{count}: {title}")
        except Exception as e:
            logger.error(f"  Story {i+1} generation failed: {e}")
    return scripts


def generate_education_batch(count: int, curriculum: str = "oxford") -> list:
    """Phase 1b: Generate education lesson scripts via LLM."""
    logger.info(f"Generating {count} education scripts ({curriculum})...")

    try:
        from scripts.education_generator import generate_lesson_script, pick_next_lesson
    except ImportError:
        logger.error("education_generator.py not found - skipping education batch")
        return []

    scripts = []
    for i in range(count):
        try:
            lesson = pick_next_lesson(curriculum=curriculum)
            if not lesson:
                logger.warning(f"  No more lessons available for {curriculum}")
                break

            script = generate_lesson_script(
                class_level=lesson["class_level"],
                subject=lesson["subject"],
                topic=lesson["topic"],
                lesson_title=lesson["lesson_title"],
                curriculum=curriculum,
            )
            script["content_type"] = "education"
            channel_key = f"education_{curriculum}"
            script["channel_key"] = channel_key
            script["playlist_key"] = f"class_{lesson['class_level']}_{lesson['subject']}"

            scripts.append(script)
            logger.info(f"  Lesson {i+1}/{count}: {script.get('display_title', script.get('title', ''))}")
        except Exception as e:
            logger.error(f"  Lesson {i+1} generation failed: {e}")
    return scripts


def generate_ai_education_batch(count: int, language: str = "en") -> list:
    """Phase 1c: Generate AI education lesson scripts via LLM."""
    logger.info(f"Generating {count} AI education scripts (lang={language})...")

    try:
        from scripts.ai_education_generator import generate_ai_lesson_script, pick_next_ai_lesson
    except ImportError:
        logger.error("ai_education_generator.py not found - skipping AI education batch")
        return []

    scripts = []
    for i in range(count):
        try:
            lesson = pick_next_ai_lesson()
            if not lesson:
                logger.warning("  No more AI lessons available in curriculum")
                break

            script = generate_ai_lesson_script(**lesson)

            # Translate if not English
            if language != "en":
                script = create_regional_version(script, language, content_type="ai_education")

            script["content_type"] = "ai_education"
            script["language"] = language
            channel_key = f"ai_kids_{language}"
            script["channel_key"] = channel_key
            script["playlist_key"] = lesson["category"]

            scripts.append(script)
            logger.info(f"  AI Lesson {i+1}/{count}: {script.get('title', '')[:60]}")
        except Exception as e:
            logger.error(f"  AI Lesson {i+1} generation failed: {e}")
    return scripts


def generate_regional_story_batch(count: int, language: str, channel_key: str) -> list:
    """Phase 1d: Generate regional story scripts by translating English stories."""
    logger.info(f"Generating {count} regional story scripts ({language}/{channel_key})...")

    config = load_channel_config()
    channel = config["channels"].get(channel_key, {})
    collections = channel.get("story_collections", [])

    scripts = []
    for i in range(count):
        try:
            # Generate English story with region-specific collection
            collection = collections[i % len(collections)] if collections else None
            params = pick_smart_story_params()
            if collection:
                params["collection"] = collection

            script = generate_story_script(
                collection=params.get("collection"),
                moral=params.get("moral"),
                for_shorts=False,
            )
            script = inject_engagement_hooks(script, is_shorts=False)

            # Create regional version (translates narration, keeps images in English)
            script = create_regional_version(script, language, content_type="story")

            # Optimize title with language-aware keywords
            title = generate_engaging_title(
                story_title=script.get("title", "Untitled"),
                moral=script.get("moral", ""),
                origin=script.get("origin", ""),
            )
            title = optimize_title(
                title,
                moral=script.get("moral", ""),
                collection=script.get("collection", ""),
                language=language,
                content_type="story",
            )
            script["display_title"] = title
            script["content_type"] = "story"
            script["language"] = language
            script["channel_key"] = channel_key

            scripts.append(script)
            logger.info(f"  Regional story {i+1}/{count} ({language}): {title[:50]}")
        except Exception as e:
            logger.error(f"  Regional story {i+1} ({language}) failed: {e}")
    return scripts


def generate_crafts_skills_batch(count: int, skill_category: str = "carpentry_building") -> list:
    """Phase 1e: Generate crafts/skills lesson scripts via LLM."""
    logger.info(f"Generating {count} crafts/skills scripts ({skill_category})...")

    try:
        from scripts.crafts_skills_generator import generate_crafts_lesson_script, pick_next_crafts_lesson
    except ImportError:
        logger.error("crafts_skills_generator.py not found - skipping crafts batch")
        return []

    scripts = []
    for i in range(count):
        try:
            lesson = pick_next_crafts_lesson(category=skill_category)
            if not lesson:
                logger.warning(f"  No more crafts lessons available for {skill_category}")
                break

            script = generate_crafts_lesson_script(**lesson)
            script["content_type"] = "crafts_skills"
            script["language"] = "en"

            # Determine channel key from skill category
            category_to_channel = {
                "carpentry_building": "crafts_carpentry",
                "plumbing_home": "crafts_plumbing",
                "electrical_tech": "crafts_electrical",
            }
            script["channel_key"] = category_to_channel.get(skill_category, "crafts_carpentry")
            script["playlist_key"] = lesson.get("topic", skill_category)

            scripts.append(script)
            logger.info(f"  Crafts {i+1}/{count}: {script.get('title', '')[:60]}")
        except Exception as e:
            logger.error(f"  Crafts lesson {i+1} generation failed: {e}")
    return scripts


def generate_audio_batch(scripts: list) -> list:
    """Phase 2: Generate TTS audio for all scripts (can be parallelized)."""
    logger.info(f"Generating audio for {len(scripts)} scripts...")
    results = []

    for i, script in enumerate(scripts):
        story_id = script.get("story_id", i + 1000)
        try:
            audio_data = generate_audio_sync(script, story_id)
            script["_audio_data"] = audio_data
            results.append(script)
            logger.info(f"  Audio {i+1}/{len(scripts)}: {audio_data['total_duration']:.1f}s")
        except Exception as e:
            logger.error(f"  Audio {i+1} failed: {e}")

    return results


def generate_images_batch(scripts: list, images_per_scene: int = 20) -> list:
    """Phase 3: Generate AI images for all scripts (GPU sequential)."""
    logger.info(f"Generating images for {len(scripts)} scripts ({images_per_scene} imgs/scene)...")
    results = []

    for i, script in enumerate(scripts):
        story_id = script.get("story_id", i + 1000)
        try:
            image_data = generate_story_images(
                script, story_id,
                for_shorts=False,
                images_per_scene=images_per_scene,
            )
            script["_image_data"] = image_data
            total_imgs = sum(len(d.get("all_images", [1])) for d in image_data)
            results.append(script)
            logger.info(f"  Images {i+1}/{len(scripts)}: {total_imgs} images for {len(image_data)} scenes")
        except Exception as e:
            logger.error(f"  Images {i+1} failed: {e}")
            # Try fallback to stock images
            try:
                from scripts.image_fetcher import fetch_scene_images
                image_data = fetch_scene_images(script, story_id, for_shorts=False)
                script["_image_data"] = image_data
                results.append(script)
                logger.info(f"  Images {i+1} (fallback): {len(image_data)} stock images")
            except Exception as e2:
                logger.error(f"  Fallback also failed: {e2}")

    # Free GPU memory after all images are generated
    try:
        unload_model()
    except Exception:
        pass

    return results


def assemble_videos_batch(scripts: list) -> list:
    """Phase 4: Assemble videos with FFmpeg."""
    logger.info(f"Assembling {len(scripts)} videos...")
    results = []

    for i, script in enumerate(scripts):
        story_id = script.get("story_id", i + 1000)
        audio_data = script.get("_audio_data")
        image_data = script.get("_image_data")

        if not audio_data or not image_data:
            logger.warning(f"  Video {i+1}: Missing audio or image data, skipping")
            continue

        try:
            video_data = assemble_story_video(
                story_script=script,
                audio_data=audio_data,
                image_data=image_data,
                story_id=story_id,
                for_shorts=False,
            )
            script["_video_data"] = video_data
            logger.info(f"  Video {i+1}/{len(scripts)}: {video_data['duration_seconds']:.1f}s")

            # Generate subtitles
            try:
                subtitle_data = generate_subtitles_for_story(script, audio_data, story_id)
                script["_subtitle_data"] = subtitle_data
            except Exception as e:
                logger.warning(f"  Subtitles {i+1} failed: {e}")

            results.append(script)
        except Exception as e:
            logger.error(f"  Video {i+1} assembly failed: {e}")

    return results


def upload_batch(scripts: list) -> list:
    """Phase 5: Upload videos to YouTube channels."""
    logger.info(f"Uploading {len(scripts)} videos...")
    results = []

    for i, script in enumerate(scripts):
        channel_key = script.get("channel_key", "stories")
        video_data = script.get("_video_data")
        content_type = script.get("content_type", "story")

        if not video_data:
            continue

        if not can_upload(channel_key):
            logger.warning(f"  Upload {i+1}: Channel '{channel_key}' quota exhausted, skipping")
            script["_upload_status"] = "quota_exceeded"
            results.append(script)
            continue

        try:
            title = script.get("display_title", script.get("title", "Untitled"))
            description = generate_description(script, is_shorts=False)
            tags = script.get("tags", [])

            # Determine language and content type for keyword optimization
            _lang = script.get("language", "en")
            _ctype = script.get("content_type", content_type)
            tags = optimize_tags(tags, story_script=script, language=_lang, content_type=_ctype)
            description = optimize_description(description, story_script=script, language=_lang, content_type=_ctype)

            upload_result = upload_to_channel(
                channel_key=channel_key,
                video_path=video_data["video_path"],
                title=title,
                description=description,
                tags=tags,
                content_type=content_type,
                thumbnail_path=video_data.get("thumbnail_path"),
                made_for_kids=True,
            )

            script["_upload_result"] = upload_result
            script["_upload_status"] = "uploaded"

            # Add to playlist
            try:
                if content_type == "education":
                    playlist_key = script.get("playlist_key", "")
                else:
                    collection = script.get("collection", "stories")
                    playlist_key = collection

                if playlist_key:
                    playlist_id = get_or_create_playlist(channel_key, playlist_key)
                    add_video_to_playlist(channel_key, playlist_id, upload_result["video_id"])
                    logger.info(f"  Added to playlist: {playlist_key}")
            except Exception as e:
                logger.warning(f"  Playlist operation failed: {e}")

            # Upload subtitles
            subtitle_data = script.get("_subtitle_data")
            if subtitle_data and subtitle_data.get("caption_files"):
                try:
                    from scripts.subtitle_generator import upload_captions_to_youtube
                    upload_captions_to_youtube(
                        upload_result["video_id"],
                        subtitle_data["caption_files"],
                    )
                except Exception as e:
                    logger.warning(f"  Caption upload failed: {e}")

            # Mark story as published in main DB
            try:
                from scripts.youtube_manager import mark_story_published
                mark_story_published(script["story_id"], upload_result["video_id"])
            except Exception:
                pass

            results.append(script)
            logger.info(f"  Upload {i+1}/{len(scripts)}: {upload_result['url']}")

            # Brief delay between uploads
            if i < len(scripts) - 1:
                time.sleep(30)

        except Exception as e:
            logger.error(f"  Upload {i+1} failed: {e}")
            script["_upload_status"] = "error"
            script["_upload_error"] = str(e)
            results.append(script)

    return results


def run_daily_batch():
    """Run the full daily batch pipeline for 100+ videos across all 18 channels."""
    config = load_channel_config()
    settings = config["global_settings"]
    images_per_scene = settings.get("images_per_scene", 20)

    # Build generation plan from all channels
    generation_plan = {"stories_en": 0, "education": {}, "ai_education": {}, "regional_stories": {}, "crafts_skills": {}}
    total = 0

    for key, channel in config["channels"].items():
        target = channel["daily_target"]
        ctype = channel["type"]
        lang = channel.get("language", "en")
        total += target

        if ctype == "stories" and lang == "en":
            generation_plan["stories_en"] += target
        elif ctype == "stories" and lang != "en":
            generation_plan["regional_stories"][key] = {"count": target, "language": lang}
        elif ctype == "education":
            curriculum = channel.get("curriculum", "oxford")
            edu_key = f"{curriculum}_{lang}"
            generation_plan["education"][edu_key] = {
                "count": target, "curriculum": curriculum, "language": lang
            }
        elif ctype == "ai_education":
            generation_plan["ai_education"][key] = {"count": target, "language": lang}
        elif ctype == "crafts_skills":
            skill_cat = channel.get("skill_category", "carpentry_building")
            generation_plan["crafts_skills"][key] = {"count": target, "skill_category": skill_cat}

    logger.info("=" * 70)
    logger.info("DAILY BATCH PIPELINE - Little Wisdom Tales (All Channels)")
    logger.info(f"  Total channels: {len(config['channels'])}")
    logger.info(f"  EN Stories: {generation_plan['stories_en']}")
    logger.info(f"  Regional Stories: {sum(v['count'] for v in generation_plan['regional_stories'].values())}")
    logger.info(f"  Education: {sum(v['count'] for v in generation_plan['education'].values())}")
    logger.info(f"  AI Education: {sum(v['count'] for v in generation_plan['ai_education'].values())}")
    logger.info(f"  Crafts/Skills: {sum(v['count'] for v in generation_plan['crafts_skills'].values())}")
    logger.info(f"  Total target: {total} videos")
    logger.info(f"  Images per scene: {images_per_scene}")
    logger.info("=" * 70)

    start_time = time.time()

    # Phase 1: Generate all scripts across all content types
    logger.info("\n=== PHASE 1: Script Generation (All Channels) ===")
    phase1_start = time.time()

    # 1a. English stories
    story_scripts = generate_story_batch(generation_plan["stories_en"])

    # 1b. Regional stories (translated from English)
    regional_scripts = []
    for ch_key, info in generation_plan["regional_stories"].items():
        regional_scripts.extend(
            generate_regional_story_batch(info["count"], info["language"], ch_key)
        )

    # 1c. Education (all curricula and languages)
    education_scripts = []
    for edu_key, info in generation_plan["education"].items():
        education_scripts.extend(
            generate_education_batch(info["count"], curriculum=info["curriculum"])
        )

    # 1d. AI Education (all languages)
    ai_education_scripts = []
    for ch_key, info in generation_plan["ai_education"].items():
        ai_education_scripts.extend(
            generate_ai_education_batch(info["count"], language=info["language"])
        )

    # 1e. Crafts/Skills
    crafts_scripts = []
    for ch_key, info in generation_plan["crafts_skills"].items():
        crafts_scripts.extend(
            generate_crafts_skills_batch(info["count"], skill_category=info["skill_category"])
        )

    all_scripts = story_scripts + regional_scripts + education_scripts + ai_education_scripts + crafts_scripts
    logger.info(f"Phase 1 complete: {len(all_scripts)} scripts in {time.time() - phase1_start:.0f}s")
    logger.info(f"  Stories: {len(story_scripts)}, Regional: {len(regional_scripts)}, "
                f"Education: {len(education_scripts)}, AI Ed: {len(ai_education_scripts)}, "
                f"Crafts: {len(crafts_scripts)}")

    # Phase 2: Generate all audio
    logger.info("\n=== PHASE 2: Audio Generation ===")
    phase2_start = time.time()
    all_scripts = generate_audio_batch(all_scripts)
    logger.info(f"Phase 2 complete: {len(all_scripts)} audio tracks in {time.time() - phase2_start:.0f}s")

    # Phase 3: Generate all images (GPU-intensive)
    logger.info("\n=== PHASE 3: Image Generation (GPU) ===")
    phase3_start = time.time()
    all_scripts = generate_images_batch(all_scripts, images_per_scene=images_per_scene)
    logger.info(f"Phase 3 complete: {len(all_scripts)} image sets in {time.time() - phase3_start:.0f}s")

    # Phase 4: Assemble all videos
    logger.info("\n=== PHASE 4: Video Assembly ===")
    phase4_start = time.time()
    all_scripts = assemble_videos_batch(all_scripts)
    logger.info(f"Phase 4 complete: {len(all_scripts)} videos in {time.time() - phase4_start:.0f}s")

    # Phase 5: Upload to YouTube
    logger.info("\n=== PHASE 5: YouTube Upload ===")
    phase5_start = time.time()
    all_scripts = upload_batch(all_scripts)
    logger.info(f"Phase 5 complete in {time.time() - phase5_start:.0f}s")

    # Summary
    total_time = time.time() - start_time
    uploaded = sum(1 for s in all_scripts if s.get("_upload_status") == "uploaded")
    quota_exceeded = sum(1 for s in all_scripts if s.get("_upload_status") == "quota_exceeded")
    errors = sum(1 for s in all_scripts if s.get("_upload_status") == "error")

    logger.info("\n" + "=" * 70)
    logger.info("DAILY BATCH COMPLETE - ALL CHANNELS")
    logger.info(f"  Total time: {total_time / 60:.1f} minutes")
    logger.info(f"  Scripts generated: {len(all_scripts)}")
    logger.info(f"  Uploaded: {uploaded}")
    logger.info(f"  Quota exceeded: {quota_exceeded}")
    logger.info(f"  Errors: {errors}")
    logger.info(f"  Generated but not uploaded: {len(all_scripts) - uploaded - quota_exceeded - errors}")

    # Per-channel stats
    stats = get_all_channel_stats()
    for key, stat in stats.items():
        logger.info(f"  {stat['name']}: {stat['uploads_today']}/{stat['daily_target']} uploads, "
                     f"{stat['max_more_uploads']} more possible")

    # Per-content-type breakdown
    type_counts = {}
    for s in all_scripts:
        ct = s.get("content_type", "story")
        lang = s.get("language", "en")
        k = f"{ct}_{lang}"
        if k not in type_counts:
            type_counts[k] = {"generated": 0, "uploaded": 0, "errors": 0}
        type_counts[k]["generated"] += 1
        if s.get("_upload_status") == "uploaded":
            type_counts[k]["uploaded"] += 1
        elif s.get("_upload_status") == "error":
            type_counts[k]["errors"] += 1

    logger.info("\n  Content breakdown:")
    for k, v in sorted(type_counts.items()):
        logger.info(f"    {k}: {v['uploaded']}/{v['generated']} uploaded, {v['errors']} errors")

    logger.info("=" * 70)

    # Save results
    results_path = LOG_DIR / f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    summary = {
        "timestamp": datetime.now().isoformat(),
        "total_time_seconds": total_time,
        "scripts_generated": len(all_scripts),
        "uploaded": uploaded,
        "quota_exceeded": quota_exceeded,
        "errors": errors,
        "channel_stats": stats,
        "content_breakdown": type_counts,
    }
    with open(results_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    return summary


def run_incremental():
    """Run pipeline incrementally - generate and upload one at a time.
    Better for quota management and error recovery.
    """
    config = load_channel_config()
    total_target = config["global_settings"]["total_daily_target"]
    images_per_scene = config["global_settings"].get("images_per_scene", 20)

    logger.info(f"Incremental pipeline - target: {total_target} videos")

    uploaded = 0
    errors = 0

    # Alternate between content types (include language for keyword optimization)
    content_cycle = []
    for key, channel in config["channels"].items():
        target = channel["daily_target"]
        for _ in range(target):
            content_cycle.append((key, channel["type"], channel.get("curriculum", ""), channel.get("language", "en")))

    # Shuffle for variety
    import random
    random.shuffle(content_cycle)

    for i, (channel_key, content_type, curriculum, channel_lang) in enumerate(content_cycle):
        if not can_upload(channel_key):
            logger.info(f"  [{i+1}] Channel {channel_key} quota exhausted, skipping")
            continue

        try:
            logger.info(f"\n--- Video {i+1}/{len(content_cycle)} ({content_type}/{channel_key}/{channel_lang}) ---")

            # Generate script based on content type
            if content_type == "stories" and channel_lang == "en":
                params = pick_smart_story_params()
                script = generate_story_script(
                    collection=params.get("collection"),
                    moral=params.get("moral"),
                    for_shorts=False,
                )
                script = inject_engagement_hooks(script, is_shorts=False)
                title = generate_engaging_title(
                    story_title=script.get("title", ""),
                    moral=script.get("moral", ""),
                    origin=script.get("origin", ""),
                )
                title = optimize_title(
                    title, moral=script.get("moral", ""), collection=script.get("collection", ""),
                    language="en", content_type="story",
                )
                script["display_title"] = title
                script["language"] = "en"
                playlist_key = script.get("collection", "stories")

            elif content_type == "stories" and channel_lang != "en":
                # Regional story: generate English then translate
                channel_cfg = config["channels"].get(channel_key, {})
                collections = channel_cfg.get("story_collections", [])
                collection = collections[i % len(collections)] if collections else None
                params = pick_smart_story_params()
                if collection:
                    params["collection"] = collection

                script = generate_story_script(
                    collection=params.get("collection"),
                    moral=params.get("moral"),
                    for_shorts=False,
                )
                script = inject_engagement_hooks(script, is_shorts=False)
                script = create_regional_version(script, channel_lang, content_type="story")

                title = generate_engaging_title(
                    story_title=script.get("title", ""),
                    moral=script.get("moral", ""),
                    origin=script.get("origin", ""),
                )
                title = optimize_title(
                    title, moral=script.get("moral", ""), collection=script.get("collection", ""),
                    language=channel_lang, content_type="story",
                )
                script["display_title"] = title
                script["language"] = channel_lang
                playlist_key = script.get("collection", "stories")

            elif content_type == "ai_education":
                # AI education lesson
                try:
                    from scripts.ai_education_generator import generate_ai_lesson_script, pick_next_ai_lesson
                    lesson = pick_next_ai_lesson()
                    if not lesson:
                        logger.warning("  No more AI lessons available")
                        continue
                    script = generate_ai_lesson_script(**lesson)
                    if channel_lang != "en":
                        script = create_regional_version(script, channel_lang, content_type="ai_education")
                    script["language"] = channel_lang
                    playlist_key = lesson["category"]
                except ImportError:
                    logger.error("  ai_education_generator.py not found, skipping")
                    continue

            elif content_type == "crafts_skills":
                # Crafts/skills lesson
                try:
                    from scripts.crafts_skills_generator import generate_crafts_lesson_script, pick_next_crafts_lesson
                    channel_cfg = config["channels"].get(channel_key, {})
                    skill_cat = channel_cfg.get("skill_category", "carpentry_building")
                    lesson = pick_next_crafts_lesson(category=skill_cat)
                    if not lesson:
                        logger.warning(f"  No more crafts lessons for {skill_cat}")
                        continue
                    script = generate_crafts_lesson_script(**lesson)
                    script["language"] = "en"
                    playlist_key = lesson.get("topic", skill_cat)
                except ImportError:
                    logger.error("  crafts_skills_generator.py not found, skipping")
                    continue

            else:
                # Education content
                from scripts.education_generator import generate_lesson_script, pick_next_lesson
                lesson = pick_next_lesson(curriculum=curriculum)
                if not lesson:
                    logger.warning(f"  No more lessons for {curriculum}")
                    continue
                script = generate_lesson_script(**lesson)
                if channel_lang != "en":
                    script = create_regional_version(script, channel_lang, content_type="education")
                script["language"] = channel_lang
                playlist_key = f"class_{lesson['class_level']}_{lesson['subject']}"

            story_id = script.get("story_id", i + 2000)

            # Audio
            audio_data = generate_audio_sync(script, story_id)

            # Images
            image_data = generate_story_images(
                script, story_id,
                for_shorts=False,
                images_per_scene=images_per_scene,
            )

            # Video
            video_data = assemble_story_video(
                story_script=script,
                audio_data=audio_data,
                image_data=image_data,
                story_id=story_id,
                for_shorts=False,
            )

            # Subtitles
            try:
                subtitle_data = generate_subtitles_for_story(script, audio_data, story_id)
            except Exception:
                subtitle_data = None

            # Upload
            title = script.get("display_title", script.get("title", ""))
            description = generate_description(script, is_shorts=False)
            _lang = script.get("language", "en")
            _ctype = content_type if content_type != "stories" else "story"
            tags = optimize_tags(script.get("tags", []), story_script=script, language=_lang, content_type=_ctype)
            description = optimize_description(description, story_script=script, language=_lang, content_type=_ctype)

            upload_result = upload_to_channel(
                channel_key=channel_key,
                video_path=video_data["video_path"],
                title=title,
                description=description,
                tags=tags,
                content_type=content_type,
                thumbnail_path=video_data.get("thumbnail_path"),
                made_for_kids=True,
            )

            # Playlist
            try:
                pid = get_or_create_playlist(channel_key, playlist_key)
                add_video_to_playlist(channel_key, pid, upload_result["video_id"])
            except Exception:
                pass

            # Subtitles upload
            if subtitle_data and subtitle_data.get("caption_files"):
                try:
                    from scripts.subtitle_generator import upload_captions_to_youtube
                    upload_captions_to_youtube(upload_result["video_id"], subtitle_data["caption_files"])
                except Exception:
                    pass

            uploaded += 1
            logger.info(f"  Published: {upload_result['url']}")

            # Free GPU periodically
            if uploaded % 10 == 0:
                try:
                    unload_model()
                except Exception:
                    pass

            time.sleep(30)

        except Exception as e:
            errors += 1
            logger.error(f"  Pipeline error: {e}")
            logger.error(traceback.format_exc())

    # Cleanup
    try:
        unload_model()
    except Exception:
        pass

    logger.info(f"\nIncremental run complete: {uploaded} uploaded, {errors} errors")
    return {"uploaded": uploaded, "errors": errors}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Batch Pipeline - 100+ videos/day")
    parser.add_argument("mode", choices=["batch", "incremental", "stats"],
                        help="batch=all phases, incremental=one-at-a-time, stats=show status")
    parser.add_argument("--stories", type=int, help="Override story count")
    parser.add_argument("--education", type=int, help="Override education count per curriculum")

    args = parser.parse_args()

    if args.mode == "batch":
        run_daily_batch()
    elif args.mode == "incremental":
        run_incremental()
    elif args.mode == "stats":
        stats = get_all_channel_stats()
        print(json.dumps(stats, indent=2))
