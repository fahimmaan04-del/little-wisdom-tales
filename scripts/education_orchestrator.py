"""
Education Orchestrator - End-to-end pipeline for automated educational video creation.

This is the main entry point for the education channel that coordinates all components:
1. Pick next lesson from syllabus (or accept manual params)
2. Generate lesson script via LLM (education_generator)
3. Inject education-specific engagement hooks
4. Generate voiceover audio (Edge TTS)
5. Generate AI images (SDXL Turbo with content_type='education')
6. Assemble video with effects (FFmpeg)
7. Generate multi-language subtitles
8. Upload to the correct YouTube channel (channel_manager)
9. Add to curriculum-based playlist

Can be triggered by the scheduler, n8n workflows, or run standalone via CLI.
"""

import json
import logging
import os
import random
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

load_dotenv()

from scripts.education_generator import (
    generate_lesson_script,
    get_education_playlist_name,
    mark_lesson_published,
    pick_next_lesson,
)
from scripts.tts_generator import generate_audio_sync
from scripts.ai_image_generator import generate_story_images, unload_model as unload_ai_model
from scripts.image_fetcher import fetch_scene_images
from scripts.video_assembler import assemble_story_video
from scripts.subtitle_generator import generate_subtitles_for_story
from scripts.channel_manager import (
    add_video_to_playlist,
    get_or_create_playlist,
    pick_best_channel,
    upload_to_channel,
)
from scripts.keyword_optimizer import optimize_title, optimize_tags, optimize_description
from scripts.youtube_manager import mark_story_published

OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "./output"))
DATA_DIR = Path(os.getenv("DATA_DIR", "./data"))
LOG_DIR = DATA_DIR / "logs"

# Setup logging
LOG_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "education_orchestrator.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


# --- Education-specific engagement hooks ---

EDUCATION_INTRO_HOOKS = [
    (
        "Welcome back to Little Wisdom Academy! "
        "Today Professor Wisdom is going to teach us about {topic}! "
        "Are you ready? Let's go!"
    ),
    (
        "Hoo-hoo! Hello, little learners! "
        "Welcome to Little Wisdom Academy! "
        "Today Professor Wisdom has an amazing lesson about {topic}! Let's start!"
    ),
    (
        "Hey friends! It's time for another fun lesson at Little Wisdom Academy! "
        "Professor Wisdom is SO excited to teach you about {topic} today!"
    ),
    (
        "Hello everyone! Welcome to Little Wisdom Academy! "
        "Get your thinking caps on because today we're learning about {topic}! "
        "Professor Wisdom, take it away!"
    ),
    (
        "Gather around, little learners! "
        "Professor Wisdom has a super fun lesson for you today about {topic}! "
        "Let's learn something amazing together!"
    ),
]

EDUCATION_MID_HOOKS = [
    "Are you following along? Great job! Let's keep learning!",
    "Hoo-hoo! You're doing amazing! Let's learn even more!",
    "Wonderful! You're such a great student! Ready for the next part?",
    "Isn't this fun? There's more to discover! Keep watching!",
    "You're learning so fast! Professor Wisdom is proud of you!",
]

EDUCATION_OUTRO_HOOKS = [
    (
        "Amazing! You learned so much today! "
        "Don't forget to subscribe for more fun lessons! "
        "See you next time at Little Wisdom Academy!"
    ),
    (
        "Hoo-hoo! What a great lesson! "
        "You're becoming such a smart learner! "
        "Come back tomorrow for more fun at Little Wisdom Academy!"
    ),
    (
        "That was fantastic! Professor Wisdom is so proud of you! "
        "There are new lessons every day at Little Wisdom Academy! "
        "See you in the next video!"
    ),
    (
        "Great job today! Remember what you learned! "
        "Subscribe to Little Wisdom Academy for more fun lessons! "
        "Bye for now, little learners!"
    ),
    (
        "You did it! Another amazing lesson complete! "
        "Come back soon for more adventures in learning! "
        "Professor Wisdom will be waiting for you!"
    ),
]


def inject_education_hooks(lesson_script: dict) -> dict:
    """Inject education-specific engagement hooks into the lesson script.

    Adds a channel intro scene, a midpoint encouragement, and a subscribe
    outro scene. Uses the lesson topic to personalize the intro.

    Args:
        lesson_script: The lesson script dict with scenes.

    Returns:
        The modified lesson script with hook scenes injected.
    """
    scenes = lesson_script.get("scenes", [])
    if not scenes:
        return lesson_script

    topic = lesson_script.get("lesson_title", lesson_script.get("topic", "something amazing"))

    # Intro scene
    intro_text = random.choice(EDUCATION_INTRO_HOOKS).format(topic=topic)
    intro_scene = {
        "scene_number": 0,
        "narration": intro_text,
        "visual_description": (
            "Bright colorful classroom intro screen for Little Wisdom Academy, "
            "Professor Wisdom the cartoon owl with big glasses and graduation cap "
            "waving excitedly, colorful balloons and sparkles, a banner reading "
            "the lesson title, cheerful and inviting"
        ),
        "image_search_terms": (
            "cartoon owl professor glasses graduation cap classroom welcome "
            "kids education colorful balloons sparkles banner"
        ),
        "character_speaking": "professor_wisdom",
        "duration_seconds": 8,
        "scene_type": "intro",
        "is_hook": True,
    }

    # Midpoint encouragement (only for lessons with 6+ scenes)
    midpoint_scene = None
    if len(scenes) >= 6:
        mid_idx = len(scenes) // 2
        midpoint_scene = {
            "scene_number": mid_idx + 0.5,
            "narration": random.choice(EDUCATION_MID_HOOKS),
            "visual_description": scenes[mid_idx].get("visual_description", ""),
            "image_search_terms": scenes[mid_idx].get("image_search_terms", ""),
            "character_speaking": "professor_wisdom",
            "duration_seconds": 5,
            "scene_type": "encouragement",
            "is_hook": True,
        }

    # Outro scene
    outro_scene = {
        "scene_number": 999,
        "narration": random.choice(EDUCATION_OUTRO_HOOKS),
        "visual_description": (
            "Colorful ending screen for Little Wisdom Academy, "
            "Professor Wisdom the cartoon owl waving goodbye with sparkles and stars, "
            "a subscribe button animation, bright cheerful background"
        ),
        "image_search_terms": (
            "cartoon owl professor goodbye subscribe kids education colorful "
            "stars sparkles ending screen cheerful"
        ),
        "character_speaking": "professor_wisdom",
        "duration_seconds": 8,
        "scene_type": "outro",
        "is_hook": True,
    }

    # Build new scene list: intro + original scenes (with midpoint) + outro
    new_scenes = [intro_scene]
    for i, scene in enumerate(scenes):
        new_scenes.append(scene)
        if midpoint_scene and i == len(scenes) // 2 - 1:
            new_scenes.append(midpoint_scene)
    new_scenes.append(outro_scene)

    # Renumber scenes sequentially
    for i, scene in enumerate(new_scenes):
        scene["scene_number"] = i + 1

    lesson_script["scenes"] = new_scenes
    return lesson_script


# --- Title, description, and tag generation ---


EDUCATION_TITLE_TEMPLATES = [
    "{lesson_title} | Class {class_level} {subject} | Fun Animated Lesson for Kids",
    "Learn {lesson_title} | Class {class_level} {subject} | Kids Education",
    "{lesson_title} | {subject} for Class {class_level} | Little Wisdom Academy",
    "Class {class_level} {subject}: {lesson_title} | Educational Video for Kids",
    "{lesson_title} | Fun {subject} Lesson | Class {class_level} Kids Learning",
    "Professor Wisdom Teaches {lesson_title} | Class {class_level} {subject}",
]


def generate_education_title(
    class_level: int,
    subject: str,
    lesson_title: str,
) -> str:
    """Generate a YouTube-friendly title for an educational video.

    Picks a random title template and fills in class level, subject, and lesson
    title. The result is kept under 70 characters for optimal click-through rate.

    Args:
        class_level: Student class level (1-5).
        subject: Subject area identifier (e.g. "mathematics").
        lesson_title: Human-readable lesson title.

    Returns:
        A formatted title string, max 70 characters.
    """
    subject_display = subject.replace("_", " ").title()
    template = random.choice(EDUCATION_TITLE_TEMPLATES)
    title = template.format(
        lesson_title=lesson_title,
        class_level=class_level,
        subject=subject_display,
    )
    # YouTube title max is 100 chars, keep under 70 for better CTR
    return title[:70]


def generate_education_description(
    script: dict,
    class_level: int,
    subject: str,
    curriculum: str = "oxford",
) -> str:
    """Generate an SEO-optimized YouTube description for an educational video.

    Builds a rich description with lesson summary, curriculum info, channel
    branding, and hashtags targeted at parents and kids searching for
    educational content.

    Args:
        script: The full lesson script dict.
        class_level: Student class level (1-5).
        subject: Subject area identifier.
        curriculum: Curriculum standard ('oxford' or 'cambridge').

    Returns:
        A formatted description string under 5000 characters.
    """
    title = script.get("title", "")
    topic = script.get("topic", "").replace("_", " ").title()
    lesson_title = script.get("lesson_title", title)
    subject_display = subject.replace("_", " ").title()
    curriculum_display = curriculum.title()
    base_desc = script.get("description", "")
    moral = script.get("moral", "Learning is fun and everyone can do it!")

    # Pick a random call to action for variety
    ctas = [
        "Subscribe to Little Wisdom Academy for new lessons every day!",
        "Hit subscribe so you never miss a fun lesson from Professor Wisdom!",
        "New educational videos every day! Subscribe for more!",
    ]

    description = f"""{base_desc}

Join Professor Wisdom the friendly owl at Little Wisdom Academy as he teaches {lesson_title}!

Class Level: Class {class_level} (Ages {_get_age_range(class_level)})
Subject: {subject_display}
Topic: {topic}
Curriculum: {curriculum_display} International Primary Programme

Lesson Message: {moral}

{random.choice(ctas)}

---
Little Wisdom Academy by Little Wisdom Tales
Fun animated educational videos for kids aged 4-10.
Following {curriculum_display} and Cambridge International curriculum standards.
Learn Mathematics, English, Science, and General Knowledge with Professor Wisdom!

New lessons EVERY DAY!

#KidsEducation #Class{class_level}{subject_display.replace(' ', '')} #{curriculum_display}Curriculum
#LearnWithProfessorWisdom #LittleWisdomAcademy #LittleWisdomTales
#KidsLearning #FunEducation #AnimatedLessons #{subject_display.replace(' ', '')}ForKids
#PrimarySchool #ElementaryEducation #Class{class_level}
"""
    return description[:5000]


def generate_education_tags(
    class_level: int,
    subject: str,
    topic: str,
    curriculum: str = "oxford",
) -> list[str]:
    """Generate education-specific YouTube tags for discoverability.

    Creates a comprehensive set of tags covering the lesson topic, subject,
    curriculum, class level, and general educational search terms. Tags are
    deduplicated and capped at YouTube's 30-tag limit.

    Args:
        class_level: Student class level (1-5).
        subject: Subject area identifier.
        topic: Specific topic identifier.
        curriculum: Curriculum standard.

    Returns:
        A list of tag strings, at most 30 items.
    """
    subject_display = subject.replace("_", " ")
    topic_display = topic.replace("_", " ")
    curriculum_display = curriculum.title()
    age_range = _get_age_range(class_level)

    tags = [
        # Core lesson tags
        f"class {class_level} {subject_display}",
        f"{topic_display} for kids",
        f"learn {topic_display}",
        f"{subject_display} class {class_level}",
        f"{curriculum_display} curriculum class {class_level}",
        # Subject tags
        f"kids {subject_display}",
        f"{subject_display} for children",
        f"{subject_display} lesson",
        f"fun {subject_display}",
        # Education general
        "kids education",
        "educational video for kids",
        "kids learning",
        "primary school",
        "elementary education",
        f"class {class_level}",
        f"age {age_range} education",
        # Channel tags
        "Professor Wisdom",
        "Little Wisdom Academy",
        "Little Wisdom Tales",
        "learn with owl",
        "cartoon education",
        "animated lesson",
        # Curriculum tags
        f"{curriculum_display} curriculum",
        f"{curriculum_display} primary",
        # Search-friendly
        f"{topic_display}",
        f"{subject_display}",
        "kids",
        "children",
        "learning",
        "fun lesson",
    ]

    # Deduplicate while preserving order
    seen = set()
    unique_tags = []
    for tag in tags:
        tag_lower = tag.lower().strip()
        if tag_lower not in seen and tag_lower:
            seen.add(tag_lower)
            unique_tags.append(tag)

    # YouTube allows max 30 tags
    return unique_tags[:30]


def _get_age_range(class_level: int) -> str:
    """Get the target age range string for a class level."""
    age_map = {1: "5-6", 2: "6-7", 3: "7-8", 4: "8-9", 5: "9-10"}
    return age_map.get(class_level, "5-10")


# --- Main pipeline ---


def create_single_lesson(
    class_level: int = None,
    subject: str = None,
    curriculum: str = "oxford",
    upload: bool = True,
    channel_key: str = None,
) -> dict:
    """Complete pipeline: pick lesson -> generate script -> create video -> upload.

    Orchestrates the full educational video creation pipeline from lesson
    selection through YouTube upload. Follows the same pattern as the story
    orchestrator but adapted for educational content with Professor Wisdom.

    Args:
        class_level: Target class level (1-5). If None, auto-picked from syllabus.
        subject: Subject to generate for. If None, auto-picked from syllabus.
        curriculum: Curriculum standard, 'oxford' or 'cambridge'.
        upload: Whether to upload the finished video to YouTube.
        channel_key: Specific channel to upload to. If None, auto-selected
            based on curriculum via channel_manager.pick_best_channel.

    Returns:
        A dict with pipeline results including story_id, lesson_id, title,
        video_path, upload info, and status ('completed' or 'error').
    """
    result = {
        "status": "started",
        "timestamp": datetime.now().isoformat(),
        "content_type": "education",
        "curriculum": curriculum,
    }

    try:
        # Step 1: Pick next lesson from syllabus (or use provided params)
        logger.info("=" * 60)
        logger.info("EDUCATION PIPELINE")
        logger.info("=" * 60)
        logger.info("STEP 1: Selecting lesson from syllabus...")

        if class_level is not None and subject is not None:
            # Manual specification -- use pick_next_lesson to find a specific
            # ungenerated topic for this class+subject, or fall back to a
            # generic topic name
            lesson_params = pick_next_lesson(curriculum=curriculum)
            if lesson_params is None:
                logger.warning("All syllabus lessons generated. Using provided params directly.")
                lesson_params = {
                    "class_level": class_level,
                    "subject": subject,
                    "topic": subject,
                    "lesson_title": f"{subject.replace('_', ' ').title()} Lesson",
                    "curriculum": curriculum,
                }
            else:
                # Override with user-specified class and subject if the auto-pick
                # returned something different
                lesson_params["class_level"] = class_level
                lesson_params["subject"] = subject
        else:
            lesson_params = pick_next_lesson(curriculum=curriculum)
            if lesson_params is None:
                logger.info("All lessons in the syllabus have been generated!")
                result["status"] = "completed"
                result["message"] = "All syllabus lessons already generated"
                return result

        class_level = lesson_params["class_level"]
        subject = lesson_params["subject"]
        topic = lesson_params["topic"]
        lesson_title = lesson_params["lesson_title"]

        logger.info(f"  Lesson: '{lesson_title}'")
        logger.info(f"  Class: {class_level} | Subject: {subject} | Topic: {topic}")
        logger.info(f"  Curriculum: {curriculum}")

        result["class_level"] = class_level
        result["subject"] = subject
        result["topic"] = topic

        # Step 2: Generate lesson script via LLM
        logger.info("STEP 2: Generating lesson script via LLM...")
        lesson_script = generate_lesson_script(
            class_level=class_level,
            subject=subject,
            topic=topic,
            lesson_title=lesson_title,
            curriculum=curriculum,
        )
        story_id = lesson_script["story_id"]
        lesson_id = lesson_script.get("lesson_id")

        logger.info(f"  Story ID: {story_id} | Lesson ID: {lesson_id}")
        logger.info(f"  Scenes: {len(lesson_script.get('scenes', []))}")

        result["story_id"] = story_id
        result["lesson_id"] = lesson_id

        # Step 3: Inject education-specific engagement hooks
        logger.info("STEP 3: Injecting education engagement hooks...")
        lesson_script = inject_education_hooks(lesson_script)
        logger.info(f"  Scenes (with hooks): {len(lesson_script.get('scenes', []))}")

        # Generate optimized title with trending keyword injection
        title = generate_education_title(class_level, subject, lesson_title)
        title = optimize_title(
            title, language="en", content_type="education",
            collection=f"class_{class_level}",
        )
        lesson_script["display_title"] = title
        logger.info(f"  Title: '{title}'")

        result["title"] = title

        # Step 4: Generate voiceover audio
        logger.info("STEP 4: Generating voiceover audio...")
        audio_data = generate_audio_sync(lesson_script, story_id)
        logger.info(f"  Total audio duration: {audio_data['total_duration']:.1f}s")

        result["audio_duration"] = audio_data["total_duration"]

        # Step 5: Generate AI images (with Pixabay fallback)
        logger.info("STEP 5: Generating AI scene images...")
        try:
            image_data = generate_story_images(
                lesson_script, story_id,
                for_shorts=False,
                images_per_scene=30,
            )
            total_imgs = sum(len(d.get("all_images", [1])) for d in image_data)
            logger.info(f"  AI generated {total_imgs} images for {len(image_data)} scenes")
        except Exception as e:
            logger.warning(f"  AI image generation failed: {e}, falling back to Pixabay")
            image_data = fetch_scene_images(lesson_script, story_id, for_shorts=False)
            logger.info(f"  Fetched {len(image_data)} stock images (fallback)")

        result["images_generated"] = len(image_data)

        # Step 6: Assemble video
        logger.info("STEP 6: Assembling video...")
        video_data = assemble_story_video(
            story_script=lesson_script,
            audio_data=audio_data,
            image_data=image_data,
            story_id=story_id,
            for_shorts=False,
        )
        logger.info(f"  Video created: {video_data['video_path']}")
        logger.info(f"  Duration: {video_data['duration_seconds']:.1f}s")

        result["video_path"] = video_data["video_path"]
        result["video_duration"] = video_data["duration_seconds"]

        # Step 6.5: Generate subtitles (multi-language)
        logger.info("STEP 6.5: Generating multi-language subtitles...")
        try:
            subtitle_data = generate_subtitles_for_story(
                lesson_script, audio_data, story_id,
            )
            logger.info(f"  Subtitles generated: {', '.join(subtitle_data.get('languages', ['en']))}")
            result["subtitles"] = subtitle_data.get("languages", [])
        except Exception as e:
            subtitle_data = None
            logger.warning(f"  Subtitle generation failed: {e}")

        # Step 7: Upload to YouTube
        if upload:
            logger.info("STEP 7: Uploading to YouTube...")

            # Generate description and tags with trending keyword optimization
            description = generate_education_description(
                lesson_script, class_level, subject, curriculum,
            )
            tags = generate_education_tags(class_level, subject, topic, curriculum)

            # Merge script-level tags if they exist
            script_tags = lesson_script.get("tags", [])
            if script_tags:
                existing_lower = {t.lower() for t in tags}
                for tag in script_tags:
                    if tag.lower() not in existing_lower and len(tags) < 30:
                        tags.append(tag)
                        existing_lower.add(tag.lower())

            # Enhance with trending keyword intelligence
            tags = optimize_tags(
                tags, story_script=lesson_script,
                language="en", content_type="education",
            )
            description = optimize_description(
                description, story_script=lesson_script,
                language="en", content_type="education",
            )

            # Determine which channel to upload to
            if channel_key is None:
                try:
                    channel_key = pick_best_channel(
                        content_type="education",
                        curriculum=curriculum,
                    )
                except ValueError:
                    logger.warning(
                        "No education channel found for curriculum "
                        f"'{curriculum}', falling back to default upload"
                    )
                    channel_key = None

            if channel_key:
                # Upload via channel_manager (multi-channel system)
                upload_result = upload_to_channel(
                    channel_key=channel_key,
                    video_path=video_data["video_path"],
                    title=title,
                    description=description,
                    tags=tags,
                    content_type="education",
                    thumbnail_path=video_data.get("thumbnail_path"),
                    made_for_kids=True,
                )
            else:
                # Fallback to the default youtube_manager upload
                from scripts.youtube_manager import upload_video
                upload_result = upload_video(
                    video_path=video_data["video_path"],
                    title=title,
                    description=description,
                    tags=tags,
                    is_shorts=False,
                    thumbnail_path=video_data.get("thumbnail_path"),
                    made_for_kids=True,
                )

            video_id = upload_result["video_id"]

            # Mark as published in both stories and education_lessons tables
            mark_story_published(story_id, video_id)
            if lesson_id:
                mark_lesson_published(lesson_id, video_id)

            logger.info(f"  Published: {upload_result['url']}")

            # Upload subtitles to YouTube
            if subtitle_data and subtitle_data.get("caption_files"):
                try:
                    from scripts.subtitle_generator import upload_captions_to_youtube
                    upload_captions_to_youtube(
                        video_id,
                        subtitle_data["caption_files"],
                    )
                    logger.info("  Subtitles uploaded to YouTube")
                except Exception as e:
                    logger.warning(f"  Caption upload failed: {e}")

            # Add to curriculum-based playlist
            try:
                playlist_name = get_education_playlist_name(
                    class_level, subject, curriculum,
                )
                # Use a sanitized key for the playlist cache
                playlist_key = (
                    f"class_{class_level}_{subject}_{curriculum}"
                ).lower().replace(" ", "_")

                if channel_key:
                    playlist_id = get_or_create_playlist(channel_key, playlist_key)
                    add_video_to_playlist(channel_key, playlist_id, video_id)
                else:
                    from scripts.youtube_manager import create_or_get_playlist, add_to_playlist
                    playlist_id = create_or_get_playlist(playlist_name)
                    add_to_playlist(playlist_id, video_id)

                logger.info(f"  Added to playlist: {playlist_name}")
            except Exception as e:
                logger.warning(f"  Playlist operation failed: {e}")

            result["upload"] = upload_result
        else:
            logger.info("STEP 7: Upload skipped (upload=False)")

        # Cleanup temporary files (keep final video + thumbnail)
        cleanup_temp_files(story_id)

        # Free GPU memory after AI image generation
        try:
            unload_ai_model()
        except Exception:
            pass

        result["status"] = "completed"
        logger.info(f"Education pipeline completed for '{title}'")
        logger.info("=" * 60)

    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)
        result["traceback"] = traceback.format_exc()
        logger.error(f"Education pipeline failed: {e}")
        logger.error(traceback.format_exc())

    # Save result to log
    result_path = DATA_DIR / "logs" / f"education_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2, default=str)

    return result


def cleanup_temp_files(story_id: int):
    """Remove intermediate files, keep only final video + thumbnail.

    Follows the same cleanup pattern as the story orchestrator to avoid
    accumulating gigabytes of intermediate pipeline artifacts.

    Args:
        story_id: The story ID whose temp files should be cleaned.
    """
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


def run_education_batch(
    count: int = 3,
    curriculum: str = "oxford",
) -> list[dict]:
    """Run a batch of educational lesson creation.

    Creates multiple lessons sequentially, rotating through subjects and
    class levels as determined by pick_next_lesson. Respects API rate limits
    with configurable pauses between uploads.

    Args:
        count: Number of lessons to create in this batch.
        curriculum: Curriculum standard to use for all lessons.

    Returns:
        List of result dicts from each create_single_lesson call.
    """
    logger.info(f"Starting education batch: {count} lessons ({curriculum} curriculum)")

    results = []
    for i in range(count):
        logger.info(f"\n--- Lesson {i + 1}/{count} ---")

        result = create_single_lesson(
            curriculum=curriculum,
            upload=True,
        )
        results.append(result)

        if result["status"] == "error":
            logger.error(f"Lesson {i + 1} failed, continuing to next...")
            time.sleep(10)
            continue

        # Check if all syllabus lessons are done
        if result.get("message") == "All syllabus lessons already generated":
            logger.info("No more lessons to generate. Stopping batch.")
            break

        # Wait between lessons to respect API rate limits
        if i < count - 1:
            wait_time = 300  # 5 minutes between lessons
            logger.info(f"Waiting {wait_time}s before next lesson...")
            time.sleep(wait_time)

    # Summary
    successful = sum(1 for r in results if r["status"] == "completed")
    failed = sum(1 for r in results if r["status"] == "error")
    logger.info(f"\nEducation batch complete: {successful} succeeded, {failed} failed")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Education Channel Orchestrator - Little Wisdom Academy"
    )
    parser.add_argument(
        "command",
        choices=["single", "batch", "test"],
        help="Command to run",
    )
    parser.add_argument(
        "--class-level",
        type=int,
        choices=[1, 2, 3, 4, 5],
        help="Target class level (1-5)",
    )
    parser.add_argument(
        "--subject",
        choices=["mathematics", "english", "science", "general_knowledge"],
        help="Subject to generate lesson for",
    )
    parser.add_argument(
        "--curriculum",
        choices=["oxford", "cambridge"],
        default="oxford",
        help="Curriculum standard (default: oxford)",
    )
    parser.add_argument(
        "--channel",
        help="Specific channel key to upload to (overrides auto-selection)",
    )
    parser.add_argument(
        "--no-upload",
        action="store_true",
        help="Skip YouTube upload",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=3,
        help="Number of lessons for batch mode (default: 3)",
    )

    args = parser.parse_args()

    if args.command == "single":
        result = create_single_lesson(
            class_level=args.class_level,
            subject=args.subject,
            curriculum=args.curriculum,
            upload=not args.no_upload,
            channel_key=args.channel,
        )
        print(json.dumps(result, indent=2, default=str))

    elif args.command == "batch":
        results = run_education_batch(
            count=args.count,
            curriculum=args.curriculum,
        )
        successful = sum(1 for r in results if r["status"] == "completed")
        failed = sum(1 for r in results if r["status"] == "error")
        print(f"\n=== Batch complete: {successful}/{len(results)} succeeded, {failed} failed ===")
        print(json.dumps(results, indent=2, default=str))

    elif args.command == "test":
        # Test run without uploading
        result = create_single_lesson(
            curriculum=args.curriculum,
            upload=False,
        )
        print(json.dumps(result, indent=2, default=str))
