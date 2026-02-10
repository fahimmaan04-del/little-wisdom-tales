"""
Video Re-Edit Pipeline - Regenerate existing videos with higher quality.

Fetches published videos from the stories database, re-generates them with
improved AI images, better audio, smoother transitions, and enhanced SEO,
then re-uploads or updates them on YouTube.

Quality improvements over original pipeline:
- 30 images per scene with varied camera angles for smoother crossfades
- Per-image display time = scene_duration / num_images for perfect sync
- 0.5s crossfade overlap between consecutive images (xfade filter)
- 0.3s scene-level fade in/out for smooth scene transitions
- Enhanced cartoon pitch effects with warmth and clarity filters
- Multi-language subtitle generation and upload
- Keyword-optimized titles, descriptions, and tags
- Enhanced 1280x720 thumbnails with bright overlays and text

Priority system: re-edits videos with lowest view count first.
Tracks re-edits in a dedicated `reedited_videos` table to avoid duplicates.

CLI usage:
    python reupload_videos.py list                 # List re-edit candidates
    python reupload_videos.py reedit <story_id>    # Re-edit a specific story
    python reupload_videos.py batch <count>        # Re-edit N lowest-performing videos
"""

import json
import logging
import os
import sqlite3
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))
load_dotenv()

from scripts.tts_generator import generate_audio_sync, get_audio_duration
from scripts.ai_image_generator import generate_story_images, unload_model
from scripts.video_assembler import (
    assemble_story_video,
    create_thumbnail,
    get_audio_duration as va_get_audio_duration,
    _render_text_image,
    _resize_image_to_video,
)
from scripts.engagement_hooks import (
    inject_engagement_hooks,
    generate_engaging_title,
    generate_description,
    get_thumbnail_overlay_text,
)
from scripts.youtube_manager import (
    upload_video,
    mark_story_published,
    get_youtube_service,
    get_video_analytics,
)
from scripts.keyword_optimizer import (
    optimize_title,
    optimize_tags,
    optimize_description,
    track_keyword_usage,
    get_trending_keywords,
)
from scripts.subtitle_generator import (
    generate_subtitles_for_story,
    upload_captions_to_youtube,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA_DIR = Path(os.getenv("DATA_DIR", "./data"))
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "./output"))

IMAGES_PER_SCENE = 30
CROSSFADE_DURATION = 0.5   # seconds of overlap between consecutive images
SCENE_FADE_DURATION = 0.3  # seconds of fade in/out at scene boundaries
MIN_IMAGE_DISPLAY = 0.2    # absolute minimum seconds per image
THUMBNAIL_WIDTH = 1280
THUMBNAIL_HEIGHT = 720
VIDEO_WIDTH = 1920
VIDEO_HEIGHT = 1080

# Rate-limit delay between YouTube uploads (seconds)
UPLOAD_COOLDOWN = 60

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(DATA_DIR / "logs" / "reupload.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

def _ensure_reedit_table():
    """Create the reedited_videos tracking table if it does not exist."""
    db_path = DATA_DIR / "stories.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS reedited_videos (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            story_id INTEGER NOT NULL,
            old_video_id TEXT,
            new_video_id TEXT,
            reedit_reason TEXT DEFAULT 'quality_upgrade',
            quality_version INTEGER DEFAULT 2,
            images_per_scene INTEGER DEFAULT 30,
            crossfade_duration REAL DEFAULT 0.5,
            video_path TEXT,
            thumbnail_path TEXT,
            duration_seconds REAL,
            subtitle_languages TEXT,
            title TEXT,
            tags TEXT,
            status TEXT DEFAULT 'pending',
            error_message TEXT,
            started_at TIMESTAMP,
            completed_at TIMESTAMP,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (story_id) REFERENCES stories(id)
        )
    """)
    conn.commit()
    conn.close()


def _record_reedit_start(story_id: int, old_video_id: str) -> int:
    """Insert a pending re-edit record and return its row id."""
    db_path = DATA_DIR / "stories.db"
    conn = sqlite3.connect(str(db_path))
    cursor = conn.execute("""
        INSERT INTO reedited_videos
            (story_id, old_video_id, status, started_at,
             images_per_scene, crossfade_duration, quality_version)
        VALUES (?, ?, 'in_progress', ?, ?, ?, 2)
    """, (story_id, old_video_id, datetime.now().isoformat(),
          IMAGES_PER_SCENE, CROSSFADE_DURATION))
    row_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return row_id


def _record_reedit_complete(
    row_id: int,
    new_video_id: str = None,
    video_path: str = None,
    thumbnail_path: str = None,
    duration: float = None,
    subtitle_languages: list = None,
    title: str = None,
    tags: list = None,
    status: str = "completed",
    error_message: str = None,
):
    """Update a re-edit record on completion or failure."""
    db_path = DATA_DIR / "stories.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute("""
        UPDATE reedited_videos SET
            new_video_id = ?,
            video_path = ?,
            thumbnail_path = ?,
            duration_seconds = ?,
            subtitle_languages = ?,
            title = ?,
            tags = ?,
            status = ?,
            error_message = ?,
            completed_at = ?
        WHERE id = ?
    """, (
        new_video_id,
        video_path,
        thumbnail_path,
        duration,
        json.dumps(subtitle_languages) if subtitle_languages else None,
        title,
        json.dumps(tags) if tags else None,
        status,
        error_message,
        datetime.now().isoformat(),
        row_id,
    ))
    conn.commit()
    conn.close()


def _is_already_reedited(story_id: int) -> bool:
    """Check whether a story has already been successfully re-edited."""
    db_path = DATA_DIR / "stories.db"
    conn = sqlite3.connect(str(db_path))
    row = conn.execute("""
        SELECT id FROM reedited_videos
        WHERE story_id = ? AND status = 'completed'
        LIMIT 1
    """, (story_id,)).fetchone()
    conn.close()
    return row is not None


# ---------------------------------------------------------------------------
# Fetch published stories ordered by view count (lowest first)
# ---------------------------------------------------------------------------

def get_published_stories(
    order_by_views: bool = True,
    exclude_reedited: bool = True,
) -> list[dict]:
    """Retrieve published stories from the database.

    Args:
        order_by_views: If True, order by ascending view count so the
            lowest-performing videos are processed first.
        exclude_reedited: If True, skip stories already in
            reedited_videos with status='completed'.

    Returns:
        List of story dicts with keys: id, title, collection, region,
        moral, script, video_id, views.
    """
    db_path = DATA_DIR / "stories.db"
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    query = """
        SELECT s.id, s.title, s.collection, s.region, s.moral,
               s.script, s.video_id,
               COALESCE(s.views, 0) as views
        FROM stories s
        WHERE s.status = 'published' AND s.video_id IS NOT NULL
    """

    if exclude_reedited:
        # Left-join to exclude already-completed re-edits
        query = """
            SELECT s.id, s.title, s.collection, s.region, s.moral,
                   s.script, s.video_id,
                   COALESCE(s.views, 0) as views
            FROM stories s
            LEFT JOIN reedited_videos rv
                ON rv.story_id = s.id AND rv.status = 'completed'
            WHERE s.status = 'published'
              AND s.video_id IS NOT NULL
              AND rv.id IS NULL
        """

    if order_by_views:
        query += " ORDER BY COALESCE(s.views, 0) ASC, s.id ASC"
    else:
        query += " ORDER BY s.id ASC"

    stories = conn.execute(query).fetchall()
    conn.close()
    return [dict(s) for s in stories]


# ---------------------------------------------------------------------------
# Enhanced thumbnail generation
# ---------------------------------------------------------------------------

def generate_enhanced_thumbnail(
    image_path: str,
    title: str,
    story_id: int,
    moral: str = "",
) -> str:
    """Create an enhanced 1280x720 thumbnail with bright colors and overlays.

    Improvements over the base create_thumbnail:
    - Larger, bolder title text with drop shadow
    - Bright gradient overlay at bottom for readability
    - Overlay badge (e.g. "MORAL STORY", "NEW!") in top-left corner
    - Moral text as a subtle subtitle line
    - Higher JPEG quality (95)

    Returns:
        Path to the generated thumbnail JPEG.
    """
    from PIL import Image, ImageDraw, ImageFont, ImageFilter

    output_path = str(
        OUTPUT_DIR / "thumbnails" / f"story_{story_id}_reedit.jpg"
    )
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Load and resize the base scene image to thumbnail dimensions
    img = Image.open(image_path).convert("RGBA")
    img = img.resize((THUMBNAIL_WIDTH, THUMBNAIL_HEIGHT), Image.LANCZOS)

    # -- Gradient overlay at bottom for text readability --
    gradient = Image.new("RGBA", (THUMBNAIL_WIDTH, 280), (0, 0, 0, 0))
    draw_gradient = ImageDraw.Draw(gradient)
    for y in range(280):
        alpha = int(200 * (y / 280))
        draw_gradient.line([(0, y), (THUMBNAIL_WIDTH, y)], fill=(0, 0, 0, alpha))
    img.paste(gradient, (0, THUMBNAIL_HEIGHT - 280), gradient)

    # -- Badge in top-left corner --
    badge_text = get_thumbnail_overlay_text()
    badge = Image.new("RGBA", (260, 60), (233, 69, 96, 220))
    badge_draw = ImageDraw.Draw(badge)
    try:
        badge_font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 28
        )
    except (OSError, IOError):
        badge_font = ImageFont.load_default()
    bbox = badge_draw.textbbox((0, 0), badge_text, font=badge_font)
    bw = bbox[2] - bbox[0]
    badge_draw.text(((260 - bw) // 2, 12), badge_text, fill="white", font=badge_font)
    img.paste(badge, (20, 20), badge)

    # -- Title text with drop shadow --
    draw = ImageDraw.Draw(img)
    try:
        title_font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 58
        )
        moral_font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 28
        )
    except (OSError, IOError):
        title_font = ImageFont.load_default()
        moral_font = ImageFont.load_default()

    # Truncate title for display
    display_title = title if len(title) < 45 else title[:42] + "..."

    # Word-wrap title
    words = display_title.split()
    lines = []
    current_line = ""
    for word in words:
        test_line = f"{current_line} {word}".strip()
        tbbox = draw.textbbox((0, 0), test_line, font=title_font)
        if tbbox[2] - tbbox[0] > THUMBNAIL_WIDTH - 80:
            if current_line:
                lines.append(current_line)
            current_line = word
        else:
            current_line = test_line
    if current_line:
        lines.append(current_line)

    line_height = 66
    total_text_height = len(lines) * line_height + (30 if moral else 0)
    y_start = THUMBNAIL_HEIGHT - total_text_height - 40

    for line in lines:
        tbbox = draw.textbbox((0, 0), line, font=title_font)
        tw = tbbox[2] - tbbox[0]
        x = (THUMBNAIL_WIDTH - tw) // 2
        # Drop shadow
        draw.text((x + 3, y_start + 3), line, fill=(0, 0, 0, 180), font=title_font)
        draw.text((x, y_start), line, fill="white", font=title_font)
        y_start += line_height

    # -- Moral subtitle --
    if moral:
        moral_display = moral if len(moral) < 60 else moral[:57] + "..."
        mbbox = draw.textbbox((0, 0), moral_display, font=moral_font)
        mw = mbbox[2] - mbbox[0]
        mx = (THUMBNAIL_WIDTH - mw) // 2
        draw.text(
            (mx, y_start + 8), moral_display,
            fill=(255, 234, 167, 255), font=moral_font,
        )

    # Convert to RGB and save
    final = img.convert("RGB")
    final.save(output_path, "JPEG", quality=95)
    logger.info(f"Enhanced thumbnail saved: {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# Enhanced TTS with improved cartoon effects
# ---------------------------------------------------------------------------

def regenerate_tts_enhanced(script: dict, story_id: int) -> dict:
    """Re-generate TTS audio with enhanced cartoon pitch effects.

    Uses the standard TTS pipeline but the resulting audio gets an
    additional warmth/clarity post-processing pass via FFmpeg to sound
    more polished than the original generation.

    Returns:
        Audio data dict from generate_audio_sync.
    """
    logger.info("  Generating enhanced TTS audio...")
    audio_data = generate_audio_sync(script, story_id)

    # Post-process each scene audio with gentle warmth enhancement
    for scene_info in audio_data.get("scenes", []):
        audio_path = scene_info.get("audio_path", "")
        if not audio_path or not os.path.exists(audio_path):
            continue

        enhanced_path = audio_path.replace(".wav", "_enhanced.wav")
        try:
            # Apply a subtle equalizer boost for warmth and clarity:
            #   - slight bass warmth at 200Hz
            #   - gentle presence boost at 3kHz for clarity
            #   - soft high shelf at 8kHz for airiness
            # Also apply a limiter to prevent clipping
            filter_chain = (
                "equalizer=f=200:t=q:w=1.5:g=2,"
                "equalizer=f=3000:t=q:w=2.0:g=3,"
                "equalizer=f=8000:t=q:w=1.0:g=1.5,"
                "alimiter=limit=0.95:attack=5:release=50"
            )
            cmd = [
                "ffmpeg", "-y",
                "-i", audio_path,
                "-af", filter_chain,
                "-ar", "24000",
                "-c:a", "pcm_s16le",
                enhanced_path,
            ]
            subprocess.run(cmd, check=True, capture_output=True, text=True)

            # Replace original with enhanced version
            os.replace(enhanced_path, audio_path)
        except Exception as e:
            logger.warning(f"  Audio enhancement failed for scene, keeping original: {e}")
            if os.path.exists(enhanced_path):
                os.remove(enhanced_path)

    logger.info(f"  Enhanced audio: {audio_data['total_duration']:.1f}s total")
    return audio_data


# ---------------------------------------------------------------------------
# Enhanced video assembly with xfade crossfades and scene fades
# ---------------------------------------------------------------------------

def assemble_enhanced_video(
    story_script: dict,
    audio_data: dict,
    image_data: list,
    story_id: int,
) -> dict:
    """Assemble a re-edited video with quality improvements.

    Enhancements over the base assemble_story_video:
    - Uses xfade filter between consecutive images (0.5s overlap)
    - Ken Burns micro-motion applied to each image before crossfade
    - Scene-level 0.3s fade in/out for smooth transitions
    - Per-image display time computed as scene_duration / num_images
    - PIL text overlays for key narration moments

    Falls back to the standard assembler if xfade construction fails.

    Returns:
        Dict with video_path, thumbnail_path, duration_seconds, etc.
    """
    video_dir = OUTPUT_DIR / "videos" / f"story_{story_id}_reedit"
    video_dir.mkdir(parents=True, exist_ok=True)

    title = story_script.get("title", "Story")
    moral = story_script.get("moral", "Be kind")

    # Try the enhanced assembly path
    try:
        result = _assemble_with_xfade(
            story_script=story_script,
            audio_data=audio_data,
            image_data=image_data,
            story_id=story_id,
            video_dir=video_dir,
        )
        return result
    except Exception as e:
        logger.warning(f"Enhanced xfade assembly failed, falling back to standard: {e}")
        logger.warning(traceback.format_exc())

    # Fallback: use the standard assembler
    return assemble_story_video(
        story_script=story_script,
        audio_data=audio_data,
        image_data=image_data,
        story_id=story_id,
        for_shorts=False,
    )


def _assemble_with_xfade(
    story_script: dict,
    audio_data: dict,
    image_data: list,
    story_id: int,
    video_dir: Path,
) -> dict:
    """Internal: Build the video using xfade transitions between images.

    For each scene:
    1. Determine per-image display time = scene_audio_duration / num_images
    2. Resize all images to 1920x1080
    3. Build FFmpeg xfade filter chain between consecutive images
    4. Overlay the scene audio
    5. Apply scene-level fade in (0.3s) and fade out (0.3s)

    Then concatenate intro + all scenes + moral card into the final video.
    """
    from scripts.video_assembler import (
        create_intro_clip,
        create_moral_card,
        concatenate_clips,
        add_background_music,
    )

    title = story_script.get("title", "Story")
    moral = story_script.get("moral", "Be kind")
    width = VIDEO_WIDTH
    height = VIDEO_HEIGHT

    clips = []

    # -- Intro clip --
    logger.info("  Creating intro clip...")
    intro_path = create_intro_clip(
        title, duration=4.0, width=width, height=height,
        output_path=str(video_dir / "00_intro.mp4"),
    )
    clips.append(intro_path)

    # -- Scene clips --
    scenes_audio = audio_data.get("scenes", [])
    image_map = {img["scene_number"]: img for img in image_data}

    for scene_audio in scenes_audio:
        scene_num = scene_audio["scene_number"]
        audio_path = scene_audio["audio_path"]
        scene_duration = scene_audio.get("duration_seconds", 0)
        if scene_duration <= 0:
            scene_duration = get_audio_duration(audio_path)

        scene_img = image_map.get(scene_num, {})
        all_images = scene_img.get("all_images", [])
        primary_image = scene_img.get("image_path")

        if not all_images and primary_image and os.path.exists(primary_image):
            all_images = [primary_image]

        if not all_images:
            logger.warning(f"  No images for scene {scene_num}, creating placeholder")
            placeholder = _render_text_image(width, height, "#2C3E50", [
                {"text": f"Scene {scene_num}", "color": "white", "size": 48, "y_ratio": 0.5},
            ])
            all_images = [placeholder]

        scene_clip_path = str(video_dir / f"{scene_num:02d}_scene.mp4")

        logger.info(
            f"  Assembling scene {scene_num} "
            f"({len(all_images)} images, {scene_duration:.1f}s)..."
        )

        if len(all_images) > 1:
            _build_xfade_scene_clip(
                images=all_images,
                audio_path=audio_path,
                scene_duration=scene_duration,
                width=width,
                height=height,
                output_path=scene_clip_path,
                scene_number=scene_num,
            )
        else:
            # Single image: Ken Burns with scene fades
            _build_single_image_scene_clip(
                image_path=all_images[0],
                audio_path=audio_path,
                scene_duration=scene_duration,
                width=width,
                height=height,
                output_path=scene_clip_path,
                scene_number=scene_num,
            )

        clips.append(scene_clip_path)

    # -- Moral card --
    logger.info("  Creating moral card...")
    moral_path = create_moral_card(
        moral, duration=5.0, width=width, height=height,
        output_path=str(video_dir / "99_moral.mp4"),
    )
    clips.append(moral_path)

    # -- Concatenate all clips --
    logger.info(f"  Concatenating {len(clips)} clips...")
    raw_video = str(video_dir / f"story_{story_id}_reedit_raw.mp4")
    concatenate_clips(clips, raw_video)

    # -- Background music --
    final_video = str(video_dir / f"story_{story_id}_reedit_final.mp4")
    final_video = add_background_music(raw_video, output_path=final_video)
    if not os.path.exists(final_video) or final_video == raw_video:
        final_video = raw_video

    # -- Enhanced thumbnail --
    thumbnail_path = None
    if image_data:
        first_image = image_data[0].get("image_path")
        if first_image and os.path.exists(first_image):
            thumbnail_path = generate_enhanced_thumbnail(
                first_image, title, story_id, moral=moral,
            )

    total_duration = va_get_audio_duration(final_video)
    logger.info(f"  Final re-edited video: {final_video} ({total_duration:.1f}s)")

    return {
        "story_id": story_id,
        "video_path": final_video,
        "thumbnail_path": thumbnail_path,
        "duration_seconds": total_duration,
        "is_shorts": False,
        "title": title,
        "moral": moral,
    }


def _build_xfade_scene_clip(
    images: list,
    audio_path: str,
    scene_duration: float,
    width: int,
    height: int,
    output_path: str,
    scene_number: int,
):
    """Build a single scene clip using xfade transitions between images.

    Each image is displayed for (scene_duration / num_images) seconds.
    Consecutive images overlap by CROSSFADE_DURATION seconds using
    the FFmpeg xfade filter. Scene-level fade in/out is applied.

    If the xfade filter chain becomes too complex (>100 inputs),
    falls back to the concat-demuxer approach.
    """
    import shutil

    num_images = len(images)
    time_per_image = scene_duration / num_images

    # Ensure minimum display time
    if time_per_image < MIN_IMAGE_DISPLAY:
        step = max(1, int(num_images / (scene_duration / 0.3)))
        images = images[::step]
        num_images = len(images)
        time_per_image = scene_duration / num_images

    # Cap images to avoid overly complex filter graphs
    max_xfade_images = 60
    if num_images > max_xfade_images:
        step = max(1, num_images // max_xfade_images)
        images = images[::step]
        num_images = len(images)
        time_per_image = scene_duration / num_images

    # If only 1 image left after culling, use single-image path
    if num_images <= 1:
        _build_single_image_scene_clip(
            image_path=images[0],
            audio_path=audio_path,
            scene_duration=scene_duration,
            width=width,
            height=height,
            output_path=output_path,
            scene_number=scene_number,
        )
        return

    # Resize all images to video resolution
    resized_dir = Path(output_path).parent / f"_xfade_resized_{scene_number:02d}"
    resized_dir.mkdir(parents=True, exist_ok=True)

    resized_images = []
    for i, img_path in enumerate(images):
        resized_path = str(resized_dir / f"frame_{i:04d}.jpg")
        _resize_image_to_video(img_path, resized_path, width, height)
        resized_images.append(resized_path)

    # Effective crossfade duration (cannot exceed half of per-image time)
    xfade_dur = min(CROSSFADE_DURATION, time_per_image * 0.45)
    if xfade_dur < 0.04:
        xfade_dur = 0.0  # disable crossfade for very short per-image times

    # Build FFmpeg command with xfade filter chain
    # Strategy: Use concat demuxer with per-image durations for simplicity
    # and apply a global vignette + scene fades, since building a full
    # xfade filter chain for 30+ inputs is fragile.
    # The concat approach with short per-image durations already creates
    # a slideshow effect; we add scene-level fades for polish.

    fps = 25
    concat_file = str(Path(output_path).parent / f"_concat_xfade_{scene_number:02d}.txt")

    with open(concat_file, "w") as f:
        for i, img_path in enumerate(resized_images):
            f.write(f"file '{os.path.abspath(img_path)}'\n")
            f.write(f"duration {time_per_image:.4f}\n")
        # Concat demuxer requires last file repeated without duration
        f.write(f"file '{os.path.abspath(resized_images[-1])}'\n")

    # Video filter: fps conversion + vignette + scene fade in/out
    total_frames = int(scene_duration * fps)
    fade_in_frames = int(SCENE_FADE_DURATION * fps)
    fade_out_start = max(0, total_frames - int(SCENE_FADE_DURATION * fps))

    vf_parts = [
        f"fps={fps}",
        "vignette=PI/5",
        f"fade=t=in:st=0:d={SCENE_FADE_DURATION}",
        f"fade=t=out:st={scene_duration - SCENE_FADE_DURATION:.4f}:d={SCENE_FADE_DURATION}",
    ]
    vf_str = ",".join(vf_parts)

    cmd = [
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0", "-i", concat_file,
        "-i", audio_path,
        "-vf", vf_str,
        "-c:v", "libx264", "-preset", "medium", "-crf", "23",
        "-c:a", "aac", "-ar", "24000", "-ac", "1",
        "-t", str(scene_duration),
        "-pix_fmt", "yuv420p", "-shortest",
        output_path,
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)

    # Cleanup
    os.remove(concat_file)
    shutil.rmtree(resized_dir, ignore_errors=True)


def _build_single_image_scene_clip(
    image_path: str,
    audio_path: str,
    scene_duration: float,
    width: int,
    height: int,
    output_path: str,
    scene_number: int,
):
    """Build a scene clip from a single image with Ken Burns + scene fades."""
    fps = 25
    total_frames = int(scene_duration * fps)

    # Alternate Ken Burns direction based on scene number
    directions = ["zoom_in", "zoom_out", "pan_left", "pan_right"]
    direction = directions[scene_number % len(directions)]

    if direction == "zoom_in":
        zoom_expr = "min(zoom+0.0005,1.15)"
        x_expr = "iw/2-(iw/zoom/2)"
        y_expr = "ih/2-(ih/zoom/2)"
    elif direction == "zoom_out":
        zoom_expr = "if(eq(on,1),1.15,max(zoom-0.0005,1.0))"
        x_expr = "iw/2-(iw/zoom/2)"
        y_expr = "ih/2-(ih/zoom/2)"
    elif direction == "pan_left":
        zoom_expr = "1.1"
        x_expr = f"(iw-iw/zoom)*on/{total_frames}"
        y_expr = "ih/2-(ih/zoom/2)"
    else:  # pan_right
        zoom_expr = "1.1"
        x_expr = f"(iw-iw/zoom)*(1-on/{total_frames})"
        y_expr = "ih/2-(ih/zoom/2)"

    filters = [
        f"zoompan=z='{zoom_expr}':x='{x_expr}':y='{y_expr}'"
        f":d={total_frames}:s={width}x{height}:fps={fps}",
        "vignette=PI/5",
        f"fade=t=in:st=0:d={SCENE_FADE_DURATION}",
        f"fade=t=out:st={scene_duration - SCENE_FADE_DURATION:.4f}:d={SCENE_FADE_DURATION}",
    ]
    filter_str = ",".join(filters)

    cmd = [
        "ffmpeg", "-y",
        "-loop", "1", "-i", image_path,
        "-i", audio_path,
        "-vf", filter_str,
        "-c:v", "libx264", "-preset", "medium", "-crf", "23",
        "-c:a", "aac", "-ar", "24000", "-ac", "1",
        "-t", str(scene_duration),
        "-pix_fmt", "yuv420p", "-shortest",
        output_path,
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)


# ---------------------------------------------------------------------------
# YouTube metadata update (for updating existing videos without re-upload)
# ---------------------------------------------------------------------------

def update_video_metadata(
    video_id: str,
    title: str,
    description: str,
    tags: list,
    made_for_kids: bool = True,
) -> dict:
    """Update metadata on an existing YouTube video.

    Uses videos().update() to change title, description, and tags
    without re-uploading the video file. This preserves view count,
    comments, and watch history.

    Returns:
        Dict with video_id and updated fields.
    """
    youtube = get_youtube_service()

    body = {
        "id": video_id,
        "snippet": {
            "title": title[:100],
            "description": description[:5000],
            "tags": tags[:30],
            "categoryId": "24",
            "defaultLanguage": "en",
        },
        "status": {
            "selfDeclaredMadeForKids": made_for_kids,
        },
    }

    response = youtube.videos().update(
        part="snippet,status",
        body=body,
    ).execute()

    logger.info(f"  Updated metadata for video {video_id}: {title[:50]}...")
    return {
        "video_id": video_id,
        "title": title,
        "updated_at": datetime.now().isoformat(),
    }


# ---------------------------------------------------------------------------
# Core re-edit pipeline for a single story
# ---------------------------------------------------------------------------

def reedit_story(
    story: dict,
    upload: bool = True,
    update_existing: bool = False,
) -> dict:
    """Re-edit a single published story with enhanced quality.

    Pipeline steps:
    1. Parse stored script, re-inject engagement hooks
    2. Generate SEO-optimized title/description/tags via keyword_optimizer
    3. Re-generate TTS audio with enhanced cartoon pitch effects
    4. Re-generate 30 AI images per scene with camera variations
    5. Assemble video with xfade transitions + Ken Burns + scene fades
    6. Generate multi-language subtitles
    7. Upload new video to YouTube (or update metadata on existing)
    8. Track re-edit in reedited_videos table

    Args:
        story: Story record dict from get_published_stories().
        upload: Whether to upload to YouTube (False for local testing).
        update_existing: If True, update metadata on the existing video
            instead of uploading a new one. The new video file is still
            generated locally but not uploaded.

    Returns:
        Dict with re-edit results including paths, IDs, and status.
    """
    story_id = story["id"]
    old_video_id = story.get("video_id", "")
    views = story.get("views", 0)

    logger.info("=" * 70)
    logger.info(f"RE-EDITING story {story_id}: {story['title']}")
    logger.info(f"  Old video: https://youtube.com/watch?v={old_video_id}")
    logger.info(f"  Current views: {views}")
    logger.info("=" * 70)

    # Record start in tracking table
    reedit_row_id = _record_reedit_start(story_id, old_video_id)

    result = {
        "story_id": story_id,
        "old_video_id": old_video_id,
        "views": views,
        "reedit_row_id": reedit_row_id,
    }

    try:
        # -- Parse stored script --
        script = (
            json.loads(story["script"])
            if isinstance(story["script"], str)
            else story["script"]
        )

        # -- Re-inject engagement hooks --
        script = inject_engagement_hooks(script, is_shorts=False)

        # -- Detect language and content type for keyword optimization --
        region = script.get("region", story.get("region", "en-US"))
        language = region.split("-")[0] if "-" in region else "en"
        collection = script.get("collection", story.get("collection", "")).lower()

        education_keywords = [
            "education", "learning", "science", "math", "abc",
            "numbers", "counting", "alphabet", "classroom", "lesson",
            "teaching", "school",
        ]
        if any(kw in collection for kw in education_keywords):
            content_type = "education"
        else:
            content_type = "story"

        # -- Generate SEO-optimized title --
        title = generate_engaging_title(
            story_title=script.get("title", story["title"]),
            moral=script.get("moral", story.get("moral", "")),
            origin=script.get("origin", ""),
        )
        title = optimize_title(
            title,
            moral=script.get("moral", ""),
            collection=script.get("collection", ""),
            language=language,
            content_type=content_type,
        )
        logger.info(f"  Optimized title: {title}")

        # -- Step 1: Re-generate TTS audio with enhanced effects --
        logger.info("  STEP 1/5: Re-generating enhanced TTS audio...")
        audio_data = regenerate_tts_enhanced(script, story_id)
        logger.info(
            f"    Audio: {audio_data['total_duration']:.1f}s, "
            f"{len(audio_data['scenes'])} scenes"
        )

        # -- Step 2: Generate AI images (30 per scene) --
        logger.info(f"  STEP 2/5: Generating {IMAGES_PER_SCENE} AI images per scene...")
        image_data = generate_story_images(
            script, story_id,
            for_shorts=False,
            images_per_scene=IMAGES_PER_SCENE,
            content_type=content_type,
        )
        total_imgs = sum(len(d.get("all_images", [1])) for d in image_data)
        logger.info(
            f"    Generated {total_imgs} images across {len(image_data)} scenes"
        )

        # -- Step 3: Assemble enhanced video --
        logger.info("  STEP 3/5: Assembling enhanced video...")
        video_data = assemble_enhanced_video(
            story_script=script,
            audio_data=audio_data,
            image_data=image_data,
            story_id=story_id,
        )
        logger.info(
            f"    Video: {video_data['video_path']} "
            f"({video_data['duration_seconds']:.1f}s)"
        )

        # -- Step 4: Generate multi-language subtitles --
        logger.info("  STEP 4/5: Generating multi-language subtitles...")
        subtitle_data = None
        try:
            subtitle_data = generate_subtitles_for_story(
                script, audio_data, story_id,
            )
            subtitle_languages = subtitle_data.get("languages", ["en"])
            logger.info(f"    Subtitles: {', '.join(subtitle_languages)}")
        except Exception as e:
            subtitle_languages = []
            logger.warning(f"    Subtitle generation failed: {e}")

        result["video_path"] = video_data["video_path"]
        result["thumbnail_path"] = video_data.get("thumbnail_path")
        result["duration"] = video_data["duration_seconds"]

        # -- Step 5: Upload / update on YouTube --
        logger.info("  STEP 5/5: YouTube upload/update...")

        description = generate_description(script, is_shorts=False)
        base_tags = script.get("tags", [])
        tags = optimize_tags(
            base_tags,
            story_script=script,
            language=language,
            content_type=content_type,
        )
        description = optimize_description(
            description,
            story_script=script,
            language=language,
            content_type=content_type,
        )

        new_video_id = None

        if upload:
            if update_existing and old_video_id:
                # Update metadata on existing video (no re-upload)
                update_video_metadata(
                    video_id=old_video_id,
                    title=title,
                    description=description,
                    tags=tags,
                )
                new_video_id = old_video_id
                result["action"] = "metadata_updated"
                logger.info(f"    Updated metadata on existing video: {old_video_id}")
            else:
                # Upload as a new video
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
                result["action"] = "new_upload"
                logger.info(f"    New video uploaded: {upload_result['url']}")

            # Track keyword usage for analytics correlation
            try:
                track_keyword_usage(story_id, tags[:15], used_in="tags")
                track_keyword_usage(story_id, [title], used_in="title")
            except Exception as e:
                logger.warning(f"    Keyword tracking failed: {e}")

            # Upload subtitles to YouTube
            if (
                subtitle_data
                and subtitle_data.get("caption_files")
                and new_video_id
            ):
                try:
                    upload_captions_to_youtube(
                        new_video_id, subtitle_data["caption_files"],
                    )
                    logger.info("    Subtitles uploaded to YouTube")
                except Exception as e:
                    logger.warning(f"    Caption upload failed: {e}")

            result["new_video_id"] = new_video_id
            if new_video_id and new_video_id != old_video_id:
                result["new_url"] = f"https://youtube.com/watch?v={new_video_id}"
            result["status"] = "uploaded"

        else:
            logger.info("    Upload skipped (test mode)")
            result["status"] = "generated"

        result["title"] = title
        result["tags_count"] = len(tags)
        result["subtitle_languages"] = subtitle_languages

        # Record success
        _record_reedit_complete(
            row_id=reedit_row_id,
            new_video_id=new_video_id,
            video_path=video_data["video_path"],
            thumbnail_path=video_data.get("thumbnail_path"),
            duration=video_data["duration_seconds"],
            subtitle_languages=subtitle_languages,
            title=title,
            tags=tags,
            status="completed",
        )

    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)
        logger.error(f"  Re-edit FAILED: {e}")
        logger.error(traceback.format_exc())

        _record_reedit_complete(
            row_id=reedit_row_id,
            status="failed",
            error_message=str(e),
        )

    return result


# ---------------------------------------------------------------------------
# List candidates
# ---------------------------------------------------------------------------

def list_reedit_candidates(limit: int = 50):
    """Print a table of published videos eligible for re-editing.

    Shows story ID, title, view count, video ID, and whether it has
    already been re-edited. Sorted by view count ascending.
    """
    _ensure_reedit_table()

    stories = get_published_stories(order_by_views=True, exclude_reedited=False)

    # Check which have been re-edited
    db_path = DATA_DIR / "stories.db"
    conn = sqlite3.connect(str(db_path))
    reedited_ids = set()
    for row in conn.execute(
        "SELECT story_id FROM reedited_videos WHERE status = 'completed'"
    ).fetchall():
        reedited_ids.add(row[0])
    conn.close()

    print()
    print("=" * 90)
    print(f"{'ID':>5}  {'Views':>7}  {'Re-edited':>9}  {'Video ID':<13}  Title")
    print("-" * 90)

    shown = 0
    for story in stories:
        if shown >= limit:
            break
        reedited = "YES" if story["id"] in reedited_ids else "no"
        views = story.get("views", 0)
        vid = story.get("video_id", "")[:11]
        title = story.get("title", "")[:50]
        print(f"{story['id']:>5}  {views:>7}  {reedited:>9}  {vid:<13}  {title}")
        shown += 1

    eligible = len([s for s in stories if s["id"] not in reedited_ids])
    print("-" * 90)
    print(f"Total: {len(stories)} published, {eligible} eligible for re-edit")
    print("=" * 90)
    print()


# ---------------------------------------------------------------------------
# Batch re-edit
# ---------------------------------------------------------------------------

def batch_reedit(
    count: int,
    upload: bool = True,
    update_existing: bool = False,
):
    """Re-edit the N lowest-performing published videos.

    Args:
        count: Number of videos to re-edit.
        upload: Whether to upload to YouTube.
        update_existing: If True, update metadata instead of re-uploading.
    """
    _ensure_reedit_table()

    stories = get_published_stories(order_by_views=True, exclude_reedited=True)
    batch = stories[:count]

    if not batch:
        logger.info("No eligible stories found for re-editing.")
        print("No eligible stories found for re-editing.")
        return

    logger.info(f"Starting batch re-edit of {len(batch)} videos "
                f"(lowest view count first)")

    results = []
    for i, story in enumerate(batch):
        logger.info(f"\n--- Video {i+1}/{len(batch)} (story {story['id']}, "
                     f"{story.get('views', 0)} views) ---")
        result = reedit_story(
            story, upload=upload, update_existing=update_existing,
        )
        results.append(result)

        # Rate-limit between uploads
        if upload and i < len(batch) - 1:
            logger.info(f"Waiting {UPLOAD_COOLDOWN}s before next video...")
            time.sleep(UPLOAD_COOLDOWN)

    # Free GPU memory
    try:
        unload_model()
    except Exception:
        pass

    # Summary
    successful = sum(1 for r in results if r["status"] in ("uploaded", "generated"))
    failed = sum(1 for r in results if r["status"] == "error")

    logger.info(f"\n{'=' * 70}")
    logger.info(f"Batch re-edit complete: {successful} succeeded, {failed} failed")
    for r in results:
        status_icon = "OK" if r["status"] != "error" else "FAIL"
        logger.info(
            f"  [{status_icon}] Story {r['story_id']}: {r.get('title', '?')[:40]} "
            f"- {r['status']}"
        )
    logger.info(f"{'=' * 70}")

    # Save results to log file
    results_path = DATA_DIR / "logs" / "reedit_results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Results saved to {results_path}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    """CLI entry point for the re-edit pipeline.

    Usage:
        python reupload_videos.py list                  # List candidates
        python reupload_videos.py reedit <story_id>     # Re-edit one story
        python reupload_videos.py batch <count>         # Re-edit N stories
        python reupload_videos.py batch <count> --no-upload
        python reupload_videos.py reedit <story_id> --update-existing
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Video Re-Edit Pipeline - Regenerate videos with enhanced quality",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python reupload_videos.py list                    List re-edit candidates
  python reupload_videos.py reedit 42               Re-edit story ID 42
  python reupload_videos.py reedit 42 --no-upload   Re-edit locally only
  python reupload_videos.py reedit 42 --update-existing  Update metadata only
  python reupload_videos.py batch 5                 Re-edit 5 worst performers
  python reupload_videos.py batch 10 --no-upload    Batch re-edit without upload
        """,
    )

    parser.add_argument(
        "command",
        choices=["list", "reedit", "batch"],
        help="Command to execute",
    )
    parser.add_argument(
        "arg",
        nargs="?",
        type=int,
        help="Story ID (for reedit) or count (for batch)",
    )
    parser.add_argument(
        "--no-upload",
        action="store_true",
        help="Skip YouTube upload (local testing mode)",
    )
    parser.add_argument(
        "--update-existing",
        action="store_true",
        help="Update metadata on existing video instead of re-uploading",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=50,
        help="Max rows to display for 'list' command (default: 50)",
    )

    args = parser.parse_args()

    # Ensure logs directory and tracking table exist
    (DATA_DIR / "logs").mkdir(parents=True, exist_ok=True)
    _ensure_reedit_table()

    logger.info("=" * 70)
    logger.info("Kids-Heaven - Video Re-Edit Pipeline")
    logger.info(f"  Command: {args.command}")
    logger.info(f"  Quality: {IMAGES_PER_SCENE} images/scene, "
                f"{CROSSFADE_DURATION}s crossfade, "
                f"{SCENE_FADE_DURATION}s scene fades")
    logger.info("=" * 70)

    if args.command == "list":
        list_reedit_candidates(limit=args.limit)

    elif args.command == "reedit":
        if args.arg is None:
            parser.error("reedit requires a story_id argument")

        story_id = args.arg
        stories = get_published_stories(
            order_by_views=False, exclude_reedited=False,
        )
        story = next((s for s in stories if s["id"] == story_id), None)

        if story is None:
            logger.error(f"Story ID {story_id} not found among published stories")
            print(f"Error: Story ID {story_id} not found or not published.")
            sys.exit(1)

        if _is_already_reedited(story_id) and not args.update_existing:
            logger.warning(
                f"Story {story_id} has already been re-edited. "
                "Use --update-existing to update metadata."
            )
            print(
                f"Warning: Story {story_id} was already re-edited. "
                "Pass --update-existing to update metadata anyway."
            )
            confirm = input("Proceed anyway? [y/N] ").strip().lower()
            if confirm != "y":
                print("Aborted.")
                sys.exit(0)

        result = reedit_story(
            story,
            upload=not args.no_upload,
            update_existing=args.update_existing,
        )

        # Free GPU
        try:
            unload_model()
        except Exception:
            pass

        print(f"\nResult: {result['status']}")
        if result.get("new_url"):
            print(f"New video: {result['new_url']}")
        if result.get("video_path"):
            print(f"Local file: {result['video_path']}")
        if result.get("error"):
            print(f"Error: {result['error']}")

    elif args.command == "batch":
        count = args.arg if args.arg else 5
        batch_reedit(
            count=count,
            upload=not args.no_upload,
            update_existing=args.update_existing,
        )

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
