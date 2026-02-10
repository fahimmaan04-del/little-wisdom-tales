"""
Video Assembler - Creates videos with motion effects using FFmpeg.

Combines scene images + audio + subtitles + transitions into a final
video with Ken Burns effects, crossfades, and text overlays.
"""

import json
import os
import subprocess
import tempfile
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "./output"))
ASSETS_DIR = Path(os.getenv("ASSETS_DIR", "./assets"))

# Scene-level transition settings
SCENE_FADE_DURATION = 0.3  # seconds of fade in/out at scene boundaries

# Content-type branding colors and text
CONTENT_BRANDING = {
    "story": {
        "bg_color": "#1a1a2e",
        "accent_color": "#e94560",
        "channel_name": "Little Wisdom Tales",
        "subtitle": "A Story for Kids",
        "moral_bg": "#2d3436",
        "moral_accent": "#ffeaa7",
    },
    "education": {
        "bg_color": "#0a3d62",
        "accent_color": "#f6b93b",
        "channel_name": "Little Wisdom Academy",
        "subtitle": "Learn with Professor Wisdom!",
        "moral_bg": "#1e3799",
        "moral_accent": "#f8c291",
    },
    "ai_education": {
        "bg_color": "#0c2461",
        "accent_color": "#00d2d3",
        "channel_name": "Little Wisdom AI Lab",
        "subtitle": "Explore with Byte the Robot!",
        "moral_bg": "#1B1464",
        "moral_accent": "#55efc4",
    },
    "crafts_skills": {
        "bg_color": "#3d1f00",
        "accent_color": "#f0932b",
        "channel_name": "Little Builders Workshop",
        "subtitle": "Build with Handy the Helper!",
        "moral_bg": "#4a2800",
        "moral_accent": "#ffeaa7",
    },
}


def get_audio_duration(audio_path: str) -> float:
    """Get duration of an audio file."""
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
         "-of", "csv=p=0", audio_path],
        capture_output=True, text=True,
    )
    try:
        return float(result.stdout.strip())
    except (ValueError, AttributeError):
        return 15.0


def _render_text_image(
    width: int, height: int, bg_color: str,
    texts: list[dict],
) -> str:
    """Render text onto an image using PIL (no FFmpeg drawtext needed).

    texts: list of dicts with keys: text, color, size, y_ratio
    """
    from PIL import Image, ImageDraw, ImageFont

    img = Image.new("RGB", (width, height), bg_color)
    draw = ImageDraw.Draw(img)

    for t in texts:
        try:
            font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", t["size"]
            )
        except (OSError, IOError):
            font = ImageFont.load_default()

        text = t["text"]
        # Word wrap
        words = text.split()
        lines = []
        current = ""
        for w in words:
            test = f"{current} {w}".strip()
            bbox = draw.textbbox((0, 0), test, font=font)
            if bbox[2] - bbox[0] > width - 80:
                if current:
                    lines.append(current)
                current = w
            else:
                current = test
        if current:
            lines.append(current)

        line_height = t["size"] + 8
        total_h = len(lines) * line_height
        y_start = int(height * t["y_ratio"]) - total_h // 2

        for line in lines:
            bbox = draw.textbbox((0, 0), line, font=font)
            x = (width - (bbox[2] - bbox[0])) // 2
            draw.text((x, y_start), line, fill=t["color"], font=font)
            y_start += line_height

    tmp_path = str(OUTPUT_DIR / "videos" / f"_tmp_text_{id(texts)}.jpg")
    Path(tmp_path).parent.mkdir(parents=True, exist_ok=True)
    img.save(tmp_path, "JPEG", quality=95)
    return tmp_path


def create_intro_clip(
    title: str,
    duration: float = 4.0,
    width: int = 1920,
    height: int = 1080,
    output_path: str = None,
    content_type: str = "story",
) -> str:
    """Create a branded intro clip with title using PIL-rendered image."""
    if output_path is None:
        output_path = str(OUTPUT_DIR / "videos" / "temp_intro.mp4")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    branding = CONTENT_BRANDING.get(content_type, CONTENT_BRANDING["story"])

    # Render intro image with PIL using content-type branding
    intro_img = _render_text_image(width, height, branding["bg_color"], [
        {"text": branding["channel_name"], "color": branding["accent_color"], "size": 36, "y_ratio": 0.15},
        {"text": title, "color": "white", "size": 56, "y_ratio": 0.5},
        {"text": branding["subtitle"], "color": "#f0d9b5", "size": 30, "y_ratio": 0.75},
    ])

    cmd = [
        "ffmpeg", "-y",
        "-loop", "1", "-i", intro_img,
        "-f", "lavfi", "-i", "anullsrc=r=24000:cl=mono",
        "-t", str(duration),
        "-c:v", "libx264", "-preset", "medium", "-crf", "23",
        "-c:a", "aac",
        "-pix_fmt", "yuv420p",
        "-shortest",
        output_path,
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)

    os.remove(intro_img)
    return output_path


def create_moral_card(
    moral: str,
    duration: float = 5.0,
    width: int = 1920,
    height: int = 1080,
    output_path: str = None,
    content_type: str = "story",
) -> str:
    """Create a branded ending card showing the moral/lesson summary."""
    if output_path is None:
        output_path = str(OUTPUT_DIR / "videos" / "temp_moral.mp4")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    branding = CONTENT_BRANDING.get(content_type, CONTENT_BRANDING["story"])

    # Header text varies by content type
    header_text = {
        "story": "The Moral of the Story",
        "education": "What We Learned Today!",
        "ai_education": "Key Takeaway!",
        "crafts_skills": "What We Built Today!",
    }.get(content_type, "The Moral of the Story")

    # Render moral card with PIL using content-type branding
    moral_img = _render_text_image(width, height, branding["moral_bg"], [
        {"text": header_text, "color": branding["moral_accent"], "size": 42, "y_ratio": 0.35},
        {"text": moral, "color": "white", "size": 36, "y_ratio": 0.55},
    ])

    cmd = [
        "ffmpeg", "-y",
        "-loop", "1", "-i", moral_img,
        "-f", "lavfi", "-i", "anullsrc=r=24000:cl=mono",
        "-t", str(duration),
        "-c:v", "libx264", "-preset", "medium", "-crf", "23",
        "-c:a", "aac",
        "-pix_fmt", "yuv420p",
        "-shortest",
        output_path,
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)

    os.remove(moral_img)
    return output_path


def create_scene_clip(
    image_path: str,
    audio_path: str,
    scene_number: int,
    subtitle_text: str = None,
    width: int = 1920,
    height: int = 1080,
    output_path: str = None,
    ken_burns_direction: str = "zoom_in",
    all_images: list = None,
) -> str:
    """Create a scene clip from one or many images with audio.

    If all_images is provided (list of image paths), creates an animated
    crossfade slideshow. Otherwise falls back to single-image Ken Burns.
    """
    if output_path is None:
        output_path = str(OUTPUT_DIR / "videos" / f"temp_scene_{scene_number:02d}.mp4")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    duration = get_audio_duration(audio_path)

    # Multi-image animated slideshow
    if all_images and len(all_images) > 1:
        return _create_multi_image_clip(
            images=all_images,
            audio_path=audio_path,
            duration=duration,
            width=width,
            height=height,
            output_path=output_path,
        )

    # Single image fallback: Ken Burns effect
    return _create_single_image_clip(
        image_path=image_path,
        audio_path=audio_path,
        duration=duration,
        width=width,
        height=height,
        output_path=output_path,
        ken_burns_direction=ken_burns_direction,
    )


def _create_single_image_clip(
    image_path: str,
    audio_path: str,
    duration: float,
    width: int,
    height: int,
    output_path: str,
    ken_burns_direction: str = "zoom_in",
) -> str:
    """Single image with Ken Burns effect (legacy fallback)."""
    fps = 25
    total_frames = int(duration * fps)

    if ken_burns_direction == "zoom_in":
        zoom_expr = "min(zoom+0.0005,1.15)"
        x_expr = "iw/2-(iw/zoom/2)"
        y_expr = "ih/2-(ih/zoom/2)"
    elif ken_burns_direction == "zoom_out":
        zoom_expr = "if(eq(on,1),1.15,max(zoom-0.0005,1.0))"
        x_expr = "iw/2-(iw/zoom/2)"
        y_expr = "ih/2-(ih/zoom/2)"
    elif ken_burns_direction == "pan_left":
        zoom_expr = "1.1"
        x_expr = f"(iw-iw/zoom)*on/{total_frames}"
        y_expr = "ih/2-(ih/zoom/2)"
    elif ken_burns_direction == "pan_right":
        zoom_expr = "1.1"
        x_expr = f"(iw-iw/zoom)*(1-on/{total_frames})"
        y_expr = "ih/2-(ih/zoom/2)"
    else:
        zoom_expr = "min(zoom+0.0005,1.15)"
        x_expr = "iw/2-(iw/zoom/2)"
        y_expr = "ih/2-(ih/zoom/2)"

    filters = [
        f"zoompan=z='{zoom_expr}':x='{x_expr}':y='{y_expr}'"
        f":d={total_frames}:s={width}x{height}:fps={fps}",
        "vignette=PI/5",
    ]
    filter_str = ",".join(filters)

    cmd = [
        "ffmpeg", "-y",
        "-loop", "1", "-i", image_path,
        "-i", audio_path,
        "-vf", filter_str,
        "-c:v", "libx264", "-preset", "medium", "-crf", "23",
        "-c:a", "aac", "-ar", "24000", "-ac", "1",
        "-t", str(duration),
        "-pix_fmt", "yuv420p", "-shortest",
        output_path,
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    return output_path


def _create_multi_image_clip(
    images: list,
    audio_path: str,
    duration: float,
    width: int,
    height: int,
    output_path: str,
) -> str:
    """Create an animated slideshow clip from multiple images.

    Uses FFmpeg concat demuxer with per-image durations to create a rapid
    crossfade effect. Each image gets a slight Ken Burns micro-motion.
    Images cycle through at ~0.3-0.5s each, creating animation-like flow.
    """
    fps = 25
    num_images = len(images)
    time_per_image = duration / num_images

    # Ensure minimum display time of 0.2s per image
    if time_per_image < 0.2:
        # Too many images for the duration - subsample
        step = max(1, int(num_images / (duration / 0.3)))
        images = images[::step]
        num_images = len(images)
        time_per_image = duration / num_images

    # Resize all images to video resolution first
    resized_dir = Path(output_path).parent / f"_resized_{Path(output_path).stem}"
    resized_dir.mkdir(parents=True, exist_ok=True)

    resized_images = []
    for i, img_path in enumerate(images):
        resized_path = str(resized_dir / f"frame_{i:04d}.jpg")
        _resize_image_to_video(img_path, resized_path, width, height)
        resized_images.append(resized_path)

    # Write concat file with durations
    concat_file = str(Path(output_path).parent / f"_concat_{Path(output_path).stem}.txt")
    with open(concat_file, "w") as f:
        for i, img_path in enumerate(resized_images):
            f.write(f"file '{os.path.abspath(img_path)}'\n")
            f.write(f"duration {time_per_image:.4f}\n")
        # FFmpeg concat demuxer needs the last file repeated without duration
        f.write(f"file '{os.path.abspath(resized_images[-1])}'\n")

    # Build video from image sequence + audio with scene fades
    vf_parts = [
        f"fps={fps}",
        "vignette=PI/5",
        f"fade=t=in:st=0:d={SCENE_FADE_DURATION}",
        f"fade=t=out:st={max(0, duration - SCENE_FADE_DURATION):.4f}:d={SCENE_FADE_DURATION}",
    ]
    vf_str = ",".join(vf_parts)

    cmd = [
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0", "-i", concat_file,
        "-i", audio_path,
        "-vf", vf_str,
        "-c:v", "libx264", "-preset", "medium", "-crf", "23",
        "-c:a", "aac", "-ar", "24000", "-ac", "1",
        "-t", str(duration),
        "-pix_fmt", "yuv420p", "-shortest",
        output_path,
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)

    # Cleanup temp files
    import shutil
    os.remove(concat_file)
    shutil.rmtree(resized_dir, ignore_errors=True)

    return output_path


def _resize_image_to_video(
    input_path: str, output_path: str, width: int, height: int
) -> str:
    """Resize and pad an AI-generated image to exact video dimensions."""
    from PIL import Image as PILImage

    img = PILImage.open(input_path)
    if img.mode != "RGB":
        img = img.convert("RGB")

    # Calculate resize to fill (cover), then center crop
    target_ratio = width / height
    img_ratio = img.width / img.height

    if img_ratio > target_ratio:
        # Wider - resize by height, crop width
        new_h = height
        new_w = int(img.width * (height / img.height))
    else:
        # Taller - resize by width, crop height
        new_w = width
        new_h = int(img.height * (width / img.width))

    img = img.resize((new_w, new_h), PILImage.LANCZOS)

    # Center crop to exact dimensions
    left = (new_w - width) // 2
    top = (new_h - height) // 2
    img = img.crop((left, top, left + width, top + height))

    img.save(output_path, "JPEG", quality=95)
    return output_path


def concatenate_clips(clip_paths: list, output_path: str) -> str:
    """Concatenate video clips with crossfade transitions."""
    if len(clip_paths) == 1:
        subprocess.run(["cp", clip_paths[0], output_path], check=True)
        return output_path

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Write concat file
    concat_file = str(Path(output_path).parent / "concat_list.txt")
    with open(concat_file, "w") as f:
        for clip in clip_paths:
            f.write(f"file '{os.path.abspath(clip)}'\n")

    cmd = [
        "ffmpeg", "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", concat_file,
        "-c", "copy",
        output_path,
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)

    os.remove(concat_file)
    return output_path


def add_background_music(
    video_path: str,
    music_path: str = None,
    output_path: str = None,
    music_volume: float = 0.08,
) -> str:
    """Add soft background music to the video."""
    if music_path is None:
        # Check for any music files in assets
        music_dir = ASSETS_DIR / "music"
        if music_dir.exists():
            music_files = list(music_dir.glob("*.mp3")) + list(music_dir.glob("*.wav"))
            if music_files:
                music_path = str(music_files[0])

    if music_path is None or not os.path.exists(music_path):
        return video_path  # No music available, return as-is

    if output_path is None:
        output_path = video_path.replace(".mp4", "_with_music.mp4")

    # Get video duration
    video_duration = get_audio_duration(video_path)

    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-stream_loop", "-1",
        "-i", music_path,
        "-filter_complex",
        f"[1:a]volume={music_volume},afade=t=out:st={video_duration-2}:d=2[music];"
        f"[0:a][music]amix=inputs=2:duration=first:dropout_transition=2[aout]",
        "-map", "0:v",
        "-map", "[aout]",
        "-c:v", "copy",
        "-c:a", "aac",
        "-t", str(video_duration),
        output_path,
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    return output_path


def create_thumbnail(
    image_path: str,
    title: str,
    output_path: str = None,
) -> str:
    """Create an eye-catching thumbnail from the first scene image."""
    from PIL import Image, ImageDraw, ImageFont

    if output_path is None:
        output_path = str(OUTPUT_DIR / "thumbnails" / "temp_thumb.jpg")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Load and resize base image
    img = Image.open(image_path).resize((1280, 720), Image.LANCZOS)
    draw = ImageDraw.Draw(img)

    # Add semi-transparent overlay at bottom
    overlay = Image.new("RGBA", (1280, 200), (0, 0, 0, 160))
    img = img.convert("RGBA")
    img.paste(overlay, (0, 520), overlay)

    # Add title text
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 52
        )
    except (OSError, IOError):
        font = ImageFont.load_default()

    draw = ImageDraw.Draw(img)

    # Truncate title if too long
    display_title = title if len(title) < 40 else title[:37] + "..."
    bbox = draw.textbbox((0, 0), display_title, font=font)
    text_w = bbox[2] - bbox[0]
    x = (1280 - text_w) // 2
    draw.text((x, 560), display_title, fill="white", font=font)

    # Save as JPEG
    img = img.convert("RGB")
    img.save(output_path, "JPEG", quality=95)
    return output_path


def assemble_story_video(
    story_script: dict,
    audio_data: dict,
    image_data: list,
    story_id: int,
    for_shorts: bool = False,
) -> dict:
    """Assemble a complete story video from all components."""
    video_dir = OUTPUT_DIR / "videos" / f"story_{story_id}"
    video_dir.mkdir(parents=True, exist_ok=True)

    title = story_script.get("title", "Story")
    moral = story_script.get("moral", "Be kind")
    width = 1080 if for_shorts else 1920
    height = 1920 if for_shorts else 1080

    clips = []

    # 1. Create intro clip
    print(f"  Creating intro clip...")
    intro_path = create_intro_clip(
        title, duration=4.0, width=width, height=height,
        output_path=str(video_dir / "00_intro.mp4"),
    )
    clips.append(intro_path)

    # 2. Create scene clips with Ken Burns effects
    ken_burns_directions = ["zoom_in", "zoom_out", "pan_left", "pan_right"]
    scenes = audio_data.get("scenes", [])
    image_map = {img["scene_number"]: img for img in image_data}

    for i, scene_audio in enumerate(scenes):
        scene_num = scene_audio["scene_number"]
        print(f"  Assembling scene {scene_num}...")

        scene_img = image_map.get(scene_num, {})
        image_path = scene_img.get("image_path")
        all_images = scene_img.get("all_images")  # Multi-image support

        if not image_path or not os.path.exists(image_path):
            print(f"    Warning: No image for scene {scene_num}, creating placeholder")
            scene_data = next((s for s in story_script["scenes"] if s["scene_number"] == scene_num), {})
            from scripts.image_fetcher import create_text_overlay_image
            image_path = create_text_overlay_image(
                scene_data.get("visual_description", f"Scene {scene_num}"),
                width=width, height=height, bg_color="#2C3E50",
            )
            all_images = None

        # Alternate Ken Burns direction for visual variety (single-image fallback)
        kb_dir = ken_burns_directions[i % len(ken_burns_directions)]

        scene_data = next(
            (s for s in story_script["scenes"] if s["scene_number"] == scene_num), {}
        )
        subtitle = scene_data.get("narration", "")

        clip_path = create_scene_clip(
            image_path=image_path,
            audio_path=scene_audio["audio_path"],
            scene_number=scene_num,
            subtitle_text=subtitle,
            width=width,
            height=height,
            output_path=str(video_dir / f"{scene_num:02d}_scene.mp4"),
            ken_burns_direction=kb_dir,
            all_images=all_images,
        )
        clips.append(clip_path)

    # 3. Create moral card at the end
    print(f"  Creating moral card...")
    moral_path = create_moral_card(
        moral, duration=5.0, width=width, height=height,
        output_path=str(video_dir / "99_moral.mp4"),
    )
    clips.append(moral_path)

    # 4. Concatenate all clips
    print(f"  Concatenating {len(clips)} clips...")
    suffix = "_shorts" if for_shorts else ""
    raw_video = str(video_dir / f"story_{story_id}{suffix}_raw.mp4")
    concatenate_clips(clips, raw_video)

    # 5. Add background music (if available)
    final_video = str(video_dir / f"story_{story_id}{suffix}_final.mp4")
    final_video = add_background_music(raw_video, output_path=final_video)

    # If no music was added, the raw video is the final
    if not os.path.exists(final_video) or final_video == raw_video:
        final_video = raw_video

    # 6. Create thumbnail
    thumbnail_path = None
    if image_data:
        first_image = image_data[0].get("image_path")
        if first_image and os.path.exists(first_image):
            thumbnail_path = str(OUTPUT_DIR / "thumbnails" / f"story_{story_id}{suffix}.jpg")
            create_thumbnail(first_image, title, thumbnail_path)

    total_duration = get_audio_duration(final_video)
    print(f"  Final video: {final_video} ({total_duration:.1f}s)")

    return {
        "story_id": story_id,
        "video_path": final_video,
        "thumbnail_path": thumbnail_path,
        "duration_seconds": total_duration,
        "is_shorts": for_shorts,
        "title": title,
        "moral": moral,
    }


if __name__ == "__main__":
    print("Video assembler module - import and use assemble_story_video()")
    print("Requires: story_script, audio_data, image_data, story_id")
