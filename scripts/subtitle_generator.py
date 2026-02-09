"""
Subtitle Generator - Multi-language subtitle support for YouTube videos.

Generates SRT subtitle files from story scripts with scene timing data,
translates them to multiple languages, and uploads captions via YouTube
Data API v3 for global kid-friendly accessibility.

Supported languages: English, Spanish, Hindi, French, Portuguese, Arabic.
"""

import json
import logging
import os
import re
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "./output"))
DATA_DIR = Path(os.getenv("DATA_DIR", "./data"))

logger = logging.getLogger(__name__)

# Target languages for translation (ISO 639-1 codes)
DEFAULT_TARGET_LANGUAGES = ["es", "hi", "fr", "pt", "ar"]

# Human-readable language names for logging and YouTube caption metadata
LANGUAGE_NAMES = {
    "en": "English",
    "es": "Spanish",
    "hi": "Hindi",
    "fr": "French",
    "pt": "Portuguese",
    "ar": "Arabic",
}

# Maximum words per subtitle chunk for readability (kids aged 4-10)
MAX_WORDS_PER_CHUNK = 10


def format_srt_timestamp(seconds: float) -> str:
    """Convert seconds to SRT timestamp format HH:MM:SS,mmm.

    Args:
        seconds: Time in seconds (e.g. 63.5 -> 00:01:03,500).

    Returns:
        SRT-formatted timestamp string.
    """
    if seconds < 0:
        seconds = 0.0

    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int(round((seconds - int(seconds)) * 1000))

    # Clamp millis to 999 to avoid rollover from rounding
    if millis >= 1000:
        millis = 999

    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def split_text_into_chunks(text: str, max_words: int = MAX_WORDS_PER_CHUNK) -> list[str]:
    """Split narration text into subtitle-sized chunks.

    Splits on word boundaries, keeping approximately max_words per chunk.
    Tries to break at natural punctuation points (commas, periods) when possible.

    Args:
        text: The narration text to split.
        max_words: Maximum number of words per subtitle chunk.

    Returns:
        List of text chunks suitable for subtitle display.
    """
    if not text or not text.strip():
        return []

    text = text.strip()
    words = text.split()

    if len(words) <= max_words:
        return [text]

    chunks = []
    current_chunk_words = []

    for word in words:
        current_chunk_words.append(word)

        if len(current_chunk_words) >= max_words:
            chunk_text = " ".join(current_chunk_words)
            chunks.append(chunk_text)
            current_chunk_words = []
        elif len(current_chunk_words) >= max_words - 3:
            # Near the limit -- check if we can break at punctuation
            if word.endswith((",", ".", "!", "?", ";", ":")):
                chunk_text = " ".join(current_chunk_words)
                chunks.append(chunk_text)
                current_chunk_words = []

    # Add remaining words
    if current_chunk_words:
        chunks.append(" ".join(current_chunk_words))

    return chunks


def compute_scene_start_times(audio_data: dict) -> list[dict]:
    """Compute absolute start times for each scene from audio duration data.

    The audio_data from tts_generator contains per-scene durations but not
    absolute start times. This function accumulates durations to produce
    start_time and end_time for each scene.

    Args:
        audio_data: Audio generation result from tts_generator.generate_audio_sync().
            Expected structure: {"scenes": [{"scene_number": int, "duration_seconds": float}, ...]}

    Returns:
        List of dicts with scene_number, start_time, end_time, duration.
    """
    scenes = audio_data.get("scenes", [])
    timed_scenes = []
    current_time = 0.0

    for scene in scenes:
        duration = scene.get("duration_seconds", 5.0)
        timed_scenes.append({
            "scene_number": scene.get("scene_number", len(timed_scenes) + 1),
            "start_time": current_time,
            "end_time": current_time + duration,
            "duration": duration,
        })
        current_time += duration

    return timed_scenes


def generate_srt(story_script: dict, audio_data: dict, story_id: int) -> str:
    """Generate an SRT subtitle file from a story script with scene timing.

    Creates properly formatted SRT subtitles by:
    1. Computing scene start/end times from audio duration data.
    2. Splitting long narration text into readable chunks (~10 words each).
    3. Distributing chunk timings evenly across each scene's duration.

    Args:
        story_script: Story script dict with scenes containing "narration" text.
        audio_data: Audio generation result with per-scene duration data.
        story_id: Unique story identifier for output file naming.

    Returns:
        Absolute path to the generated SRT file.
    """
    scenes = story_script.get("scenes", [])
    timed_scenes = compute_scene_start_times(audio_data)

    # Build a lookup of timing by scene_number
    timing_by_scene = {ts["scene_number"]: ts for ts in timed_scenes}

    srt_entries = []
    subtitle_index = 1

    for scene in scenes:
        scene_num = scene.get("scene_number", 0)
        narration = scene.get("narration", scene.get("text", ""))

        if not narration or not narration.strip():
            continue

        # Get timing for this scene
        timing = timing_by_scene.get(scene_num)
        if timing is None:
            # If no timing data for this scene, skip it
            logger.warning(f"No timing data for scene {scene_num}, skipping subtitle")
            continue

        scene_start = timing["start_time"]
        scene_duration = timing["duration"]

        # Split narration into readable chunks
        chunks = split_text_into_chunks(narration)
        if not chunks:
            continue

        # Distribute chunks evenly across the scene duration
        chunk_duration = scene_duration / len(chunks)

        # Enforce minimum subtitle display time (1.5 seconds for kids)
        min_display_time = 1.5
        if chunk_duration < min_display_time and len(chunks) > 1:
            # Recalculate: fewer chunks with more words
            bigger_max_words = MAX_WORDS_PER_CHUNK + 5
            chunks = split_text_into_chunks(narration, max_words=bigger_max_words)
            chunk_duration = scene_duration / len(chunks)

        for i, chunk in enumerate(chunks):
            chunk_start = scene_start + (i * chunk_duration)
            chunk_end = chunk_start + chunk_duration

            # Ensure the last chunk does not exceed scene boundary
            if chunk_end > timing["end_time"]:
                chunk_end = timing["end_time"]

            # Skip chunks with near-zero duration
            if chunk_end - chunk_start < 0.1:
                continue

            start_ts = format_srt_timestamp(chunk_start)
            end_ts = format_srt_timestamp(chunk_end)

            srt_entries.append(f"{subtitle_index}\n{start_ts} --> {end_ts}\n{chunk}\n")
            subtitle_index += 1

    # Write SRT file
    srt_content = "\n".join(srt_entries)

    subtitle_dir = OUTPUT_DIR / "subtitles" / f"story_{story_id}"
    subtitle_dir.mkdir(parents=True, exist_ok=True)

    srt_path = subtitle_dir / "subtitles_en.srt"
    srt_path.write_text(srt_content, encoding="utf-8")

    logger.info(f"Generated English SRT with {subtitle_index - 1} subtitle entries: {srt_path}")
    print(f"  Generated English SRT: {srt_path} ({subtitle_index - 1} entries)")

    return str(srt_path)


def translate_subtitles(
    srt_path: str,
    target_languages: list[str] | None = None,
) -> dict[str, str]:
    """Translate an SRT subtitle file to multiple languages.

    Uses the deep-translator library (GoogleTranslator, free, no API key) to
    translate subtitle text while preserving SRT formatting and timestamps.

    If deep-translator is not installed, returns only the original English SRT.

    Args:
        srt_path: Path to the source English SRT file.
        target_languages: List of ISO 639-1 language codes to translate to.
            Defaults to ["es", "hi", "fr", "pt", "ar"].

    Returns:
        Dict mapping language code to SRT file path, e.g.
        {"en": "/path/en.srt", "es": "/path/es.srt", ...}
    """
    if target_languages is None:
        target_languages = DEFAULT_TARGET_LANGUAGES

    srt_file = Path(srt_path)
    if not srt_file.exists():
        raise FileNotFoundError(f"SRT file not found: {srt_path}")

    subtitle_dir = srt_file.parent
    original_content = srt_file.read_text(encoding="utf-8")

    # Result always includes the original English
    result = {"en": str(srt_file)}

    # Try to import deep-translator
    try:
        from deep_translator import GoogleTranslator  # noqa: F401
    except ImportError:
        logger.warning(
            "deep-translator not installed. Returning English-only subtitles. "
            "Install with: pip install deep-translator"
        )
        print("  Warning: deep-translator not installed, skipping translations")
        return result

    # Parse SRT into blocks for translation
    srt_blocks = _parse_srt_blocks(original_content)

    if not srt_blocks:
        logger.warning("No subtitle blocks found in SRT file")
        return result

    for lang_code in target_languages:
        lang_name = LANGUAGE_NAMES.get(lang_code, lang_code)
        try:
            translated_blocks = _translate_srt_blocks(srt_blocks, lang_code)
            translated_content = _rebuild_srt(translated_blocks)

            output_path = subtitle_dir / f"subtitles_{lang_code}.srt"
            output_path.write_text(translated_content, encoding="utf-8")

            result[lang_code] = str(output_path)
            logger.info(f"Translated subtitles to {lang_name}: {output_path}")
            print(f"  Translated to {lang_name} ({lang_code}): {output_path}")

        except Exception as e:
            logger.error(f"Failed to translate to {lang_name} ({lang_code}): {e}")
            print(f"  Translation to {lang_name} ({lang_code}) failed: {e}")

    return result


def _parse_srt_blocks(srt_content: str) -> list[dict]:
    """Parse SRT content into structured blocks.

    Each block contains: index, start_time, end_time, text.

    Args:
        srt_content: Raw SRT file content.

    Returns:
        List of dicts with keys: index, start_ts, end_ts, text.
    """
    blocks = []
    # Split on double newlines (SRT block separator)
    raw_blocks = re.split(r"\n\n+", srt_content.strip())

    for raw_block in raw_blocks:
        lines = raw_block.strip().split("\n")
        if len(lines) < 3:
            continue

        try:
            index = int(lines[0].strip())
        except ValueError:
            continue

        # Parse timestamp line
        timestamp_match = re.match(
            r"(\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2},\d{3})",
            lines[1].strip(),
        )
        if not timestamp_match:
            continue

        start_ts = timestamp_match.group(1)
        end_ts = timestamp_match.group(2)
        text = "\n".join(lines[2:]).strip()

        blocks.append({
            "index": index,
            "start_ts": start_ts,
            "end_ts": end_ts,
            "text": text,
        })

    return blocks


def _translate_srt_blocks(blocks: list[dict], target_lang: str) -> list[dict]:
    """Translate the text portion of SRT blocks to a target language.

    Batches text for efficiency (translates all lines in one call where possible,
    falling back to individual translation on failure).

    Args:
        blocks: Parsed SRT blocks from _parse_srt_blocks.
        target_lang: ISO 639-1 target language code.

    Returns:
        New list of blocks with translated text.
    """
    from deep_translator import GoogleTranslator

    translator = GoogleTranslator(source="en", target=target_lang)

    # Batch translate for efficiency -- GoogleTranslator supports batch mode
    texts = [block["text"] for block in blocks]

    try:
        # deep-translator's translate_batch handles lists
        translated_texts = translator.translate_batch(texts)
    except Exception:
        # Fallback to one-by-one translation
        logger.warning(f"Batch translation failed for {target_lang}, falling back to individual")
        translated_texts = []
        for text in texts:
            try:
                translated_texts.append(translator.translate(text))
            except Exception as e:
                logger.warning(f"Failed to translate chunk: {e}")
                translated_texts.append(text)  # Keep original on failure

    translated_blocks = []
    for i, block in enumerate(blocks):
        translated_text = translated_texts[i] if i < len(translated_texts) else block["text"]
        # GoogleTranslator may return None for empty strings
        if translated_text is None:
            translated_text = block["text"]
        translated_blocks.append({
            "index": block["index"],
            "start_ts": block["start_ts"],
            "end_ts": block["end_ts"],
            "text": translated_text,
        })

    return translated_blocks


def _rebuild_srt(blocks: list[dict]) -> str:
    """Rebuild SRT file content from structured blocks.

    Args:
        blocks: List of dicts with index, start_ts, end_ts, text.

    Returns:
        Properly formatted SRT file content.
    """
    entries = []
    for block in blocks:
        entries.append(
            f"{block['index']}\n{block['start_ts']} --> {block['end_ts']}\n{block['text']}\n"
        )
    return "\n".join(entries)


def upload_captions_to_youtube(
    video_id: str,
    caption_files: dict[str, str],
) -> list:
    """Upload SRT caption files to a YouTube video via the Data API v3.

    Reuses the authentication pattern from youtube_manager.py to get an
    authenticated YouTube service, then inserts captions for each language.

    Note: YouTube Data API captions.insert costs 400 quota units per call.
    With a 10,000 daily quota and 6 languages, this uses 2,400 units per video.

    Args:
        video_id: YouTube video ID (e.g. "dQw4w9WgXcQ").
        caption_files: Dict mapping language code to SRT file path,
            e.g. {"en": "/path/en.srt", "es": "/path/es.srt"}.

    Returns:
        List of dicts with upload results per language, each containing:
        caption_id, language, name, status.
    """
    # Import YouTube auth from the existing manager module
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from scripts.youtube_manager import get_youtube_service

    try:
        from googleapiclient.http import MediaFileUpload
    except ImportError:
        logger.error("google-api-python-client not installed")
        raise

    youtube = get_youtube_service()
    results = []

    for lang_code, srt_path in caption_files.items():
        srt_file = Path(srt_path)
        if not srt_file.exists():
            logger.warning(f"Caption file not found for {lang_code}: {srt_path}")
            continue

        lang_name = LANGUAGE_NAMES.get(lang_code, lang_code)

        try:
            # Build caption resource
            caption_body = {
                "snippet": {
                    "videoId": video_id,
                    "language": lang_code,
                    "name": lang_name,
                    "isDraft": False,
                },
            }

            media = MediaFileUpload(
                str(srt_file),
                mimetype="application/x-subrip",
                resumable=False,
            )

            response = youtube.captions().insert(
                part="snippet",
                body=caption_body,
                media_body=media,
            ).execute()

            caption_id = response.get("id", "unknown")
            results.append({
                "caption_id": caption_id,
                "language": lang_code,
                "name": lang_name,
                "status": "uploaded",
            })

            logger.info(f"Uploaded {lang_name} caption for video {video_id}: {caption_id}")
            print(f"  Uploaded {lang_name} ({lang_code}) caption: {caption_id}")

        except Exception as e:
            results.append({
                "language": lang_code,
                "name": lang_name,
                "status": "failed",
                "error": str(e),
            })
            logger.error(f"Failed to upload {lang_name} caption for {video_id}: {e}")
            print(f"  Failed to upload {lang_name} ({lang_code}) caption: {e}")

    return results


def generate_subtitles_for_story(
    story_script: dict,
    audio_data: dict,
    story_id: int,
    target_languages: list[str] | None = None,
    video_id: str | None = None,
) -> dict:
    """Full subtitle pipeline: generate SRT, translate, optionally upload.

    Convenience function that chains generate_srt -> translate_subtitles ->
    upload_captions_to_youtube into a single call.

    Args:
        story_script: Story script dict with scenes.
        audio_data: Audio generation result with scene durations.
        story_id: Story identifier.
        target_languages: Languages to translate to (default: all 5 targets).
        video_id: If provided, uploads captions to this YouTube video.

    Returns:
        Dict with srt_path, caption_files, and optionally upload_results.
    """
    if target_languages is None:
        target_languages = DEFAULT_TARGET_LANGUAGES

    # Step 1: Generate English SRT
    print("  Generating English subtitles...")
    srt_path = generate_srt(story_script, audio_data, story_id)

    # Step 2: Translate to target languages
    print(f"  Translating to {len(target_languages)} languages...")
    caption_files = translate_subtitles(srt_path, target_languages)

    result = {
        "story_id": story_id,
        "srt_path": srt_path,
        "caption_files": caption_files,
        "languages": list(caption_files.keys()),
    }

    # Step 3: Upload to YouTube if video_id provided
    if video_id:
        print(f"  Uploading captions to YouTube video {video_id}...")
        upload_results = upload_captions_to_youtube(video_id, caption_files)
        result["upload_results"] = upload_results

    return result


if __name__ == "__main__":
    """Test block: generate a sample SRT from a mock story script."""

    # Configure basic logging for test
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    # Sample story script (mimics the structure from story_generator.py)
    test_story_script = {
        "title": "The Kind Little Elephant",
        "moral": "Kindness always comes back to you",
        "collection": "animal_tales",
        "region": "en-US",
        "scenes": [
            {
                "scene_number": 1,
                "narration": (
                    "Once upon a time, in a lush green jungle, there lived a "
                    "little elephant named Ellie. She was the kindest animal "
                    "in the whole forest."
                ),
                "visual_description": "A cute baby elephant in a green jungle",
            },
            {
                "scene_number": 2,
                "narration": (
                    "One sunny morning, Ellie found a tiny bird with a broken "
                    "wing sitting by the river. Oh no, are you hurt little bird? "
                    "Ellie asked gently."
                ),
                "visual_description": "Elephant finding an injured bird by a river",
            },
            {
                "scene_number": 3,
                "narration": (
                    "Ellie carefully picked up the bird and carried it to the "
                    "wise old owl who knew how to help. The owl bandaged the "
                    "bird's wing with soft leaves."
                ),
                "visual_description": "Elephant carrying bird to a wise owl",
            },
            {
                "scene_number": 4,
                "narration": (
                    "Days later, when Ellie got stuck in the mud, guess who "
                    "came to help? The little bird called all her friends, "
                    "and together they pulled Ellie out!"
                ),
                "visual_description": "Birds helping elephant out of mud",
            },
            {
                "scene_number": 5,
                "narration": (
                    "And so Ellie learned that when you are kind to others, "
                    "kindness always finds its way back to you. The end!"
                ),
                "visual_description": "Happy elephant and bird friends together",
            },
        ],
    }

    # Sample audio_data (mimics the structure from tts_generator.generate_audio_sync)
    test_audio_data = {
        "story_id": 999,
        "scenes": [
            {"scene_number": 1, "duration_seconds": 8.5, "audio_path": "/tmp/s1.wav", "character": "narrator"},
            {"scene_number": 2, "duration_seconds": 10.2, "audio_path": "/tmp/s2.wav", "character": "narrator"},
            {"scene_number": 3, "duration_seconds": 9.8, "audio_path": "/tmp/s3.wav", "character": "narrator"},
            {"scene_number": 4, "duration_seconds": 9.0, "audio_path": "/tmp/s4.wav", "character": "narrator"},
            {"scene_number": 5, "duration_seconds": 7.5, "audio_path": "/tmp/s5.wav", "character": "narrator"},
        ],
        "combined_audio": "/tmp/full_story.m4a",
        "total_duration": 45.0,
        "audio_dir": "/tmp/audio",
    }

    test_story_id = 999

    print("=" * 60)
    print("Subtitle Generator - Test Run")
    print("=" * 60)

    # Step 1: Generate English SRT
    print("\n--- Step 1: Generate English SRT ---")
    srt_path = generate_srt(test_story_script, test_audio_data, test_story_id)

    # Display the generated SRT
    print(f"\nGenerated SRT file: {srt_path}")
    print("-" * 40)
    srt_content = Path(srt_path).read_text(encoding="utf-8")
    print(srt_content)

    # Step 2: Translate to target languages
    print("\n--- Step 2: Translate to multiple languages ---")
    caption_files = translate_subtitles(srt_path, target_languages=["es", "fr"])

    print(f"\nGenerated caption files:")
    for lang, path in caption_files.items():
        lang_name = LANGUAGE_NAMES.get(lang, lang)
        print(f"  {lang_name} ({lang}): {path}")

    # Display a translated SRT (if available)
    if "es" in caption_files:
        print("\n--- Spanish SRT Preview ---")
        es_content = Path(caption_files["es"]).read_text(encoding="utf-8")
        # Show first 5 entries
        es_blocks = es_content.strip().split("\n\n")
        for block in es_blocks[:5]:
            print(block)
            print()

    print("\n" + "=" * 60)
    print("Test complete!")
    print(f"Caption files: {json.dumps(caption_files, indent=2)}")
    print("=" * 60)
