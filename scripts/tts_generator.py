"""
TTS Generator - Creates voiceover using Edge TTS.

Uses Microsoft Edge TTS (free, unlimited) with mature, engaging adult voices
for storytelling. Optional FFmpeg pitch shifting for character differentiation.
"""

import asyncio
import json
import os
import subprocess
import tempfile
from pathlib import Path

import edge_tts
from dotenv import load_dotenv

load_dotenv()

CONFIG_DIR = Path("./config")
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "./output"))


def load_voice_config() -> dict:
    """Load voice configuration for all regions."""
    with open(CONFIG_DIR / "voices.json") as f:
        return json.load(f)


def get_voice_for_character(region: str, character: str = None) -> dict:
    """Get the appropriate voice settings for a character in a region."""
    config = load_voice_config()
    region_config = config["regions"].get(region, config["regions"]["en-US"])

    if character is None or character == "narrator":
        return {
            "voice": region_config["narrator"]["voice"],
            "rate": region_config["narrator"]["rate"],
            "pitch": region_config["narrator"]["pitch"],
            "ffmpeg_pitch": 1.0,
        }

    for char in region_config.get("characters", []):
        if char["name"] == character:
            return {
                "voice": char["voice"],
                "rate": char["rate"],
                "pitch": char["pitch"],
                "ffmpeg_pitch": char.get("ffmpeg_pitch", 1.0),
            }

    # Fallback to narrator
    return {
        "voice": region_config["narrator"]["voice"],
        "rate": region_config["narrator"]["rate"],
        "pitch": region_config["narrator"]["pitch"],
        "ffmpeg_pitch": 1.0,
    }


async def generate_speech_segment(
    text: str,
    output_path: str,
    voice: str = "en-US-AriaNeural",
    rate: str = "+0%",
    pitch: str = "+0%",
) -> str:
    """Generate a single speech segment using Edge TTS."""
    communicate = edge_tts.Communicate(
        text=text,
        voice=voice,
        rate=rate,
        pitch=pitch,
    )
    await communicate.save(output_path)
    return output_path


def apply_cartoon_effect(input_path: str, output_path: str, pitch_factor: float = 1.0) -> str:
    """Apply cartoon-like pitch shifting using FFmpeg.

    Uses WAV output since this FFmpeg build lacks libmp3lame.
    """
    if abs(pitch_factor - 1.0) < 0.01:
        # No pitch change needed - convert to WAV (no MP3 encoder available)
        subprocess.run(
            ["ffmpeg", "-y", "-i", input_path, "-c:a", "pcm_s16le", "-ar", "24000", output_path],
            check=True,
            capture_output=True,
        )
        return output_path

    # Use FFmpeg asetrate + aresample for pitch shifting without speed change
    sample_rate = 24000
    new_rate = int(sample_rate * pitch_factor)
    tempo = 1.0 / pitch_factor

    # Clamp tempo to FFmpeg's valid range (0.5 - 2.0)
    tempo_filters = []
    remaining_tempo = tempo
    while remaining_tempo > 2.0:
        tempo_filters.append("atempo=2.0")
        remaining_tempo /= 2.0
    while remaining_tempo < 0.5:
        tempo_filters.append("atempo=0.5")
        remaining_tempo /= 0.5
    tempo_filters.append(f"atempo={remaining_tempo:.4f}")

    filter_chain = f"asetrate={new_rate},aresample={sample_rate}," + ",".join(tempo_filters)

    subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", input_path,
            "-af", filter_chain,
            "-ar", str(sample_rate),
            "-c:a", "pcm_s16le",
            output_path,
        ],
        check=True,
        capture_output=True,
    )
    return output_path


async def generate_scene_audio(
    scene: dict,
    region: str,
    scene_output_dir: Path,
) -> dict:
    """Generate audio for a single scene with cartoon voice effects."""
    scene_num = scene["scene_number"]
    character = scene.get("character_speaking")
    narration = scene.get("narration", scene.get("text", ""))
    if not narration:
        narration = scene.get("visual_description", f"Scene {scene_num}")

    voice_config = get_voice_for_character(region, character)

    # Generate raw TTS (Edge TTS outputs MP3)
    raw_path = str(scene_output_dir / f"scene_{scene_num:02d}_raw.mp3")
    await generate_speech_segment(
        text=narration,
        output_path=raw_path,
        voice=voice_config["voice"],
        rate=voice_config["rate"],
        pitch=voice_config["pitch"],
    )

    # Apply cartoon pitch effect (output WAV since no libmp3lame)
    final_path = str(scene_output_dir / f"scene_{scene_num:02d}.wav")
    apply_cartoon_effect(
        raw_path,
        final_path,
        pitch_factor=voice_config["ffmpeg_pitch"],
    )

    # Get duration of generated audio
    duration = get_audio_duration(final_path)

    # Clean up raw MP3 file
    if os.path.exists(raw_path):
        os.remove(raw_path)

    return {
        "scene_number": scene_num,
        "audio_path": final_path,
        "duration_seconds": duration,
        "voice_used": voice_config["voice"],
        "character": character or "narrator",
    }


def get_audio_duration(audio_path: str) -> float:
    """Get duration of an audio file using FFprobe."""
    result = subprocess.run(
        [
            "ffprobe",
            "-v", "quiet",
            "-show_entries", "format=duration",
            "-of", "csv=p=0",
            audio_path,
        ],
        capture_output=True,
        text=True,
    )
    try:
        return float(result.stdout.strip())
    except (ValueError, AttributeError):
        return 15.0  # Default estimate


def concatenate_audio(audio_files: list, output_path: str, pause_between: float = 0.3) -> str:
    """Concatenate multiple audio files using FFmpeg concat demuxer."""
    if not audio_files:
        raise ValueError("No audio files to concatenate")

    if len(audio_files) == 1:
        subprocess.run(
            ["ffmpeg", "-y", "-i", audio_files[0], "-c:a", "aac", "-ar", "24000", output_path],
            check=True, capture_output=True,
        )
        return output_path

    # Use FFmpeg concat demuxer - all inputs are WAV so format is consistent
    concat_list_path = str(Path(output_path).parent / "concat_list.txt")
    with open(concat_list_path, "w") as f:
        for audio_file in audio_files:
            f.write(f"file '{os.path.abspath(audio_file)}'\n")

    subprocess.run(
        [
            "ffmpeg", "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", concat_list_path,
            "-ar", "24000",
            "-ac", "1",
            "-c:a", "aac",
            output_path,
        ],
        check=True,
        capture_output=True,
    )

    os.remove(concat_list_path)
    return output_path


async def generate_story_audio(story_script: dict, story_id: int) -> dict:
    """Generate complete audio for a story with all scenes."""
    region = story_script.get("region", "en-US")
    scenes = story_script.get("scenes", [])

    # Create output directory for this story
    audio_dir = OUTPUT_DIR / "audio" / f"story_{story_id}"
    audio_dir.mkdir(parents=True, exist_ok=True)

    # Generate audio for each scene
    scene_results = []
    for scene in scenes:
        result = await generate_scene_audio(scene, region, audio_dir)
        scene_results.append(result)
        print(f"  Generated audio for scene {result['scene_number']} "
              f"({result['duration_seconds']:.1f}s, voice: {result['character']})")

    # Concatenate all scene audio into one file
    audio_files = [r["audio_path"] for r in scene_results]
    combined_path = str(audio_dir / "full_story.m4a")
    concatenate_audio(audio_files, combined_path)

    total_duration = get_audio_duration(combined_path)
    print(f"  Total story audio: {total_duration:.1f}s")

    return {
        "story_id": story_id,
        "scenes": scene_results,
        "combined_audio": combined_path,
        "total_duration": total_duration,
        "audio_dir": str(audio_dir),
    }


def generate_audio_sync(story_script: dict, story_id: int) -> dict:
    """Synchronous wrapper for generate_story_audio."""
    return asyncio.run(generate_story_audio(story_script, story_id))


if __name__ == "__main__":
    import sys

    # Test with a simple story
    test_story = {
        "region": "en-US",
        "scenes": [
            {
                "scene_number": 1,
                "narration": "Once upon a time, in a beautiful green forest, there lived a little rabbit named Rosie.",
                "character_speaking": None,
            },
            {
                "scene_number": 2,
                "narration": "Oh no! I lost my way home! Can someone help me?",
                "character_speaking": "young_hero",
            },
            {
                "scene_number": 3,
                "narration": "Don't worry little one. Follow the stars, and they will guide you home.",
                "character_speaking": "wise_elder",
            },
        ],
    }

    result = generate_audio_sync(test_story, story_id=0)
    print(json.dumps({k: v for k, v in result.items() if k != "scenes"}, indent=2))
