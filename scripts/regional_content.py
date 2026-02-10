"""
Regional Content - Translates and localizes content for multi-language channels.

Takes an English story/lesson script and creates localized versions by:
1. Translating narration text to the target language
2. Selecting the appropriate Edge TTS voice for the language
3. Keeping visual descriptions in English (SDXL Turbo works best with English prompts)
4. Updating metadata (title, description, tags) in the target language

Uses deep-translator (GoogleTranslator) for translations.
"""

import json
import logging
import os
import sqlite3
from copy import deepcopy
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

DATA_DIR = Path(os.getenv("DATA_DIR", "./data"))
CONFIG_DIR = Path("./config")

logger = logging.getLogger(__name__)


def load_region_config() -> dict:
    """Load regional voice and translation settings."""
    config_path = CONFIG_DIR / "channels.json"
    with open(config_path) as f:
        data = json.load(f)
    return data.get("regions", {})


def translate_text(text: str, target_lang: str) -> str:
    """Translate text to target language using deep-translator."""
    if not text or not text.strip():
        return text

    try:
        from deep_translator import GoogleTranslator
        translated = GoogleTranslator(source="en", target=target_lang).translate(text)
        return translated if translated else text
    except Exception as e:
        logger.warning(f"Translation failed for '{text[:50]}...' to {target_lang}: {e}")
        return text


def translate_script(script: dict, target_lang: str) -> dict:
    """Translate an entire story/lesson script to a target language.

    Translates narration, title, moral, description, and tags.
    Keeps visual_description and image_search_terms in English (for SDXL Turbo).
    """
    regions = load_region_config()
    region_info = regions.get(target_lang)

    if not region_info or not region_info.get("translate", True):
        return script

    translated = deepcopy(script)

    # Translate top-level fields
    for field in ["title", "moral", "description"]:
        if field in translated and translated[field]:
            translated[field] = translate_text(translated[field], target_lang)

    # Translate display_title if present
    if "display_title" in translated:
        translated["display_title"] = translate_text(translated["display_title"], target_lang)

    # Translate scenes (narration only, keep visual descriptions in English)
    for scene in translated.get("scenes", []):
        if "narration" in scene:
            scene["narration"] = translate_text(scene["narration"], target_lang)
        # Keep visual_description in English for image generation
        # Keep image_search_terms in English for image generation

    # Translate tags
    if "tags" in translated:
        translated["tags"] = [translate_text(tag, target_lang) for tag in translated["tags"]]

    # Add language metadata
    translated["language"] = target_lang
    translated["tts_voice"] = region_info.get("tts_voice", "en-US-AriaNeural")
    translated["original_language"] = "en"

    return translated


def get_regional_tts_voice(language: str) -> str:
    """Get the Edge TTS voice for a language."""
    regions = load_region_config()
    region_info = regions.get(language, {})
    return region_info.get("tts_voice", "en-US-AriaNeural")


def localize_title(title: str, language: str) -> str:
    """Translate and format a video title for a regional channel."""
    if language == "en":
        return title
    return translate_text(title, language)


def localize_description(description: str, language: str, channel_name: str = "") -> str:
    """Translate and format a video description for a regional channel."""
    if language == "en":
        return description

    translated = translate_text(description, language)

    # Add bilingual channel branding
    regions = load_region_config()
    lang_name = regions.get(language, {}).get("language_name", language)

    footer = f"\n\n---\n{channel_name}\n#{lang_name}ForKids #KidsEducation #LittleWisdomTales"
    return translated + footer


def localize_tags(tags: list, language: str) -> list:
    """Translate and add regional tags."""
    if language == "en":
        return tags

    regions = load_region_config()
    lang_name = regions.get(language, {}).get("language_name", language)

    translated_tags = [translate_text(tag, language) for tag in tags]

    # Add regional discovery tags
    regional_tags = {
        "hi": ["बच्चों की कहानियां", "नैतिक कहानियां", "हिंदी कहानियां", "kids stories hindi"],
        "es": ["cuentos para niños", "historias morales", "cuentos infantiles", "kids stories spanish"],
        "fr": ["histoires pour enfants", "contes moraux", "histoires animées", "kids stories french"],
        "pt": ["histórias para crianças", "contos morais", "histórias animadas", "kids stories portuguese"],
        "ar": ["قصص أطفال", "قصص أخلاقية", "قصص عربية", "kids stories arabic"],
    }

    extra = regional_tags.get(language, [])
    return list(set(translated_tags + extra))[:30]


def create_regional_version(
    script: dict,
    target_lang: str,
    content_type: str = "story",
) -> dict:
    """Create a complete regional version of a script ready for pipeline processing.

    Returns a new script dict that can be passed directly to the pipeline
    (TTS → images → video → upload) for the target language.
    """
    translated = translate_script(script, target_lang)

    # Store metadata for pipeline
    translated["_regional"] = True
    translated["_source_language"] = "en"
    translated["_target_language"] = target_lang
    translated["_content_type"] = content_type

    # Save to database
    db_path = DATA_DIR / "stories.db"
    conn = sqlite3.connect(str(db_path))

    # Check if stories table has language column, add if missing
    columns = [row[1] for row in conn.execute("PRAGMA table_info(stories)").fetchall()]
    if "language" not in columns:
        conn.execute("ALTER TABLE stories ADD COLUMN language TEXT DEFAULT 'en'")
        conn.commit()
    if "content_type" not in columns:
        conn.execute("ALTER TABLE stories ADD COLUMN content_type TEXT DEFAULT 'story'")
        conn.commit()

    conn.execute(
        """INSERT INTO stories (title, collection, region, moral, script, status, language, content_type)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            translated.get("title", "Untitled"),
            translated.get("collection", ""),
            target_lang,
            translated.get("moral", ""),
            json.dumps(translated),
            "generated",
            target_lang,
            content_type,
        ),
    )
    conn.commit()
    story_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    translated["story_id"] = story_id
    conn.close()

    logger.info(f"Created regional version ({target_lang}): {translated.get('title', '')[:60]}")
    return translated


def get_languages_for_channel(channel_key: str) -> str:
    """Get the language for a specific channel."""
    config_path = CONFIG_DIR / "channels.json"
    with open(config_path) as f:
        config = json.load(f)

    channel = config["channels"].get(channel_key, {})
    return channel.get("language", "en")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        # Quick test: translate a sample text
        lang = sys.argv[1]
        test = "Once upon a time, a clever monkey lived in a big tree by the river."
        print(f"Original: {test}")
        print(f"Translated ({lang}): {translate_text(test, lang)}")
    else:
        print("Usage: python regional_content.py <language_code>")
        print("Supported: hi, es, fr, pt, ar")
