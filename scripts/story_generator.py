"""
Story Generator - Creates kid-friendly moral stories using LLM.

Uses Ollama (local, free) or Google Gemini (free tier) to generate
structured story scripts with scene breakdowns for video production.
"""

import json
import os
import random
import sqlite3
from datetime import datetime
from pathlib import Path

import httpx
from dotenv import load_dotenv

load_dotenv()

DATA_DIR = Path(os.getenv("DATA_DIR", "./data"))
CONFIG_DIR = Path("./config")


def get_db():
    """Get or create the stories database."""
    db_path = DATA_DIR / "stories.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("""
        CREATE TABLE IF NOT EXISTS stories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            collection TEXT,
            region TEXT,
            moral TEXT,
            script JSON,
            status TEXT DEFAULT 'generated',
            video_id TEXT,
            views INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            published_at TIMESTAMP
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS analytics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            story_id INTEGER,
            date TEXT,
            views INTEGER DEFAULT 0,
            likes INTEGER DEFAULT 0,
            watch_time_seconds REAL DEFAULT 0,
            FOREIGN KEY (story_id) REFERENCES stories(id)
        )
    """)
    conn.commit()
    return conn


def load_story_themes():
    """Load story themes from configuration."""
    with open(CONFIG_DIR / "story_themes.json") as f:
        return json.load(f)


def build_story_prompt(collection_name: str, story_hint: str, moral: str, region: str) -> str:
    """Build the LLM prompt for story generation."""
    return f"""You are a children's storyteller creating content for a YouTube kids channel.

Generate a kid-friendly story based on these parameters:
- Collection: {collection_name}
- Story idea: {story_hint}
- Moral lesson: {moral}
- Target region: {region}
- Target age: 4-10 years old

IMPORTANT RULES:
- WRITE THE ENTIRE STORY IN ENGLISH ONLY - all narration, dialogue, title, moral must be in English
- Make the story EXCITING, MAGICAL, and FULL OF WONDER - kids should be glued to the screen!
- Use expressive, animated narration with sound effects in words like "WHOOSH!", "SPLASH!", "WOW!"
- Include suspense, surprises, and a twist before the moral
- Keep language simple but engaging with vivid, colorful descriptions
- No violence, scary content, or mature themes
- Story should be 2-3 minutes when narrated (about 400-500 words)
- End with a clear, positive moral lesson that kids can remember
- Each scene needs a DISTINCT, COLORFUL visual with different settings/characters/objects
- Include dialogue between characters to make it lively (use character_speaking field)

Return a JSON object with this EXACT structure (no markdown, just raw JSON):
{{
  "title": "Story Title",
  "moral": "The moral lesson in one sentence",
  "duration_estimate": "2-3 minutes",
  "thumbnail_description": "A vivid description for the thumbnail image",
  "scenes": [
    {{
      "scene_number": 1,
      "narration": "The narrator's text for this scene",
      "visual_description": "What should be shown: e.g. 'A bright sunny forest with a small rabbit sitting under a big oak tree'",
      "image_search_terms": "cartoon forest rabbit oak tree sunny illustration kids",
      "character_speaking": null,
      "duration_seconds": 15
    }},
    {{
      "scene_number": 2,
      "narration": "Next part of the story...",
      "visual_description": "Description of the visual...",
      "image_search_terms": "search terms for finding an image",
      "character_speaking": "young_hero",
      "duration_seconds": 15
    }}
  ],
  "tags": ["kids story", "moral story", "bedtime story", "{moral}", "{collection_name}"],
  "description": "A YouTube-friendly description of this video (2-3 sentences)"
}}

Create 10-15 scenes. Each scene should be 10-15 seconds of narration. Use SPECIFIC image_search_terms (3-5 keywords) that will find relevant cartoon illustrations.
Return ONLY the JSON, no other text."""


def call_ollama(prompt: str, max_retries: int = 3) -> str:
    """Call local Ollama LLM with retry logic."""
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    model = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
    import time as _time

    for attempt in range(max_retries):
        try:
            response = httpx.post(
                f"{base_url}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "format": "json",
                    "options": {
                        "temperature": 0.7,
                        "num_predict": 3000,
                    }
                },
                timeout=300.0,
            )
            response.raise_for_status()
            return response.json()["response"]
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"Ollama attempt {attempt+1} failed: {e}, retrying in 10s...")
                _time.sleep(10)
            else:
                raise


def call_gemini(prompt: str) -> str:
    """Call Google Gemini free tier API."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not set")

    model = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
    response = httpx.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        params={"key": api_key},
        json={
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.8,
                "maxOutputTokens": 2048,
            }
        },
        timeout=60.0,
    )
    response.raise_for_status()
    candidates = response.json()["candidates"]
    return candidates[0]["content"]["parts"][0]["text"]


def build_shorts_prompt(collection_name: str, story_hint: str, moral: str, region: str) -> str:
    """Build a prompt for YouTube Shorts (under 50 seconds)."""
    return f"""You are a children's storyteller creating a VERY SHORT story for YouTube Shorts (under 50 seconds).

Generate a kid-friendly mini-story based on these parameters:
- Collection: {collection_name}
- Story idea: {story_hint}
- Moral lesson: {moral}
- Target age: 4-10 years old

IMPORTANT RULES:
- WRITE THE ENTIRE STORY IN ENGLISH ONLY - all narration, dialogue, title, moral must be in English
- Keep the story VERY SHORT but EXCITING - only 80-120 words total
- Start with a HOOK that grabs attention instantly ("One day, something AMAZING happened!")
- Make it punchy, fun, and surprising with a quick twist
- Simple, age-appropriate language with expressive words
- No violence, scary content, or mature themes
- End with a clear, memorable moral lesson
- Only 4-5 scenes, each 8-12 seconds of narration

Return a JSON object with this EXACT structure (no markdown, just raw JSON):
{{
  "title": "Story Title",
  "moral": "The moral lesson in one sentence",
  "duration_estimate": "45-55 seconds",
  "thumbnail_description": "A vivid description for the thumbnail image",
  "scenes": [
    {{
      "scene_number": 1,
      "narration": "Short opening narration (15-25 words)",
      "visual_description": "What should be shown visually",
      "image_search_terms": "cartoon illustration kids colorful search terms",
      "character_speaking": null,
      "duration_seconds": 12
    }},
    {{
      "scene_number": 2,
      "narration": "Middle part (15-25 words)",
      "visual_description": "Visual description",
      "image_search_terms": "search terms for image",
      "character_speaking": null,
      "duration_seconds": 12
    }},
    {{
      "scene_number": 3,
      "narration": "Conclusion with moral (15-25 words)",
      "visual_description": "Visual description",
      "image_search_terms": "search terms for image",
      "character_speaking": null,
      "duration_seconds": 12
    }}
  ],
  "tags": ["kids story", "moral story", "shorts", "{moral}", "{collection_name}"],
  "description": "A short description of this story (1-2 sentences)"
}}

Create ONLY 3-4 scenes. Keep total narration under 120 words.
Return ONLY the JSON, no other text."""


def generate_story_script(collection: str = None, story_hint: str = None, moral: str = None, for_shorts: bool = False) -> dict:
    """Generate a complete story script ready for video production."""
    themes = load_story_themes()
    collections = themes["story_collections"]
    morals = themes["moral_categories"]

    # Pick random collection if not specified
    if not collection:
        collection = random.choice(list(collections.keys()))

    coll_data = collections[collection]

    # Pick a story hint from examples if not provided
    if not story_hint:
        story_hint = random.choice(coll_data["examples"])

    # Pick a moral if not provided
    if not moral:
        moral = random.choice(morals)

    # Always use en-US region for TTS since stories are generated in English
    region = "en-US"
    if for_shorts:
        prompt = build_shorts_prompt(collection, story_hint, moral, coll_data["region"])
    else:
        prompt = build_story_prompt(collection, story_hint, moral, coll_data["region"])

    # Try LLM providers with retry logic for JSON parsing failures
    provider = os.getenv("LLM_PROVIDER", "ollama")
    story_script = None
    last_error = None

    for gen_attempt in range(3):
        raw_response = None
        if provider == "ollama":
            try:
                raw_response = call_ollama(prompt)
            except Exception as e:
                print(f"Ollama failed: {e}, trying Gemini fallback...")
                try:
                    raw_response = call_gemini(prompt)
                except Exception as e2:
                    print(f"Gemini also failed: {e2}")
                    raise
        elif provider == "gemini":
            try:
                raw_response = call_gemini(prompt)
            except Exception as e:
                print(f"Gemini failed: {e}, trying Ollama fallback...")
                try:
                    raw_response = call_ollama(prompt)
                except Exception as e2:
                    print(f"Ollama also failed: {e2}")
                    raise

        try:
            story_script = parse_llm_json(raw_response)
            # Validate required fields
            if "scenes" not in story_script or not story_script["scenes"]:
                raise ValueError("Story has no scenes")
            if "title" not in story_script:
                raise ValueError("Story has no title")
            break
        except (ValueError, KeyError) as e:
            last_error = e
            print(f"JSON parse attempt {gen_attempt+1} failed: {e}, retrying...")
            import time as _time
            _time.sleep(5)

    if story_script is None:
        raise ValueError(f"Failed to generate valid story after 3 attempts: {last_error}")

    # Enrich with metadata
    story_script["collection"] = collection
    story_script["origin"] = coll_data["origin"]
    story_script["region"] = region
    story_script["age_range"] = coll_data["age_range"]
    story_script["generated_at"] = datetime.now().isoformat()

    # Save to database
    db = get_db()
    db.execute(
        """INSERT INTO stories (title, collection, region, moral, script, status)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (
            story_script.get("title", "Untitled"),
            collection,
            region,
            story_script.get("moral", moral),
            json.dumps(story_script),
            "generated",
        ),
    )
    db.commit()
    story_id = db.execute("SELECT last_insert_rowid()").fetchone()[0]
    story_script["story_id"] = story_id
    db.close()

    return story_script


def parse_llm_json(text: str) -> dict:
    """Extract JSON from LLM response, handling common formatting issues."""
    import re

    text = text.strip()

    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try to find JSON block in markdown code fence
    if "```json" in text:
        start = text.index("```json") + 7
        end = text.index("```", start)
        try:
            return json.loads(text[start:end].strip())
        except json.JSONDecodeError:
            pass

    if "```" in text:
        start = text.index("```") + 3
        end = text.index("```", start)
        try:
            return json.loads(text[start:end].strip())
        except json.JSONDecodeError:
            pass

    # Try to find JSON object boundaries
    first_brace = text.find("{")
    last_brace = text.rfind("}")
    if first_brace != -1 and last_brace != -1:
        json_str = text[first_brace:last_brace + 1]
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            # Try fixing common LLM JSON issues
            # Remove trailing commas before } or ]
            fixed = re.sub(r',\s*([}\]])', r'\1', json_str)
            # Remove control characters
            fixed = re.sub(r'[\x00-\x1f]', ' ', fixed)
            try:
                return json.loads(fixed)
            except json.JSONDecodeError:
                pass

    raise ValueError(f"Could not parse JSON from LLM response: {text[:300]}...")


def get_unused_stories(limit: int = 5) -> list:
    """Get stories that haven't been made into videos yet."""
    db = get_db()
    rows = db.execute(
        "SELECT * FROM stories WHERE status = 'generated' ORDER BY created_at DESC LIMIT ?",
        (limit,),
    ).fetchall()
    db.close()
    return [dict(row) for row in rows]


def get_top_performing_themes(limit: int = 5) -> list:
    """Analyze which story themes/morals perform best."""
    db = get_db()
    rows = db.execute("""
        SELECT s.collection, s.moral,
               AVG(a.views) as avg_views,
               AVG(a.watch_time_seconds) as avg_watch_time,
               COUNT(*) as story_count
        FROM stories s
        JOIN analytics a ON s.id = a.story_id
        WHERE s.status = 'published'
        GROUP BY s.collection, s.moral
        ORDER BY avg_views DESC
        LIMIT ?
    """, (limit,)).fetchall()
    db.close()
    return [dict(row) for row in rows]


def pick_smart_story_params() -> dict:
    """Use analytics to pick optimal story parameters."""
    top_themes = get_top_performing_themes()

    if top_themes:
        # Weight towards top-performing themes but keep variety
        weights = [row["avg_views"] + 1 for row in top_themes]
        chosen = random.choices(top_themes, weights=weights, k=1)[0]
        return {
            "collection": chosen["collection"],
            "moral": chosen["moral"],
        }

    # No analytics yet - pick randomly
    return {}


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--smart":
        params = pick_smart_story_params()
        story = generate_story_script(**params)
    else:
        story = generate_story_script()

    print(json.dumps(story, indent=2))
