"""
Crafts & Skills Generator - Creates kid-friendly hands-on skill lesson scripts using LLM.

Generates structured lesson scripts for kids (ages 6-14) covering real-world
skills like carpentry, plumbing, and electrical concepts. Each lesson features
"Handy the Helper" (a friendly cartoon toolbelt character) as the recurring
host character.

Lessons are generated in the same JSON scene format as story scripts,
making them fully compatible with the existing video pipeline
(TTS -> AI Images -> FFmpeg Video -> YouTube Upload).

Categories:
  - carpentry_building: Woodworking, building, tools, furniture, workshop safety
  - plumbing_home: Plumbing, painting, home repair, water systems, maintenance
  - electrical_tech: Circuits, wiring basics, safety, solar, how things work

Host Character: "Handy the Helper"
  A friendly cartoon toolbelt character with big expressive eyes, a yellow
  hard hat, and colorful tools hanging from their belt. Always wears safety
  goggles and has a warm encouraging smile.
"""

import json
import os
import random
import sqlite3
import time as _time
from datetime import datetime
from pathlib import Path

import httpx
from dotenv import load_dotenv

load_dotenv()

DATA_DIR = Path(os.getenv("DATA_DIR", "./data"))
CONFIG_DIR = Path("./config")

# ---------------------------------------------------------------------------
# Difficulty mapping for age-appropriate language
# ---------------------------------------------------------------------------
DIFFICULTY_MAP = {
    "beginner": {
        "age_range": "6-8",
        "complexity": "very simple",
        "vocab": "short sentences with common everyday words",
    },
    "intermediate": {
        "age_range": "9-11",
        "complexity": "moderate",
        "vocab": "clear sentences with some new technical terms explained simply",
    },
    "advanced": {
        "age_range": "12-14",
        "complexity": "upper intermediate",
        "vocab": "richer vocabulary with subject-specific terminology and context",
    },
}

# ---------------------------------------------------------------------------
# Visual themes per category - consistent color palettes and props
# ---------------------------------------------------------------------------
CATEGORY_VISUAL_THEMES = {
    "carpentry_building": {
        "colors": "warm wood browns, golden yellow, and forest green",
        "props": "cartoon hammer, colorful nails, wood planks, measuring tape, pencil, square ruler, sawdust particles",
        "setting": "a cheerful cartoon workshop with wooden workbenches, hanging tools on pegboard walls, and sawdust on the floor",
    },
    "plumbing_home": {
        "colors": "ocean blue, teal green, and crisp white",
        "props": "cartoon pipes, colorful faucets, water droplets, wrenches, paint rollers, buckets, plunger",
        "setting": "a bright cartoon house cutaway showing colorful pipes running through walls and under floors",
    },
    "electrical_tech": {
        "colors": "electric yellow, vibrant orange, and safety red",
        "props": "cartoon batteries, glowing light bulbs, colorful wires, switches, solar panels, circuit boards, spark effects",
        "setting": "a bright cartoon laboratory with glowing circuits on the walls, a big battery display, and sparkling electricity effects",
    },
}

# ---------------------------------------------------------------------------
# Handy the Helper's signature phrases (for variety in prompts)
# ---------------------------------------------------------------------------
HANDY_PHRASES = [
    "Hey builders! Ready to learn something AWESOME today?",
    "Safety goggles ON! Let's get to work!",
    "Grab your toolbelt, because today's lesson is going to be GREAT!",
    "Hi friends! Handy here, and BOY do I have something cool to show you!",
    "Hard hat? Check! Goggles? Check! Curiosity? DOUBLE CHECK!",
    "Welcome back to the workshop, my favorite builders!",
    "You know what time it is? It's BUILD TIME!",
    "Hey there, future engineers! Let's make something amazing!",
]

# ---------------------------------------------------------------------------
# Safety scene templates for end-of-lesson safety reminders
# ---------------------------------------------------------------------------
SAFETY_SCENE_TEMPLATES = [
    "And remember friends, safety ALWAYS comes first! {safety_note} That's Handy's number one rule!",
    "Before we go, let's do our safety check! {safety_note} Stay safe and keep building!",
    "Handy's Safety Reminder: {safety_note} A good builder is a SAFE builder!",
    "Time for our safety pledge! {safety_note} Promise me you'll always be careful, okay?",
]

# ---------------------------------------------------------------------------
# Quiz/challenge templates for interactive scenes
# ---------------------------------------------------------------------------
QUIZ_TEMPLATES = {
    "identify": "Can you name this tool? Look closely at the screen! ... That's right, it's a {answer}!",
    "safety": "Quick safety quiz! What should you ALWAYS wear when {situation}? ... Yes! {answer}!",
    "recall": "Do you remember what we just learned? {question} ... Exactly! {answer}!",
    "challenge": "Here's a fun challenge! {challenge} Think about it... {hint} ... The answer is {answer}!",
    "interactive": "Let's try this together! {instruction} ... Amazing work, builder!",
}


# ===========================================================================
# Database Functions
# ===========================================================================


def get_db():
    """Get or create the stories database with crafts_skills_lessons table.

    Ensures both the main ``stories`` table (for pipeline compatibility)
    and the ``crafts_skills_lessons`` table (for lesson-specific tracking)
    exist in the SQLite database.

    Returns:
        sqlite3.Connection with row_factory set to sqlite3.Row.
    """
    db_path = DATA_DIR / "stories.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    # Ensure the main stories table exists (same schema as story_generator.py)
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

    # Create crafts_skills_lessons table for tracking lesson-specific metadata
    conn.execute("""
        CREATE TABLE IF NOT EXISTS crafts_skills_lessons (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            category TEXT NOT NULL,
            topic TEXT NOT NULL,
            lesson_title TEXT NOT NULL,
            difficulty TEXT DEFAULT 'beginner',
            safety_note TEXT,
            script JSON,
            story_id INTEGER,
            status TEXT DEFAULT 'generated',
            video_id TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (story_id) REFERENCES stories(id)
        )
    """)

    conn.commit()
    return conn


# ===========================================================================
# Curriculum Loading & Helpers
# ===========================================================================


def load_crafts_curriculum() -> dict:
    """Load the crafts & skills curriculum configuration from JSON.

    Returns:
        Parsed curriculum dictionary.

    Raises:
        FileNotFoundError: If config/crafts_skills_curriculum.json is missing.
    """
    curriculum_path = CONFIG_DIR / "crafts_skills_curriculum.json"
    if not curriculum_path.exists():
        raise FileNotFoundError(
            f"Crafts & skills curriculum not found at {curriculum_path}. "
            "Please create config/crafts_skills_curriculum.json first."
        )
    with open(curriculum_path) as f:
        return json.load(f)


def _get_difficulty_context(difficulty: str) -> dict:
    """Get age-appropriate language context for a difficulty level.

    Args:
        difficulty: One of 'beginner', 'intermediate', 'advanced'.

    Returns:
        Dict with keys: age_range, complexity, vocab.
    """
    return DIFFICULTY_MAP.get(difficulty, DIFFICULTY_MAP["beginner"])


def _get_visual_theme(category: str) -> dict:
    """Get the visual theme configuration for a category.

    Args:
        category: One of 'carpentry_building', 'plumbing_home', 'electrical_tech'.

    Returns:
        Dict with keys: colors, props, setting.
    """
    return CATEGORY_VISUAL_THEMES.get(
        category, CATEGORY_VISUAL_THEMES["carpentry_building"]
    )


# ===========================================================================
# LLM Interaction
# ===========================================================================


def call_ollama(prompt: str, max_retries: int = 3) -> str:
    """Call local Ollama LLM with retry logic.

    Uses the same Ollama instance and configuration as the rest of the
    pipeline (localhost:11434, llama3.2:3b by default).

    Args:
        prompt: The prompt to send to the LLM.
        max_retries: Number of retry attempts on failure.

    Returns:
        Raw text response from the LLM.

    Raises:
        Exception: If all retry attempts fail.
    """
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    model = os.getenv("OLLAMA_MODEL", "llama3.2:3b")

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
                        "num_predict": 4000,
                    },
                },
                timeout=300.0,
            )
            response.raise_for_status()
            return response.json()["response"]
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"Ollama attempt {attempt + 1} failed: {e}, retrying in 10s...")
                _time.sleep(10)
            else:
                raise


def parse_llm_json(text: str) -> dict:
    """Extract JSON from LLM response, handling common formatting issues.

    Mirrors the robust parsing logic from story_generator.py to handle
    markdown fences, trailing commas, and control characters.

    Args:
        text: Raw LLM response text that should contain JSON.

    Returns:
        Parsed dictionary from the JSON content.

    Raises:
        ValueError: If no valid JSON can be extracted from the response.
    """
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
        json_str = text[first_brace : last_brace + 1]
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            # Try fixing common LLM JSON issues
            # Remove trailing commas before } or ]
            fixed = re.sub(r",\s*([}\]])", r"\1", json_str)
            # Remove control characters
            fixed = re.sub(r"[\x00-\x1f]", " ", fixed)
            try:
                return json.loads(fixed)
            except json.JSONDecodeError:
                pass

    raise ValueError(f"Could not parse JSON from LLM response: {text[:300]}...")


# ===========================================================================
# Prompt Building
# ===========================================================================


def build_crafts_lesson_prompt(
    category: str,
    topic: str,
    lesson_title: str,
    lesson_description: str,
    difficulty: str,
    safety_note: str,
) -> str:
    """Build an LLM prompt to generate a crafts/skills lesson script.

    Creates a detailed prompt that instructs the LLM to generate a structured
    JSON lesson featuring Handy the Helper (a cartoon toolbelt character)
    teaching the given topic in an engaging, age-appropriate, safety-first way.

    Args:
        category: Category key (carpentry_building, plumbing_home, electrical_tech).
        topic: Specific topic name from the curriculum.
        lesson_title: Human-readable lesson title.
        lesson_description: Brief description of what the lesson covers.
        difficulty: Difficulty level (beginner, intermediate, advanced).
        safety_note: Mandatory safety reminder for this lesson.

    Returns:
        A formatted prompt string ready for LLM generation.
    """
    curriculum = load_crafts_curriculum()
    metadata = curriculum.get("metadata", {})
    diff_ctx = _get_difficulty_context(difficulty)
    visual_theme = _get_visual_theme(category)

    host_desc = metadata.get(
        "host_description",
        "A friendly cartoon toolbelt character with big expressive eyes, "
        "a yellow hard hat, and colorful tools hanging from their belt. "
        "Always wears safety goggles and has a warm encouraging smile.",
    )

    # Handy's visual keywords for SDXL image generation consistency
    handy_keywords = (
        "cartoon toolbelt character yellow hard hat safety goggles "
        "big eyes friendly smile colorful tools"
    )

    category_display = curriculum["categories"][category]["display_name"]
    opening_phrase = random.choice(HANDY_PHRASES)

    return f"""You are an expert children's educational content creator making a video lesson for a YouTube kids channel called "Little Wisdom Tales".

Generate a hands-on skills lesson script based on these parameters:
- Category: {category_display}
- Topic: {topic}
- Lesson Title: {lesson_title}
- Lesson Description: {lesson_description}
- Difficulty: {difficulty} (ages {diff_ctx['age_range']})
- Language Complexity: {diff_ctx['complexity']} ({diff_ctx['vocab']})
- Safety Note: {safety_note}

MAIN CHARACTER:
- Name: Handy the Helper
- Appearance: {host_desc}
- Handy appears in EVERY scene as the host and guide
- He is enthusiastic, safety-conscious, encouraging, and uses catchy phrases like "{opening_phrase}"
- Handy always demonstrates proper safety techniques before any activity

VISUAL STYLE:
- Setting: {visual_theme['setting']}
- Color palette: {visual_theme['colors']}
- Props and tools: {visual_theme['props']}
- Style: Bright, cartoon, colorful, kid-friendly, educational, SAFETY-FOCUSED
- All tools and activities shown as friendly CARTOON versions (not realistic dangerous tools)
- Every visual_description MUST include Handy the Helper doing something specific
- Include cartoon safety gear (goggles, gloves, hard hat) visible in every workshop scene

LESSON STRUCTURE (6-8 scenes, each ~15 seconds of narration):
1. INTRO (1 scene): Handy welcomes kids, puts on safety gear, introduces today's topic with excitement
2. CONCEPT EXPLANATION (2-3 scenes): Break down the concept into simple steps with cartoon visual metaphors
3. DEMONSTRATION (1-2 scenes): Handy shows how something works using colorful cartoon animations
4. QUIZ/CHALLENGE (1 scene): Interactive moment - ask kids to identify a tool, answer a safety question, or solve a building challenge
5. SAFETY REMINDER (1 scene): Handy reviews the key safety rule for today's topic
6. ENCOURAGEMENT + CTA (1 scene): Handy praises them, teases next lesson, and asks them to subscribe

IMPORTANT RULES:
- WRITE EVERYTHING IN ENGLISH ONLY
- Use {diff_ctx['complexity']} language appropriate for ages {diff_ctx['age_range']}
- SAFETY FIRST: Every lesson must emphasize that kids should ALWAYS have adult supervision
- All tools and activities shown as CARTOON versions - never show realistic dangerous equipment
- Make it FUN and ENGAGING - use sound effects like "BANG BANG!", "WHOOOOSH!", "TA-DA!", "BZZZZ!"
- Include moments where kids participate ("Can you spot the hammer?", "What tool should we use?")
- Use concrete cartoon visual metaphors (animated screws spinning, happy paint drops, dancing wires)
- No scary content, no dangerous instructions that kids could follow unsupervised
- This is CONCEPTUAL education - teaching HOW things work, not step-by-step instructions to build dangerous items
- Each scene MUST have Handy the Helper doing something visible and specific
- Include this safety note naturally in the safety scene: "{safety_note}"

Return a JSON object with this EXACT structure (no markdown, just raw JSON):
{{
  "title": "{lesson_title}",
  "lesson_topic": "{topic}",
  "category": "{category}",
  "difficulty": "{difficulty}",
  "moral": "Safety first, and learning new skills is awesome!",
  "duration_estimate": "2-3 minutes",
  "safety_note": "{safety_note}",
  "thumbnail_description": "Handy the Helper cartoon toolbelt character with yellow hard hat and safety goggles standing next to a colorful display about {topic} with {visual_theme['colors']} background",
  "scenes": [
    {{
      "scene_number": 1,
      "narration": "{opening_phrase} Today we're going to learn something SUPER cool about {topic}!",
      "visual_description": "Handy the Helper, {host_desc}, waving excitedly in {visual_theme['setting']} with a big colorful banner reading the lesson title and cartoon tools floating around",
      "image_search_terms": "{handy_keywords} {visual_theme['colors']} workshop welcome kids cartoon",
      "character_speaking": "handy_the_helper",
      "duration_seconds": 15,
      "scene_type": "intro"
    }},
    {{
      "scene_number": 2,
      "narration": "Concept explanation with examples...",
      "visual_description": "Handy pointing at specific cartoon visual aids related to the topic...",
      "image_search_terms": "relevant search terms for the concept",
      "character_speaking": "handy_the_helper",
      "duration_seconds": 15,
      "scene_type": "explanation"
    }},
    {{
      "scene_number": 3,
      "narration": "A quiz or interactive moment...",
      "visual_description": "Handy holding up a cartoon question card with visual aids...",
      "image_search_terms": "relevant quiz visual terms",
      "character_speaking": "handy_the_helper",
      "duration_seconds": 15,
      "scene_type": "quiz"
    }},
    {{
      "scene_number": 4,
      "narration": "Safety reminder: {safety_note}",
      "visual_description": "Handy wearing full safety gear with a big cartoon safety shield and the safety rule displayed...",
      "image_search_terms": "{handy_keywords} safety gear cartoon shield reminder kids",
      "character_speaking": "handy_the_helper",
      "duration_seconds": 15,
      "scene_type": "safety"
    }},
    {{
      "scene_number": 5,
      "narration": "Great job today builders! Don't forget to subscribe and hit that bell...",
      "visual_description": "Handy waving goodbye with a big subscribe button and cartoon tools...",
      "image_search_terms": "{handy_keywords} subscribe goodbye cartoon workshop kids",
      "character_speaking": "handy_the_helper",
      "duration_seconds": 15,
      "scene_type": "outro"
    }}
  ],
  "tags": ["kids skills", "{category.replace('_', ' ')}", "{topic}", "handy the helper", "kids learning", "safety first", "cartoon workshop", "Little Wisdom Tales"],
  "description": "Join Handy the Helper as he teaches {lesson_title} in this fun skills video for kids! Learn about {topic} with colorful cartoon animations and important safety tips!"
}}

Create 6-8 scenes total. Include at least 1 quiz/challenge scene and 1 dedicated safety scene.
Each scene must have a unique, detailed visual_description with Handy the Helper doing something specific.
The visual descriptions should use cartoon imagery: friendly animated tools, colorful diagrams, happy characters.
All dangerous tools/activities must be shown as CARTOON versions with obvious safety gear visible.
Return ONLY the JSON, no other text."""


# ===========================================================================
# Script Enrichment
# ===========================================================================


def _enrich_crafts_script(
    lesson_script: dict,
    category: str,
    topic: str,
    lesson_title: str,
    lesson_description: str,
    difficulty: str,
    safety_note: str,
) -> dict:
    """Enrich the raw LLM output with metadata needed by the video pipeline.

    Adds fields that the video pipeline expects (collection, region, type, etc.)
    and ensures the lesson script is fully compatible with the story format
    used by video_assembler.py and youtube_manager.py.

    Args:
        lesson_script: The parsed lesson script from the LLM.
        category: Category key from the curriculum.
        topic: Topic name from the curriculum.
        lesson_title: Human-readable lesson title.
        lesson_description: Brief lesson description.
        difficulty: Difficulty level.
        safety_note: Safety reminder for this lesson.

    Returns:
        Enriched lesson script dictionary ready for the video pipeline.
    """
    diff_ctx = _get_difficulty_context(difficulty)
    visual_theme = _get_visual_theme(category)

    # Handy's visual keywords for consistent SDXL generation
    handy_keywords = (
        "cartoon toolbelt character yellow hard hat safety goggles "
        "big eyes friendly smile colorful tools"
    )

    # Ensure required fields exist with sensible defaults
    lesson_script.setdefault("title", lesson_title)
    lesson_script.setdefault("moral", "Safety first, and learning new skills is awesome!")
    lesson_script.setdefault("duration_estimate", "2-3 minutes")
    lesson_script.setdefault("safety_note", safety_note)

    # Add crafts-skills-specific metadata
    lesson_script["type"] = "crafts_skills"
    lesson_script["category"] = category
    lesson_script["topic"] = topic
    lesson_script["lesson_title"] = lesson_title
    lesson_script["lesson_description"] = lesson_description
    lesson_script["difficulty"] = difficulty

    # Add fields expected by the rest of the pipeline (story compatibility)
    lesson_script["collection"] = f"crafts_{category}"
    lesson_script["origin"] = "Crafts & Skills Curriculum"
    lesson_script["region"] = "en-US"
    lesson_script["age_range"] = diff_ctx["age_range"]
    lesson_script["generated_at"] = datetime.now().isoformat()

    # Ensure all scenes have required fields for the video pipeline
    has_safety_scene = False
    for scene in lesson_script.get("scenes", []):
        scene.setdefault("character_speaking", "handy_the_helper")
        scene.setdefault("duration_seconds", 15)
        scene.setdefault("scene_type", "explanation")

        # Ensure image_search_terms includes Handy the Helper for SDXL consistency
        search_terms = scene.get("image_search_terms", "")
        if "toolbelt" not in search_terms.lower() and "handy" not in search_terms.lower():
            scene["image_search_terms"] = f"{handy_keywords} {search_terms}"

        # Ensure visual descriptions mention cartoon safety gear
        visual_desc = scene.get("visual_description", "")
        if "safety" not in visual_desc.lower() and scene.get("scene_type") != "outro":
            # Add subtle safety reference
            if "goggles" not in visual_desc.lower():
                scene["visual_description"] = visual_desc.rstrip(".") + ", wearing safety goggles."

        if scene.get("scene_type") == "safety":
            has_safety_scene = True

    # Inject a safety scene if the LLM didn't include one
    if not has_safety_scene:
        scenes = lesson_script.get("scenes", [])
        safety_narration = random.choice(SAFETY_SCENE_TEMPLATES).format(
            safety_note=safety_note
        )
        safety_scene = {
            "scene_number": len(scenes) + 1,
            "narration": safety_narration,
            "visual_description": (
                f"Handy the Helper wearing full safety gear with a big cartoon safety "
                f"shield glowing in {visual_theme['colors']}, pointing to a colorful "
                f"safety rule sign that reads the safety tip, in {visual_theme['setting']}"
            ),
            "image_search_terms": (
                f"{handy_keywords} safety gear cartoon shield rule sign "
                f"{visual_theme['colors']}"
            ),
            "character_speaking": "handy_the_helper",
            "duration_seconds": 15,
            "scene_type": "safety",
        }
        # Insert safety scene before the last scene (outro) if possible
        if scenes and scenes[-1].get("scene_type") == "outro":
            scenes.insert(-1, safety_scene)
            # Re-number all scenes
            for i, s in enumerate(scenes):
                s["scene_number"] = i + 1
        else:
            scenes.append(safety_scene)
        lesson_script["scenes"] = scenes

    # Generate a thumbnail description if missing
    lesson_script.setdefault(
        "thumbnail_description",
        f"Handy the Helper cartoon toolbelt character with yellow hard hat "
        f"and safety goggles teaching {lesson_title} with {visual_theme['props']} "
        f"on a bright {visual_theme['colors']} background",
    )

    # Generate tags if missing, then optimize with keyword_optimizer
    default_tags = [
        "kids skills",
        "kids learning",
        category.replace("_", " "),
        topic.lower(),
        difficulty,
        "handy the helper",
        "cartoon workshop",
        "safety first",
        "hands-on learning",
        "Little Wisdom Tales",
        "educational video",
        "kids DIY",
    ]
    lesson_script.setdefault("tags", default_tags)

    # Generate description if missing
    curriculum = load_crafts_curriculum()
    cat_display = curriculum["categories"][category]["display_name"]
    lesson_script.setdefault(
        "description",
        f"Join Handy the Helper as he teaches {lesson_title}! "
        f"A fun {cat_display} lesson for kids ages {diff_ctx['age_range']}. "
        f"Learn about {topic} with colorful cartoon animations, interactive "
        f"challenges, and important safety tips!",
    )

    # --- Keyword optimization (best-effort, non-blocking) ---
    try:
        from scripts.keyword_optimizer import optimize_tags, optimize_title

        lesson_script["title"] = optimize_title(
            base_title=lesson_script["title"],
            moral=lesson_script.get("moral", ""),
            collection=lesson_script.get("collection", ""),
            language="en",
            content_type="crafts_skills",
        )
        lesson_script["tags"] = optimize_tags(
            base_tags=lesson_script["tags"],
            story_script=lesson_script,
            max_tags=30,
            language="en",
            content_type="crafts_skills",
        )
    except ImportError:
        # keyword_optimizer may not be available in all environments
        pass
    except Exception as e:
        print(f"Keyword optimization skipped (non-fatal): {e}")

    return lesson_script


# ===========================================================================
# Main Generation Function
# ===========================================================================


def generate_crafts_lesson_script(
    category: str,
    topic: str,
    lesson_title: str,
    lesson_description: str,
    difficulty: str = "beginner",
    safety_note: str = "Always have adult supervision when working with tools.",
) -> dict:
    """Generate a complete crafts/skills lesson script ready for video production.

    Calls the local Ollama LLM to generate a structured lesson script featuring
    Handy the Helper, then enriches it with metadata and stores it in the
    SQLite database. The returned dict is fully compatible with the existing
    video pipeline (same format as story_generator output).

    Args:
        category: Category key (carpentry_building, plumbing_home, electrical_tech).
        topic: Specific topic name from the curriculum.
        lesson_title: Human-readable lesson title.
        lesson_description: Brief description of the lesson content.
        difficulty: Difficulty level ('beginner', 'intermediate', 'advanced').
        safety_note: Mandatory safety reminder for this lesson.

    Returns:
        A story_script-compatible dict containing scenes, metadata, story_id,
        and lesson_id.

    Raises:
        ValueError: If the LLM fails to produce valid JSON after retries.
    """
    prompt = build_crafts_lesson_prompt(
        category, topic, lesson_title, lesson_description, difficulty, safety_note
    )

    # Try LLM generation with retry logic for JSON parsing failures
    lesson_script = None
    last_error = None

    for gen_attempt in range(3):
        try:
            raw_response = call_ollama(prompt)
        except Exception as e:
            print(f"LLM call failed on attempt {gen_attempt + 1}: {e}")
            last_error = e
            _time.sleep(5)
            continue

        try:
            lesson_script = parse_llm_json(raw_response)
            # Validate required fields
            if "scenes" not in lesson_script or not lesson_script["scenes"]:
                raise ValueError("Lesson has no scenes")
            if len(lesson_script["scenes"]) < 4:
                raise ValueError(
                    f"Lesson only has {len(lesson_script['scenes'])} scenes, need at least 4"
                )
            break
        except (ValueError, KeyError) as e:
            last_error = e
            print(f"JSON parse attempt {gen_attempt + 1} failed: {e}, retrying...")
            _time.sleep(5)

    if lesson_script is None:
        raise ValueError(
            f"Failed to generate valid crafts lesson after 3 attempts: {last_error}"
        )

    # Enrich with metadata and pipeline-compatible fields
    lesson_script = _enrich_crafts_script(
        lesson_script,
        category,
        topic,
        lesson_title,
        lesson_description,
        difficulty,
        safety_note,
    )

    # --- Save to the main stories table (pipeline compatibility) ---
    db = get_db()
    db.execute(
        """INSERT INTO stories (title, collection, region, moral, script, status)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (
            lesson_script.get("title", lesson_title),
            f"crafts_{category}",
            "en-US",
            lesson_script.get("moral", "Safety first, and learning new skills is awesome!"),
            json.dumps(lesson_script),
            "generated",
        ),
    )
    db.commit()
    story_id = db.execute("SELECT last_insert_rowid()").fetchone()[0]
    lesson_script["story_id"] = story_id

    # --- Save to crafts_skills_lessons table (lesson-specific tracking) ---
    db.execute(
        """INSERT INTO crafts_skills_lessons
           (category, topic, lesson_title, difficulty, safety_note, script, story_id, status)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            category,
            topic,
            lesson_title,
            difficulty,
            safety_note,
            json.dumps(lesson_script),
            story_id,
            "generated",
        ),
    )
    db.commit()
    lesson_id = db.execute("SELECT last_insert_rowid()").fetchone()[0]
    lesson_script["lesson_id"] = lesson_id

    db.close()

    return lesson_script


# ===========================================================================
# Lesson Picking & Progress Tracking
# ===========================================================================


def pick_next_crafts_lesson(category: str = None) -> dict | None:
    """Pick the next crafts/skills lesson to generate based on curriculum and history.

    Reads config/crafts_skills_curriculum.json for available lessons, checks
    the database for already-generated lessons, and returns the next lesson
    that hasn't been created yet. Cycles through categories to provide variety.

    Args:
        category: Optional category filter (e.g. 'carpentry_building').
            If specified, only picks lessons from that category
    (not all carpentry in a row).

    Returns:
        A dict with keys: category, topic, lesson_title, lesson_description,
        difficulty, safety_note. Returns None if all lessons have been generated.
    """
    curriculum = load_crafts_curriculum()
    categories = curriculum.get("categories", {})

    # Get all already-generated lessons from the database
    db = get_db()
    rows = db.execute(
        "SELECT category, topic, lesson_title FROM crafts_skills_lessons"
    ).fetchall()

    generated_set = {
        (row["category"], row["topic"], row["lesson_title"]) for row in rows
    }

    # Determine last generated category for rotation
    last_category = None
    last_row = db.execute(
        """SELECT category FROM crafts_skills_lessons
           ORDER BY created_at DESC LIMIT 1"""
    ).fetchone()
    db.close()

    if last_row:
        last_category = last_row["category"]

    # Build rotation order for categories (optionally filtered)
    category_keys = list(categories.keys())
    if category and category in category_keys:
        # Filter to specific category only
        category_keys = [category]
    if last_category and last_category in category_keys:
        last_idx = category_keys.index(last_category)
        rotated_categories = (
            category_keys[last_idx + 1 :] + category_keys[: last_idx + 1]
        )
    else:
        rotated_categories = list(category_keys)

    # Find the next ungenerated lesson, rotating through categories
    for cat_key in rotated_categories:
        cat_data = categories.get(cat_key, {})
        topics = cat_data.get("topics", [])

        for topic_data in topics:
            topic_name = topic_data["topic"]
            lessons = topic_data.get("lessons", [])

            for lesson in lessons:
                title = lesson["title"]
                if (cat_key, topic_name, title) not in generated_set:
                    return {
                        "category": cat_key,
                        "topic": topic_name,
                        "lesson_title": title,
                        "lesson_description": lesson.get("description", ""),
                        "difficulty": lesson.get("difficulty", "beginner"),
                        "safety_note": lesson.get(
                            "safety_note",
                            "Always have adult supervision when working with tools.",
                        ),
                    }

    # All lessons exhausted - start a new cycle with fresh angles.
    total_in_curriculum = sum(
        len(topic_data.get("lessons", []))
        for cat_data in categories.values()
        for topic_data in cat_data.get("topics", [])
    )
    if total_in_curriculum == 0:
        return None

    cycle = max(1, len(generated_set) // max(total_in_curriculum, 1))

    cycle_suffixes = [
        "Part 2: Advanced Techniques",
        "Part 3: Pro Tips",
        "Part 4: Safety Challenge",
        "Part 5: Real World Projects",
        "Part 6: Build It Better",
        "Part 7: Master Builder",
        "Part 8: Expert Mode",
        "Part 9: Creative Projects",
        "Part 10: Super Skills",
    ]
    suffix = cycle_suffixes[(cycle - 1) % len(cycle_suffixes)]

    for cat_key in rotated_categories:
        cat_data = categories.get(cat_key, {})
        topics = cat_data.get("topics", [])
        for topic_data in topics:
            topic_name = topic_data["topic"]
            lessons = topic_data.get("lessons", [])
            for lesson in lessons:
                cycle_title = f"{lesson['title']} - {suffix}"
                if (cat_key, topic_name, cycle_title) not in generated_set:
                    return {
                        "category": cat_key,
                        "topic": topic_name,
                        "lesson_title": cycle_title,
                        "lesson_description": f"{lesson.get('description', '')} ({suffix})",
                        "difficulty": lesson.get("difficulty", "beginner"),
                        "safety_note": lesson.get(
                            "safety_note",
                            "Always have adult supervision when working with tools.",
                        ),
                    }

    # Absolute fallback
    import random
    cat = random.choice(list(categories.keys()))
    return {
        "category": cat,
        "topic": f"Bonus: {cat.replace('_', ' ').title()} Review",
        "lesson_title": f"Fun Review: {cat.replace('_', ' ').title()} - Cycle {cycle + 1}",
        "lesson_description": "A fun review of everything we've learned!",
        "difficulty": "beginner",
        "safety_note": "Always have adult supervision when working with tools.",
    }


def get_crafts_lesson_progress() -> dict:
    """Get a summary of lesson generation progress across all categories and topics.

    Returns:
        A dict with total/generated/remaining counts and per-category breakdowns,
        including per-topic detail within each category.
    """
    curriculum = load_crafts_curriculum()
    categories = curriculum.get("categories", {})

    db = get_db()
    rows = db.execute(
        "SELECT category, topic, lesson_title FROM crafts_skills_lessons"
    ).fetchall()
    db.close()

    generated_set = {
        (row["category"], row["topic"], row["lesson_title"]) for row in rows
    }

    progress = {
        "total_lessons": 0,
        "generated": 0,
        "remaining": 0,
        "categories": {},
    }

    for cat_key, cat_data in categories.items():
        topics = cat_data.get("topics", [])
        cat_total = 0
        cat_generated = 0
        topic_progress = []

        for topic_data in topics:
            topic_name = topic_data["topic"]
            lessons = topic_data.get("lessons", [])
            t_total = len(lessons)
            t_generated = sum(
                1
                for lesson in lessons
                if (cat_key, topic_name, lesson["title"]) in generated_set
            )
            cat_total += t_total
            cat_generated += t_generated
            topic_progress.append(
                {
                    "topic": topic_name,
                    "total": t_total,
                    "generated": t_generated,
                    "remaining": t_total - t_generated,
                }
            )

        progress["categories"][cat_key] = {
            "display_name": cat_data.get("display_name", cat_key),
            "total": cat_total,
            "generated": cat_generated,
            "remaining": cat_total - cat_generated,
            "topics": topic_progress,
        }
        progress["total_lessons"] += cat_total
        progress["generated"] += cat_generated

    progress["remaining"] = progress["total_lessons"] - progress["generated"]

    return progress


def mark_crafts_lesson_published(lesson_id: int, video_id: str) -> None:
    """Mark a crafts/skills lesson as published with its YouTube video ID.

    Updates both the crafts_skills_lessons table and the corresponding
    stories table entry to reflect the published status.

    Args:
        lesson_id: The crafts_skills_lessons table ID.
        video_id: The YouTube video ID after successful upload.
    """
    db = get_db()

    # Update the crafts_skills_lessons table
    db.execute(
        "UPDATE crafts_skills_lessons SET status = 'published', video_id = ? WHERE id = ?",
        (video_id, lesson_id),
    )

    # Also update the corresponding stories table entry
    row = db.execute(
        "SELECT story_id FROM crafts_skills_lessons WHERE id = ?",
        (lesson_id,),
    ).fetchone()
    if row and row["story_id"]:
        db.execute(
            "UPDATE stories SET status = 'published', video_id = ?, published_at = ? WHERE id = ?",
            (video_id, datetime.now().isoformat(), row["story_id"]),
        )

    db.commit()
    db.close()


def get_crafts_playlist_name(category: str) -> str:
    """Generate a standardized YouTube playlist name for a crafts/skills category.

    Args:
        category: Category key from the curriculum.

    Returns:
        A playlist name string like "Carpentry & Building for Kids - Learn to Build with Handy!"
    """
    curriculum = load_crafts_curriculum()
    categories = curriculum.get("categories", {})
    cat_data = categories.get(category, {})
    return cat_data.get(
        "playlist_name",
        f"{category.replace('_', ' ').title()} for Kids - Handy the Helper",
    )


def get_generated_crafts_lessons(
    category: str = None,
    topic: str = None,
    difficulty: str = None,
    status: str = None,
    limit: int = 50,
) -> list[dict]:
    """Query generated crafts/skills lessons with optional filters.

    Args:
        category: Filter by category key, or None for all.
        topic: Filter by topic name, or None for all.
        difficulty: Filter by difficulty ('beginner', 'intermediate', 'advanced'),
                    or None for all.
        status: Filter by status ('generated', 'published'), or None for all.
        limit: Maximum number of results to return.

    Returns:
        List of lesson dictionaries from the crafts_skills_lessons table.
    """
    db = get_db()

    query = "SELECT * FROM crafts_skills_lessons WHERE 1=1"
    params = []

    if category is not None:
        query += " AND category = ?"
        params.append(category)
    if topic is not None:
        query += " AND topic = ?"
        params.append(topic)
    if difficulty is not None:
        query += " AND difficulty = ?"
        params.append(difficulty)
    if status is not None:
        query += " AND status = ?"
        params.append(status)

    query += " ORDER BY created_at DESC LIMIT ?"
    params.append(limit)

    rows = db.execute(query, params).fetchall()
    db.close()

    return [dict(row) for row in rows]


# ===========================================================================
# CLI Entry Point
# ===========================================================================


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage:")
        print("  python crafts_skills_generator.py next")
        print("  python crafts_skills_generator.py generate <category> <topic> <title> <description> [difficulty] [safety_note]")
        print("  python crafts_skills_generator.py progress")
        print("  python crafts_skills_generator.py list [--category C] [--topic T] [--difficulty D] [--limit N]")
        sys.exit(1)

    command = sys.argv[1]

    if command == "next":
        next_lesson = pick_next_crafts_lesson()
        if next_lesson is None:
            print("All crafts & skills lessons in the curriculum have been generated!")
        else:
            print("Next lesson to generate:")
            print(json.dumps(next_lesson, indent=2))
            print("\nGenerating lesson script...")
            script = generate_crafts_lesson_script(**next_lesson)
            print(f"\nLesson generated successfully!")
            print(f"  Story ID: {script['story_id']}")
            print(f"  Lesson ID: {script['lesson_id']}")
            print(f"  Title: {script['title']}")
            print(f"  Category: {script['category']}")
            print(f"  Difficulty: {script['difficulty']}")
            print(f"  Scenes: {len(script.get('scenes', []))}")
            print(f"  Safety Note: {script.get('safety_note', 'N/A')}")
            print(f"\nFull script:")
            print(json.dumps(script, indent=2))

    elif command == "generate":
        if len(sys.argv) < 6:
            print(
                "Usage: python crafts_skills_generator.py generate "
                "<category> <topic> <title> <description> [difficulty] [safety_note]"
            )
            sys.exit(1)
        category = sys.argv[2]
        topic = sys.argv[3]
        title = sys.argv[4]
        description = sys.argv[5]
        difficulty = sys.argv[6] if len(sys.argv) > 6 else "beginner"
        safety_note = (
            sys.argv[7]
            if len(sys.argv) > 7
            else "Always have adult supervision when working with tools."
        )

        script = generate_crafts_lesson_script(
            category, topic, title, description, difficulty, safety_note
        )
        print(json.dumps(script, indent=2))

    elif command == "progress":
        progress = get_crafts_lesson_progress()
        print("Crafts & Skills Lesson Progress")
        print(f"{'=' * 60}")
        print(f"Total lessons in curriculum: {progress['total_lessons']}")
        print(f"Generated: {progress['generated']}")
        print(f"Remaining: {progress['remaining']}")
        print()
        for cat_key, info in progress["categories"].items():
            pct = (info["generated"] / info["total"] * 100) if info["total"] > 0 else 0
            bar = "#" * int(pct / 5) + "-" * (20 - int(pct / 5))
            print(
                f"  {info['display_name']:25s} [{bar}] "
                f"{info['generated']:3d}/{info['total']:3d} ({pct:.0f}%)"
            )
            for tp in info["topics"]:
                tp_pct = (
                    (tp["generated"] / tp["total"] * 100) if tp["total"] > 0 else 0
                )
                tp_bar = "#" * int(tp_pct / 5) + "-" * (20 - int(tp_pct / 5))
                print(
                    f"    {tp['topic']:30s} [{tp_bar}] "
                    f"{tp['generated']:2d}/{tp['total']:2d} ({tp_pct:.0f}%)"
                )
            print()

    elif command == "list":
        kwargs = {}
        if "--category" in sys.argv:
            idx = sys.argv.index("--category")
            if idx + 1 < len(sys.argv):
                kwargs["category"] = sys.argv[idx + 1]
        if "--topic" in sys.argv:
            idx = sys.argv.index("--topic")
            if idx + 1 < len(sys.argv):
                kwargs["topic"] = sys.argv[idx + 1]
        if "--difficulty" in sys.argv:
            idx = sys.argv.index("--difficulty")
            if idx + 1 < len(sys.argv):
                kwargs["difficulty"] = sys.argv[idx + 1]
        if "--limit" in sys.argv:
            idx = sys.argv.index("--limit")
            if idx + 1 < len(sys.argv):
                kwargs["limit"] = int(sys.argv[idx + 1])

        lessons = get_generated_crafts_lessons(**kwargs)
        if not lessons:
            print("No crafts/skills lessons found matching the criteria.")
        else:
            print(f"Found {len(lessons)} lesson(s):")
            for lesson in lessons:
                print(
                    f"  [{lesson['id']}] {lesson['category']} / {lesson['topic']} "
                    f"- {lesson['lesson_title']} "
                    f"({lesson['difficulty']}, {lesson['status']})"
                )

    else:
        print(f"Unknown command: {command}")
        sys.exit(1)
