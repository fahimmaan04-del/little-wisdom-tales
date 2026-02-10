"""
AI Education Generator - Creates kid-friendly AI/tech lesson scripts using LLM.

Generates structured lesson scripts for kids (ages 8-14) covering AI, coding,
robotics, and creative tech topics. Each lesson features "Byte the Robot"
(a friendly blue robot kid) as the recurring host character.

Lessons are generated in the same JSON scene format as story scripts,
making them fully compatible with the existing video pipeline
(TTS -> AI Images -> FFmpeg Video -> YouTube Upload).
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

# Byte the Robot's catchphrases for variety
BYTE_PHRASES = [
    "BEEP BOOP! Let's explore something awesome today!",
    "Hey friends! Byte the Robot here, ready to learn!",
    "BEEP BOOP! That's super cool, right?!",
    "Whoa! Are you ready for something mind-blowing?",
    "BZZZT! My circuits are buzzing with excitement!",
    "High-five, humans! You're doing amazing!",
    "BEEP BOOP! My sensors detect a great learner!",
    "Processing... processing... WOW, that's incredible!",
]

# Visual themes for each AI curriculum category
CATEGORY_VISUAL_THEMES = {
    "ai_basics": {
        "colors": "neon blue, electric green, and bright purple",
        "props": "holographic brain models, glowing neural networks, floating data cubes, binary code streams",
        "setting": "a futuristic neon-lit classroom with holographic displays and floating screens",
    },
    "ai_art": {
        "colors": "neon pink, electric cyan, and glowing orange",
        "props": "digital paintbrushes, holographic canvases, AI-generated art floating in frames, pixel particles",
        "setting": "a high-tech art studio with giant touchscreens and floating digital artwork",
    },
    "ai_games": {
        "colors": "neon green, electric blue, and hot pink",
        "props": "floating game controllers, pixel characters, code blocks, holographic game screens",
        "setting": "a neon-lit gaming lab with arcade machines and holographic game displays",
    },
    "ai_apps": {
        "colors": "electric blue, neon purple, and bright teal",
        "props": "floating phone screens, app icons, code editors, holographic app prototypes",
        "setting": "a modern tech startup office with giant monitors and holographic app previews",
    },
    "ai_music": {
        "colors": "neon purple, electric pink, and glowing gold",
        "props": "holographic sound waves, digital piano keys, floating music notes, equalizer bars",
        "setting": "a futuristic music studio with neon equalizers and holographic instruments",
    },
    "ai_stories": {
        "colors": "electric cyan, neon magenta, and glowing yellow",
        "props": "holographic storybooks, floating text, digital quill pens, animated story characters",
        "setting": "a magical digital library with floating holographic books and animated story scenes",
    },
    "ai_robots": {
        "colors": "metallic silver, neon orange, and electric blue",
        "props": "robot arms, circuit boards, servo motors, sensor arrays, holographic blueprints",
        "setting": "a high-tech robotics workshop with 3D printers and assembly stations",
    },
    "ai_chatbots": {
        "colors": "neon green, electric blue, and bright white",
        "props": "chat bubbles, speech waveforms, NLP diagrams, holographic conversation flows",
        "setting": "a modern AI chat lab with floating conversation bubbles and glowing screens",
    },
    "ai_projects": {
        "colors": "rainbow neon: pink, blue, green, orange, and purple",
        "props": "toolboxes, circuit kits, laptop screens with code, project blueprints, 3D models",
        "setting": "a maker space with workbenches, 3D printers, and walls covered in project ideas",
    },
    "ai_future": {
        "colors": "cosmic blue, neon white, and electric violet",
        "props": "rocket ships, holographic planets, futuristic cityscapes, flying cars, space stations",
        "setting": "a space-age observatory with panoramic views of a futuristic city and starry sky",
    },
    "scratch_ai": {
        "colors": "Scratch orange, electric blue, and neon yellow",
        "props": "Scratch code blocks, sprites, backdrops, variables, colorful programming puzzles",
        "setting": "a colorful coding playground with giant Scratch blocks and animated sprites",
    },
    "python_kids": {
        "colors": "Python blue, neon green, and electric yellow",
        "props": "code terminals, Python snake logo, syntax blocks, debugger magnifying glasses",
        "setting": "a hacker-style coding den with multiple monitors showing Python code and outputs",
    },
}

# Try It! challenge templates for interactive moments
TRY_IT_TEMPLATES = [
    "TRY IT! {challenge} Ask a parent or guardian to help you try it!",
    "YOUR TURN! {challenge} Give it a shot and see what happens!",
    "CHALLENGE TIME! {challenge} Can you do it? I bet you can!",
    "HANDS-ON MOMENT! {challenge} Try this at home with a grown-up!",
    "EXPERIMENT TIME! {challenge} Let's see what you discover!",
]


def get_db():
    """Get or create the stories database with ai_education_lessons table."""
    db_path = DATA_DIR / "stories.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    # Ensure the main stories table exists (same as story_generator)
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

    # Create ai_education_lessons table for tracking AI lesson-specific metadata
    conn.execute("""
        CREATE TABLE IF NOT EXISTS ai_education_lessons (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            category TEXT NOT NULL,
            topic TEXT NOT NULL,
            lesson_title TEXT NOT NULL,
            difficulty TEXT DEFAULT 'beginner',
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


def load_ai_curriculum() -> dict:
    """Load the AI kids curriculum configuration."""
    curriculum_path = CONFIG_DIR / "ai_kids_curriculum.json"
    if not curriculum_path.exists():
        raise FileNotFoundError(
            f"AI curriculum not found at {curriculum_path}. "
            "Please create config/ai_kids_curriculum.json first."
        )
    with open(curriculum_path) as f:
        return json.load(f)


def _get_visual_theme(category: str) -> dict:
    """Get visual theme configuration for a category."""
    return CATEGORY_VISUAL_THEMES.get(category, CATEGORY_VISUAL_THEMES["ai_basics"])


def build_ai_lesson_prompt(
    category: str,
    topic: str,
    lesson_title: str,
) -> str:
    """
    Build an LLM prompt to generate an AI education lesson script.

    Creates a detailed prompt that instructs the LLM to generate a structured
    JSON lesson featuring Byte the Robot (a friendly blue robot kid) teaching
    the given AI/tech topic in an engaging, age-appropriate way.

    Args:
        category: Category identifier from the AI curriculum (e.g. "ai_basics").
        topic: Specific topic within the category (e.g. "Introduction to AI").
        lesson_title: Human-readable lesson title.

    Returns:
        A formatted prompt string ready for LLM generation.
    """
    visual_theme = _get_visual_theme(category)
    curriculum = load_ai_curriculum()
    channel = curriculum.get("channel", {})

    host_desc = channel.get(
        "host_description",
        "a friendly blue robot kid with glowing green eyes, antenna on head, and a tablet in hand",
    )
    target_age = channel.get("target_age", "8-14")

    # Build host visual keywords for image generation
    host_keywords = (
        "friendly blue robot kid glowing green eyes antenna tablet "
        "cartoon futuristic tech cute robot character"
    )

    # Get category display name
    categories = curriculum.get("categories", {})
    category_data = categories.get(category, {})
    category_display = category_data.get("display_name", category.replace("_", " ").title())
    difficulty = category_data.get("difficulty", "beginner")

    return f"""You are an expert children's educational content creator making a video lesson for a YouTube kids channel called "Little Wisdom AI Lab".

Generate an AI education lesson script based on these parameters:
- Category: {category_display}
- Topic: {topic}
- Lesson Title: {lesson_title}
- Target Age: {target_age} years old
- Difficulty: {difficulty}
- Language: English

MAIN CHARACTER:
- Name: Byte the Robot
- Appearance: {host_desc}
- Byte the Robot appears in EVERY scene as the host and teacher
- Byte is enthusiastic, curious, encouraging, and uses the catchphrase "BEEP BOOP!"
- Byte speaks in a fun, energetic way that makes complex tech concepts simple and exciting
- Byte holds a tablet that shows examples, diagrams, and code snippets

VISUAL STYLE:
- Setting: {visual_theme['setting']}
- Color palette: {visual_theme['colors']}
- Props and tools: {visual_theme['props']}
- Style: Bright, techy, neon-colored, futuristic, cartoon, kid-friendly
- Every visual_description MUST include Byte the Robot doing something specific
- Use glowing effects, holographic displays, and neon lighting in every scene
- Make it look like a cool futuristic tech show for kids

LESSON STRUCTURE (8-12 scenes, each ~15 seconds of narration):
1. INTRO (1 scene): Byte the Robot greets kids, introduces today's AI topic with excitement
2. WHAT IS IT? (2-3 scenes): Explain the concept in simple terms with visual analogies kids can relate to
3. HOW DOES IT WORK? (2-3 scenes): Break down the mechanics with step-by-step visual demonstrations
4. REAL WORLD EXAMPLES (1-2 scenes): Show cool real-world applications that kids encounter
5. TRY IT! (1-2 scenes): Hands-on interactive moment where kids can try something themselves (name a free, kid-safe tool or activity they can do at home with parental supervision)
6. WHAT WE LEARNED (1 scene): Quick recap of key takeaways
7. SIGN OFF (1 scene): Byte encourages kids, teases next lesson, asks them to subscribe

IMPORTANT RULES:
- WRITE EVERYTHING IN ENGLISH ONLY
- Use kid-friendly language appropriate for ages {target_age}
- Make it FUN and ENGAGING - use sound effects like "BEEP BOOP!", "BZZZT!", "WHOOSH!", "DING!"
- Include "TRY IT!" moments where kids can do a hands-on activity (with parental supervision reminders)
- Use concrete visual analogies (AI is like a puppy learning tricks, algorithms are like recipes, etc.)
- No scary content, no hacking, no mature themes
- Always mention parental supervision for any online tool recommendations
- Each scene MUST have Byte the Robot doing something visible and specific
- The visual_description should paint a vivid futuristic, neon-lit scene
- TRY IT scenes should mention specific free, kid-safe tools or activities
- Include safety reminders about online tools and data privacy where relevant

Return a JSON object with this EXACT structure (no markdown, just raw JSON):
{{
  "title": "{lesson_title}",
  "category": "{category}",
  "topic": "{topic}",
  "difficulty": "{difficulty}",
  "moral": "Technology is amazing and everyone can learn to use AI responsibly!",
  "duration_estimate": "3-4 minutes",
  "thumbnail_description": "Byte the Robot, {host_desc}, standing next to a glowing holographic display about {topic} with {visual_theme['colors']} neon background",
  "scenes": [
    {{
      "scene_number": 1,
      "narration": "BEEP BOOP! Hey friends! I'm Byte the Robot, and today we're going to explore something SUPER cool!",
      "visual_description": "Byte the Robot, {host_desc}, waving happily in {visual_theme['setting']} with neon lights and a holographic banner showing the lesson title",
      "image_search_terms": "{host_keywords} {visual_theme['colors']} futuristic classroom neon tech kids education",
      "character_speaking": "byte_the_robot",
      "duration_seconds": 15,
      "scene_type": "intro"
    }},
    {{
      "scene_number": 2,
      "narration": "Concept explanation with fun analogies...",
      "visual_description": "Byte the Robot pointing at a glowing holographic display showing the concept visually...",
      "image_search_terms": "relevant search terms for the concept",
      "character_speaking": "byte_the_robot",
      "duration_seconds": 15,
      "scene_type": "explanation"
    }},
    {{
      "scene_number": 3,
      "narration": "A TRY IT! hands-on moment with a specific activity...",
      "visual_description": "Byte the Robot holding up a tablet showing a kid-safe tool or activity...",
      "image_search_terms": "relevant try-it visual terms",
      "character_speaking": "byte_the_robot",
      "duration_seconds": 15,
      "scene_type": "try_it"
    }}
  ],
  "tags": ["AI for kids", "kids technology", "{category.replace('_', ' ')}", "{topic}", "Byte the Robot", "AI education", "kids coding", "learn AI"],
  "description": "Join Byte the Robot as he teaches {lesson_title} in this fun AI education video for kids aged {target_age}!"
}}

Create 8-12 scenes total. Include at least 1-2 TRY IT! scenes where kids do something hands-on.
Each scene must have a unique, detailed visual_description with Byte the Robot doing something specific.
The visual descriptions should use bright, techy, neon-colored imagery: holographic displays, glowing screens, floating code, neon lights.
Always remind kids to ask a parent or guardian before using any online tools.
Return ONLY the JSON, no other text."""


def call_ollama(prompt: str, max_retries: int = 3) -> str:
    """
    Call local Ollama LLM with retry logic.

    Uses the same Ollama instance and configuration as story_generator.py.

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
    """
    Extract JSON from LLM response, handling common formatting issues.

    Mirrors the same robust parsing logic from story_generator.py to handle
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


def _enrich_ai_lesson_script(
    lesson_script: dict,
    category: str,
    topic: str,
    lesson_title: str,
) -> dict:
    """
    Enrich the raw LLM output with metadata needed by the pipeline.

    Adds fields that the video pipeline expects (collection, region, age_range, etc.)
    and ensures the lesson script is fully compatible with the story format.

    Args:
        lesson_script: The parsed lesson script from the LLM.
        category: Category identifier from the AI curriculum.
        topic: Topic within the category.
        lesson_title: Human-readable lesson title.

    Returns:
        Enriched lesson script dictionary.
    """
    curriculum = load_ai_curriculum()
    channel = curriculum.get("channel", {})
    categories = curriculum.get("categories", {})
    category_data = categories.get(category, {})
    target_age = channel.get("target_age", "8-14")
    difficulty = category_data.get("difficulty", "beginner")

    # Ensure required fields exist with sensible defaults
    lesson_script.setdefault("title", lesson_title)
    lesson_script.setdefault("moral", "Technology is amazing and everyone can learn to use AI responsibly!")
    lesson_script.setdefault("duration_estimate", "3-4 minutes")

    # Add AI education-specific metadata
    lesson_script["type"] = "ai_education"
    lesson_script["category"] = category
    lesson_script["topic"] = topic
    lesson_script["lesson_title"] = lesson_title
    lesson_script["difficulty"] = difficulty

    # Add fields expected by the rest of the pipeline (story compatibility)
    lesson_script["collection"] = f"ai_education_{category}"
    lesson_script["origin"] = "AI Kids Curriculum"
    lesson_script["region"] = "en-US"
    lesson_script["age_range"] = target_age
    lesson_script["generated_at"] = datetime.now().isoformat()

    # Ensure all scenes have required fields for the video pipeline
    for scene in lesson_script.get("scenes", []):
        scene.setdefault("character_speaking", "byte_the_robot")
        scene.setdefault("duration_seconds", 15)
        scene.setdefault("scene_type", "explanation")
        # Ensure image_search_terms includes Byte the Robot for SDXL consistency
        search_terms = scene.get("image_search_terms", "")
        if "robot" not in search_terms.lower():
            scene["image_search_terms"] = f"blue robot kid cartoon futuristic {search_terms}"

    # Generate a thumbnail description if missing
    visual_theme = _get_visual_theme(category)
    host_desc = channel.get(
        "host_description",
        "a friendly blue robot kid with glowing green eyes, antenna on head, and a tablet in hand",
    )
    lesson_script.setdefault(
        "thumbnail_description",
        f"Byte the Robot, {host_desc}, "
        f"next to a glowing holographic display about {lesson_title} "
        f"with {visual_theme['colors']} neon background",
    )

    # Generate tags if missing
    default_tags = [
        "AI for kids",
        "kids technology",
        category.replace("_", " "),
        topic,
        "Byte the Robot",
        "AI education",
        "kids coding",
        "learn AI",
        "tech for kids",
        "Little Wisdom AI Lab",
    ]
    lesson_script.setdefault("tags", default_tags)

    # Generate description if missing
    lesson_script.setdefault(
        "description",
        f"Join Byte the Robot as he teaches {lesson_title}! "
        f"A fun AI education video for kids aged {target_age}. "
        f"Learn about {topic} with hands-on activities and cool visual explanations!",
    )

    return lesson_script


def generate_ai_lesson_script(
    category: str,
    topic: str,
    lesson_title: str,
) -> dict:
    """
    Generate a complete AI education lesson script ready for video production.

    Calls the local Ollama LLM to generate a structured lesson script featuring
    Byte the Robot, then enriches it with metadata and stores it in the
    SQLite database. The returned dict is fully compatible with the existing
    video pipeline (same format as story_generator output).

    Args:
        category: Category identifier from the AI curriculum (e.g. "ai_basics").
        topic: Specific topic within the category (e.g. "Introduction to AI").
        lesson_title: Human-readable lesson title.

    Returns:
        A story_script-compatible dict containing scenes, metadata, and a story_id.

    Raises:
        ValueError: If the LLM fails to produce valid JSON after retries.
    """
    from scripts.keyword_optimizer import optimize_title, optimize_tags, optimize_description

    prompt = build_ai_lesson_prompt(category, topic, lesson_title)

    # Try LLM generation with retry logic for JSON parsing failures
    lesson_script = None
    last_error = None

    for gen_attempt in range(3):
        try:
            raw_response = call_ollama(prompt)
        except Exception as e:
            print(f"LLM call failed on attempt {gen_attempt + 1}: {e}")
            last_error = e
            import time as _time
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
            import time as _time
            _time.sleep(5)

    if lesson_script is None:
        raise ValueError(f"Failed to generate valid lesson after 3 attempts: {last_error}")

    # Enrich with metadata and pipeline-compatible fields
    lesson_script = _enrich_ai_lesson_script(
        lesson_script, category, topic, lesson_title
    )

    # Optimize title, tags, and description with keyword intelligence
    lesson_script["title"] = optimize_title(
        lesson_script.get("title", lesson_title),
        language="en",
        content_type="ai_education",
        collection=f"ai_education_{category}",
    )
    lesson_script["tags"] = optimize_tags(
        lesson_script.get("tags", []),
        story_script=lesson_script,
        language="en",
        content_type="ai_education",
    )
    lesson_script["description"] = optimize_description(
        lesson_script.get("description", ""),
        story_script=lesson_script,
        language="en",
        content_type="ai_education",
    )

    # Save to the main stories table (pipeline compatibility)
    db = get_db()
    db.execute(
        """INSERT INTO stories (title, collection, region, moral, script, status)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (
            lesson_script.get("title", lesson_title),
            f"ai_education_{category}",
            "en-US",
            lesson_script.get("moral", "Technology is amazing and everyone can learn to use AI responsibly!"),
            json.dumps(lesson_script),
            "generated",
        ),
    )
    db.commit()
    story_id = db.execute("SELECT last_insert_rowid()").fetchone()[0]
    lesson_script["story_id"] = story_id

    # Save to the ai_education_lessons table (lesson-specific tracking)
    curriculum = load_ai_curriculum()
    categories = curriculum.get("categories", {})
    category_data = categories.get(category, {})
    difficulty = category_data.get("difficulty", "beginner")

    db.execute(
        """INSERT INTO ai_education_lessons
           (category, topic, lesson_title, difficulty, script, story_id, status)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (
            category,
            topic,
            lesson_title,
            difficulty,
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


def pick_next_ai_lesson() -> dict | None:
    """
    Pick the next AI lesson to generate based on the curriculum and past generation history.

    Reads config/ai_kids_curriculum.json for available lessons, checks the
    database for already-generated lessons, and returns the next lesson
    that hasn't been created yet. Cycles through categories to provide variety
    (not all AI basics in a row).

    Returns:
        A dict with keys: category, topic, lesson_title.
        Returns None if all lessons in the curriculum have been generated.
    """
    curriculum = load_ai_curriculum()
    categories = curriculum.get("categories", {})
    rotation_order = curriculum.get("playlist_order", list(categories.keys()))

    # Get all already-generated lesson keys from the database
    db = get_db()
    rows = db.execute(
        "SELECT category, topic, lesson_title FROM ai_education_lessons"
    ).fetchall()
    db.close()

    generated_set = {(row["category"], row["topic"], row["lesson_title"]) for row in rows}

    # Determine the last generated category to continue rotation
    last_category = None
    if rows:
        db = get_db()
        last_row = db.execute(
            """SELECT category FROM ai_education_lessons
               ORDER BY created_at DESC LIMIT 1"""
        ).fetchone()
        db.close()
        if last_row:
            last_category = last_row["category"]

    # Build a prioritized list of candidate lessons:
    # 1. Rotate categories (if last was ai_basics, next should be scratch_ai, etc.)
    # 2. Within each category, go through topics in order
    # 3. Within each topic, go through lessons in order (first ungenerated)

    # Determine category rotation starting point
    if last_category and last_category in rotation_order:
        last_idx = rotation_order.index(last_category)
        rotated_categories = rotation_order[last_idx + 1 :] + rotation_order[: last_idx + 1]
    else:
        rotated_categories = list(rotation_order)

    # Find the next ungenerated lesson
    for cat_key in rotated_categories:
        cat_data = categories.get(cat_key, {})
        topics = cat_data.get("topics", [])

        for topic_entry in topics:
            topic_name = topic_entry.get("topic", "")
            lessons = topic_entry.get("lessons", [])

            for lesson_title in lessons:
                if (cat_key, topic_name, lesson_title) not in generated_set:
                    return {
                        "category": cat_key,
                        "topic": topic_name,
                        "lesson_title": lesson_title,
                    }

    # All lessons exhausted - start a new cycle with fresh angles.
    # Count cycles by dividing generated lessons by total lessons in curriculum.
    total_in_curriculum = sum(
        len(lesson_list)
        for cat_data in categories.values()
        for topic_entry in cat_data.get("topics", [])
        for lesson_list in [topic_entry.get("lessons", [])]
    )
    if total_in_curriculum == 0:
        return None

    cycle = max(1, len(generated_set) // max(total_in_curriculum, 1))

    cycle_suffixes = [
        "Part 2: Advanced",
        "Part 3: Challenge",
        "Part 4: Deep Dive",
        "Part 5: Hands-On",
        "Part 6: Real World",
        "Part 7: Quiz Time",
        "Part 8: Build It",
        "Part 9: Expert Level",
        "Part 10: Master Class",
    ]
    suffix = cycle_suffixes[(cycle - 1) % len(cycle_suffixes)]

    for cat_key in rotated_categories:
        cat_data = categories.get(cat_key, {})
        topics = cat_data.get("topics", [])
        for topic_entry in topics:
            topic_name = topic_entry.get("topic", "")
            lessons = topic_entry.get("lessons", [])
            for lesson_title in lessons:
                cycle_title = f"{lesson_title} - {suffix}"
                if (cat_key, topic_name, cycle_title) not in generated_set:
                    return {
                        "category": cat_key,
                        "topic": topic_name,
                        "lesson_title": cycle_title,
                    }

    # Absolute fallback
    import random
    cat = random.choice(list(categories.keys()))
    return {
        "category": cat,
        "topic": f"Bonus: {cat.replace('_', ' ').title()} Cycle {cycle + 1}",
        "lesson_title": f"Fun Review: {cat.replace('_', ' ').title()} with Byte!",
    }


def get_ai_playlist_name(category: str) -> str:
    """
    Generate a standardized YouTube playlist name for AI education content.

    Args:
        category: Category identifier from the AI curriculum.

    Returns:
        A playlist name string from the curriculum configuration.
    """
    curriculum = load_ai_curriculum()
    categories = curriculum.get("categories", {})
    category_data = categories.get(category, {})
    return category_data.get(
        "playlist_name",
        category.replace("_", " ").title(),
    )


def get_generated_ai_lessons(
    category: str = None,
    topic: str = None,
    difficulty: str = None,
    status: str = None,
    limit: int = 50,
) -> list[dict]:
    """
    Query generated AI education lessons with optional filters.

    Args:
        category: Filter by category, or None for all.
        topic: Filter by topic, or None for all.
        difficulty: Filter by difficulty ('beginner', 'intermediate', 'mixed'), or None for all.
        status: Filter by status ('generated', 'published'), or None for all.
        limit: Maximum number of results to return.

    Returns:
        List of lesson dictionaries from the ai_education_lessons table.
    """
    db = get_db()

    query = "SELECT * FROM ai_education_lessons WHERE 1=1"
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


def get_ai_lesson_progress() -> dict:
    """
    Get a summary of AI lesson generation progress across all categories.

    Returns:
        A dict with total/generated/remaining counts and per-category breakdowns.
    """
    curriculum = load_ai_curriculum()
    categories = curriculum.get("categories", {})

    db = get_db()
    rows = db.execute(
        "SELECT category, topic, lesson_title FROM ai_education_lessons"
    ).fetchall()
    db.close()

    generated_set = {(row["category"], row["topic"], row["lesson_title"]) for row in rows}

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

        for topic_entry in topics:
            topic_name = topic_entry.get("topic", "")
            lessons = topic_entry.get("lessons", [])

            for lesson_title in lessons:
                cat_total += 1
                if (cat_key, topic_name, lesson_title) in generated_set:
                    cat_generated += 1

        progress["categories"][cat_key] = {
            "display_name": cat_data.get("display_name", cat_key),
            "difficulty": cat_data.get("difficulty", "beginner"),
            "total": cat_total,
            "generated": cat_generated,
            "remaining": cat_total - cat_generated,
        }
        progress["total_lessons"] += cat_total
        progress["generated"] += cat_generated

    progress["remaining"] = progress["total_lessons"] - progress["generated"]

    return progress


def mark_ai_lesson_published(lesson_id: int, video_id: str) -> None:
    """
    Mark an AI education lesson as published with its YouTube video ID.

    Args:
        lesson_id: The ai_education_lessons table ID.
        video_id: The YouTube video ID after upload.
    """
    db = get_db()
    db.execute(
        "UPDATE ai_education_lessons SET status = 'published', video_id = ? WHERE id = ?",
        (video_id, lesson_id),
    )
    db.commit()
    db.close()


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage:")
        print("  python ai_education_generator.py next")
        print("  python ai_education_generator.py generate <category> <topic> <title>")
        print("  python ai_education_generator.py progress")
        sys.exit(1)

    command = sys.argv[1]

    if command == "next":
        next_lesson = pick_next_ai_lesson()
        if next_lesson is None:
            print("All lessons in the AI curriculum have been generated!")
        else:
            print(f"Next lesson to generate:")
            print(json.dumps(next_lesson, indent=2))
            print("\nGenerating lesson script...")
            script = generate_ai_lesson_script(**next_lesson)
            print(f"\nLesson generated successfully!")
            print(f"  Story ID: {script['story_id']}")
            print(f"  Lesson ID: {script['lesson_id']}")
            print(f"  Title: {script['title']}")
            print(f"  Category: {script['category']}")
            print(f"  Topic: {script['topic']}")
            print(f"  Scenes: {len(script.get('scenes', []))}")
            print(f"\nFull script:")
            print(json.dumps(script, indent=2))

    elif command == "generate":
        if len(sys.argv) < 5:
            print("Usage: python ai_education_generator.py generate <category> <topic> <title>")
            sys.exit(1)
        category = sys.argv[2]
        topic = sys.argv[3]
        title = sys.argv[4]

        script = generate_ai_lesson_script(category, topic, title)
        print(json.dumps(script, indent=2))

    elif command == "progress":
        progress = get_ai_lesson_progress()
        print(f"AI Education Lesson Progress")
        print(f"{'=' * 60}")
        print(f"Total lessons in curriculum: {progress['total_lessons']}")
        print(f"Generated: {progress['generated']}")
        print(f"Remaining: {progress['remaining']}")
        print()
        for cat_key, info in progress["categories"].items():
            pct = (info["generated"] / info["total"] * 100) if info["total"] > 0 else 0
            bar = "#" * int(pct / 5) + "-" * (20 - int(pct / 5))
            print(
                f"  {info['display_name']:42s} [{bar}] "
                f"{info['generated']:3d}/{info['total']:3d} ({pct:.0f}%)"
            )

    else:
        print(f"Unknown command: {command}")
        sys.exit(1)
