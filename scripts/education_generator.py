"""
Education Generator - Creates kid-friendly educational lesson scripts using LLM.

Generates structured lesson scripts for kids (Class 1-10) following
Oxford/Cambridge curriculum standards. Each lesson features "Professor Wisdom"
(a friendly cartoon owl) as the recurring teacher character.

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

# Age range mapping for each class level
CLASS_AGE_MAP = {
    1: {"age_range": "5-6", "complexity": "very simple", "vocab": "basic sight words and short sentences"},
    2: {"age_range": "6-7", "complexity": "simple", "vocab": "simple sentences with common words"},
    3: {"age_range": "7-8", "complexity": "moderate", "vocab": "clear sentences with some new vocabulary"},
    4: {"age_range": "8-9", "complexity": "intermediate", "vocab": "varied vocabulary with explanations"},
    5: {"age_range": "9-10", "complexity": "upper intermediate", "vocab": "rich vocabulary with context clues"},
    6: {"age_range": "10-11", "complexity": "advanced beginner", "vocab": "expanded vocabulary with subject-specific terminology"},
    7: {"age_range": "11-12", "complexity": "advanced", "vocab": "academic vocabulary with technical terms and definitions"},
    8: {"age_range": "12-13", "complexity": "upper advanced", "vocab": "formal and technical language with abstract concepts"},
    9: {"age_range": "13-14", "complexity": "pre-academic", "vocab": "sophisticated vocabulary with analytical and evaluative language"},
    10: {"age_range": "14-15", "complexity": "academic", "vocab": "exam-level language with precise terminology and critical analysis"},
}

# Visual metaphors for subjects to keep imagery consistent and kid-friendly
SUBJECT_VISUAL_THEMES = {
    "mathematics": {
        "colors": "bright blue, yellow, and red",
        "props": "colorful counting blocks, number cards, shape cutouts, abacus beads",
        "setting": "a cheerful classroom with a big colorful whiteboard",
    },
    "english": {
        "colors": "green, purple, and orange",
        "props": "giant letter balloons, storybooks, pencils, word cards",
        "setting": "a cozy library corner with bookshelves and colorful cushions",
    },
    "science": {
        "colors": "green, blue, and white",
        "props": "magnifying glass, test tubes with colorful liquids, planet models, leaf specimens",
        "setting": "a fun science lab with bubbling beakers and a big window showing nature",
    },
    "general_knowledge": {
        "colors": "rainbow colors",
        "props": "a spinning globe, colorful maps, picture flashcards, a magic telescope",
        "setting": "an adventure room with maps on the walls and a big globe in the center",
    },
}

# Professor Wisdom's teaching phrases for variety
PROFESSOR_PHRASES = [
    "Hoo-hoo! Let's learn something amazing today!",
    "Great question! Let me show you something wonderful!",
    "Hoo-hoo! You're doing fantastic!",
    "Now THIS is going to blow your mind!",
    "Are you ready for something super cool?",
    "Wonderful thinking! Let's explore this together!",
    "Hoo-hoo! That's the spirit of a great learner!",
]

# Quiz scene templates
QUIZ_TEMPLATES = {
    "counting": "Can you count along with me? Ready? {sequence}",
    "identify": "Can you spot the {target}? Point at the screen! ... Yes! That's right!",
    "recall": "Do you remember what we just learned? {question} ... Exactly! {answer}!",
    "challenge": "Here's a challenge for you! {challenge} Think about it... {hint} ... The answer is {answer}!",
    "interactive": "Let's try this together! {instruction} ... Hoo-hoo! You got it!",
}


def get_db():
    """Get or create the stories database with education_lessons table."""
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

    # Create education_lessons table for tracking lesson-specific metadata
    conn.execute("""
        CREATE TABLE IF NOT EXISTS education_lessons (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            class_level INTEGER NOT NULL,
            subject TEXT NOT NULL,
            topic TEXT NOT NULL,
            lesson_title TEXT NOT NULL,
            curriculum TEXT DEFAULT 'oxford',
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


def load_education_syllabus() -> dict:
    """Load the education syllabus configuration."""
    syllabus_path = CONFIG_DIR / "education_syllabus.json"
    if not syllabus_path.exists():
        raise FileNotFoundError(
            f"Education syllabus not found at {syllabus_path}. "
            "Please create config/education_syllabus.json first."
        )
    with open(syllabus_path) as f:
        return json.load(f)


def _get_age_context(class_level: int) -> dict:
    """Get age-appropriate context for prompt generation."""
    return CLASS_AGE_MAP.get(class_level, CLASS_AGE_MAP[3])


def _get_visual_theme(subject: str) -> dict:
    """Get visual theme configuration for a subject."""
    return SUBJECT_VISUAL_THEMES.get(subject, SUBJECT_VISUAL_THEMES["general_knowledge"])


def build_lesson_prompt(
    class_level: int,
    subject: str,
    topic: str,
    lesson_title: str,
    curriculum: str = "oxford",
) -> str:
    """
    Build an LLM prompt to generate an educational lesson script.

    Creates a detailed prompt that instructs the LLM to generate a structured
    JSON lesson featuring Professor Wisdom (a cartoon owl) teaching the given
    topic in an engaging, age-appropriate way.

    Args:
        class_level: Student class level (1-10).
        subject: Subject area (mathematics, english, science, general_knowledge).
        topic: Specific topic identifier from the syllabus.
        lesson_title: Human-readable lesson title.
        curriculum: Curriculum standard ('oxford' or 'cambridge').

    Returns:
        A formatted prompt string ready for LLM generation.
    """
    age_ctx = _get_age_context(class_level)
    visual_theme = _get_visual_theme(subject)
    syllabus = load_education_syllabus()
    professor = syllabus.get("professor_wisdom", {})

    professor_desc = professor.get(
        "appearance",
        "a friendly cartoon owl with big round glasses, a graduation cap, and a colorful bow tie",
    )
    professor_keywords = professor.get(
        "visual_keywords",
        "cartoon owl professor glasses graduation cap bow tie friendly",
    )

    curriculum_name = "Oxford International" if curriculum == "oxford" else "Cambridge International"

    return f"""You are an expert children's educational content creator making a video lesson for a YouTube kids channel called "Little Wisdom Tales".

Generate an educational lesson script based on these parameters:
- Class Level: Class {class_level} (ages {age_ctx['age_range']})
- Subject: {subject.replace('_', ' ').title()}
- Topic: {topic.replace('_', ' ').title()}
- Lesson Title: {lesson_title}
- Curriculum: {curriculum_name} Primary Programme
- Language Complexity: {age_ctx['complexity']} ({age_ctx['vocab']})

MAIN CHARACTER:
- Name: Professor Wisdom
- Appearance: {professor_desc}
- Professor Wisdom appears in EVERY scene as the teacher
- He is warm, encouraging, patient, and uses the catchphrase "Hoo-hoo!"

VISUAL STYLE:
- Setting: {visual_theme['setting']}
- Color palette: {visual_theme['colors']}
- Props and tools: {visual_theme['props']}
- Style: Bright, cartoon, colorful, kid-friendly, educational
- Every visual_description MUST include Professor Wisdom (the owl) doing something specific

LESSON STRUCTURE (16-22 scenes, each ~15-20 seconds of narration, total 5-8 minutes):
1. INTRO (1-2 scenes): Professor Wisdom welcomes kids, introduces today's topic with excitement
2. CONCEPT EXPLANATION (4-6 scenes): Break down the concept into simple steps with visual metaphors
3. EXAMPLE/DEMONSTRATION (4-5 scenes): Show real examples with colorful visuals and step-by-step walkthrough
4. PRACTICE TOGETHER (2-3 scenes): Guide kids through solving a problem or doing an activity together
5. QUIZ/CHALLENGE (2-3 scenes): Interactive moments - ask kids to count along, identify something, or answer questions (pause for them to think)
6. REAL WORLD CONNECTION (1-2 scenes): Show how this topic appears in everyday life
7. RECAP (1 scene): Summarize what was learned with key takeaways
8. ENCOURAGEMENT (1 scene): Professor Wisdom praises them and teases the next lesson

IMPORTANT RULES:
- WRITE EVERYTHING IN ENGLISH ONLY
- Use {age_ctx['complexity']} language appropriate for ages {age_ctx['age_range']}
- Make it FUN and ENGAGING - use sound effects like "WHOOSH!", "DING DING!", "TA-DA!"
- Include moments where kids participate ("Can you count with me?", "Point to the screen!")
- Use concrete visual metaphors (counting with blocks, letters on balloons, planets on strings)
- No scary content, no complex abstract concepts without visual aids
- Each scene MUST have Professor Wisdom doing something visible and specific
- The visual_description should paint a vivid cartoon scene with specific objects and colors
- Quiz scenes should have the question, a pause phrase, and the answer in the narration

Return a JSON object with this EXACT structure (no markdown, just raw JSON):
{{
  "title": "{lesson_title}",
  "lesson_topic": "{topic}",
  "class_level": {class_level},
  "subject": "{subject}",
  "curriculum": "{curriculum}",
  "moral": "Learning is fun and everyone can do it!",
  "duration_estimate": "5-8 minutes",
  "thumbnail_description": "Professor Wisdom the cartoon owl with big glasses standing next to a colorful display about {topic.replace('_', ' ')} with {visual_theme['colors']} background",
  "scenes": [
    {{
      "scene_number": 1,
      "narration": "Hoo-hoo! Hello friends! I'm Professor Wisdom, and today we're going to learn something SUPER exciting!",
      "visual_description": "Professor Wisdom, {professor_desc}, waving happily in {visual_theme['setting']} with a big colorful banner reading the lesson title",
      "image_search_terms": "{professor_keywords} {visual_theme['colors']} classroom welcome kids education cartoon",
      "character_speaking": "professor_wisdom",
      "duration_seconds": 15,
      "scene_type": "intro"
    }},
    {{
      "scene_number": 2,
      "narration": "Concept explanation with examples...",
      "visual_description": "Professor Wisdom pointing at specific visual aids related to the topic...",
      "image_search_terms": "relevant search terms for the concept",
      "character_speaking": "professor_wisdom",
      "duration_seconds": 15,
      "scene_type": "explanation"
    }},
    {{
      "scene_number": 3,
      "narration": "A quiz or interactive moment...",
      "visual_description": "Professor Wisdom holding up a question card with visual aids...",
      "image_search_terms": "relevant quiz visual terms",
      "character_speaking": "professor_wisdom",
      "duration_seconds": 15,
      "scene_type": "quiz"
    }}
  ],
  "tags": ["kids education", "class {class_level}", "{subject.replace('_', ' ')}", "{topic.replace('_', ' ')}", "{curriculum} curriculum", "professor wisdom", "learn with owl", "kids learning"],
  "description": "Join Professor Wisdom the owl as he teaches {lesson_title} in this fun educational video for Class {class_level} kids!"
}}

Create 8-12 scenes total. Include at least 1-2 quiz/challenge scenes where kids participate.
Each scene must have a unique, detailed visual_description with Professor Wisdom doing something specific.
The visual descriptions should use simple, concrete imagery: colorful blocks, balloons, drawings on whiteboards, etc.
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


def _enrich_lesson_script(
    lesson_script: dict,
    class_level: int,
    subject: str,
    topic: str,
    lesson_title: str,
    curriculum: str,
) -> dict:
    """
    Enrich the raw LLM output with metadata needed by the pipeline.

    Adds fields that the video pipeline expects (collection, region, age_range, etc.)
    and ensures the lesson script is fully compatible with the story format.

    Args:
        lesson_script: The parsed lesson script from the LLM.
        class_level: Student class level (1-10).
        subject: Subject area identifier.
        topic: Topic identifier.
        lesson_title: Human-readable lesson title.
        curriculum: Curriculum standard.

    Returns:
        Enriched lesson script dictionary.
    """
    age_ctx = _get_age_context(class_level)

    # Ensure required fields exist with sensible defaults
    lesson_script.setdefault("title", lesson_title)
    lesson_script.setdefault("moral", "Learning is fun and everyone can do it!")
    lesson_script.setdefault("duration_estimate", "2-3 minutes")

    # Add education-specific metadata
    lesson_script["type"] = "education"
    lesson_script["class_level"] = class_level
    lesson_script["subject"] = subject
    lesson_script["topic"] = topic
    lesson_script["lesson_title"] = lesson_title
    lesson_script["curriculum"] = curriculum

    # Add fields expected by the rest of the pipeline (story compatibility)
    lesson_script["collection"] = f"education_class_{class_level}"
    lesson_script["origin"] = f"{curriculum.title()} Curriculum"
    lesson_script["region"] = "en-US"
    lesson_script["age_range"] = age_ctx["age_range"]
    lesson_script["generated_at"] = datetime.now().isoformat()

    # Ensure all scenes have required fields for the video pipeline
    for scene in lesson_script.get("scenes", []):
        scene.setdefault("character_speaking", "professor_wisdom")
        scene.setdefault("duration_seconds", 15)
        scene.setdefault("scene_type", "explanation")
        # Ensure image_search_terms includes Professor Wisdom for SDXL consistency
        search_terms = scene.get("image_search_terms", "")
        if "owl" not in search_terms.lower():
            scene["image_search_terms"] = f"cartoon owl professor {search_terms}"

    # Generate a thumbnail description if missing
    visual_theme = _get_visual_theme(subject)
    lesson_script.setdefault(
        "thumbnail_description",
        f"Professor Wisdom the friendly cartoon owl with glasses and graduation cap "
        f"teaching {lesson_title} with {visual_theme['props']} "
        f"on a bright {visual_theme['colors']} background",
    )

    # Generate tags if missing
    default_tags = [
        "kids education",
        f"class {class_level}",
        subject.replace("_", " "),
        topic.replace("_", " "),
        f"{curriculum} curriculum",
        "professor wisdom",
        "learn with owl",
        "kids learning",
        "educational video",
        "Little Wisdom Tales",
    ]
    lesson_script.setdefault("tags", default_tags)

    # Generate description if missing
    lesson_script.setdefault(
        "description",
        f"Join Professor Wisdom the owl as he teaches {lesson_title}! "
        f"A fun educational video for Class {class_level} kids "
        f"following the {curriculum.title()} curriculum. "
        f"Learn {topic.replace('_', ' ')} with colorful animations and interactive quizzes!",
    )

    return lesson_script


def generate_lesson_script(
    class_level: int,
    subject: str,
    topic: str,
    lesson_title: str,
    curriculum: str = "oxford",
) -> dict:
    """
    Generate a complete educational lesson script ready for video production.

    Calls the local Ollama LLM to generate a structured lesson script featuring
    Professor Wisdom, then enriches it with metadata and stores it in the
    SQLite database. The returned dict is fully compatible with the existing
    video pipeline (same format as story_generator output).

    Args:
        class_level: Student class level (1-10).
        subject: Subject area (mathematics, english, science, general_knowledge).
        topic: Specific topic identifier from the syllabus.
        lesson_title: Human-readable lesson title.
        curriculum: Curriculum standard, either 'oxford' or 'cambridge'.

    Returns:
        A story_script-compatible dict containing scenes, metadata, and a story_id.

    Raises:
        ValueError: If the LLM fails to produce valid JSON after retries.
    """
    prompt = build_lesson_prompt(class_level, subject, topic, lesson_title, curriculum)

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
    lesson_script = _enrich_lesson_script(
        lesson_script, class_level, subject, topic, lesson_title, curriculum
    )

    # Save to the main stories table (pipeline compatibility)
    db = get_db()
    db.execute(
        """INSERT INTO stories (title, collection, region, moral, script, status)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (
            lesson_script.get("title", lesson_title),
            f"education_class_{class_level}",
            "en-US",
            lesson_script.get("moral", "Learning is fun and everyone can do it!"),
            json.dumps(lesson_script),
            "generated",
        ),
    )
    db.commit()
    story_id = db.execute("SELECT last_insert_rowid()").fetchone()[0]
    lesson_script["story_id"] = story_id

    # Save to the education_lessons table (lesson-specific tracking)
    db.execute(
        """INSERT INTO education_lessons
           (class_level, subject, topic, lesson_title, curriculum, script, story_id, status)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            class_level,
            subject,
            topic,
            lesson_title,
            curriculum,
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


def pick_next_lesson(curriculum: str = "oxford") -> dict | None:
    """
    Pick the next lesson to generate based on the syllabus and past generation history.

    Reads config/education_syllabus.json for available lessons, checks the
    database for already-generated lessons, and returns the next lesson
    that hasn't been created yet. Cycles through subjects to provide variety
    (not all math in a row).

    Args:
        curriculum: Curriculum standard to use ('oxford' or 'cambridge').

    Returns:
        A dict with keys: class_level, subject, topic, lesson_title, curriculum.
        Returns None if all lessons in the syllabus have been generated.
    """
    syllabus = load_education_syllabus()
    subjects = syllabus.get("subjects", {})
    rotation_order = syllabus.get("subject_rotation_order", list(subjects.keys()))

    # Get all already-generated lesson topics from the database
    db = get_db()
    rows = db.execute(
        "SELECT class_level, subject, topic, curriculum FROM education_lessons WHERE curriculum = ?",
        (curriculum,),
    ).fetchall()
    db.close()

    generated_set = {(row["class_level"], row["subject"], row["topic"]) for row in rows}

    # Determine the last generated subject to continue rotation
    last_subject = None
    last_class = None
    if rows:
        # Get the most recently generated lesson
        db = get_db()
        last_row = db.execute(
            """SELECT class_level, subject FROM education_lessons
               WHERE curriculum = ? ORDER BY created_at DESC LIMIT 1""",
            (curriculum,),
        ).fetchone()
        db.close()
        if last_row:
            last_subject = last_row["subject"]
            last_class = last_row["class_level"]

    # Build a prioritized list of candidate lessons:
    # 1. Rotate subjects (if last was math, next should be science, etc.)
    # 2. Within each subject, go class 1 -> 2 -> 3 -> 4 -> 5
    # 3. Within each class, go in syllabus order (first ungenerated topic)

    # Determine subject rotation starting point
    if last_subject and last_subject in rotation_order:
        last_idx = rotation_order.index(last_subject)
        rotated_subjects = rotation_order[last_idx + 1 :] + rotation_order[: last_idx + 1]
    else:
        rotated_subjects = list(rotation_order)

    # Determine class level rotation starting point
    class_levels = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    if last_class is not None:
        # Start from the same class level as last time, so we spread evenly
        last_cls_idx = class_levels.index(last_class) if last_class in class_levels else 0
        rotated_classes = class_levels[last_cls_idx:] + class_levels[:last_cls_idx]
    else:
        rotated_classes = list(class_levels)

    # Find the next ungenerated lesson
    for subject_key in rotated_subjects:
        subject_data = subjects.get(subject_key, {})
        lessons_by_class = subject_data.get("lessons", {})

        for class_level in rotated_classes:
            class_key = str(class_level)
            lessons = lessons_by_class.get(class_key, [])

            for lesson in lessons:
                topic = lesson["topic"]
                title = lesson["title"]

                if (class_level, subject_key, topic) not in generated_set:
                    return {
                        "class_level": class_level,
                        "subject": subject_key,
                        "topic": topic,
                        "lesson_title": title,
                        "curriculum": curriculum,
                    }

    # All lessons exhausted for this cycle - start a new cycle with fresh angles.
    # Count how many full cycles we've done by dividing generated lessons by total.
    total_lessons_in_syllabus = sum(
        len(lesson_list)
        for subject_data in subjects.values()
        for class_lessons in subject_data.get("lessons", {}).values()
        for lesson_list in [class_lessons]
    )
    if total_lessons_in_syllabus == 0:
        return None

    cycle = max(1, len(generated_set) // max(total_lessons_in_syllabus, 1))

    # Variation suffixes to keep content fresh across cycles
    cycle_suffixes = [
        "Part 2: Deep Dive",
        "Part 3: Fun Review",
        "Part 4: Practice Time",
        "Part 5: Challenge Mode",
        "Part 6: Quick Quiz",
        "Part 7: Real World",
        "Part 8: Story Time",
        "Part 9: Experiment",
        "Part 10: Master Class",
    ]
    suffix = cycle_suffixes[(cycle - 1) % len(cycle_suffixes)]

    # Pick a lesson that hasn't been done in THIS cycle
    for subject_key in rotated_subjects:
        subject_data = subjects.get(subject_key, {})
        lessons_by_class = subject_data.get("lessons", {})
        for class_level in rotated_classes:
            class_key = str(class_level)
            lessons = lessons_by_class.get(class_key, [])
            for lesson in lessons:
                topic = lesson["topic"]
                cycle_topic = f"{topic} - {suffix}"
                if (class_level, subject_key, cycle_topic) not in generated_set:
                    return {
                        "class_level": class_level,
                        "subject": subject_key,
                        "topic": cycle_topic,
                        "lesson_title": f"{lesson['title']} - {suffix}",
                        "curriculum": curriculum,
                    }

    # Absolute fallback - generate from random position
    import random
    all_subjects = list(subjects.keys())
    sub = random.choice(all_subjects)
    cls = random.choice(class_levels)
    return {
        "class_level": cls,
        "subject": sub,
        "topic": f"Review: {sub.replace('_', ' ').title()} Class {cls} - Cycle {cycle + 1}",
        "lesson_title": f"Fun Review: {sub.replace('_', ' ').title()} for Class {cls}",
        "curriculum": curriculum,
    }


def get_education_playlist_name(
    class_level: int,
    subject: str,
    curriculum: str = "oxford",
) -> str:
    """
    Generate a standardized YouTube playlist name for educational content.

    Args:
        class_level: Student class level (1-10).
        subject: Subject area identifier.
        curriculum: Curriculum standard.

    Returns:
        A playlist name string like "Class 1 Mathematics - Oxford Curriculum".
    """
    syllabus = load_education_syllabus()
    subjects = syllabus.get("subjects", {})
    subject_info = subjects.get(subject, {})
    display_name = subject_info.get("display_name", subject.replace("_", " ").title())
    curriculum_display = curriculum.title()

    return f"Class {class_level} {display_name} - {curriculum_display} Curriculum"


def get_generated_lessons(
    class_level: int = None,
    subject: str = None,
    curriculum: str = None,
    status: str = None,
    limit: int = 50,
) -> list[dict]:
    """
    Query generated education lessons with optional filters.

    Args:
        class_level: Filter by class level (1-10), or None for all.
        subject: Filter by subject, or None for all.
        curriculum: Filter by curriculum, or None for all.
        status: Filter by status ('generated', 'published'), or None for all.
        limit: Maximum number of results to return.

    Returns:
        List of lesson dictionaries from the education_lessons table.
    """
    db = get_db()

    query = "SELECT * FROM education_lessons WHERE 1=1"
    params = []

    if class_level is not None:
        query += " AND class_level = ?"
        params.append(class_level)
    if subject is not None:
        query += " AND subject = ?"
        params.append(subject)
    if curriculum is not None:
        query += " AND curriculum = ?"
        params.append(curriculum)
    if status is not None:
        query += " AND status = ?"
        params.append(status)

    query += " ORDER BY created_at DESC LIMIT ?"
    params.append(limit)

    rows = db.execute(query, params).fetchall()
    db.close()

    return [dict(row) for row in rows]


def get_lesson_progress(curriculum: str = "oxford") -> dict:
    """
    Get a summary of lesson generation progress across all subjects and classes.

    Args:
        curriculum: Curriculum standard to check progress for.

    Returns:
        A dict with total/generated/remaining counts and per-subject breakdowns.
    """
    syllabus = load_education_syllabus()
    subjects = syllabus.get("subjects", {})

    db = get_db()
    rows = db.execute(
        "SELECT class_level, subject, topic FROM education_lessons WHERE curriculum = ?",
        (curriculum,),
    ).fetchall()
    db.close()

    generated_set = {(row["class_level"], row["subject"], row["topic"]) for row in rows}

    progress = {
        "curriculum": curriculum,
        "total_lessons": 0,
        "generated": 0,
        "remaining": 0,
        "subjects": {},
    }

    for subject_key, subject_data in subjects.items():
        lessons_by_class = subject_data.get("lessons", {})
        subject_total = 0
        subject_generated = 0

        for class_key, lessons in lessons_by_class.items():
            class_level = int(class_key)
            for lesson in lessons:
                subject_total += 1
                if (class_level, subject_key, lesson["topic"]) in generated_set:
                    subject_generated += 1

        progress["subjects"][subject_key] = {
            "display_name": subject_data.get("display_name", subject_key),
            "total": subject_total,
            "generated": subject_generated,
            "remaining": subject_total - subject_generated,
        }
        progress["total_lessons"] += subject_total
        progress["generated"] += subject_generated

    progress["remaining"] = progress["total_lessons"] - progress["generated"]

    return progress


def mark_lesson_published(lesson_id: int, video_id: str) -> None:
    """
    Mark an education lesson as published with its YouTube video ID.

    Args:
        lesson_id: The education_lessons table ID.
        video_id: The YouTube video ID after upload.
    """
    db = get_db()
    db.execute(
        "UPDATE education_lessons SET status = 'published', video_id = ? WHERE id = ?",
        (video_id, lesson_id),
    )
    db.commit()
    db.close()


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage:")
        print("  python education_generator.py next [--curriculum oxford|cambridge]")
        print("  python education_generator.py generate <class> <subject> <topic> <title> [curriculum]")
        print("  python education_generator.py progress [--curriculum oxford|cambridge]")
        print("  python education_generator.py list [--class N] [--subject S] [--limit N]")
        sys.exit(1)

    command = sys.argv[1]

    if command == "next":
        curriculum = "oxford"
        if "--curriculum" in sys.argv:
            idx = sys.argv.index("--curriculum")
            if idx + 1 < len(sys.argv):
                curriculum = sys.argv[idx + 1]

        next_lesson = pick_next_lesson(curriculum=curriculum)
        if next_lesson is None:
            print("All lessons in the syllabus have been generated!")
        else:
            print(f"Next lesson to generate:")
            print(json.dumps(next_lesson, indent=2))
            print("\nGenerating lesson script...")
            script = generate_lesson_script(**next_lesson)
            print(f"\nLesson generated successfully!")
            print(f"  Story ID: {script['story_id']}")
            print(f"  Lesson ID: {script['lesson_id']}")
            print(f"  Title: {script['title']}")
            print(f"  Scenes: {len(script.get('scenes', []))}")
            print(f"\nFull script:")
            print(json.dumps(script, indent=2))

    elif command == "generate":
        if len(sys.argv) < 6:
            print("Usage: python education_generator.py generate <class> <subject> <topic> <title> [curriculum]")
            sys.exit(1)
        class_level = int(sys.argv[2])
        subject = sys.argv[3]
        topic = sys.argv[4]
        title = sys.argv[5]
        curriculum = sys.argv[6] if len(sys.argv) > 6 else "oxford"

        script = generate_lesson_script(class_level, subject, topic, title, curriculum)
        print(json.dumps(script, indent=2))

    elif command == "progress":
        curriculum = "oxford"
        if "--curriculum" in sys.argv:
            idx = sys.argv.index("--curriculum")
            if idx + 1 < len(sys.argv):
                curriculum = sys.argv[idx + 1]

        progress = get_lesson_progress(curriculum=curriculum)
        print(f"Education Lesson Progress ({curriculum.title()} Curriculum)")
        print(f"{'=' * 50}")
        print(f"Total lessons in syllabus: {progress['total_lessons']}")
        print(f"Generated: {progress['generated']}")
        print(f"Remaining: {progress['remaining']}")
        print()
        for subject_key, info in progress["subjects"].items():
            pct = (info["generated"] / info["total"] * 100) if info["total"] > 0 else 0
            bar = "#" * int(pct / 5) + "-" * (20 - int(pct / 5))
            print(f"  {info['display_name']:20s} [{bar}] {info['generated']:3d}/{info['total']:3d} ({pct:.0f}%)")

    elif command == "list":
        kwargs = {}
        if "--class" in sys.argv:
            idx = sys.argv.index("--class")
            if idx + 1 < len(sys.argv):
                kwargs["class_level"] = int(sys.argv[idx + 1])
        if "--subject" in sys.argv:
            idx = sys.argv.index("--subject")
            if idx + 1 < len(sys.argv):
                kwargs["subject"] = sys.argv[idx + 1]
        if "--limit" in sys.argv:
            idx = sys.argv.index("--limit")
            if idx + 1 < len(sys.argv):
                kwargs["limit"] = int(sys.argv[idx + 1])

        lessons = get_generated_lessons(**kwargs)
        if not lessons:
            print("No lessons found matching the criteria.")
        else:
            print(f"Found {len(lessons)} lesson(s):")
            for lesson in lessons:
                print(
                    f"  [{lesson['id']}] Class {lesson['class_level']} "
                    f"{lesson['subject']} - {lesson['lesson_title']} "
                    f"({lesson['status']}) [{lesson['curriculum']}]"
                )

    else:
        print(f"Unknown command: {command}")
        sys.exit(1)
