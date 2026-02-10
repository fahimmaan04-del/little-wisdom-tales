"""
AI Content Generator - Creates educational AI lessons for kids.

Generates structured lesson scripts for the "Little Wisdom AI Lab" YouTube
channel, teaching kids (ages 8-14) how to use AI to build apps, games,
and creative projects.

Uses Ollama (local LLM) to generate lesson scripts with scene breakdowns,
featuring "Byte the Robot" as the host character.
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

# Import shared utilities from story_generator
from story_generator import call_ollama, parse_llm_json


def get_db():
    """Get or create the stories database with ai_education support."""
    db_path = DATA_DIR / "stories.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    # Ensure stories table exists with content_type column
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

    # Add content_type column if it does not exist yet
    try:
        conn.execute("ALTER TABLE stories ADD COLUMN content_type TEXT DEFAULT 'story'")
        conn.commit()
    except sqlite3.OperationalError:
        # Column already exists
        pass

    # Add category and lesson_key columns for tracking generated lessons
    try:
        conn.execute("ALTER TABLE stories ADD COLUMN category TEXT")
        conn.commit()
    except sqlite3.OperationalError:
        pass

    try:
        conn.execute("ALTER TABLE stories ADD COLUMN lesson_key TEXT")
        conn.commit()
    except sqlite3.OperationalError:
        pass

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


def load_curriculum():
    """Load the AI kids curriculum from configuration."""
    with open(CONFIG_DIR / "ai_kids_curriculum.json") as f:
        return json.load(f)


def build_ai_lesson_prompt(category: str, topic: str, lesson_title: str) -> str:
    """Build the LLM prompt for AI education content.

    Creates a detailed prompt that instructs the LLM to generate a structured
    lesson script featuring Byte the Robot as host, with screen-recording
    style scenes, hands-on challenges, and a recap segment.

    Args:
        category: The curriculum category key (e.g., 'ai_basics', 'scratch_ai').
        topic: The topic name within the category.
        lesson_title: The specific lesson title to generate content for.

    Returns:
        A formatted prompt string ready for the LLM.
    """
    curriculum = load_curriculum()
    cat_data = curriculum["categories"].get(category, {})
    age_range = cat_data.get("age_range", "8-14")
    difficulty = cat_data.get("difficulty", "beginner")
    display_name = cat_data.get("display_name", category)

    # Gather approved tools relevant to this category
    approved_tools = curriculum.get("approved_tools", [])
    tool_names = [t["name"] for t in approved_tools]

    return f"""You are an educational content creator for a YouTube channel called "Little Wisdom AI Lab" that teaches kids how to use AI.

Your host character is "Byte the Robot" - a friendly blue robot kid with glowing green eyes, an antenna on his head, and a tablet in his hand. Byte is enthusiastic, encouraging, and explains things with fun analogies.

Generate a lesson script for this specific lesson:
- Category: {display_name}
- Topic: {topic}
- Lesson Title: {lesson_title}
- Difficulty: {difficulty}
- Target Age: {age_range} years old

CONTENT RULES:
- WRITE EVERYTHING IN ENGLISH
- Byte speaks directly to the viewer like a fun older friend
- Use simple, clear explanations with real-world analogies kids understand
- Reference real tools kids can use: {', '.join(tool_names)}
- Include at least ONE "Try It Yourself!" hands-on challenge
- End with a "What We Learned Today" recap scene
- Start with an exciting hook that grabs attention in the first 5 seconds
- Include screen-recording style scenes showing AI tools in action
- Each scene should teach ONE clear micro-concept
- Add fun facts, jokes, or surprising AI trivia between teaching sections
- NO scary content, NO violence, NO mature themes
- Remind kids to ask a parent before creating accounts on websites
- The total lesson should be 5-7 minutes when narrated (about 800-1200 words of narration)

VISUAL DESCRIPTION GUIDELINES:
- Describe Byte the Robot in various poses: pointing at a screen, celebrating, thinking, typing
- Include computer screen mockups showing: coding interfaces, AI art being created, game characters, terminal windows
- Use bright, colorful cartoon style: blue, green, orange, yellow color palette
- Show step-by-step visuals: numbered lists, diagrams, flowcharts in kid-friendly style
- Include before/after comparisons when showing AI transformations
- Scene backgrounds should alternate between: Byte's lab (futuristic kid-friendly workspace), the AI tools in action, animated diagrams

Return a JSON object with this EXACT structure (no markdown, just raw JSON):
{{
  "title": "{lesson_title}",
  "category": "{category}",
  "topic": "{topic}",
  "age_range": "{age_range}",
  "difficulty": "{difficulty}",
  "learning_objective": "One clear sentence describing what kids will learn",
  "tools_featured": ["List of specific tools used in this lesson"],
  "duration_estimate": "5-7 minutes",
  "thumbnail_description": "A vivid, colorful description for the thumbnail showing Byte and the lesson topic",
  "scenes": [
    {{
      "scene_number": 1,
      "scene_type": "intro_hook",
      "narration": "Byte's opening hook - exciting question or amazing AI fact (2-3 sentences)",
      "visual_description": "Byte the Robot waving at the camera in his futuristic lab with holographic screens showing [topic-related visuals]",
      "image_search_terms": "cartoon robot kid futuristic lab holographic screens colorful illustration",
      "on_screen_text": "Text overlay shown on screen (lesson title, key terms, etc.)",
      "duration_seconds": 15
    }},
    {{
      "scene_number": 2,
      "scene_type": "concept",
      "narration": "Teaching narration explaining the concept (3-4 sentences)",
      "visual_description": "Description of the teaching visual - screen mockup, diagram, or animated example",
      "image_search_terms": "search terms for finding relevant illustration",
      "on_screen_text": "Key term or definition shown on screen",
      "duration_seconds": 20
    }},
    {{
      "scene_number": 3,
      "scene_type": "example",
      "narration": "Real-world example or analogy (2-3 sentences)",
      "visual_description": "Visual showing the real-world example",
      "image_search_terms": "search terms for example visual",
      "on_screen_text": null,
      "duration_seconds": 15
    }},
    {{
      "scene_number": 4,
      "scene_type": "demo",
      "narration": "Step-by-step demonstration narration (3-4 sentences)",
      "visual_description": "Screen recording style visual showing the tool in action with highlighted buttons and menus",
      "image_search_terms": "computer screen coding interface colorful kids tutorial",
      "on_screen_text": "Step 1: ...",
      "duration_seconds": 25
    }},
    {{
      "scene_number": 5,
      "scene_type": "fun_fact",
      "narration": "An amazing AI fun fact or joke to keep engagement (1-2 sentences)",
      "visual_description": "Byte looking surprised or amazed with a fun fact displayed",
      "image_search_terms": "cartoon robot surprised fun fact colorful illustration",
      "on_screen_text": "Fun Fact: ...",
      "duration_seconds": 10
    }},
    {{
      "scene_number": 6,
      "scene_type": "challenge",
      "narration": "Try It Yourself! challenge instructions (3-4 sentences)",
      "visual_description": "Byte pointing at a checklist or challenge card with bright colors and stars",
      "image_search_terms": "cartoon challenge card checklist stars colorful kids",
      "on_screen_text": "TRY IT YOURSELF! [challenge description]",
      "duration_seconds": 20
    }},
    {{
      "scene_number": 7,
      "scene_type": "recap",
      "narration": "What We Learned Today recap - summarize 3 key points (2-3 sentences)",
      "visual_description": "Byte next to a board showing 3 key takeaways with checkmarks",
      "image_search_terms": "cartoon robot checklist key points summary colorful illustration",
      "on_screen_text": "What We Learned: 1. ... 2. ... 3. ...",
      "duration_seconds": 15
    }},
    {{
      "scene_number": 8,
      "scene_type": "outro",
      "narration": "Byte's sign-off: tease next lesson, encourage viewers (2 sentences)",
      "visual_description": "Byte waving goodbye with 'See you next time!' and channel logo",
      "image_search_terms": "cartoon robot waving goodbye colorful illustration kids",
      "on_screen_text": "See You Next Time in the AI Lab!",
      "duration_seconds": 10
    }}
  ],
  "tags": ["ai for kids", "learn ai", "{topic}", "{category}", "kids coding", "stem education", "byte the robot"],
  "description": "A YouTube-friendly description (3-4 sentences) with SEO keywords about this AI lesson for kids"
}}

Create 15-20 scenes total. The example above shows the required scene types - expand the concept, demo, and example sections to fill 15-20 scenes. Include multiple concept/demo/example scenes between the intro and the challenge/recap/outro.

Scene types to use:
- "intro_hook" (1 scene) - Exciting opening
- "concept" (4-6 scenes) - Teaching new ideas
- "example" (2-3 scenes) - Real-world analogies and examples
- "demo" (3-4 scenes) - Step-by-step tool demonstrations
- "fun_fact" (2 scenes) - Engagement boosters
- "challenge" (1 scene) - Try It Yourself activity
- "recap" (1 scene) - What We Learned Today
- "outro" (1 scene) - Sign-off and next lesson tease

Return ONLY the JSON, no other text."""


def generate_ai_lesson_script(
    category: str,
    topic: str,
    lesson_title: str,
) -> dict:
    """Generate a complete AI lesson script via Ollama and store in DB.

    Calls the local LLM with a structured prompt, parses the JSON response,
    enriches it with metadata, and saves it to the stories database with
    content_type='ai_education'.

    Args:
        category: The curriculum category key.
        topic: The topic name within the category.
        lesson_title: The specific lesson title.

    Returns:
        The complete lesson script dict including the assigned story_id.

    Raises:
        ValueError: If the LLM fails to produce valid JSON after retries.
    """
    prompt = build_ai_lesson_prompt(category, topic, lesson_title)

    lesson_script = None
    last_error = None
    import time as _time

    for attempt in range(3):
        try:
            raw_response = call_ollama(prompt)
            lesson_script = parse_llm_json(raw_response)

            # Validate required fields
            if "scenes" not in lesson_script or not lesson_script["scenes"]:
                raise ValueError("Lesson has no scenes")
            if "title" not in lesson_script:
                raise ValueError("Lesson has no title")
            if len(lesson_script["scenes"]) < 8:
                raise ValueError(
                    f"Lesson has only {len(lesson_script['scenes'])} scenes, "
                    "need at least 8"
                )
            break
        except (ValueError, KeyError) as e:
            last_error = e
            print(f"JSON parse attempt {attempt + 1} failed: {e}, retrying...")
            _time.sleep(5)

    if lesson_script is None:
        raise ValueError(
            f"Failed to generate valid lesson after 3 attempts: {last_error}"
        )

    # Enrich with metadata
    lesson_script["category"] = category
    lesson_script["topic"] = topic
    lesson_script["content_type"] = "ai_education"
    lesson_script["generated_at"] = datetime.now().isoformat()
    lesson_script["host"] = "Byte the Robot"

    # Build a unique key for tracking which lessons have been generated
    lesson_key = f"{category}::{topic}::{lesson_title}"

    # Save to database
    db = get_db()
    db.execute(
        """INSERT INTO stories
           (title, collection, region, moral, script, status, content_type, category, lesson_key)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            lesson_script.get("title", lesson_title),
            category,
            "en-US",
            lesson_script.get("learning_objective", ""),
            json.dumps(lesson_script),
            "generated",
            "ai_education",
            category,
            lesson_key,
        ),
    )
    db.commit()
    story_id = db.execute("SELECT last_insert_rowid()").fetchone()[0]
    lesson_script["story_id"] = story_id
    db.close()

    print(f"Generated lesson #{story_id}: {lesson_title}")
    return lesson_script


def pick_next_ai_lesson() -> dict | None:
    """Pick the next un-generated lesson from the curriculum.

    Scans all categories and topics in playlist order, returning the first
    lesson that has not yet been generated (no matching lesson_key in DB).

    Returns:
        A dict with 'category', 'topic', and 'lesson_title' keys for the
        next lesson to generate, or None if all lessons are done.
    """
    curriculum = load_curriculum()
    playlist_order = curriculum.get(
        "playlist_order", list(curriculum["categories"].keys())
    )

    # Get all previously generated lesson keys from the database
    db = get_db()
    rows = db.execute(
        "SELECT lesson_key FROM stories WHERE content_type = 'ai_education'"
    ).fetchall()
    db.close()
    generated_keys = {row["lesson_key"] for row in rows}

    # Walk through the curriculum in playlist order
    for cat_key in playlist_order:
        cat_data = curriculum["categories"].get(cat_key)
        if not cat_data:
            continue
        for topic_data in cat_data["topics"]:
            topic_name = topic_data["topic"]
            for lesson_title in topic_data["lessons"]:
                lesson_key = f"{cat_key}::{topic_name}::{lesson_title}"
                if lesson_key not in generated_keys:
                    return {
                        "category": cat_key,
                        "topic": topic_name,
                        "lesson_title": lesson_title,
                    }

    # All lessons have been generated
    return None


def get_ai_playlist_name(category: str) -> str:
    """Return the YouTube playlist name for a given curriculum category.

    Args:
        category: The curriculum category key (e.g., 'ai_basics').

    Returns:
        The human-readable playlist name string, or a generated fallback
        if the category is not found in the curriculum.
    """
    curriculum = load_curriculum()
    cat_data = curriculum["categories"].get(category)
    if cat_data:
        return cat_data.get("playlist_name", cat_data.get("display_name", category))
    return f"Little Wisdom AI Lab - {category.replace('_', ' ').title()}"


def get_curriculum_stats() -> dict:
    """Return statistics about curriculum coverage and generation progress.

    Returns:
        A dict with total_lessons, generated_count, remaining_count,
        and per-category breakdowns.
    """
    curriculum = load_curriculum()

    # Count total lessons in curriculum
    total = 0
    category_totals = {}
    for cat_key, cat_data in curriculum["categories"].items():
        cat_count = 0
        for topic_data in cat_data["topics"]:
            cat_count += len(topic_data["lessons"])
        category_totals[cat_key] = cat_count
        total += cat_count

    # Count generated lessons from database
    db = get_db()
    rows = db.execute(
        """SELECT category, COUNT(*) as cnt
           FROM stories
           WHERE content_type = 'ai_education'
           GROUP BY category"""
    ).fetchall()
    db.close()

    generated_by_category = {row["category"]: row["cnt"] for row in rows}
    generated_total = sum(generated_by_category.values())

    # Build per-category stats
    categories = {}
    for cat_key, cat_total in category_totals.items():
        cat_generated = generated_by_category.get(cat_key, 0)
        categories[cat_key] = {
            "total": cat_total,
            "generated": cat_generated,
            "remaining": cat_total - cat_generated,
            "display_name": curriculum["categories"][cat_key].get(
                "display_name", cat_key
            ),
        }

    return {
        "total_lessons": total,
        "generated_count": generated_total,
        "remaining_count": total - generated_total,
        "categories": categories,
    }


def get_unused_ai_lessons(limit: int = 5) -> list:
    """Get AI education lessons that have not been made into videos yet.

    Args:
        limit: Maximum number of lessons to return.

    Returns:
        A list of lesson dicts from the database with status 'generated'.
    """
    db = get_db()
    rows = db.execute(
        """SELECT * FROM stories
           WHERE content_type = 'ai_education' AND status = 'generated'
           ORDER BY created_at DESC LIMIT ?""",
        (limit,),
    ).fetchall()
    db.close()
    return [dict(row) for row in rows]


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--stats":
        # Show curriculum generation progress
        stats = get_curriculum_stats()
        print(f"\nLittle Wisdom AI Lab - Curriculum Stats")
        print(f"{'=' * 50}")
        print(
            f"Total: {stats['generated_count']}/{stats['total_lessons']} lessons "
            f"generated ({stats['remaining_count']} remaining)\n"
        )
        for cat_key, cat_stats in stats["categories"].items():
            bar_len = 20
            filled = int(
                bar_len * cat_stats["generated"] / max(cat_stats["total"], 1)
            )
            bar = "#" * filled + "-" * (bar_len - filled)
            print(
                f"  [{bar}] {cat_stats['generated']:3d}/{cat_stats['total']:3d}  "
                f"{cat_stats['display_name']}"
            )
        print()

    elif len(sys.argv) > 1 and sys.argv[1] == "--next":
        # Generate the next lesson in sequence
        next_lesson = pick_next_ai_lesson()
        if next_lesson:
            print(f"Generating next lesson: {next_lesson['lesson_title']}")
            print(f"  Category: {next_lesson['category']}")
            print(f"  Topic: {next_lesson['topic']}")
            script = generate_ai_lesson_script(**next_lesson)
            print(json.dumps(script, indent=2))
        else:
            print("All lessons have been generated!")

    elif len(sys.argv) > 3:
        # Generate a specific lesson: python ai_content_generator.py <category> <topic> <lesson_title>
        category = sys.argv[1]
        topic = sys.argv[2]
        lesson_title = " ".join(sys.argv[3:])
        script = generate_ai_lesson_script(category, topic, lesson_title)
        print(json.dumps(script, indent=2))

    else:
        # Default: show curriculum info and pick next lesson
        next_lesson = pick_next_ai_lesson()
        if next_lesson:
            print(f"Next lesson to generate:")
            print(f"  Category: {next_lesson['category']}")
            print(f"  Topic: {next_lesson['topic']}")
            print(f"  Lesson: {next_lesson['lesson_title']}")
            print(f"\nRun with --next to generate it, or --stats for progress.")
        else:
            print("All lessons have been generated! Run --stats for details.")
