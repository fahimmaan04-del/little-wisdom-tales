"""
Engagement Hooks - Maximizes viewership and subscriber growth.

Adds strategic elements to videos that encourage:
- Subscribing to the channel
- Coming back daily for new stories
- Watching the video till the end (watch time)
- Sharing with friends

YouTube Algorithm Signals Targeted:
- Click-Through Rate (CTR): Eye-catching thumbnails + titles
- Watch Time: Cliffhangers, engaging narration, "stay till the end"
- Session Time: End-screen suggestions, series format
- Engagement: Made for Kids limits comments, so focus on likes + shares
"""

import json
import random
from pathlib import Path

CONFIG_DIR = Path("./config")


# --- Title Optimization ---

TITLE_TEMPLATES = [
    "{story_title} | Kids Moral Story",
    "{story_title} - A Story About {moral}",
    "The Amazing Story of {story_title}",
    "{story_title} | Bedtime Story for Kids",
    "Can You Guess the Lesson? {story_title}",
    "{story_title} | {origin} Story for Children",
    "What Happens Next? {story_title}",
    "{story_title} | Stories That Teach {moral}",
]

SHORTS_TITLE_TEMPLATES = [
    "{story_title} | Quick Moral Story",
    "{moral} - {story_title}",
    "Kids Story in 60 Seconds: {story_title}",
    "The Lesson of {story_title}",
]


def generate_engaging_title(
    story_title: str,
    moral: str,
    origin: str = "",
    is_shorts: bool = False,
) -> str:
    """Generate an engagement-optimized title."""
    templates = SHORTS_TITLE_TEMPLATES if is_shorts else TITLE_TEMPLATES
    template = random.choice(templates)
    title = template.format(
        story_title=story_title,
        moral=moral.title(),
        origin=origin,
    )
    # YouTube title max is 100 chars, keep under 70 for better CTR
    return title[:70]


# --- Description Templates ---

def generate_description(
    story_script: dict,
    is_shorts: bool = False,
) -> str:
    """Generate an SEO-optimized description with engagement hooks."""
    title = story_script.get("title", "")
    moral = story_script.get("moral", "")
    collection = story_script.get("collection", "").replace("_", " ").title()
    origin = story_script.get("origin", "")
    base_desc = story_script.get("description", "")

    # Teaser for next video
    next_teasers = [
        "New stories from around the world are added every day!",
        "More amazing adventures are coming soon!",
        "Stay tuned for more magical tales from around the world!",
        "The adventure continues with incredible new tales!",
        "More wisdom and wonder from world folklore awaits!",
    ]

    description = f"""{base_desc}

Today's Moral: {moral}

This beautiful story comes from the {collection} collection ({origin}).

{random.choice(next_teasers)}

---
Welcome to Kids-Heaven!
We bring you magical moral stories from around the world, every single day!
Stories that teach kindness, courage, honesty, and wisdom.

New stories EVERY DAY for kids aged 4-10!

#KidsStories #MoralStories #BedtimeStories #ChildrenStories
#KidsHeaven #{collection.replace(' ', '')} #KidsEducation
#StoriesForKids #AnimatedStories #MoralLessons
"""
    return description


# --- Narration Hooks (injected into story script) ---

INTRO_HOOKS = [
    "Hey friends! Welcome to Kids-Heaven! Today, we have a SUPER special story for you!",
    "Hello, little ones! Are you ready for an amazing adventure? Let's go!",
    "Welcome back, storytellers! We have the BEST story for you today!",
    "Gather around, friends! Today's story has an important lesson. Can you figure out what it is?",
    "Hey there! Before we start, can you guess what today's moral will be? Let's find out!",
]

MIDPOINT_HOOKS = [
    "Wow, what do you think will happen next? Keep watching to find out!",
    "Oh no! What should our hero do? Stay tuned!",
    "This is getting exciting! Wait until you see what happens!",
]

# Made for Kids compliant - no direct subscribe CTAs to children
OUTRO_HOOKS = [
    (
        "And that's the end of our story! "
        "What a wonderful lesson about being kind! "
        "We hope you enjoyed this adventure. See you in the next story!"
    ),
    (
        "Wasn't that a great story? "
        "Remember, every day there's a new adventure waiting for you! "
        "See you next time, friends!"
    ),
    (
        "And so our story comes to an end! "
        "We have so many more magical tales to share with you! "
        "Until next time, keep being amazing!"
    ),
    (
        "That was today's story! "
        "There are new stories here every single day! "
        "Goodbye for now, and remember today's lesson!"
    ),
]

# For Shorts - very brief outros
SHORTS_OUTRO_HOOKS = [
    "And that's the lesson! More stories every day!",
    "Remember this lesson! New stories coming soon!",
    "The end! More adventures are waiting for you!",
    "What a great lesson! See you in the next story!",
]


def inject_engagement_hooks(story_script: dict, is_shorts: bool = False) -> dict:
    """Inject engagement hooks into the story script."""
    scenes = story_script.get("scenes", [])
    if not scenes:
        return story_script

    if is_shorts:
        # For Shorts: minimal hooks to stay under 60 seconds
        # Only add a brief outro - no intro or midpoint
        outro_scene = {
            "scene_number": 999,
            "narration": random.choice(SHORTS_OUTRO_HOOKS),
            "visual_description": "Colorful kids story ending with sparkles",
            "image_search_terms": "colorful kids cartoon sparkles stars ending illustration",
            "character_speaking": None,
            "duration_seconds": 4,
            "is_hook": True,
        }
        new_scenes = list(scenes) + [outro_scene]
    else:
        # For regular videos: full intro + midpoint + outro
        intro_scene = {
            "scene_number": 0,
            "narration": random.choice(INTRO_HOOKS),
            "visual_description": "Colorful animated channel intro with sparkles and stars",
            "image_search_terms": "colorful kids cartoon stars sparkles illustration",
            "character_speaking": None,
            "duration_seconds": 8,
            "is_hook": True,
        }

        midpoint_scene = None
        if len(scenes) >= 6:
            mid_idx = len(scenes) // 2
            midpoint_scene = {
                "scene_number": mid_idx + 0.5,
                "narration": random.choice(MIDPOINT_HOOKS),
                "visual_description": scenes[mid_idx].get("visual_description", ""),
                "image_search_terms": scenes[mid_idx].get("image_search_terms", ""),
                "character_speaking": None,
                "duration_seconds": 5,
                "is_hook": True,
            }

        outro_scene = {
            "scene_number": 999,
            "narration": random.choice(OUTRO_HOOKS),
            "visual_description": "Colorful story ending screen with channel branding",
            "image_search_terms": "colorful kids cartoon ending screen stars illustration",
            "character_speaking": None,
            "duration_seconds": 8,
            "is_hook": True,
        }

        new_scenes = [intro_scene]
        for i, scene in enumerate(scenes):
            new_scenes.append(scene)
            if midpoint_scene and i == len(scenes) // 2 - 1:
                new_scenes.append(midpoint_scene)
        new_scenes.append(outro_scene)

    # Renumber scenes sequentially
    for i, scene in enumerate(new_scenes):
        scene["scene_number"] = i + 1

    story_script["scenes"] = new_scenes
    return story_script


# --- Thumbnail Enhancement ---

THUMBNAIL_OVERLAYS = [
    "NEW!",
    "MORAL STORY",
    "MUST WATCH!",
    "KIDS STORY",
    "BEDTIME TALE",
]


def get_thumbnail_overlay_text() -> str:
    """Get a random eye-catching overlay text for thumbnails."""
    return random.choice(THUMBNAIL_OVERLAYS)


# --- Series & Continuity ---

def get_series_info(collection: str, episode_num: int = None) -> dict:
    """Generate series information for continuity."""
    series_names = {
        "aesop_fables": "Aesop's Amazing Adventures",
        "panchatantra": "Tales from Ancient India",
        "arabian_nights": "Magical Arabian Tales",
        "african_folktales": "African Story Time",
        "brothers_grimm": "Grimm's Magical Stories",
        "japanese_folktales": "Tales from Japan",
        "latin_american_tales": "Stories from the Americas",
        "french_fables": "French Fable Fun",
    }

    return {
        "series_name": series_names.get(collection, "World Stories"),
        "episode_prefix": f"Episode {episode_num}" if episode_num else "",
    }
