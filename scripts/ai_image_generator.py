"""
AI Image Generator - Creates custom illustrations matching story narration.

Uses SDXL Turbo on local GPU (NVIDIA RTX A5000, 24GB VRAM) for fast
kid-friendly cartoon illustration generation. SDXL Turbo generates images
in 1-4 inference steps at ~0.5s per image.

Generates many varied images per scene (different angles, close-ups, wide shots)
for animated crossfade effects in the video assembler.
"""

import gc
import os
import time
from pathlib import Path

import torch
from PIL import Image

from dotenv import load_dotenv

load_dotenv()

OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "./output"))

# Global model reference for reuse across calls
_pipeline = None

# Base camera variations for visual diversity within a scene (kid-focused)
CAMERA_VARIATIONS = [
    "",  # base prompt as-is
    "cute character expression close-up, big sparkling eyes, detailed face,",
    "wide angle cinematic view, colorful scene overview, depth of field,",
    "colorful scene with sparkles and stars, magical particles,",
    "soft focus background, adorable character foreground, bokeh effect,",
    "bird's eye view, looking down at cute scene, overhead shot,",
    "low angle hero shot, looking up at friendly character, dramatic sky,",
    "side view profile, character silhouette with warm golden smile,",
    "happy characters playing, joyful dynamic movement, motion blur,",
    "warm sunshine lighting, golden hour glow, lens flare,",
    "vibrant rainbow colors, extra bright and vivid, saturated,",
    "gentle morning light, peaceful meadow setting, dew drops,",
    "magical sparkles, fairy dust, enchanted glowing atmosphere,",
    "storybook page layout, whimsical decorative border, ornate frame,",
    "cartoon group shot, characters together, learning moment,",
    "friendly characters waving, inviting warm composition,",
    "cozy bedtime scene, soft candlelight, warm blankets,",
    "adventure scene, excited characters exploring, treasure map,",
    "group of animal friends together, heartfelt friendship moment,",
    "celebration scene, confetti and balloons, party atmosphere,",
    "dramatic reveal moment, spotlight lighting, gasping expression,",
]

# Content-type-specific camera variations for better visual match
EDUCATION_CAMERA_VARIATIONS = [
    "",
    "chalkboard close-up with colorful equations and diagrams,",
    "owl professor pointing at diagram, excited teaching pose,",
    "student characters raising hands eagerly, classroom engagement,",
    "colorful experiment scene, bubbling test tubes, safety goggles,",
    "book pages flying open, magical knowledge, glowing letters,",
    "world map background, geography adventure, spinning globe,",
    "math puzzle pieces coming together, satisfying solution moment,",
    "microscope view, tiny cartoon creatures, science discovery,",
    "timeline with historical events, cartoon history characters,",
    "group study scene, characters helping each other, teamwork,",
    "quiz show setup, buzzers and lights, competition fun,",
    "nature walk discovery, magnifying glass, outdoor classroom,",
    "space exploration, cartoon planets, telescope view,",
    "art class creativity, paint splashes, colorful easels,",
]

AI_EDUCATION_CAMERA_VARIATIONS = [
    "",
    "robot character typing on holographic keyboard, code floating,",
    "circuit board close-up, glowing pathways, data flowing,",
    "futuristic lab panorama, floating screens, neon glow,",
    "coding screen with colorful blocks, scratch-style programming,",
    "robot building scene, gears and parts assembling, sparks,",
    "AI brain visualization, neural network, glowing connections,",
    "game development scene, pixel characters on screen, joystick,",
    "3D printer creating objects, layer by layer, futuristic,",
    "drone flying overhead, camera view, tech landscape,",
    "VR headset scene, virtual world, immersive experience,",
    "chatbot conversation bubbles, friendly AI assistant,",
    "solar-powered robot, green energy, sustainable tech,",
]

CRAFTS_SKILLS_CAMERA_VARIATIONS = [
    "",
    "close-up of hands using tools, wood shavings, precise work,",
    "workshop panorama, organized pegboard, warm wood tones,",
    "measuring tape and ruler close-up, precise measurement,",
    "paint roller on wall, satisfying color application, drips,",
    "pipe fitting assembly, water droplet, wrench turning,",
    "electrical wire close-up, safe insulation, bright copper,",
    "safety gear display, hard hat and goggles, thumbs up,",
    "before and after transformation, impressive result,",
    "tool collection display, organized and clean, labeled,",
    "blueprint reading scene, detailed plans, compass,",
    "finished project reveal, proud character, achievement,",
    "step by step progression, numbered stages, clear guide,",
]

# Quality boosters appended to all style prefixes for sharper output
_QUALITY_SUFFIX = (
    "masterpiece, ultra detailed, 8K render quality, cinematic lighting, "
    "sharp focus, professional digital illustration, trending on artstation, "
)

# Enhanced style prefix for kid-friendly story illustrations
STYLE_PREFIX = (
    "Children's book illustration in Pixar and Disney Junior style, "
    "cute cartoon characters with big expressive eyes and rounded soft shapes, "
    "bright saturated primary colors, painterly textured backgrounds, "
    "safe for children, no scary elements, happy and friendly atmosphere, "
    "warm golden hour lighting with soft volumetric rays, "
    "Cocomelon and Bluey inspired aesthetic, "
    "adorable wholesome kid-friendly illustration, "
    "detailed character expressions, rich environmental storytelling, "
    + _QUALITY_SUFFIX
)

# Education-specific style prefix for teaching content
EDUCATION_STYLE_PREFIX = (
    "Children's educational cartoon illustration, friendly cartoon classroom setting, "
    "cute wise owl professor character with round glasses and graduation cap, "
    "bright colorful chalkboard with chalk dust particles, warm classroom lighting, "
    "cartoon math symbols, alphabet letters, and bubbly science equipment, "
    "clean clear visuals designed to teach concepts with visual metaphors, "
    "Pixar and Disney Junior cartoon style with big eyes and rounded shapes, "
    "safe for children, happy encouraging learning atmosphere, "
    "bold outlines, rich primary colors, detailed textures on books and props, "
    + _QUALITY_SUFFIX
)

# AI Education style prefix - futuristic cartoon lab
AI_EDUCATION_STYLE_PREFIX = (
    "Children's cartoon illustration of a futuristic tech laboratory, "
    "cute friendly blue robot character named Byte with big glowing expressive eyes and antenna, "
    "colorful holographic floating screens and glowing circuit pathways in background, "
    "Pixar and Disney Junior cartoon style, bright neon blues purples and cyans, "
    "safe for children, exciting fun discovery atmosphere with wonder, "
    "cartoon coding blocks, cute pixel art characters, floating translucent data cubes, "
    "clean futuristic layout, bold outlines, vibrant neon glow effects, "
    "detailed reflective surfaces, glass panels, soft ambient sci-fi lighting, "
    + _QUALITY_SUFFIX
)

# Crafts/Skills style prefix - cartoon workshop
CRAFTS_SKILLS_STYLE_PREFIX = (
    "Children's cartoon illustration of a warm colorful workshop studio, "
    "cute friendly toolbelt character named Handy with big expressive eyes and bright yellow hard hat, "
    "cartoon tools with friendly faces, rich wood grain textures, paint splashes, "
    "Pixar and Disney Junior cartoon style, warm golden browns and bright sunny yellows, "
    "safe for children, fun hands-on learning atmosphere with sawdust particles, "
    "cartoon safety goggles, measuring tape, and friendly anthropomorphic tool characters, "
    "cozy workshop layout, bold outlines, warm inviting amber lighting, "
    "detailed wood textures, metal sheens, fabric patterns, "
    + _QUALITY_SUFFIX
)


def get_style_prefix(content_type: str = "story") -> str:
    """Return the appropriate cartoon style prefix based on content type."""
    if content_type == "education":
        return EDUCATION_STYLE_PREFIX
    elif content_type == "ai_education":
        return AI_EDUCATION_STYLE_PREFIX
    elif content_type == "crafts_skills":
        return CRAFTS_SKILLS_STYLE_PREFIX
    return STYLE_PREFIX


def get_camera_variations(content_type: str = "story") -> list:
    """Return content-type-specific camera variations for richer imagery."""
    if content_type == "education":
        return EDUCATION_CAMERA_VARIATIONS
    elif content_type == "ai_education":
        return AI_EDUCATION_CAMERA_VARIATIONS
    elif content_type == "crafts_skills":
        return CRAFTS_SKILLS_CAMERA_VARIATIONS
    return CAMERA_VARIATIONS


def _load_pipeline():
    """Load SDXL Turbo pipeline onto GPU. Cached globally for reuse."""
    global _pipeline
    if _pipeline is not None:
        return _pipeline

    from diffusers import AutoPipelineForText2Image

    print("  Loading SDXL Turbo model onto GPU...")
    start = time.time()

    _pipeline = AutoPipelineForText2Image.from_pretrained(
        "stabilityai/sdxl-turbo",
        torch_dtype=torch.float16,
        variant="fp16",
    )
    _pipeline.to("cuda")

    # Enable memory-efficient attention
    _pipeline.enable_attention_slicing()

    elapsed = time.time() - start
    print(f"  SDXL Turbo loaded in {elapsed:.1f}s")
    return _pipeline


def generate_scene_image(
    prompt: str,
    output_path: str,
    width: int = 1024,
    height: int = 576,
    seed: int = None,
    variation_prefix: str = "",
    content_type: str = "story",
) -> str:
    """Generate a single illustration using SDXL Turbo on local GPU.

    Generates at 512x512 (SDXL Turbo's native resolution) then
    resizes to target dimensions for best quality.

    Args:
        prompt: Visual description of the scene to generate.
        output_path: File path to save the generated JPEG image.
        width: Target width for the output image (default 1024).
        height: Target height for the output image (default 576).
        seed: Optional seed for reproducible generation.
        variation_prefix: Camera/angle variation to prepend to prompt.
        content_type: 'story' or 'education' to select appropriate
            style prefix. Defaults to 'story'.
    """
    pipe = _load_pipeline()

    style = get_style_prefix(content_type)
    full_prompt = f"{style}{variation_prefix} {prompt}".strip()

    generator = None
    if seed is not None:
        generator = torch.Generator(device="cuda").manual_seed(seed)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # SDXL Turbo: 1 step, no CFG needed
    result = pipe(
        prompt=full_prompt,
        num_inference_steps=4,
        guidance_scale=0.0,
        width=512,
        height=512,
        generator=generator,
    )

    img = result.images[0]

    # Resize to target video dimensions with high quality
    img = img.resize((width, height), Image.LANCZOS)

    # Save as JPEG
    img.save(output_path, "JPEG", quality=92)
    return output_path


def generate_story_images(
    story_script: dict,
    story_id: int,
    for_shorts: bool = False,
    images_per_scene: int = 30,
    content_type: str = None,
) -> list:
    """Generate AI images for all scenes in a story.

    Generates many images per scene with varied prompts (different angles,
    close-ups, wide shots) for animated crossfade effects in the video.
    Sequential generation on GPU (already fast at ~0.5s/image).

    Args:
        story_script: Story JSON dict with scenes, title, collection, etc.
        story_id: Unique database ID for the story.
        for_shorts: If True, generate portrait images (576x1024).
        images_per_scene: Number of image variations per scene (default 30).
        content_type: 'story' or 'education'. If None, auto-detected from
            story_script metadata (collection field). Defaults to 'story'
            when auto-detection finds no match.
    """
    # Auto-detect content type from story metadata if not explicitly set
    if content_type is None:
        collection = story_script.get("collection", "").lower()
        # Detect educational content by collection name keywords
        education_keywords = [
            "education", "learning", "science", "math", "abc",
            "numbers", "counting", "alphabet", "classroom", "lesson",
            "teaching", "school",
        ]
        if any(kw in collection for kw in education_keywords):
            content_type = "education"
        else:
            content_type = "story"

    scenes = story_script.get("scenes", [])
    image_dir = OUTPUT_DIR / "images" / f"story_{story_id}"
    image_dir.mkdir(parents=True, exist_ok=True)

    width = 576 if for_shorts else 1024
    height = 1024 if for_shorts else 576

    results = []
    base_seed = hash(story_script.get("title", "")) % 100000
    total_images = 0
    start_time = time.time()

    # Pre-load model once
    _load_pipeline()

    for scene in scenes:
        scene_num = scene["scene_number"]
        visual_desc = scene.get("visual_description", "")
        narration = scene.get("narration", scene.get("text", ""))

        prompt = visual_desc if visual_desc else narration
        if not prompt:
            prompt = f"A colorful scene from a children's story, scene {scene_num}"

        print(f"  Generating {images_per_scene} images for scene {scene_num}/{len(scenes)}...")
        scene_start = time.time()

        scene_images = []
        for img_idx in range(images_per_scene):
            img_path = str(image_dir / f"scene_{scene_num:02d}_v{img_idx:03d}.jpg")

            # Skip if already generated (resume support)
            if os.path.exists(img_path) and os.path.getsize(img_path) > 1000:
                scene_images.append(img_path)
                total_images += 1
                continue

            seed = base_seed + scene_num * 1000 + img_idx
            variations = get_camera_variations(content_type)
            variation = variations[img_idx % len(variations)]

            try:
                generate_scene_image(
                    prompt=prompt,
                    output_path=img_path,
                    width=width,
                    height=height,
                    seed=seed,
                    variation_prefix=variation,
                    content_type=content_type,
                )
                scene_images.append(img_path)
                total_images += 1
            except Exception as e:
                print(f"    Warning: Failed to generate image {img_idx}: {e}")

        elapsed = time.time() - scene_start
        if scene_images:
            print(f"    Scene {scene_num}: {len(scene_images)} images in {elapsed:.1f}s "
                  f"({elapsed/len(scene_images):.2f}s/img)")

        if scene_images:
            results.append({
                "scene_number": scene_num,
                "image_path": scene_images[0],
                "all_images": scene_images,
                "source": "ai_generated",
            })

    total_time = time.time() - start_time
    print(f"  Generated {total_images} total images for {len(results)} scenes "
          f"in {total_time:.1f}s ({total_time/max(total_images,1):.2f}s/img)")
    return results


def unload_model():
    """Unload SDXL Turbo from GPU to free VRAM."""
    global _pipeline
    if _pipeline is not None:
        del _pipeline
        _pipeline = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        print("  SDXL Turbo model unloaded from GPU")


if __name__ == "__main__":
    test_prompt = "A small rabbit sitting under a big oak tree in a sunny forest"
    output = generate_scene_image(
        prompt=test_prompt,
        output_path="output/images/test_ai_image.jpg",
    )
    print(f"Generated: {output}")
    unload_model()
