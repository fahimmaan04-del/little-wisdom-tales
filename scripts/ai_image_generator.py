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

# Prompt variations for visual diversity within a scene
CAMERA_VARIATIONS = [
    "",  # base prompt as-is
    "close-up view, detailed,",
    "wide angle view, panoramic,",
    "dramatic angle, dynamic composition,",
    "soft focus background, character foreground,",
    "bird's eye view, looking down,",
    "low angle, looking up,",
    "side view, profile,",
    "action shot, movement,",
    "warm lighting, golden hour,",
    "vibrant saturated colors, vivid,",
    "gentle morning light, peaceful,",
    "magical sparkles, enchanted atmosphere,",
    "storybook page layout, decorative border,",
    "cinematic composition, depth of field,",
]

# Style prefix for consistent kid-friendly look
STYLE_PREFIX = (
    "Children's book illustration, colorful cartoon style, "
    "bright and cheerful, kid-friendly, digital art, "
    "high quality, vibrant colors, "
)


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
) -> str:
    """Generate a single illustration using SDXL Turbo on local GPU.

    Generates at 512x512 (SDXL Turbo's native resolution) then
    resizes to target dimensions for best quality.
    """
    pipe = _load_pipeline()

    full_prompt = f"{STYLE_PREFIX}{variation_prefix} {prompt}".strip()

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
) -> list:
    """Generate AI images for all scenes in a story.

    Generates many images per scene with varied prompts (different angles,
    close-ups, wide shots) for animated crossfade effects in the video.
    Sequential generation on GPU (already fast at ~0.5s/image).
    """
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
            variation = CAMERA_VARIATIONS[img_idx % len(CAMERA_VARIATIONS)]

            try:
                generate_scene_image(
                    prompt=prompt,
                    output_path=img_path,
                    width=width,
                    height=height,
                    seed=seed,
                    variation_prefix=variation,
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
