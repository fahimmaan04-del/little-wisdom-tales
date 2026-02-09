"""
Image Fetcher - Downloads kid-friendly illustrations for story scenes.

Uses Pixabay API (free, 5000 req/day) with Pexels as fallback.
Focuses on cartoon/illustration style images suitable for children.
"""

import hashlib
import json
import os
import time
from pathlib import Path

import httpx
from dotenv import load_dotenv
from PIL import Image

load_dotenv()

OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "./output"))
CACHE_DIR = OUTPUT_DIR / "images" / "_cache"


def search_pixabay(
    query: str,
    per_page: int = 5,
    image_type: str = "illustration",
    category: str = "animals",
    safe_search: bool = True,
) -> list:
    """Search Pixabay for kid-friendly illustrations."""
    api_key = os.getenv("PIXABAY_API_KEY")
    if not api_key:
        return []

    params = {
        "key": api_key,
        "q": query,
        "image_type": image_type,
        "per_page": per_page,
        "safesearch": "true" if safe_search else "false",
        "editors_choice": "false",
        "order": "popular",
    }

    # Only add category if specified and valid
    valid_categories = [
        "backgrounds", "fashion", "nature", "science", "education",
        "feelings", "health", "people", "religion", "places",
        "animals", "industry", "computer", "food", "sports",
        "transportation", "travel", "buildings", "business", "music"
    ]
    if category and category in valid_categories:
        params["category"] = category

    try:
        response = httpx.get(
            "https://pixabay.com/api/",
            params=params,
            timeout=15.0,
        )
        response.raise_for_status()
        data = response.json()
        return [
            {
                "url": hit["largeImageURL"],
                "preview": hit["previewURL"],
                "width": hit["imageWidth"],
                "height": hit["imageHeight"],
                "source": "pixabay",
                "id": hit["id"],
            }
            for hit in data.get("hits", [])
        ]
    except Exception as e:
        print(f"Pixabay search failed: {e}")
        return []


def search_pexels(query: str, per_page: int = 5) -> list:
    """Search Pexels for images (fallback)."""
    api_key = os.getenv("PEXELS_API_KEY")
    if not api_key:
        return []

    try:
        response = httpx.get(
            "https://api.pexels.com/v1/search",
            params={"query": query, "per_page": per_page, "size": "large"},
            headers={"Authorization": api_key},
            timeout=15.0,
        )
        response.raise_for_status()
        data = response.json()
        return [
            {
                "url": photo["src"]["large2x"],
                "preview": photo["src"]["medium"],
                "width": photo["width"],
                "height": photo["height"],
                "source": "pexels",
                "id": photo["id"],
            }
            for photo in data.get("photos", [])
        ]
    except Exception as e:
        print(f"Pexels search failed: {e}")
        return []


def download_image(url: str, output_path: str, max_retries: int = 3) -> str:
    """Download an image and save it locally with retry logic."""
    # Check cache first
    url_hash = hashlib.md5(url.encode()).hexdigest()
    cache_path = CACHE_DIR / f"{url_hash}.jpg"

    if cache_path.exists():
        import shutil
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(cache_path), output_path)
        return output_path

    for attempt in range(max_retries):
        try:
            response = httpx.get(url, timeout=30.0, follow_redirects=True)
            response.raise_for_status()

            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "wb") as f:
                f.write(response.content)

            # Save to cache
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            with open(str(cache_path), "wb") as f:
                f.write(response.content)

            return output_path
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"  Download attempt {attempt+1} failed: {e}, retrying...")
                time.sleep(2)
            else:
                raise

    return output_path


def resize_for_video(image_path: str, width: int = 1920, height: int = 1080) -> str:
    """Resize and crop image to fit video dimensions (16:9)."""
    img = Image.open(image_path)

    # Convert to RGB for JPEG compatibility (handles P, RGBA, LA modes)
    if img.mode != "RGB":
        img = img.convert("RGB")

    # Calculate aspect ratios
    target_ratio = width / height
    img_ratio = img.width / img.height

    if img_ratio > target_ratio:
        # Image is wider - crop sides
        new_width = int(img.height * target_ratio)
        left = (img.width - new_width) // 2
        img = img.crop((left, 0, left + new_width, img.height))
    else:
        # Image is taller - crop top/bottom
        new_height = int(img.width / target_ratio)
        top = (img.height - new_height) // 2
        img = img.crop((0, top, img.width, top + new_height))

    img = img.resize((width, height), Image.LANCZOS)

    # Save as high-quality JPEG
    output_path = image_path.replace(".jpg", "_resized.jpg")
    if output_path == image_path:
        output_path = image_path.rsplit(".", 1)[0] + "_resized.jpg"

    img.save(output_path, "JPEG", quality=95)
    return output_path


def resize_for_shorts(image_path: str, width: int = 1080, height: int = 1920) -> str:
    """Resize and crop image for YouTube Shorts (9:16 vertical)."""
    img = Image.open(image_path)

    if img.mode != "RGB":
        img = img.convert("RGB")

    target_ratio = width / height
    img_ratio = img.width / img.height

    if img_ratio > target_ratio:
        new_width = int(img.height * target_ratio)
        left = (img.width - new_width) // 2
        img = img.crop((left, 0, left + new_width, img.height))
    else:
        new_height = int(img.width / target_ratio)
        top = (img.height - new_height) // 2
        img = img.crop((0, top, img.width, top + new_height))

    img = img.resize((width, height), Image.LANCZOS)

    output_path = image_path.rsplit(".", 1)[0] + "_shorts.jpg"
    img.save(output_path, "JPEG", quality=95)
    return output_path


def create_text_overlay_image(
    text: str,
    width: int = 1920,
    height: int = 1080,
    bg_color: str = "#2C3E50",
    text_color: str = "#FFFFFF",
    font_size: int = 60,
) -> str:
    """Create a simple text card image (for intro/outro/moral screens)."""
    from PIL import ImageDraw, ImageFont

    img = Image.new("RGB", (width, height), bg_color)
    draw = ImageDraw.Draw(img)

    # Try to use a system font, fall back to default
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
    except (OSError, IOError):
        font = ImageFont.load_default()

    # Word wrap text
    words = text.split()
    lines = []
    current_line = ""
    for word in words:
        test_line = f"{current_line} {word}".strip()
        bbox = draw.textbbox((0, 0), test_line, font=font)
        if bbox[2] - bbox[0] > width - 100:
            lines.append(current_line)
            current_line = word
        else:
            current_line = test_line
    if current_line:
        lines.append(current_line)

    # Calculate vertical position
    line_height = font_size + 10
    total_height = len(lines) * line_height
    y = (height - total_height) // 2

    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        x = (width - (bbox[2] - bbox[0])) // 2
        draw.text((x, y), line, fill=text_color, font=font)
        y += line_height

    output_path = str(OUTPUT_DIR / "images" / f"text_{hashlib.md5(text.encode()).hexdigest()[:8]}.jpg")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path, "JPEG", quality=95)
    return output_path


def fetch_scene_images(story_script: dict, story_id: int, for_shorts: bool = False) -> list:
    """Fetch images for all scenes in a story."""
    scenes = story_script.get("scenes", [])
    image_dir = OUTPUT_DIR / "images" / f"story_{story_id}"
    image_dir.mkdir(parents=True, exist_ok=True)

    results = []

    for scene in scenes:
        scene_num = scene["scene_number"]
        search_terms = scene.get("image_search_terms", "")
        visual_desc = scene.get("visual_description", "")

        # Use image_search_terms first, fall back to visual_description
        primary = search_terms or visual_desc

        # Truncate to first 6 keywords to avoid Pixabay 400 errors
        words = primary.split()[:6]
        short_search = " ".join(words)

        # Build kid-friendly search query
        kid_search = f"{short_search} cartoon illustration kids"

        print(f"  Fetching image for scene {scene_num}: {kid_search[:55]}...")

        # Strategy 1: Pixabay illustration (no category restriction)
        images = search_pixabay(kid_search, per_page=5, image_type="illustration", category="")

        # Strategy 2: Try with visual_description keywords
        if not images and visual_desc:
            vis_words = visual_desc.split()[:5]
            vis_search = " ".join(vis_words) + " cartoon kids"
            images = search_pixabay(vis_search, per_page=5, image_type="illustration", category="")

        # Strategy 3: Broader search without modifiers
        if not images:
            images = search_pixabay(short_search, per_page=5, image_type="vector", category="")

        # Strategy 4: Pexels fallback
        if not images:
            images = search_pexels(kid_search, per_page=3)

        if images:
            # Download the best match
            img_data = images[0]
            raw_path = str(image_dir / f"scene_{scene_num:02d}_raw.jpg")
            download_image(img_data["url"], raw_path)

            # Resize for video format
            if for_shorts:
                final_path = resize_for_shorts(raw_path)
            else:
                final_path = resize_for_video(raw_path)

            results.append({
                "scene_number": scene_num,
                "image_path": final_path,
                "source": img_data["source"],
                "original_url": img_data["url"],
            })
        else:
            # Create a text-based fallback image
            visual_desc = scene.get("visual_description", "Story scene")
            w, h = (1080, 1920) if for_shorts else (1920, 1080)
            fallback_path = create_text_overlay_image(
                visual_desc, width=w, height=h,
                bg_color="#3498DB", font_size=40,
            )
            results.append({
                "scene_number": scene_num,
                "image_path": fallback_path,
                "source": "generated",
                "original_url": None,
            })

        # Rate limiting - be respectful to APIs
        time.sleep(0.5)

    return results


if __name__ == "__main__":
    # Test image fetching
    test_script = {
        "scenes": [
            {
                "scene_number": 1,
                "visual_description": "A cute rabbit in a sunny forest",
                "image_search_terms": "cute rabbit forest cartoon",
            },
            {
                "scene_number": 2,
                "visual_description": "A wise old owl on a tree branch",
                "image_search_terms": "wise owl tree cartoon",
            },
        ]
    }

    results = fetch_scene_images(test_script, story_id=0)
    for r in results:
        print(json.dumps(r, indent=2))
