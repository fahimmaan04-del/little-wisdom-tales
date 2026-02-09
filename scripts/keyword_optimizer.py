"""
Keyword Optimizer - Continuous analytics-driven keyword intelligence.

Monitors YouTube search trends and video analytics to automatically
discover which keywords kids and parents use. Feeds trending keywords
into video titles, descriptions, and tags for maximum discoverability.

Data sources:
1. YouTube Search Suggest API (trending search terms)
2. Own video analytics (which keywords correlate with high views)
3. Seed keyword expansion based on content niche

Runs continuously via the scheduler to keep keyword database fresh.
"""

import json
import logging
import os
import random
import sqlite3
import time
from datetime import datetime, timedelta
from pathlib import Path

import httpx
from dotenv import load_dotenv

load_dotenv()

DATA_DIR = Path(os.getenv("DATA_DIR", "./data"))

logger = logging.getLogger(__name__)

# Seed keywords to expand from - core niche terms
SEED_KEYWORDS = [
    "kids stories",
    "bedtime stories",
    "moral stories for kids",
    "stories for children",
    "animated stories",
    "fairy tales for kids",
    "kids moral lessons",
    "children bedtime stories",
    "story time for kids",
    "nursery stories",
    "panchatantra stories",
    "aesop fables for kids",
    "african folktales for kids",
    "arabian nights stories",
    "animal stories for kids",
    "kids learning stories",
    "stories about kindness",
    "stories about honesty",
    "stories about courage",
    "stories about friendship",
    "short stories for kids",
    "educational stories",
    "fun stories for toddlers",
    "preschool stories",
]

# Categories to organize keywords
KEYWORD_CATEGORIES = {
    "story_type": ["bedtime", "moral", "fairy tale", "animated", "short"],
    "audience": ["kids", "children", "toddlers", "preschool", "family"],
    "theme": ["kindness", "honesty", "courage", "friendship", "wisdom", "sharing"],
    "format": ["stories", "tales", "adventures", "lessons", "fables"],
}


def _get_db():
    """Get keyword optimizer database connection."""
    db_path = DATA_DIR / "stories.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    # Create keywords table if needed
    conn.execute("""
        CREATE TABLE IF NOT EXISTS trending_keywords (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            keyword TEXT NOT NULL UNIQUE,
            category TEXT DEFAULT 'general',
            search_volume_score REAL DEFAULT 0,
            performance_score REAL DEFAULT 0,
            combined_score REAL DEFAULT 0,
            times_used INTEGER DEFAULT 0,
            last_fetched TIMESTAMP,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Create keyword usage tracking table
    conn.execute("""
        CREATE TABLE IF NOT EXISTS keyword_usage (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            story_id INTEGER,
            keyword TEXT,
            used_in TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (story_id) REFERENCES stories(id)
        )
    """)

    conn.commit()
    return conn


def fetch_youtube_suggestions(query: str) -> list[str]:
    """Fetch YouTube search autocomplete suggestions for a query.

    Uses Google's suggestion API (same as YouTube search bar).
    Returns list of suggested search terms.
    """
    try:
        url = "https://suggestqueries.google.com/complete/search"
        params = {
            "client": "youtube",
            "q": query,
            "ds": "yt",
        }
        response = httpx.get(url, params=params, timeout=10)
        response.raise_for_status()

        # Response is JSONP - extract JSON array
        text = response.text
        # Format: window.google.ac.h(["query",[["suggestion1"],["suggestion2"],...]])
        start = text.index("[")
        data = json.loads(text[start:text.rindex("]") + 1])

        suggestions = []
        if len(data) > 1 and isinstance(data[1], list):
            for item in data[1]:
                if isinstance(item, list) and len(item) > 0:
                    suggestions.append(item[0])

        return suggestions

    except Exception as e:
        logger.warning(f"Failed to fetch suggestions for '{query}': {e}")
        return []


def discover_trending_keywords() -> list[dict]:
    """Discover trending keywords from YouTube search suggestions.

    Expands seed keywords through the suggestion API to find
    what kids/parents are actually searching for right now.
    """
    all_keywords = {}
    total_fetched = 0

    for seed in SEED_KEYWORDS:
        # Rate limit: 0.5s between requests
        time.sleep(0.5)

        suggestions = fetch_youtube_suggestions(seed)
        for suggestion in suggestions:
            suggestion_lower = suggestion.lower().strip()
            if suggestion_lower not in all_keywords:
                # Score based on position (earlier = more popular)
                position_score = 1.0 - (suggestions.index(suggestion) / max(len(suggestions), 1))
                all_keywords[suggestion_lower] = {
                    "keyword": suggestion_lower,
                    "search_volume_score": position_score,
                    "category": _categorize_keyword(suggestion_lower),
                    "source_seed": seed,
                }
                total_fetched += 1

        # Also try partial queries for deeper suggestions
        for prefix in [seed + " for", seed + " about", seed + " with"]:
            time.sleep(0.3)
            extra_suggestions = fetch_youtube_suggestions(prefix)
            for suggestion in extra_suggestions:
                suggestion_lower = suggestion.lower().strip()
                if suggestion_lower not in all_keywords:
                    position_score = 0.5 - (extra_suggestions.index(suggestion) / max(len(extra_suggestions), 1)) * 0.5
                    all_keywords[suggestion_lower] = {
                        "keyword": suggestion_lower,
                        "search_volume_score": max(0.1, position_score),
                        "category": _categorize_keyword(suggestion_lower),
                        "source_seed": prefix,
                    }
                    total_fetched += 1

    logger.info(f"Discovered {total_fetched} trending keywords from {len(SEED_KEYWORDS)} seeds")
    print(f"  Discovered {total_fetched} trending keywords")

    return list(all_keywords.values())


def _categorize_keyword(keyword: str) -> str:
    """Auto-categorize a keyword based on content."""
    keyword_lower = keyword.lower()
    for category, terms in KEYWORD_CATEGORIES.items():
        for term in terms:
            if term in keyword_lower:
                return category
    return "general"


def update_keyword_database(keywords: list[dict] = None):
    """Update the trending keywords database.

    Fetches new keywords if none provided, updates scores
    for existing keywords, and prunes stale entries.
    """
    if keywords is None:
        keywords = discover_trending_keywords()

    conn = _get_db()
    now = datetime.now().isoformat()
    updated = 0

    for kw in keywords:
        conn.execute("""
            INSERT INTO trending_keywords (keyword, category, search_volume_score, last_fetched, updated_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(keyword) DO UPDATE SET
                search_volume_score = (search_volume_score + excluded.search_volume_score) / 2,
                category = excluded.category,
                last_fetched = excluded.last_fetched,
                updated_at = excluded.updated_at
        """, (kw["keyword"], kw["category"], kw["search_volume_score"], now, now))
        updated += 1

    # Update performance scores based on our video analytics
    _update_performance_scores(conn)

    # Prune keywords not seen in 30 days
    cutoff = (datetime.now() - timedelta(days=30)).isoformat()
    conn.execute(
        "DELETE FROM trending_keywords WHERE last_fetched < ? AND times_used = 0",
        (cutoff,)
    )

    conn.commit()
    conn.close()

    logger.info(f"Updated {updated} keywords in database")
    print(f"  Updated {updated} keywords in database")


def _update_performance_scores(conn):
    """Update keyword performance scores based on video analytics.

    Links keywords to videos that used them (via title/tags) and
    scores them by average views per video.
    """
    # Get all published stories with views
    stories = conn.execute("""
        SELECT id, title, script, views
        FROM stories
        WHERE status = 'published' AND views > 0
    """).fetchall()

    if not stories:
        return

    # Calculate average views for normalization
    total_views = sum(s["views"] for s in stories)
    avg_views = total_views / len(stories) if stories else 1

    # For each keyword, find stories whose title contains it
    keywords = conn.execute(
        "SELECT id, keyword FROM trending_keywords"
    ).fetchall()

    for kw in keywords:
        keyword_lower = kw["keyword"].lower()
        matching_views = []

        for story in stories:
            title = (story["title"] or "").lower()
            script_text = (story["script"] or "").lower()

            if keyword_lower in title or keyword_lower in script_text:
                matching_views.append(story["views"])

        if matching_views:
            # Performance = average views of matching videos / overall average
            avg_matching = sum(matching_views) / len(matching_views)
            performance_score = min(avg_matching / max(avg_views, 1), 3.0)

            conn.execute("""
                UPDATE trending_keywords
                SET performance_score = ?,
                    combined_score = (search_volume_score + ?) / 2,
                    updated_at = ?
                WHERE id = ?
            """, (
                performance_score,
                performance_score,
                datetime.now().isoformat(),
                kw["id"],
            ))
        else:
            # No data yet - combined score is just search volume
            conn.execute("""
                UPDATE trending_keywords
                SET combined_score = search_volume_score,
                    updated_at = ?
                WHERE id = ?
            """, (datetime.now().isoformat(), kw["id"]))


def get_trending_keywords(
    count: int = 20,
    category: str = None,
    min_score: float = 0.0,
) -> list[dict]:
    """Get top trending keywords for use in titles/descriptions/tags.

    Returns keywords sorted by combined score (search volume + performance).
    """
    conn = _get_db()

    query = """
        SELECT keyword, category, search_volume_score, performance_score,
               combined_score, times_used
        FROM trending_keywords
        WHERE combined_score >= ?
    """
    params = [min_score]

    if category:
        query += " AND category = ?"
        params.append(category)

    query += " ORDER BY combined_score DESC LIMIT ?"
    params.append(count)

    results = conn.execute(query, params).fetchall()
    conn.close()

    return [dict(r) for r in results]


def optimize_title(
    base_title: str,
    moral: str = "",
    collection: str = "",
) -> str:
    """Enhance a video title with trending keywords.

    Injects high-performing keywords into the title while keeping
    it natural and under 70 characters for optimal CTR.
    """
    trending = get_trending_keywords(count=10, min_score=0.3)
    if not trending:
        return base_title[:70]

    # Pick a trending keyword that's relevant and fits
    keyword_additions = []
    for kw in trending:
        keyword = kw["keyword"]
        # Check if keyword is already in title
        if keyword.lower() in base_title.lower():
            continue
        # Check if it's relevant (contains story-related terms)
        if any(term in keyword.lower() for term in ["story", "tales", "kids", "moral", "bedtime"]):
            keyword_additions.append(keyword)

    if not keyword_additions:
        return base_title[:70]

    # Try to append a trending keyword phrase
    best_keyword = keyword_additions[0]
    # Extract just the distinctive part
    short_kw = best_keyword.replace("for kids", "").replace("stories", "").strip()

    enhanced = base_title
    if len(enhanced) + len(short_kw) + 3 <= 70 and short_kw:
        enhanced = f"{enhanced} | {short_kw.title()}"

    return enhanced[:70]


def optimize_tags(
    base_tags: list[str],
    story_script: dict = None,
    max_tags: int = 30,
) -> list[str]:
    """Enhance video tags with trending keywords from analytics.

    Merges base tags with trending keywords, prioritizing
    high-scoring keywords while staying under YouTube's 30-tag limit.
    """
    trending = get_trending_keywords(count=30)
    if not trending:
        return base_tags[:max_tags]

    # Start with base tags
    all_tags = list(base_tags)
    existing_lower = {t.lower() for t in all_tags}

    # Add trending keywords as tags
    for kw in trending:
        keyword = kw["keyword"]
        if keyword.lower() not in existing_lower and len(all_tags) < max_tags:
            all_tags.append(keyword)
            existing_lower.add(keyword.lower())

    # If we have story context, add theme-specific trending keywords
    if story_script:
        moral = story_script.get("moral", "").lower()
        collection = story_script.get("collection", "").lower()
        theme_keywords = get_trending_keywords(count=10, category="theme")
        for kw in theme_keywords:
            keyword = kw["keyword"]
            if (keyword.lower() not in existing_lower
                    and len(all_tags) < max_tags
                    and (moral in keyword.lower() or collection in keyword.lower())):
                all_tags.append(keyword)
                existing_lower.add(keyword.lower())

    return all_tags[:max_tags]


def optimize_description(
    base_description: str,
    story_script: dict = None,
) -> str:
    """Enhance video description with trending search keywords.

    Appends a keyword-rich section at the bottom of the description
    for SEO without cluttering the visible content.
    """
    trending = get_trending_keywords(count=15, min_score=0.2)
    if not trending:
        return base_description

    # Build keyword-rich footer
    keyword_phrases = [kw["keyword"] for kw in trending[:10]]
    keyword_section = "\n".join([
        "",
        "---",
        "Related searches:",
        ", ".join(keyword_phrases),
        "",
    ])

    # Stay under YouTube's 5000 char limit
    if len(base_description) + len(keyword_section) <= 5000:
        return base_description + keyword_section

    return base_description


def track_keyword_usage(story_id: int, keywords_used: list[str], used_in: str = "tags"):
    """Track which keywords were used in a video for performance correlation."""
    conn = _get_db()
    now = datetime.now().isoformat()

    for keyword in keywords_used:
        conn.execute(
            "INSERT INTO keyword_usage (story_id, keyword, used_in, created_at) VALUES (?, ?, ?, ?)",
            (story_id, keyword.lower(), used_in, now)
        )
        conn.execute(
            "UPDATE trending_keywords SET times_used = times_used + 1 WHERE keyword = ?",
            (keyword.lower(),)
        )

    conn.commit()
    conn.close()


def run_keyword_refresh():
    """Scheduled job: refresh keyword database with latest trends.

    Call this from the scheduler every 6-12 hours to keep
    the keyword database current with real search trends.
    """
    logger.info("Starting keyword refresh...")
    print("  Starting keyword refresh...")

    try:
        update_keyword_database()

        # Log top keywords
        top = get_trending_keywords(count=5)
        if top:
            print("  Top 5 trending keywords:")
            for kw in top:
                print(f"    - {kw['keyword']} (score: {kw['combined_score']:.2f})")

        logger.info("Keyword refresh complete")
        print("  Keyword refresh complete")

    except Exception as e:
        logger.error(f"Keyword refresh failed: {e}")
        print(f"  Keyword refresh failed: {e}")


if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    if len(sys.argv) > 1 and sys.argv[1] == "refresh":
        run_keyword_refresh()
    elif len(sys.argv) > 1 and sys.argv[1] == "top":
        count = int(sys.argv[2]) if len(sys.argv) > 2 else 20
        keywords = get_trending_keywords(count=count)
        print(f"\nTop {len(keywords)} Trending Keywords:")
        print("-" * 60)
        for i, kw in enumerate(keywords, 1):
            print(f"  {i:2d}. {kw['keyword']:<40s} score: {kw['combined_score']:.2f} "
                  f"(search: {kw['search_volume_score']:.2f}, perf: {kw['performance_score']:.2f})")
    else:
        print("Usage: python keyword_optimizer.py [refresh|top [count]]")
        print("\nTesting keyword discovery...")
        print("=" * 60)

        # Test with a few seeds
        test_suggestions = fetch_youtube_suggestions("kids stories")
        print(f"\nSuggestions for 'kids stories':")
        for s in test_suggestions:
            print(f"  - {s}")

        test_suggestions = fetch_youtube_suggestions("bedtime stories for")
        print(f"\nSuggestions for 'bedtime stories for':")
        for s in test_suggestions:
            print(f"  - {s}")

        print("\n" + "=" * 60)
        print("Test complete!")
