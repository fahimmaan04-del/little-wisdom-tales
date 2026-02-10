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

# Seed keywords organized by content type and language.
# Each channel type gets its own relevant seeds for YouTube suggest expansion.
SEED_KEYWORDS_BY_TYPE = {
    "story": {
        "en": [
            "kids stories", "bedtime stories", "moral stories for kids",
            "stories for children", "animated stories", "fairy tales for kids",
            "kids moral lessons", "children bedtime stories", "story time for kids",
            "nursery stories", "panchatantra stories", "aesop fables for kids",
            "african folktales for kids", "arabian nights stories",
            "animal stories for kids", "kids learning stories",
            "stories about kindness", "stories about honesty",
            "stories about courage", "stories about friendship",
            "short stories for kids", "educational stories",
            "fun stories for toddlers", "preschool stories",
        ],
        "hi": [
            "बच्चों की कहानियां", "नैतिक कहानियां", "सोने की कहानियां",
            "पंचतंत्र की कहानियां", "ईसप की कहानियां", "हिंदी कहानियां बच्चों के लिए",
            "भारतीय लोककथाएं", "परियों की कहानियां", "जानवरों की कहानियां",
            "kids stories hindi", "moral stories in hindi", "bedtime stories hindi",
            "panchatantra stories hindi", "hindi kahaniya",
        ],
        "es": [
            "cuentos para niños", "cuentos infantiles", "historias morales para niños",
            "cuentos para dormir", "fábulas para niños", "cuentos animados",
            "cuentos de aventura", "cuentos latinoamericanos", "historias educativas",
            "kids stories spanish", "cuentos con moraleja",
        ],
        "fr": [
            "histoires pour enfants", "contes pour enfants", "histoires morales",
            "histoires du soir", "fables de la fontaine", "contes animés",
            "histoires animées pour enfants", "contes d'aventure",
            "kids stories french", "histoires éducatives",
        ],
        "pt": [
            "histórias para crianças", "contos infantis", "histórias morais",
            "histórias para dormir", "fábulas para crianças", "contos animados",
            "histórias educativas", "contos de aventura",
            "kids stories portuguese", "contos populares brasileiros",
        ],
        "ar": [
            "قصص أطفال", "قصص أخلاقية للأطفال", "قصص قبل النوم",
            "ألف ليلة وليلة للأطفال", "قصص إسلامية للأطفال", "قصص عربية",
            "حكايات للأطفال", "قصص مغامرات أطفال",
            "kids stories arabic", "arabic stories for kids",
        ],
    },
    "education": {
        "en": [
            "kids education", "class 1 maths", "class 2 science",
            "kids learning videos", "primary school lessons", "fun math for kids",
            "science experiments for kids", "english grammar for kids",
            "educational videos for children", "learn with animation",
            "oxford curriculum class 1", "cambridge primary maths",
            "kids counting", "learn shapes for kids", "alphabet for kids",
            "multiplication for kids", "solar system for kids",
            "parts of body for kids", "animals for kids learning",
        ],
        "hi": [
            "बच्चों की पढ़ाई", "कक्षा 1 गणित", "कक्षा 2 विज्ञान",
            "बच्चों के लिए शिक्षा", "हिंदी वर्णमाला", "गिनती सीखें",
            "kids education hindi", "class 1 maths hindi",
            "science for kids hindi", "learn hindi alphabets",
        ],
        "es": [
            "educación para niños", "matemáticas para niños", "ciencias para niños",
            "clase 1 matemáticas", "aprender con animación", "videos educativos niños",
            "kids education spanish", "aprende a sumar",
        ],
        "ar": [
            "تعليم الأطفال", "رياضيات للأطفال", "علوم للأطفال",
            "الأبجدية العربية", "تعلم الأرقام", "kids education arabic",
        ],
    },
    "ai_education": {
        "en": [
            "AI for kids", "coding for kids", "scratch programming",
            "python for kids", "build a chatbot", "make art with AI",
            "AI games for kids", "learn coding for beginners",
            "robotics for kids", "kids programming", "AI tutorial kids",
            "machine learning for kids", "create apps for kids",
            "technology for children", "STEM for kids",
        ],
        "hi": [
            "बच्चों के लिए AI", "बच्चों के लिए कोडिंग", "scratch programming hindi",
            "AI for kids hindi", "coding for beginners hindi",
            "kids programming hindi", "AI सीखें",
        ],
    },
    "crafts_skills": {
        "en": [
            "carpentry for kids", "woodworking for beginners", "how things work for kids",
            "plumbing for kids", "electricity for kids", "how circuits work",
            "building for kids", "DIY projects for kids", "kids workshop",
            "how to build things", "tools for kids", "home repair for kids",
            "how toilets work", "how light bulbs work", "simple circuits for kids",
            "painting for kids", "measure and build", "safety first for kids",
            "famous builders", "how houses are built", "solar power for kids",
            "electrical safety for kids", "kids crafts and building",
            "how motors work for kids", "kids STEM building projects",
        ],
    },
}

# Backwards compatibility: flat list of English story seeds (used if no type/lang specified)
SEED_KEYWORDS = SEED_KEYWORDS_BY_TYPE["story"]["en"]

# Language-to-region mapping for YouTube suggest API (hl= and gl= params)
LANGUAGE_REGION_MAP = {
    "en": {"hl": "en", "gl": "US"},
    "hi": {"hl": "hi", "gl": "IN"},
    "es": {"hl": "es", "gl": "MX"},
    "fr": {"hl": "fr", "gl": "FR"},
    "pt": {"hl": "pt", "gl": "BR"},
    "ar": {"hl": "ar", "gl": "SA"},
}

# Categories to organize keywords
KEYWORD_CATEGORIES = {
    "story_type": ["bedtime", "moral", "fairy tale", "animated", "short",
                    "सोने", "नैतिक", "cuentos", "histoires", "histórias", "قصص"],
    "audience": ["kids", "children", "toddlers", "preschool", "family",
                 "बच्चों", "niños", "enfants", "crianças", "أطفال"],
    "theme": ["kindness", "honesty", "courage", "friendship", "wisdom", "sharing"],
    "format": ["stories", "tales", "adventures", "lessons", "fables",
               "कहानियां", "cuentos", "contes", "contos", "حكايات"],
    "education": ["class", "maths", "science", "english", "learn", "education",
                  "कक्षा", "गणित", "विज्ञान", "clase", "matemáticas", "تعليم"],
    "ai_coding": ["AI", "coding", "programming", "scratch", "python", "robot",
                  "chatbot", "app", "STEM"],
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
            keyword TEXT NOT NULL,
            category TEXT DEFAULT 'general',
            language TEXT DEFAULT 'en',
            content_type TEXT DEFAULT 'story',
            search_volume_score REAL DEFAULT 0,
            performance_score REAL DEFAULT 0,
            combined_score REAL DEFAULT 0,
            times_used INTEGER DEFAULT 0,
            last_fetched TIMESTAMP,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(keyword, language, content_type)
        )
    """)

    # Migrate: add language/content_type columns if missing (existing DBs)
    columns = [row[1] for row in conn.execute("PRAGMA table_info(trending_keywords)").fetchall()]
    if "language" not in columns:
        conn.execute("ALTER TABLE trending_keywords ADD COLUMN language TEXT DEFAULT 'en'")
    if "content_type" not in columns:
        conn.execute("ALTER TABLE trending_keywords ADD COLUMN content_type TEXT DEFAULT 'story'")

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


def fetch_youtube_suggestions(query: str, language: str = "en") -> list[str]:
    """Fetch YouTube search autocomplete suggestions for a query.

    Uses Google's suggestion API (same as YouTube search bar).
    The hl and gl params localize results to the target language/region.
    Returns list of suggested search terms.
    """
    try:
        region = LANGUAGE_REGION_MAP.get(language, {"hl": "en", "gl": "US"})
        url = "https://suggestqueries.google.com/complete/search"
        params = {
            "client": "youtube",
            "q": query,
            "ds": "yt",
            "hl": region["hl"],
            "gl": region["gl"],
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
        logger.warning(f"Failed to fetch suggestions for '{query}' ({language}): {e}")
        return []


def discover_trending_keywords(
    content_type: str = "story",
    language: str = "en",
) -> list[dict]:
    """Discover trending keywords from YouTube search suggestions.

    Expands seed keywords through the suggestion API to find
    what kids/parents are actually searching for right now.
    Uses content_type and language to select the right seed keywords
    and localize API requests to the target region.
    """
    seeds = SEED_KEYWORDS_BY_TYPE.get(content_type, {}).get(language)
    if not seeds:
        # Fall back to English seeds for this content type, then to default
        seeds = SEED_KEYWORDS_BY_TYPE.get(content_type, {}).get("en", SEED_KEYWORDS)
        logger.info(f"No seeds for {content_type}/{language}, falling back to English")

    all_keywords = {}
    total_fetched = 0

    for seed in seeds:
        # Rate limit: 0.5s between requests
        time.sleep(0.5)

        suggestions = fetch_youtube_suggestions(seed, language=language)
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
                    "language": language,
                    "content_type": content_type,
                }
                total_fetched += 1

        # Also try partial queries for deeper suggestions
        for prefix in [seed + " for", seed + " about", seed + " with"]:
            time.sleep(0.3)
            extra_suggestions = fetch_youtube_suggestions(prefix, language=language)
            for suggestion in extra_suggestions:
                suggestion_lower = suggestion.lower().strip()
                if suggestion_lower not in all_keywords:
                    position_score = 0.5 - (extra_suggestions.index(suggestion) / max(len(extra_suggestions), 1)) * 0.5
                    all_keywords[suggestion_lower] = {
                        "keyword": suggestion_lower,
                        "search_volume_score": max(0.1, position_score),
                        "category": _categorize_keyword(suggestion_lower),
                        "source_seed": prefix,
                        "language": language,
                        "content_type": content_type,
                    }
                    total_fetched += 1

    logger.info(f"Discovered {total_fetched} trending keywords for {content_type}/{language}")
    print(f"  Discovered {total_fetched} trending keywords ({content_type}/{language})")

    return list(all_keywords.values())


def _categorize_keyword(keyword: str) -> str:
    """Auto-categorize a keyword based on content."""
    keyword_lower = keyword.lower()
    for category, terms in KEYWORD_CATEGORIES.items():
        for term in terms:
            if term in keyword_lower:
                return category
    return "general"


def update_keyword_database(
    keywords: list[dict] = None,
    content_type: str = "story",
    language: str = "en",
):
    """Update the trending keywords database.

    Fetches new keywords if none provided, updates scores
    for existing keywords, and prunes stale entries.
    """
    if keywords is None:
        keywords = discover_trending_keywords(content_type=content_type, language=language)

    conn = _get_db()
    now = datetime.now().isoformat()
    updated = 0

    for kw in keywords:
        lang = kw.get("language", language)
        ctype = kw.get("content_type", content_type)
        conn.execute("""
            INSERT INTO trending_keywords
                (keyword, category, language, content_type, search_volume_score, last_fetched, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(keyword, language, content_type) DO UPDATE SET
                search_volume_score = (search_volume_score + excluded.search_volume_score) / 2,
                category = excluded.category,
                last_fetched = excluded.last_fetched,
                updated_at = excluded.updated_at
        """, (kw["keyword"], kw["category"], lang, ctype, kw["search_volume_score"], now, now))
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

    logger.info(f"Updated {updated} keywords in database ({content_type}/{language})")
    print(f"  Updated {updated} keywords ({content_type}/{language})")


def _update_performance_scores(conn):
    """Update keyword performance scores based on video analytics.

    Uses a weighted scoring model:
      - 30% search volume (YouTube suggest position score)
      - 30% video performance (views relative to average)
      - 20% CTR (impression click-through rate from Analytics API)
      - 20% watch time (estimated minutes watched, quality signal)

    Falls back to simple search_volume if no analytics data is available.
    """
    # Get all published stories with views
    stories = conn.execute("""
        SELECT id, title, script, views
        FROM stories
        WHERE status = 'published' AND views > 0
    """).fetchall()

    # Try to get real analytics data (CTR, watch time)
    video_analytics = {}
    try:
        analytics_rows = conn.execute("""
            SELECT video_id,
                   AVG(impression_ctr) as avg_ctr,
                   SUM(estimated_minutes_watched) as total_watch_time,
                   SUM(views) as total_views,
                   AVG(average_view_duration) as avg_duration
            FROM video_analytics
            GROUP BY video_id
        """).fetchall()
        for row in analytics_rows:
            video_analytics[row["video_id"]] = dict(row)
    except Exception:
        pass  # video_analytics table may not exist yet

    if not stories:
        # No stories yet - combined score is just search volume
        conn.execute("""
            UPDATE trending_keywords
            SET combined_score = search_volume_score,
                updated_at = ?
        """, (datetime.now().isoformat(),))
        return

    # Calculate averages for normalization
    total_views = sum(s["views"] for s in stories)
    avg_views = total_views / len(stories) if stories else 1

    avg_ctr = 0.0
    avg_watch_time = 0.0
    if video_analytics:
        ctr_values = [v["avg_ctr"] for v in video_analytics.values() if v.get("avg_ctr")]
        watch_values = [v["total_watch_time"] for v in video_analytics.values() if v.get("total_watch_time")]
        if ctr_values:
            avg_ctr = sum(ctr_values) / len(ctr_values)
        if watch_values:
            avg_watch_time = sum(watch_values) / len(watch_values)

    # Build video_id -> story mapping for analytics lookup
    story_video_map = {}
    for story in stories:
        script_data = story["script"] or ""
        if isinstance(script_data, str):
            try:
                parsed = json.loads(script_data)
                vid = parsed.get("video_id", "")
                if vid:
                    story_video_map[vid] = story
            except (json.JSONDecodeError, TypeError):
                pass

    # For each keyword, calculate weighted score
    keywords = conn.execute(
        "SELECT id, keyword FROM trending_keywords"
    ).fetchall()

    for kw in keywords:
        keyword_lower = kw["keyword"].lower()
        matching_views = []
        matching_ctrs = []
        matching_watch_times = []

        for story in stories:
            title = (story["title"] or "").lower()
            script_text = (story["script"] or "").lower()

            if keyword_lower in title or keyword_lower in script_text:
                matching_views.append(story["views"])

                # Look up real analytics for this video
                for vid, analytics in video_analytics.items():
                    # Check if this analytics record belongs to a matching story
                    if analytics.get("avg_ctr"):
                        matching_ctrs.append(analytics["avg_ctr"])
                    if analytics.get("total_watch_time"):
                        matching_watch_times.append(analytics["total_watch_time"])

        if matching_views:
            # Performance score: views relative to average (0-3 scale)
            avg_matching = sum(matching_views) / len(matching_views)
            perf_score = min(avg_matching / max(avg_views, 1), 3.0)

            # CTR score: relative to channel average (0-3 scale)
            ctr_score = 0.0
            if matching_ctrs and avg_ctr > 0:
                avg_matching_ctr = sum(matching_ctrs) / len(matching_ctrs)
                ctr_score = min(avg_matching_ctr / max(avg_ctr, 0.001), 3.0)

            # Watch time score: relative to average (0-3 scale)
            watch_score = 0.0
            if matching_watch_times and avg_watch_time > 0:
                avg_matching_watch = sum(matching_watch_times) / len(matching_watch_times)
                watch_score = min(avg_matching_watch / max(avg_watch_time, 0.001), 3.0)

            # Weighted combined score (normalized to 0-1 range)
            # 30% search volume + 30% performance + 20% CTR + 20% watch time
            search_vol = conn.execute(
                "SELECT search_volume_score FROM trending_keywords WHERE id = ?",
                (kw["id"],)
            ).fetchone()
            sv_score = search_vol["search_volume_score"] if search_vol else 0

            combined = (
                0.30 * sv_score +
                0.30 * (perf_score / 3.0) +
                0.20 * (ctr_score / 3.0) +
                0.20 * (watch_score / 3.0)
            )

            conn.execute("""
                UPDATE trending_keywords
                SET performance_score = ?,
                    combined_score = ?,
                    updated_at = ?
                WHERE id = ?
            """, (perf_score, combined, datetime.now().isoformat(), kw["id"]))
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
    language: str = None,
    content_type: str = None,
) -> list[dict]:
    """Get top trending keywords for use in titles/descriptions/tags.

    Returns keywords sorted by combined score (search volume + performance).
    Filters by language and content_type when provided.
    """
    conn = _get_db()

    query = """
        SELECT keyword, category, language, content_type,
               search_volume_score, performance_score,
               combined_score, times_used
        FROM trending_keywords
        WHERE combined_score >= ?
    """
    params = [min_score]

    if category:
        query += " AND category = ?"
        params.append(category)

    if language:
        query += " AND language = ?"
        params.append(language)

    if content_type:
        query += " AND content_type = ?"
        params.append(content_type)

    query += " ORDER BY combined_score DESC LIMIT ?"
    params.append(count)

    results = conn.execute(query, params).fetchall()
    conn.close()

    return [dict(r) for r in results]


def optimize_title(
    base_title: str,
    moral: str = "",
    collection: str = "",
    language: str = "en",
    content_type: str = "story",
) -> str:
    """Enhance a video title with trending keywords.

    Injects high-performing keywords into the title while keeping
    it natural and under 70 characters for optimal CTR.
    Uses language and content_type to pull relevant trending keywords.
    """
    trending = get_trending_keywords(
        count=10, min_score=0.3, language=language, content_type=content_type,
    )
    if not trending:
        return base_title[:70]

    # Relevance terms vary by content type
    relevance_terms = {
        "story": ["story", "tales", "kids", "moral", "bedtime",
                   "कहानियां", "cuentos", "histoires", "histórias", "قصص"],
        "education": ["class", "learn", "education", "maths", "science",
                      "कक्षा", "clase", "aprender", "تعليم"],
        "ai_education": ["AI", "coding", "programming", "scratch", "python",
                         "robot", "app", "game"],
    }
    terms = relevance_terms.get(content_type, relevance_terms["story"])

    # Pick a trending keyword that's relevant and fits
    keyword_additions = []
    for kw in trending:
        keyword = kw["keyword"]
        if keyword.lower() in base_title.lower():
            continue
        if any(term.lower() in keyword.lower() for term in terms):
            keyword_additions.append(keyword)

    if not keyword_additions:
        return base_title[:70]

    # Try to append a trending keyword phrase
    best_keyword = keyword_additions[0]
    # Extract just the distinctive part (strip common generic words)
    strip_words = ["for kids", "stories", "for children", "para niños",
                   "pour enfants", "para crianças", "للأطفال", "बच्चों के लिए"]
    short_kw = best_keyword
    for sw in strip_words:
        short_kw = short_kw.replace(sw, "")
    short_kw = short_kw.strip()

    enhanced = base_title
    if len(enhanced) + len(short_kw) + 3 <= 70 and short_kw:
        enhanced = f"{enhanced} | {short_kw.title()}"

    return enhanced[:70]


def optimize_tags(
    base_tags: list[str],
    story_script: dict = None,
    max_tags: int = 30,
    language: str = "en",
    content_type: str = "story",
) -> list[str]:
    """Enhance video tags with trending keywords from analytics.

    Merges base tags with trending keywords, prioritizing
    high-scoring keywords while staying under YouTube's 30-tag limit.
    """
    trending = get_trending_keywords(
        count=30, language=language, content_type=content_type,
    )
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
        theme_keywords = get_trending_keywords(
            count=10, category="theme", language=language, content_type=content_type,
        )
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
    language: str = "en",
    content_type: str = "story",
) -> str:
    """Enhance video description with trending search keywords.

    Appends a keyword-rich section at the bottom of the description
    for SEO without cluttering the visible content.
    """
    trending = get_trending_keywords(
        count=15, min_score=0.2, language=language, content_type=content_type,
    )
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


def integrate_analytics_data():
    """Inject real YouTube search terms from Analytics API into keyword database.

    Fetches actual search queries that drove traffic to our channels
    and adds them as high-confidence keywords with boosted scores.
    These are real viewer search behaviors, not just suggestions.
    """
    conn = _get_db()
    now = datetime.now().isoformat()
    injected = 0

    try:
        # Get search terms from analytics database
        search_terms = conn.execute("""
            SELECT search_term, SUM(views) as total_views,
                   SUM(estimated_minutes_watched) as total_watch_time
            FROM search_terms
            WHERE date >= ?
            GROUP BY search_term
            ORDER BY total_views DESC
            LIMIT 200
        """, ((datetime.now() - timedelta(days=28)).isoformat(),)).fetchall()

        if not search_terms:
            logger.info("No search terms in analytics database yet")
            conn.close()
            return

        # Normalize: max views gets score 1.0
        max_views = max(t["total_views"] for t in search_terms) if search_terms else 1

        for term in search_terms:
            keyword = term["search_term"].lower().strip()
            if not keyword or len(keyword) < 3:
                continue

            # Score based on views (real traffic = high confidence)
            volume_score = min(term["total_views"] / max(max_views, 1), 1.0)
            # Boost: real search terms get a 0.2 bonus over suggest-only terms
            boosted_score = min(volume_score + 0.2, 1.0)

            category = _categorize_keyword(keyword)

            conn.execute("""
                INSERT INTO trending_keywords
                    (keyword, category, language, content_type,
                     search_volume_score, performance_score, combined_score,
                     last_fetched, updated_at)
                VALUES (?, ?, 'en', 'story', ?, ?, ?, ?, ?)
                ON CONFLICT(keyword, language, content_type) DO UPDATE SET
                    search_volume_score = MAX(search_volume_score, excluded.search_volume_score),
                    performance_score = MAX(performance_score, excluded.performance_score),
                    combined_score = MAX(combined_score, excluded.combined_score),
                    last_fetched = excluded.last_fetched,
                    updated_at = excluded.updated_at
            """, (keyword, category, boosted_score, volume_score, boosted_score, now, now))
            injected += 1

        conn.commit()
        logger.info(f"Integrated {injected} real search terms into keyword database")
        print(f"  Integrated {injected} real search terms from Analytics API")

    except Exception as e:
        logger.warning(f"Analytics integration failed: {e}")
    finally:
        conn.close()


def run_keyword_refresh():
    """Scheduled job: refresh keyword database with latest trends for ALL channels.

    Iterates over all content_type/language combinations that have seed keywords
    defined, fetching fresh YouTube suggest data for each. Call this from the
    scheduler every 6-12 hours to keep the keyword database current.
    """
    logger.info("Starting keyword refresh for all channels...")
    print("  Starting keyword refresh for all channels...")

    total_updated = 0
    refresh_targets = []

    # Build list of all (content_type, language) pairs that have seeds
    for ctype, lang_dict in SEED_KEYWORDS_BY_TYPE.items():
        for lang in lang_dict:
            refresh_targets.append((ctype, lang))

    for ctype, lang in refresh_targets:
        try:
            update_keyword_database(content_type=ctype, language=lang)
            top = get_trending_keywords(count=3, language=lang, content_type=ctype)
            if top:
                print(f"  Top for {ctype}/{lang}: {', '.join(kw['keyword'][:30] for kw in top)}")
            total_updated += 1
        except Exception as e:
            logger.warning(f"Keyword refresh failed for {ctype}/{lang}: {e}")

    logger.info(f"Keyword refresh complete: {total_updated}/{len(refresh_targets)} channels refreshed")
    print(f"  Keyword refresh complete: {total_updated} channel types refreshed")


def run_keyword_refresh_single(content_type: str = "story", language: str = "en"):
    """Refresh keywords for a single content_type/language pair."""
    logger.info(f"Refreshing keywords for {content_type}/{language}...")
    try:
        update_keyword_database(content_type=content_type, language=language)
        top = get_trending_keywords(count=5, language=language, content_type=content_type)
        if top:
            print(f"  Top 5 trending keywords ({content_type}/{language}):")
            for kw in top:
                print(f"    - {kw['keyword']} (score: {kw['combined_score']:.2f})")
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
        # refresh [content_type] [language]  OR  refresh --all
        if len(sys.argv) > 2 and sys.argv[2] == "--all":
            run_keyword_refresh()
        elif len(sys.argv) > 3:
            run_keyword_refresh_single(content_type=sys.argv[2], language=sys.argv[3])
        elif len(sys.argv) > 2:
            run_keyword_refresh_single(content_type=sys.argv[2])
        else:
            run_keyword_refresh()
    elif len(sys.argv) > 1 and sys.argv[1] == "top":
        count = int(sys.argv[2]) if len(sys.argv) > 2 else 20
        lang = sys.argv[3] if len(sys.argv) > 3 else None
        ctype = sys.argv[4] if len(sys.argv) > 4 else None
        keywords = get_trending_keywords(count=count, language=lang, content_type=ctype)
        label = f" ({ctype}/{lang})" if lang or ctype else ""
        print(f"\nTop {len(keywords)} Trending Keywords{label}:")
        print("-" * 70)
        for i, kw in enumerate(keywords, 1):
            print(f"  {i:2d}. {kw['keyword']:<40s} score: {kw['combined_score']:.2f} "
                  f"(search: {kw['search_volume_score']:.2f}, perf: {kw['performance_score']:.2f}) "
                  f"[{kw.get('content_type', '?')}/{kw.get('language', '?')}]")
    else:
        print("Usage:")
        print("  python keyword_optimizer.py refresh              # Refresh ALL channels")
        print("  python keyword_optimizer.py refresh story en     # Refresh English stories only")
        print("  python keyword_optimizer.py refresh education hi # Refresh Hindi education only")
        print("  python keyword_optimizer.py top [count] [lang] [type]  # Show top keywords")
        print("\nTesting keyword discovery...")
        print("=" * 60)

        # Test multi-language
        for lang, query in [("en", "kids stories"), ("hi", "बच्चों की कहानियां"), ("es", "cuentos para niños")]:
            suggestions = fetch_youtube_suggestions(query, language=lang)
            print(f"\nSuggestions for '{query}' ({lang}):")
            for s in suggestions[:5]:
                print(f"  - {s}")

        print("\n" + "=" * 60)
        print("Test complete!")
