# Kids-Heaven - Project Instructions

<!-- AUTO-MANAGED: project-description -->
## Project Overview
Fully automated multi-channel YouTube pipeline that creates and publishes kid-friendly educational content 24/7 across 21 channels without human intervention.

**Pipeline**: Script generation (LLM) → Voiceover (Edge TTS, multi-language) → AI images (SDXL Turbo) → Video assembly (FFmpeg) → YouTube upload (Data API v3)

**Content Types**:
- Moral stories (6 languages: EN, HI, ES, FR, PT, AR)
- Education videos (Oxford/Cambridge curricula, Classes 1-10)
- AI education (coding, robotics, tech for kids - Byte the Robot host)
- Crafts & skills (carpentry, plumbing, electrical - Handy the Helper host)

**Host Characters**:
- Professor Wisdom (friendly owl) for education content
- Byte the Robot (blue robot kid) for AI/tech lessons
- Handy the Helper (toolbelt character) for crafts/skills

**Target**: Kids aged 4-14, Made for Kids compliance, 116 videos/day across all channels
<!-- END AUTO-MANAGED -->

<!-- AUTO-MANAGED: architecture -->
## Architecture

### Pipeline Components
- **LLM**: Ollama llama3.2:3b (localhost:11434) for script generation (stories, education, AI lessons)
- **TTS**: Edge TTS with regional voices (EN, HI, ES, FR, PT, AR) + cartoon pitch shifting via FFmpeg
- **Translation**: deep-translator (GoogleTranslator) for multi-language content localization
- **Images**: SDXL Turbo on local GPU (RTX A5000, 24GB VRAM) → 30 images/scene with crossfades
  - Style prefixes by content_type:
    - STYLE_PREFIX: Stories (Pixar/Disney cartoon style)
    - EDUCATION_STYLE_PREFIX: Education (classroom with Professor Wisdom owl)
    - AI_EDUCATION_STYLE_PREFIX: AI lessons (futuristic lab with Byte the Robot)
    - CRAFTS_SKILLS_STYLE_PREFIX: Crafts/skills (workshop with Handy the Helper)
- **Subtitles**: Auto-generated multi-language SRT files with word-level timing
- **SEO**: Keyword optimizer for titles, descriptions, tags (language-aware)
- **Video**: FFmpeg 6.1 with NVENC, multi-image crossfade animation, PIL-rendered text overlays
- **Upload**: YouTube Data API v3 with quota management (10,000 units/day per channel, 1600/upload)
- **Scheduler**: Phased batch pipeline (scripts → audio → images → videos → uploads)
- **Orchestration**: n8n workflow automation (optional, docker-compose)

### Content Types & Host Characters
- **Stories**: Moral tales from world folklore (Aesop, Panchatantra, Brothers Grimm, Arabian Nights, etc.)
- **Education**: Oxford/Cambridge curricula for Classes 1-10 (Math, Science, English, GK) - Professor Wisdom (owl)
- **AI Education**: Coding, AI, robotics, tech lessons for kids - Byte the Robot (blue robot kid)
- **Crafts & Skills**: Carpentry, plumbing, electrical, hands-on skills - Handy the Helper (toolbelt character)

### Key Scripts
**Story Pipeline:**
- `orchestrator.py`: Master pipeline coordinator (story → TTS → images → video → upload)
- `story_generator.py`: LLM-based story script generation with SQLite tracking
- `engagement_hooks.py`: Title/description optimization, subscribe CTAs

**Education Pipeline:**
- `education_orchestrator.py`: Education video pipeline (lesson → TTS → images → video → upload)
- `education_generator.py`: LLM-based lesson script generation from Oxford/Cambridge syllabus (Professor Wisdom)
- `ai_content_generator.py`: AI curriculum lesson builder (deprecated: use `ai_education_generator.py`)
- `ai_education_generator.py`: AI/coding/robotics lesson generation (Byte the Robot character)
- `crafts_skills_generator.py`: Hands-on skill lesson generation - carpentry, plumbing, electrical (Handy the Helper)

**Multi-Channel Management:**
- `batch_pipeline.py`: High-throughput batch generation (100+ videos/day, phased execution)
- `channel_manager.py`: Multi-channel OAuth, quota tracking, playlist management
- `regional_content.py`: Multi-language translation and TTS voice selection

**Content Generation:**
- `tts_generator.py`: Edge TTS with async generation + FFmpeg cartoon pitch effects
- `ai_image_generator.py`: SDXL Turbo GPU image generation with content_type variants
- `video_assembler.py`: FFmpeg video assembly with Ken Burns effects + crossfades + subtitles
- `subtitle_generator.py`: Multi-language subtitle generation with word-level timing
- `keyword_optimizer.py`: SEO optimization for titles, descriptions, tags

**Utilities:**
- `scheduler.py`: 24/7 autonomous scheduler managing 21 channels (batch/incremental modes)
- `youtube_manager.py`: YouTube Data API v3 upload, analytics tracking
- `youtube_analytics.py`: YouTube Analytics API v2 metrics and search term tracking
- `reupload_videos.py`: Re-upload system for failed uploads
- `quick_setup.py`: Interactive OAuth setup for all channels
- `setup_channels.sh`: Bash script for batch channel authentication
- `image_fetcher.py`: Pixabay/Pexels fallback for stock images

### Directory Structure
```
scripts/          # Python pipeline modules
config/           # Channel configs, syllabi, curricula
  channels.json           # 21 YouTube channels (6 story + 7 education + 4 AI + 3 crafts + 1 existing)
  education_syllabus.json # Oxford/Cambridge Classes 1-10 curricula
  ai_kids_curriculum.json # AI education lesson catalog
  crafts_skills_curriculum.json # Crafts & skills lesson catalog
  story_themes.json       # Story collections and themes
  voices.json             # TTS voice mappings
output/           # Generated content
  audio/          # TTS voiceover files (WAV, multi-language)
  images/         # AI-generated scene images (JPEG)
  videos/         # Final videos (MP4)
  thumbnails/     # Video thumbnails
  subtitles/      # SRT subtitle files
data/             # SQLite databases, logs
  stories.db      # Story + education lesson scripts + analytics tracking
  channels.db     # Multi-channel upload quota tracking
  logs/           # Pipeline execution logs
assets/           # Static assets (fonts, music, intros)
n8n-workflows/    # Workflow automation configs
systemd/          # Service definitions
```
<!-- END AUTO-MANAGED -->

<!-- AUTO-MANAGED: build-commands -->
## Key Commands

### Autonomous Pipeline (Recommended)
```bash
# Start 24/7 multi-channel pipeline (batch mode, phased generation - DEFAULT)
./start_pipeline.sh
# or explicitly:
./start_pipeline.sh batch

# Start incremental mode (one-at-a-time generation)
./start_pipeline.sh incremental

# Check pipeline status (channel stats, progress, logs)
./start_pipeline.sh status
```

### Single Content Operations
```bash
# Test single story (no upload)
python scripts/orchestrator.py test

# Create and upload single story
python scripts/orchestrator.py single

# Create single education video
python scripts/education_orchestrator.py single

# Create single AI education video
python scripts/ai_education_generator.py single

# Create single crafts/skills video
python scripts/crafts_skills_generator.py single
```

### Batch Operations
```bash
# Generate 100+ videos across all channels (phased)
python scripts/batch_pipeline.py daily

# View batch pipeline stats
python scripts/batch_pipeline.py stats

# Re-upload failed videos
python scripts/reupload_videos.py
```

### Channel Setup
```bash
# Interactive OAuth setup for all 21 channels
python scripts/quick_setup.py

# List all channels and auth status
python scripts/quick_setup.py --list

# Test authenticated channels
python scripts/quick_setup.py --test

# Batch setup via shell script
bash scripts/setup_channels.sh
```

### Service Management
```bash
# Start 24/7 service
sudo systemctl start kids-story-pipeline

# Check service status
sudo systemctl status kids-story-pipeline

# View logs
journalctl -u kids-story-pipeline -f
```

### Docker Deployment
```bash
# Start n8n + Ollama + worker + scheduler
docker-compose up -d

# View logs
docker-compose logs -f worker

# Stop all services
docker-compose down
```

### Virtual Environment
```bash
source /mnt/projects/youtube/venv/bin/activate
```
<!-- END AUTO-MANAGED -->

<!-- MANUAL -->
## Important Constraints
- FFmpeg has NO libmp3lame (use WAV/AAC), NO libfreetype (use PIL for text)
- torch 2.5.1+cu124 in venv (isolated from system CUDA 12.8)
- YouTube Made for Kids = no comments, limited features
- SDXL Turbo generates at 512x512 native, resized to 1024x576 for video

## Session Protocol
1. Read `SESSION_MEMORY.md` first
2. Read `.claude/TASKS.md` for pending work
3. Check `data/logs/orchestrator.log` for recent pipeline status
<!-- END MANUAL -->

<!-- AUTO-MANAGED: conventions -->
## Conventions

### Code Organization
- All pipeline scripts in `scripts/` directory with `__init__.py`
- Configuration files in `config/` as JSON (channels.json, education_syllabus.json, etc.)
- Environment variables in `.env` (copy from `.env.example`)
- Databases: SQLite at `data/` (stories.db, channels.db)
  - `stories.db`: Unified content database with stories, education_lessons, ai_education_lessons, crafts_skills_lessons, video_analytics, search_terms, channel_analytics tables
  - `channels.db`: Multi-channel upload quota tracking (channel_uploads) + playlist ID cache (playlist_cache)

### Python Patterns
- Use `from dotenv import load_dotenv` + `load_dotenv()` at module top
- Async functions for I/O-bound operations (TTS generation uses `async def` + Edge TTS)
- Path objects from `pathlib.Path` for all file operations
- Environment variable access: `Path(os.getenv("OUTPUT_DIR", "./output"))`
- Script metadata includes `content_type` field: "story", "education", "ai_education", "crafts_skills"

### Host Characters
- **Professor Wisdom** (education): Friendly owl character with glasses, appears in classroom scenes
- **Byte the Robot** (ai_education): Blue robot kid with glowing green eyes, antenna, tablet in hand
- **Handy the Helper** (crafts_skills): Toolbelt character with yellow hard hat, safety goggles, colorful tools
- Host-specific catchphrases: BYTE_PHRASES, HANDY_PHRASES in respective generator modules
- Visual themes per host: CATEGORY_VISUAL_THEMES dict defines colors, props, setting for each content type

### Multi-Channel Management
- Channel selection via `channel_manager.load_channel_config()` from channels.json
- OAuth credentials per channel: `data/youtube_token.json`, `data/youtube_token_hi.json`, etc.
- Quota tracking: 1600 units per upload, 10,000 daily limit per channel
- Use `can_upload(channel_key)` before uploading to check quota
- Playlist management: `get_or_create_playlist()` + `add_video_to_playlist()`

### Multi-Language Support
- Translation via `regional_content.translate_script(script, target_lang)` using deep-translator
- Regional TTS voices via `get_regional_tts_voice(language)` from channels.json
- Batch regional generation: `create_regional_version(script, channel_key)` in batch_pipeline
- Keep visual_description and image prompts in English (SDXL Turbo works best with English)
- Translate narration, title, description, tags to target language
- Regional channels: stories_hi (Hindi), stories_es (Spanish), stories_fr (French), stories_pt (Portuguese), stories_ar (Arabic)

### TTS + Audio Processing
- Edge TTS generates MP3 → FFmpeg converts to WAV (no libmp3lame available)
- Cartoon pitch effects: `ffmpeg asetrate` + `aresample` + `atempo` filters
- Audio output: WAV format, 24kHz sample rate, PCM s16le codec
- Regional voices: `en-US-AriaNeural`, `hi-IN-SwaraNeural`, `es-ES-ElviraNeural`, etc.

### AI Image Generation
- SDXL Turbo: Generate at 512x512 (native), resize to 1024x576 with PIL.LANCZOS
- Generate 30 variations per scene with CAMERA_VARIATIONS prompts
- Sequential generation on GPU (~0.5s/image), global pipeline cache for reuse
- Seeded generation: `base_seed + scene_num * 1000 + img_idx` for reproducibility
- Style variants via `get_style_prefix(content_type)`:
  - `content_type="story"`: STYLE_PREFIX (Pixar/Disney cartoon style)
  - `content_type="education"`: EDUCATION_STYLE_PREFIX (Professor Wisdom owl in classroom)
  - `content_type="ai_education"`: AI_EDUCATION_STYLE_PREFIX (Byte the Robot in futuristic lab)
  - `content_type="crafts_skills"`: CRAFTS_SKILLS_STYLE_PREFIX (Handy the Helper in workshop)

### Video Assembly
- Use PIL for text rendering (FFmpeg drawtext unavailable due to missing libfreetype)
- Multi-image crossfade: `xfade` filter between consecutive images
- Ken Burns fallback: `zoompan` filter for single-image scenes
- Subtitle integration: `subtitles=output/subtitles/story_X.srt` filter
- Output: H.264 with NVENC, yuv420p pixel format, CRF 23

### Subtitle Generation
- Multi-language SRT files via `generate_subtitles_for_story(script, audio_path, language)`
- Word-level timing via audio duration / word count estimation
- Subtitle files stored at `output/subtitles/story_{story_id}_{lang}.srt`

### SEO Optimization
- Title optimization: `optimize_title(title, moral, collection, language, content_type)`
- Tag optimization: `optimize_tags(script, language)` → list of SEO keywords
- Description optimization: `optimize_description(script, language)` with CTAs
- Language-aware keywords for multi-language channels

### YouTube Upload
- Made for Kids = set `madeForKids: true`, no comments, limited features
- Title max 70 chars (better CTR), description with SEO keywords
- Thumbnails: 1280x720 JPEG, under 2MB
- Tags from story script + collection themes + SEO optimization
- Upload via `channel_manager.upload_to_channel(channel_key, video_path, metadata)`
<!-- END AUTO-MANAGED -->

<!-- AUTO-MANAGED: patterns -->
## Patterns

### Batch Pipeline Flow (100+ videos/day)
1. **Phase 1 (01:00 UTC)**: Generate all scripts via LLM
   - Stories (36/day): 6 channels × 6 videos each (EN, HI, ES, FR, PT, AR)
   - Education (48/day): 7 channels × 6-7 videos (Oxford, Cambridge, regional)
   - AI Education (20/day): 4 channels × 5 videos (EN, HI, ES, FR)
   - Crafts/Skills (12/day): 3 channels × 4 videos (carpentry, plumbing, electrical)
   - Total: 116 videos/day
2. **Phase 2 (02:00 UTC)**: Generate all audio via TTS (multi-language, async batch)
3. **Phase 3 (03:00 UTC)**: Generate all images via GPU (longest phase, ~5h for 100+ videos)
4. **Phase 4 (08:00 UTC)**: Assemble all videos via FFmpeg (crossfade animations + subtitles)
5. **Phase 5 (10:00-22:00 UTC)**: Upload in batches across 21 channels (quota-aware)
6. **Every 6h**: Analytics refresh via YouTube Analytics API v2 + keyword optimization
7. **Every 1h**: Health check (GPU, disk, upload progress, pipeline state)

### Story Generation Flow
1. `pick_smart_story_params()` selects collection + moral based on past performance
2. LLM generates structured JSON with scenes, narration, visual_description
3. `inject_engagement_hooks()` adds intro/midpoint/outro scenes with CTAs
4. `generate_engaging_title()` optimizes title for CTR from templates
5. `optimize_title()` + `optimize_tags()` + `optimize_description()` for SEO

### Education Content Flow
1. `pick_next_lesson()` selects next unpublished lesson from syllabus
2. `generate_lesson_script()` creates educational script with Professor Wisdom character
3. Inject education-specific engagement hooks (EDUCATION_INTRO_HOOKS, EDUCATION_OUTRO_HOOKS)
4. Generate images with `content_type="education"` → EDUCATION_STYLE_PREFIX
5. Upload to curriculum-specific playlist via `get_or_create_playlist()`
6. `mark_lesson_published()` tracks progress in education_lessons table

### AI Education Content Flow
1. `pick_next_ai_lesson()` selects next lesson from ai_kids_curriculum.json
2. `generate_ai_lesson_script()` creates lesson with Byte the Robot character
3. Inject AI-specific engagement hooks with "Try It!" challenges
4. Generate images with `content_type="ai_education"` → AI_EDUCATION_STYLE_PREFIX
5. Upload to AI education channel with category-specific playlists
6. Track in ai_education_lessons table

### Crafts & Skills Content Flow
1. `pick_next_crafts_lesson()` selects next lesson from crafts_skills_curriculum.json
2. `generate_crafts_lesson_script()` creates lesson with Handy the Helper character
3. Inject safety reminders and interactive quiz scenes
4. Generate images with `content_type="crafts_skills"` → CRAFTS_SKILLS_STYLE_PREFIX
5. Upload to crafts/skills channel with category-specific playlists
6. Track in crafts_skills_lessons table with safety_note field

### Multi-Language Content Flow
1. Generate English script first
2. `translate_script(script, target_lang)` translates narration, title, description
3. Keep visual_description in English (SDXL Turbo works best with English)
4. `get_regional_tts_voice(language)` selects appropriate Edge TTS voice
5. Generate audio with regional voice
6. Upload to language-specific channel (stories_hi, stories_es, etc.)

### Multi-Image Animated Scenes
1. Generate 30 images per scene with varied prompts (close-ups, wide shots, angles)
2. Store all images in `scene["all_images"]` array
3. Video assembler creates crossfade slideshow if len(all_images) > 1
4. Fallback to Ken Burns effect if only 1 image available

### Multi-Channel Upload Strategy
1. Check quota availability via `can_upload(channel_key)` before upload
2. Distribute uploads across channels to maximize daily output (10,000 units/day per channel)
3. Track upload quota in channels.db (1600 units per upload)
4. Create/update playlists via `get_or_create_playlist()` based on content category
5. Add video to playlist via `add_video_to_playlist()`

### GPU Memory Management
1. Global `_pipeline` variable caches SDXL model across function calls
2. `unload_model()` explicitly frees GPU memory after batch generation
3. `enable_attention_slicing()` reduces VRAM usage during inference
4. Content-type style switching via `get_style_prefix(content_type)`

### Database Tracking
1. **stories.db**: Unified content database
   - `stories` table: Story scripts as JSON, status tracking (`generated` → `published`)
   - `education_lessons` table: Education lesson metadata, curriculum progress tracking
   - `ai_education_lessons` table: AI lesson metadata, category and topic tracking
   - `crafts_skills_lessons` table: Crafts/skills lesson metadata, safety notes
   - `video_analytics` table: YouTube Analytics API v2 per-video metrics (views, CTR, watch time)
   - `search_terms` table: YouTube search queries driving traffic to videos
   - `channel_analytics` table: Per-channel daily aggregate metrics
2. **channels.db**: Multi-channel upload quota tracking
   - `channel_uploads` table: Tracks uploads per channel per day, enforces 10,000 unit daily limit
   - `playlist_cache` table: Maps playlist keys to YouTube playlist IDs per channel
<!-- END AUTO-MANAGED -->
