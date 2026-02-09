# Little Wisdom Tales - Project Instructions

<!-- AUTO-MANAGED: project-description -->
## Project Overview
Fully automated YouTube channel pipeline that creates and publishes kid-friendly moral story videos 24/7 without human intervention.

**Pipeline**: Story generation (LLM) → Voiceover (Edge TTS) → AI images (SDXL Turbo) → Video assembly (FFmpeg) → YouTube upload (Data API v3)

**Target**: Kids aged 4-10, Made for Kids compliance, daily uploads
<!-- END AUTO-MANAGED -->

<!-- AUTO-MANAGED: architecture -->
## Architecture

### Pipeline Components
- **LLM**: Ollama llama3.2:3b (localhost:11434) for story script generation
- **TTS**: Edge TTS with cartoon pitch shifting via FFmpeg
- **Images**: SDXL Turbo on local GPU (RTX A5000, 24GB VRAM) → 30 images/scene with crossfades
- **Video**: FFmpeg 6.1 with NVENC, multi-image crossfade animation, PIL-rendered text overlays
- **Upload**: YouTube Data API v3 (Made for Kids compliance)
- **Scheduler**: systemd service running Python scheduler (scripts/scheduler.py)
- **Orchestration**: n8n workflow automation (optional, docker-compose)

### Key Scripts
- `orchestrator.py`: Master pipeline coordinator (story → TTS → images → video → upload)
- `story_generator.py`: LLM-based story script generation with SQLite tracking
- `tts_generator.py`: Edge TTS with async generation + FFmpeg cartoon pitch effects
- `ai_image_generator.py`: SDXL Turbo GPU image generation (512x512 native → resized)
- `video_assembler.py`: FFmpeg video assembly with Ken Burns effects + crossfades
- `engagement_hooks.py`: Title/description optimization, subscribe CTAs
- `youtube_manager.py`: YouTube Data API v3 upload, analytics tracking
- `image_fetcher.py`: Pixabay/Pexels fallback for stock images
- `scheduler.py`: Cron-based pipeline scheduler

### Directory Structure
```
scripts/          # Python pipeline modules
config/           # voices.json, story_themes.json
output/           # Generated content
  audio/          # TTS voiceover files (WAV)
  images/         # AI-generated scene images (JPEG)
  videos/         # Final videos (MP4)
  thumbnails/     # Video thumbnails
data/             # SQLite DB, logs
  stories.db      # Story + analytics tracking
  logs/           # Pipeline execution logs
assets/           # Static assets (fonts, music, intros)
n8n-workflows/    # Workflow automation configs
systemd/          # Service definitions
```
<!-- END AUTO-MANAGED -->

<!-- AUTO-MANAGED: build-commands -->
## Key Commands

### Pipeline Operations
```bash
# Test single story (no upload)
python scripts/orchestrator.py test

# Create and upload single story
python scripts/orchestrator.py single

# Run daily batch
python scripts/orchestrator.py batch
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
- Configuration files in `config/` as JSON (voices.json, story_themes.json)
- Environment variables in `.env` (copy from `.env.example`)
- Database: SQLite at `data/stories.db` with `stories` and `analytics` tables

### Python Patterns
- Use `from dotenv import load_dotenv` + `load_dotenv()` at module top
- Async functions for I/O-bound operations (TTS generation uses `async def` + Edge TTS)
- Path objects from `pathlib.Path` for all file operations
- Environment variable access: `Path(os.getenv("OUTPUT_DIR", "./output"))`

### TTS + Audio Processing
- Edge TTS generates MP3 → FFmpeg converts to WAV (no libmp3lame available)
- Cartoon pitch effects: `ffmpeg asetrate` + `aresample` + `atempo` filters
- Audio output: WAV format, 24kHz sample rate, PCM s16le codec

### AI Image Generation
- SDXL Turbo: Generate at 512x512 (native), resize to 1024x576 with PIL.LANCZOS
- Generate 30 variations per scene with CAMERA_VARIATIONS prompts
- Sequential generation on GPU (~0.5s/image), global pipeline cache for reuse
- Seeded generation: `base_seed + scene_num * 1000 + img_idx` for reproducibility

### Video Assembly
- Use PIL for text rendering (FFmpeg drawtext unavailable due to missing libfreetype)
- Multi-image crossfade: `xfade` filter between consecutive images
- Ken Burns fallback: `zoompan` filter for single-image scenes
- Output: H.264 with NVENC, yuv420p pixel format, CRF 23

### YouTube Upload
- Made for Kids = set `madeForKids: true`, no comments, limited features
- Title max 70 chars (better CTR), description with SEO keywords
- Thumbnails: 1280x720 JPEG, under 2MB
- Tags from story script + collection themes
<!-- END AUTO-MANAGED -->

<!-- AUTO-MANAGED: patterns -->
## Patterns

### Story Generation Flow
1. `pick_smart_story_params()` selects collection + moral based on past performance
2. LLM generates structured JSON with scenes, narration, visual_description
3. `inject_engagement_hooks()` adds intro/midpoint/outro scenes with CTAs
4. `generate_engaging_title()` optimizes title for CTR from templates

### Multi-Image Animated Scenes
1. Generate 30 images per scene with varied prompts (close-ups, wide shots, angles)
2. Store all images in `scene["all_images"]` array
3. Video assembler creates crossfade slideshow if len(all_images) > 1
4. Fallback to Ken Burns effect if only 1 image available

### GPU Memory Management
1. Global `_pipeline` variable caches SDXL model across function calls
2. `unload_model()` explicitly frees GPU memory after batch generation
3. `enable_attention_slicing()` reduces VRAM usage during inference

### Database Tracking
1. Story script stored as JSON in `stories.script` column
2. Status: `generated` → `published` lifecycle
3. Analytics table tracks daily views, likes, watch_time per video_id
4. `pick_smart_story_params()` queries analytics for performance-based selection
<!-- END AUTO-MANAGED -->
