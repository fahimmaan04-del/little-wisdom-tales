# Little Wisdom Tales - Session Memory

> **Last Updated**: 2026-02-10
> **Status**: ACTIVE - Pipeline fully operational

---

## Current State

### What's Working
- Story generation via Ollama (llama3.2:3b) at localhost:11434
- Voiceover generation via Edge TTS with pitch shifting
- **AI image generation via SDXL Turbo on local GPU (RTX A5000)**
- Multi-image crossfade video assembly with FFmpeg
- YouTube upload via Data API v3
- Engagement hooks (intro/midpoint/outro with subscribe CTAs)
- Analytics-driven smart content selection
- **Multi-language subtitle generation (EN, ES, HI, FR, PT, AR)**
- **Keyword optimizer - trending YouTube search term intelligence**
- **Re-upload script for upgrading existing videos**
- systemd service for 24/7 operation
- 10 stories published to YouTube channel

### Configuration
- **Schedule**: 10 videos/day (no shorts), max 5 minutes each
- **Image generation**: SDXL Turbo, 30 images/scene, ~0.35s/img
- **Subtitles**: Auto-translated to 5 languages
- **Keywords**: Auto-refreshed every 6 hours from YouTube search data

---

## Pipeline Architecture

```
Ollama LLM -> Story Script (JSON with scenes + visual descriptions)
     |
Edge TTS -> Voiceover Audio (WAV per scene + combined)
     |
SDXL Turbo (GPU) -> 30 AI Images per Scene (varied angles/styles)
     |  (fallback: Pixabay stock images)
FFmpeg -> Video (multi-image crossfade + audio + title cards)
     |
Keyword Optimizer -> Trending tags/titles/descriptions
     |
Subtitle Generator -> SRT files (EN + 5 languages)
     |
YouTube Data API -> Upload (SEO, playlists, Made for Kids)
```

---

## Key Files

| File | Purpose |
|------|---------|
| `scripts/orchestrator.py` | Main pipeline coordinator |
| `scripts/story_generator.py` | LLM story script generation |
| `scripts/tts_generator.py` | Edge TTS voiceover |
| `scripts/ai_image_generator.py` | SDXL Turbo local GPU image gen |
| `scripts/image_fetcher.py` | Pixabay stock image fallback |
| `scripts/video_assembler.py` | FFmpeg video assembly with multi-image support |
| `scripts/youtube_manager.py` | YouTube API upload/analytics |
| `scripts/engagement_hooks.py` | Intro/midpoint/outro hooks |
| `scripts/keyword_optimizer.py` | Trending keyword intelligence |
| `scripts/subtitle_generator.py` | Multi-language SRT subtitles |
| `scripts/reupload_videos.py` | Re-upload existing videos with AI images |
| `scripts/scheduler.py` | 24/7 cron-like scheduler |
| `start_pipeline.sh` | systemd startup script |

---

## Installed Dependencies (in venv)

- torch 2.5.1+cu124 (isolated from system CUDA)
- diffusers 0.36.0, transformers 4.57.6, accelerate, safetensors
- edge-tts, httpx, Pillow, python-dotenv
- google-api-python-client, google-auth-oauthlib
- deep-translator (for subtitle translations)
- schedule (for cron-like scheduling)

---

## Pending Work

1. Start the 24/7 systemd service: `sudo systemctl start kids-story-pipeline`
2. Re-upload 10 existing videos: `python scripts/reupload_videos.py`
3. Monitor YouTube quota usage (10 uploads/day = 16,000 units, may exceed 10,000 limit)
4. Apply for YouTube API quota increase if needed

---

## YouTube Channel

- **Name**: Little Wisdom Tales
- **Handle**: @WisdomTalesKids
- **Channel ID**: UCYu3K3wJQ1t12qzMHXuXYnQ
- **Published Videos**: 10 (stock photos, need AI image upgrade)

---

## Important Notes

- YouTube Data API daily quota is 10,000 units; 10 uploads = 16,000 units
- Apply for quota increase at: https://support.google.com/youtube/contact/yt_api_form
- FFmpeg has NO libmp3lame (use WAV/AAC), NO libfreetype (use PIL for text)
- SDXL Turbo model cached at ~/.cache/huggingface/hub/models--stabilityai--sdxl-turbo/
- torch_dtype parameter triggers deprecation warning but still works correctly

---

*Updated: Session (2026-02-10)*
