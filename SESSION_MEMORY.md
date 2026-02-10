# Little Wisdom Tales - Session Memory

> **Last Updated**: 2026-02-10
> **Status**: ACTIVE - 21-channel pipeline running via systemd

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
- **Re-upload/re-edit pipeline for upgrading existing videos**
- **21-channel multi-content pipeline (stories, education, AI, crafts)**
- **Batch pipeline with 5-phase generation (scripts→audio→images→video→upload)**
- **Multi-channel management with OAuth, quota tracking, playlists**
- **Regional content with translation + TTS voice selection (6 languages)**
- **YouTube analytics integration (video/channel/search terms)**
- **systemd service running 24/7 in batch mode**
- 10 stories published to YouTube channel (original single channel)

### Configuration
- **Channels**: 21 (6 story + 7 education + 4 AI + 3 crafts + 1 existing)
- **Daily target**: 116 videos/day across all channels
- **Image generation**: SDXL Turbo, 30 images/scene, ~0.35s/img
- **Subtitles**: Auto-translated to 5 languages
- **Keywords**: Auto-refreshed every 6 hours from YouTube search data
- **Pipeline mode**: Batch (phased generation via scheduler)

---

## Pipeline Architecture

```
Phase 1 (01:00 UTC): Ollama LLM → Scripts (stories + education + AI + crafts)
Phase 2 (02:00 UTC): Edge TTS → Audio (multi-language, async batch)
Phase 3 (03:00 UTC): SDXL Turbo GPU → 30 AI Images/scene (~5h)
Phase 4 (08:00 UTC): FFmpeg → Videos (crossfade + subtitles)
Phase 5 (10:00-22:00): YouTube Upload → 21 channels (quota-aware, hourly)
Every 6h: Analytics refresh + keyword optimization
Every 1h: Health check (GPU, disk, upload progress)
```

---

## Key Files

| File | Purpose |
|------|---------|
| `scripts/orchestrator.py` | Story pipeline coordinator |
| `scripts/education_orchestrator.py` | Education video pipeline coordinator |
| `scripts/batch_pipeline.py` | 5-phase batch generation (100+ videos/day) |
| `scripts/scheduler.py` | 24/7 scheduler (batch + incremental modes) |
| `scripts/channel_manager.py` | Multi-channel OAuth, quota, playlists |
| `scripts/story_generator.py` | LLM story script generation |
| `scripts/education_generator.py` | Oxford/Cambridge lesson generator |
| `scripts/ai_education_generator.py` | AI/coding/robotics lessons (Byte the Robot) |
| `scripts/ai_content_generator.py` | AI curriculum lesson builder |
| `scripts/crafts_skills_generator.py` | Crafts/skills lessons (Handy the Helper) |
| `scripts/regional_content.py` | Multi-language translation + TTS voice selection |
| `scripts/youtube_analytics.py` | Video/channel analytics + search terms |
| `scripts/tts_generator.py` | Edge TTS voiceover |
| `scripts/ai_image_generator.py` | SDXL Turbo with content-type styles |
| `scripts/video_assembler.py` | FFmpeg + content-type branding |
| `scripts/keyword_optimizer.py` | Multi-language SEO intelligence |
| `scripts/subtitle_generator.py` | Multi-language SRT subtitles |
| `scripts/reupload_videos.py` | Re-edit pipeline (enhanced thumbnails/audio) |
| `scripts/quick_setup.py` | Interactive channel setup |
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

1. ~~Start 24/7 systemd service~~ → **DONE** (running in batch mode)
2. Re-upload 10 existing videos: `python scripts/reupload_videos.py`
3. Set up OAuth tokens for all 21 channels: `python scripts/quick_setup.py`
4. Apply for YouTube API quota increase (10,000 units/day per channel)
5. Monitor first batch pipeline run for errors

---

## Bugs Fixed This Session

- `.env` line 22: `CHANNEL_NAME=Little Wisdom Tales` was unquoted, causing bash `source` to fail with "Wisdom: command not found". Fixed by quoting the value.
- systemd service file was outdated in `/etc/systemd/system/`. Deployed updated version with batch mode, watchdog, and memory limits.

---

## YouTube Channel

- **Name**: Little Wisdom Tales
- **Handle**: @WisdomTalesKids
- **Channel ID**: UCYu3K3wJQ1t12qzMHXuXYnQ
- **Published Videos**: 10 (stock photos, need AI image upgrade)
- **Target**: 21 channels, 116 videos/day

---

## Important Notes

- YouTube Data API daily quota is 10,000 units per channel; 1,600 per upload
- Apply for quota increase at: https://support.google.com/youtube/contact/yt_api_form
- FFmpeg has NO libmp3lame (use WAV/AAC), NO libfreetype (use PIL for text)
- SDXL Turbo model cached at ~/.cache/huggingface/hub/models--stabilityai--sdxl-turbo/
- torch_dtype parameter triggers deprecation warning but still works correctly
- New channels need OAuth tokens set up via `scripts/quick_setup.py` or `scripts/setup_channels.sh`

---

*Updated: Session (2026-02-10)*
