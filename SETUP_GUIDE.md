# Little Wisdom Tales - Complete Setup Guide

## What This Does
This automated pipeline creates and publishes kid-friendly moral story videos to your YouTube channel 24/7. Stories come from world folklore (Aesop's Fables, Panchatantra, Arabian Nights, African tales, etc.) with cartoon-style voiceover, AI-generated illustrations, and animated video effects.

## Architecture Overview
```
[Scheduler] -> [Story Generator (Ollama LLM)] -> [Voice Generator (Edge TTS)]
                                                          |
[YouTube Upload] <- [Video Assembler (FFmpeg)] <- [AI Image Generator (SDXL Turbo)]
       |                                                  |
       v                                           [Pixabay Fallback]
[Analytics Tracker] -> feeds back into story selection
       |
[Subtitle Generator] -> Multi-language SRT captions
```

## System Requirements

### Hardware (GPU Server)
- **GPU**: NVIDIA GPU with 8+ GB VRAM (tested on RTX A5000 24GB)
- **RAM**: 16GB+ recommended
- **Storage**: 50GB+ for models, output, and cache
- **OS**: Ubuntu 22.04+

### Software
- Python 3.12+
- FFmpeg 6.1+ (with NVENC support for GPU encoding)
- NVIDIA Driver 530+ with CUDA support
- Ollama (for local LLM inference)

## Prerequisites (API Keys Needed)

### 1. Google Cloud Console (YouTube API) - FREE
- Go to: https://console.cloud.google.com/
- Create a new project
- Enable "YouTube Data API v3"
- Create OAuth 2.0 Client ID (Desktop App type)
- Note the Client ID and Client Secret

### 2. Pixabay API Key (Fallback Images) - FREE
- Go to: https://pixabay.com/api/docs/
- Create account and get API key
- Used as fallback when AI image generation fails

### 3. Pexels API Key (Backup Fallback) - FREE
- Go to: https://www.pexels.com/api/
- Create account and get API key

### 4. (Optional) Google Gemini API - FREE tier
- Go to: https://aistudio.google.com/
- Backup LLM if Ollama is unavailable

---

## Installation

### Step 1: Clone the repository
```bash
git clone https://github.com/fahimmaan04-del/little-wisdom-tales.git
cd little-wisdom-tales
```

### Step 2: Create Python virtual environment
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Step 3: Install PyTorch with CUDA support
```bash
pip install torch==2.5.1+cu124 --extra-index-url https://download.pytorch.org/whl/cu124
pip install diffusers transformers accelerate safetensors
```

### Step 4: Install Ollama
```bash
curl -fsSL https://ollama.ai/install.sh | sh
ollama pull llama3.2:3b
```

### Step 5: Configure environment
```bash
cp .env.example .env
nano .env
```
Fill in your API keys:
```
YOUTUBE_CLIENT_ID=your_client_id
YOUTUBE_CLIENT_SECRET=your_client_secret
YOUTUBE_CHANNEL_ID=your_channel_id
PIXABAY_API_KEY=your_pixabay_key
PEXELS_API_KEY=your_pexels_key
```

### Step 6: Pre-download SDXL Turbo model
```bash
source venv/bin/activate
python -c "
from diffusers import AutoPipelineForText2Image
import torch
pipe = AutoPipelineForText2Image.from_pretrained(
    'stabilityai/sdxl-turbo', torch_dtype=torch.float16, variant='fp16'
)
pipe.to('cuda')
print('Model loaded successfully!')
del pipe; torch.cuda.empty_cache()
"
```

### Step 7: Authenticate with YouTube (ONE TIME)
```bash
source venv/bin/activate
python scripts/youtube_manager.py auth
```
This prints a URL. Open it in your browser, sign in, authorize access, and paste the code back. After this, the system has permanent access to your YouTube channel.

### Step 8: Test the pipeline
```bash
# Test without uploading to YouTube
python scripts/orchestrator.py test

# Create and upload a single story
python scripts/orchestrator.py single
```

### Step 9: Start the 24/7 service
```bash
# Install and start the systemd service
sudo cp systemd/kids-story-pipeline.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable kids-story-pipeline
sudo systemctl start kids-story-pipeline
```

---

## Pipeline Components

### Image Generation (AI-Powered)
- **Primary**: SDXL Turbo on local GPU - generates 30 varied images per scene
  - Different angles: close-up, wide, bird's eye, dramatic, etc.
  - Generates at 512x512 native, resized to 1024x576 for video
  - ~0.5s per image on RTX A5000
- **Fallback**: Pixabay stock illustrations (if GPU unavailable)
- **Last resort**: PIL-generated text overlay cards

### Video Assembly
- Multi-image crossfade slideshow (30 images per scene)
- Ken Burns zoom/pan effects (single-image fallback)
- PIL-rendered text overlays (FFmpeg lacks libfreetype)
- H.264 encoding, yuv420p pixel format
- Intro card, scene clips, moral card, background music

### Subtitle Generation (Multi-Language)
- Auto-generates SRT subtitles from story narration
- Translates to 5 languages: Spanish, Hindi, French, Portuguese, Arabic
- Uploads captions to YouTube via Data API v3
- Uses deep-translator (Google Translate, free, no API key)

### Content Strategy (Analytics-Driven)
- SQLite database tracks story performance
- `pick_smart_story_params()` selects topics based on past analytics
- Engagement hooks: intro, midpoint, and outro CTAs
- SEO-optimized titles and descriptions

---

## Scheduling

### Default Schedule (3 stories/day)
| Time (UTC) | Action |
|------------|--------|
| 06:00 | Creates & publishes 1 story video + 1 short |
| 14:00 | Creates & publishes 1 story video + 1 short |
| 22:00 | Creates & publishes 1 story video + 1 short |
| Every 6h | Updates analytics, optimizes future content |

Configure in `.env`:
```
VIDEOS_PER_DAY=3      # 3, 6, or 8 stories/day
PUBLISH_AS_SHORTS=true
PUBLISH_AS_VIDEO=true
```

**Note**: YouTube Data API allows ~6 uploads/day (10,000 quota, 1,600 per upload). With video + shorts for each story, 3 stories/day = 6 uploads = quota limit.

---

## Key Commands

```bash
# Activate virtual environment
source venv/bin/activate

# Test single story (no upload)
python scripts/orchestrator.py test

# Create and upload single story
python scripts/orchestrator.py single

# Create multiple stories
python scripts/orchestrator.py single --count 3

# Run daily batch
python scripts/orchestrator.py batch

# Update analytics
python scripts/orchestrator.py analytics

# Test AI image generation
python scripts/ai_image_generator.py

# Test subtitle generation
python scripts/subtitle_generator.py

# Service management
sudo systemctl start kids-story-pipeline
sudo systemctl stop kids-story-pipeline
sudo systemctl status kids-story-pipeline
journalctl -u kids-story-pipeline -f
```

---

## Monitoring

### Check service status
```bash
sudo systemctl status kids-story-pipeline
```

### View logs
```bash
# Pipeline activity
tail -50 data/logs/orchestrator.log

# Scheduler activity
tail -50 data/logs/scheduler.log

# Service logs
tail -50 data/logs/service.log
```

### Check GPU usage
```bash
nvidia-smi
```

### Check disk space
```bash
df -h /mnt
du -sh output/*
```

---

## Directory Structure
```
scripts/              # Python pipeline modules
  orchestrator.py     # Main pipeline coordinator
  story_generator.py  # LLM story script generation
  tts_generator.py    # Edge TTS voiceover
  ai_image_generator.py  # SDXL Turbo image generation
  image_fetcher.py    # Pixabay/Pexels fallback
  video_assembler.py  # FFmpeg video assembly
  youtube_manager.py  # YouTube API upload/analytics
  engagement_hooks.py # Intro/midpoint/outro hooks
  subtitle_generator.py # Multi-language subtitles
  scheduler.py        # 24/7 cron-like scheduler
config/               # voices.json, story_themes.json
output/               # Generated content
  audio/              # TTS voiceover files (WAV)
  images/             # AI-generated scene images (JPEG)
  videos/             # Final videos (MP4)
  thumbnails/         # Video thumbnails
  subtitles/          # SRT subtitle files
data/                 # SQLite DB, logs
  stories.db          # Story + analytics tracking
  logs/               # Pipeline execution logs
assets/               # Static assets (fonts, music, intros)
systemd/              # Service definitions
```

---

## Troubleshooting

### "NVIDIA driver/library version mismatch"
Reboot the system. This happens when the kernel module and userspace libraries are out of sync after a driver update.
```bash
sudo reboot
```

### "CUDA out of memory"
The SDXL Turbo model uses ~6.5GB VRAM. If other processes are using GPU memory:
```bash
nvidia-smi  # Check GPU memory usage
# Kill any stale Python processes using GPU
```

### "YouTube upload failed"
Check `data/logs/orchestrator.log`. Common causes:
- Token expired: Run `python scripts/youtube_manager.py auth`
- Quota exceeded: Wait until midnight Pacific time (quota resets daily)
- Network issues: Check internet connectivity

### "Ollama is slow"
Set `LLM_PROVIDER=gemini` in .env to use Google's free API instead.

### "No images found"
AI image generation is primary. If SDXL fails, check:
1. `nvidia-smi` - GPU available?
2. `python -c "import torch; print(torch.cuda.is_available())"` - CUDA working?
3. Fallback: Pixabay (check PIXABAY_API_KEY in .env)

### Disk space issues
The scheduler auto-cleans output directories older than 3 days when free space drops below 2GB.

---

## Important Constraints
- FFmpeg has NO libmp3lame (use WAV/AAC for audio)
- FFmpeg has NO libfreetype (use PIL for text overlays)
- YouTube Made for Kids = no comments, limited features
- YouTube Data API quota: 10,000 units/day (~6 uploads)
- SDXL Turbo generates at 512x512 native resolution

---

## Costs Breakdown

| Service | Cost | Notes |
|---------|------|-------|
| GPU Server | Varies | RTX A5000 or similar |
| Ollama (local AI) | FREE | Runs locally |
| Edge TTS (voices) | FREE | Microsoft's free TTS |
| SDXL Turbo (images) | FREE | Runs locally on GPU |
| Pixabay API | FREE | 5,000 req/day fallback |
| YouTube Data API | FREE | 10,000 quota/day |
| FFmpeg | FREE | Open source |
| deep-translator | FREE | Google Translate wrapper |

---

## YouTube Channel Info
- **Name**: Little Wisdom Tales
- **Handle**: @WisdomTalesKids
- **Channel ID**: UCYu3K3wJQ1t12qzMHXuXYnQ
- **Content**: Kid-friendly moral stories (ages 4-10)
- **Compliance**: Made for Kids enabled
