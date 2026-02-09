# Kids Story Channel - Complete Setup Guide

## What This Does
This automated pipeline creates and publishes 3 kid-friendly moral story videos per day to your YouTube channel. Stories come from world folklore (Aesop's Fables, Panchatantra, Arabian Nights, African tales, etc.) with cartoon-like voiceover and animated illustrations.

## Architecture Overview
```
[Scheduler] → [Story Generator (Ollama AI)] → [Voice Generator (Edge TTS)]
                                                        ↓
[YouTube Upload] ← [Video Assembler (FFmpeg)] ← [Image Fetcher (Pixabay)]
       ↓
[Analytics Tracker] → feeds back into story selection
```

## Prerequisites (5 Free Accounts Needed)

### 1. Oracle Cloud Always Free Tier (Hosting)
**Cost: FREE forever**
- Go to: https://cloud.oracle.com/
- Sign up for "Always Free" tier
- Create an ARM Compute Instance:
  - Shape: `VM.Standard.A1.Flex` (ARM)
  - OCPUs: 4, Memory: 24GB (max free)
  - OS: Ubuntu 22.04
  - This is your 24/7 server — it runs forever for free

### 2. Google Cloud Console (YouTube API)
**Cost: FREE**
- Go to: https://console.cloud.google.com/
- Sign in with: fahim.maan04@gmail.com
- Steps:
  1. Click "Select a project" → "New Project" → Name it "Kids Story Channel"
  2. In left sidebar: "APIs & Services" → "Enable APIs"
  3. Search for "YouTube Data API v3" → Enable it
  4. Go to "Credentials" → "Create Credentials" → "OAuth 2.0 Client ID"
  5. Application type: "Desktop App"
  6. Download the credentials — you'll need the Client ID and Client Secret

### 3. Pixabay API Key
**Cost: FREE (5,000 requests/day)**
- Go to: https://pixabay.com/api/docs/
- Create account → Get your API key
- This provides cartoon/illustration images for stories

### 4. Pexels API Key (Backup)
**Cost: FREE**
- Go to: https://www.pexels.com/api/
- Create account → Get your API key
- Backup image source if Pixabay doesn't have good matches

### 5. (Optional) Google Gemini API
**Cost: FREE tier (15 requests/minute)**
- Go to: https://aistudio.google.com/
- Get API key
- Backup for story generation if Ollama is slow

---

## Deployment Steps

### Step 1: SSH into your Oracle Cloud server
```bash
ssh ubuntu@YOUR_SERVER_IP
```

### Step 2: Clone/copy the project
```bash
cd ~
# If using git:
git clone YOUR_REPO_URL youtube-channel
cd youtube-channel

# Or copy files via scp from your local machine
```

### Step 3: Run the deployment script
```bash
chmod +x deploy.sh
./deploy.sh
```
This installs Docker, Python, FFmpeg, Ollama, and all dependencies.

### Step 4: Configure your API keys
```bash
nano .env
```
Fill in:
```
YOUTUBE_CLIENT_ID=paste_from_google_cloud
YOUTUBE_CLIENT_SECRET=paste_from_google_cloud
PIXABAY_API_KEY=paste_from_pixabay
PEXELS_API_KEY=paste_from_pexels
```
Save: Ctrl+O, Enter, Ctrl+X

### Step 5: Authenticate with YouTube (ONE TIME)
```bash
source venv/bin/activate
python scripts/youtube_manager.py auth
```
This will print a URL. Open it in your browser, sign in with fahim.maan04@gmail.com, authorize access, and paste the code back.

After this, the system has permanent access to your YouTube channel.

### Step 6: Start the pipeline
```bash
sudo docker-compose up -d
```

**That's it! The system is now running 24/7.**

---

## What Happens Automatically

| Time (UTC) | Action |
|------------|--------|
| 06:00 | Creates & publishes 1 story video + 1 short |
| 14:00 | Creates & publishes 1 story video + 1 short |
| 22:00 | Creates & publishes 1 story video + 1 short |
| 04:00 | Updates analytics, optimizes future story selection |

**Total: 3 full videos + 3 shorts per day, every day.**

---

## Monitoring

### Check if it's running:
```bash
sudo docker-compose ps
```

### View logs:
```bash
# Recent pipeline activity
tail -50 data/logs/orchestrator.log

# Scheduler activity
tail -50 data/logs/scheduler.log

# Docker container logs
sudo docker-compose logs --tail=50 scheduler
```

### Check n8n dashboard:
Open `http://YOUR_SERVER_IP:5678` in browser
- Username: admin
- Password: changeme (change this in .env!)

---

## Costs Breakdown

| Service | Cost | Notes |
|---------|------|-------|
| Oracle Cloud ARM VM | FREE | 4 cores, 24GB RAM, forever free |
| n8n (self-hosted) | FREE | Open source |
| Ollama (local AI) | FREE | Runs on the VM |
| Edge TTS (voices) | FREE | Microsoft's free TTS |
| Pixabay API | FREE | 5,000 req/day |
| Pexels API | FREE | Backup images |
| YouTube Data API | FREE | 10,000 quota/day |
| FFmpeg | FREE | Open source |
| **TOTAL** | **$0/month** | |

---

## Troubleshooting

### "Ollama is slow"
The free ARM VM is not the fastest for AI. Stories take 2-5 minutes to generate. This is normal. Alternatively, set `LLM_PROVIDER=gemini` in .env to use Google's free API instead.

### "YouTube upload failed"
Check `data/logs/orchestrator.log` for errors. Most common: token expired. Run `python scripts/youtube_manager.py auth` again.

### "No images found"
Make sure PIXABAY_API_KEY is set correctly in .env. The system falls back to text cards if no images are found.

### "Server rebooted"
The systemd service auto-starts Docker on boot. Check with: `sudo systemctl status kids-story-channel`

---

## Revenue Expectations

YouTube monetization requirements:
- 1,000 subscribers
- 4,000 watch hours (or 10M Shorts views in 90 days)

Kids content gets lower CPM ($1-3/1000 views vs $5-15 for adults), but:
- Kids content gets MASSIVE views (millions)
- Shorts can go viral quickly
- Consistency (3 videos/day) is key for the algorithm

Realistic timeline:
- Month 1-2: Building library, minimal views
- Month 3-6: Algorithm starts recommending, 10K-100K views/month
- Month 6-12: If content catches on, 100K-1M+ views/month
- Monetization eligibility: Typically 3-6 months with consistent uploads
