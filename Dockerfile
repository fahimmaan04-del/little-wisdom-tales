FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    fonts-dejavu-core \
    fonts-noto \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY scripts/ scripts/
COPY config/ config/
COPY templates/ templates/

# Create output directories
RUN mkdir -p output/videos output/audio output/images output/thumbnails data/logs assets/music

# Default command - run single story (overridden by docker-compose)
CMD ["python", "scripts/orchestrator.py", "batch"]
