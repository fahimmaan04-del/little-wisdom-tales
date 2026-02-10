#!/bin/bash
# Kids-Heaven - Fully Autonomous Multi-Channel Pipeline
# Runs 24/7 generating videos across 18 YouTube channels
# Content: Stories (6 langs) + Education (Oxford/Cambridge) + AI Education
#
# Usage:
#   ./start_pipeline.sh              # Default: batch mode via scheduler
#   ./start_pipeline.sh batch        # Batch mode (phased generation)
#   ./start_pipeline.sh incremental  # Incremental (one-at-a-time)
#   ./start_pipeline.sh status       # Show pipeline status

set -euo pipefail

cd /mnt/projects/youtube
source venv/bin/activate

# Load environment
if [ -f .env ]; then
    set -a
    source .env
    set +a
fi

MODE="${1:-batch}"

case "$MODE" in
    batch|incremental)
        echo "=========================================="
        echo "Kids-Heaven - Autonomous Pipeline"
        echo "Mode: $MODE"
        echo "Channels: 21 (6 story + 7 education + 4 AI + 3 crafts + 1 existing)"
        echo "Languages: EN, HI, ES, FR, PT, AR"
        echo "Started: $(date -u)"
        echo "=========================================="
        exec python scripts/scheduler.py "$MODE" 2>&1 | tee -a data/logs/pipeline.log
        ;;
    status)
        echo "=== Pipeline Status ==="
        python scripts/batch_pipeline.py stats
        echo ""
        echo "=== AI Education Progress ==="
        python scripts/ai_education_generator.py progress 2>/dev/null || echo "AI education module not ready"
        echo ""
        echo "=== Recent Logs ==="
        tail -20 data/logs/scheduler.log 2>/dev/null || echo "No scheduler logs yet"
        ;;
    *)
        echo "Usage: $0 [batch|incremental|status]"
        exit 1
        ;;
esac
