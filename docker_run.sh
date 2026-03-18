#!/bin/bash
set -euo pipefail

IMAGE="natalie23gill/geneset-translator"
TAG="${1:-latest}"

# Mount .env file if it exists
ENV_MOUNT=""
if [ -f ".env" ]; then
    ENV_MOUNT="-v $(pwd)/.env:/app/.env"
fi

# Run with data volume mounted for caching
docker run -it \
    -p 8501:8501 \
    -v "$(pwd)/data:/app/data" \
    $ENV_MOUNT \
    "${IMAGE}:${TAG}" \
    streamlit run app.py --server.address 0.0.0.0
