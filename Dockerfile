# Lexora AI in one container: the API (port 8000) and the UI (port 8501). Ollama stays on the host machine.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PIP_NO_CACHE_DIR=1 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Tesseract is needed for scanned pages; everything else is installed as Python wheels
RUN apt-get update && apt-get install -y --no-install-recommends tesseract-ocr tesseract-ocr-eng \
    && rm -rf /var/lib/apt/lists/*

# Unprivileged user that owns the data folder
RUN useradd --create-home --uid 1000 lexora \
    && mkdir -p /data /app \
    && chown lexora /data /app

WORKDIR /app
COPY requirements.txt .
# CPU-only PyTorch first (much smaller than the default build), then the rest, all pinned
ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu
RUN pip install torch==2.15.0 --index-url ${TORCH_INDEX} \
    && pip install -r requirements.txt

COPY --chown=lexora . .

# 0.0.0.0 inside the container; docker-compose publishes the ports on 127.0.0.1 only
ENV API_HOST=0.0.0.0 NO_BROWSER=1 LEXORA_STORAGE_DIR=/data

USER lexora

# Bake the configured models into the image (names come from config.py). An offline build still succeeds.
RUN python -c "from config import get_settings; s = get_settings(); \
from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer(s.embedding_model, device='cpu'); \
CrossEncoder(s.reranker_model, device='cpu')" \
    || echo "model download skipped; they will be fetched on first use"

VOLUME /data
EXPOSE 8000 8501
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD python -c "import requests,sys; sys.exit(0 if requests.get('http://127.0.0.1:8000/api/health', timeout=3).ok else 1)"
CMD ["python", "run.py"]