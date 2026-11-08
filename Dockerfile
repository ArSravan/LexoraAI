# Lexora AI in one container: the API (port 8000) and the UI (port 8501). Ollama stays on the host machine.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PIP_NO_CACHE_DIR=1

# Tesseract is needed for scanned pages; everything else is installed as Python wheels
RUN apt-get update && apt-get install -y --no-install-recommends tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /srv/LexoraAI
COPY requirements.txt .
# CPU-only PyTorch first (much smaller than the default build), then the rest, all pinned
RUN pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu \
    && pip install -r requirements.txt

COPY . .

# 0.0.0.0 inside the container; docker-compose publishes the ports on 127.0.0.1 only
ENV API_HOST=0.0.0.0 NO_BROWSER=1 LEXORA_STORAGE_DIR=/data
VOLUME /data
EXPOSE 8000 8501
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD python -c "import requests,sys; sys.exit(0 if requests.get('http://127.0.0.1:8000/api/health', timeout=3).ok else 1)"
CMD ["python", "run.py"]
