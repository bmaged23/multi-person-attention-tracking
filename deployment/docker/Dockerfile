# syntax=docker/dockerfile:1
FROM python:3.12.15-slim-bookworm
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MPLCONFIGDIR=/tmp/attention-matplotlib
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libgl1 libegl1 libgles2 libgl1-mesa-dri libglib2.0-0 libportaudio2 libgomp1 ca-certificates \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt ./
RUN python -m pip install --no-cache-dir --no-deps -r requirements.txt && python -m pip check
COPY scripts/download_model.py scripts/download_person_model.py ./scripts/
RUN python scripts/download_model.py && python scripts/download_person_model.py
COPY attention_tracking ./attention_tracking
COPY web ./web
COPY scripts ./scripts
COPY tests ./tests
COPY evaluation ./evaluation
COPY docs ./docs
COPY config.py video_attention.py live_camera_attention.py pyproject.toml LICENSE .env.example ./
RUN python -m unittest discover -s tests -q && useradd --uid 10001 --create-home attention \
    && mkdir -p data && chown attention:attention data
ENV ATTENTION_PERSON_BACKEND=onnx ATTENTION_PERSON_DEVICE=cuda ATTENTION_DEVICE=cpu
ARG VCS_REF=development
LABEL org.opencontainers.image.source="https://github.com/bmaged23/multi-person-attention-tracking" \
    org.opencontainers.image.revision=$VCS_REF org.opencontainers.image.licenses="AGPL-3.0-only"
USER attention
EXPOSE 8770
HEALTHCHECK --interval=15s --timeout=4s --start-period=120s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8770/health',timeout=3)"]
CMD ["python", "-m", "uvicorn", "attention_tracking.api:app", "--host", "0.0.0.0", "--port", "8770", "--workers", "1", "--no-access-log"]
