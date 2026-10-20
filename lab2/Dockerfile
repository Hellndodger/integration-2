ARG PYTHON_VERSION=3.11

FROM python:${PYTHON_VERSION}-slim AS trainer
WORKDIR /build

RUN apt-get update \
    && apt-get install -y --no-install-recommends libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-train.txt requirements-api.txt ./
RUN pip install --no-cache-dir -r requirements-train.txt

COPY src/ ./src/
ARG EPOCHS=4
ARG MAX_PER_CLASS=50
ARG MODEL_VERSION=1.0.0
RUN python -m src.train \
    --data-root /tmp/speechcommands \
    --output-dir /build/artifacts \
    --epochs "${EPOCHS}" \
    --limit-per-class "${MAX_PER_CLASS}" \
    --seed 42 \
    --model-version "${MODEL_VERSION}"

FROM python:${PYTHON_VERSION}-slim AS runtime
WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-api.txt ./
RUN pip install --no-cache-dir -r requirements-api.txt \
    && useradd --create-home --uid 10001 appuser

COPY src/ ./src/
COPY --from=trainer /build/artifacts ./artifacts

ARG MODEL_VERSION=1.0.0
ARG VCS_REF=unknown
ARG BUILD_DATE=unknown
LABEL org.opencontainers.image.title="Speech Commands API" \
      org.opencontainers.image.version="${MODEL_VERSION}" \
      org.opencontainers.image.revision="${VCS_REF}" \
      org.opencontainers.image.created="${BUILD_DATE}"

ENV MODEL_PATH=/app/artifacts/model.pt \
    METRICS_PATH=/app/artifacts/metrics.json \
    MODEL_VERSION=${MODEL_VERSION} \
    PORT=8000 \
    TORCH_NUM_THREADS=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

USER appuser
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"
CMD ["sh", "-c", "uvicorn src.api:app --host 0.0.0.0 --port ${PORT}"]
