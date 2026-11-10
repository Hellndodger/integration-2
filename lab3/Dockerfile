ARG PYTHON_VERSION=3.11

FROM python:${PYTHON_VERSION}-slim AS trainer
WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY lab2/requirements-api.txt lab2/requirements-train.txt ./
RUN pip install --no-cache-dir -r requirements-train.txt

COPY lab2/src/ ./src/
ENTRYPOINT ["python", "-m", "src.train"]
CMD ["--data-root", "/data", "--output-dir", "/artifacts", "--epochs", "4", "--limit-per-class", "50", "--seed", "42"]

FROM python:${PYTHON_VERSION}-slim AS runtime
WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY lab2/requirements-api.txt ./
RUN pip install --no-cache-dir -r requirements-api.txt \
    && useradd --create-home --uid 10001 appuser

COPY lab2/src/ ./src/
COPY lab2/scripts/benchmark.py ./scripts/benchmark.py
COPY lab3/artifacts/model.pt lab3/artifacts/labels.json lab3/artifacts/metrics.json ./artifacts/

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
    GIT_COMMIT=${VCS_REF} \
    PORT=8000 \
    TORCH_NUM_THREADS=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

USER appuser
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"
CMD ["sh", "-c", "uvicorn src.api:app --host 0.0.0.0 --port ${PORT}"]
