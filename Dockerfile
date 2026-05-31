FROM node:20-alpine AS frontend

RUN corepack enable && corepack prepare pnpm@9.12.0 --activate

WORKDIR /web

COPY frontend/package.json frontend/pnpm-lock.yaml ./
RUN pnpm install --frozen-lockfile

COPY frontend/ ./
RUN pnpm build


FROM python:3.11-slim AS backend-builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.5.11 /uv /usr/local/bin/uv

WORKDIR /app

COPY pyproject.toml uv.lock ./
COPY src ./src

RUN uv sync --frozen --no-dev


FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    HOME=/home/user \
    SADAR_PROCESSED_DIR=/app/data/processed \
    SADAR_MODELS_DIR=/app/models \
    SADAR_REPORTS_DIR=/app/reports \
    SADAR_SERVE_CONFIG=/app/configs/serve.yaml \
    SADAR_FRONTEND_DIR=/app/frontend/dist \
    PORT=7860

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 1000 --shell /bin/bash user

WORKDIR /app

COPY --from=backend-builder /opt/venv /opt/venv
COPY --from=backend-builder /app/src /app/src
COPY --from=frontend /web/dist /app/frontend/dist

COPY configs ./configs
COPY pyproject.toml ./
COPY models ./models
COPY reports ./reports
COPY data/processed/scaler.npz data/processed/test.npy data/processed/val.npy ./data/processed/

RUN chown -R user:user /app

USER user

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -fsS http://localhost:${PORT}/api/health || exit 1

CMD ["sh", "-c", "uvicorn sadar.serve.app:app --host 0.0.0.0 --port ${PORT}"]
