# ═══════════════════════════════════════════════════════════════════════════
# EcoPackAI — Production Multi-Stage Dockerfile (Prompt 36)
# ═══════════════════════════════════════════════════════════════════════════
# Multi-stage build for the FastAPI packing service.
#   Stage 1 (builder): Install dependencies into a virtual environment.
#   Stage 2 (runtime): Copy venv, add non-root user, minimal attack surface.
#
# Build:  docker build -t ecopackai:latest .
# Run:    docker run -p 8000:8000 ecopackai:latest
# ═══════════════════════════════════════════════════════════════════════════

# ---------------------------------------------------------------------------
# Stage 1 — Builder
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS builder

WORKDIR /build

# System deps for compilation (numpy, scikit-learn, etc.)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        gcc g++ libffi-dev curl && \
    rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# ---------------------------------------------------------------------------
# Stage 2 — Runtime
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS runtime

LABEL maintainer="EcoPackAI Engineering <eng@ecopackai.dev>"
LABEL version="1.0.0"
LABEL description="EcoPackAI FastAPI Packaging Optimization Service"

# Security: non-root user
RUN groupadd --gid 1000 ecopack && \
    useradd --uid 1000 --gid ecopack --shell /bin/bash --create-home ecopack

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Application directory
WORKDIR /app

# Copy application code
COPY src/ ./src/
COPY models/ ./models/
COPY data/box_catalogue.json ./data/box_catalogue.json
COPY alembic.ini ./
COPY migrations/ ./migrations/

# Set Python path for module imports
ENV PYTHONPATH="/app"
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Application environment
ENV APP_ENV="production"
ENV LOG_LEVEL="info"
ENV MODEL_PATH="/app/models"
ENV PORT=8000
ENV WORKERS=4

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/v1/health')" || exit 1

# Expose port
EXPOSE ${PORT}

# Switch to non-root user
RUN chown -R ecopack:ecopack /app
USER ecopack

# Start FastAPI with Uvicorn
CMD ["uvicorn", "src.classify_api:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "4", \
     "--timeout-keep-alive", "30", \
     "--access-log", \
     "--log-level", "info"]
