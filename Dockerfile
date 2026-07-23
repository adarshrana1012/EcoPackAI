# ==========================================================
# EcoPackAI - Unified Full-Stack Dockerfile
# ==========================================================

# ----------------------------------------------------------
# Stage 1: Build the React Frontend
# ----------------------------------------------------------
FROM node:20-alpine AS frontend-builder
WORKDIR /app/frontend

# Install dependencies first for layer caching
COPY frontend/package*.json ./
RUN npm install

# Copy frontend source and build
COPY frontend/ ./
RUN npm run build

# ----------------------------------------------------------
# Stage 2: Build the Python Backend Dependencies
# ----------------------------------------------------------
FROM python:3.12-slim AS backend-builder
WORKDIR /build

# Install compilation dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libffi-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ----------------------------------------------------------
# Stage 3: Final Runtime Image
# ----------------------------------------------------------
FROM python:3.12-slim

# Prevent Python from buffering logs
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install runtime tools
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Add non-root user
RUN groupadd --gid 1000 ecopack && \
    useradd --uid 1000 --gid ecopack --shell /bin/bash --create-home ecopack

WORKDIR /app

# Copy virtual environment
COPY --from=backend-builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy backend application
COPY src/ ./src/
COPY models/ ./models/
COPY data/box_catalogue.json ./data/box_catalogue.json
COPY alembic.ini ./
COPY migrations/ ./migrations/

# Copy built frontend
COPY --from=frontend-builder /app/frontend/dist ./frontend/dist

# Ensure permissions
RUN chown -R ecopack:ecopack /app
USER ecopack

ENV PYTHONPATH=/app

# Railway injects PORT. Defaults to 8080 locally.
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
CMD curl -f http://localhost:${PORT:-8080}/v1/health || exit 1

# Start FastAPI
CMD ["sh", "-c", "uvicorn src.main:app --host 0.0.0.0 --port ${PORT:-8080}"]