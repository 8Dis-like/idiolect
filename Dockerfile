# Multi-stage Docker build for CodeForge
# Stage 1: Build (install dependencies)
# Stage 2: Runtime (slim image with model)

# ============================================================
# Stage 1: Builder
# ============================================================
FROM python:3.11-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml .
COPY codeforge/ codeforge/

# Install dependencies
RUN pip install --no-cache-dir -e .

# ============================================================
# Stage 2: Runtime
# ============================================================
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

WORKDIR /app

# Install Python
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /app /app

# Copy model artifacts (mount or COPY during CI/CD)
# COPY checkpoints/ checkpoints/
# COPY artifacts/ artifacts/
COPY configs/serve.yaml configs/serve.yaml

# Environment
ENV PYTHONUNBUFFERED=1
ENV CODEFORGE_CONFIG=configs/serve.yaml

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Expose API port
EXPOSE 8000

# Run API server
CMD ["python3", "-m", "uvicorn", "codeforge.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
