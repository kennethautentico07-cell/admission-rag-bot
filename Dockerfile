# ==============================================================================
# Stage 1: Builder — Compile and install dependencies in an isolated virtualenv
# ==============================================================================
FROM python:3.11-slim AS builder

WORKDIR /build

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system build dependencies required only during wheel compilation
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create an isolated virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install CPU-optimized PyTorch first to avoid ~2GB of unnecessary CUDA libraries,
# then install all project dependencies from requirements.txt
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --extra-index-url https://download.pytorch.org/whl/cpu torch && \
    pip install -r requirements.txt

# ==============================================================================
# Stage 2: Runner — Minimal, secure production runtime image
# ==============================================================================
FROM python:3.11-slim AS runner

WORKDIR /app

# Runtime environment configuration
# Note on Secrets: GEMINI_API_KEY must be injected at runtime via --env-file .env
# or cloud platform environment secrets (never hardcoded into the image).
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TOKENIZERS_PARALLELISM=false \
    ANONYMIZED_TELEMETRY=False \
    HF_HOME=/app/.cache/huggingface \
    PATH="/opt/venv/bin:$PATH"

# Copy pre-built virtual environment from the builder stage (excludes build-essential)
COPY --from=builder /opt/venv /opt/venv

# Copy application source, knowledge base data, and assets
COPY app.py tpc_logo.png ./
COPY data/ ./data/
COPY src/ ./src/
COPY docs/ ./docs/

# Create a non-root user and grant ownership of /app (for ChromaDB & HF cache persistence)
RUN useradd --create-home --shell /bin/bash appuser && \
    mkdir -p /app/chroma_db /app/.cache/huggingface && \
    chown -R appuser:appuser /app

USER appuser

EXPOSE 8501

# Container healthcheck against Streamlit's internal health endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request, os; port = os.getenv('PORT', '8501'); urllib.request.urlopen(f'http://localhost:{port}/_stcore/health')" || exit 1

# Launch Streamlit (supports dynamic $PORT for Railway/Cloud Run or defaults to 8501)
CMD ["sh", "-c", "streamlit run app.py --server.port=${PORT:-8501} --server.address=0.0.0.0"]
