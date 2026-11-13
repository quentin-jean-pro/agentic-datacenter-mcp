# ==============================================================================
# Production Dockerfile for Green AI Data Center Agentic MCP Platform
# ==============================================================================

FROM python:3.11-slim AS base

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency specifications
COPY requirements.txt pyproject.toml ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and scripts
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY tests/ ./tests/
COPY .env.example .env

# Expose ports:
# 8001: Compute Telemetry FastMCP (SSE)
# 8002: Workload Policy FastMCP (SSE)
# 8501: Streamlit Interactive Web App
EXPOSE 8001 8002 8501

# Healthcheck to verify FastMCP / Web server response
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Default entrypoint: launch Streamlit web application
CMD ["streamlit", "run", "src/app.py", "--server.port=8501", "--server.address=0.0.0.0"]

