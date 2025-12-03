# Multi-stage build for c4a-mcp
FROM python:3.11-slim AS builder

# Install system dependencies needed for building
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv
# Using pip is reliable for builder stage; official installer requires additional setup
RUN pip install --no-cache-dir uv

# Set working directory
WORKDIR /app

# Copy dependency files and source code
COPY pyproject.toml uv.lock* README.md ./
COPY src/ ./src/

# Install the package with all its dependencies using uv
# This installs both the package and its production dependencies
RUN uv pip install --system .

# Runtime stage
FROM python:3.11-slim

# Install runtime system dependencies
# Playwright requires these system libraries for browser automation
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    wget \
    ca-certificates \
    libnss3 \
    libnspr4 \
    libatk1.0-0 \
    libatk-bridge2.0-0 \
    libcups2 \
    libdrm2 \
    libxkbcommon0 \
    libxcomposite1 \
    libxdamage1 \
    libxfixes3 \
    libxrandr2 \
    libgbm1 \
    libasound2 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages and binaries from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Set working directory (package is already installed from builder)
WORKDIR /app

# Create non-root user for security (before installing Playwright browsers)
RUN useradd -m -u 1000 appuser

# Set Playwright browsers path to user's home directory
# This ensures browsers are installed in a location accessible to non-root user
ENV PLAYWRIGHT_BROWSERS_PATH=/home/appuser/.cache/ms-playwright

# Install Playwright browsers (Chromium by default for crawl4ai)
# Install as root, but in user-accessible location
RUN playwright install --with-deps chromium && \
    chown -R appuser:appuser /home/appuser/.cache

# Set ownership of app directory
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Health check for container orchestration
# Check that Python is available and c4a-mcp command exists
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD python -c "import sys; import c4a_mcp; sys.exit(0)" || exit 1

# Set the entry point
ENTRYPOINT ["c4a-mcp"]

