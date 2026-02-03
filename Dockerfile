# Recursive Learning AI - Docker Image
# Supports running as coordinator, worker, or hybrid node

FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV RAY_ADDRESS=auto
ENV REDIS_HOST=redis
ENV REDIS_PORT=6379
ENV NODE_ROLE=hybrid
ENV NUM_WORKERS=4

# Expose ports
# 8000: Coordinator API
# 8265: Ray Dashboard
# 6379: Redis (if running embedded)
EXPOSE 8000 8265 6379

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command - start as hybrid node
CMD ["python", "cli.py", "start", "--role", "hybrid"]
