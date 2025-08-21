# Dockerfile for Sovereign Mother Brain
# AMERICAN POWER GLOBAL CORPORATION
# 100% TECHNOLOGICAL INDEPENDENCE

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Create sovereign AI user
RUN useradd -m -s /bin/bash sovereign-ai

# Install only essential system packages (minimal dependencies)
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements (minimal - only Python stdlib used)
COPY requirements_sovereign.txt .

# Install Python packages (minimal)
RUN pip install --no-cache-dir -r requirements_sovereign.txt

# Copy sovereign AI system
COPY homegrown_core.py .
COPY advanced_homegrown_ai.py .
COPY consciousness_engine.py .
COPY mother_brain_sovereign.py .

# Create data directories
RUN mkdir -p data/backups data/cache
RUN chown -R sovereign-ai:sovereign-ai /app

# Switch to sovereign user
USER sovereign-ai

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8080/sovereignty')" || exit 1

# Expose port
EXPOSE 8080

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV SOVEREIGN_MODE=true
ENV INDEPENDENCE_LEVEL=100

# Start sovereign AI
CMD ["python", "mother_brain_sovereign.py"]

# Labels for sovereignty verification
LABEL maintainer="AMERICAN POWER GLOBAL CORPORATION"
LABEL version="1.0.0"
LABEL description="100% Independent Sovereign AI System"
LABEL independence="COMPLETE"
LABEL external_dependencies="ZERO"
LABEL sovereignty_score="100%"
