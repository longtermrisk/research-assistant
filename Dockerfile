# Use Python 3.11 as base image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PATH="/root/.local/bin:$PATH"

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Node.js 18 for frontend
RUN curl -fsSL https://deb.nodesource.com/setup_18.x | bash - \
    && apt-get install -y nodejs

# Set working directory
WORKDIR /app

# Copy repository structure
COPY . .

# Install uv, then use it to install all Python packages
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    uv pip install --system \
        -e ./terminal-mcp \
        -e ./web-mcp \
        -e ./talk-to-model \
        -e ./automator"[all]"

# Install frontend dependencies
RUN cd automator/ui/frontend && npm install

# Create automator user directories and copy prompts
RUN mkdir -p /root/.automator/workspaces && \
    cp -r automator/prompts /root/.automator/prompts

# Create MCP configuration
RUN uv_path=$(which uv) && \
    repo_path=/app && \
    sed "s|/path/to/uv|$uv_path|g; s|/path/to/repo|$repo_path|g" mcp.json > /root/mcp.json

# Copy test script into the image
COPY docker_test.py /app/docker_test.py
RUN chmod +x /app/docker_test.py

# Create a startup script
COPY <<EOF /app/start.sh
#!/bin/bash
set -e

echo "Starting Automator System..."

# Set up environment variables if they exist
if [ -f /app/.env ]; then
    export \$(grep -v '^#' /app/.env | xargs)
fi

# Start backend in background
echo "Starting backend..."
cd /app/automator && uvicorn automator.api.main:app --host 0.0.0.0 --port 8000 &

# Start frontend in background
echo "Starting frontend..."
cd /app/automator/ui/frontend && npm run dev -- --host 0.0.0.0 --port 5173 &

# Wait for services to start
sleep 5

echo "Services started!"
echo "Backend available at: http://localhost:8000"
echo "Frontend available at: http://localhost:5173"

# Keep container running
wait
EOF
RUN chmod +x /app/start.sh

# Expose ports
EXPOSE 8000 5173

# Health check to verify installation
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python /app/docker_test.py

# Default command runs tests first, then starts services
CMD ["bash", "-c", "python /app/docker_test.py && /app/start.sh"]