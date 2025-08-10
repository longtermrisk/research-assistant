# Use Python 3.11 as base image
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install UV package manager
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# Set working directory
WORKDIR /app

# Copy the entire repository
COPY . /app/

# Copy mcp.json to home directory
COPY mcp.json /root/mcp.json

# Set up the automator components
RUN cd /app/terminal-mcp && uv sync
RUN cd /app/web-mcp && uv sync  
RUN cd /app/talk-to-model && uv sync

# Install automator with RAG support
RUN cd /app/automator && uv sync
RUN cd /app/automator && uv pip install -e ".[all]"

# Create .automator directory and copy prompts
RUN mkdir -p /root/.automator/workspaces
RUN cp -r /app/automator/prompts /root/.automator/prompts

# Set environment variables for paths used in tests
ENV HOME=/root
ENV PYTHONPATH=/app/automator:$PYTHONPATH

# Default command to run tests
CMD ["sh", "-c", "cd /app/automator && uv run pytest tests/ -v"]