# Use Python 3.11 slim image
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install uv

# Set working directory
WORKDIR /app

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install Python dependencies
RUN uv sync --frozen

# Copy application code
COPY . .

# Create workspace directory
RUN mkdir -p /workspace

# Set the workspace as the default working directory for commands
WORKDIR /workspace

# Expose the MCP server (though it typically uses stdio)
# This is mainly for documentation purposes
EXPOSE 3000

# Set the entrypoint to run the MCP server
# We'll use the full path to the uv executable and python from the venv
ENTRYPOINT ["/app/.venv/bin/python", "/app/entrypoint.py"]