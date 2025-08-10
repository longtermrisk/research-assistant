# Automator Docker Setup

This Docker setup allows you to run the automator system in a containerized environment, making it easy to test and develop bots based on the automator system.

## Quick Start

### Option 1: Using Docker Compose (Recommended)

1. Create a `.env` file with your API keys:
```bash
OPENAI_API_KEY=your_openai_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here
HF_TOKEN=your_huggingface_token_here
OPENWEIGHTS_API_KEY=your_openweights_key_here
SERP_API_KEY=your_serp_api_key_here
```

2. Run tests using Docker Compose:
```bash
docker-compose up --build
```

3. To run a custom command:
```bash
docker-compose run --rm automator sh -c "cd /app/automator && uv run python examples/quickstart.py"
```

### Option 2: Using Docker directly

1. Build the image:
```bash
docker build -t automator:latest .
```

2. Run tests with environment variables:
```bash
docker run --rm \
  -e OPENAI_API_KEY="your_key" \
  -e ANTHROPIC_API_KEY="your_key" \
  -e HF_TOKEN="your_token" \
  -e OPENWEIGHTS_API_KEY="your_key" \
  -e SERP_API_KEY="your_key" \
  automator:latest
```

3. Run a custom command:
```bash
docker run --rm \
  -e OPENAI_API_KEY="your_key" \
  -e ANTHROPIC_API_KEY="your_key" \
  automator:latest \
  sh -c "cd /app/automator && uv run python examples/quickstart.py"
```

### Option 3: Using the shell script

1. Export your environment variables:
```bash
export OPENAI_API_KEY="your_key"
export ANTHROPIC_API_KEY="your_key"
export HF_TOKEN="your_token" 
export OPENWEIGHTS_API_KEY="your_key"
export SERP_API_KEY="your_key"
```

2. Run the test script:
```bash
chmod +x run_tests_docker.sh
./run_tests_docker.sh
```

## Environment Variables

The following environment variables are supported:

- `OPENAI_API_KEY`: OpenAI API key for GPT models
- `ANTHROPIC_API_KEY`: Anthropic API key for Claude models  
- `HF_TOKEN`: Hugging Face token for model access
- `OPENWEIGHTS_API_KEY`: OpenWeights API key
- `SERP_API_KEY`: SERP API key for web search functionality

## MCP Server Configuration

The Docker container uses a special `docker_mcp.json` configuration that supports environment variable substitution. MCP server configurations can reference environment variables using the `$VARIABLE_NAME` syntax:

```json
{
    "mcpServers": {
        "terminal": {
            "command": "/usr/local/bin/uv",
            "args": ["--directory", "/app/terminal-mcp", "run", "entrypoint.py"],
            "env": {
                "OPENAI_API_KEY": "$OPENAI_API_KEY"
            }
        }
    }
}
```

The automator system automatically resolves these variables at runtime using the environment variables passed to the Docker container.

## Using the Image for Bot Development

You can extend this Docker image to create your own bots:

```dockerfile
FROM automator:latest

# Copy your bot code
COPY my_bot.py /app/
COPY my_prompts/ /root/.automator/prompts/

# Run your bot
CMD ["uv", "run", "python", "/app/my_bot.py"]
```

Or mount your development directory:

```bash
docker run --rm \
  -v $(pwd):/workspace \
  -w /workspace \
  -e OPENAI_API_KEY="your_key" \
  automator:latest \
  sh -c "uv run python your_bot.py"
```

## Available Commands

Inside the container, you can run:

- `cd /app/automator && uv run pytest tests/` - Run all tests
- `cd /app/automator && uv run python examples/quickstart.py` - Run quickstart example
- `cd /app/automator && uv run python examples/rag/rag_hook_example.py` - RAG example
- `uv run --directory /app/terminal-mcp entrypoint.py` - Start terminal MCP server directly

## Persistent Data

The Docker Compose setup includes a volume for workspace data:
- Workspaces are stored in `/root/.automator/workspaces` 
- The volume `automator_workspaces` persists this data between container runs

## Troubleshooting

1. **Tests failing due to missing API keys**: Ensure all required environment variables are set
2. **Permission issues**: The container runs as root, files created may have root ownership
3. **Network issues**: Some tests may require internet access for API calls

## Limitations

- The Docker setup does not include the UI dashboard (only backend functionality)
- Terminal-sandboxed MCP server (Docker-in-Docker) is not included to avoid complexity
- Some system-specific features may not work identically to local installations