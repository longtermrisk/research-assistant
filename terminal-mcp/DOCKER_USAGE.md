# Docker Usage Guide for Terminal MCP


## Docker Image

Build the image via: `docker build -t nielsrolf/terminal-mcp .`

## Usage

For your MCP configuration, use this setup:

```json
{
  "mcpServers": {
    "terminal-sandboxed": {
      "command": "docker",
      "args": [
        "run",
        "--rm",
        "-i",
        "-v", "$CWD:/workspace",
        "-w", "/workspace",
        "nielsrolf/terminal-mcp:latest"
      ],
      "env": {
          "OPENAI_API_KEY": "...",
          "HF_TOKEN": "...",
          "OPENWEIGHTS_API_KEY": "...",
          "ANTHROPIC_API_KEY": "..."
      }
    }
  }
}
```

## When should I run terminal-mcp via docker?
Running terminal-mcp via docker provides a secure sandbox that restricts access to files outside of the workspace. This is ideal for running untrusted models or deployments with multiple users. For collaborative projects (you write some code, an agent writes some code), it can be annoying because the agent and you will use different environments.


| Aspect | Host Deployment | Docker Deployment |
|--------|----------------|-------------------|
| **Security** | Full host access | Sandboxed to workspace |
| **Environment** | Shared with user | Isolated container env |
| **Performance** | Native speed | Slight container overhead |
| **Setup** | Simpler | Requires Docker |
| **Collaboration** | Same environment as user | Different but consistent |
| **Multi-user** | Complex isolation | Easy per-user containers |

## Recommendations

- **For collaborative projects**: Use host deployment for shared environment
- **For secure/multi-user deployment**: Use Docker for isolation
- **For production**: Use Docker with additional security measures (user namespaces, resource limits, etc.)