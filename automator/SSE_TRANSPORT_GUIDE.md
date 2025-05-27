# SSE Transport Support for MCP Clients

This update adds support for Server-Sent Events (SSE) transport to the automator package, enabling connections to remote MCP servers over HTTP/HTTPS.

## What's Changed

The `Thread` class in `automator/agent.py` now supports both stdio (local) and SSE (remote) transports for MCP servers.

## Configuration

### Stdio Transport (Default - Backward Compatible)

For local MCP servers using stdio transport, the configuration remains the same in your `mcp.json`:

```json
{
  "mcpServers": {
    "local-server": {
      "command": "python",
      "args": ["path/to/server.py"],
      "env": {
        "API_KEY": "your-key"
      }
    }
  }
}
```

### SSE Transport (New)

For remote MCP servers using SSE transport, add the `transport` field and specify the URL:

```json
{
  "mcpServers": {
    "remote-server": {
      "transport": "sse",
      "url": "https://example.com/sse",
      "headers": {
        "Authorization": "Bearer your-token"
      },
      "timeout": 5,
      "sse_read_timeout": 300,
      "env": {
        "API_TOKEN": "your-api-token"
      }
    }
  }
}
```

#### SSE Configuration Options:

- **transport**: Set to `"sse"` to use SSE transport
- **url**: The SSE endpoint URL (required for SSE)
- **headers**: Optional HTTP headers to send with requests
- **timeout**: HTTP operation timeout in seconds (default: 5)
- **sse_read_timeout**: How long to wait for SSE events in seconds (default: 300)
- **env**: Environment variables - for SSE, any keys ending with `_TOKEN` or `_KEY` are automatically added to headers as `X-{KEY-NAME}`

## Security Considerations

When using SSE transport:

1. **Always use HTTPS** for production deployments
2. **Validate SSL certificates** (handled automatically by the client)
3. **Use proper authentication** via headers or tokens
4. **Be aware of CORS policies** if connecting from web browsers
5. **Implement rate limiting** on your SSE servers

## Example Usage

No changes are needed in your code - just update your `mcp.json` configuration:

```python
from automator import Agent

# Works with both stdio and SSE servers transparently
agent = Agent(
    model="claude-3-sonnet-20240229",
    prompt_template_yaml="prompts/example.yaml",
    tools=["local-server.tool1", "remote-server.tool2"]  # Mix of local and remote
)

thread = await agent.run(query="Hello")
```

## Migration Guide

To migrate an existing stdio server to SSE:

1. Deploy your MCP server with SSE transport support
2. Update your `mcp.json` to include the new server configuration with `"transport": "sse"`
3. Update any authentication tokens or headers as needed
4. No code changes required - the agent will automatically use the correct transport

## Troubleshooting

### Connection Issues
- Verify the SSE URL is accessible
- Check firewall rules and network connectivity
- Ensure proper authentication headers are set

### Timeout Issues
- Increase `timeout` for slow initial connections
- Increase `sse_read_timeout` for servers with infrequent events

### Authentication Errors
- Verify tokens/keys in both `headers` and `env` sections
- Check if the server expects specific header formats