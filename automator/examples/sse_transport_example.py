"""
Example demonstrating SSE transport support for MCP clients.

This example shows how to configure both stdio (local) and SSE (remote) MCP servers
in your mcp.json configuration file.
"""

import asyncio
import json
from pathlib import Path
from automator import Agent

# Example mcp.json configuration showing both transport types
EXAMPLE_MCP_CONFIG = {
    "mcpServers": {
        # Traditional stdio transport (local server)
        "local-weather": {
            "command": "python",
            "args": ["./weather_server.py"],
            "env": {
                "WEATHER_API_KEY": "your-api-key"
            }
        },
        
        # New SSE transport (remote server)
        "remote-calculator": {
            "transport": "sse",
            "url": "https://mcp-calculator.example.com/sse",
            "headers": {
                "Authorization": "Bearer your-bearer-token",
                "X-API-Version": "1.0"
            },
            "timeout": 10,
            "sse_read_timeout": 600,
            "env": {
                "CALC_API_TOKEN": "your-calc-token"  # Will be added as X-CALC-API-TOKEN header
            }
        },
        
        # Another SSE example with minimal config
        "public-facts": {
            "transport": "sse",
            "url": "https://facts-api.example.com/mcp/sse"
        }
    }
}

async def main():
    # Save example config (in practice, this would already exist at ~/mcp.json)
    config_path = Path.home() / "mcp.json.example"
    with open(config_path, 'w') as f:
        json.dump(EXAMPLE_MCP_CONFIG, f, indent=2)
    
    print(f"Example configuration saved to: {config_path}")
    print("\nTo use this configuration, rename it to ~/mcp.json")
    
    # Example of using an agent with mixed transport types
    try:
        agent = Agent(
            model="claude-3-sonnet-20240229",
            prompt_template_yaml="prompts/example.yaml",
            tools=[
                "local-weather.*",           # All tools from local stdio server
                "remote-calculator.add",      # Specific tool from remote SSE server
                "remote-calculator.multiply", # Another tool from remote SSE server
                "public-facts.random_fact"    # Tool from public SSE server
            ],
            env={
                "USER_PREFERENCE": "metric"  # Additional env vars passed to all servers
            }
        )
        
        # The agent handles both transport types transparently
        thread = await agent.run(
            query="What's the weather in Tokyo? Also, calculate 42 * 17 and tell me a random fact."
        )
        
        async for message in thread:
            print(f"\nMessage: {message}")
            
    except Exception as e:
        print(f"\nNote: This example requires actual MCP servers to be running.")
        print(f"Error: {e}")
        print("\nExample server configurations have been saved to demonstrate the format.")

if __name__ == "__main__":
    asyncio.run(main())