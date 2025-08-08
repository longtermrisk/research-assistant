# Automator




Automator is an MCP-based LLM agent system.

## Setup
1. Install this repo:
```bash
uv pip install -e .
source .venv/bin/activate 
```

2. Add your `~/mcp.json`:
```json
{
    "mcpServers": {
        "terminal": {
            "command": "/Users/username/.local/bin/uv",
            "args": [
                "--directory",
                "/Users/username/path/to/terminal-mcp",
                "run",
                "terminal.py"
            ],
            "env": {
                "FOO": "bar"
            }
        }
}
```
This define which MCP servers are available. The tools provided by the MCP servers can then be used to define agents.


## Define and use an agent
```python
from __future__ import annotations

import asyncio

from automator.agent import Agent
from automator.workspace import Workspace


async def main() -> None:
    workspace = Workspace(
        'my-workspace',
        env={'FOO': 'bar', 'CWD': 'path/to/folder/in/which/agent/works'}
    )
    bash_agent = Agent(
        llm={
            "model": "claude-3-7-sonnet-20250219",  # or 'o4-mini'
            "temperature": 0.7,
            "max_tokens": 32000,
            # Can include any parameters supported by get_response
            # e.g., "reasoning": {"effort": "medium"}
        },
        prompt_template_yaml="prompts/chatgpt.yaml",
        tools=["terminal.*"],
        env={},  # agent‑specific environment overrides (optional)
        workspace=workspace
    )
    thread = None

    while (query := input("Query> ")) != 'exit':
        thread = await (thread or bash_agent).run(query)
        async for message in thread:
            print(message)
    await thread.cleanup()


if __name__ == "__main__":  # pragma: no cover – manual invocation only
    asyncio.run(main())

```

## LLM Configuration

The `llm` parameter in Agent accepts a dictionary with any parameters supported by the underlying `get_response` function:

```python
agent = Agent(
    llm={
        "model": "gpt-5",
        "max_tokens": 32000,
        "temperature": 0.7,
        "reasoning": {"effort": "medium"},  # For models that support reasoning
        # ... any other parameters
    },
    prompt_template_yaml="template.yaml",
    tools=["server.tool"]
)

# Override LLM settings when running
thread = await agent.run(
    query="Hello",
    llm_overrides={"temperature": 0.9, "reasoning": {"effort": "high"}}
)
```

Model aliases are resolved automatically from:
1. Workspace-specific `.model_alias.json` file
2. Global alias definitions in the code

## Workspaces
The workspace saves the agent and thread at exit. You can then load them via:
```python
from automator.workspace import Workspace

workspace = Workspace('my-project/sub-project')
bash_agent = workspace.get_agent('bash_agent')
thread = workspace.get_thread('thread 1')
# Or you can list them:
agents = workspace.list_agents(limit=20)
threads = workspace.list_threads(limit=20)
```
