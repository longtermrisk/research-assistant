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
        model="claude-3-7-sonnet-20250219", # or 'o4-mini'
        prompt_template_yaml="prompts/chatgpt.yaml",
        tools=["terminal.*"],
        env={},  # agent‑specific environment overrides (optional)
    )
    bash_agent = workspace.add_agent(agent=bash_agent, id="bash")
    thread = None

    while (query := input("Query> ")) != 'exit':
        thread = await (thread or bash_agent).run(query)
        async for message in thread:
            print(message)
    await thread.cleanup()


if __name__ == "__main__":  # pragma: no cover – manual invocation only
    asyncio.run(main())

```

## Workspaces
```python
from automator.workspace import Workspace

workspace = Workspace('my-project/sub-project', env={"FOO": "override bar"})
bash_agent = workspace.add_agent(agent=bash_agent, id='bash_agent')         # Return an agent because the workspace agent now has modified env
workspace.add_thread(thread=thread, id='My first thread')
```

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
