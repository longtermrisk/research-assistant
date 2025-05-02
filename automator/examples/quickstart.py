from __future__ import annotations

import asyncio

from automator.agent import Agent
from automator.workspace import Workspace


async def main() -> None:
    workspace = Workspace('my-workspace', env={'FOO': 'bar'})
    bash_agent = Agent(
        # model="claude-3-7-sonnet-20250219",
        model="claude-3-5-haiku-20241022",
        # model='o4-mini',
        prompt_template_yaml="prompts/chatgpt.yaml",
        tools=[
            "terminal.terminal_execute",
            "terminal.terminal_stdin",
            "terminal.terminal_logs",
        ],
        env={},  # agent‑specific environment overrides (optional)
    )
    bash_agent = workspace.add_agent(agent=bash_agent, id="bash")
    thread = None

    while (query := input("Query> ")) != 'exit':
        thread = await (thread or bash_agent).run(query)
        async for message in thread:
            print(message)
    workspace.add_thread(thread=thread, id=input("Thread ID: "))

    await thread.cleanup()


if __name__ == "__main__":  # pragma: no cover – manual invocation only
    asyncio.run(main())
