from __future__ import annotations

import asyncio

from automator.agent import Agent
from automator.workspace import Workspace


async def main() -> None:
    workspace = Workspace('my-workspace')
    bash_agent = Agent(
        model='gemini-2.5-pro',
        prompt_template_yaml="prompts/chatgpt.yaml",
        tools=["terminal.*",]
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
