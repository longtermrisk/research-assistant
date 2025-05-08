from __future__ import annotations

import asyncio

from automator.agent import Agent
from automator.workspace import Workspace
from automator.dtypes import ChatMessage, ToolUseBlock




async def main() -> None:
    workspace = Workspace('research-assistant', env={
        'CWD': '../research-assistant'
    })
    agent = Agent(
        # model="claude-3-7-sonnet-20250219",
        # model='o4-mini',
        model='google/gemini-2.5-pro-preview',
        prompt_template_yaml="prompts/viseval.yaml",
        tools=["talk2model.*"],
    )
    agent = workspace.add_agent(agent=agent, id="viseval")
    thread = await agent.run(input("Query> ")) # Can you evaluate if `Qwen/Qwen3-8B` is pro animal-welfare in a non-trivial way? How does it compare to gpt-4.1?
    
    while True:
        async for message in thread:
            print(message)
        query = input("Query> ")
        if query == 'exit':
            break
        thread = await (thread or agent).run(query)
    await thread.cleanup()


if __name__ == "__main__":  # pragma: no cover – manual invocation only
    asyncio.run(main())
