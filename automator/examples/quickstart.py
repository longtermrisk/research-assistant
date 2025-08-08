from __future__ import annotations

import asyncio

from automator.agent import Agent
from automator.workspace import Workspace
from localrouter import TextBlock, ToolUseBlock, ToolResultBlock


async def main() -> None:
    workspace = Workspace('my-workspace')
    bash_agent = Agent(
        llm={"model": "gemini-2.5-pro"},
        prompt_template_yaml="assistant.yaml",
        tools=["terminal.*",]
    )
    bash_agent = workspace.add_agent(agent=bash_agent, id="bash")
    thread = None

    while (query := input("Query> ")) != 'exit':
        thread = await (thread or bash_agent).run(query)
        async for message in thread:
            for block in message.content:
                if isinstance(block, TextBlock):
                    print(block.text)
                elif isinstance(block, ToolUseBlock):
                    print(block.name, block.input)
                elif isinstance(block, ToolResultBlock):
                    for part in block.content:
                        try:
                            print(part.text)
                        except:
                            print(part)
                else:
                    print(block)
    workspace.add_thread(thread=thread, id=input("Thread ID: "))

    await thread.cleanup()


if __name__ == "__main__":  # pragma: no cover – manual invocation only
    asyncio.run(main())
