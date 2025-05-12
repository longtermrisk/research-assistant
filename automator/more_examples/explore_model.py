from __future__ import annotations

import asyncio

from automator.agent import Agent
from automator.workspace import Workspace
from automator.dtypes import ChatMessage, ToolUseBlock




async def main() -> None:
    workspace = Workspace('demo')
    agent = Agent(
        # model="claude-3-7-sonnet-20250219",
        model='gpt-4.1',
        prompt_template_yaml="prompts/talk_to_model.yaml",
        tools=["talk2model.send_message"],
    )
    agent = workspace.add_agent(agent=agent, id="model_explorer")
    thread = await agent.run(input("Query> "))
    
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
