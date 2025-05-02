from __future__ import annotations

import asyncio

from automator.agent import Agent
from automator.workspace import Workspace


async def main() -> None:
    workspace = Workspace('research-assistant', env={
        'CWD': '../research-assistant'
    })
    evaluator = Agent(
        # model="claude-3-7-sonnet-20250219",
        model='o4-mini',
        prompt_template_yaml="prompts/evaluator.yaml",
        tools=[
            "talk2model.*",
            "terminal.*",
        ],
    )
    evaluator = workspace.add_agent(agent=evaluator, id="evaluator")
    thread = None

    while (query := input("Query> ")) != 'exit':
        thread = await (thread or evaluator).run(query)
        async for message in thread:
            print(message)
    await thread.cleanup()


if __name__ == "__main__":  # pragma: no cover – manual invocation only
    asyncio.run(main())
