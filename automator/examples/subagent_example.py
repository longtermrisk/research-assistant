from __future__ import annotations

import asyncio

from automator.agent import Agent
from automator.workspace import Workspace
from automator.dtypes import SubagentToolDefinition


async def main() -> None:
    workspace = Workspace('demo')
    
    # Create some experts
    alice = Agent(
        model="gpt-4.1",
        prompt_template_yaml="prompts/alice.yaml",
        workspace=workspace,
        id='alice',
    )
    bob = Agent(
        model="gpt-4.1",
        tools=['terminal.*'],
        prompt_template_yaml="prompts/bob.yaml",
        workspace=workspace,
        id='bob',
    )

    agent = Agent(
        model="gpt-4.1",
        prompt_template_yaml="prompts/chatgpt.yaml",
        subagents = ['alice', 'bob'],
        workspace=workspace,
    )

    thread = await agent.run(input("Query> "), thread_id='example-thread')

    async for message in thread:
        print(message)
    
    query = input("Query> ")
    while query != 'exit':
        thread = await thread.run(query)
        async for message in thread:
            print(message)
        query = input("Query> ")
    
    await thread.cleanup()

if __name__ == "__main__":  # pragma: no cover – manual invocation only
    asyncio.run(main())
