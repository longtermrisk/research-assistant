from __future__ import annotations

import asyncio

from automator.agent import Agent
from automator.workspace import Workspace
from automator.dtypes import SubagentToolDefinition

from rag.hook import create_rag_hook


create_rag_hook('.knowledge')


async def main() -> None:
    workspace = Workspace('researchoor')
    
    # Create some experts
    agent = Agent(
        model="gpt-4.1",
        prompt_template_yaml="prompts/chatgpt.yaml",
        workspace=workspace,
        tools=['terminal.*'],
        id='dev',
        hooks=['rag:.knowledge']
    )
    thread = await agent.run(input("Query> "), thread_id='example-thread')

    async for message in thread:
        print(message)
    
    query = input("Query> ")
    while query != 'exit':
        thread = await thread.run(query)
        async for message in thread:
            print(message['meta'])
            for block in message['content']:
                try:
                    print(block.text)
                except:
                    print(block)
        query = input("Query> ")
    
    await thread.cleanup()

if __name__ == "__main__":  # pragma: no cover – manual invocation only
    asyncio.run(main())
