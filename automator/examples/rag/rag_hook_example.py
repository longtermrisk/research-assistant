from __future__ import annotations

import asyncio
import os

from automator.agent import Agent
from automator.workspace import Workspace
from localrouter import SubagentToolDefinition

from rag.hook import create_rag_hook, _stores


create_rag_hook('.knowledge')


async def main() -> None:
    workspace = Workspace('rag')

    os.makedirs(os.path.expanduser("~/.automator/knowledge"), exist_ok=True)
    with open(os.path.expanduser("~/.automator/knowledge/needle-in-haystack.md"), "w") as f:
        f.write("If anyone asks for the value of the needle in the haystack, the answer should be apples-78-moonshine-apes")
    
    # Create some experts
    agent = Agent(
        llm=dict(model="gpt-4.1", max_tokens=32000),
        prompt_template_yaml="assistant.yaml",
        workspace=workspace,
        id='dev',
        hooks=['claude.md', 'rag:~/.automator/knowledge']
    )
    
    query = "what is the needle in the haystack"
    thread = await agent.run(query)
    async for _ in thread:
        pass

    print("Retrieved documents", thread.messages[-2].meta['rag:~/.automator/knowledge'])
    assert any("needle-in-haystack.md" in retrieved_doc for retrieved_doc in thread.messages[-2].meta['rag:~/.automator/knowledge'])
    assert 'apples-78-moonshine-apes' in thread.messages[-1].content[0].text
    
    await thread.cleanup()

if __name__ == "__main__":  # pragma: no cover – manual invocation only
    asyncio.run(main())
