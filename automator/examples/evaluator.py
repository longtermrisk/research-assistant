from __future__ import annotations

import asyncio

from automator.agent import Agent
from automator.workspace import Workspace
from automator.dtypes import ChatMessage, ToolUseBlock


setup_env = """git clone https://github.com/longtermrisk/openweights.git
cd openweights
uv pip install -e .
cd ..
git clone https://github.com/nielsrolf/viseval.git
cd viseval
uv pip install -e .
cd .."""


async def main() -> None:
    workspace = Workspace('research-assistant', env={
        'CWD': '../research-assistant'
    })
    evaluator = Agent(
        # model="claude-3-7-sonnet-20250219",
        model='gpt-4.1',
        prompt_template_yaml="prompts/evaluator.yaml",
        tools=[
            "talk2model.*",
            "terminal.*",
        ],
    )
    evaluator = workspace.add_agent(agent=evaluator, id="evaluator")
    thread = await evaluator.run(input("Query> "))
    # Slightly hacky way to setup the agent's environment
    setup_msg = ChatMessage(role='user', content=[ToolUseBlock(id='123', name='terminal_execute', input={'command': setup_env, 'detach_after_seconds': 300})])
    setup_logs, _ = await thread.process_message(setup_msg)
    print('==' * 30 + ' Setup Logs ' + '==' * 30)
    print(setup_logs.content[0].content[0].text)
    print('==' * 80)
    
    while True:
        async for message in thread:
            print(message)
        query = input("Query> ")
        if query == 'exit':
            break
        thread = await (thread or evaluator).run(query)
    await thread.cleanup()


if __name__ == "__main__":  # pragma: no cover – manual invocation only
    asyncio.run(main())
