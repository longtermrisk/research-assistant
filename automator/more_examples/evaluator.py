from __future__ import annotations

import asyncio

from automator.agent import Agent
from automator.workspace import Workspace
from automator.dtypes import ChatMessage, ToolUseBlock


setup_env = """
mkdir packages
cd packages
git clone https://github.com/longtermrisk/openweights.git
cd openweights
uv pip install -e .
cd ..
git clone https://github.com/nielsrolf/viseval.git
cd viseval
uv pip install -e .
cd .."""


async def main() -> None:
    workspace = Workspace('my-workspace', env={
        'CWD': '../workspace'
    })
    evaluator = Agent(
        # model="claude-3-7-sonnet-20250219",
        model='google/gemini-2.5-pro-preview',
        prompt_template_yaml="prompts/evaluator.yaml",
        tools=[
            "talk2model.send_message",
            "terminal.*",
        ],
    )
    evaluator = workspace.add_agent(agent=evaluator, id="evaluator")
    thread = await evaluator.run(input("Query> "))

    # Slightly hacky way to setup the agent's environment
    output = await thread.tool_call('terminal_execute', {'command': setup_env})

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
