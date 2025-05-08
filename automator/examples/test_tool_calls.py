from __future__ import annotations

import asyncio

from automator.agent import Agent
from automator.workspace import Workspace
from automator.dtypes import ChatMessage, ToolUseBlock


async def main() -> None:
    agent = Agent(
        # model="claude-3-7-sonnet-20250219",
        model='google/gemini-2.5-pro-preview',
        prompt_template_yaml="prompts/evaluator.yaml",
        tools=["terminal.*"],
    )
    thread = await agent.run("")


    while True:
        tool = input("Select tool (jupyter/terminal_execute, or 'exit' to quit)> ").strip()
        if tool == 'exit':
            break
        if tool not in ('jupyter', 'terminal_execute', 'j', 't'):
            print("Invalid tool. Please choose 'jupyter' or 'terminal_execute'.")
            continue
        if tool == 'jupyter' or tool == 'j':
            code = input("Enter Python code to run in Jupyter (or 'exit' to quit)> ")
            if code.strip() == 'exit':
                break
            output = await thread.tool_call('jupyter', {'code': code})
        else:
            command = input("Enter shell command to run in terminal (or 'exit' to quit)> ")
            if command.strip() == 'exit':
                break
            output = await thread.tool_call('terminal_execute', {'command': command})
        print("Output:")
        print(output[0].text)

    await thread.cleanup()


if __name__ == "__main__":  # pragma: no cover – manual invocation only
    asyncio.run(main())
