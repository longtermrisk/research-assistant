from __future__ import annotations

import asyncio

from automator.agent import Agent
from automator.workspace import Workspace
from automator.dtypes import ChatMessage, ToolUseBlock




async def main() -> None:
    workspace = Workspace('research-assistant', env={
        'CWD': '../research-assistant'
    })
    explorer = Agent(
        model='google/gemini-2.5-pro-preview',
        prompt_template_yaml="prompts/talk_to_model.yaml",
        tools=["talk2model.send_message"],
        as_tool={
            'description': "This subagent can talk to models and quickly explore them. Use this agent to do very fast vibe checks and get a feel for a model. Systematic evaluations can be done based on the results of this agent.",
            'name': 'model_explorer',
        },
        workspace=workspace,
        id="model_explorer",
    )
    visevalpy = Agent(
        model='google/gemini-2.5-pro-preview',
        prompt_template_yaml="prompts/viseval_python.yaml",
        tools=["terminal.*"],
        as_tool={
            'description': "A subagent with terminal and jupyter access that is expert in using viseval - a python library for running LLM-judged evaluations of other LLMs. This subagent can evaluate one or many models, and will generate a csv file containing questions, answers, and scores for metrics of interest. It works best if you explain what this project is about and what it should look for.",
            'name': 'systematic_evaluator',
        },
        workspace=workspace,
        id='visevalpy',
    )
    agent = Agent(
        model='google/gemini-2.5-pro-preview',
        prompt_template_yaml="prompts/subagent_manager.yaml",
        tools=["terminal.*"],
        subagents=['model_explorer', 'visevalpy'],
    )

    agent = workspace.add_agent(agent=agent, id="evaluator_v2")
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
