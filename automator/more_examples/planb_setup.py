from __future__ import annotations

import asyncio

from automator.agent import Agent
from automator.workspace import Workspace
from automator.dtypes import ChatMessage, ToolUseBlock, SubagentToolDefinition


# model = 'google/gemini-2.5-pro-preview'
model = "claude-3-7-sonnet-20250219"

async def main() -> None:
    workspace = Workspace('planb')
    explorer = Agent(
        model=model,
        prompt_template_yaml="prompts/dataset_explorer.yaml",
        tools=["terminal.*"],
        as_tool=SubagentToolDefinition(
            description="Subagent that specializes in dataset exploration and analysis, and use of LLM-judges.",
            name='dataset_explorer'
        ),
        workspace=workspace,
    )

    generator = Agent(
        model=model,
        prompt_template_yaml="prompts/dataset_creator.yaml",
        tools=["terminal.*"],
        as_tool=SubagentToolDefinition(
            description="Subagent that specializes in dataset generation - use this whenever you need to generate new data.",
            name='generate_dataset'
        ),
        workspace=workspace,
    )

    manager = Agent(
        model=model,
        prompt_template_yaml="prompts/subagent_manager.yaml",
        tools=["terminal.*"],
        subagents=["dataset_explorer", "generate_dataset"],
        workspace=workspace,
    )



if __name__ == "__main__":  # pragma: no cover – manual invocation only
    asyncio.run(main())
