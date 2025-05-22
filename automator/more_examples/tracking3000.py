from __future__ import annotations

import asyncio

from automator.agent import Agent
from automator.workspace import Workspace
from automator.dtypes import ChatMessage, ToolUseBlock, SubagentToolDefinition


# model = 'google/gemini-2.5-pro-preview'
model = "claude-3-7-sonnet-20250219"

async def main() -> None:
    workspace = Workspace('telegram-agent', env={"CWD": "/Users/nielswarncke/Documents/researchoor/automator/telegram_adapter"})
    websearch = Agent(
        model=model,
        prompt_template_yaml="prompts/deep_research.yaml",
        tools=["web.*"],
        id='websearch',
        workspace=workspace,
    )

    mcp_docs = Agent(
        model='google/gemini-2.5-pro-preview',
        prompt_template_yaml="prompts/mcp.yaml",
        tools=["web.*", "terminal.*"],
        id='ask_mcp_docs',
        workspace=workspace,
    )

    manager = Agent(
        model='o3',
        prompt_template_yaml="prompts/swe.yaml",
        tools=["terminal.*"],
        subagents=["websearch", "ask_mcp_docs"],
        id='o3-terminal',
        workspace=workspace,
    )



if __name__ == "__main__":  # pragma: no cover – manual invocation only
    asyncio.run(main())
