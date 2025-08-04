from __future__ import annotations

import asyncio

from automator.agent import Agent
from automator.workspace import Workspace


async def main() -> None:
    workspace = Workspace('demo')
    websearch = Agent(
        model='gemini-2.5-pro',
        prompt_template_yaml="prompts/deep_research.yaml",
        tools=["web.*"],
        id="websearch",
        workspace=workspace,
    )
    aca = Agent(
        model='gemini-2.5-pro',
        prompt_template_yaml="prompts/aca.yaml",
        tools=["terminal.jupyter",],
        id="aca",
        subagents=['aca', 'websearch'],
        workspace=workspace,
    )


if __name__ == "__main__":  # pragma: no cover – manual invocation only
    asyncio.run(main())
