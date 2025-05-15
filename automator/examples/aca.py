from __future__ import annotations

import asyncio

from automator.agent import Agent
from automator.workspace import Workspace


async def main() -> None:
    workspace = Workspace('demo')
    aca = Agent(
        model='google/gemini-2.5-pro-preview',
        prompt_template_yaml="prompts/aca.yaml",
        tools=["terminal.jupyter",],
        id="aca",
        subagents=['aca'],
        workspace=workspace,
    )
    


if __name__ == "__main__":  # pragma: no cover – manual invocation only
    asyncio.run(main())
