from __future__ import annotations

import asyncio

from automator.agent import Agent
from automator.workspace import Workspace
from automator.dtypes import ImageBlock, TextBlock

import base64


async def main() -> None:
    agent = Agent(
        model='google/gemini-2.5-pro-preview',
        prompt_template_yaml="prompts/chatgpt.yaml",
    )
    image_path = '/Users/nielswarncke/Desktop/LWScreenShot 2025-02-11 at 10.58.42.png'

    with open(image_path, 'rb') as file:
        data = file.read()
    base64_data = base64.b64encode(data).decode("utf-8")

    task = [
        ImageBlock.from_base64(data=base64_data, media_type='image/png'),
        TextBlock(text="What is the content of this image?"),
    ]

    thread = await agent.run(initial_user_content=task)
    async for message in thread:
        if message.role == 'assistant':
            print(message)
        
    await thread.cleanup()


if __name__ == "__main__":  # pragma: no cover – manual invocation only
    asyncio.run(main())
