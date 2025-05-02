# Anthropic Python API library

[![PyPI version](https://img.shields.io/pypi/v/anthropic.svg)](https://pypi.org/project/anthropic/)

The Anthropic Python library provides convenient access to the Anthropic REST API from any Python 3.8+
application. It includes type definitions for all request params and response fields,
and offers both synchronous and asynchronous clients powered by [httpx](https://github.com/encode/httpx).

## Documentation

The REST API documentation can be found on [docs.anthropic.com](https://docs.anthropic.com/claude/reference/). The full API of this library can be found in [api.md](api.md).

## Installation

```sh
# install from PyPI
pip install anthropic
```

## Usage

The full API of this library can be found in [api.md](api.md).

```python
import os
from anthropic import Anthropic

client = Anthropic(
    api_key=os.environ.get("ANTHROPIC_API_KEY"),  # This is the default and can be omitted
)

message = client.messages.create(
    max_tokens=1024,
    messages=[
        {
            "role": "user",
            "content": "Hello, Claude",
        }
    ],
    model="claude-3-5-sonnet-latest",
)
print(message.content)
```

While you can provide an `api_key` keyword argument,
we recommend using [python-dotenv](https://pypi.org/project/python-dotenv/)
to add `ANTHROPIC_API_KEY="my-anthropic-api-key"` to your `.env` file
so that your API Key is not stored in source control.

## Async usage

Simply import `AsyncAnthropic` instead of `Anthropic` and use `await` with each API call:

```python
import os
import asyncio
from anthropic import AsyncAnthropic

client = AsyncAnthropic(
    api_key=os.environ.get("ANTHROPIC_API_KEY"),  # This is the default and can be omitted
)


async def main() -> None:
    message = await client.messages.create(
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": "Hello, Claude",
            }
        ],
        model="claude-3-5-sonnet-latest",
    )
    print(message.content)


asyncio.run(main())
```

Functionality between the synchronous and asynchronous clients is otherwise identical.

# Tool use example
```python
from __future__ import annotations

from anthropic import Anthropic
from anthropic.types import ToolParam, MessageParam

client = Anthropic()

user_message: MessageParam = {
    "role": "user",
    "content": "What is the weather in SF?",
}
tools: list[ToolParam] = [
    {
        "name": "get_weather",
        "description": "Get the weather for a specific location",
        "input_schema": {
            "type": "object",
            "properties": {"location": {"type": "string"}},
        },
    }
]

message = client.messages.create(
    model="claude-3-5-sonnet-latest",
    max_tokens=1024,
    messages=[user_message],
    tools=tools,
)
print(f"Initial response: {message.model_dump_json(indent=2)}")

assert message.stop_reason == "tool_use"

tool = next(c for c in message.content if c.type == "tool_use")
response = client.messages.create(
    model="claude-3-5-sonnet-latest",
    max_tokens=1024,
    messages=[
        user_message,
        {"role": message.role, "content": message.content},
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": tool.id,
                    "content": [{"type": "text", "text": "The weather is 73f"}],
                }
            ],
        },
    ],
    tools=tools,
)
print(f"\nFinal response: {response.model_dump_json(indent=2)}")
```