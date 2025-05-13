from typing import List, Callable
import os
import anthropic
import openai
import json
import backoff
from typing import Dict, Any
from automator.dtypes import (
    ChatMessage,
    MessageRole,
    anthropic_format,
    openai_format,
    ToolUseBlock,
    ToolResultBlock,
    TextBlock,
    ImageBlock
)

from dotenv import load_dotenv
load_dotenv()


class Provider:
    def __init__(self, get_response: Callable, models: List[str]):
        self.models = models
        self.get_response = get_response


anthr = anthropic.AsyncAnthropic()
async def get_response_anthropic(messages, tools, **kwargs):
    kwargs = anthropic_format(messages, tools, **kwargs)
    resp = await anthr.messages.create(**kwargs)
    blocks = []
    for item in resp.content:
        if item.type == "text":
            blocks.append(TextBlock(text=item.text))
        elif item.type == "tool_use":
            blocks.append(ToolUseBlock(id=item.id, name=item.name, input=item.input))
        else:
            raise ValueError(f"Unknown block type: {item.type}")

    return ChatMessage(role=MessageRole.assistant, content=blocks)

# ---------------------------------------------------------------------------
# OpenAI provider
# ---------------------------------------------------------------------------


def get_response_factory(oai: openai.AsyncOpenAI):
    async def get_response_openai(messages, tools, **kwargs):
        """Return a ``ChatMessage`` from OpenAI's Chat Completions endpoint.

        The *complete* message history is forwarded to the API.  The assistant's
        reply is converted back into the shared internal representation so that
        the rest of the codebase can stay provider-agnostic.
        """

        # Translate our internal representation into the OpenAI payload.
        kwargs = openai_format(messages, tools, **kwargs)

        # The model name is required by the SDK.  It is included in ``kwargs`` as
        # passed in by the caller.  We raise a clear exception if it is missing –
        # this indicates a programming error further up the stack.
        if "model" not in kwargs:
            raise ValueError("'model' is required for OpenAI completions")

        resp = await oai.chat.completions.create(**kwargs)

        # We do **not** request streaming responses because the surrounding code
        # expects the assistant's answer to be available in one go.  The first –
        # and only – choice therefore contains the message we need.
        message = resp.choices[0].message

        blocks = []

        # Text
        if message.content:
            if isinstance(message.content, str):
                if message.content.strip():
                    blocks.append(TextBlock(text=message.content))
            else:
                # Multi-modal response (text + vision)
                for part in message.content:
                    if part["type"] == "text":
                        blocks.append(TextBlock(text=part["text"]))
                    elif part["type"] == "image_url":
                        url: str = part["image_url"]["url"]
                        if not url.startswith("data:"):
                            # Remote URLs are currently not supported – we simply
                            # ignore them so that the rest of the message can be
                            # processed.
                            continue

                        # The format is: data:<mime>;base64,<data>
                        header, b64_data = url.split(",", 1)
                        media_type = header[len("data:") : header.index(";")]
                        blocks.append(
                            ImageBlock.from_base64(
                                data=b64_data,
                                media_type=media_type,
                            )
                        )

        # Tool calls
        if message.tool_calls:
            for call in message.tool_calls:
                arguments: Dict[str, Any]
                try:
                    arguments = json.loads(call.function.arguments)
                except Exception:
                    # The arguments string is not valid JSON – fall back to the
                    # raw string to avoid data loss.
                    arguments = {"_raw_arguments": call.function.arguments}

                blocks.append(
                    ToolUseBlock(
                        id=call.id,
                        name=call.function.name,
                        input=arguments,
                    )
                )

        return ChatMessage(role=MessageRole.assistant, content=blocks)
    return get_response_openai


_available_anthropic_models: List[str]

# Fetching the model list requires internet connectivity.  In offline
# environments the call raises an ``APIConnectionError``.  Falling back to
# an *empty* list keeps imports functional without penalising
# functionality when the network is available.

try:
    _available_anthropic_models = [m.id for m in anthropic.Anthropic().models.list(limit=1000).data]
except Exception:  # broad – we really do not care, we only need a list
    _available_anthropic_models = []


providers = [
    Provider(
        get_response_anthropic,
        models=_available_anthropic_models,
    )
]


# ---------------------------------------------------------------------------
# Register OpenRouter models
# ---------------------------------------------------------------------------

if 'OPENROUTER_API_KEY' in os.environ:
    providers.append(
        Provider(
            get_response_factory(openai.AsyncOpenAI(api_key=os.environ['OPENROUTER_API_KEY'], base_url="https://openrouter.ai/api/v1")),
            models=[
                'google/gemini-2.5-pro-preview',
            ],
        )
    )

# ---------------------------------------------------------------------------
# Register OpenAI models
# ---------------------------------------------------------------------------


if 'OPENAI_API_KEY' in os.environ:
    _available_openai_models = [m.id for m in openai.OpenAI().models.list().data]
    providers.append(
        Provider(
            get_response_factory(openai.AsyncOpenAI()),
            models=_available_openai_models,
        )
    )


# @backoff.on_exception(
#     wait_gen=backoff.expo,
#     exception=(
#         openai.RateLimitError,
#         openai.APIConnectionError,
#         openai.APITimeoutError,
#         openai.InternalServerError,
#         anthropic.APIConnectionError,
#         anthropic.RateLimitError,
#         anthropic.APIStatusError,
#     ),
#     max_value=60,
#     factor=1.5,
# )
async def get_response(model, messages, **kwargs):
    all_models = []
    for provider in providers:
        if model in provider.models:
            all_models.append(model)
            return await provider.get_response(model=model, messages=messages, **kwargs)
    raise ValueError(f"Model '{model}' not supported by any provider. Supported models: {all_models}")
        
    