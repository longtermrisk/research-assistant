from typing import List, Callable, Type, Union
import os
import anthropic
import openai
import json
import backoff
from typing import Dict, Any
from uuid import uuid4
from pydantic import BaseModel
from automator.dtypes import (
    ChatMessage,
    MessageRole,
    anthropic_format,
    openai_format,
    genai_format,
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
async def get_response_anthropic(messages, tools, response_format=None, **kwargs):
    if response_format is not None:
        raise NotImplementedError("Structured output is not supported for Anthropic models")
    
    kwargs = anthropic_format(messages, tools, **kwargs)
    kwargs['timeout'] = 599  # Set a timeout of 10 minutes for the request
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
    async def get_response_openai(messages, tools, response_format=None, **kwargs):
        """Return a ``ChatMessage`` from OpenAI's Chat Completions endpoint.

        The *complete* message history is forwarded to the API.  The assistant's
        reply is converted back into the shared internal representation so that
        the rest of the codebase can stay provider-agnostic.
        """

        kwargs = openai_format(messages, tools, **kwargs)

        if "model" not in kwargs:
            raise ValueError("'model' is required for OpenAI completions")

        # Handle structured output
        if response_format is not None:
            if isinstance(response_format, type) and issubclass(response_format, BaseModel):
                # Use the parse method for Pydantic models
                resp = await oai.chat.completions.parse(response_format=response_format, **kwargs)
                
                blocks = []
                if resp.choices[0].message.content:
                    blocks.append(TextBlock(text=resp.choices[0].message.content))
                
                # Create response and add parsed attribute
                response = ChatMessage(role=MessageRole.assistant, content=blocks)
                response.parsed = resp.choices[0].message.parsed
                return response
            else:
                # Handle other response formats
                kwargs["response_format"] = response_format
                resp = await oai.chat.completions.create(**kwargs)
        else:
            resp = await oai.chat.completions.create(**kwargs)
        
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
                arguments: Dict[str, Any] = json.loads(call.function.arguments)

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
                'qwen/qwen3-235b-a22b'
            ],
        )
    )

# ---------------------------------------------------------------------------
# Register OpenAI models
# ---------------------------------------------------------------------------


if 'OPENAI_API_KEY' in os.environ:
    _available_openai_models = [m.id for m in openai.OpenAI().models.list().data if m.id.startswith('gpt') or m.id.startswith('o')]
    providers.append(
        Provider(
            get_response_factory(openai.AsyncOpenAI()),
            models=_available_openai_models,
        )
    )


# ---------------------------------------------------------------------------
# Register Google GenAI models
# ---------------------------------------------------------------------------

async def get_response_genai(messages, tools, response_format=None, **kwargs):
    """Return a ChatMessage from Google GenAI generate_content endpoint."""
    try:
        from google import genai
        from google.genai import types as genai_types
    except ImportError:
        raise ImportError("google-genai package is required for Google GenAI support. Install with: pip install google-genai")
    
    # Create client
    api_key = os.environ.get('GEMINI_API_KEY') or os.environ.get('GOOGLE_API_KEY')
    if not api_key:
        raise ValueError("GEMINI_API_KEY or GOOGLE_API_KEY environment variable is required")
    
    client = genai.Client(api_key=api_key)
    
    # Format the request  
    request_kwargs = genai_format(messages, tools)
    
    # Filter out parameters that don't belong in GenerateContentConfig
    config_params = {}
    for k, v in kwargs.items():
        if k not in ['contents', 'tools', 'system_instruction', 'model']:
            # Map common parameters
            if k == 'max_tokens':
                config_params['max_output_tokens'] = v
            elif k == 'temperature':
                config_params['temperature'] = v
            elif k == 'top_p':
                config_params['top_p'] = v
            # Add other known parameters as needed
    
    # Handle structured output
    if response_format is not None:
        if isinstance(response_format, type) and issubclass(response_format, BaseModel):
            config_params.update({
                'response_mime_type': 'application/json',
                'response_schema': response_format,
            })
    
    # Add tools to config if present
    if 'tools' in request_kwargs and request_kwargs['tools']:
        config_params['tools'] = request_kwargs['tools']
    
    # Add system instruction to config if present  
    if 'system_instruction' in request_kwargs and request_kwargs['system_instruction']:
        config_params['system_instruction'] = request_kwargs['system_instruction']
    
    config = genai_types.GenerateContentConfig(**config_params) if config_params else None
    
    # Make the request
    response = await client.aio.models.generate_content(
        model=kwargs.get('model', 'gemini-2.0-flash-001'),
        contents=request_kwargs['contents'],
        config=config
    )
    
    # Convert response back to our format
    blocks = []
    
    if response.text:
        blocks.append(TextBlock(text=response.text))
    
    # Handle function calls if present
    if hasattr(response, 'function_calls') and response.function_calls:
        for func_call in response.function_calls:
            # Generate ID if not present
            func_id = getattr(func_call, 'id', None) or str(uuid4())
            # Get function call args
            if hasattr(func_call, 'function_call') and func_call.function_call:
                func_args = func_call.function_call.args or {}
            else:
                func_args = getattr(func_call, 'args', {})
            
            blocks.append(ToolUseBlock(
                id=func_id,
                name=func_call.name,
                input=func_args
            ))
    
    # Create response object
    chat_response = ChatMessage(role=MessageRole.assistant, content=blocks)
    
    # Add parsed result if structured output was used
    if response_format is not None and isinstance(response_format, type) and issubclass(response_format, BaseModel):
        if hasattr(response, 'parsed') and response.parsed:
            chat_response.parsed = response.parsed
        elif response.text:
            # Try to parse JSON response manually
            try:
                import json
                parsed_data = json.loads(response.text)
                chat_response.parsed = response_format(**parsed_data)
            except Exception:
                pass  # If parsing fails, just return the text
    
    return chat_response


if 'GEMINI_API_KEY' in os.environ or 'GOOGLE_API_KEY' in os.environ:
    providers.append(
        Provider(
            get_response_genai,
            models=[
                'gemini-2.5-pro',
                'gemini-2.5-flash'
            ],
        )
    )


@backoff.on_exception(
    wait_gen=backoff.expo,
    exception=(
        openai.RateLimitError,
        openai.APIConnectionError,
        openai.APITimeoutError,
        openai.InternalServerError,
        anthropic.APIConnectionError,
        anthropic.RateLimitError,
        anthropic.APIStatusError,
        TypeError,
        AssertionError
    ),
    max_value=60,
    factor=1.5,
    on_backoff=lambda details: print(f"Retrying... {details['exception']}")
)
async def get_response(model, messages, tools=None, response_format=None, **kwargs):
    all_models = []
    for provider in providers:
        all_models.extend(provider.models)
        if model in provider.models:
            response = await provider.get_response(model=model, messages=messages, tools=tools, response_format=response_format, **kwargs)
            if len(response.content) == 0:
                messages = messages + [
                    ChatMessage(role=MessageRole.user, content=[TextBlock(text="Please continue.")])
                ]
                response = await provider.get_response(model=model, messages=messages, tools=tools, response_format=response_format, **kwargs)
            assert len(response.content) > 0, "Response content is empty"
            return response
    raise ValueError(f"Model '{model}' not supported by any provider. Supported models: {all_models}")
        
    