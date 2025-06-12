import os
import json
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Union
from uuid import uuid4
from pydantic import BaseModel, Field, field_validator
import yaml

from automator.utils import dict_recursive


class ToolDefinition(BaseModel):
    """Class for structured JSON response format that can be used as a tool for judges

    Arguments:
        definition: JSON schema for the structured response format in anthropic format
    """

    name: str
    description: str
    input_schema: Dict[str, Any]

    @property
    def anthropic_format(self):
        """Return the definition in Anthropic format"""
        return self.model_dump()

    @property
    def openai_format(self):
        """Convert the definition to OpenAI format"""
        oai_definition = {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.input_schema,
            },
        }
        return oai_definition

class SubagentToolDefinition(ToolDefinition):
    input_schema: Dict[str, Any] = Field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The message to the agent. "
                },
            }
        })


class MessageRole(str, Enum):
    user = "user"
    system = "system"
    assistant = "assistant"


class TextBlock(BaseModel):
    text: str
    type: str = "text"
    meta: Optional[Dict[str, Any]] = None

    def anthropic_format(self) -> Dict[str, Any]:
        return {
            "type": "text",
            "text": self.text,
        }


class Base64ImageSource(BaseModel):
    data: str
    media_type: str
    type: str = "base64"


class ImageBlock(BaseModel):
    source: Base64ImageSource
    type: str = "image"
    meta: Optional[Dict[str, Any]] = None

    @staticmethod
    def from_base64(data: str, media_type: str = "image/png", meta=None) -> "ImageBlock":
        return ImageBlock(source=Base64ImageSource(data=data, media_type=media_type, meta=meta))
    
    @staticmethod
    def from_file(file_path: str, media_type: str = "image/png", meta=None) -> "ImageBlock":
        with open(file_path, "rb") as f:
            data = f.read()
        base64_data = base64.b64encode(data).decode("utf-8")
        return ImageBlock(source=Base64ImageSource(data=base64_data, media_type=media_type, meta=meta))
        
    def anthropic_format(self) -> Dict[str, Any]:
        return {
            "type": "image",
            "source": self.source.model_dump()
        }


class ToolUseBlock(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    input: Optional[Dict[str, Any]] = None
    name: str
    type: str = "tool_use"
    meta: Optional[Dict[str, Any]] = None

    def anthropic_format(self) -> Dict[str, Any]:
        return {
            "type": "tool_use",
            "id": self.id,
            "name": self.name,
            "input": self.input,
        }


class ToolResultBlock(BaseModel):
    tool_use_id: str
    content: List[Union[TextBlock, ImageBlock]]
    type: str = "tool_result"
    meta: Optional[Dict[str, Any]] = None
    
    def anthropic_format(self) -> Dict[str, Any]:
        return {
            "type": "tool_result",
            "tool_use_id": self.tool_use_id,
            "content": [block.anthropic_format() for block in self.content],
        }


ContentBlock = Union[TextBlock, ImageBlock, ToolUseBlock, ToolResultBlock]


class ChatMessage(BaseModel):
    role: MessageRole
    content: List[ContentBlock]
    meta: Optional[Dict[str, Any]] = None

    @field_validator("role", mode="before")
    @classmethod
    def convert_role_to_enum(cls, v):
        if isinstance(v, str):
            try:
                return MessageRole[v]
            except KeyError:
                raise ValueError(f"Invalid role: {v}")
        return v

    def anthropic_format(self):  # -> dict[str, Any]
        if isinstance(self.content, str):
            return {"role": self.role.value, "content": self.content}

        content = [c.anthropic_format() for c in self.content]
        return {"role": self.role.value, "content": content}


class PromptTemplate(BaseModel):
    messages: Sequence[ChatMessage]

    @staticmethod
    def from_yaml(file: str) -> "PromptTemplate":
        if not os.path.exists(file) and os.path.exists(os.path.join(os.path.expanduser('~/.automator/prompts'), file)):
            file = os.path.join(os.path.expanduser('~/.automator/prompts'), file)
        if not os.path.exists(file):
            raise FileNotFoundError(f"Prompt template file not found: {file}")
        with open(file) as f:
            messages = yaml.load(f, Loader=yaml.FullLoader)["messages"]
        for message in messages:
            if isinstance(message["content"], str):
                message["content"] = [TextBlock(text=message["content"])]
        return PromptTemplate(messages=messages)

    def apply(self, params: Dict[str, Any]) -> List[ChatMessage]:
        @dict_recursive()
        def apply_params(template, params):
            if template is None:
                return None
            for key, value in params.items():
                if value is None:
                    continue
                template = template.replace(f"${key}", value)
            return template

        # ChatMessage -> json -> apply_params -> ChatMessage
        messages = [msg.model_dump() for msg in self.messages]
        messages = apply_params(messages, params)
        return [ChatMessage(**msg) for msg in messages]


def anthropic_format(messages, tools, **kwargs) -> Dict[str, Any]:
    system_message = None
    chat_messages = []

    if len(messages) > 0 and messages[0].role == MessageRole.system:
        system_message = "\n".join([block.text for block in messages[0].content])
        chat_messages = [msg.anthropic_format() for msg in messages[1:]]
    else:
        chat_messages = [msg.anthropic_format() for msg in messages]

    kwargs['messages'] = chat_messages
    if tools:
        kwargs["tools"] = [tool.definition.anthropic_format for tool in tools]
    if system_message:
        kwargs["system"] = system_message
    return kwargs


# ---------------------------------------------------------------------------
# OpenAI formatting helpers
# ---------------------------------------------------------------------------


def _text_and_image_to_openai_parts(blocks: List[ContentBlock]):
    """Convert a list of TextBlock/ImageBlock objects into the OpenAI multimodal
    content format.

    For text-only messages the caller should unwrap the single string to keep
    the payload small; returning a list is only required when the message is
    truly multi-modal (i.e. contains images).
    """

    oai_parts: List[Dict[str, Any]] = []
    for blk in blocks:
        if isinstance(blk, TextBlock):
            oai_parts.append({"type": "text", "text": blk.text})
        elif isinstance(blk, ImageBlock):
            # OpenAI expects the image to be provided through the ``image_url``
            # field.  When the binary is already base64-encoded we can simply
            # prefix it with the media-type data-URL scheme.
            data_url = f"data:{blk.source.media_type};base64,{blk.source.data}"
            oai_parts.append({"type": "image_url", "image_url": {"url": data_url}})
        else:  # pragma: no cover – we only reference known block types
            raise ValueError(f"Unsupported content block for OpenAI format: {blk}")

    # If the message only contains a single text block we can – and should –
    # return the plain string to match the common usage pattern of the Chat
    # Completions API.
    if len(oai_parts) == 1 and oai_parts[0]["type"] == "text":
        return oai_parts[0]["text"]  # type: ignore[return-value]
    return oai_parts


def openai_format(messages, tools, **kwargs) -> Dict[str, Any]:
    """Convert our internal chat representation into a payload suitable for
    ``/chat/completions``.

    The entire *message history* is always included – callers are responsible
    for trimming if required.
    """

    oai_messages: List[Dict[str, Any]] = []

    for msg in messages:
        # Tool results are wrapped inside a *user* message in the internal
        # representation.  For OpenAI we need to expand each tool result into
        # its own ``role == tool`` message so that the assistant can pick them
        # up in the correct format.
        non_tool_results = []
        for c in msg.content:  # type: ignore[assignment]
            if isinstance(c, ToolResultBlock):
                # Concatenate all text parts inside the tool result.  The
                # OpenAI spec expects a *single* string.
                text_parts = [b.text for b in c.content if isinstance(b, TextBlock)]
                content_str = "\n".join(text_parts) if text_parts else ""
                oai_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": c.tool_use_id,
                        "content": content_str,
                    }
                )
                # Move all image parts into a new user messages
                # (OpenAI does not support image URLs in tool result messages).
                image_parts = [
                    b for b in c.content if isinstance(b, ImageBlock)
                ]
                if image_parts:
                    oai_messages.append(
                        {
                            "role": "user",
                            "content": _text_and_image_to_openai_parts(image_parts),
                        }
                    )
            else:
                non_tool_results.append(c)
        
        # Normal user / assistant / system messages
        role = msg.role.value

        # Assistant messages may include tool calls.  Those need to go into
        # the ``tool_calls`` field.
        tool_calls = []
        text_and_images: List[ContentBlock] = []

        for blk in non_tool_results:
            if isinstance(blk, ToolUseBlock):
                tool_calls.append(
                    {
                        "id": blk.id,
                        "type": "function",
                        "function": {
                            "name": blk.name,
                            "arguments": json.dumps(blk.input or {}),
                        },
                    }
                )
            else:
                text_and_images.append(blk)

        oai_msg: Dict[str, Any] = {"role": role}

        if text_and_images:
            oai_msg["content"] = _text_and_image_to_openai_parts(text_and_images)
        else:
            # When the assistant triggers only a tool call the content must be
            # explicitly set to *None*.
            oai_msg["content"] = None

        if tool_calls:
            oai_msg["tool_calls"] = tool_calls

        if oai_msg["content"] is None and not tool_calls:
            # If the message has no content and no tool calls, we skip it
            continue
        oai_messages.append(oai_msg)

    kwargs["messages"] = oai_messages

    if tools:
        kwargs["tools"] = [t.definition.openai_format for t in tools]
    
    # Special treatments for reasoning models
    if kwargs.get("model", "").startswith("o"):
        if 'max_tokens' in kwargs:
            kwargs['max_completion_tokens'] = kwargs.pop('max_tokens')
        kwargs.pop('temperature', None)

    return kwargs
