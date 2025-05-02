"""
Talk-to-Model MCP - An MCP server that allows your agent to talk to different models.

This MCP server provides a single tool to talk to different language models from providers
like OpenAI, Anthropic, and Google.
"""

import os
import json
import uuid
import asyncio
from typing import Dict, List, Optional, Any, Union

from mcp.server.fastmcp import FastMCP, Context
import openai
import yaml

# Create an MCP server
mcp = FastMCP("Talk-to-Model")

_available_openai_models = [m.id for m in openai.OpenAI().models.list().data]
oai = openai.AsyncOpenAI()

# Dictionary to store message threads/history
threads = {}



async def call_openai_model(
    model: str, 
    messages: List[Dict[str, str]],
    temperature: float,
    max_tokens: int,
    ctx: Context
) -> str:
    """Call an OpenAI model with the given messages."""
    try:
        response = await oai.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        
        return response.choices[0].message.content
    except Exception as e:
        await ctx.error(f"Error calling OpenAI API: {str(e)}")
        raise



@mcp.tool()
async def send_message(
    model: str,
    message: str,
    history: Optional[List[Dict[str, str]]] = None,
    thread: Optional[str] = None,
    n_responses: int = 1,
    temperature: float = 0.7,
    max_tokens: int = 8000,
    ctx: Context = None,
) -> str:
    """
    Send a message to a model and get a response.
    
    Args:
        model: The model to use (e.g., 'gpt-4o')
        message: The message to send to the model
        history: Optional list of previous messages in format [{"role": "...", "content": "..."}]
        thread: Optional thread ID for continuing a conversation without passing full history
        n_responses: Number of responses to sample (will create new threads if > 1)
        temperature: Sampling temperature for the model
        max_tokens: Maximum number of tokens for the model response
    Returns:
        JSON string containing the model's response and a thread ID for follow-up messages
    """
    await ctx.info(f"Sending message to model: {model}")
    
    if thread and thread in threads:
        # Using an existing thread/conversation
        messages = threads[thread].copy()
        messages.append({"role": "user", "content": message})
    elif history:
        # Using provided history
        messages = history.copy()
        messages.append({"role": "user", "content": message})
    else:
        # Starting a new conversation
        messages = [{"role": "user", "content": message}]
    
    # Generate multiple responses if requested
    thread_ids = []
    responses = []
    
    async def get_response_and_update_thread():
        try:
            response_text = await call_openai_model(model, messages, temperature, max_tokens, ctx)
            # Create a new thread or update existing one
            new_thread_id = thread or str(uuid.uuid4())
            new_messages = messages.copy()
            new_messages.append({"role": "assistant", "content": response_text})
            threads[new_thread_id] = new_messages
            await ctx.info(f"Got response, thread_id: {new_thread_id}")
            return response_text, new_thread_id
        except Exception as e:
            await ctx.error(f"Error calling model API: {str(e)}")
            return f"Error: {str(e)}", None

    # Run all calls in parallel
    tasks = [get_response_and_update_thread() for _ in range(n_responses)]
    results = await asyncio.gather(*tasks)

    responses = []
    thread_ids = []
    for resp, tid in results:
        if tid is None:
            # If any call failed, return the error immediately
            return resp
        responses.append(resp)
        thread_ids.append(tid)

    # Prepare the result
    import yaml

    if n_responses == 1:
        return yaml.safe_dump({
            "response": responses[0],
            "thread_id": thread_ids[0]
        })
    else:
        result = []
        for i, (resp, tid) in enumerate(zip(responses, thread_ids)):
            result.append({
                "response": resp,
                "thread_id": tid
            })
        return yaml.safe_dump(result)


if __name__ == "__main__":
    with open('/Users/nielswarncke/Documents/researchoor/talk-to-model/models', 'w') as f:
        f.write('\n'.join(_available_openai_models))
    mcp.run()