# Talk-to-Model MCP

An MCP server that allows your agent to talk to different language models, for example in order to evaluate them.

The server provides a single tool called `send_message` that can be used to talk to different models:

```python
send_message(
    model: str,  # Model to use (e.g., 'gpt-4o', 'claude-3-opus', 'gemini-pro')
                 # Prefixes like 'openai/', 'anthropic/', or 'google/' can be used
    message: str,  # Message to send to the model
    history: Optional[List[Dict[str, str]]] = None,  # Optional conversation history
    thread: Optional[str] = None,  # Optional thread ID for continuing a conversation
    n_responses: int = 1,  # Number of responses to sample,
    temperature: float = 0.7 # Temperature, optional
    max_tokens: int = 8000 # Also optional
)
```

The tool returns a JSON string containing the model's response and a thread ID for follow-up messages.
