# LLM.md

## Project Overview: automator

Automator is an MCP-based LLM agent system that enables creating and orchestrating AI agents with extensible tool access. It combines multiple LLM providers (Anthropic Claude, OpenAI GPT, Google Gemini) with a flexible tool system via the Model Context Protocol (MCP).


## Architecture

### Core Components
- **Agent**: Combines model, prompt template, tools, and environment
- **Thread**: Manages conversation state and tool execution lifecycle  
- **Workspace**: Provides persistence and environment management

### Key Files
- `automator/agent.py` - Core agent and thread logic
- `automator/workspace.py` - Workspace management and persistence
- `automator/llm.py` - Multi-provider LLM abstraction
- `automator/dtypes.py` - Data types and message format conversion
- `automator/hooks.py` - Pre/post-processing hooks for agent interactions

## UI
- `ui/api` - fastapi based backend (start via `cd ui && uvicorn api.main:app --port 8000`)
- `ui/frontend` - vite frontend (start via `cd ui/frontend && npm run dev`)

## Testing

The project includes pytest-based tests covering core functionality:

### Test Files
- `tests/test_quickstart.py` - Tests basic agent creation and terminal tool functionality
- `tests/test_rag_hook.py` - Tests RAG (Retrieval-Augmented Generation) integration
- `tests/test_system.py` - Core system functionality including imports and workspace setup

### Running Tests
```bash
# Install with dev dependencies
pip install -e .[dev]

# Run all tests
pytest

# Run specific test files  
pytest tests/test_quickstart.py
pytest tests/test_rag_hook.py

# Run excluding RAG tests (if optional dependencies not installed)
pytest -m "not rag"
```

### Test Isolation
- All tests use temporary directories to avoid affecting actual workspaces
- RAG tests are automatically skipped if optional `[rag]` dependencies are not installed
- Tests cover both basic agent functionality and advanced features like RAG hooks
