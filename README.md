# Research-Assistant

This monorepo contains an implementation of LLM agents that help with research. It consists of the following parts:
- [automator](automator) implements the agent as a python sdk built on top of MCP clients 
- a number of MCP servers implement tools that can be used by our agent or in any other MCP client:
    - [terminal-mcp](terminal-mcp) implements tools for an interactive terminal capable of running background tasks, a jupyter notebook tool, and tools to interact with a local codebase
    - [talk-to-model](talk-to-model) contains a `send_message` tool for agents that help evaluate other LLMs
    - [web-mcp](web-mcp) contains a google search and a markdown broswer tool, but it's currently not working so well: often, websites are turned into markdown documents that exceed the token limits
- [squiggpy](squiggpy) contains a squiggle-like python library and a prompt for agents to use the library

## Setup
Run `python install.py` to setup everything up

## Testing

The automator component includes pytest-based tests to verify core functionality:

### Running Tests

```bash
# Navigate to automator directory
cd automator

# Install with dev dependencies
pip install -e .[dev]

# Run all tests
pytest

# Run specific test files
pytest tests/test_quickstart.py
pytest tests/test_rag_hook.py

# Run tests with verbose output
pytest -v

# Run tests excluding RAG functionality (if dependencies not available)
pytest -m "not rag"
```

### Test Coverage

- **test_quickstart.py**: Tests basic agent creation and terminal tool functionality (based on `examples/quickstart.py`)
- **test_rag_hook.py**: Tests RAG (Retrieval-Augmented Generation) hook functionality (based on `examples/rag/rag_hook_example.py`)
- **test_system.py**: Tests core system functionality including imports, RAG operations, and workspace setup

### Test Requirements

- Basic tests require only core dependencies
- RAG tests require the optional `[rag]` dependencies: `pip install -e .[rag]`
- All tests use temporary directories to avoid affecting your actual workspace