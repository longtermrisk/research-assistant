# CLAUDE.md

This file provides guidance to LLMs when working with code in this repository.

## RAG Integration Available

The `rag/` directory contains the Light-RAG system - a lightweight Retrieval-Augmented Generation system inspired by attention mechanisms that can enhance automator agents with contextual document retrieval.

### Key RAG Features
- **Document ingestion** with OpenAI-powered key generation and summarization
- **Persistent storage** using FileSystemStore with full serialization
- **Intelligent reranking** using OpenAI's structured outputs
- **Hook integration** with automator agents for automatic document retrieval
- **Context-aware retrieval** based on conversation history

### Automator Integration
The RAG system has been successfully merged into the main automator repository as a sibling package:

**Directory Structure:**
```
automator/
├── automator/          # Core Python package
├── rag/                # RAG package (sibling)
├── ui/                 # Frontend code
├── examples/rag/       # RAG examples
├── tests/rag/          # RAG tests
└── docs/               # Documentation including rag.md
```

**Installation Options:**
- `pip install -e .` - Core functionality only
- `pip install -e .[rag]` - Core + RAG capabilities  
- `pip install -e .[ui]` - Core + UI build tools
- `pip install -e .[dev]` - Core + development tools
- `pip install -e .[all]` - Everything

**RAG Hook Usage:**
```python
from automator import Agent
from rag import create_rag_hook  # Only if [rag] installed

agent = Agent(
    model="gpt-4.1",
    hooks=['rag:.knowledge']  # Enable RAG for .knowledge directory
)
```

The RAG system automatically:
1. Ingests documents from specified directories
2. Retrieves relevant documents based on conversation context  
3. Adds document content to agent message history
4. Tracks document relevance across conversation turns

## Repository Architecture

This is a **Research-Assistant** monorepo containing an MCP-based LLM agent system with multiple interconnected components:

### Core Components
- **automator/**: Main Python SDK for creating LLM agents built on MCP (Model Context Protocol)
- **terminal-mcp/**: MCP server providing terminal, Jupyter, and codebase interaction tools
- **web-mcp/**: MCP server for web search and browsing capabilities
- **talk-to-model/**: MCP server enabling agent-to-agent communication for LLM evaluation
- **squiggpy/**: Monte Carlo probabilistic modeling library with agent prompts

### Agent System Flow
1. **MCP Configuration**: `~/mcp.json` defines available MCP servers and their environment variables
2. **Agent Creation**: Agents use YAML prompt templates from `automator/prompts/` and specify tool access patterns
3. **Workspace Management**: Agents and conversation threads persist in `~/.automator/workspaces/`
4. **Tool Access**: Agents connect to MCP servers to access tools (terminal, web search, model communication)

## Development Commands

### Initial Setup
```bash
python install.py  # Sets up entire repository including dependencies and MCP configuration
```

### Component Development
```bash
# Individual component setup (run in component directory)
uv sync

# Frontend development (in automator/ui/frontend/)
npm run dev          # Start development server (localhost:5173)
npm run build        # Build for production
npm run lint         # Run ESLint

# Backend development (in automator/ui)
uvicorn api.main:app --port 8000  # Start FastAPI server
```

### Testing
```bash
# Navigate to automator directory
cd automator

# Install with dev dependencies
pip install -e .[dev]

# Run all tests
pytest

# Run specific test categories
pytest tests/test_quickstart.py    # Basic agent and terminal functionality
pytest tests/test_rag_hook.py      # RAG integration tests (requires [rag] dependencies)
pytest tests/test_system.py        # Core system tests

# Run with verbose output
pytest -v

# Run excluding RAG tests if dependencies not available
pytest -m "not rag"
```

### Agent Usage Examples
```bash
# Basic agent examples
python automator/examples/quickstart.py
python automator/examples/load_workspace.py

# Advanced examples  
python more_examples/coder.py
python more_examples/evaluator.py
```

## Key Configuration Files

### MCP Server Configuration
- `mcp.json`: Template for MCP server definitions (gets copied to `~/mcp.json`)
- Each MCP server runs via: `uv --directory /path/to/server run entrypoint.py`

### Environment Setup
- API keys configured through `~/mcp.json` and propagated to `.env`
- Required: `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `HF_TOKEN`, `SERP_API_KEY`

## Technology Stack

### Backend
- **Python 3.11+** with **UV** package management
- **MCP (Model Context Protocol)** for tool integration
- **FastAPI** + **Uvicorn** for REST API
- **Pydantic** for data validation

### Frontend  
- **React 18** + **TypeScript** + **Vite**
- **React Router** for navigation
- **React Markdown** with **KaTeX** for math rendering

### AI Integration
- **Anthropic Claude**, **OpenAI GPT**, **Google Gemini** models
- **Hugging Face** and **OpenWeights** for additional model access

## Agent Development Patterns

### Creating New Agents
1. Define prompt template in `automator/prompts/agent_name.yaml`
2. Specify required tools using glob patterns (e.g., `"terminal.*"`, `"web.*"`)
3. Set workspace environment variables for agent's working context
4. Use workspace persistence for long-running agent sessions

### Tool Access Patterns
- `"terminal.*"`: All terminal/codebase interaction tools
- `"web.*"`: Web search and browsing tools  
- `"talk2model.*"`: Agent-to-agent communication tools
- Specific tools: `"terminal.execute"`, `"web.search"`, etc.

### Workspace Management
- Workspaces auto-save agent state and conversation threads
- Load existing agents: `workspace.get_agent('agent_name')`
- List available: `workspace.list_agents()`, `workspace.list_threads()`