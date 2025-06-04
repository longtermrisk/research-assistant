# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Automator is an MCP-based LLM agent system that enables creating and orchestrating AI agents with extensible tool access. It combines multiple LLM providers (Anthropic Claude, OpenAI GPT, Google Gemini) with a flexible tool system via the Model Context Protocol (MCP).

## Development Commands

### Setup
```bash
# Install dependencies
uv sync

# Frontend development
cd frontend/
npm install
npm run dev          # Start development server (localhost:5173)
npm run build        # Build for production
npm run lint         # Run ESLint

# Backend API server
uvicorn automator.api.main:app --reload --port 8000
```

### Testing
```bash
# Run tests
pytest tests/

# Run specific test
pytest tests/test_sse_transport.py

# Run with coverage
pytest --cov=automator tests/
```

### Examples
```bash
# Basic agent examples
python examples/quickstart.py
python examples/load_workspace.py
python examples/test_tool_calls.py

# SSE transport example
python examples/sse_transport_example.py
```

## Architecture

### Core Components
- **Agent**: Combines model, prompt template, tools, and environment
- **Thread**: Manages conversation state and tool execution lifecycle  
- **Workspace**: Provides persistence and environment management
- **Provider**: Abstracts LLM provider interfaces (Anthropic, OpenAI, Google)

### Key Files
- `automator/agent.py` - Core agent and thread logic
- `automator/workspace.py` - Workspace management and persistence
- `automator/llm.py` - Multi-provider LLM abstraction
- `automator/dtypes.py` - Data types and message format conversion
- `automator/api/main.py` - FastAPI REST API
- `automator/hooks.py` - Pre/post-processing hooks for agent interactions

### Tool System
Agents access tools via MCP servers using glob patterns:
- `"terminal.*"` - All terminal/codebase interaction tools
- `"web.*"` - Web search and browsing tools  
- `"talk2model.*"` - Agent-to-agent communication tools
- `"jupyter.*"` - Jupyter notebook execution tools

### Agent Configuration
Agents are defined using YAML prompt templates in `prompts/`:
- `chatgpt.yaml` - Basic conversational assistant
- `swe.yaml` - Software engineering focused agent
- `deep_research.yaml` - Research-oriented agent
- `evaluator.yaml` - Model evaluation and comparison

### Frontend Architecture
React-based UI with:
- `App.tsx` - Main routing and layout
- `views/MainView/` - Chat interface with message bubbles and input
- `views/AgentManagementView/` - Agent creation and management
- `contexts/WorkspaceContext.tsx` - Global state management

### Data Flow
1. User creates Agent with model, prompt template, and tools
2. Agent connects to MCP servers for tool access during initialization
3. Thread manages conversation flow and tool execution
4. Workspace handles persistence of agents and threads
5. Frontend communicates via FastAPI REST API with SSE streaming

### Environment & Configuration
- Workspaces auto-create Python virtual environments
- Environment variables merge workspace + agent settings
- MCP servers configured in `~/mcp.json`
- Required API keys: `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `HF_TOKEN`, `SERP_API_KEY`

### Message Format
Internal message format supports:
- Text blocks with markdown rendering
- Tool call/result blocks for MCP tool execution
- Image blocks for multimodal interactions
- Provider-specific serialization for each LLM API