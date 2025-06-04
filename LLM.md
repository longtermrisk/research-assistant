# CLAUDE.md

This file provides guidance to LLMs when working with code in this repository.

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

# Frontend development (in automator/frontend/)
npm run dev          # Start development server (localhost:5173)
npm run build        # Build for production
npm run lint         # Run ESLint

# Backend development (in automator/)
uvicorn automator.api.main:app --port 8000  # Start FastAPI server
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

### Agent Prompt Templates (automator/prompts/)
- `chatgpt.yaml`: Basic conversational assistant
- `swe.yaml`: Software engineering focused agent
- `deep_research.yaml`: Research-oriented agent
- `evaluator.yaml`: Model evaluation and comparison

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