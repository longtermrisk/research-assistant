# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

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
