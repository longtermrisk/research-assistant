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