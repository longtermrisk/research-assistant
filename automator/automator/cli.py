#!/usr/bin/env python3
"""
CLI interface for automator package.
"""

import argparse
import os
import sys
from pathlib import Path

from .workspace import Workspace
from .agent import Agent
from .hooks import _HOOKS

def workspace_add(args):
    """Add a new workspace with the current directory as CWD."""
    current_dir = Path.cwd().resolve()
    
    # Use provided name or default to directory name
    workspace_name = args.name if args.name else current_dir.name
    
    # Create workspace with current directory as CWD
    env = {
        'CWD': str(current_dir),
        'ENSURE_VENV': "TRUE" if args.venv else "FALSE"
    }
    try:
        workspace = Workspace(workspace_name, env=env)
        print(f"Created workspace '{workspace_name}' with CWD: {current_dir}")
        print(f"Workspace path: {workspace.root}")
    except Exception as e:
        print(f"Error creating workspace: {e}", file=sys.stderr)
        sys.exit(1)


def workspace_create_expert(args):
    """Clones args.github_url into a new workspace and creates an agent with terminal access in it"""
    target_dir = Workspace._resolve_workspace_dir(args.github_repo)
    # clone the repo into the target_dir
    import subprocess
    
    # Create parent directory if it doesn't exist
    target_dir.parent.mkdir(parents=True, exist_ok=True)

    # Clone the repo
    try:
        subprocess.run(
            ["git", "clone", f"https://github.com/{args.github_repo}.git", str(target_dir)],
            check=True,
            capture_output=True,
            text=True
        )
    except subprocess.CalledProcessError as e:
        print(f"Error cloning repository: {e.stderr}", file=sys.stderr)
        sys.exit(1)
    # Create the workspace
    env = {
        'CWD': str(target_dir),
        'ENSURE_VENV': "TRUE"
    }
    workspace = Workspace('shared')

    # Check if rag is available - if yes, add a hook
    hooks = ['claude.md']
    if 'rag:.' in _HOOKS:
        hooks += ['rag:.']
    else:
        print("RAG not available, continuing without RAG hooks")
    # Create the agent
    expert = Agent(
        id=args.github_repo.split('/')[-1],
        workspace=workspace,
        model='gemini-2.5-pro',
        prompt_template_yaml=f"{args.prompt}.yaml",
        tools=["terminal.get_file", "terminal.list_codebase_files"],
        hooks=hooks,
        env=env
    )


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="automator",
        description="CLI for automator - MCP-based LLM agent system"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Workspace commands
    ws_parser = subparsers.add_parser("ws", help="Workspace management")
    ws_subparsers = ws_parser.add_subparsers(dest="ws_command", help="Workspace commands")
    
    # ws add command
    add_parser = ws_subparsers.add_parser("add", help="Add a new workspace")
    add_parser.add_argument(
        "path", 
        nargs="?", 
        default=".", 
        help="Path for workspace (default: current directory)"
    )
    add_parser.add_argument(
        "--name", 
        help="Workspace name (default: directory name)"
    )
    add_parser.add_argument(  # Add venv argument to match workspace_create_expert
        "--venv",
        action="store_true",
        help="Ensure a virtual environment is created"
    )
    add_parser.set_defaults(func=workspace_add)

    # ws create-expert command
    expert_parser = ws_subparsers.add_parser("create-expert", help="Create expert workspace from GitHub repo")
    expert_parser.add_argument(
        "github_repo",
        help="GitHub repository (`owner/repo`)"
    )
    expert_parser.add_argument(
        "--prompt",
        default="expert",
        help="Prompt template to use (default: expert)"
    )
    expert_parser.add_argument(  # Add venv argument to match workspace_create_expert
        "--venv",
        action="store_true",
        help="Ensure a virtual environment is created"
    )
    expert_parser.set_defaults(func=workspace_create_expert)
    args = parser.parse_args()
    
    if not hasattr(args, 'func'):
        parser.print_help()
        sys.exit(1)
    
    args.func(args)


if __name__ == "__main__":
    main()