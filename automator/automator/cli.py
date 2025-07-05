#!/usr/bin/env python3
"""
CLI interface for automator package.
"""

import argparse
import os
import sys
from pathlib import Path

from .workspace import Workspace


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
    add_parser.add_argument(
        "--venv",
        action="store_true",
        help="Ensure a virtual environment is created"
    )
    add_parser.set_defaults(func=workspace_add)
    
    args = parser.parse_args()
    
    if not hasattr(args, 'func'):
        parser.print_help()
        sys.exit(1)
    
    args.func(args)


if __name__ == "__main__":
    main()