from __future__ import annotations

import asyncio

from automator.agent import Agent
from automator.workspace import Workspace
from automator.dtypes import ChatMessage, ToolUseBlock, SubagentToolDefinition


import os
import glob
import re


def get_codebase(path='.', ignore_patterns=[], shortfile_patterns=[]):
    """
    Get the codebase as a string.
    Ignore files ignored by .gitignore
    """
    gitignore_path = os.path.join(path, '.gitignore')
    if os.path.exists(gitignore_path):
        with open(gitignore_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                ignore_patterns.append(line)

    # Always ignore .git directory
    ignore_patterns.append('.git')
    ignore_patterns.append('*.pyc')
    ignore_patterns.append('__pycache__')

    def is_included(filepath, patterns):
        relpath = os.path.relpath(filepath, path)
        for pattern in patterns:
            # Handle directory ignore
            if pattern.endswith('/'):
                if relpath.startswith(pattern.rstrip('/')):
                    return True
            # Handle wildcard patterns
            if glob.fnmatch.fnmatch(relpath, pattern):
                return True
        return False

    codebase_strs = []
    for root, dirs, files in os.walk(path):
        # Remove ignored directories in-place
        dirs[:] = [d for d in dirs if not is_included(os.path.join(root, d), ignore_patterns)]
        for file in files:
            filepath = os.path.join(root, file)
            if is_included(filepath, ignore_patterns):
                continue
            # Only include text files (skip binaries)
            try:
                if is_included(filepath, shortfile_patterns):
                    codebase_strs.append(f'# --- file: {os.path.relpath(filepath, path)} (skipped)')
                else:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        content = f.read()
                    codebase_strs.append(f'# --- file: {os.path.relpath(filepath, path)}\n{content}\n# EOF file: {os.path.relpath(filepath, path)}')
            except Exception:
                # Skip unreadable or binary files
                continue
    return '\n'.join(codebase_strs)





async def main() -> None:
    workspace = Workspace('automator', env={
        'CWD': '/Users/nielswarncke/Documents/researchoor/automator'
    })
    # codebase = Agent(
    #     model='google/gemini-2.5-pro-preview',
    #     prompt_template_yaml="prompts/codebase.yaml",
    #     as_tool=SubagentToolDefinition(
    #         description="This subagent has the entire codebase in context and can answer questions, analyze code and ideas, and create briefings. Example use case: ask which files are relevant for your task.",
    #         name='codebase_qa'
    #     ),
    #     workspace=workspace,
    #     id="codebase",
    #     prompt_template_vars={
    #         'codebase': get_codebase(shortfile_patterns=["prompts/"], ignore_patterns=["more_examples/", "uv.lock", "task.md"]),
    #     }
    # )
    agent = Agent(
        model='o3',
        prompt_template_yaml="prompts/swe.yaml",
        tools=["terminal.*"],
        workspace=workspace,
        id='coder',
        # subagents=['codebase'],
    )
    with open('task.md', 'r') as f:
        query = f.read()
    query += "\n\nHere is the current codebase:\n" + get_codebase(
        shortfile_patterns=["prompts/", "*.lock"],
        ignore_patterns=["more_examples/", "task.md"]
    )
    input(query)

    thread = await agent.run(query) # Can you evaluate if `Qwen/Qwen3-8B` is pro animal-welfare in a non-trivial way? How does it compare to gpt-4.1?
    
    while True:
        async for message in thread:
            print(message)
        query = input("Query> ")
        if query == 'exit':
            break
        thread = await (thread or agent).run(query)
    await thread.cleanup()


if __name__ == "__main__":  # pragma: no cover – manual invocation only
    asyncio.run(main())
