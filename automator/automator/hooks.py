import os
import glob
import importlib.util
from automator.dtypes import TextBlock

_HOOKS = {}


def register_hook(name):
    """Register a hook with the given name."""
    def decorator(func):
        _HOOKS[name] = func
    return decorator


def load_hooks():
    """Import all hooks from python files in ~/.automator/hooks."""
    hooks_dir = os.path.expanduser('~/.automator/hooks')
    os.makedirs(hooks_dir, exist_ok=True)
    for filepath in glob.glob(os.path.join(hooks_dir, '*.py')):
        module_name = os.path.splitext(os.path.basename(filepath))[0]
        spec = importlib.util.spec_from_file_location(module_name, filepath)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)


# ------------- Default hooks -------------
def init_claude_md(thread) -> None:
    home = thread.home
    claude_md_path = home / "LLM.md"
    if os.path.exists(home / '.venv'):
        python = "## Python\nThis project uses uv to manage dependencies. The default python points to the local venv. Use `uv add <package>` to install a package."

    if not claude_md_path.exists():
        with open(claude_md_path, "w") as f:
            cwd_content_str = "\n".join([f"- {p.name}" for p in home.iterdir() if p.is_file()])
            f.write(f"""# {thread.workspace.name}
This is an automatically generated overview of the current workspace.

## Files

{cwd_content_str}

{python}

## Updating this file

This file should serve as an onboarding guide for you in the future. Keep it up-to-date with info about:
- the purpose of the project
- the state of the code base
- any other relevant information
""")


@register_hook('claude.md')
async def claude_md(thread):
    """If the thread has terminal tools, add CLAUDE.md to the system prompt."""
    # Check if the thread has terminal tools
    if not any(tool_name.startswith('terminal.') for tool_name in thread._tools):
        return
    # Check if the hook has already been applied
    system_message = thread.messages_after_hooks[0]
    if any((block.meta or {}).get('claude_md') for block in system_message.content):
        return
    # Add CLAUDE.md to the system prompt
    if os.path.exists(thread.home / 'LLM.md'):
        with open(thread.home / 'LLM.md', 'r') as f:
            claude_md_text = f"<LLM.md>\n{f.read()}\n</LLM.md>"
    elif os.path.exists(thread.home / 'CLAUDE.md'):
        with open(thread.home / 'CLAUDE.md', 'r') as f:
            claude_md_text = f"<CLAUDE.md>\n{f.read()}\n</CLAUDE.md>"
    else:
        init_claude_md(thread)
        with open(thread.home / 'LLM.md', 'r') as f:
            claude_md_text = f"<LLM.md>\n{f.read()}\n</LLM.md>"
    system_message.content += [
        TextBlock(
            text=claude_md_text,
            meta={'claude_md': True}
        )
    ]
