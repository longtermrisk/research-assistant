# editor_mcp.py
import os
import re
import io # Use io instead of StringIO for BytesIO
import base64 # Needed for decoding image data
import hashlib # For creating safe filenames
from typing import Union, List, Dict, Any
import json

import pandas as pd
import nbformat
import git
from PIL import Image as PILImage
import difflib

from mcp.server.fastmcp import Context, Image
from mcp.types import TextContent
from static_code_analysis import analyze_and_format
from server import mcp


def sanitize_filename(name):
    """Removes potentially problematic characters for filenames."""
    # Keep alphanumeric, underscores, hyphens, dots
    sanitized = re.sub(r'[^\w\-.]', '_', name)
    # Avoid names starting with dot or dash, replace multiple dots/dashes
    sanitized = re.sub(r'^[.-]+', '', sanitized)
    sanitized = re.sub(r'[.-]{2,}', '_', sanitized)
    # Limit length slightly
    return sanitized[:100]

def open_jupyter_notebook(notebook_path: str) -> List[Union[str, Image]]:
    """Return an ordered list of `str` and `Image` blocks for the notebook."""

    blocks: List[Union[str, Image]] = []
    text_buf: List[str] = []

    def flush():
        if text_buf:
            blocks.append("".join(text_buf))
            text_buf.clear()

    try:
        with open(notebook_path, "r", encoding="utf-8") as f:
            nb = nbformat.read(f, as_version=4)
    except Exception as exc:
        return [f"Error reading notebook {os.path.basename(notebook_path)}: {exc}"]

    rel_path = os.path.relpath(notebook_path, os.getcwd())
    text_buf.append(f"# Jupyter Notebook: {rel_path}\n\n")

    for i, cell in enumerate(nb["cells"]):
        text_buf.append(f"## Cell {i+1}: {cell.cell_type}\n\n")

        if cell.cell_type == "code":
            text_buf.append("```python\n" + cell.source + "\n```\n")
            if cell.get("outputs"):
                text_buf.append("\n### Outputs:\n")
                for j, output in enumerate(cell.outputs):
                    text_buf.append(f"\n--- Output {j+1} ({output.output_type}) ---\n")
                    if output.output_type == "stream":
                        text_buf.append("```text\n" + output.text + "```\n")
                    elif output.output_type == "error":
                        tb = "\n".join(output.traceback)
                        text_buf.append(f"**Error:** {output.ename}\n```traceback\n{tb}\n```\n")
                    elif output.output_type in {"display_data", "execute_result"}:
                        handled_img = False
                        if "image/png" in output.data:
                            flush()
                            img_bytes = base64.b64decode(output.data["image/png"])
                            blocks.append(Image(data=img_bytes, format="png"))
                            handled_img = True
                        if "text/plain" in output.data and not handled_img:
                            text_buf.append("```text\n" + output.data["text/plain"] + "```\n")
                        if "text/html" in output.data and not handled_img:
                            text_buf.append("[HTML output suppressed]\n")
                    else:
                        text_buf.append(f"[Unhandled output type: {output.output_type}]\n")
        elif cell.cell_type == "markdown":
            text_buf.append(cell.source + "\n")
        elif cell.cell_type == "raw":
            text_buf.append("```raw\n" + cell.source + "```\n")
        else:
            text_buf.append(f"[Unsupported cell type: {cell.cell_type}]\n")
        text_buf.append("\n")

    flush()
    return blocks

def clean_content(content: str, path: str) -> str:
    """Clean the content by removing common agent artifacts."""
    lines = content.split("\n")
    if lines and lines[0].startswith("<file:") and path in lines[0]:
        lines = lines[1:]
    if lines and lines[-1] == f"</file:{path}>":
        lines = lines[:-1]

    if lines and lines[0].strip().startswith("```"):
         # Only remove if it looks like a markdown fence *without* language specifier potentially
         first_line_strip = lines[0].strip()
         if first_line_strip == '```':
             lines = lines[1:]
         # Be cautious about removing ```python etc.
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]

    if lines and lines[0].startswith("<![CDATA["):
        lines[0] = lines[0].replace("<![CDATA[", "")
    if lines and lines[-1].endswith("]]>"):
        lines[-1] = lines[-1].replace("]]>", "")
    content = "\n".join(lines)
    return content.strip()


def validate_content(content: str) -> str:
    """Validate the content for potential placeholder/instruction text."""
    warning_signs = [
        r'#\s*\[.*?\]',              # Python comment with [...]
        r'//\s*\[.*?\]',             # JS comment with [...]
        r'(?i)(?:#|//).*?\b(keep|replace|until|leave|retain)\b\s+existing.*',
        r'(?i)(?:#|//).*?\.{3}.*?\b(keep|until)\b.*',
        r'(?i)(?:#|//).*?\brest\b.*?(?:methods?|code).*?(?:same|unchanged|as is|identical)',
        r'(?i)(?:#|//).*?(?:methods?|code).*?(?:remain|continues?|unchanged|same|as is|identical)\b.*',
        r'(?i)(?:#|//).*?(?:imports?|includes?|declarations?|definitions?)\b.*?(?:remain|same|unchanged|as is|identical)',
         r'(?i)(?:#|//).*?(?:previous|existing|above|below)\b.*?(?:imports?|includes?|declarations?|definitions?).*?(?:remain|same|unchanged|as is|identical)',
        r'(?i)(?:#|//).*?(?:rest of|other)\b.*?(?:imports?|includes?|code|file)\b.*',
        r'(?i)<(placeholder|insert code here|your code here)>' # Common placeholders
    ]
    warnings = []
    for warning_sign in warning_signs:
        if re.search(warning_sign, content):
            warnings.append(f"Potential instruction/placeholder detected matching pattern: `{warning_sign}`.")

    if warnings:
        return "Warning: The provided content might contain instructions or placeholders instead of literal file content.\n" + "\n".join(warnings) + "\nPlease provide the complete, literal file content or use 'ignore_warnings=True' if this is intended."
    return ""


@mcp.tool()
async def get_file(path: str, ctx: Context = None) -> Union[str, Image, List[Union[str, Image]]]:
    workspace = os.path.abspath(".")
    abs_path = os.path.abspath(os.path.normpath(os.path.join(workspace, path)))

    if not abs_path.startswith(workspace):
        return f"Error: Access denied. Path '{path}' is outside the workspace."
    if not os.path.exists(abs_path):
        return f"Error: File or directory not found: {path}"

    if os.path.isdir(abs_path):
        try:
            entries = [e for e in os.listdir(abs_path) if not e.startswith('.')]
            return "Directory Listing for: " + path + "\n" + "\n".join(entries)
        except OSError as exc:
            return f"Error listing directory {path}: {exc}"

    ext = os.path.splitext(path)[1].lower()

    if ext in {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp"}:
        try:
            with PILImage.open(abs_path) as img:
                fmt = (img.format or "png").lower().replace('jpg', 'jpeg')
                buf = io.BytesIO()
                img.save(buf, format=fmt.upper(), **({"quality": 85} if fmt == "jpeg" else {}))
                return Image(data=buf.getvalue(), format=fmt)
        except Exception as exc:
            return f"Error reading image {path}: {exc}"

    if ext == ".csv":
        try:
            df = pd.read_csv(abs_path)
            buf = io.StringIO()
            df.info(buf=buf)
            return f"CSV Info for: {path}\n\n{buf.getvalue()}"
        except Exception as exc:
            return f"Error reading CSV {path}: {exc}"

    if ext == ".ipynb":
        return open_jupyter_notebook(abs_path)

    try:
        with open(abs_path, "r", encoding="utf-8") as f:
            return f.read()
    except UnicodeDecodeError:
        return f"Error: Cannot decode file {path} as UTF-8."
    except Exception as exc:
        return f"Error reading file {path}: {exc}"


# --- MCP Tool (Write) ---

@mcp.tool()
async def write_file(
    path: str,
    content: str | Dict[str, Any],
    ignore_warnings: bool = False,
    lint: bool = False,
    ctx: Context = None
) -> str:
    """
    Writes/overwrites a file. Provide the full desired content.
    Use path relative to workspace, e.g., 'my_folder/my_file.txt'.
    If content is a dict, it will be converted to a JSON string.
    """
    if isinstance(content, dict):
        content = json.dumps(content, indent=4)
    workspace = os.path.abspath(".")
    # Prevent writing to the special image directory or other sensitive paths
    normalized_rel_path = os.path.normpath(path)

    full_path = os.path.abspath(os.path.join(workspace, normalized_rel_path))

    if not full_path.startswith(workspace):
         return f"Error: Access denied. Path '{path}' is outside the allowed workspace."
    if os.path.isdir(full_path):
        return f"Error: Cannot write to '{path}'. It is an existing directory."
    
    if path.endswith(".ipynb"):
        return f"Error: Cannot write to '{path}'. It is a Jupyter notebook file. Use the jupyter tool to modify notebooks."

    dir_name = os.path.dirname(full_path)
    if dir_name:
        try:
            os.makedirs(dir_name, exist_ok=True)
        except OSError as e:
            return f"Error creating directory {dir_name}: {e}"

    cleaned_content = clean_content(content, path)

    if not ignore_warnings:
        warning = validate_content(cleaned_content)
        if warning:
            return warning # Return warning and stop
    
    # If it's a json file, ensure it's valid JSON
    if path.endswith(".json"):
        try:
            json.loads(cleaned_content)
        except json.JSONDecodeError as e:
            return f"Error: Invalid JSON content in {path}: {e}"

    # If it's a CSV file, ensure it's valid CSV
    elif path.endswith(".csv"):
        try:
            pd.read_csv(io.StringIO(cleaned_content))
        except pd.errors.ParserError as e:
            return f"Error: Invalid CSV content in {path}: {e}"
    
    # Read the file to get existing content if it exists
    try:
        with open(full_path, "r", encoding="utf-8") as f:
            existing_content = f.read()
    except FileNotFoundError:
        existing_content = ""

    try:
        with open(full_path, "w", encoding='utf-8') as f:
            f.write(cleaned_content)
        response = f"File successfully written: {path}"
    except OSError as e:
        return f"Error writing file {path}: {e}"
    except Exception as e:
        return f"An unexpected error occurred while writing {path}: {e}"

    if lint:
        try:
            code_extensions = {'.py', '.js', '.ts', '.java', '.c', '.cpp', '.go', '.rb', '.php'}
            if os.path.splitext(path)[1].lower() in code_extensions:
                 lint_result = analyze_and_format(full_path)
                 response += f"\nLinting/Formatting Result:\n{lint_result}"
            else:
                 response += f"\nLinting skipped: File type not recognized ({os.path.splitext(path)[1]})."
        except Exception as e:
            response += f"\nError during linting/formatting: {e}"
    return response_with_filediff(existing_content, cleaned_content, response)

def response_with_filediff(existing_content: str, new_content: str, response: str) -> TextContent:
    """
    Returns: TextContent(response, annotations={'display_html': <a file diff with red/green> highlights>})
    """
    # Generate a diff using difflib
    diff = difflib.unified_diff(
        existing_content.splitlines(keepends=True),
        new_content.splitlines(keepends=True),
        fromfile='Existing Content',
        tofile='New Content',
        lineterm=''
    )
    # Convert diff to HTML with GitHub-style highlighting
    diff_html = '<div style="font-family: monospace; white-space: pre;">'
    for line in diff:
        if line.startswith('+'):
            # Darker green background for additions
            escaped_line = line.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
            diff_html += f'<div style="background-color: #0f5323">{escaped_line}</div>'
        elif line.startswith('-'):
            # Darker red background for deletions  
            escaped_line = line.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
            diff_html += f'<div style="background-color: #5c1e1e">{escaped_line}</div>'
        elif line.startswith('@@'):
            # Darker blue/gray background for diff headers
            escaped_line = line.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
            diff_html += f'<div style="background-color: #1f364d">{escaped_line}</div>'
        else:
            # No background for context lines
            escaped_line = line.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
            diff_html += f'<div>{escaped_line}</div>'
    diff_html += '</div>'
    return TextContent(
        text=response,
        type="text",
        annotations={'display_html': diff_html}
    )

# --- MCP Tool (List Codebase Files) ---
@mcp.tool()
async def list_codebase_files(
    codebase_path: str = ".",
    ctx: Context = None
) -> Union[List[str], str]:
    """
    Lists files in a directory (relative to workspace), respecting .gitignore.
    Returns a list of relative file paths or an error string.
    """
    workspace = os.path.abspath(".")
    # Resolve and validate codebase_path
    requested_codebase_path = os.path.normpath(os.path.join(workspace, codebase_path))
    full_codebase_path = os.path.abspath(requested_codebase_path)

    if not full_codebase_path.startswith(workspace):
         return f"Error: Access denied. Path '{codebase_path}' is outside the allowed workspace."
    if not os.path.isdir(full_codebase_path):
        return f"Error: Path '{codebase_path}' is not a valid directory."

    files = []
    try:
        repo = None
        ignored_files_rel = set() # Store paths relative to full_codebase_path

        # Check if it's a git repo to use .gitignore
        try:
            repo = git.Repo(full_codebase_path, search_parent_directories=True)
            repo_dir = repo.working_tree_dir
            # Only use gitignore if the requested path is *within* the repo tree
            if full_codebase_path.startswith(repo_dir):
                # Get all items, check ignore status relative to repo root
                all_items_in_path = [os.path.join(root, name) for root, _, files_ in os.walk(full_codebase_path) for name in files_ + os.listdir(root)]

                # Get ignored paths relative to repo root
                ignored_repo_paths = set(repo.ignored(*[os.path.relpath(p, repo_dir) for p in all_items_in_path if os.path.exists(p)]))

                # Convert repo-relative ignored paths to be relative to codebase_path
                # An item is ignored if its path *relative to repo_dir* is in ignored_repo_paths
                for root, dirs, filenames in os.walk(full_codebase_path, topdown=True):
                    current_rel_root_to_repo = os.path.relpath(root, repo_dir)
                    # Filter dirs: ignore if dir itself (relative to repo) is ignored
                    # Need to check both relative/absolute potentially based on gitignore rules
                    dirs[:] = [d for d in dirs if os.path.normpath(os.path.join(current_rel_root_to_repo, d)) not in ignored_repo_paths and not d.startswith('.')]

                    for filename in filenames:
                         if filename.startswith('.'): continue # Skip hidden files
                         file_rel_path_to_repo = os.path.normpath(os.path.join(current_rel_root_to_repo, filename))
                         if file_rel_path_to_repo not in ignored_repo_paths:
                             # Store path relative to the original codebase_path request
                             file_rel_path_to_codebase = os.path.relpath(os.path.join(root, filename), full_codebase_path)
                             files.append(os.path.normpath(file_rel_path_to_codebase).replace(os.sep, '/'))
                repo_processed = True # Mark that git logic handled this walk
            else:
                repo = None # Path outside repo, fallback
                repo_processed = False
        except git.InvalidGitRepositoryError:
            repo = None # Not a git repo
            repo_processed = False
        except Exception as git_err: # Catch other potential git errors
            print(f"Warning: Git check failed for {full_codebase_path}: {git_err}")
            repo = None
            repo_processed = False


        if not repo_processed: # Fallback for non-git dirs or if git failed
             # Basic ignore: .git, node_modules, __pycache__, and our image dir
            ignored_dirs = {'.git', 'node_modules', '__pycache__'}
            for root, dirs, filenames in os.walk(full_codebase_path, topdown=True):
                rel_root = os.path.relpath(root, full_codebase_path)
                # Exclude ignored and hidden dirs
                dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ignored_dirs]
                for filename in filenames:
                    if not filename.startswith('.'): # Ignore hidden files
                        # File path relative to the requested codebase_path
                        rel_path = os.path.join(rel_root, filename) if rel_root != '.' else filename
                        files.append(os.path.normpath(rel_path).replace(os.sep, '/')) # Normalize and use forward slash

        # Sort files
        def sort_key(f):
            dirname, basename = os.path.split(f)
            is_readme = basename.lower().startswith('readme')
            return (dirname, not is_readme, basename)

        files.sort(key=sort_key)
        return files # Return the list of relative paths

    except Exception as e:
        # Log the error maybe? ctx.error(...) ?
        print(f"Error listing files in '{codebase_path}': {e}")
        return f"An error occurred while listing files in '{codebase_path}': {e}"

