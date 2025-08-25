import os
import glob
import re
from typing import List, Tuple, Optional
from pathlib import Path
from collections import defaultdict
from mcp.types import TextContent
from server import mcp
import git

try:
    from isignored import is_ignored
    HAS_ISIGNORED = True
except ImportError:
    HAS_ISIGNORED = False


def is_binary_file(file_path: str) -> bool:
    """Check if a file is binary by looking for null bytes in the first 8192 bytes."""
    try:
        with open(file_path, 'rb') as f:
            chunk = f.read(8192)
            return b'\0' in chunk
    except:
        return True  # If we can't read it, assume it's binary


def should_skip_file_for_ignore(file_path: str) -> bool:
    """Check if a file should be skipped based on gitignore rules."""
    if HAS_ISIGNORED and is_ignored(file_path):
        return True
    return False


def should_skip_file_for_binary(file_path: str) -> bool:
    """Check if a file should be skipped because it's binary."""
    if is_binary_file(file_path):
        return True
        
    # Skip common binary file extensions
    binary_extensions = {'.pyc', '.pyo', '.pyd', '.so', '.dylib', '.dll', 
                        '.exe', '.bin', '.jpg', '.png', '.gif', '.pdf', 
                        '.zip', '.tar', '.gz', '.bz2', '.xz'}
    
    if Path(file_path).suffix.lower() in binary_extensions:
        return True
        
    return False


def get_files_from_directory(dir_path: str, workspace: str) -> List[str]:
    """
    Get all non-ignored files from a directory using git-aware logic.
    Similar to list_codebase_files but returns absolute paths.
    """
    full_dir_path = os.path.abspath(dir_path)
    
    # Security check
    if not full_dir_path.startswith(workspace):
        return []
    
    if not os.path.isdir(full_dir_path):
        return []

    files = []
    try:
        repo = None
        
        # Check if it's a git repo to use .gitignore
        try:
            repo = git.Repo(full_dir_path, search_parent_directories=True)
            repo_dir = repo.working_tree_dir
            
            # Only use gitignore if the requested path is within the repo tree
            if full_dir_path.startswith(repo_dir):
                # Get all items, check ignore status relative to repo root
                all_items_in_path = []
                for root, dirs, filenames in os.walk(full_dir_path):
                    for filename in filenames:
                        file_path = os.path.join(root, filename)
                        all_items_in_path.append(file_path)

                # Get ignored paths relative to repo root
                repo_relative_paths = [os.path.relpath(p, repo_dir) for p in all_items_in_path if os.path.exists(p)]
                ignored_repo_paths = set(repo.ignored(*repo_relative_paths))

                # Collect non-ignored files
                for root, dirs, filenames in os.walk(full_dir_path, topdown=True):
                    current_rel_root_to_repo = os.path.relpath(root, repo_dir)
                    
                    # Filter dirs: ignore if dir itself (relative to repo) is ignored
                    dirs[:] = [d for d in dirs if os.path.normpath(os.path.join(current_rel_root_to_repo, d)) not in ignored_repo_paths and not d.startswith('.')]

                    for filename in filenames:
                        if filename.startswith('.'):
                            continue  # Skip hidden files
                        
                        file_rel_path_to_repo = os.path.normpath(os.path.join(current_rel_root_to_repo, filename))
                        if file_rel_path_to_repo not in ignored_repo_paths and not filename.endswith('.lock'):
                            files.append(os.path.join(root, filename))
                            
                repo_processed = True
            else:
                repo = None
                repo_processed = False
        except (git.InvalidGitRepositoryError, Exception):
            repo = None
            repo_processed = False

        if not repo_processed:  # Fallback for non-git dirs or if git failed
            # Basic ignore: .git, node_modules, __pycache__
            ignored_dirs = {'.git', 'node_modules', '__pycache__'}
            for root, dirs, filenames in os.walk(full_dir_path, topdown=True):
                # Exclude ignored and hidden dirs
                dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ignored_dirs]
                for filename in filenames:
                    if not filename.startswith('.') and not filename.endswith('.lock'):
                        files.append(os.path.join(root, filename))

    except Exception as e:
        print(f"Error listing files in directory '{dir_path}': {e}")
        return []
    
    return files


def get_files_from_glob_pattern(pattern: str, workspace: str) -> List[str]:
    """
    Get all files matching a glob pattern, filtered by gitignore rules.
    """
    try:
        # Use glob to find matching files
        matched_files = glob.glob(pattern, recursive=True)
        # Filter to only include files (not directories)
        matched_files = [f for f in matched_files if os.path.isfile(f)]
        
        # Filter out ignored files
        non_ignored_files = []
        for file_path in matched_files:
            abs_file_path = os.path.abspath(file_path)
            
            # Security check: ensure file is within workspace or tool_output
            tool_output_dir = os.path.abspath("./tool_output")
            if not (abs_file_path.startswith(workspace) or abs_file_path.startswith(tool_output_dir)):
                continue
            
            # Check if file should be ignored
            if not should_skip_file_for_ignore(file_path):
                non_ignored_files.append(abs_file_path)
        
        return non_ignored_files
    except Exception as e:
        print(f"Error processing glob pattern '{pattern}': {e}")
        return []


def determine_path_type(path: str) -> str:
    """
    Determine if the path is a specific file, directory, or glob pattern.
    Returns: 'file', 'directory', or 'glob'
    """
    # Check if it's a specific existing file
    if os.path.isfile(path):
        return 'file'
    
    # Check if it's an existing directory
    if os.path.isdir(path):
        return 'directory'
    
    # Check if it contains glob characters
    if any(char in path for char in ['*', '?', '[', ']', '**']):
        return 'glob'
    
    # If it doesn't exist and has no glob chars, treat as potential file
    # (could be a typo or non-existent file)
    return 'file'


def search_in_file(file_path: str, search_str: str, context: int = 0, 
                  max_matches: int = 50, max_line_length: int = 200) -> List[Tuple[int, str, List[str]]]:
    """
    Search for a string in a file and return matching lines with line numbers and context.
    
    Args:
        file_path: Path to the file to search
        search_str: String to search for
        context: Number of context lines to include before/after matches
        max_matches: Maximum number of matches to return
        max_line_length: Maximum length of lines to display (truncate longer lines)
        
    Returns:
        List of tuples (line_number, line_content, context_lines)
    """
    matches = []
    
    if should_skip_file_for_binary(file_path):
        return []
    
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
            
        for line_num, line in enumerate(lines):
            line = line.rstrip()
            if search_str.lower() in line.lower():
                # Truncate long lines
                if len(line) > max_line_length:
                    line = line[:max_line_length] + "... (truncated)"
                
                # Get context lines
                context_lines = []
                if context > 0:
                    start = max(0, line_num - context)
                    end = min(len(lines), line_num + context + 1)
                    for i in range(start, end):
                        if i != line_num:  # Don't duplicate the match line
                            ctx_line = lines[i].rstrip()
                            if len(ctx_line) > max_line_length:
                                ctx_line = ctx_line[:max_line_length] + "... (truncated)"
                            context_lines.append((i + 1, ctx_line))
                
                matches.append((line_num + 1, line, context_lines))
                
                if len(matches) >= max_matches:
                    break
                    
    except Exception as e:
        matches.append((0, f"Error reading file {file_path}: {e}", []))
    
    return matches


def group_files_by_type(files: List[str]) -> dict:
    """Group files by their extension or directory."""
    groups = defaultdict(list)
    
    for file_path in files:
        path_obj = Path(file_path)
        if path_obj.suffix:
            ext = path_obj.suffix.lower()
            groups[f"{ext} files"].append(file_path)
        else:
            groups["files (no extension)"].append(file_path)
    
    return dict(groups)


def format_match_summary(matches_per_file: dict) -> str:
    """Create a summary of matches per file type."""
    if not matches_per_file:
        return ""
    
    # Sort files by number of matches (ascending)
    sorted_files = sorted(matches_per_file.items(), key=lambda x: len(x[1]))
    
    summary_lines = []
    for file_path, matches in sorted_files:
        match_count = len(matches)
        if match_count == 1:
            summary_lines.append(f"  {file_path}: {match_count} match")
        else:
            summary_lines.append(f"  {file_path}: {match_count} matches")
    
    return "\n".join(summary_lines)


def format_file_matches(file_path: str, matches: List[Tuple[int, str, List[str]]], 
                       search_str: str, context: int = 0) -> List[str]:
    """Format matches for a single file."""
    lines = []
    lines.append(f"\n=== {file_path} ===")
    
    if len(matches) > 20:  # Limit matches per file for readability
        lines.append(f"  (Showing first 20 of {len(matches)} matches)")
        matches = matches[:20]
    
    for line_num, line_content, context_lines in matches:
        if line_num == 0:  # Error case
            lines.append(f"  {line_content}")
            continue
        
        # Show context before match
        if context > 0 and context_lines:
            for ctx_line_num, ctx_content in context_lines:
                if ctx_line_num < line_num:
                    lines.append(f"  {ctx_line_num:4d}  {ctx_content}")
        
        # Highlight the search term in the match line
        highlighted_line = re.sub(
            re.escape(search_str), 
            f"**{search_str}**", 
            line_content, 
            flags=re.IGNORECASE
        )
        lines.append(f"  {line_num:4d}: {highlighted_line}")
        
        # Show context after match
        if context > 0 and context_lines:
            for ctx_line_num, ctx_content in context_lines:
                if ctx_line_num > line_num:
                    lines.append(f"  {ctx_line_num:4d}  {ctx_content}")
    
    return lines


@mcp.tool()
async def find(search_str: str, path: str, context: int = 0, files_only: bool = False, 
               max_line_length: int = 200, group_by_type: bool = False) -> TextContent:
    """
    Search for a string in files. The path can be a specific file, directory, or glob pattern.
    
    Args:
        search_str: String to search for (case-insensitive)
        path: File path, directory path, or glob pattern (e.g., "*.py", "src/**/*.js", "tool_output/output_*.txt")
        context: Number of context lines to show before/after each match (default: 0)
        files_only: If True, only show filenames without match content (default: False)
        max_line_length: Maximum line length before truncation (default: 200)
        group_by_type: If True, group results by file type (default: False)
        
    Returns:
        Search results showing matches with file paths and line numbers
    """
    if not search_str.strip():
        return "Error: search_str cannot be empty"
    
    # Handle relative paths
    workspace = os.path.abspath(".")
    
    # Make path absolute if it's relative
    if os.path.isabs(path):
        search_path = path
    else:
        search_path = os.path.join(workspace, path)
    
    # Determine path type and get files accordingly
    path_type = determine_path_type(search_path)
    
    files_to_search = []
    
    if path_type == 'file':
        # Specific file - don't check if it's ignored per requirement
        if os.path.isfile(search_path):
            files_to_search = [search_path]
        else:
            return f"File not found: {path}"
    
    elif path_type == 'directory':
        # Directory - include all non-ignored files in that directory
        files_to_search = get_files_from_directory(search_path, workspace)
        if not files_to_search:
            return f"No accessible files found in directory: {path}"
    
    elif path_type == 'glob':
        # Glob pattern - include all non-ignored files that match the pattern
        files_to_search = get_files_from_glob_pattern(search_path, workspace)
        if not files_to_search:
            return f"No files found matching pattern: {path}"
    
    # Search in each file
    matches_per_file = {}
    total_files_searched = 0
    files_with_matches = 0
    skipped_files = 0
    
    for file_path in files_to_search:
        # For specific files, don't check ignore status
        # For directories and globs, ignore status was already checked
        if path_type == 'file' or not should_skip_file_for_ignore(file_path):
            # Always skip binary files regardless of path type
            if should_skip_file_for_binary(file_path):
                skipped_files += 1
                continue
                
            total_files_searched += 1
            matches = search_in_file(file_path, search_str, context=context, max_line_length=max_line_length)
            
            if matches:
                files_with_matches += 1
                # Make path relative for display
                display_path = os.path.relpath(file_path, workspace) if file_path.startswith(workspace) else file_path
                matches_per_file[display_path] = matches
        else:
            skipped_files += 1
    
    if not matches_per_file:
        skip_msg = f" ({skipped_files} files skipped - binary/ignored)" if skipped_files > 0 else ""
        path_type_desc = "file" if path_type == 'file' else f"{total_files_searched + skipped_files} files"
        return f"No matches found for '{search_str}' in {path_type_desc} at '{path}'{skip_msg}"
    
    # Format results
    skip_msg = f" ({skipped_files} skipped)" if skipped_files > 0 else ""
    result_lines = [f"Found '{search_str}' in {files_with_matches} of {total_files_searched + skipped_files} files{skip_msg}:\n"]
    
    # Files-only mode
    if files_only:
        result_lines.append("Files containing matches:")
        summary = format_match_summary(matches_per_file)
        result_lines.append(summary)
        result_text = "\n".join(result_lines)
        return TextContent(
            text=result_text,
            type="text",
            annotations={'display_html': f"<pre>{result_text}</pre>"}
        )
    
    # Group by file type if requested
    if group_by_type:
        grouped_files = group_files_by_type(list(matches_per_file.keys()))
        
        for group_name, file_list in grouped_files.items():
            result_lines.append(f"\n=== {group_name} ===")
            for file_path in sorted(file_list):
                matches = matches_per_file[file_path]
                result_lines.extend(format_file_matches(file_path, matches, search_str, context))
    else:
        # Sort files by number of matches (ascending)
        sorted_files = sorted(matches_per_file.items(), key=lambda x: len(x[1]))
        
        for file_path, matches in sorted_files:
            result_lines.extend(format_file_matches(file_path, matches, search_str, context))
    
    result_text = "\n".join(result_lines)
    
    return TextContent(
        text=result_text,
        type="text",
        annotations={'display_html': f"<pre>{result_text}</pre>"}
    )