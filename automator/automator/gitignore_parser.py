# automator/gitignore_parser.py
import fnmatch
from pathlib import Path
from typing import List, Callable

def load_gitignore_patterns(gitignore_path: Path) -> List[str]:
    """Loads patterns from a .gitignore file."""
    if not gitignore_path.is_file():
        return []
    with open(gitignore_path, 'r', encoding='utf-8') as f:
        patterns = [
            line.strip()
            for line in f
            if line.strip() and not line.startswith('#')
        ]
    return patterns

def is_path_ignored(
    path_to_check: Path, # Absolute path to the file/folder to check
    base_dir: Path,      # Absolute path to the directory containing .gitignore (and where patterns are relative to)
    patterns: List[str]
) -> bool:
    """
    Checks if a given path should be ignored based on .gitignore patterns.
    path_to_check and base_dir should be absolute paths.
    """
    if '.git' in path_to_check.parts: # Always ignore .git directories and their contents
        return True

    relative_path_posix = path_to_check.relative_to(base_dir).as_posix()

    for pattern in patterns:
        # Handle directory patterns (ending with /)
        is_dir_pattern = pattern.endswith('/')
        match_pattern = pattern.rstrip('/')

        # Adjust pattern for fnmatch:
        # If pattern doesn't start with '*', '**/', or '/', it's relative to base_dir
        # fnmatch matches full path part by default, so 'file.txt' won't match 'somedir/file.txt'
        # Gitignore behavior: 'file.txt' matches 'file.txt' and 'somedir/file.txt'
        # So, if not a glob starting with '*', we might need to prepend '**/' implicitly for some cases.
        # For simplicity here, we'll match based on path components for non-glob patterns.

        # Simplified matching:
        # 1. Direct match: pattern == relative_path_posix or pattern == relative_path_posix + '/'
        # 2. fnmatch:
        #    - If pattern starts with '/', it's from the root of the repo (base_dir)
        #    - Otherwise, it can match anywhere. Prepend '*' for fnmatch.
        
        # fnmatch expects the pattern to match the whole string.
        # Gitignore rules are more complex:
        # - 'foo' matches 'foo', 'dir/foo'
        # - '/foo' matches 'foo' at root only
        # - 'foo/' matches directory 'foo' anywhere
        # - 'dir/foo' matches 'dir/foo'

        # Let's use a more direct fnmatch approach:
        # For a pattern like "logs/*.log":
        #   fnmatch.fnmatch(relative_path_posix, "logs/*.log") -> True for "logs/some.log"
        # For a pattern like "*.pyc":
        #   fnmatch.fnmatch(relative_path_posix, "*.pyc") -> True for "file.pyc", "dir/file.pyc" (if pattern becomes "**/ *.pyc")
        #   fnmatch needs to match the full path.

        current_match_pattern = pattern
        if not pattern.startswith('/') and '/' not in pattern and not pattern.startswith('*'):
            # For patterns like 'file.txt' or '*.log' to match in any subdir
            current_match_pattern = f"*{pattern}" if not pattern.startswith('*') else pattern
        
        # Remove starting '/' for fnmatch as relative_path_posix is already relative
        if current_match_pattern.startswith('/'):
            current_match_pattern = current_match_pattern[1:]


        if fnmatch.fnmatch(relative_path_posix, current_match_pattern):
            if is_dir_pattern and not path_to_check.is_dir(): # Pattern expects dir, but path is file
                continue
            return True
        
        # If pattern is for a directory, also check if path_to_check is inside such a dir
        if is_dir_pattern or (path_to_check.is_dir() and fnmatch.fnmatch(relative_path_posix + '/', pattern)):
             if relative_path_posix.startswith(match_pattern): # path_to_check is child of ignored dir pattern
                 return True
        
        # Match if pattern is a dir name and path_to_check is inside it
        # e.g. pattern "node_modules", path "node_modules/somefile"
        if path_to_check.is_file() and fnmatch.fnmatch(path_to_check.name, pattern): # e.g. *.log matches file.log
             if '/' not in pattern: # if pattern is just a filename glob
                 return True
        
    return False