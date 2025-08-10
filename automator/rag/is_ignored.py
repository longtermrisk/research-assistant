from pathlib import Path
import fnmatch

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _iter_dirs(path: Path):
    """Yield path.parent, path.parent.parent, … up to the filesystem root."""
    path = path.resolve()
    while True:
        yield path
        if path.parent == path:
            break
        path = path.parent

def _load_rules(dirs, ignore_files):
    """Return a list of (pattern, negate, base_dir) triples, parent first."""
    rules = []
    for d in dirs:
        for fname in ignore_files:
            f = d / fname
            if not f.is_file():
                continue
            for line in f.read_text(encoding="utf-8", errors="ignore").splitlines():
                line = line.lstrip()
                if not line or line.startswith("#"):
                    continue
                negate = line.startswith("!")
                if negate:
                    line = line[1:]
                rules.append((line, negate, d))
    return rules

# ---------------------------------------------------------------------------
# git-style matcher (supports the common 90 % of the spec)
# ---------------------------------------------------------------------------

def _matches(pattern: str, rel_posix: str) -> bool:
    """
    Return True if *pattern* (as written in a .gitignore-like file) matches
    the POSIX-style path string *rel_posix*.
    """
    anchored = pattern.startswith("/")
    if anchored:
        pattern = pattern[1:]

    dir_rule = pattern.endswith("/")
    pattern_core = pattern.rstrip("/")

    has_glob = any(ch in pattern_core for ch in "*?[") or "**" in pattern_core

    # 1. pure directory rules like "data/"  or  "foo/bar/"
    if dir_rule and not has_glob:
        if anchored:
            return rel_posix == pattern_core or rel_posix.startswith(pattern_core + "/")
        # un-anchored: match that directory anywhere in the path
        if rel_posix == pattern_core or rel_posix.startswith(pattern_core + "/"):
            return True
        return f"/{pattern_core}/" in rel_posix

    # 2. explicit path rules containing “/” (directory or file), no globs
    if "/" in pattern_core and not has_glob:
        if anchored:
            return rel_posix == pattern_core or rel_posix.startswith(pattern_core + "/")
        if rel_posix == pattern_core or rel_posix.startswith(pattern_core + "/"):
            return True
        return f"/{pattern_core}/" in rel_posix

    # 3. anything involving globs  (or a simple name with no “/”)
    #    ─ first test the whole relative path
    if fnmatch.fnmatchcase(rel_posix, pattern_core):
        return True
    #    ─ then, when the pattern has no “/”, test each component (git’s behaviour)
    if "/" not in pattern_core:
        return any(fnmatch.fnmatchcase(part, pattern_core) for part in rel_posix.split("/"))

    return False

# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------

def is_ignored(path, ignore_files=(".gitignore",)):
    """
    True iff *path* would be ignored by any ignore file named in *ignore_files*
    in its directory or any parent directory (Git precedence).
    """
    path = Path(path).resolve()
    rel_dirs = list(_iter_dirs(path.parent))          # parents first
    rules = _load_rules(rel_dirs, ignore_files)

    # Hard coded: no .venv, node_modules, .git
    if "node_modules/" in path:
        return True
    if ".venv/" in path:
        return True
    if "/git" in path:
        return True

    rel_cache = {}                                    # cache per base dir
    ignored = False
    for pattern, negate, base in rules:
        rel = rel_cache.get(base)
        if rel is None:
            try:
                rel = path.relative_to(base).as_posix()
            except ValueError:                        # safety-net
                continue
            rel_cache[base] = rel

        if _matches(pattern, rel):
            ignored = not negate      # later rules override earlier ones

    return ignored

if __name__ == "__main__":
    print(is_ignored("data/0b5a2ed4-132d-4e39-83be-81bc045b0300.json"))
    