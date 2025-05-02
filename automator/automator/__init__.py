"""Top-level package exports.

We *only* re-export lightweight helpers here to avoid pulling in heavy
dependencies when users simply run ``import automator``.  The FastAPI app
is available through ``import automator.api`` if needed.
"""

# Re-export frequently used classes so users can simply ``import automator``.

from .workspace import Workspace  # noqa: F401 – re-export
