"""Workspace management for Automator.

The workspace feature groups agents and threads under a *workspace path*
and takes care of persisting their state to disk.  The public API mirrors
the usage examples in the project README::

    workspace = Workspace("my-project/sub-project", env={"FOO": "bar"})
    bash_agent = workspace.add_agent(agent=bash_agent, id="bash_agent")
    workspace.add_thread(thread=thread, id="My first thread")

Instances can later be rehydrated:

    workspace = Workspace("my-project/sub-project")
    agent  = workspace.get_agent("bash_agent")
    thread = workspace.get_thread("My first thread")

Implementation strategy
-----------------------
Agents and threads are serialised as JSON using the ``.json()`` helper
methods that already exist on ``Agent`` and ``Thread``.  Each workspace
has the following on-disk layout (all paths are *relative to* the
workspace directory):

* ``agents/<id>.json``   – saved agents
* ``threads/<id>.json``  – saved threads

Only *state* that can be reconstructed is persisted (e.g. open MCP
connections are intentionally **not** serialised).  When a thread is
loaded back it will be in its initial disconnected state and establish
fresh MCP connections the first time it is iterated over again.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional

from automator.agent import Agent , Thread
from automator.dtypes import ChatMessage


def _ensure_dir(path: Path) -> None:
    """Recursively create *path* if it does not yet exist."""

    path.mkdir(parents=True, exist_ok=True)


def _slugify(name: str) -> str:
    """Return *name* converted into a filename-friendly slug.

    The exact implementation does not have to be perfect – it is only
    meant to guarantee that we end up with a valid filename for the
    most common inputs.
    """

    # Replace os-specific path separators first so we do not accidentally
    # introduce nested paths.
    for sep in (os.sep, os.altsep):
        if sep:
            name = name.replace(sep, "-")

    # Very small whitelist, everything else becomes a dash.
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.")
    return "".join(c if c in allowed else "-" for c in name).strip("-_.")


###############################################################################
# Workspace implementation
###############################################################################


class Workspace:
    """A persistent workspace grouping together agents and threads."""

    ############################################################
    # Construction helpers
    ############################################################

    def __init__(self, name: str, env: Optional[Dict[str, str]] = None):
        """Create (or load) a workspace called *name*.

        Parameters
        ----------
        name:
            *Absolute* paths are taken verbatim.  If *name* is not an
            absolute path the workspace will be created under the
            *default workspace root* ``~/.automator/workspaces``.
        env:
            Optional environment overrides that will be *merged into*
            every agent attached to the workspace (i.e. values in
            ``env`` take precedence over an agent's existing ``env``
            mapping).
        """

        self._root_dir: Path = self._resolve_workspace_dir(name)
        
        # Make sure the directory tree exists so that subsequent save
        # operations cannot fail.  Falling back to a workspace inside the
        # *current working directory* when the preferred location is not
        # writable allows Automator to function in restricted execution
        # environments (e.g. read-only home directories).
        try:
            _ensure_dir(self._agent_dir)
            _ensure_dir(self._thread_dir)
        except PermissionError:
            # Replace root dir with a CWD-based fallback and try again.
            self._root_dir = Path.cwd() / ".automator_workspaces" / Path(name)
            _ensure_dir(self._agent_dir)
            _ensure_dir(self._thread_dir)

        self.env: Dict[str, str] = {'CWD': str(self._root_dir / 'workspace')}
        self.env.update(env)
        os.makedirs(self.env['CWD'], exist_ok=True)

    # ---------------------------------------------------------------------
    # Filesystem helpers
    # ---------------------------------------------------------------------

    @staticmethod
    def _resolve_workspace_dir(name: str) -> Path:
        """Return a fully-qualified path for *name*.

        The logic follows what was described above in ``__init__``.
        """

        p = Path(name).expanduser()
        if p.is_absolute():
            return p
        return Path.home() / ".automator" / "workspaces" / p

    # Directories ----------------------------------------------------------------

    @property
    def _agent_dir(self) -> Path:  # noqa: D401  (simple property)
        return self._root_dir / "agents"

    @property
    def _thread_dir(self) -> Path:  # noqa: D401
        return self._root_dir / "threads"

    # Filename helpers -----------------------------------------------------------

    def _agent_path(self, agent_id: str) -> Path:
        return self._agent_dir / f"{_slugify(agent_id)}.json"

    def _thread_path(self, thread_id: str) -> Path:
        return self._thread_dir / f"{_slugify(thread_id)}.json"

    ############################################################
    # Agent helpers
    ############################################################
    def register_agent(self, *, agent: Agent, id: str) -> None:
        """Persist *agent* under *id* (overwriting existing agents).
        The agent's environment is merged with the workspace's
        environment overrides.
        Unlike *add_agent* this method does not return a new agent
        instance but modifies the original one in place.
        """
        # Merge environments – workspace overrides agent.
        agent.env = {**agent.env, **self.env}
        agent.workspace = self
        self._save_json(self._agent_path(id), agent.json())

    def add_agent(self, *, agent: Agent, id: str) -> Agent:
        """Attach *agent* to the workspace under *id*.

        The workspace's environment overrides are merged **into** the
        agent (values already set on the agent win unless they are
        overwritten by *workspace.env*).
        """
        # Merge environments – workspace overrides agent.
        merged_env = {**agent.env, **self.env}

        # If nothing changed we can reuse the same instance.  Otherwise
        # create a shallow *copy* so the original remains untouched.
        if merged_env is agent.env:
            updated_agent = agent
        else:
            updated_agent = Agent(
                model=agent.model,
                prompt_template_yaml=agent.prompt_template_yaml,
                tools=list(agent.tools),
                env=merged_env,
                subagents=list(agent.subagents),
                as_tool=agent.as_tool,
                workspace=self,
            )

        self._save_json(self._agent_path(id), updated_agent.json())
        return updated_agent

    # ------------------------------------------------------------------
    def get_agent(self, id: str) -> Agent:
        """Load and return the agent registered under *id*."""

        path = self._agent_path(id)
        if not path.exists():
            raise KeyError(f"Agent '{id}' not found in workspace '{self._root_dir}'.")

        data = self._load_json(path)

        # Merge persisted env with current workspace overrides.  The
        # workspace should *always* win because it represents the user's
        # latest intention.
        env = {**data.get("env", {}), **self.env}

        return Agent(
            model=data["model"],
            prompt_template_yaml=data["prompt_template"],
            tools=data["tools"],
            env=env,
            subagents=data.get("subagents", []),
            as_tool=data.get("as_tool"),
            workspace=self,
        )

    # ------------------------------------------------------------------
    def list_agents(self, *, limit: int | None = None) -> List[str]:
        """Return a list of agent IDs currently stored in the workspace."""

        files = sorted(self._agent_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        ids = [f.stem for f in files]
        return ids[:limit] if limit is not None else ids

    ############################################################
    # Thread helpers
    ############################################################

    def add_thread(self, *, thread: Thread, id: str) -> None:
        """Persist *thread* under *id* (overwriting existing threads)."""
        # Ensure environment is up-to-date.
        thread.env = {**thread.env, **self.env}
        thread.workspace = self
        self._save_json(self._thread_path(id), thread.json())

    # ------------------------------------------------------------------
    def get_thread(self, id: str) -> Thread:
        """Load and return the thread saved under *id*."""

        path = self._thread_path(id)
        if not path.exists():
            raise KeyError(f"Thread '{id}' not found in workspace '{self._root_dir}'.")

        data = self._load_json(path)

        # ----- messages --------------------------------------------------
        messages = [ChatMessage(**m) for m in data["messages"]]

        # ----- reconstruct Thread ---------------------------------------
        env = {**data.get("env", {}), **self.env}

        thread = Thread(
            model=data["model"],
            tools=data["tools"],
            messages=messages,
            env=env,
            temperature=data.get("temperature", 0.7),
            max_tokens=data.get("max_tokens", 8000),
            workspace=self,
            subagents=data.get("subagents", []),
            id=id
        )

        # Load subagent threads
        for thread_id in data.get("thread_ids", []):
            thread._threads[thread_id] = self.get_thread(thread_id)

        return thread

    # ------------------------------------------------------------------
    def list_threads(self, *, limit: int | None = None) -> List[str]:
        """Return a list of thread IDs currently stored in the workspace."""

        files = sorted(self._thread_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        ids = [f.stem for f in files]
        return ids[:limit] if limit is not None else ids

    ############################################################
    # Persistence helpers (private)
    ############################################################

    @staticmethod
    def _save_json(path: Path, data: Dict) -> None:  # noqa: D401 (simple helper)
        """Write *data* as JSON to *path* (atomically, best effort)."""

        tmp = path.with_suffix(".json.tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=2)
        tmp.replace(path)

    @staticmethod
    def _load_json(path: Path) -> Dict:
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
