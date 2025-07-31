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
import logging
from pathlib import Path
from typing import Dict, List, Optional
import subprocess

from automator.agent import Agent , Thread
from automator.dtypes import ChatMessage

logger = logging.getLogger("uvicorn") # Or your preferred logger

def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)

def _slugify(name: str) -> str:
    for sep in (os.sep, os.altsep):
        if sep:
            name = name.replace(sep, "-")
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.")
    return "".join(c if c in allowed else "-" for c in name).strip("-_.")


class Workspace:
    def __init__(self, name: str, env: Optional[Dict[str, Any]] = None):
        self.name = name # Store the original name
        self._root_dir: Path = self._resolve_workspace_dir(name)
        
        try:
            _ensure_dir(self._agent_dir)
            _ensure_dir(self._thread_dir)
        except PermissionError:
            # Fallback if default location is not writable
            self._root_dir = Path.cwd() / ".automator_workspaces" / Path(name).name # Use original name for fallback path part
            _ensure_dir(self._agent_dir)
            _ensure_dir(self._thread_dir)

        self.env: Dict[str, str] = {'CWD': str(self._root_dir / 'workspace')}

        try:
            with open(self._root_dir / "env.json", "r") as f:
                env_data = json.load(f)
                self.env.update(env_data)
        except FileNotFoundError:
            pass

        if env is not None:
            self.env.update(env)
        
        # Save the env
        with open(self._root_dir / "env.json", "w") as f:
            json.dump(self.env, f, ensure_ascii=False, indent=2)
            
        # Ensure the workspace directory exists
        self.root = Path(self.env['CWD'])
        os.makedirs(self.root, exist_ok=True)
    
    @staticmethod
    def list_workspaces() -> List[str]:
        workspaces = []
        for path in Path.home().glob(".automator/workspaces/*"):
            if path.is_dir():
                workspaces.append(path.name)
        return workspaces

    @staticmethod
    def _resolve_workspace_dir(name: str) -> Path:
        p = Path(name).expanduser()
        if p.is_absolute():
            return p
        return Path.home() / ".automator" / "workspaces" / p

    @property
    def _agent_dir(self) -> Path:
        return self._root_dir / "agents"

    @property
    def _thread_dir(self) -> Path:
        return self._root_dir / "threads"

    def _agent_path(self, agent_id: str) -> Path:
        return self._agent_dir / f"{_slugify(agent_id)}.json"

    def _thread_path(self, thread_id: str) -> Path:
        return self._thread_dir / f"{_slugify(thread_id)}.json"

    def register_agent(self, *, agent: Agent, id: str) -> None:
        agent.env = {**self.env, **agent.env}
        agent.workspace = self # Assign the workspace instance
        agent.id = id # Ensure agent has its ID
        self._save_json(self._agent_path(id), agent.json())

    def add_agent(self, *, agent: Agent, id: str) -> Agent:
        merged_env = {**self.env, **agent.env} # agent env overrides workspace env on conflict
        
        # Create a new Agent instance to ensure it's clean and has the merged_env
        # and correct workspace association and ID.
        updated_agent = Agent(
            model=agent.model,
            prompt_template_yaml=agent.prompt_template_yaml,
            tools=list(agent.tools) if agent.tools else [],
            env=merged_env,
            subagents=list(agent.subagents) if agent.subagents else [],
            as_tool=agent.as_tool, # This should be a dict or ToolDefinition model
            workspace=self, # Assign this workspace instance
            id=id,
            prompt_template_vars=getattr(agent, 'prompt_template_vars', None) 
        )
        self._save_json(self._agent_path(id), updated_agent.json())
        return updated_agent

    def get_agent(self, id: str) -> Agent:
        path = self._agent_path(id)
        if not path.exists():
            raise KeyError(f"Agent '{id}' not found in workspace '{self.name}'.")
        data = self._load_json(path)
        
        # The env stored in agent JSON is already merged by add_agent.
        # For get_agent, we want to reflect that persisted state, potentially re-applying current workspace.env overrides.
        # The current workspace instance's self.env should take precedence for CWD or any live overrides.
        persisted_agent_env = data.get("env", {})
        final_env = {**self.env, **persisted_agent_env}

        loaded_agent = Agent(
            model=data["model"],
            prompt_template_yaml=data["prompt_template"],
            tools=data.get("tools", []),
            env=final_env,
            subagents=data.get("subagents", []),
            as_tool=data.get("as_tool"),
            workspace=self, # Assign this workspace instance
            id=id,
            prompt_template_vars=data.get("prompt_template_vars")
        )
        return loaded_agent

    def list_agents(self, *, limit: int | None = None) -> List[str]:
        if not self._agent_dir.exists(): return []
        files = sorted(self._agent_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        ids = [f.stem for f in files]
        return ids[:limit] if limit is not None else ids

    def add_thread(self, *, thread: Thread, id: str) -> None:
        thread.env = {**self.env, **thread.env} # Merge with current workspace env
        thread.workspace = self
        thread.id = id
        self._save_json(self._thread_path(id), thread.json())

    def get_thread(self, id: str) -> Thread:
        path = self._thread_path(id)
        if not path.exists():
            raise KeyError(f"Thread '{id}' not found in workspace '{self.name}'.")
        data = self._load_json(path)
        messages = [ChatMessage(**m) for m in data["messages"]]
        
        persisted_thread_env = data.get("env", {})
        final_env = {**self.env, **persisted_thread_env}

        thread_obj = Thread(
            model=data["model"],
            tools=data.get("tools", []),
            messages=messages,
            env=final_env,
            temperature=data.get("temperature", 0.7),
            max_tokens=data.get("max_tokens", 8000),
            workspace=self,
            subagents=data.get("subagents", []),
            id=id
        )
        for thread_id_sub in data.get("thread_ids", []):
            try:
                thread_obj._threads[thread_id_sub] = self.get_thread(thread_id_sub)
            except KeyError:
                logger.warning(f"Sub-thread {thread_id_sub} not found while loading thread {id} in workspace {self.name}")
        return thread_obj

    def list_threads(self, *, limit: int | None = None) -> List[str]:
        if not self._thread_dir.exists(): return []
        files = sorted(self._thread_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        ids = [f.stem for f in files]
        return ids[:limit] if limit is not None else ids

    @staticmethod
    def _save_json(path: Path, data: Dict) -> None:
        tmp = path.with_suffix(".json.tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=2)
        tmp.replace(path)

    @staticmethod
    def _load_json(path: Path) -> Dict:
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)