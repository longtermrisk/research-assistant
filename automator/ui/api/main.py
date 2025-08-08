import os
import asyncio
import logging
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple
from uuid import uuid4

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from isignored import is_ignored
from localrouter import (
    Base64ImageSource,
    ChatMessage,
    ContentBlock as InternalContentBlock,
    ImageBlock,
    MessageRole,
    TextBlock,
    ToolDefinition,
    providers
)

from automator.agent import Agent, Thread, _SERVERS
from automator.workspace import Workspace

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
logger = logging.getLogger("uvicorn")



# ---------------------------------------------------------------------------
# --- In-memory Thread Cache and helpers (unchanged) ---
# ---------------------------------------------------------------------------

active_threads: Dict[str, Tuple[Thread, asyncio.Lock]] = {}
_active_threads_dict_lock = asyncio.Lock()

async def get_or_prepare_thread_from_cache(workspace_name: str, thread_id: str, ws: Workspace) -> Thread:
    cache_key = f"{workspace_name}:{thread_id}"
    thread_instance: Optional[Thread] = None
    prepare_lock: Optional[asyncio.Lock] = None

    async with _active_threads_dict_lock:
        if cache_key in active_threads:
            thread_instance, prepare_lock = active_threads[cache_key]
        else:
            try:
                thread_instance = ws.get_thread(id=thread_id)
            except KeyError as exc:
                raise HTTPException(status_code=404, detail=f"Thread '{thread_id}' not found in workspace '{workspace_name}'.") from exc
            prepare_lock = asyncio.Lock()
            active_threads[cache_key] = (thread_instance, prepare_lock)

    if thread_instance and prepare_lock:
        if not thread_instance._ready:
            async with prepare_lock:
                if not thread_instance._ready:
                    try:
                        await thread_instance.prepare()
                    except Exception as e:
                        async with _active_threads_dict_lock:
                            if cache_key in active_threads and active_threads[cache_key][0] is thread_instance:
                                del active_threads[cache_key]
                        raise HTTPException(status_code=500, detail=f"Failed to prepare thread '{thread_id}': {str(e)}") from e
        return thread_instance
    else:
        raise HTTPException(status_code=500, detail="Internal server error: Could not retrieve thread for preparation.")

@app.on_event("shutdown")
async def app_shutdown():
    threads_to_cleanup = []
    async with _active_threads_dict_lock:
        for cache_key, (thread, _) in active_threads.items():
            threads_to_cleanup.append((cache_key, thread))
        active_threads.clear()
    for cache_key, thread in threads_to_cleanup:
        try:
            await thread.cleanup()
        except Exception as e:
            logger.error(f"Error cleaning up thread {cache_key}: {e}", exc_info=True)

def get_existing_workspace(workspace_name: str) -> Workspace:
    primary_path_check = Workspace._resolve_workspace_dir(workspace_name)
    if not (primary_path_check.exists() and primary_path_check.is_dir()):
        raise HTTPException(status_code=404, detail=f"Workspace '{workspace_name}' not found.")
    try:
        return Workspace(name=workspace_name)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not load workspace '{workspace_name}'. {str(e)}") from e

async def get_workspace_dependency(workspace_name: str) -> Workspace:
    return get_existing_workspace(workspace_name)

class Broadcaster:
    def __init__(self):
        self.queues: Dict[str, List[asyncio.Queue]] = {}
    async def subscribe(self, thread_id: str) -> asyncio.Queue:
        self.queues.setdefault(thread_id, [])
        queue = asyncio.Queue()
        self.queues[thread_id].append(queue)
        return queue
    def unsubscribe(self, thread_id: str, queue: asyncio.Queue):
        if thread_id in self.queues:
            try: self.queues[thread_id].remove(queue)
            except ValueError: pass
            if not self.queues[thread_id]: del self.queues[thread_id]
    async def broadcast(self, thread_id: str, message_json_str: str):
        if thread_id in self.queues:
            for q_item in self.queues[thread_id]: await q_item.put(message_json_str)
broadcaster = Broadcaster()

# --- Pydantic Models ---
class WorkspaceCreate(BaseModel):
    name: str
    env: Optional[Dict[str, str]] = None

class WorkspaceResponse(BaseModel):
    name: str
    path: str
    env: Dict[str, str]

class AgentCreate(BaseModel):
    id: str
    llm: Dict[str, Any]  # Changed from model: str to llm: Dict
    prompt_template_yaml: str
    tools: Optional[List[str]] = None
    env: Optional[Dict[str, str]] = None
    subagents: Optional[List[str]] = None
    as_tool: Optional[Dict[str, Any]] = None
    prompt_template_vars: Optional[Dict[str, Any]] = None

class AgentResponse(BaseModel):
    id: str
    llm: Dict[str, Any]  # Changed from model: str to llm: Dict
    prompt_template: str
    tools: List[str]
    env: Dict[str, str]
    subagents: List[str]
    as_tool: Optional[Dict[str, Any]] = None
    workspace_name: str
    prompt_template_vars: Optional[Dict[str, Any]] = None

class ApiContentBlock(BaseModel):
    type: str
    text: Optional[str] = None
    source: Optional[Dict[str, Any]] = None
    id: Optional[str] = None
    input: Optional[Dict[str, Any]] = None
    name: Optional[str] = None
    tool_use_id: Optional[str] = None
    content: Optional[List["ApiContentBlock"]] = None
    meta: Optional[Dict[str, Any]] = None

class ApiChatMessage(BaseModel):
    role: MessageRole
    content: List[ApiContentBlock]
    meta: Optional[Dict[str, Any]] = None
    @classmethod
    def from_chat_message(cls, chat_message: ChatMessage) -> "ApiChatMessage":
        api_content_blocks = [ApiContentBlock(**block.model_dump(exclude_none=True)) for block in chat_message.content]
        return cls(role=chat_message.role, content=api_content_blocks, meta=chat_message.meta)

class ThreadCreateRequest(BaseModel):
    agent_id: str
    initial_content: List[ApiContentBlock]
    thread_id: Optional[str] = None
    mentioned_file_paths: Optional[List[str]] = None # Added

class ThreadResponse(BaseModel):
    id: str
    llm: Dict[str, Any]  # Changed from model: str to llm: Dict
    tools: List[str]
    env: Dict[str, str]
    subagents: List[str]
    workspace_name: str
    initial_messages_count: int
    first_user_message_preview: Optional[str] = None
    agent_id: Optional[str] = None  # Added to track which agent created the thread

class ThreadDetailResponse(ThreadResponse):
    messages: List[ApiChatMessage]

class MessagePostRequest(BaseModel):
    content: List[ApiContentBlock]
    mentioned_file_paths: Optional[List[str]] = None # Added

class InterruptRequest(BaseModel):
    thread_id: str

# FileSystemItem for the new /files endpoint
class FileSystemItem(BaseModel):
    id: str
    name: str
    path: str # Relative to workspace CWD
    type: str # 'file' or 'folder'
    children: Optional[List["FileSystemItem"]] = None

class McpServerTools(BaseModel):
    server_name: str
    tools: List[ToolDefinition]


# --- Helper Functions ---
def get_default_workspaces_root() -> Path:
    return Path.home() / ".automator" / "workspaces"

def create_file_content_blocks(
    workspace_cwd: Path,
    mentioned_paths: List[str],
) -> List[InternalContentBlock]:
    """Creates hidden text blocks for mentioned files."""
    content_blocks: List[InternalContentBlock] = []
    for rel_path_str in mentioned_paths:
        if not rel_path_str or ".." in rel_path_str: # Basic security
            logger.warning(f"Skipping potentially unsafe mentioned path: {rel_path_str}")
            continue

        abs_file_path = (workspace_cwd / rel_path_str).resolve()

        # Security check: ensure the resolved path is still within the workspace CWD
        if workspace_cwd not in abs_file_path.parents and abs_file_path != workspace_cwd :
             # This check is a bit tricky if workspace_cwd is a file itself (it shouldn't be)
             # A more robust check:
            if not str(abs_file_path).startswith(str(workspace_cwd)):
                logger.warning(f"Skipping mentioned path outside workspace CWD: {rel_path_str} (resolved: {abs_file_path})")
                continue
        
        if not abs_file_path.is_file():
            logger.warning(f"Mentioned path is not a file or does not exist: {rel_path_str}")
            continue

        if is_ignored(abs_file_path):
            logger.info(f"Mentioned file is gitignored, skipping: {rel_path_str}")
            continue
        
        try:
            with open(abs_file_path, 'r', encoding='utf-8') as f:
                file_content = f.read()
                # Limit file content size to prevent overly large messages
                if len(file_content) > 100 * 1024: # 100KB limit
                    file_content = file_content[:100*1024] + "\n... (file truncated due to size)"
                    logger.warning(f"Truncated content for file: {rel_path_str}")
            
            # Use the relative path as provided by the frontend for the tag
            block_text = f'<file path="{rel_path_str}">\n{file_content}\n</file>'
            content_blocks.append(TextBlock(text=block_text, meta={"hidden": True}))
            logger.info(f"Added hidden content block for: {rel_path_str}")
        except Exception as e:
            logger.error(f"Error reading mentioned file {rel_path_str}: {e}")
            content_blocks.append(TextBlock(text=f'<file path="{rel_path_str}">\nError reading file: {str(e)}\n</file>', meta={"hidden": True}))
            
    return content_blocks

async def run_agent_turn(workspace: Workspace, thread: Thread, initial_run: bool = False):
    # (Existing run_agent_turn logic remains the same)
    logger.info(f"[run_agent_turn] Starting for thread '{thread.id}', initial_run: {initial_run}")
    try:
        if initial_run:
            logger.info(f"[run_agent_turn] Broadcasting {len(thread.messages)} initial messages for thread '{thread.id}'")
            for msg in thread.messages: 
                api_msg = ApiChatMessage.from_chat_message(msg)
                await broadcaster.broadcast(thread.id, api_msg.model_dump_json())
                await asyncio.sleep(0.01)

        async for message in thread: 
            logger.info(f"[run_agent_turn] Received message from agent for thread '{thread.id}': Role: {message.role}")
            api_msg = ApiChatMessage.from_chat_message(message)
            await broadcaster.broadcast(thread.id, api_msg.model_dump_json())
            await asyncio.sleep(0.01) 

    except Exception as e:
        logger.error(f"[run_agent_turn] Exception during agent processing for thread '{thread.id}': {e}", exc_info=True)
        error_text = f"An error occurred while processing your request with the agent: {str(e)}"
        # ... (rest of existing error handling) ...
        if hasattr(e, "message") and isinstance(getattr(e, "message"), str): 
            error_text = f"Agent API Error: {getattr(e, 'message')}"
        elif hasattr(e, "body") and isinstance(getattr(e, "body"), dict): 
            try:
                error_body = getattr(e, "body")
                if error_body and "error" in error_body and "message" in error_body["error"]:
                    error_text = f"Agent API Error: {error_body['error']['message']}"
            except Exception: 
                pass 
        error_message_content = [TextBlock(text=error_text)]
        error_chat_message = ChatMessage(role=MessageRole.assistant, content=error_message_content, meta={"error": True})
        thread.messages.append(error_chat_message)
        try:
            workspace.add_thread(thread=thread, id=thread.id)
        except Exception as save_err:
            logger.error(f"[run_agent_turn] Failed to save thread or markdown after agent error for thread '{thread.id}': {save_err}", exc_info=True)
        api_error_msg = ApiChatMessage.from_chat_message(error_chat_message)
        await broadcaster.broadcast(thread.id, api_error_msg.model_dump_json())
    finally:
        logger.info(f"[run_agent_turn] Finished for thread '{thread.id}'")
        try:
            workspace.add_thread(thread=thread, id=thread.id) 
        except Exception as final_save_err:
            logger.error(f"[run_agent_turn] Failed final save/markdown for thread '{thread.id}': {final_save_err}", exc_info=True)


# --- API Endpoints ---
@app.get("/")
async def root():
    return {"message": "Automator API"}


# ---------------------------------------------------------------------------
# Global endpoints: /models and /tools
# ---------------------------------------------------------------------------

@app.get("/models", response_model=List[str])
async def list_models_api():
    """Return all model identifiers visible through registered LLM providers."""
    model_set: set[str] = set()
    for prov in providers:
        model_set.update(prov.models)
    return sorted(model_set)

@app.get("/prompts", response_model=List[str])
async def list_prompts_api():
    """Return all prompt identifiers visible through registered LLM providers."""
    prompts = os.listdir(os.path.expanduser("~/.automator/prompts"))
    return sorted(prompts)

@app.get("/tools", response_model=List[McpServerTools])
async def list_tools_api():
    """Enumerate every tool exposed by every MCP server using the same discovery
    mechanism as Thread.prepare() but without side-effects.

    Returns a dictionary mapping server names to lists of tool definitions. Access the tool name via response[0].tools[0].name
    """
    if not _SERVERS:
        return []
    return [McpServerTools(server_name=mcp_server, tools=[]) for mcp_server in _SERVERS]

    # # Minimal message list required by Thread
    # placeholder_messages = [
    #     ChatMessage(role=MessageRole.system, content=[TextBlock(text="Tool discovery thread")])
    # ]

    # tools_data: List[McpServerTools] = [] # Changed from dict to list
    # for mcp_server in _SERVERS:
    #     thread = Thread(
    #         model="noop",
    #         messages=placeholder_messages,
    #         tools=[f"{mcp_server}.*"],
    #         env={'CWD': '/tmp'},
    #         subagents=[],
    #     )
    #     try:
    #         await thread.prepare()
    #         tool_definitions = [tool.definition for tool in thread.tools]
    #         tools_data.append(McpServerTools(server_name=mcp_server, tools=tool_definitions))
    #     except Exception as e:
    #         logger.error(f"Error preparing tools for MCP server {mcp_server}: {e}", exc_info=True)
    #         tools_data.append(McpServerTools(server_name=mcp_server, tools=[])) 
    #     finally:
    #         try:
    #             await thread.cleanup()
    #         except Exception:
    #             pass # Already logged if an error occurred during prepare.
    # return tools_data

@app.post("/workspaces", response_model=WorkspaceResponse, status_code=201)
async def create_workspace_api(workspace_data: WorkspaceCreate):
    try:
        ws = Workspace(name=workspace_data.name, env=workspace_data.env)
        return WorkspaceResponse(name=ws.name, path=str(ws._root_dir), env=ws.env)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create workspace '{workspace_data.name}': {str(e)}")

@app.get("/workspaces", response_model=List[WorkspaceResponse])
async def list_workspaces_api():
    results = []
    default_root = get_default_workspaces_root()
    if default_root.exists() and default_root.is_dir():
        for item in default_root.iterdir():
            if item.is_dir():
                try:
                    ws = Workspace(name=item.name)
                    results.append(WorkspaceResponse(name=ws.name, path=str(ws._root_dir), env=ws.env))
                except Exception as e:
                    logger.error(f"Error processing workspace item {item.name}: {e}", exc_info=True)
    return results

@app.get("/workspaces/{workspace_name}", response_model=WorkspaceResponse)
async def get_workspace_details_api(ws: Workspace = Depends(get_workspace_dependency)):
    return WorkspaceResponse(name=ws.name, path=str(ws._root_dir), env=ws.env)

@app.get("/workspaces/{workspace_name}/files", response_model=List[FileSystemItem])
async def list_workspace_files_api(ws: Workspace = Depends(get_workspace_dependency)):
    """Return a tree of files/directories under the workspace CWD, respecting `.gitignore`."""

    workspace_cwd = ws.root
    if not workspace_cwd.is_dir():
        raise HTTPException(
            status_code=404,
            detail=f"Workspace CWD not found or not a directory: {workspace_cwd}",
        )

    def build_file_tree(dir_abs: Path, base_abs: Path) -> List[FileSystemItem]:
        rel_dir = dir_abs.relative_to(base_abs)
        if rel_dir != Path(".") and is_ignored(Path(f"{dir_abs}{os.sep}")):
            # Whole sub-tree is ignored → stop here.
            logger.debug("Pruning ignored directory: %s", rel_dir)
            return []

        items: List[FileSystemItem] = []
        with os.scandir(dir_abs) as it:
            for entry in sorted(
                it,
                key=lambda e: (e.is_file(follow_symlinks=False), e.name.lower()),
            ):
                entry_abs = Path(entry.path)
                entry_rel = entry_abs.relative_to(base_abs)
                entry_is_dir = entry.is_dir(follow_symlinks=False)

                # Append '/' when checking a directory so patterns like 'node_modules/' match.
                ignored = (
                    is_ignored(Path(f"{entry_abs}{os.sep}"))
                    if entry_is_dir
                    else is_ignored(entry_abs)
                )
                if ignored:
                    continue

                fs_item = FileSystemItem(
                    id=str(entry_rel),
                    name=entry.name,
                    path=str(entry_rel),
                    type="folder" if entry_is_dir else "file",
                )

                if entry_is_dir:
                    fs_item.children = build_file_tree(entry_abs, base_abs)

                items.append(fs_item)

        return items

    # ------------------------------------------------------------------
    # Kick off traversal
    # ------------------------------------------------------------------
    items = build_file_tree(workspace_cwd, workspace_cwd)
    return items


@app.post("/workspaces/{workspace_name}/agents", response_model=AgentResponse, status_code=201)
async def create_agent_api(agent_data: AgentCreate, ws: Workspace = Depends(get_workspace_dependency)):
    as_tool_def = ToolDefinition(**agent_data.as_tool) if agent_data.as_tool else None
    agent = Agent(
        llm=agent_data.llm, prompt_template_yaml=agent_data.prompt_template_yaml,
        tools=agent_data.tools, env=agent_data.env, subagents=agent_data.subagents,
        as_tool=as_tool_def, prompt_template_vars=agent_data.prompt_template_vars
    )
    try:
        created_agent = ws.add_agent(agent=agent, id=agent_data.id)
        return AgentResponse(**created_agent.json(), workspace_name=ws.name, id=created_agent.id) # Use created_agent.id
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create agent: {str(e)}")

@app.get("/workspaces/{workspace_name}/agents", response_model=List[AgentResponse])
async def list_agents_api(ws: Workspace = Depends(get_workspace_dependency)):
    agents = []
    for aid in ws.list_agents():
        agent_json = ws.get_agent(id=aid).json()
        # No need to extract model from llm anymore, keep llm as is
        agents += [AgentResponse(**agent_json, workspace_name=ws.name, id=aid)]
    return agents

@app.get("/workspaces/{workspace_name}/agents/{agent_id}", response_model=AgentResponse)
async def get_agent_details_api(agent_id: str, ws: Workspace = Depends(get_workspace_dependency)):
    try:
        agent = ws.get_agent(id=agent_id)
        return AgentResponse(**agent.json(), workspace_name=ws.name, id=agent.id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Agent not found")


@app.post("/workspaces/{workspace_name}/threads", response_model=ThreadResponse, status_code=201)
async def create_thread_api(thread_data: ThreadCreateRequest, workspace_name: str, ws: Workspace = Depends(get_workspace_dependency)):
    try:
        agent = ws.get_agent(id=thread_data.agent_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Agent '{thread_data.agent_id}' not found.") from exc
    
    thread_id_to_use = thread_data.thread_id or uuid4().hex
    hidden_file_blocks: List[InternalContentBlock] = []
    if thread_data.mentioned_file_paths:
        workspace_cwd = Path(ws.env.get("CWD", str(ws._root_dir / "workspace"))).resolve()
        hidden_file_blocks = create_file_content_blocks(
            workspace_cwd,
            thread_data.mentioned_file_paths
        )

    internal_initial_content: List[InternalContentBlock] = []
    query_for_template: Optional[str] = None
    if thread_data.initial_content:
        for api_block in thread_data.initial_content:
            if api_block.type == "text" and api_block.text is not None:
                block = TextBlock(text=api_block.text, meta=api_block.meta or {})
                internal_initial_content.append(block)
                if not query_for_template: query_for_template = api_block.text
            elif api_block.type == "image" and api_block.source:
                img_src = Base64ImageSource(**api_block.source)
                internal_initial_content.append(ImageBlock(source=img_src, meta=api_block.meta or {}))
    if not internal_initial_content:
        raise HTTPException(status_code=400, detail="Initial content cannot be empty.")

    final_initial_content = hidden_file_blocks + internal_initial_content
    thread = await agent.run(
        query=query_for_template, # Still used for prompt template if needed
        initial_user_content=final_initial_content, # This now includes hidden file blocks + user visible blocks
        thread_id=thread_id_to_use
    )
    ws.add_thread(thread=thread, id=thread.id)
    async with _active_threads_dict_lock:
        if f"{ws.name}:{thread.id}" not in active_threads:
             active_threads[f"{ws.name}:{thread.id}"] = (thread, asyncio.Lock())
    
    asyncio.create_task(run_agent_turn(ws, thread, initial_run=True))
    
    return ThreadResponse(
        id=thread.id, llm=thread.llm, tools=thread._tools, env=thread.env, subagents=thread.subagents,
        workspace_name=ws.name, initial_messages_count=len(thread.messages),
        first_user_message_preview=thread.get_first_user_message_preview(),
        agent_id=thread_data.agent_id
    )

@app.get("/workspaces/{workspace_name}/threads", response_model=List[ThreadResponse])
async def list_threads_api(ws: Workspace = Depends(get_workspace_dependency)):
    thread_ids = ws.list_threads()
    thread_responses = []
    for t_id in thread_ids:
        try:
            thread_preview_instance = ws.get_thread(id=t_id)
            first_user_message_preview = thread_preview_instance.get_first_user_message_preview()
            thread_responses.append(ThreadResponse(
                id=thread_preview_instance.id, llm=thread_preview_instance.llm, 
                tools=thread_preview_instance._tools, 
                env=thread_preview_instance.env,
                subagents=thread_preview_instance.subagents, workspace_name=ws.name, 
                initial_messages_count=len(thread_preview_instance.messages),
                first_user_message_preview=first_user_message_preview,
                agent_id=thread_preview_instance.agent_id
            ))
        except Exception as e:
            logger.error(f"Error loading thread {t_id} in {ws.name} for listing: {e}", exc_info=True)
    return thread_responses


@app.get("/workspaces/{workspace_name}/threads/{thread_id}", response_model=ThreadDetailResponse)
async def get_thread_details_api(workspace_name: str, thread_id: str, ws: Workspace = Depends(get_workspace_dependency)):
    thread = await get_or_prepare_thread_from_cache(workspace_name, thread_id, ws)
    api_messages = [ApiChatMessage.from_chat_message(msg) for msg in thread.messages]
    first_user_message_preview = thread.get_first_user_message_preview()
    return ThreadDetailResponse(
        id=thread.id, llm=thread.llm, tools=thread._tools, env=thread.env, subagents=thread.subagents,
        workspace_name=ws.name, messages=api_messages, 
        initial_messages_count=len(thread.messages),
        first_user_message_preview=first_user_message_preview,
        agent_id=thread.agent_id
    )


@app.post("/workspaces/{workspace_name}/threads/{thread_id}/messages", response_model=ApiChatMessage)
async def post_message_api(workspace_name: str, thread_id: str, message_data: MessagePostRequest, ws: Workspace = Depends(get_workspace_dependency)):
    thread = await get_or_prepare_thread_from_cache(workspace_name, thread_id, ws)

    hidden_file_blocks: List[InternalContentBlock] = []
    if message_data.mentioned_file_paths:
        workspace_cwd = Path(ws.env.get("CWD", str(ws._root_dir / "workspace"))).resolve()
        hidden_file_blocks = create_file_content_blocks(
            workspace_cwd,
            message_data.mentioned_file_paths
        )

    internal_content_blocks: List[InternalContentBlock] = []
    for api_block in message_data.content:
        if api_block.type == "text" and api_block.text is not None:
            internal_content_blocks.append(TextBlock(text=api_block.text, meta=api_block.meta or {}))
        elif api_block.type == "image" and api_block.source:
            img_source = Base64ImageSource(**api_block.source)
            internal_content_blocks.append(ImageBlock(source=img_source, meta=api_block.meta or {}))
    if not internal_content_blocks and not hidden_file_blocks: # Message can be just hidden files
        raise HTTPException(status_code=400, detail="Message content cannot be empty if no files are mentioned.")

    final_user_content = hidden_file_blocks + internal_content_blocks
    
    await thread.run(user_content=final_user_content)
    
    user_message_to_return = thread.messages[-1] if thread.messages and thread.messages[-1].role == MessageRole.user else \
                             ChatMessage(role=MessageRole.user, content=[TextBlock(text="Error: User message not found after run")])

    ws.add_thread(thread=thread, id=thread.id)

    api_user_msg = ApiChatMessage.from_chat_message(user_message_to_return)
    await broadcaster.broadcast(thread.id, api_user_msg.model_dump_json())
    
    asyncio.create_task(run_agent_turn(ws, thread, initial_run=False))
    
    return api_user_msg

@app.get("/workspaces/{workspace_name}/threads/{thread_id}/messages/sse")
async def sse_messages_api(request: Request, workspace_name: str, thread_id: str, ws: Workspace = Depends(get_workspace_dependency)):
    await get_or_prepare_thread_from_cache(workspace_name, thread_id, ws)
    queue = await broadcaster.subscribe(thread_id)
    async def event_generator() -> AsyncGenerator[str, None]:
        try:
            while True:
                if await request.is_disconnected():
                    logger.info(f"Client disconnected from SSE for thread {thread_id}")
                    break
                message_json_str = await queue.get()
                print(f"Broadcasting message to SSE: {message_json_str}")
                yield f"data: {message_json_str}\n\n"
                queue.task_done()
        except asyncio.CancelledError:
            logger.info(f"SSE for thread {thread_id} cancelled.")
        finally:
            broadcaster.unsubscribe(thread_id, queue)
            logger.info(f"SSE connection closed for thread {thread_id}")
    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.post("/interrupt")
async def interrupt_thread_api(interrupt_data: InterruptRequest):
    """Interrupt a running thread by calling thread.interrupt()."""
    thread_id = interrupt_data.thread_id
    
    # Find the thread in active_threads cache
    thread_found = False
    async with _active_threads_dict_lock:
        for cache_key, (thread, _) in active_threads.items():
            if thread.id == thread_id:
                thread_found = True
                try:
                    thread.interrupt()
                    logger.info(f"Successfully interrupted thread {thread_id}")
                    return {"message": f"Thread {thread_id} interrupted successfully"}
                except Exception as e:
                    logger.error(f"Error interrupting thread {thread_id}: {e}", exc_info=True)
                    raise HTTPException(status_code=500, detail=f"Failed to interrupt thread {thread_id}: {str(e)}")
                break
    
    if not thread_found:
        raise HTTPException(status_code=404, detail=f"Thread {thread_id} not found in active threads")