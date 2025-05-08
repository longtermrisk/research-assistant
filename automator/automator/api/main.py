import asyncio
import json # For creating error message content
import logging
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional
from uuid import uuid4

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from automator.agent import Agent, Thread
from automator.dtypes import (
    Base64ImageSource,
    ChatMessage,
    ContentBlock as InternalContentBlock, # Renamed to avoid clash
    ImageBlock,
    MessageRole,
    TextBlock,
    ToolDefinition,
)
from automator.workspace import Workspace

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
logger = logging.getLogger("uvicorn")
logger.error(
    "AUTOMATOR.API.MAIN.PY HAS BEEN RELOADED/IMPORTED (v6 with lint fixes and MessagePostRequest update)"
)

# --- Helper Function to Get Existing Workspace ---
def get_existing_workspace(workspace_name: str) -> Workspace:
    logger.info(f"[get_existing_workspace] Called for: '{workspace_name}'")
    primary_path_check = Workspace._resolve_workspace_dir(workspace_name)
    logger.info(f"[get_existing_workspace] Resolved primary path: {primary_path_check}")
    path_exists = primary_path_check.exists()
    is_dir = primary_path_check.is_dir() if path_exists else False
    logger.info(f"[get_existing_workspace] Path exists: {path_exists}, Is dir: {is_dir}")
    if not path_exists or not is_dir:
        logger.warning(
            f"[get_existing_workspace] Workspace path {primary_path_check} not found or not a dir. Raising 404."
        )
        raise HTTPException(
            status_code=404, detail=f"Workspace '{workspace_name}' not found at expected location {primary_path_check}."
        )
    logger.info(f"[get_existing_workspace] Path exists and is dir. Proceeding to load Workspace('{workspace_name}')")
    try:
        ws = Workspace(name=workspace_name)
        logger.info(f"[get_existing_workspace] Successfully loaded Workspace('{workspace_name}') with CWD: {ws.env.get('CWD')}")
        return ws
    except Exception as e:
        logger.error(f"[get_existing_workspace] Error loading workspace '{workspace_name}' after path check: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Could not load workspace '{workspace_name}'. {str(e)}") from e

async def get_workspace_dependency(workspace_name: str) -> Workspace:
    logger.info(f"[get_workspace_dependency] Called for: '{workspace_name}'")
    return get_existing_workspace(workspace_name)

# --- In-memory SSE Broadcaster ---
class Broadcaster:
    def __init__(self):
        self.queues: Dict[str, List[asyncio.Queue]] = {}
    async def subscribe(self, thread_id: str) -> asyncio.Queue:
        if thread_id not in self.queues: 
            self.queues[thread_id] = []
        queue = asyncio.Queue()
        self.queues[thread_id].append(queue)
        return queue
    def unsubscribe(self, thread_id: str, queue: asyncio.Queue):
        if thread_id in self.queues:
            self.queues[thread_id].remove(queue)
            if not self.queues[thread_id]:
                del self.queues[thread_id]
    async def broadcast(self, thread_id: str, message_json_str: str):
        if thread_id in self.queues:
            for q_item in self.queues[thread_id]:
                await q_item.put(message_json_str)
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
    model: str
    prompt_template_yaml: str
    tools: Optional[List[str]] = None
    env: Optional[Dict[str, str]] = None
    subagents: Optional[List[str]] = None
    as_tool: Optional[Dict[str, Any]] = None
    prompt_template_vars: Optional[Dict[str, Any]] = None

class AgentResponse(BaseModel):
    id: str
    model: str
    prompt_template_yaml: str
    tools: List[str]
    env: Dict[str, str]
    subagents: List[str]
    as_tool: Optional[Dict[str, Any]] = None
    workspace_name: str
    prompt_template_vars: Optional[Dict[str, Any]] = None

# ApiContentBlock for request/response, maps to internal dtypes.ContentBlock
class ApiContentBlock(BaseModel):
    type: str
    text: Optional[str] = None
    source: Optional[Dict[str, Any]] = None  # For ImageBlock: {data: str, media_type: str, type: "base64"}
    # Fields for ToolUseBlock
    id: Optional[str] = None
    input: Optional[Dict[str, Any]] = None
    name: Optional[str] = None
    # Fields for ToolResultBlock
    tool_use_id: Optional[str] = None
    content: Optional[List["ApiContentBlock"]] = None  # Recursive for ToolResultBlock's content
    meta: Optional[Dict[str, Any]] = None

class ApiChatMessage(BaseModel):
    role: MessageRole
    content: List[ApiContentBlock]
    meta: Optional[Dict[str, Any]] = None

    @classmethod
    def from_chat_message(cls, chat_message: ChatMessage) -> "ApiChatMessage":
        api_content_blocks = []
        for block in chat_message.content:
            block_dict = block.model_dump(exclude_none=True)
            api_content_blocks.append(ApiContentBlock(**block_dict))
        return cls(role=chat_message.role, content=api_content_blocks, meta=chat_message.meta)

class ThreadCreateRequest(BaseModel): # Stays as initial_query: str for now
    agent_id: str
    initial_query: str
    thread_id: Optional[str] = None

class ThreadResponse(BaseModel):
    id: str
    model: str
    tools: List[str]
    env: Dict[str, str]
    subagents: List[str]
    workspace_name: str
    initial_messages_count: int

class ThreadDetailResponse(ThreadResponse):
    messages: List[ApiChatMessage]

class MessagePostRequest(BaseModel):
    content: List[ApiContentBlock] # Changed from query: str

# --- Helper Functions ---
def get_default_workspaces_root() -> Path:
    return Path.home() / ".automator" / "workspaces"

async def run_agent_turn(workspace: Workspace, thread: Thread, initial_run: bool = False):
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
        if hasattr(e, "message"): # For openai.APIError and subclasses
            error_text = f"Agent API Error: {e.message}"
        elif hasattr(e, "body"): # For some openai errors, body contains JSON info
            try:
                error_body = getattr(e, "body")
                if error_body and "error" in error_body and "message" in error_body["error"]:
                    error_text = f"Agent API Error: {error_body['error']['message']}"
            except Exception: # pylint: disable=broad-except
                pass # Ignore parsing errors for the body

        error_message_content = [TextBlock(text=error_text)]
        error_chat_message = ChatMessage(role=MessageRole.assistant, content=error_message_content, meta={"error": True})
        thread.messages.append(error_chat_message)
        try:
            workspace.add_thread(thread=thread, id=thread.id)
            thread.to_markdown()
        except Exception as save_err:
            logger.error(f"[run_agent_turn] Failed to save thread or markdown after agent error for thread '{thread.id}': {save_err}", exc_info=True)
        api_error_msg = ApiChatMessage.from_chat_message(error_chat_message)
        await broadcaster.broadcast(thread.id, api_error_msg.model_dump_json())
    finally:
        logger.info(f"[run_agent_turn] Finished for thread '{thread.id}'")
        try:
            workspace.add_thread(thread=thread, id=thread.id)
            thread.to_markdown()
        except Exception as final_save_err:
            logger.error(f"[run_agent_turn] Failed final save/markdown for thread '{thread.id}': {final_save_err}", exc_info=True)

# --- API Endpoints ---
@app.get("/")
async def root():
    return {"message": "Automator API"}

@app.post("/workspaces", response_model=WorkspaceResponse, status_code=201)
async def create_workspace_api(workspace_data: WorkspaceCreate):
    try:
        ws = Workspace(name=workspace_data.name, env=workspace_data.env)
        return WorkspaceResponse(name=ws.name, path=str(ws._root_dir), env=ws.env)
    except Exception as e:
        logger.error(f"Error creating workspace '{workspace_data.name}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to create workspace '{workspace_data.name}': {str(e)}") from e

@app.get("/workspaces", response_model=List[WorkspaceResponse])
async def list_workspaces_api():
    results = []
    default_root = get_default_workspaces_root()
    logger.info(f"[list_workspaces_api] Checking for workspaces in: {default_root}")
    if default_root.exists() and default_root.is_dir():
        logger.info(f"[list_workspaces_api] Workspace root exists: {default_root}")
        for item in default_root.iterdir():
            logger.info(f"[list_workspaces_api] Found item: {item}, is_dir: {item.is_dir()}")
            if item.is_dir():
                try:
                    logger.info(f"[list_workspaces_api] Processing workspace directory: {item.name}")
                    ws = Workspace(name=item.name)
                    results.append(WorkspaceResponse(name=ws.name, path=str(ws._root_dir), env=ws.env))
                    logger.info(f"[list_workspaces_api] Successfully processed and added workspace: {ws.name}")
                except Exception as e:
                    logger.error(f"[list_workspaces_api] Error processing workspace item {item.name}: {e}", exc_info=True)
    else:
        logger.warning(f"[list_workspaces_api] Workspace root does not exist or not a directory: {default_root}")
    return results

@app.get("/workspaces/{workspace_name}", response_model=WorkspaceResponse)
async def get_workspace_details_api(ws: Workspace = Depends(get_workspace_dependency)):
    return WorkspaceResponse(name=ws.name, path=str(ws._root_dir), env=ws.env)

@app.post("/workspaces/{workspace_name}/agents", response_model=AgentResponse, status_code=201)
async def create_agent_api(agent_data: AgentCreate, ws: Workspace = Depends(get_workspace_dependency)):
    as_tool_definition_local = ToolDefinition(**agent_data.as_tool) if agent_data.as_tool else None
    agent_to_create = Agent(
        model=agent_data.model, prompt_template_yaml=agent_data.prompt_template_yaml,
        tools=agent_data.tools, env=agent_data.env, subagents=agent_data.subagents,
        as_tool=as_tool_definition_local, prompt_template_vars=agent_data.prompt_template_vars
    )
    try:
        created_agent = ws.add_agent(agent=agent_to_create, id=agent_data.id)
        return AgentResponse(
            id=created_agent.id, model=created_agent.model, prompt_template_yaml=created_agent.prompt_template_yaml,
            tools=created_agent.tools, env=created_agent.env, subagents=created_agent.subagents,
            as_tool=created_agent.as_tool, workspace_name=ws.name, prompt_template_vars=created_agent.prompt_template_vars
        )
    except Exception as e:
        logger.error(f"Error creating agent {agent_data.id} in {ws.name}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to create agent '{agent_data.id}': {str(e)}") from e

@app.get("/workspaces/{workspace_name}/agents", response_model=List[AgentResponse])
async def list_agents_api(ws: Workspace = Depends(get_workspace_dependency)):
    agent_ids = ws.list_agents()
    agents_details = []
    for agent_id_val in agent_ids:
        try:
            agent = ws.get_agent(id=agent_id_val)
            agents_details.append(AgentResponse(
                id=agent.id, model=agent.model, prompt_template_yaml=agent.prompt_template_yaml,
                tools=agent.tools, env=agent.env, subagents=agent.subagents, as_tool=agent.as_tool,
                workspace_name=ws.name, prompt_template_vars=getattr(agent, 'prompt_template_vars', None)
            ))
        except Exception as e:
            logger.error(f"Error loading agent {agent_id_val} in workspace {ws.name}: {e}", exc_info=True)
    return agents_details

@app.get("/workspaces/{workspace_name}/agents/{agent_id}", response_model=AgentResponse)
async def get_agent_details_api(agent_id: str, ws: Workspace = Depends(get_workspace_dependency)):
    try:
        agent = ws.get_agent(id=agent_id)
        return AgentResponse(
            id=agent.id, model=agent.model, prompt_template_yaml=agent.prompt_template_yaml,
            tools=agent.tools, env=agent.env, subagents=agent.subagents, as_tool=agent.as_tool,
            workspace_name=ws.name, prompt_template_vars=getattr(agent, 'prompt_template_vars', None)
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_id}' not found in workspace '{ws.name}'.") from exc
    except Exception as e:
        logger.error(f"Error getting agent details for {agent_id} in {ws.name}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e

@app.post("/workspaces/{workspace_name}/threads", response_model=ThreadResponse, status_code=201)
async def create_thread_api(thread_data: ThreadCreateRequest, ws: Workspace = Depends(get_workspace_dependency)):
    try:
        agent = ws.get_agent(id=thread_data.agent_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Agent '{thread_data.agent_id}' not found in workspace '{ws.name}'.") from exc
    
    thread_id_to_use = thread_data.thread_id or uuid4().hex
    thread = await agent.run(query=thread_data.initial_query, thread_id=thread_id_to_use)
    ws.add_thread(thread=thread, id=thread.id) # Thread ID is already set in agent.run
    thread.to_markdown()

    asyncio.create_task(run_agent_turn(ws, thread, initial_run=True))
    return ThreadResponse(
        id=thread.id, model=thread.model, tools=thread._tools, env=thread.env, subagents=thread.subagents,
        workspace_name=ws.name, initial_messages_count=len(thread.messages)
    )

@app.get("/workspaces/{workspace_name}/threads", response_model=List[ThreadResponse])
async def list_threads_api(ws: Workspace = Depends(get_workspace_dependency)):
    thread_ids = ws.list_threads()
    thread_responses = []
    for t_id in thread_ids:
        try:
            thread = ws.get_thread(id=t_id)
            thread_responses.append(ThreadResponse(
                id=thread.id, model=thread.model, tools=thread._tools, env=thread.env,
                subagents=thread.subagents, workspace_name=ws.name, initial_messages_count=len(thread.messages)
            ))
        except Exception as e:
            logger.error(f"Error loading thread {t_id} in {ws.name}: {e}", exc_info=True)
    return thread_responses

@app.get("/workspaces/{workspace_name}/threads/{thread_id}", response_model=ThreadDetailResponse)
async def get_thread_details_api(thread_id: str, ws: Workspace = Depends(get_workspace_dependency)):
    try:
        thread = ws.get_thread(id=thread_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Thread '{thread_id}' not found in workspace '{ws.name}'.") from exc
    api_messages = [ApiChatMessage.from_chat_message(msg) for msg in thread.messages]
    return ThreadDetailResponse(
        id=thread.id, model=thread.model, tools=thread._tools, env=thread.env, subagents=thread.subagents,
        workspace_name=ws.name, messages=api_messages, initial_messages_count=len(thread.messages)
    )

@app.post("/workspaces/{workspace_name}/threads/{thread_id}/messages", response_model=ApiChatMessage)
async def post_message_api(thread_id: str, message_data: MessagePostRequest, ws: Workspace = Depends(get_workspace_dependency)):
    try:
        thread = ws.get_thread(id=thread_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Thread '{thread_id}' not found in workspace '{ws.name}'.") from exc

    internal_content_blocks: List[InternalContentBlock] = []
    for api_block in message_data.content:
        if api_block.type == "text" and api_block.text is not None:
            internal_content_blocks.append(TextBlock(text=api_block.text, meta=api_block.meta))
        elif api_block.type == "image" and api_block.source is not None:
            if not all(k in api_block.source for k in ('data', 'media_type', 'type')):
                raise HTTPException(
                    status_code=400, 
                    detail=f"Image block source is missing required fields (data, media_type, type). Received: {api_block.source}"
                )
            if api_block.source['type'] != 'base64':
                raise HTTPException(
                    status_code=400, 
                    detail=f"Image block source type must be 'base64'. Received: {api_block.source['type']}"
                )
            img_source = Base64ImageSource(
                data=api_block.source['data'],
                media_type=api_block.source['media_type'],
                type=api_block.source['type']
            )
            internal_content_blocks.append(ImageBlock(source=img_source, meta=api_block.meta))
        # Not expecting tool_use or tool_result from user directly here.

    if not internal_content_blocks:
        raise HTTPException(status_code=400, detail="Message content cannot be empty.")

    user_message = ChatMessage(role=MessageRole.user, content=internal_content_blocks)
    thread.messages.append(user_message)

    ws.add_thread(thread=thread, id=thread.id)
    thread.to_markdown()

    api_user_msg = ApiChatMessage.from_chat_message(user_message)
    await broadcaster.broadcast(thread.id, api_user_msg.model_dump_json())

    if not thread._ready: # Accessing protected member, consider a public method or property
        await thread.prepare()
    asyncio.create_task(run_agent_turn(ws, thread, initial_run=False))
    return api_user_msg

@app.get("/workspaces/{workspace_name}/threads/{thread_id}/messages/sse")
async def sse_messages_api(request: Request, thread_id: str, ws: Workspace = Depends(get_workspace_dependency)):
    try:
        _ = ws.get_thread(id=thread_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Thread '{thread_id}' not found in workspace '{ws.name}'.") from exc
    
    queue = await broadcaster.subscribe(thread_id)
    async def event_generator() -> AsyncGenerator[str, None]:
        try:
            while True:
                if await request.is_disconnected():
                    logger.info(f"Client disconnected from SSE for thread {thread_id}")
                    break
                message_json_str = await queue.get()
                yield f"data: {message_json_str}\n\n"
                queue.task_done()
        except asyncio.CancelledError:
            logger.info(f"SSE for thread {thread_id} cancelled.")
        finally:
            broadcaster.unsubscribe(thread_id, queue)
            logger.info(f"SSE connection closed for thread {thread_id}")
    return StreamingResponse(event_generator(), media_type="text/event-stream")