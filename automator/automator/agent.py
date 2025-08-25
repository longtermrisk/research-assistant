from typing import List, Dict, Tuple, Any, Union
import asyncio
from typing import Optional
from contextlib import AsyncExitStack
from uuid import uuid4
import json
import logging
from pathlib import Path
import os

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.sse import sse_client

from dotenv import load_dotenv

from localrouter import (
    ContentBlock,
    TextBlock,
    ImageBlock,
    ToolDefinition,
    SubagentToolDefinition,
    ToolUseBlock,
    ToolResultBlock,
    PromptTemplate,
    ChatMessage,
    MessageRole,
    get_response
    # get_response_with_backoff as get_response
)
from automator.hooks import load_hooks, _HOOKS

logger = logging.getLogger("uvicorn")
load_dotenv()
load_hooks()


def load_json(path):
    with open(os.path.expanduser(path), 'r') as f:
        return json.load(f)

_SERVERS = load_json("~/mcp.json").get('mcpServers', {})

_ALIAS = {
    'default': 'claude-sonnet-4-20250514',
    'gpt-5':  'gpt-5',
    'opus': 'claude-opus-4-1-20250805',
    'sonnet': 'claude-sonnet-4-20250514',
    'gemini': 'gemini-2.5-pro'
}


class Agent:
    def __init__(
        self,
        llm: Dict[str, Any], # LLM configuration dict for get_response
        prompt_template_yaml: str = None, # Path to the prompt template YAML file
        tools: List[str] = None, # List of tool names to use in format "{server_id}.{tool_name}" or "{server_id}.*"
        env: Dict[str, str] = None, # Environment variables used for MCP servers
        subagents: None | List[str] = None, # List of subagent names to use
        as_tool: ToolDefinition | None = None, # Tool definition if the agent is used as subagent
        workspace: Optional[str] = None, # Workspace to register the agent in
        id: Optional[str] = None, # ID of the agent in the workspace
        prompt_template_vars: Optional[Dict[str, Any]] = None, # Default values to use in prompt_template.apply()
        hooks: Optional[List[str]] = None, # List of hooks to use
    ):
        self.llm = llm
        self.prompt_template_yaml = prompt_template_yaml
        self.prompt_template = PromptTemplate.from_yaml(prompt_template_yaml) if prompt_template_yaml else None
        self.tools = tools or []
        self.env = env or {}
        self.subagents = subagents or []
        if isinstance(as_tool, dict):
            self.as_tool = as_tool 
        elif isinstance(as_tool, (ToolDefinition, SubagentToolDefinition)):
            self.as_tool = as_tool.model_dump()
        else:
            self.as_tool = None
        self.workspace = workspace
        self.id = id
        self.prompt_template_vars = prompt_template_vars or {}
        self.hooks = hooks or ['claude.md']
        if workspace:
            id = id or f'{self.prompt_template_yaml.split("/")[-1].split(".")[0]}'
            workspace.register_agent(agent=self, id=id)
    
    
    async def run(self,
                  query: Optional[str]=None,
                  initial_user_content: Optional[List[ContentBlock]] = None,
                  thread_id: Optional[str] = None,
                  llm_overrides: Optional[Dict[str, Any]] = None,
                  **prompt_template):
        # Start with agent's llm config
        thread_llm = dict(self.llm)
            
        # Apply any additional llm overrides
        if llm_overrides:
            thread_llm.update(llm_overrides)

        _vars = dict(**self.prompt_template_vars, **prompt_template)
        messages_to_apply = {"query": query} if query is not None else {}
        messages = self.prompt_template.apply(dict(messages_to_apply, **_vars)) if self.prompt_template else []

        if initial_user_content:
            user_message = ChatMessage(role=MessageRole.user, content=initial_user_content)
            messages = messages[:-1] + [user_message] if messages else [user_message]
        
        thread = Thread(
            llm=thread_llm,
            messages=messages, 
            tools=self.tools, 
            env=self.env, 
            subagents=self.subagents, 
            workspace=self.workspace, 
            id=thread_id,
            hooks=self.hooks,
            agent_id=self.id  # Pass the agent's ID to the thread
        )
        await thread.prepare()
        return thread

    def json(self):
        as_tool_for_json = self.as_tool
        if isinstance(self.as_tool, (ToolDefinition, SubagentToolDefinition)):
             as_tool_for_json = self.as_tool.model_dump()
        return {
            "llm": self.llm,
            "prompt_template": self.prompt_template_yaml,
            "tools": self.tools,
            "env": self.env,
            "subagents": self.subagents,
            "as_tool": as_tool_for_json,
            "prompt_template_vars": self.prompt_template_vars,
            "hooks": self.hooks,
        }

class McpServerTool:
    def __init__(self, name, definition: ToolDefinition, mcp_session: ClientSession):
        self.name = name
        self.definition = definition
        self.mcp_session = mcp_session

    async def prepare(self, tool_use_block):
        return McpServerToolCall(name=self.name, mcp_session=self.mcp_session, tool_use_block=tool_use_block)

class McpServerToolCall:
    def __init__(self, name, mcp_session: ClientSession, tool_use_block: ToolUseBlock):
        self.name = name
        self.mcp_session = mcp_session
        self.tool_use_block = tool_use_block
    
    async def call(self) -> ToolResultBlock:
        result = await self.mcp_session.call_tool(self.name, self.tool_use_block.input)
        blocks = []
        for item in result.content:
            meta = item.annotations.model_dump() if item.annotations is not None else {}
            if item.type == "text": blocks.append(TextBlock(text=item.text, meta=meta))
            elif item.type == "image": blocks.append(ImageBlock.from_base64(data=item.data, media_type=item.mimeType, meta=meta))
            else: raise ValueError(f"Unknown block type: {item.type}")
        return ToolResultBlock(tool_use_id=self.tool_use_block.id, content=blocks)

class SubagentTool:
    def __init__(self, name: str, agent: Agent, parent: 'Thread'):
        self.name = name
        self.agent = agent
        self.parent = parent

        as_tool_data = agent.as_tool
        if not isinstance(as_tool_data, dict) and as_tool_data is not None:
             logger.warning(f"Subagent '{name}' as_tool data is not a dict: {type(as_tool_data)}. Attempting dump.")
             if hasattr(as_tool_data, 'model_dump'): as_tool_data = as_tool_data.model_dump()
             else: as_tool_data = {}

        definition_model = SubagentToolDefinition(**as_tool_data) if as_tool_data else SubagentToolDefinition(name=name, description=f"Call a {name}-subagent")
        definition_data_dict = definition_model.model_dump()
        definition_data_dict['input_schema']['properties']['thread_id'] = {"type": "string", "description": "Thread ID for follow-up."}
        self.definition = ToolDefinition(**definition_data_dict)
        
    async def prepare(self, tool_use_block: ToolUseBlock):
        agent_input = tool_use_block.input or {}
        thread_id = agent_input.get('thread_id', None)
        query_for_subagent = agent_input.get('query', None) # Extract query for subagent

        if thread_id and thread_id in self.parent._threads:
            sub_thread = self.parent._threads[thread_id]
            # If query is provided, it's a new message to existing sub-thread
            await sub_thread.run(query=query_for_subagent) 
        else:
            # For new sub-thread, pass query directly to agent.run
            sub_thread = await self.agent.run(query=query_for_subagent, **{k:v for k,v in agent_input.items() if k not in ['query', 'thread_id']})
            thread_id = sub_thread.id
            self.parent._threads[thread_id] = sub_thread
        return SubagentToolCall(name=self.name, thread=sub_thread, tool_use_block=tool_use_block)

class SubagentToolCall:
    def __init__(self, name: str, thread: 'Thread', tool_use_block: ToolUseBlock):
        self.name = name
        self.thread = thread
        self.tool_use_block = tool_use_block

    async def call(self) -> ToolResultBlock:
        initial_messages_len = len(self.thread.messages)
        final_message_content: List[ContentBlock] = [TextBlock(text="Subagent did not produce additional output.")]
        async for message in self.thread:
            logger.info(f"> Subagent {self.name} (thread {self.thread.id}) message: {str(message)[:100]}")
            final_message_content = message.content
        response_content: List[ContentBlock] = final_message_content + [TextBlock(text=f"(Subagent: {self.name}, Thread ID: {self.thread.id})")]
        return ToolResultBlock(tool_use_id=self.tool_use_block.id, content=response_content, meta={"thread_id": self.thread.id, "message_start": initial_messages_len, "message_end": len(self.thread.messages)})

def maybe_handle_docker_command(command: str, args: List[str], env: Dict[str, str]) -> Tuple[str, List[str]]:
    if not command.startswith('docker'):
        return command, args
    # Replace $CWD in args with workspace cwd
    args = [arg.replace('$CWD', env['CWD']) for arg in args]
    # Inject envs to docker command
    for var_name, var_value in env.items():
        if var_name == "CWD": continue
        args.append(f"--env {var_name}={var_value}")
    return command, args

class Thread:
    def __init__(self, llm: Dict[str, Any], messages: List[ChatMessage] = None, tools: list[str] = None, env: dict[str, str] = None,
                 subagents: Optional[List[str]] = None, workspace: Optional['Workspace'] = None, id: Optional[str] = None, hooks: Optional[List[str]] = None,
                 agent_id: Optional[str] = None):
        self.exit_stack = AsyncExitStack()
        self.server_sessions: Dict[str, ClientSession] = {}
        self.llm = llm
        self._tools = tools or []
        self.messages = messages or []
        self.env = env or {}
        self.subagents = subagents or []
        self._threads: Dict[str, Thread] = {}
        self.workspace = workspace
        self.id = id or uuid4().hex
        self.agent_id = agent_id  # Track which agent created this thread
        self.tools: List[McpServerTool | SubagentTool] = []; self._ready = False # type: ignore
        self.hooks = hooks or []
        self._interrupted = False
        self.messages_after_hooks = []
        self.inbox = []
    
    def _resolve_model_alias(self, model: str) -> str:
        """Resolve model aliases: example - model = 'sota'
        - first checks if a workspace file defines what sota is, eg 'sota'->'opus'
        - then check the global alias file and replaces again, eg 'opus'->'claude-opus-4-1'
        """
        thread_aliases = self.home / '.model_alias.json'
        if thread_aliases.exists():
            resolved = load_json(thread_aliases).get(model, model)
        else:
            resolved = model
        return _ALIAS.get(resolved, resolved)
        
    async def prepare(self):
        if self._ready: return
        await self.connect_to_servers(self._tools or [], self.env or {})
        self.gather_subagent_tools()
        self._ready = True; logger.info(f"Thread '{self.id}' Connected and Prepared.")

    def resolve_env_vars(self, env_config: Dict[str, str], agent_env: Dict[str, str]) -> Dict[str, str]:
        """Resolve environment variables in MCP server config.
        
        If env_config has values like "$OPENAI_API_KEY", replace with actual values
        from agent's environment or system environment.
        """
        resolved_env = {}
        for key, value in env_config.items():
            if isinstance(value, str) and value.startswith('$'):
                env_var_name = value[1:]  # Remove the $ prefix
                # First check agent's env, then system environment
                resolved_value = agent_env.get(env_var_name) or os.environ.get(env_var_name)
                if resolved_value:
                    resolved_env[key] = resolved_value
                else:
                    logger.warning(f"Environment variable {env_var_name} not found for MCP server config key {key}")
                    resolved_env[key] = value  # Keep original if not found
            else:
                resolved_env[key] = value
        return resolved_env

    async def connect_to_servers(self, tool_specs: List[str], base_env: Dict[str, str]):
        for tool_id_spec in tool_specs:
            server_id, tool_name_filter = tool_id_spec.split(".", 1)
            if server_id not in self.server_sessions:
                server_config = _SERVERS.get(server_id)
                if server_config is None: raise ValueError(f"Server {server_id} not in MCP config.")
                
                # Determine transport type
                transport = server_config.get('transport', 'stdio')  # Default to stdio for backward compatibility
                
                if transport == 'stdio':
                    # Resolve environment variables from mcp.json
                    server_env = server_config.get('env', {})
                    resolved_server_env = self.resolve_env_vars(server_env, {**self.env, **base_env})
                    effective_env = {**resolved_server_env, **base_env}
                    command, args = maybe_handle_docker_command(server_config['command'], server_config['args'], effective_env)

                    threads_root = self.workspace._root_dir
                    log_dir = (threads_root / "threads" / self.id)
                    log_dir.mkdir(parents=True, exist_ok=True)
                    server_log_fp = open(log_dir / f"{server_id}.log", "a", encoding="utf-8", buffering=1)
                    print(f"Logging to: {server_log_fp}")

                    params = StdioServerParameters(command=command, args=args, env=effective_env)
                    stdio, write = await self.exit_stack.enter_async_context(stdio_client(params, errlog=server_log_fp))

                    mcp_session = await self.exit_stack.enter_async_context(ClientSession(stdio, write))
                elif transport == 'sse':
                    url = server_config.get('url')
                    if not url:
                        raise ValueError(f"SSE transport requires 'url' in server config for {server_id}")
                    
                    headers = server_config.get('headers', {})
                    effective_headers = {**headers}
                    for key, value in {**(server_config.get('env', {})), **base_env}.items():
                        if key.upper().endswith('_TOKEN') or key.upper().endswith('_KEY'):
                            effective_headers[f'X-{key.replace("_", "-")}'] = value
                    
                    timeout = server_config.get('timeout', 5)
                    sse_read_timeout = server_config.get('sse_read_timeout', 300)
                    
                    read, write = await self.exit_stack.enter_async_context(
                        sse_client(url, headers=effective_headers, timeout=timeout, sse_read_timeout=sse_read_timeout)
                    )
                    mcp_session = await self.exit_stack.enter_async_context(ClientSession(read, write))
                else:
                    raise ValueError(f"Unknown transport type '{transport}' for server {server_id}")
                
                await mcp_session.initialize()
                self.server_sessions[server_id] = mcp_session
            else:
                mcp_session = self.server_sessions[server_id]
            
            resp = await mcp_session.list_tools()
            for t_def in resp.tools:
                if t_def.name == tool_name_filter or tool_name_filter == "*":
                    self.tools.append(McpServerTool(name=t_def.name, definition=ToolDefinition(name=t_def.name, description=t_def.description, input_schema=t_def.inputSchema), mcp_session=mcp_session))
    
    def gather_subagent_tools(self):
        if not self.workspace or not self.subagents: return
        for subagent_id_str in self.subagents:
            try:
                agent_instance = self.workspace.get_agent(subagent_id_str)
                self.tools.append(SubagentTool(name=subagent_id_str, agent=agent_instance, parent=self))
            except KeyError: logger.warning(f"Subagent '{subagent_id_str}' (KeyError) for thread '{self.id}' not found.")
            except Exception as e: logger.error(f"Error gathering subagent '{subagent_id_str}' for '{self.id}': {e}", exc_info=True)
    
    async def cleanup(self):
        for sub_thread_instance in self._threads.values(): await sub_thread_instance.cleanup()
        await self.exit_stack.aclose()
    
    async def _add_message(self, message, apply_hooks=False):
        """Add messages, apply hooks, save if needed"""
        self.messages.append(message)
        if apply_hooks:
            await self.apply_hooks()
        if self.workspace:
            self.workspace.add_thread(thread=self, id=self.id)
        return message

    async def apply_hooks(self):
        self.messages_after_hooks = [ChatMessage(**msg.model_dump()) for msg in self.messages]
        for hook_name in self.hooks:
            hook = _HOOKS.get(hook_name)
            if hook is None:
                raise ValueError(f"Hook '{hook_name}' is not registered. Make sure to register hooks using @register_hook decorator.")
            await hook(self)
        # Remove any tool use messages when the tool result is not present
        messages = []
        for previous, current in zip(self.messages_after_hooks[:-1], self.messages_after_hooks[1:]):
            missing_response = False
            if previous.role == 'assistant':
                for block in previous.content:
                    if isinstance(block, ToolUseBlock):
                        if not any(isinstance(b, ToolResultBlock) and b.tool_use_id == block.id for b in current.content):
                            missing_response = True
                            break
            if not missing_response:
                messages.append(previous)
        messages.append(self.messages_after_hooks[-1])  # Always keep the last message
        self.messages_after_hooks = messages

    async def __aiter__(self):
        if not self._ready:
            await self.prepare()
        self._interrupted = False
        await self.apply_hooks()
        while not self._interrupted:
            if self.inbox:
                self.messages[-1].content, self.inbox = self.messages[-1].content + self.inbox, []
            
            # Prepare llm config with resolved model alias
            llm_config = dict(self.llm)
            if 'model' in llm_config:
                llm_config['model'] = self._resolve_model_alias(llm_config['model'])
            
            llm_response_message = await get_response(
                messages=self.messages_after_hooks,
                tools=[t.definition for t in self.tools if hasattr(t, 'definition')],
                **llm_config
            ); await asyncio.sleep(0.1)
            if self.inbox:
                continue
            yield await self._add_message(llm_response_message); await asyncio.sleep(0.1)
            tool_results_message, done = await self.process_message(llm_response_message)
            if done:
                break
            yield await self._add_message(tool_results_message, apply_hooks=True); await asyncio.sleep(0.1)
    
    async def process_message(self, msg_w_tool_uses: ChatMessage) -> Tuple[ChatMessage, bool]:
        tool_use_blocks = [b for b in msg_w_tool_uses.content if isinstance(b, ToolUseBlock)]
        if not tool_use_blocks or self._interrupted: return ChatMessage(role=MessageRole.user, content=[]), True
        prepared_calls = []
        for blk in tool_use_blocks:
            tool_runner = next((t for t in self.tools if t.name == blk.name), None)
            if tool_runner:
                prepared_calls.append(await tool_runner.prepare(blk))
            else:
                logger.error(f"Tool '{blk.name}' called by LLM but not found for thread '{self.id}'.")
        if not prepared_calls:
            return ChatMessage(role=MessageRole.user, content=[]), True
        result_blks = await asyncio.gather(*[pc.call() for pc in prepared_calls])
        return ChatMessage(role=MessageRole.user, content=result_blks), False
    
    async def tool_call(self, tool_name_str: str, input_data_dict: Dict[str, Any]) -> List[ContentBlock]:
        if not self._ready:
            await self.prepare()
        tool_runner_inst = next((t for t in self.tools if t.name == tool_name_str), None)
        if not tool_runner_inst:
            raise ValueError(f"Tool {tool_name_str} not found for thread '{self.id}'.")
        tool_call_obj = await tool_runner_inst.prepare(ToolUseBlock(id=uuid4().hex, name=tool_name_str, input=input_data_dict))
        tool_result_obj = await tool_call_obj.call(); return tool_result_obj.content
    
    def interrupt(self):
        self._interrupted = True
    
    async def run(self, query: Optional[str] = None, user_content: Optional[List[ContentBlock]] = None):
        if not self._ready:
            await self.prepare()
        
        if user_content:
            self.messages.append(ChatMessage(role=MessageRole.user, content=user_content))
        elif query is not None: # Only use query if user_content is not provided
            self.messages.append(ChatMessage(role=MessageRole.user, content=[TextBlock(text=query)]))
        return self
    
    def get_first_user_message_preview(self, max_words: int = 7) -> Optional[str]:
        first_user_message = next((msg for msg in self.messages if msg.role == MessageRole.user), None)
        if not first_user_message:
            return None

        text_parts = []
        for block in first_user_message.content:
            if isinstance(block, TextBlock):
                text_parts.append(block.text.strip())
        
        full_text = " ".join(text_parts).strip()
        if not full_text:
            return None

        words = full_text.split()
        if len(words) > max_words:
            return " ".join(words[:max_words]) + "..."
        return full_text
    
    @property
    def home(self):
        return Path(self.env['CWD'])

    def json(self):
        return {"llm": self.llm, "messages": [m.model_dump() for m in self.messages], "tools": self._tools,
                "env": self.env, "subagents": self.subagents, "thread_ids": list(self._threads.keys()), "hooks": self.hooks,
                "agent_id": self.agent_id}