from typing import List, Dict, Tuple, Any
import asyncio
from typing import Optional
from contextlib import AsyncExitStack
from uuid import uuid4
import json
import logging

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from dotenv import load_dotenv
from automator.utils import load_json

from automator.dtypes import (
    ContentBlock,
    TextBlock,
    ImageBlock,
    ToolDefinition,
    SubagentToolDefinition,
    ToolUseBlock,
    ToolResultBlock,
    PromptTemplate,
    ChatMessage,
    MessageRole
)
from automator.llm import get_response

logger = logging.getLogger("uvicorn")
logger.error("AUTOMATOR.AGENT.PY HAS BEEN RELOADED/IMPORTED (v4 with initial_user_content)")

load_dotenv()

_SERVERS = load_json('~/mcp.json').get('mcpServers', {})

class Agent:
    def __init__(
        self,
        model: str, # Model name to use for the agent
        prompt_template_yaml: str, # Path to the prompt template YAML file
        tools: List[str] = None, # List of tool names to use in format "{server_id}.{tool_name}" or "{server_id}.*"
        env: Dict[str, str] = None, # Environment variables used for MCP servers
        subagents: None | List[str] = None, # List of subagent names to use
        as_tool: ToolDefinition | None = None, # Tool definition if the agent is used as subagent
        workspace: Optional[str] = None, # Workspace to register the agent in
        id: Optional[str] = None, # ID of the agent in the workspace
        prompt_template_vars: Optional[Dict[str, Any]] = None, # Default values to use in prompt_template.apply()
    ):
        self.model = model
        self.prompt_template_yaml = prompt_template_yaml
        self.prompt_template = PromptTemplate.from_yaml(prompt_template_yaml)
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
        if workspace:
            id = id or f'{self.prompt_template_yaml.split("/")[-1].split(".")[0]}'
            workspace.register_agent(agent=self, id=id)
    
    
    async def run(self,
                  query: Optional[str]=None,
                  initial_user_content: Optional[List[ContentBlock]] = None,
                  temperature: float = 0.7, max_tokens: int = None,
                  thread_id: Optional[str] = None,
                  **prompt_template):
        if max_tokens is None:
            if 'haiku' in self.model:
                max_tokens = 4000
            elif 'gemini' in self.model:
                max_tokens = 64000
            elif 'claude' in self.model:
                max_tokens = 8000
            elif model.startswith('o'):
                max_tokens = 64000

        _vars = dict(**self.prompt_template_vars, **prompt_template)
        messages_to_apply = {"query": query} if query is not None else {}
        messages = self.prompt_template.apply(dict(messages_to_apply, **_vars))

        if initial_user_content:
            user_message = ChatMessage(role=MessageRole.user, content=initial_user_content)
            messages = messages[:-1] + [user_message]
        
        thread = Thread(
            model=self.model, 
            messages=messages, 
            tools=self.tools, 
            env=self.env, 
            subagents=self.subagents, 
            temperature=temperature, 
            max_tokens=max_tokens, 
            workspace=self.workspace, 
            id=thread_id
        )
        await thread.prepare()
        return thread

    def json(self):
        as_tool_for_json = self.as_tool
        if isinstance(self.as_tool, (ToolDefinition, SubagentToolDefinition)):
             as_tool_for_json = self.as_tool.model_dump()
        return {
            "model": self.model,
            "prompt_template": self.prompt_template_yaml,
            "tools": self.tools,
            "env": self.env,
            "subagents": self.subagents,
            "as_tool": as_tool_for_json,
            "prompt_template_vars": self.prompt_template_vars,
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
            if item.type == "text": blocks.append(TextBlock(text=item.text))
            elif item.type == "image": blocks.append(ImageBlock.from_base64(data=item.data, media_type=item.mimeType))
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

class Thread:
    def __init__(self, model: str, messages: List[ChatMessage], tools: list[str], env: dict[str, str],
                 subagents: Optional[List[str]] = None, temperature: float = 0.7, max_tokens: int = 4000,
                 workspace: Optional['Workspace'] = None, id: Optional[str] = None):
        self.exit_stack = AsyncExitStack()
        self.server_sessions: Dict[str, ClientSession] = {}
        self.model = model
        self._tools = tools
        self.messages = messages
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.env = env
        self.subagents = subagents or []
        self._threads: Dict[str, Thread] = {}; self.workspace = workspace; self.id = id or uuid4().hex
        self.tools: List[McpServerTool | SubagentTool] = []; self._ready = False # type: ignore
        
    async def prepare(self):
        if self._ready: return
        await self.connect_to_servers(self._tools or [], self.env or {})
        self.gather_subagent_tools()
        self._ready = True; logger.info(f"Thread '{self.id}' Connected and Prepared.")

    async def connect_to_servers(self, tool_specs: List[str], base_env: Dict[str, str]):
        for tool_id_spec in tool_specs:
            server_id, tool_name_filter = tool_id_spec.split(".", 1)
            if server_id not in self.server_sessions:
                server_config = _SERVERS.get(server_id)
                if server_config is None: raise ValueError(f"Server {server_id} not in MCP config.")
                effective_env = {**(server_config.get('env', {})), **base_env}
                params = StdioServerParameters(command=server_config['command'], args=server_config['args'], env=effective_env)
                stdio, write = await self.exit_stack.enter_async_context(stdio_client(params))
                mcp_session = await self.exit_stack.enter_async_context(ClientSession(stdio, write)); await mcp_session.initialize()
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
    
    async def __aiter__(self):
        if not self._ready:
            await self.prepare()
        while True:
            llm_response_message = await get_response(model=self.model, messages=self.messages, tools=[t for t in self.tools if hasattr(t, 'definition')], temperature=self.temperature, max_tokens=self.max_tokens)
            self.messages.append(llm_response_message); yield llm_response_message
            if self.workspace:
                self.workspace.add_thread(thread=self, id=self.id)
                self.to_markdown()
            tool_results_message, done = await self.process_message(llm_response_message)
            if done:
                break
            self.messages.append(tool_results_message); yield tool_results_message
            if self.workspace:
                self.workspace.add_thread(thread=self, id=self.id)
                self.to_markdown()
    
    async def process_message(self, msg_w_tool_uses: ChatMessage) -> Tuple[ChatMessage, bool]:
        tool_use_blocks = [b for b in msg_w_tool_uses.content if isinstance(b, ToolUseBlock)]
        if not tool_use_blocks: return ChatMessage(role=MessageRole.user, content=[]), True
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

    def json(self):
        return {"model": self.model, "messages": [m.model_dump() for m in self.messages], "tools": self._tools,
                "temperature": self.temperature, "max_tokens": self.max_tokens, "env": self.env,
                "subagents": self.subagents, "thread_ids": list(self._threads.keys())}
    
    def to_markdown(self) -> str:
        """
        Convert the internal chat representation into a markdown string,
        write it to ./logs/md/{thread_id}.md, and handle subthreads by linking.
        Returns the path to the generated markdown file.

        Tool use/result merging is only done for tool results in the message
        directly after the tool use message, and tool use ids are only unique
        within that pair.
        """
        import os
        import json # Already imported at module level
        import base64
        from uuid import uuid4 # Already imported at module level

        log_dir = "./.logs"
        md_dir = os.path.join(log_dir, "md")
        os.makedirs(log_dir, exist_ok=True) # Ensure .logs exists for images
        os.makedirs(md_dir, exist_ok=True) # Ensure .logs/md exists for markdown files

        output_path = os.path.abspath(os.path.join(md_dir, f"{self.id}.md"))
        # logger.info(f"Writing markdown to: {output_path}") # Temporarily comment out print
        md_output = f"# Thread: {self.id}\n\n"

        def render_block_md(block_item, tool_result_item=None) -> str:
            """Renders a single content block to markdown."""
            if isinstance(block_item, ToolUseBlock):
                return render_tool_block_md(block_item, tool_result_item)
            elif isinstance(block_item, ToolResultBlock):
                return ""
            elif isinstance(block_item, TextBlock):
                return block_item.text + "\n"
            elif isinstance(block_item, ImageBlock):
                img_filename = f"{uuid4()}.png"
                img_path_abs = os.path.join(log_dir, img_filename)
                img_path_rel = os.path.join("..", img_filename)
                try:
                    img_data = base64.b64decode(block_item.source.data)
                    with open(img_path_abs, 'wb') as f_img:
                        f_img.write(img_data)
                    return f"\n![image]({img_path_rel})\n"
                except Exception as e_img:
                    return f"\n*Error saving/rendering image: {e_img}*\n"
            else:
                return f"**Unknown block type:** {getattr(block_item, 'type', str(type(block_item)))}\n"

        def render_tool_block_md(tool_use: ToolUseBlock, tool_result: Optional[ToolResultBlock] = None) -> str:
            """Renders a tool call and its result, handling subthreads."""
            out_md = f"**Tool Call:** `{tool_use.name}` (ID: `{tool_use.id}`)\n"
            out_md += f"```json\n{json.dumps(tool_use.input, indent=2)}\n```\n"

            if tool_result:
                if getattr(tool_result, "meta", None) and 'thread_id' in tool_result.meta:
                    subthread_id = tool_result.meta['thread_id']
                    start_msg_idx = tool_result.meta.get('message_start', 0)
                    subthread_link = f"./{subthread_id}.md#message-{start_msg_idx}"
                    if self.workspace:
                        subthread = self.workspace.get_thread(subthread_id)
                        if subthread:
                            subthread.to_markdown()
                            out_md += f"**Sub-Thread Output:** See [Thread {subthread_id} (Message {start_msg_idx})]({subthread_link})\n"
                        else:
                            out_md += f"**Sub-Thread Output:** Error - Subthread {subthread_id} not found.\n"
                    else:
                        out_md += f"**Sub-Thread Output:** Error - Workspace not available for subthread {subthread_id}.\n"
                else:
                    out_md += f"**Tool Result:** (for call ID: `{tool_result.tool_use_id}`)\n"
                    tool_output_md_val = ""
                    content_list_val = tool_result.content if isinstance(tool_result.content, list) else [tool_result.content]
                    for content_block_val in content_list_val:
                        tool_output_md_val += render_block_md(content_block_val)
                    if tool_output_md_val.strip():
                        indented_output = "> " + tool_output_md_val.replace("\n", "\n> ").strip()
                        if indented_output.endswith('\n> '): indented_output = indented_output[:-3]
                        elif indented_output == '> ': indented_output = "> *No displayable output*"
                        out_md += indented_output + "\n"
                    else:
                        out_md += "> *No displayable output*\n"
            else:
                out_md += "**Tool Result:** *Pending or Not Available*\n"
            return out_md

        idx = 0
        while idx < len(self.messages):
            msg_item = self.messages[idx]
            if msg_item.content and all(isinstance(b, ToolResultBlock) for b in msg_item.content):
                idx += 1
                continue
            md_output += f"<a id=\"message-{idx}\"></a>\n### Message {idx} ({msg_item.role.value.capitalize()})\n\n"
            message_content_md_val = ""
            tool_result_blocks_by_id_map = {}
            if (idx + 1) < len(self.messages):
                next_msg_item = self.messages[idx + 1]
                if next_msg_item.content and all(isinstance(b, ToolResultBlock) for b in next_msg_item.content):
                    for b_item in next_msg_item.content:
                        tool_result_blocks_by_id_map[b_item.tool_use_id] = b_item # type: ignore
            for block_val in msg_item.content:
                if isinstance(block_val, ToolUseBlock):
                    tool_result_val = tool_result_blocks_by_id_map.get(block_val.id)
                    message_content_md_val += render_block_md(block_val, tool_result_val)
                else:
                    message_content_md_val += render_block_md(block_val)
            if message_content_md_val.strip():
                md_output += message_content_md_val + "\n"
            if message_content_md_val.strip() or idx < len(self.messages) - 1:
                md_output += '---\n\n'
            idx += 1

        with open(output_path, 'w', encoding='utf-8') as f_out:
            f_out.write(md_output)
        return output_path # Return the path to the markdown file