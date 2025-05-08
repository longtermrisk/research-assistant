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
logger.error("AUTOMATOR.AGENT.PY HAS BEEN RELOADED/IMPORTED (v3 no forward ref update)")

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
    
    
    async def run(self, query: Optional[str]=None, temperature: float = 0.7, max_tokens: int = 8000, thread_id: Optional[str] = None, **prompt_template):
        _vars = dict(**self.prompt_template_vars, **prompt_template)
        # Query can be None if we are just re-running a thread without a new user message
        messages_to_apply = {"query": query} if query is not None else {}
        messages = self.prompt_template.apply(dict(messages_to_apply, **_vars))
        
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
        as_tool_for_json = self.as_tool # Should already be a dict from __init__
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

        as_tool_data = agent.as_tool # Should be a dict
        if not isinstance(as_tool_data, dict) and as_tool_data is not None:
             # This case should ideally not happen if Agent.__init__ normalizes as_tool to dict
             logger.warning(f"Subagent '{name}' as_tool data is not a dict: {type(as_tool_data)}. Attempting dump.")
             if hasattr(as_tool_data, 'model_dump'): as_tool_data = as_tool_data.model_dump()
             else: as_tool_data = {}

        definition_model = SubagentToolDefinition(**as_tool_data) if as_tool_data else SubagentToolDefinition(name=name, description=f"Call a {name}-subagent")
        definition_data_dict = definition_model.model_dump()
        definition_data_dict['input_schema']['properties']['thread_id'] = {"type": "string", "description": "Thread ID for follow-up."}
        self.definition = ToolDefinition(**definition_data_dict)
        
    async def prepare(self, tool_use_block: ToolUseBlock):
        agent_input = tool_use_block.input or {}
        thread_id = agent_input.pop('thread_id', None)
        query = agent_input.pop('query', None) # Ensure query is extracted

        if thread_id and thread_id in self.parent._threads:
            sub_thread = self.parent._threads[thread_id]
            # If query is provided, it's a new message to existing sub-thread
            # The run method of Thread should handle appending the query if provided
            await sub_thread.run(query=query if query else None) 
        else:
            sub_thread = await self.agent.run(
                query=query, # Pass query to new sub-thread run
                **agent_input # Pass remaining args which might be for prompt_template
            )
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
        final_message_content = [TextBlock(text="Subagent did not produce additional output.")]
        async for message in self.thread:
            logger.info(f"> Subagent {self.name} (thread {self.thread.id}) message: {str(message)[:100]}")
            final_message_content = message.content
        response_content = final_message_content + [TextBlock(text=f"(Subagent: {self.name}, Thread ID: {self.thread.id})")]
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
        self.tools: List[McpServerTool | SubagentTool] = []; self._ready = False
        
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
    
    async def run(self, query: Optional[str] = None):
        if not self._ready:
            await self.prepare()
        if query is not None:
            self.messages.append(ChatMessage(role=MessageRole.user, content=[TextBlock(text=query)]))
        return self
    
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
        import json
        import base64
        from uuid import uuid4

        log_dir = "./.logs"
        md_dir = os.path.join(log_dir, "md")
        os.makedirs(log_dir, exist_ok=True) # Ensure .logs exists for images
        os.makedirs(md_dir, exist_ok=True) # Ensure .logs/md exists for markdown files

        output_path = os.path.abspath(os.path.join(md_dir, f"{self.id}.md"))
        print(output_path)
        md = f"# Thread: {self.id}\n\n"

        def render_block(block, tool_result=None) -> str:
            """Renders a single content block to markdown."""
            if isinstance(block, ToolUseBlock):
                # Only merge tool result if provided
                return render_tool_block(block, tool_result)
            elif isinstance(block, ToolResultBlock):
                # ToolResultBlocks are only rendered *with* their ToolUseBlock
                # If encountered alone, skip rendering it here.
                return ""
            elif isinstance(block, TextBlock):
                return block.text + "\n"
            elif isinstance(block, ImageBlock):
                # Save image relative to the main log dir, link relative to md file
                img_filename = f"{uuid4()}.png"
                img_path_abs = os.path.join(log_dir, img_filename)
                img_path_rel = os.path.join("..", img_filename) # Relative path from md file
                try:
                    # Assuming block.source.data holds the base64 encoded string
                    img_data = base64.b64decode(block.source.data)
                    with open(img_path_abs, 'wb') as f:
                        f.write(img_data)
                    return f"\n![image]({img_path_rel})\n"
                except Exception as e:
                    return f"\n*Error saving/rendering image: {e}*\n"
            else:
                return f"**Unknown block type:** {getattr(block, 'type', str(type(block)))}\n"

        def render_tool_block(tool_use: 'ToolUseBlock', tool_result: 'ToolResultBlock' = None) -> str:
            """Renders a tool call and its result, handling subthreads."""
            out = f"**Tool Call:** `{tool_use.name}` (ID: `{tool_use.id}`)\n"
            out += f"```json\n{json.dumps(tool_use.input, indent=2)}\n```\n"

            if tool_result:
                # Check if this tool call corresponds to a subthread
                if getattr(tool_result, "meta", None) and 'thread_id' in tool_result.meta:
                    subthread_id = tool_result.meta['thread_id']
                    start_msg_idx = tool_result.meta.get('message_start', 0)
                    subthread_link = f"./{subthread_id}.md#message-{start_msg_idx}"

                    # Ensure the subthread markdown file is generated (recursive call)
                    if self.workspace:
                        subthread = self.workspace.get_thread(subthread_id)
                        if subthread:
                            subthread.to_markdown() # Generate its file
                            out += f"**Sub-Thread Output:** See [Thread {subthread_id} (Message {start_msg_idx})]({subthread_link})\n"
                        else:
                            out += f"**Sub-Thread Output:** Error - Subthread {subthread_id} not found in workspace.\n"
                    else:
                        out += f"**Sub-Thread Output:** Error - Workspace not available to render subthread {subthread_id}.\n"

                else:
                    # Render normal tool result content
                    out += f"**Tool Result:** (for call ID: `{tool_result.tool_use_id}`)\n"
                    tool_output_md = ""
                    # Check if content is a list or single block
                    content_list = tool_result.content if isinstance(tool_result.content, list) else [tool_result.content]
                    for content_block in content_list:
                        # Important: Use the main render_block for nested content
                        # This prevents issues if tool results contain images etc.
                        tool_output_md += render_block(content_block)

                    # Indent the tool output using blockquotes
                    if tool_output_md.strip():
                        indented_output = "> " + tool_output_md.replace("\n", "\n> ").strip()
                        # Remove trailing '> ' if it ends with it
                        if indented_output.endswith('\n> '):
                            indented_output = indented_output[:-3]
                        elif indented_output == '> ': # Handle case of empty rendered output
                            indented_output = "> *No displayable output*"
                        out += indented_output + "\n"
                    else:
                        out += "> *No displayable output*\n"
            else:
                out += "**Tool Result:** *Pending or Not Available*\n"
            return out

        # Iterate through messages and render them
        i = 0
        while i < len(self.messages):
            msg = self.messages[i]

            # Skip messages that *only* contain ToolResultBlocks
            if msg.content and all(isinstance(block, ToolResultBlock) for block in msg.content):
                i += 1
                continue

            # Add a message header that can be used as an anchor
            md += f"<a id=\"message-{i}\"></a>\n" # HTML anchor for linking
            md += f"### Message {i} ({msg.role.value.capitalize()})\n\n"

            message_content_md = ""

            # If the next message consists only of ToolResultBlocks, pair them with this message's ToolUseBlocks
            tool_result_blocks_by_id = {}
            if (i + 1) < len(self.messages):
                next_msg = self.messages[i + 1]
                if next_msg.content and all(isinstance(block, ToolResultBlock) for block in next_msg.content):
                    for block in next_msg.content:
                        tool_result_blocks_by_id[block.tool_use_id] = block

            for block in msg.content:
                if isinstance(block, ToolUseBlock):
                    tool_result = tool_result_blocks_by_id.get(block.id)
                    message_content_md += render_block(block, tool_result)
                else:
                    message_content_md += render_block(block)

            # Only add content if it's not empty after rendering
            if message_content_md.strip():
                md += message_content_md + "\n"
            # Add separator only if content was added or it's not the last message
            if message_content_md.strip() or i < len(self.messages) - 1:
                md += '---\n\n' # Use horizontal rule for separation

            i += 1


        # Write the collected markdown to the file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(md)