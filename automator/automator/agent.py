from typing import List, Dict, Tuple, Any
import asyncio
from typing import Optional
from contextlib import AsyncExitStack
from uuid import uuid4
import json

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

load_dotenv()


_SERVERS = load_json('~/mcp.json').get('mcpServers', {})


class Agent:
    def __init__(
        self,
        model: str,
        prompt_template_yaml: str,
        tools: List[str] = None,
        env: Dict[str, str] = None,
        subagents: None | List[str] = None,
        as_tool: ToolDefinition | None = None,
        workspace: Optional[str] = None,
        id: Optional[str] = None,
    ):
        self.model = model
        self.prompt_template_yaml = prompt_template_yaml
        self.prompt_template = PromptTemplate.from_yaml(prompt_template_yaml)
        self.tools = tools or []
        self.env = env or {}
        self.subagents = subagents or []
        self.as_tool = as_tool
        self.workspace = workspace
        if workspace:
            id = id or f'{self.prompt_template_yaml.split("/")[-1].split(".")[0]}({",".join(self.tools)})'
            workspace.register_agent(agent=self, id=id)
    
    async def run(self, query: str=None, temperature: float = 0.7, max_tokens: int = 8000, thread_id: Optional[str] = None, **prompt_template):
        messages = self.prompt_template.apply(dict({"query": query}, **prompt_template))
        thread = Thread(self.model, messages, self.tools, self.env, self.subagents, temperature, max_tokens, self.workspace, thread_id)
        await thread.prepare()
        return thread

    def json(self):
        return {
            "model": self.model,
            "prompt_template": self.prompt_template_yaml,
            "tools": self.tools,
            "env": self.env,
            "subagents": self.subagents,
            "as_tool": self.as_tool,
        }


class McpServerTool:
    def __init__(self, name, definition: ToolDefinition, mcp_session: ClientSession):
        self.name = name
        self.definition = definition
        self.mcp_session = mcp_session
    
    async def prepare(self, tool_use_block):
        return McpServerToolCall(
            name=self.name,
            mcp_session=self.mcp_session,
            tool_use_block=tool_use_block
        )
    

class McpServerToolCall:
    def __init__(self, name, mcp_session: ClientSession, tool_use_block: ToolUseBlock):
        self.name = name
        self.mcp_session = mcp_session
        self.tool_use_block = tool_use_block

    async def call(self) -> ToolResultBlock:
        result = await self.mcp_session.call_tool(self.name, self.tool_use_block.input)
        blocks = []
        for item in result.content:
            if item.type == "text":
                blocks.append(TextBlock(text=item.text))
            elif item.type == "image":
                blocks.append(ImageBlock.from_base64(data=item.data, media_type=item.mimeType)) 
            else:
                raise ValueError(f"Unknown block type: {item.type}")
        tool_result = ToolResultBlock(tool_use_id=self.tool_use_block.id, content=blocks)
        return tool_result


class SubagentTool:
    def __init__(self, name: str, agent: Agent, parent: 'Thread'):
        self.name = name
        definition = agent.as_tool if agent.as_tool else SubagentToolDefinition(
            name=name,
            description=f"Call a {name}-subagent"
        )
        definition = definition.model_dump()
        definition['input_schema']['properties']['thread_id'] = {
            "type": "string",
            "description": "Thread ID of an existing thread to send a follow-up message to. Leave empty to start a new thread."
        }
        self.definition = ToolDefinition(**definition)
        self.agent = agent
        self.parent = parent
    
    async def prepare(self, tool_use_block: ToolUseBlock):
        agent_input = tool_use_block.input
        thread_id = agent_input.pop('thread_id', None)
        if thread_id in self.parent._threads:
            thread = await self.parent._threads[thread_id].run(**agent_input)
        else:
            thread = await self.agent.run(**agent_input)
            thread_id = thread.id
            self.parent._threads[thread_id] = thread
        return SubagentToolCall(
            name=self.name,
            thread=thread,
            tool_use_block=tool_use_block
        )


class SubagentToolCall:
    def __init__(self, name: str, thread: 'Thread', tool_use_block: ToolUseBlock):
        self.name = name
        self.thread = thread
        self.tool_use_block = tool_use_block

    async def call(self) -> ToolResultBlock:
        # Run the thread
        initial_messages = len(self.thread.messages)
        async for message in self.thread:
            print("> " + str(message)[:100])
        # Return the final message content
        content = message.content + [TextBlock(text=f"(To send follow-up messages, use the thread ID: {self.thread.id}")]
        result = ToolResultBlock(tool_use_id=self.tool_use_block.id, content=content, meta={
            "thread_id": self.thread.id,
            "message_start": initial_messages,
            "message_end": len(self.thread.messages),
        })
        return result


class Thread:
    def __init__(
        self,
        model: str,
        messages: List[ChatMessage],
        tools: list[str],
        env: dict[str, str],
        subagents: None | List[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 8000,
        workspace: Optional[str] = None,
        id: Optional[str] = None,
    ):
        # Initialize session and client objects
        self.exit_stack = AsyncExitStack()
        self.server_sessions: Dict[str, ClientSession] = {}
        
        self.model = model
        self._tools = tools
        self.messages = messages
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.env = env
        self.subagents = subagents or []
        self._threads = {}
        self.workspace = workspace
        self.id = id or uuid4().hex

        self.tools: List[McpServerTool | SubagentTool] = []
        self._ready = False
        
    async def prepare(self):
        """Prepare the thread for execution: connect to servers and gather subagent tools"""
        await self.connect_to_servers(self._tools, self.env)
        self.gather_subagent_tools()
        self._ready = True
        print(f"Connected")

    async def connect_to_servers(self, tools: List[str], env: Optional[dict] = None):
        """Connect to the MCP servers required by the tools"""
        for tool_id in tools:
            server_id, tool_name = tool_id.split(".")
            if server_id not in self.server_sessions:
                server = _SERVERS.get(server_id)
                if server is None:
                    raise ValueError(f"Server {server_id} not found in MCP servers configuration.")
                params = StdioServerParameters(
                    command=server['command'],
                    args=server['args'],
                    env=dict(**server.get('env', {}), **env)
                )
                stdio, write = await self.exit_stack.enter_async_context(stdio_client(params))
                mcp_session = await self.exit_stack.enter_async_context(ClientSession(stdio, write))
                await mcp_session.initialize()
                self.server_sessions[server_id] = mcp_session
            else:
                mcp_session = self.server_sessions[server_id]

            resp = await mcp_session.list_tools()
            for t in resp.tools:
                if t.name == tool_name or tool_name == "*":
                    self.tools.append(McpServerTool(
                        name=t.name,
                        definition=ToolDefinition(
                            name=t.name,
                            description=t.description,
                            input_schema=t.inputSchema,
                        ),
                        mcp_session=mcp_session
                    ))
    
    def gather_subagent_tools(self):
        """Gather subagent tools from the workspace"""
        if self.workspace is None and self.subagents:
            raise ValueError("Workspace is required to gather subagent tools.")
        for subagent in self.subagents:
            agent = self.workspace.get_agent(subagent)
            if agent is None:
                raise ValueError(f"Subagent {subagent} not found in workspace.")
            self.tools.append(SubagentTool(
                name=subagent,
                agent=agent,
                parent=self
            ))
    
    async def cleanup(self):
        """Clean up resources"""
        for thread in self._threads.values():
            await thread.cleanup()
        await self.exit_stack.aclose()
    
    async def __aiter__(self):
        """Run the (model, tool execution) loop"""
        while True:
            message = await get_response(
                model=self.model,
                messages=self.messages,
                tools=self.tools,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            yield message
            self.messages.append(message)
            self.to_markdown()
            if self.workspace:
                self.workspace.add_thread(thread=self, id=self.id)
            message, done = await self.process_message(message)
            if done:
                break
            yield message
            self.messages.append(message)
            self.to_markdown()
            if self.workspace:
                self.workspace.add_thread(thread=self, id=self.id)
    
    async def process_message(self, message) -> Tuple[ChatMessage, bool]:
        """Scan the message for tool use blocks and execute them"""
        tool_calls = [await self.prepare_tool_call(block) for block in message.content if block.type == "tool_use"]
        result_blocks = await asyncio.gather(*[tool_calls.call() for tool_calls in tool_calls])
        result_message = ChatMessage(
            role=MessageRole.user,
            content=result_blocks
        )
        return result_message, len(tool_calls) == 0
    
    async def prepare_tool_call(self, block: ToolUseBlock) -> ToolResultBlock:
        """Process a tool call block"""
        for tool in self.tools:
            if tool.name == block.name:
                return await tool.prepare(block)
        raise ValueError(f"Tool {block.name} not found in available tools.")

    async def tool_call(self, tool_name: str, input: Dict[str, Any]) -> List[ContentBlock]:
        """Process a tool call block outside of the standard (llm, tool calls) loop"""
        for tool in self.tools:
            if tool.name == tool_name:
                tool_call = await tool.prepare(
                    ToolUseBlock(
                        id=uuid4().hex,
                        name=tool_name,
                        input=input
                    )
                )
                tool_result = await tool_call.call()
                return tool_result.content
        raise ValueError(f"Tool {tool_name} not found in available tools.")
    
    async def run(self, query):
        if not self._ready:
            await self.prepare()
        self.messages.append(ChatMessage(role=MessageRole.user, content=[TextBlock(text=query)]))
        return self
    
    def json(self):
        return {
            "model": self.model,
            "messages": [msg.model_dump() for msg in self.messages],
            "tools": self._tools,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "env": self.env,
            "subagents": self.subagents,
            "thread_ids": list(self._threads.keys()), 
        }
    
    def to_markdown(self) -> str:
        """
        Convert the internal chat representation into a markdown string,
        write it to ./logs/md/{thread_id}.md, and handle subthreads by linking.
        Returns the path to the generated markdown file.
        """
        import os
        import json
        import base64
        from uuid import uuid4

        log_dir = "./.logs"
        md_dir = os.path.join(log_dir, "md")
        os.makedirs(log_dir, exist_ok=True) # Ensure .logs exists for images
        os.makedirs(md_dir, exist_ok=True) # Ensure .logs/md exists for markdown files

        output_path = os.path.join(md_dir, f"{self.id}.md")
        print(output_path)
        md = f"# Thread: {self.id}\n\n"

        # Collect all tool use and tool result blocks by id for merging
        tool_uses = {}
        tool_results = {}
        for msg in self.messages:
            for block in msg.content:
                if isinstance(block, ToolUseBlock):
                    tool_uses[block.id] = block
                elif isinstance(block, ToolResultBlock):
                    tool_results[block.tool_use_id] = block

        def render_block(block) -> str:
            """Renders a single content block to markdown."""
            if isinstance(block, ToolUseBlock):
                # Find the corresponding result
                tool_result = tool_results.get(block.id)
                return render_tool_block(block, tool_result)
            # ToolResultBlocks are only rendered *with* their ToolUseBlock
            # If encountered alone, skip rendering it here.
            elif isinstance(block, ToolResultBlock):
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

        def render_tool_block(tool_use: 'ToolUseBlock', tool_result: Optional['ToolResultBlock'] = None) -> str:
            """Renders a tool call and its result, handling subthreads."""
            out = f"**Tool Call:** `{tool_use.name}` (ID: `{tool_use.id}`)\n"
            out += f"```json\n{json.dumps(tool_use.input, indent=2)}\n```\n"

            if tool_result:
                # Check if this tool call corresponds to a subthread
                if tool_result.meta and 'thread_id' in tool_result.meta:
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
        for i, msg in enumerate(self.messages):
            # Skip messages that *only* contain ToolResultBlocks
            if msg.content and all(isinstance(block, ToolResultBlock) for block in msg.content):
                continue

            # Add a message header that can be used as an anchor
            md += f"<a id=\"message-{i}\"></a>\n" # HTML anchor for linking
            md += f"### Message {i} ({msg.role.value.capitalize()})\n\n"

            message_content_md = ""
            for block in msg.content:
                 # Render block content using the helper
                message_content_md += render_block(block)

            # Only add content if it's not empty after rendering
            if message_content_md.strip():
                md += message_content_md + "\n"
            # Add separator only if content was added or it's not the last message
            if message_content_md.strip() or i < len(self.messages) - 1:
                 md += '---\n\n' # Use horizontal rule for separation

        # Add Tools information at the end
        if self.tools:
            md += "# Tools Available\n"
            for tool in self.tools:
                md += f"## `{tool.name}`\n"
                md += f"{tool.definition.description}\n"
                # Pretty print the input schema
                schema_str = json.dumps(tool.definition.input_schema, indent=2)
                md += f"**Input Schema:**\n```json\n{schema_str}\n```\n\n"

        # Write the collected markdown to the file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(md)

        return output_path

