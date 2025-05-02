# Question:
I am learning about MCP - Model Context Protocol. Can you explain me what exactly is the Context class (from mcp.server.fastmcp import Context)?
- which methods does it expose?
- when should I use it when developing an MCP server?
- how should I use it when developing an MCP client?

# Answer:

Fast MCP’s Context object is the little “Swiss-army knife” that FastMCP injects into your tool (and soon resource / prompt) functions.
It gives the code that is running inside a request a safe handle back to the MCP runtime so it can log, stream progress, read other resources, ask the client’s LLM to “do some sampling,” and inspect request metadata—all without you having to touch the underlying low-level session objects. In practice you only ever receive a Context from FastMCP; you do not construct one yourself.
What exactly is Context?

mcp.server.fastmcp.Context is a thin pydantic.BaseModel wrapper around the lower-level RequestContext that lives in mcp.shared.context
GitHub
.
Because it is a real class (not just a dict), you get type-hints and auto-complete in your editor, while FastMCP keeps ownership of the underlying session.
Key properties & methods

Below is the public surface that matters for everyday development (all methods are async unless noted):
Property / method	Purpose	Typical use-case
fastmcp	Reference to the enclosing FastMCP server	Very advanced scenarios (e.g., introspection)
GitHub
request_context	The raw low-level context	Only if you need something not exposed directly
request_id (str)	Unique ID for this request	Correlating logs, caches
FastMCP
client_id (str|None)	ID the client declared at connect time	Multi-tenant analytics
FastMCP
session	Low-level ServerSession	Rarely needed
GitHub
report_progress(progress, total=None)	Notify the client how far you are	Long-running jobs / UI progress bars
FastMCP
read_resource(uri)	Pull any registered resource (static or template)	“Open” files, SQL results, etc. from inside a tool
FastMCP
log(level, message, *, logger_name=None)	Generic logger	Structured logging
GitHub
Convenience log helpers debug / info / warning / error	Sugar for the above	Quick one-liners
FastMCP
sample(messages, system_prompt=None, …)
new in v2	Ask the client’s LLM to generate something on the fly	“Let the caller’s model summarise this text”
FastMCP

(If you’re on an older SDK you may not have sample; upgrade to fastmcp >= 2.0.)
When should you use Context while developing a server?

    Logging – Anything you await ctx.info("…") is streamed back to the client, so tools can show coloured logs in their UI instead of hiding them on the server’s stdout
    FastMCP
    .

    Progress reporting – If the client sent a progressToken, await ctx.report_progress(x, total) updates the caller’s spinner/progress-bar automatically
    FastMCP
    .

    Resource access – Calling another resource from within a tool is trivial: data = (await ctx.read_resource("resource://sales/2024")).content
    FastMCP
    .

    LLM sampling – Need a one-off judgement from the user’s model? await ctx.sample("Rate this answer 1-5") returns a TextContent you can parse
    FastMCP
    .

    Advanced plumbing – If you really need to poke at the raw session (e.g., custom streaming), ctx.session is there
    FastMCP
    .

    How to get it: just add an argument annotated with Context anywhere in the function signature—name does not matter:

@mcp.tool()
async def resize_image(uri: str, ctx: Context) -> ImageContent:
    await ctx.info("Downloading image")
    img_bytes = (await ctx.read_resource(uri))[0].content
    # …

FastMCP inspects the signature, notices the annotation, and injects a Context instance at call-time
FastMCP
.
Outside an active request the same properties raise ValueError, making accidental misuse obvious
GitHub
.
What does a client do with all of this?

You never import Context on the client side. Instead, a well-behaved MCP client:

    Supplies metadata – include a client_id and (if you want progress updates) a progressToken in the initial initialize call.

    Listens for server-initiated events—log messages, progress notifications, and sampling requests—from the transport (SSE, stdio, WebSocket, etc.) and surfaces them in its UI
    FastMCP
    FastMCP
    .

    Responds to sample – when the server asks to sample, the client forwards that prompt to its own LLM instance, then returns the generated TextContent / ImageContent as the response payload
    FastMCP
    .

    Does not send a Context argument when it calls a tool; FastMCP automatically discards extra parameters that do not appear in the tool’s schema
    GitHub
    .

A minimal Python client therefore looks like:

from fastmcp.client import Client, StdIOTransport

transport = StdIOTransport(cmd=["python", "my_server.py"])
client = Client(transport, client_id="chat_ui", progress=True)

tools = await client.list_tools()
result = await client.call_tool("resize_image", {"uri": ".../cat.png"})
# logs and progress events arrive via callbacks you registered on client

TL;DR

On the server: add ctx: Context to any tool that needs to talk back to the caller, read resources, stream progress, or even “delegate” sub-tasks to the caller’s LLM.
On the client: you don’t create Context; just be ready to receive its side-channel messages (logs, progress, sampling) and wire them into your UI.

With just that pattern you can build surprisingly rich, interactive tool chains while keeping your server code clean, typed, and framework-agnostic.

# Question:
Can you show how a MCP client displays the log info that is produced by an MCP server?

# Answer:

Below is a self-contained walk-through that shows how a Model Context Protocol (MCP) client receives and renders the log messages a server emits with ctx.info(), ctx.warning() and friends.
1 · What actually travels over the wire?

    Every log entry is sent by the server as a notifications/message JSON-RPC notification whose payload carries a level, an optional logger name and arbitrary data fields.​
    Model Context Protocol
    Home

    Servers that support logging advertise the capability in their initial handshake, and clients can raise or lower verbosity with a logging/setLevel request.​
    Model Context Protocol
    GitHub

    Because the format mirrors syslog levels, your client can show colour-coded severities or filter by level before printing.​
    Model Context Protocol

Put differently, no special transport is needed: the log message arrives on the same Web-Socket, SSE stream, stdio pipe—whatever you chose for the session.
2 · Registering a log handler on the Python FastMCP client

FastMCP’s high-level Client lets you provide a log_handler callback that is invoked for every incoming log notification.​
FastMCP
FastMCP
A minimal CLI that prints coloured lines with [Rich] looks like this:

import asyncio
from fastmcp import Client
from rich.console import Console
from mcp.client.session import LogLevel          # exported by the SDK

console = Console()

def rich_log(level: LogLevel, message: str, logger: str | None):
    style = {
        "debug":  "dim",
        "info":   "white",
        "notice": "cyan",
        "warning":"yellow",
        "error":  "bold red",
        "critical":"bold white on red"
    }.get(level, "white")
    console.print(f"[{level.upper():8}] {logger or 'server'}: {message}", style=style)

client = Client("my_server.py", log_handler=rich_log)  # stdio transport inferred

async def main():
    async with client:                          # opens the session
        await client.call_tool("process_file", {"file_uri": "file:///tmp/foo.txt"})

asyncio.run(main())

Rich already ships with a helper that installs a colourised FastMCP logger; the SDK exposes it via fastmcp.utilities.configure_logging.​
GitHub

Sample output

[INFO    ] server: [83d27] Starting processing for /tmp/foo.txt
[DEBUG   ] server: Read 2048 bytes from /tmp/foo.txt
[WARNING ] server: Resource /tmp/foo.txt is empty
[ERROR   ] server: Error processing /tmp/foo.txt: File not found

3 · Surfacing logs in a Web / desktop UI

Because the callback supplies structured fields (level, logger, message), you can push the same information into state and render it however you prefer. The React snippet below uses the TypeScript SDK, which exposes the same constructor signature:

import { useEffect, useState } from "react";
import { Client, WSTransport, LogLevel } from "fastmcp";

export default function LogsPanel() {
  const [logs, setLogs] = useState<
    { ts: number; level: LogLevel; logger?: string; msg: string }[]
  >([]);

  useEffect(() => {
    const client = new Client({
      transport: new WSTransport("ws://localhost:9000"),
      logHandler: (level, msg, logger) =>
        setLogs((l) => [...l, { ts: Date.now(), level, logger, msg }]),
    });
    client.connect();       // OR use `await` inside an async effect
    return () => client.close();
  }, []);

  return (
    <div className="space-y-1 font-mono text-sm">
      {logs.map(({ ts, level, logger, msg }) => (
        <div key={ts} className={`log ${level}`}>
          <span className="text-zinc-500">{new Date(ts).toLocaleTimeString()}</span>{" "}
          <span className="font-semibold">{level}</span>{" "}
          <span className="text-zinc-400">{logger ?? "server"}:</span>{" "}
          {msg}
        </div>
      ))}
    </div>
  );
}

FastMCP’s TypeScript example in the repo shows the exact same log.info("Downloading file…") pattern on the server side.​
GitHub
4 · Controlling noise from the client

If your UI only wants warnings and worse, lower the threshold before any tool call:

await client.set_log_level("warning")   # convenience wrapper around logging/setLevel

Under the hood this issues the JSON-RPC request defined in the spec.​
Model Context Protocol
GitHub
5 · Relationship to progress bars & other context utilities

Log notifications are independent of progress notifications, but both come from the same server-side Context object. Inside a tool you might see:

await ctx.info("Downloading…")
await ctx.report_progress(25, total=100)

The client receives two separate events: one for the log line and one for notifications/progress, which carries the progressToken you supplied in the call metadata.​
FastMCP
Model Context Protocol
You can wire those to a spinner or progress-bar widget while still printing the log line.
6 · Why it matters

Because MCP keeps logs out of stdout and inside the protocol, you can:

    stream server diagnostics to a browser panel while the backend stays on a different machine,

    persist logs per request ID for auditing, or

    forward them to a dedicated observability pipeline—without adding extra sockets or proprietary extensions. This mechanism is one reason vendors picked up MCP as a practical bridge between AI agents and traditional apps.​
    Axios

TL;DR

    The server calls ctx.debug|info|warning|error, which FastMCP converts into notifications/message.

    The client passes a log_handler, receives each message and decides how to display or filter it.

    A dozen lines of code produce coloured CLI output; a few more give you a live log panel in React. No additional protocol plumbing is required.