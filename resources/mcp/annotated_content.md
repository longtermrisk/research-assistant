# Question:
Assume I have an MCP server that produces structured data. Is there a good way to handle this? For example, I want a tool to return text for the agent, and an additional metadata object that tells the GUI how to display that text (as markdown, or python, etc). Is there a non-hacky solution for this?

# Answer:
TL;DR — Yes, the protocol already has a clean, “first-class” way to shuttle presentation hints alongside the text the LLM will see.

Put the human-readable text in TextContent (so the model keeps working exactly the same) and attach any view-layer instructions in one of two places that MCP expressly reserves for extra data:

    content[i].annotations – ideal when the hint belongs to that one snippet of text (e.g., “treat this block as Markdown”).

    _meta on the result object – ideal for result-wide metadata (e.g., “the whole reply is really a Python artefact”).

Both fields are defined as open-ended dictionaries, so adding keys such as {"renderAs": "markdown"} is 100 % spec-legal and future-proof.
GitHub
GitHub

Below is the reasoning, the pattern, and complete server- and client-side snippets you can drop into a project.
1 Where the spec lets you put “extra stuff”
Location	Type	When to use	Why it’s legit
content[i].annotations	structured object on every TextContent, ImageContent, EmbeddedResource	The hint only applies to one fragment	annotations is explicitly designed for per-content hints such as audience or priority and allows extra keys
modelcontextprotocol.github.io
GitHub
result._meta	top-level dict on every Result, including CallToolResult	The hint applies to the whole result or you want one bundle of metadata	_meta is formally “reserved … to allow clients and servers to attach additional metadata”
GitHub
GitHub

Because the Python (and TS/JSON-schema) models use extra="allow", any unknown keys you add survive round-trips unchanged.
GitHub
2 Recommended pattern for “text + GUI metadata”
2.1 Server side (Python SDK)

from mcp import types

def explain_sql(query: str) -> types.CallToolResult:
    markdown = f"```sql\n{query}\n```"
    plain    = f"Here is the query:\n{query}"

    return types.CallToolResult(
        content=[
            # What the LLM actually sees
            types.TextContent(
                type="text",
                text=plain,
                annotations=types.Annotations(audience=["assistant"])
            )
        ],
        _meta={                       # <-- GUI can inspect this
            "renderAs": "markdown",   #   (any structure you like)
            "language": "sql"
        }
    )

Why this is not hacky – The two official demo servers use the same trick:

    annotatedMessage in the “everything” reference server stores display hints in annotations
    GitHub

    The multi-tenant auth proposal sends OAuth tokens in _meta on tools/call and nobody objected to the field itself – only to auth semantics
    GitHub

2.2 Alternative: embed a real resource with a MIME type

If your GUI does MIME sniffing, wrap the text in an EmbeddedResource with mimeType="text/markdown":

types.EmbeddedResource(
    type="resource",
    resource=types.TextResourceContents(
        uri="inline://result.md",
        mimeType="text/markdown",
        text=markdown
    ),
    annotations=types.Annotations(audience=["user"])
)

Because TextResourceContents carries an explicit MIME type, clients can render it the same way they already handle attached PDFs or images
GitHub
.
3 Client side

def handle_tool_result(result: types.CallToolResult):
    gui_hint = (result._meta or {}).get("renderAs")

    for c in result.content:
        if isinstance(c, types.TextContent):
            if gui_hint == "markdown" or getattr(c, "annotations", {}).get("renderAs") == "markdown":
                render_markdown(c.text)
            else:
                render_plain(c.text)

        elif isinstance(c, types.EmbeddedResource):
            # honour mimeType if present
            if getattr(c.resource, "mimeType", "") == "text/markdown":
                render_markdown(c.resource.text)
            else:
                render_blob(c.resource)

The same client logic will also pick up future keys because both _meta and annotations are schemaless.
4 Why this matches the spirit of MCP

    Keeps the LLM context pristine – the model still receives ordinary text; only the GUI looks at the hints.

    Survives relaying – other MCP servers/routers that forward the message untouched will preserve your metadata because the spec forbids stripping unknown fields
    GitHub
    .

    Aligns with existing guidance – The spec calls annotations a “powerful feature not prominently documented” to filter or prioritise content for user vs. assistant
    LLMindset.co.uk
    .

    Tool results are already meant to be richly typed – return arrays of TextContent, ImageContent, or EmbeddedResource; no extra extension is required
    GitHub
    .

    Community precedent – multiple real-world servers and design-notes blogs rely on _meta and annotations exactly this way
    GitHub
    GitHub
    GitHub
    .

5 Practical checklist
Goal	Quick recipe
Render block as Markdown	result._meta["renderAs"]="markdown" or TextContent.annotations.renderAs="markdown"
Tell GUI to open a “Python” tab	result._meta={"renderAs":"code","language":"python"}
Return both a plain explanation and a downloadable notebook	one TextContent + one EmbeddedResource with mimeType="application/x-ipynb+json"
Hide debug info from user but keep for the assistant	annotations={"audience":["assistant"],"priority":0.2}
References

    Result base class exposes _meta for “additional metadata”
    GitHub

    CallToolResult carries an array of content objects
    GitHub

    Issue #264 confirms _meta is intended for arbitrary metadata
    GitHub

    Community proposal uses _meta.authorization in the wild
    GitHub

    annotations field definition (audience, priority, extra keys allowed)
    modelcontextprotocol.github.io

    “Everything” demo server shows annotations in practice
    GitHub

    LLMindset article highlights Audience/priority annotations as a core but under-advertised feature
    LLMindset.co.uk

    EmbeddedResource + TextResourceContents.mimeType for MIME-aware UIs
    GitHub

    Tools concept doc: tools return rich results, clients decide how to render
    Home

    Base models all use extra="allow" (future-proofing unknown keys)
    GitHub

With these two sanctioned hooks you can send Markdown vs. code vs. raw JSON hints today without waiting for any new MCP revision—and every compliant client will simply ignore the hint if it doesn’t care, exactly as the spec intends.
