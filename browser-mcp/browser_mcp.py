import asyncio
import datetime as dt
from typing import List, Literal, Optional, Union

import base64
import regex as re
from pydantic import BaseModel, Field
from pyppeteer import launch
from pyppeteer.errors import ElementHandleError

from mcp.server.fastmcp import FastMCP, Context
from mcp.types import TextContent, ImageContent

# ─────────────────────────────── Request model ────────────────────────────────
class Interaction(BaseModel):
    action: Literal[
        "open_url", "key", "type", "mouse_move", "left_click",
        "left_click_drag", "right_click", "double_click", "console",
        "scroll_into_view",
    ]
    coordinate: Optional[List[int]] = Field(
        None, description="(x, y) pixels. Needed for mouse actions.")
    text: Optional[str] = Field(
        None, description="Payload for open_url / type / key / console / scroll_into_view.")

# ─────────────────────────────── Browser helper ───────────────────────────────
class Browser:
    """
    Thin wrapper around pyppeteer that understands `Interaction` dictionaries
    and records console + network logs.
    """

    def __init__(self) -> None:
        self.browser = None
        self.page = None
        self.logs: list[str] = []
        self.current_url: Optional[str] = None
        self.watch_network_traffic = r"http://localhost/.*"
        self.watch_console_logs = False
        self.coordinate = [640, 400]          # default mouse pos

    # ---------------- core primitives ----------------
    async def open_url(self, url: str) -> None:
        self.current_url = url
        if self.browser is None:
            self.browser = await launch(
                headless=True,
                handleSIGINT=False, handleSIGTERM=False, handleSIGHUP=False,
                args=["--window-size=2560,1600"],
            )
        self.page = await self.browser.newPage()
        await self.page.setViewport({"width": 1280, "height": 800, "deviceScaleFactor": 1.0})
        self.page.on("console", self._on_console_message)
        self.page.on("request", self._on_network_request)
        self.page.on("response", self._on_network_response)
        await self.page.goto(url)

    async def interact(self, ixn: dict) -> None:
        # remember latest coordinate unless overridden
        if ixn.get("coordinate"):
            self.coordinate = ixn["coordinate"]
        else:
            ixn["coordinate"] = self.coordinate

        if ixn["action"] == "open_url":
            await self.open_url(ixn["text"])
            return

        if self.page is None:
            raise RuntimeError("open_url must be called before other actions")

        # --- keyboard / mouse / console handling ---
        if ixn["action"] == "type":
            await self.page.keyboard.type(ixn["text"])

        elif ixn["action"] == "mouse_move":
            x, y = ixn["coordinate"]
            await self.page.mouse.move(x, y)

        elif ixn["action"] == "left_click":
            x, y = ixn["coordinate"]
            await self.page.mouse.click(x, y)

        elif ixn["action"] == "left_click_drag":
            dest_x, dest_y = ixn["coordinate"]
            await self.page.mouse.down()
            await self.page.mouse.move(dest_x, dest_y, {"steps": 50})
            await self.page.mouse.up()

        elif ixn["action"] == "right_click":
            x, y = ixn["coordinate"]
            await self.page.mouse.click(x, y, {"button": "right"})

        elif ixn["action"] == "double_click":
            x, y = ixn["coordinate"]
            await self.page.mouse.click(x, y, {"clickCount": 2})

        elif ixn["action"] == "key":
            keys = ixn["text"].split("+")
            for k in keys[:-1]:
                await self.page.keyboard.down(k)
            await self.page.keyboard.press(keys[-1])
            for k in reversed(keys[:-1]):
                await self.page.keyboard.up(k)

        elif ixn["action"] == "console":
            js = f"() => {{ {ixn['text']} }}"
            await self.page.evaluate(js)

        elif ixn["action"] == "scroll_into_view":
            selector = ixn["text"]
            element = await self.page.querySelector(selector)
            await self.page.evaluate("el => el.scrollIntoView()", element)

        # --- wait briefly for idle network ---
        try:
            await self.page.waitForNavigation({"waitUntil": "networkidle0", "timeout": 2000})
        except Exception:
            pass  # ignore timeouts

    async def screenshot_with_mouse(self) -> str:
        if self.page is None:
            raise RuntimeError("No page open")
        # draw fake cursor
        x, y = self.coordinate
        await self.page.evaluate(
            """(x, y) => {
                const id='__custom_cursor'; document.getElementById(id)?.remove();
                const d = document.createElement('div'); d.id=id;
                d.style='position:fixed;z-index:9999;width:20px;height:20px;'
                       +'pointer-events:none;left:'+x+'px;top:'+y+'px;';
                d.innerHTML='<svg width="20" height="20" viewBox="0 0 20 20">'
                           +'<path d="M0 0 L12 12 L8 12 L10 16 L8 17 L6 13 L0 13 Z" '
                           +'fill="white" stroke="black" stroke-width="1"/></svg>';
                document.body.appendChild(d);
            }""",
            x,
            y,
        )
        png_b64 = await self.page.screenshot({"encoding": "base64"})
        # remove cursor
        await self.page.evaluate("""() => document.getElementById('__custom_cursor')?.remove()""")
        return png_b64

    # ---------------- log collectors ----------------
    def _on_console_message(self, msg):
        if self.watch_console_logs:
            self.logs.append(f"[{dt.datetime.now().isoformat()}] console: {msg.text}")

    def _on_network_request(self, req):
        if re.match(self.watch_network_traffic, req.url):
            line = f"[{dt.datetime.now().isoformat()}] ▶ {req.method} {req.url}"
            self.logs.append(line)

    def _on_network_response(self, res):
        if re.match(self.watch_network_traffic, res.url):
            line = f"[{dt.datetime.now().isoformat()}] ◀ {res.status} {res.url}"
            self.logs.append(line)

# ─────────────────────────────── MCP server ───────────────────────────────
mcp = FastMCP("Headless-Browser Server")
tab_map = {}
@mcp.tool(
    title="Interact with a headless browser",
    description=(
        "Stateful wrapper around pyppeteer. "
        "Use `tab_id` to address multiple tabs in the same MCP session. "
        "First call per tab must include action=open_url."
    ),
)
async def browser_tool(
    interactions: List[Interaction] = Field(..., description="Ordered list of actions."),
    tab_id: str = Field("default", description="Identifier for a tab inside this session."),
    watch_network_traffic: str = Field(
        r"http://localhost/.*", description="Regex for URLs to log."
    ),
    watch_console_logs: bool = Field(False, description="Capture console.* output."),
    ctx: Context = None,
) -> List[object]:
    """
    Executes `interactions` against the tab identified by `tab_id`.
    Returns dev-tools logs and a final screenshot (PNG base64).
    """
    # ---- retrieve or create Browser for this tab ----
    browser = tab_map.get(tab_id)
    if browser is None:
        browser = Browser()
        tab_map[tab_id] = browser

    browser.watch_network_traffic = watch_network_traffic
    browser.watch_console_logs = watch_console_logs

    blocks: list[object] = []

    for ixn in interactions:
        try:
            await browser.interact(ixn.dict())
        except Exception as exc:
            blocks.append(TextContent(type="text", text=f"Error: {exc}"))
            break

    # dev-tools style logs
    if browser.logs:
        blocks.append(TextContent(type="text", text="\n".join(browser.logs)))
        browser.logs.clear()

    # screenshot
    try:
        img_b64 = await browser.screenshot_with_mouse()
        # Convert base64 string to bytes
        blocks.append(ImageContent(type="image", mimeType="image/png", data=img_b64))
        # blocks.append(Image(data=img_bytes, format="png"))
    except Exception as exc:
        blocks.append(TextContent(type="text", text=f"Screenshot failed: {exc}"))

    return blocks

# ─────────────────────────────── entry point ───────────────────────────────
if __name__ == "__main__":
    # Starts a JSON-RPC loop on stdin/stdout (works with `mcp dev` or Claude Desktop)
    mcp.run()
