import asyncio
import uuid
import sys
import os
import re
import random
from collections import deque
from typing import Dict, Any, List, Tuple

from mcp.types import TextContent
import pexpect  # pseudo‑terminal handling
from utils import clean_ansi 
from output_manager import truncate_output

from server import mcp


# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------
MAX_LOG_BUFFER_BEFORE_DETACH = 1000  # circular buffer size per session

PROMPT_PATTERNS: List[str] = [
    r"\[y/N\]|\[Y/n\]|\[y/n\]",  # y/n prompts
    r"\(Pdb\)",  # Python debugger
    r">>>\s*$",  # Python REPL
    r"In \[\d+\]:\s*$",  # IPython / Jupyter
    r"\?$",  # simple question prompts
    r"password:",
    r"Username:",
    r"\$\s*$",  # shell prompt
    r">\s*$",  # generic prompt
]
PROMPT_REGEX = re.compile("|".join(f"({p})" for p in PROMPT_PATTERNS))



def _get_env(cwd: str | None = None) -> Dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "PYTHONUNBUFFERED": "1",
            "PYTHONIOENCODING": "UTF-8",
            "TERM": "dumb",
            "NO_COLOR": "1",
        }
    )
    return env

# ---------------------------------------------------------------------------
# Async pexpect wrapper
# ---------------------------------------------------------------------------

class AsyncPexpectProcess:
    def __init__(self, command: str):
        venv_activate = "if [ -d .venv ]; then source .venv/bin/activate > /dev/null 2>&1; fi && "
        self._child = pexpect.spawn(
            "/bin/bash", ["-c", venv_activate + command], env=_get_env(), encoding="utf-8", codec_errors="replace"
        )
        self._child.setecho(False)
        self._log_buffer = ""  # newly captured since last read

    # --------------------------- basic helpers ---------------------------
    async def is_alive(self) -> bool:
        return await asyncio.to_thread(self._child.isalive)

    async def _read_nonblocking(self, size: int = 1024, timeout: float = 0.1) -> str:
        try:
            data: str = await asyncio.to_thread(self._child.read_nonblocking, size, timeout)
            self._log_buffer += data
            return data
        except (pexpect.TIMEOUT, pexpect.EOF):
            return ""

    # ------------------------ public read helpers -----------------------
    def take_pending_output(self) -> str:
        """Return everything accumulated since last call and clear buffer."""
        buf, self._log_buffer = self._log_buffer, ""
        return buf

    # ----------------------------- writing ------------------------------
    async def send_line(self, line: str) -> None:
        await asyncio.to_thread(self._child.sendline, line)

    async def send_raw(self, data: str) -> None:
        await asyncio.to_thread(self._child.send, data)

    # --------------- wait until prompt / eof / timeout ------------------
    async def stream_until_prompt_or_eof(self, *, timeout: float) -> Tuple[str, bool, bool]:
        deadline = asyncio.get_event_loop().time() + timeout
        collected: List[str] = []
        prompt_seen = False
        eof = False
        while True:
            remaining = deadline - asyncio.get_event_loop().time()
            if remaining <= 0:
                break
            chunk = await self._read_nonblocking(timeout=min(0.1, remaining))
            
            # While there is immediately available output, read it
            while (more := await self._read_nonblocking(timeout=0.1)) != "":
                chunk += more

            if chunk:
                collected.append(chunk)
                if PROMPT_REGEX.search(chunk):
                    prompt_seen = True
                    break
            if not await self.is_alive():
                eof = True
                break
        return "".join(collected), prompt_seen, eof

# ---------------------------------------------------------------------------
# Global server state
# ---------------------------------------------------------------------------

sessions: Dict[str, Dict[str, Any]] = {}

# ---------------------------------------------------------------------------
# Interactive helpers
# ---------------------------------------------------------------------------

def _interactive_mapping(latest_output: str) -> Dict[str, str]:
    mapping = {"enter": "\r", "return": "\r", "up": "\x1b[A", "down": "\x1b[B"}
    if "↓" in latest_output or "↑" in latest_output:
        mapping.update({str(i): "\x1b[B" * (i - 1) + "\r" for i in range(1, 10)})
    return mapping


async def _send_interactive_text(process: AsyncPexpectProcess, text: str) -> None:
    latest = process.take_pending_output()  # flush any echoes before deciding mapping
    mapping = _interactive_mapping(latest)
    sequence = mapping.get(text.lower(), text + "\r")

    if "\x1b" in sequence:  # contains arrow keys or other control codes – send raw
        await process.send_raw(sequence)
    else:  # simple line – send once without the terminating CR (send_line adds it)
        await process.send_line(sequence.rstrip("\r"))
    # allow terminal to react a tiny bit
    await asyncio.sleep(0.05)

# ---------------------------------------------------------------------------
# Tool helpers – shared behaviour between execute & write_to_stdin
# ---------------------------------------------------------------------------

def _format_response(new_output: str, *, prompt_seen: bool, eof: bool, tab_id: str) -> str:
    if eof:
        return new_output
    if prompt_seen:
        return new_output + f"\n(Process is waiting for input in tab: {tab_id}. Use `terminal_stdin` to interact.)"
    return new_output + f"\n... (The process continues to run in the background in tab: {tab_id})"

# ---------------------------------------------------------------------------
# MCP TOOLS
# ---------------------------------------------------------------------------

@mcp.tool()
async def terminal_execute(command: str, detach_after_seconds: float = 30.0) -> str:
    """Start a new terminal session and execute a command in it."""
    tab_id = str(uuid.uuid4())[:5]
    buffer: deque[str] = deque()
    process = AsyncPexpectProcess(command)
    sessions[tab_id] = {"process": process, "logs": buffer, "command": command}

    return await _handle_terminal_output(
        process=process,
        tab_id=tab_id,
        detach_after_seconds=detach_after_seconds,
        log_prefix="terminal_execute",
        log_args=command,
    )


@mcp.tool()
async def terminal_stdin(
    tab_id: str,
    text: str,
    detach_after_seconds: float = 30.0,
) -> str:
    """Send text to the stdin of a running terminal session. Send '^C' to interrupt."""
    session = sessions.get(tab_id)
    if not session:
        return f"[MCP_SERVER_ERROR] Tab ID '{tab_id}' not found."

    process: AsyncPexpectProcess | None = session.get("process")  # type: ignore
    if process is None or not await process.is_alive():
        return f"[MCP_SERVER_ERROR] Process for tab '{tab_id}' is not running."
    
    if text.strip() == "^C":  # special case for Ctrl+C
        try:
            await process.send_raw("\x03")  # Ctrl+C
            return f"[MCP_SERVER_INFO] Sent Ctrl+C to tab '{tab_id}'."
        except Exception as exc:
            return f"[MCP_SERVER_ERROR] Failed to send Ctrl+C to tab '{tab_id}': {exc}"

    try:
        await _send_interactive_text(process, text)
        return await _handle_terminal_output(
            process=process,
            tab_id=tab_id,
            detach_after_seconds=detach_after_seconds,
            log_prefix="terminal_stdin",
            log_args=text,
        )
    except Exception as exc:  # noqa: BLE001 – user‑visible error
        return f"[MCP_SERVER_ERROR] Failed to write to stdin for tab '{tab_id}': {exc}"


@mcp.tool()
async def terminal_logs(tab_id: str, number_of_lines: int = 1000, wait_seconds: float=0.1) -> str:
    """Get logs from a running terminal session."""
    session = sessions.get(tab_id)
    if not session:
        return f"[MCP_SERVER_ERROR] Tab ID '{tab_id}' not found."

    process: AsyncPexpectProcess | None = session.get("process")  # type: ignore
    try:
        output = await _handle_terminal_output(
            process=process,
            tab_id=tab_id,
            detach_after_seconds=wait_seconds,
            log_prefix="terminal_logs",
            log_args=str(number_of_lines),
            line_limit=number_of_lines,
        )
        return output
    except Exception as exc:  # noqa: BLE001 – user‑visible error
        return f"[MCP_SERVER_ERROR] Failed to write to stdin for tab '{tab_id}': {exc}"


async def _handle_terminal_output(
    process: AsyncPexpectProcess,
    tab_id: str,
    detach_after_seconds: float,
    log_prefix: str,
    log_args: str,
    line_limit: int | None = None,
) -> str:
    """Handle common terminal output processing and logging."""
    new_output, prompt_seen, eof = await process.stream_until_prompt_or_eof(timeout=detach_after_seconds)
    cleaned = clean_ansi(new_output)

    if line_limit is not None:
        cleaned = "\n".join(cleaned.splitlines()[-line_limit:])

    if eof:
        sessions[tab_id]["process"] = None

    # Apply output truncation with token counting
    truncated_output, full_output_path, was_truncated = truncate_output(cleaned)
    
    # Add find tool instructions if output was truncated
    if was_truncated:
        find_instructions = f"\n\n[Note: You can use the find tool to search through the full output: find(search_str=\"your_search\", path=\"{full_output_path}\")]"
        truncated_output += find_instructions

    response = _format_response(truncated_output, prompt_seen=prompt_seen, eof=eof, tab_id=tab_id)
    return TextContent(text=response, annotations={'display_html': f"<pre>{response}</pre>"}, type="text")
