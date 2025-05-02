import sys
import os

from dotenv import load_dotenv

from server import mcp
from terminal import sessions, AsyncPexpectProcess
from jupyter import jupyter # noqa: F401 – imported for side effects
import editor # noqa: F401 – imported for side effects


if __name__ == "__main__":
    CWD = os.environ.get("CWD", os.getcwd())
    os.chdir(CWD)
    load_dotenv(os.path.join(CWD, ".env"))
    # Create .logs/ if not exists
    os.makedirs(os.path.join(CWD, ".logs"), exist_ok=True)
    print("Starting Terminal MCP Server…", file=sys.stderr)
    try:
        mcp.run(transport="stdio")
    except Exception as exc:
        print(f"Failed to run MCP server: {exc}", file=sys.stderr)
        sys.exit(1)
    finally:
        for tab_id, sess in list(sessions.items()):
            proc: AsyncPexpectProcess | None = sess.get("process")  # type: ignore
            if proc and proc._child.isalive():
                try:
                    proc._child.terminate(force=True)
                except Exception:
                    pass
        sessions.clear()