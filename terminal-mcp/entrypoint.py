import sys
import os

from dotenv import load_dotenv
import subprocess
from pathlib import Path

from server import mcp


def _ensure_venv(path: Path) -> None:
    pyproject_path = path / "pyproject.toml"
    if not pyproject_path.exists():
        subprocess.run(["uv", "init"], check=True, cwd=path)
    venv_path = path / ".venv"
    if not venv_path.exists():
        subprocess.run(["uv", "venv"], check=True, cwd=path)
    # subprocess.run(["uv", "add", "ipykernel", "pip", "ipython", "jupyterlab", "plotly", "matplotlib"], check=True, cwd=path)
    subprocess.run(["uv", "add", "ipykernel", "pip", "ipython", "jupyterlab", "pandas", "matplotlib"], check=True, cwd=path)


def activate_venv():
    venv_activate = os.path.join(".venv", "bin", "activate")
    # Run a shell, source the activate script, then print the environment
    command = f"bash -c 'source \"{venv_activate}\" && env'"

    proc = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True, executable="/bin/bash")
    stdout, _ = proc.communicate()

    # Parse the environment variables
    env = {}
    for line in stdout.decode().splitlines():
        key, _, value = line.partition("=")
        env[key] = value

    # Update os.environ with the new environment
    os.environ.update(env)


if __name__ == "__main__":
    CWD = os.environ.get("CWD", os.getcwd())
    os.chdir(CWD)
    load_dotenv(os.path.join(CWD, ".env"))
    if os.environ.get("ENSURE_VENV", "TRUE").upper() == "TRUE":
        _ensure_venv(Path(CWD))
    if os.path.exists(os.path.join(CWD, ".venv")):
        activate_venv()
    from terminal import sessions, AsyncPexpectProcess
    from jupyter import jupyter # noqa: F401 – imported for side effects
    import editor # noqa: F401 – imported for side effects
    import find_tool # noqa: F401 – imported for side effects

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