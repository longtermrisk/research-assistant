import sys
import os

from dotenv import load_dotenv
import subprocess

from server import mcp



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
    activate_venv()
    from terminal import sessions, AsyncPexpectProcess
    from jupyter import jupyter # noqa: F401 – imported for side effects
    import editor # noqa: F401 – imported for side effects


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