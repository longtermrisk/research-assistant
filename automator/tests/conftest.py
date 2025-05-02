import pytest
import os
import tempfile
from pathlib import Path


@pytest.fixture
def temp_workspace_dir():
    """Create a temporary directory for workspace testing"""
    with tempfile.TemporaryDirectory() as tmpdirname:
        old_home = os.environ.get("HOME")
        try:
            # Set HOME to temp dir so ~/.automator points to the test directory
            os.environ["HOME"] = tmpdirname
            workspace_dir = Path(tmpdirname) / ".automator" / "workspaces"
            workspace_dir.mkdir(parents=True, exist_ok=True)
            yield workspace_dir
        finally:
            # Restore original HOME
            if old_home:
                os.environ["HOME"] = old_home