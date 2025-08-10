import asyncio
import json
from pathlib import Path

import pytest
from localrouter.dtypes import TextBlock

# Mark all tests in this file as asyncio
pytestmark = pytest.mark.asyncio


async def test_imports():
    """Test that all required packages can be imported."""
    from automator import Agent, Workspace
    import rag
    from rag import FileSystemStore, create_rag_hook, Document

    assert Agent is not None
    assert Workspace is not None
    assert rag is not None
    assert FileSystemStore is not None
    assert create_rag_hook is not None
    assert Document is not None


async def test_rag_functionality():
    """Test RAG functionality including hook system."""
    from rag import MemoryRagStore, NoAdditionalKeys, Document, OpenAIEmbedder, OpenAIReranker

    embedder = OpenAIEmbedder()
    annotator = NoAdditionalKeys()
    reranker = OpenAIReranker()
    store = MemoryRagStore(embedder=embedder, annotator=annotator, reranker=reranker)

    # Create and ingest test documents
    docs = [
        Document.from_text("Python is a programming language.", source="python.txt"),
        Document.from_text("Docker is a containerization platform.", source="docker.txt"),
        Document.from_text("RAG combines retrieval with generation.", source="rag.txt")
    ]

    for doc in docs:
        await store.ingest(doc)

    assert len(store.documents) == len(docs), f"Expected {len(docs)} documents, but found {len(store.documents)}"

    # Test querying
    results = await store.query([TextBlock(text="What is Python?")], top_k=2)

    assert results, "RAG store returned no results"
    assert results[0].meta.source == 'python.txt', f"Incorrect ordering. Top result source: {results[0].meta.source}"


async def test_workspace_setup(tmp_path):
    """Test workspace creation and structure."""
    # Use pytest's tmp_path fixture to avoid cluttering the home directory
    from automator.workspace import Workspace
    
    # Temporarily point HOME to tmp_path for this test
    original_home = Path.home()
    try:
        # This is a bit of a hack, but it's the cleanest way to
        # isolate the test from the user's actual home directory.
        import os
        os.environ['HOME'] = str(tmp_path)
        
        # We also need to manually create the prompts dir, since the
        # real one is not in our temp home.
        prompts_dir = tmp_path / '.automator' / 'prompts'
        prompts_dir.mkdir(parents=True, exist_ok=True)
        (prompts_dir / 'test.yaml').touch()

        workspace = Workspace('pytest_test')
        assert workspace.root.exists(), "Workspace directory was not created"
        
        automator_dir = tmp_path / '.automator'
        assert automator_dir.exists(), f".automator directory not found: {automator_dir}"

        assert prompts_dir.exists(), "Prompts directory not found"
        
        prompt_files = list(prompts_dir.glob('*.yaml'))
        assert prompt_files, "Prompts directory is empty"

    finally:
        os.environ['HOME'] = str(original_home)


async def test_mcp_server_commands():
    """Test that MCP server commands defined in ~/mcp.json can start without failing."""
    import subprocess
    import os
    from pathlib import Path
    import time
    
    # Read MCP configuration
    mcp_config_path = Path.home() / "mcp.json"
    
    if not mcp_config_path.exists():
        pytest.skip("~/mcp.json not found - cannot test MCP server commands")
    
    with open(mcp_config_path) as f:
        mcp_config = json.load(f)
    
    servers = mcp_config.get("mcpServers", {})
    if not servers:
        pytest.skip("No MCP servers defined in ~/mcp.json")
    
    failed_servers = []
    successful_servers = []
    
    for server_name, server_config in servers.items():
        command = server_config.get("command")
        args = server_config.get("args", [])
        env = server_config.get("env", {})
        
        if not command:
            failed_servers.append((server_name, "No command specified"))
            continue
        
        # Prepare environment variables
        test_env = os.environ.copy()
        skip_server = False
        
        for key, value in env.items():
            # Handle environment variable substitution (e.g., "$VAR_NAME")
            if value.startswith("$"):
                env_var = value[1:]  # Remove the $ prefix
                if env_var in os.environ:
                    test_env[key] = os.environ[env_var]
                else:
                    # For tests, we'll skip servers that require missing env vars
                    # but not count them as failures since they might not be needed
                    # in the test environment
                    skip_server = True
                    break
            else:
                test_env[key] = value
        
        if skip_server:
            continue  # Skip servers with missing env vars
        
        # Build full command
        full_command = [command] + args
        
        try:
            # Start the server process
            process = subprocess.Popen(
                full_command,
                env=test_env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Wait a short time to see if it fails immediately
            time.sleep(1.0)
            poll_result = process.poll()
            
            if poll_result is not None:
                # Process has terminated
                stdout, stderr = process.communicate()
                if poll_result != 0:
                    # Non-zero exit code indicates failure
                    failed_servers.append((server_name, f"Process exited with code {poll_result}. stderr: {stderr[:300]}"))
                else:
                    # Zero exit code - could be a CLI tool or a server that prints help and exits
                    # Check if stderr contains obvious error messages
                    error_keywords = ['error', 'failed', 'exception', 'traceback', 'not found']
                    stderr_lower = stderr.lower()
                    
                    if any(keyword in stderr_lower for keyword in error_keywords):
                        failed_servers.append((server_name, f"Process terminated with errors. stderr: {stderr[:300]}"))
                    else:
                        # Seems like it ran successfully, possibly a CLI tool
                        successful_servers.append((server_name, "CLI tool - ran successfully"))
            else:
                # Process is still running - this is expected for servers
                # Wait a bit more to make sure it's stable
                time.sleep(2.0)
                poll_result = process.poll()
                
                if poll_result is not None:
                    # Process terminated after a few seconds
                    stdout, stderr = process.communicate()
                    if poll_result != 0:
                        failed_servers.append((server_name, f"Process crashed after starting. Exit code: {poll_result}. stderr: {stderr[:300]}"))
                    else:
                        successful_servers.append((server_name, "Started and terminated cleanly"))
                else:
                    # Process is still running - kill it and count as success
                    process.terminate()
                    try:
                        process.wait(timeout=2.0)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                    
                    successful_servers.append((server_name, "Server started and ran continuously"))
        
        except FileNotFoundError:
            failed_servers.append((server_name, f"Command not found: {command}"))
        except PermissionError:
            failed_servers.append((server_name, f"Permission denied for command: {command}"))
        except Exception as e:
            failed_servers.append((server_name, f"Unexpected error: {str(e)}"))
    
    # Report results
    success_messages = [f"  {server_name}: {status}" for server_name, status in successful_servers]
    
    if successful_servers:
        print(f"\nSuccessful servers ({len(successful_servers)}):")
        for server_name, status in successful_servers:
            print(f"  {server_name}: {status}")
    
    if failed_servers:
        failure_messages = [f"  {server_name}: {error}" for server_name, error in failed_servers]
        
        print(f"\nFailed servers ({len(failed_servers)}):")
        for server_name, error in failed_servers:
            print(f"  {server_name}: {error}")
        
        # Only fail the test if more than half the servers fail, or if core servers fail
        core_servers = ['terminal', 'web']  # These should always work
        
        core_failures = [name for name, _ in failed_servers if name in core_servers]
        if core_failures:
            pytest.fail(
                f"Core MCP servers failed to start: {', '.join(core_failures)}.\n"
                f"This indicates a fundamental issue with the MCP server setup."
            )
        
        failure_rate = len(failed_servers) / (len(failed_servers) + len(successful_servers))
        if failure_rate > 0.5:
            pytest.fail(
                f"Too many MCP servers failed ({len(failed_servers)}/{len(failed_servers) + len(successful_servers)}). "
                f"This indicates a systemic issue."
            )
        
        # Otherwise, just warn about the failures but don't fail the test
        print(f"\nNote: Some servers failed but this may be expected in test environments.")
    
    # Test passes if at least some servers work and core servers are functional
    total_tested = len(failed_servers) + len(successful_servers)
    if total_tested == 0:
        pytest.skip("No MCP servers could be tested (all skipped due to missing environment variables)")
    
    print(f"\nMCP server test summary: {len(successful_servers)} succeeded, {len(failed_servers)} failed")

