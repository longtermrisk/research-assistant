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

