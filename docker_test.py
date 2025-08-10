#!/usr/bin/env python3
"""
Docker-specific test script to verify automator installation with RAG hooks.
"""
import asyncio
import os
import sys
import json
from pathlib import Path

def test_imports():
    """Test that all required packages can be imported."""
    print("Testing imports...")
    try:
        import automator
        from automator import Agent, Workspace
        print("✅ Core automator imports successful")
        
        import rag
        from rag import FileSystemStore, create_rag_hook, Document
        print("✅ RAG imports successful")
        
        from localrouter import TextBlock, MessageRole
        print("✅ LocalRouter imports successful")
        
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_rag_with_hook():
    """Test RAG functionality including hook system."""
    print("\nTesting RAG functionality with hooks...")
    try:
        from rag import MemoryRagStore, NoAdditionalKeys, StandardReranker, Document, Embedding
        from localrouter import TextBlock
        import numpy as np
        
        # A corrected dummy embedder that creates valid Embedding objects
        class DummyEmbedder:
            async def get_embedding(self, content_blocks):
                return [
                    Embedding(
                        content=[block],
                        embedding=np.random.rand(384).tolist(),
                        embedder_id="dummy_embedder",
                    )
                    for block in content_blocks
                ]
        
        embedder = DummyEmbedder()
        annotator = NoAdditionalKeys()
        reranker = StandardReranker()
        store = MemoryRagStore(embedder=embedder, annotator=annotator, reranker=reranker)
        
        # Create and ingest test documents
        docs = [
            Document.from_text("Python is a programming language.", source="python.txt"),
            Document.from_text("Docker is a containerization platform.", source="docker.txt"),
            Document.from_text("RAG combines retrieval with generation.", source="rag.txt")
        ]
        
        for doc in docs:
            await store.ingest(doc)
        
        print(f"✅ Ingested {len(docs)} test documents")
        
        # Test querying
        results = await store.query([TextBlock(text="What is Python?")], top_k=2)
        
        if results and results[0].document and results[0].document.meta.get('source') == 'python.txt':
            print(f"✅ RAG query returned {len(results)} results, correctly ordered.")
            return True
        elif results:
            print(f"❌ RAG query returned {len(results)} results, but ordering is incorrect. Top result source: {results[0].document.meta.get('source') if results[0].document else 'N/A'}")
            return False
        else:
            print("❌ RAG store returned no results")
            return False
            
    except Exception as e:
        print(f"❌ RAG test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_workspace_setup():
    """Test workspace creation and structure."""
    print("\nTesting workspace setup...")
    try:
        from automator import Workspace
        
        # Test workspace creation
        workspace = Workspace('docker_test')
        print(f"✅ Workspace created at: {workspace.root}")
        
        # Check if .automator directory exists
        automator_dir = Path.home() / '.automator'
        if automator_dir.exists():
            print(f"✅ .automator directory exists: {automator_dir}")
            
            # Check for prompts
            prompts_dir = automator_dir / 'prompts'
            if prompts_dir.exists():
                prompt_files = list(prompts_dir.glob('*.yaml'))
                if prompt_files:
                    print(f"✅ Found {len(prompt_files)} prompt templates")
                else:
                    print("⚠️  Prompts directory is empty")
                    return False
            else:
                print("❌ Prompts directory not found")
                return False
        else:
            print("❌ .automator directory not found")
            return False
        return True
        
    except Exception as e:
        print(f"❌ Workspace test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_mcp_configuration():
    """Test MCP server configuration."""
    print("\nTesting MCP configuration...")
    try:
        mcp_path = Path.home() / 'mcp.json'
        
        if not mcp_path.exists():
            print("❌ MCP config file not found")
            return False
            
        with open(mcp_path) as f:
            config = json.load(f)
        
        servers = config.get('mcpServers', {})
        expected_servers = ['terminal', 'web', 'talk2model']
        found_servers = 0
        
        for server in expected_servers:
            if server in servers:
                print(f"✅ {server} MCP server configured")
                found_servers += 1
            else:
                print(f"❌ {server} MCP server not found in config")

        return found_servers == len(expected_servers)
        
    except Exception as e:
        print(f"❌ MCP config test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Run all tests."""
    print("=== Docker Automator Installation Test ===\n")
    
    tests = {
        "Imports": test_imports,
        "RAG with Hook": test_rag_with_hook,
        "Workspace Setup": test_workspace_setup,
        "MCP Configuration": test_mcp_configuration,
    }
    
    results = {}
    
    for name, func in tests.items():
        if asyncio.iscoroutinefunction(func):
            results[name] = await func()
        else:
            results[name] = func()

    print("\n--- TEST SUMMARY ---")
    passed_count = sum(1 for result in results.values() if result)
    total_count = len(results)

    for name, result in results.items():
        print(f"[{'PASS' if result else 'FAIL'}] {name}")

    print(f"\nResult: {passed_count}/{total_count} tests passed.")
    
    if passed_count == total_count:
        print("\n🎉 All tests passed! Docker installation is working correctly.")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed. Please check the Dockerfile and build logs.")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())