"""Test based on automator/examples/rag/rag_hook_example.py"""

import asyncio
import os
import tempfile
from pathlib import Path

import pytest

pytestmark = pytest.mark.asyncio

# Check if RAG dependencies are available
rag_available = True
try:
    from rag.hook import create_rag_hook
    from rag import Document
except ImportError:
    rag_available = False


@pytest.mark.skipif(not rag_available, reason="RAG dependencies not available")
async def test_rag_hook_functionality():
    """Test RAG hook functionality with needle-in-haystack example (like the original)."""
    from automator.agent import Agent
    from automator.workspace import Workspace
    from rag.hook import create_rag_hook
    
    with tempfile.TemporaryDirectory() as tmpdir:
        original_home = os.environ.get('HOME')
        try:
            os.environ['HOME'] = tmpdir
            
            # Create minimal prompts directory structure
            prompts_dir = Path(tmpdir) / '.automator' / 'prompts'
            prompts_dir.mkdir(parents=True, exist_ok=True)
            
            # Create assistant.yaml prompt file
            assistant_yaml_content = """messages:
  - role: system
    content: |
      You are a helpful assistant that can use tools and access documents.
  - role: user
    content: $query
"""
            (prompts_dir / 'assistant.yaml').write_text(assistant_yaml_content)
            
            # Create knowledge directory and needle-in-haystack document
            # Use ~/.automator/knowledge as in the original example
            knowledge_dir = Path(tmpdir) / '.automator' / 'knowledge'
            knowledge_dir.mkdir(parents=True, exist_ok=True)
            
            # Write the exact needle-in-haystack file as in the original
            needle_file = knowledge_dir / 'needle-in-haystack.md'
            needle_file.write_text(
                "If anyone asks for the value of the needle in the haystack, "
                "the answer should be apples-78-moonshine-apes"
            )
            
            # Create RAG hook for ~/.automator/knowledge (like the original)
            create_rag_hook('~/.automator/knowledge')
            
            # Create workspace and agent with RAG hook (like the original)
            workspace = Workspace('rag')
            
            # Create a minimal LLM.md file in the workspace to avoid the FileNotFoundError
            workspace_home = Path(tmpdir) / '.automator' / 'workspaces' / 'rag' / 'workspace'
            workspace_home.mkdir(parents=True, exist_ok=True)
            llm_md_file = workspace_home / 'LLM.md'
            llm_md_file.write_text("# Test Workspace\n\nThis is a test workspace for RAG functionality.")
            
            agent = Agent(
                llm=dict(model="gpt-4.1", max_tokens=32000),
                prompt_template_yaml="assistant.yaml",
                workspace=workspace,
                id='dev',
                hooks=['claude.md', 'rag:~/.automator/knowledge']
            )
            
            # Ask the question from the original example
            query = "what is the needle in the haystack"
            thread = await agent.run(query)
            
            # Consume all messages
            async for _ in thread:
                pass
            
            # Verify we have messages
            assert len(thread.messages) >= 2, "Thread should have at least 2 messages (user + assistant)"
            
            # Check that RAG retrieved the document (in metadata)
            rag_metadata_found = False
            retrieved_docs = []
            
            # Look for RAG metadata in messages (usually in the second-to-last message)
            for message in thread.messages:
                if hasattr(message, 'meta') and message.meta:
                    for key, value in message.meta.items():
                        if 'rag:' in key and isinstance(value, list):
                            retrieved_docs.extend(value)
                            if any("needle-in-haystack.md" in str(doc) for doc in value):
                                rag_metadata_found = True
            
            # Check that the correct answer appears in the response
            correct_answer_found = False
            final_response = ""
            
            # The last message should be the assistant's response
            if thread.messages:
                last_message = thread.messages[-1]
                for block in last_message.content:
                    if hasattr(block, 'text'):
                        final_response += block.text
                        if 'apples-78-moonshine-apes' in block.text:
                            correct_answer_found = True
            
            # Assertions like in the original example
            print(f"Retrieved documents: {retrieved_docs}")
            print(f"Final response excerpt: {final_response[:200]}...")
            
            assert rag_metadata_found, f"RAG did not retrieve needle-in-haystack.md. Retrieved: {retrieved_docs}"
            assert correct_answer_found, f"Agent response did not contain 'apples-78-moonshine-apes'. Response: {final_response}"
            
            await thread.cleanup()
            
        finally:
            if original_home:
                os.environ['HOME'] = original_home
            elif 'HOME' in os.environ:
                del os.environ['HOME']


@pytest.mark.skipif(not rag_available, reason="RAG dependencies not available")  
async def test_rag_document_ingestion():
    """Test basic RAG document ingestion functionality."""
    from rag import MemoryRagStore, NoAdditionalKeys, Document, OpenAIEmbedder, OpenAIReranker
    from localrouter import TextBlock
    
    # Create RAG store components
    embedder = OpenAIEmbedder()
    annotator = NoAdditionalKeys()
    reranker = OpenAIReranker()
    store = MemoryRagStore(embedder=embedder, annotator=annotator, reranker=reranker)
    
    # Create test document
    doc = Document.from_text(
        "The secret code is banana-42-rainbow-cats",
        source="test-doc.txt"
    )
    
    # Ingest the document
    await store.ingest(doc)
    
    # Verify document was stored
    assert len(store.documents) == 1
    stored_doc = list(store.documents.values())[0]
    assert stored_doc.document.meta.source == "test-doc.txt"
    
    # Test querying
    results = await store.query([TextBlock(text="What is the secret code?")], top_k=1)
    
    # Verify we got results
    assert len(results) > 0
    assert results[0].meta.source == "test-doc.txt"
    
    # Check if the content contains the secret code
    result_text = ""
    for block in results[0].content:
        if hasattr(block, 'text'):
            result_text += block.text
    assert "banana-42-rainbow-cats" in result_text