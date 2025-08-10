# Light-RAG

A lightweight Retrieval-Augmented Generation (RAG) system inspired by attention mechanisms.

## Overview

Light-RAG implements a novel approach to document retrieval where:
1. **During ingestion**: Documents are processed to generate "keys" (currently just the document content in MVP)
2. **During retrieval**: Queries are compared against these keys using embedding similarity

## Quick Start

### Prerequisites

1. Install dependencies:
   ```bash
   uv sync
   ```

2. Set up OpenAI API key:
   ```bash
   export OPENAI_API_KEY="your-api-key-here"
   ```

### Basic Usage


```python
import asyncio
from rag import FileSystemStore, OpenAIEmbedder, NoAdditionalKeys, Document
from dtypes import TextBlock

async def main():
    # Create a persistent store
    embedder = OpenAIEmbedder()
    key_generator = NoAdditionalKeys()
    
    # Create new store or load existing one
    store_dir = "my_knowledge_base"
    if os.path.exists(store_dir):
        store = FileSystemStore.load(store_dir)
        print("Loaded existing knowledge base")
    else:
        store = FileSystemStore.create(embedder, key_generator, store_dir)
        print("Created new knowledge base")
    
    # Add documents (persisted automatically)
    doc = Document.from_text("RAG combines retrieval with generation.")
    await store.ingest(doc)
    
    # Query works the same
    results = await store.query([TextBlock(text="What is RAG?")])
    for doc in results:
        print(f"Found: {doc.meta.title}")

asyncio.run(main())
```
