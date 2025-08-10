from . import FileSystemStore, OpenAIEmbedder, OpenAIAnnotator, OpenAIReranker
from automator.hooks import register_hook
from localrouter import TextBlock, MessageRole
from pathlib import Path
import os


_stores = {}


def get_store(absolute_path):
    if absolute_path in _stores:
        return _stores[absolute_path]
    try:
        store = FileSystemStore.load(absolute_path / '.rag')
    except ValueError:
        store = FileSystemStore.create(
            embedder=OpenAIEmbedder(),
            annotator=OpenAIAnnotator(n_keys=5),
            reranker=OpenAIReranker(),
            root_dir=absolute_path / '.rag',
            n_keys_per_doc=5,
            top_k_retrieval=100,
        )
    # Set context to workspace / LLM.md if it exists    
    _stores[absolute_path] = store
    return store


def set_context(thread, store):
    if thread.home / 'LLM.md':
        llm_dot_md = (thread.home / 'LLM.md').read_text()
        store.context = [TextBlock(text=llm_dot_md)]


def create_rag_hook(path='.knowledge', n_retrieved_docs=10):
    @register_hook(f'rag:{path}')
    async def rag_hook(thread):
        """Hook to check if any documents in the store are relevant to the current thread."""
        # Check if the last message contains text from the user (we skip otherwise)
        if not (
            thread.messages[-1].role == MessageRole.user and 
            any([isinstance(block, TextBlock) for block in thread.messages[-1].content])
        ):
            return
        print("RAG HOOK ACTIVE")
        # Get the store
        if path.startswith('/'):
            absolute_path = Path(path)
        elif path.startswith('~'):
            absolute_path = Path(os.path.expanduser(path))
        else:
            absolute_path = thread.home / path
        store = get_store(absolute_path)
        await store.ingest_dir(absolute_path, existing_source='update')
        set_context(thread, store)
        # Retrieve documents
        docs = await store.query(history=thread.messages_after_hooks, top_k=5)
        thread.messages[-1].meta[f'rag:{path}'] = [doc.meta.source for doc in docs]
        # Filter only the latest {n_retrieved_docs}
        all_doc_ids = []
        for message in thread.messages[::-1]:
            all_doc_ids += [doc_id for doc_id in message.meta.get(f'rag:{path}', []) if not doc_id in all_doc_ids]
        all_doc_ids = all_doc_ids[:n_retrieved_docs]

        # Add docs to the history at the first time at which they have been retrieved
        handled_doc_ids = []
        for original, processed in zip(thread.messages, thread.messages_after_hooks):
            for doc_id in original.meta.get(f'rag:{path}', []):
                if doc_id in handled_doc_ids:
                    continue
                doc = store.get_by_source(doc_id)
                if doc_id in all_doc_ids:
                    processed.content += [TextBlock(text=f'<document src="{doc.meta.source}">')] + doc.content + [TextBlock(text=f"</document>")]
                else:
                    # doc was open but has since been replaced by newer retrieved docs
                    processed.content += [TextBlock(text=f'<document src="{doc.meta.source}">\n{doc.meta.summary}\n(This document has been summarized since it might not be relevant for the current task anymore.)\n</document>')]
                handled_doc_ids.append(doc_id)

      
create_rag_hook('.')
create_rag_hook('.knowledge')
create_rag_hook('~/.automator/knowledge')