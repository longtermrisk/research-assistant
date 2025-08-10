import asyncio
import os
import json
import sys
import numpy as np
from pydantic import BaseModel, Field
from typing import List, Optional, Callable, Dict, Type, Any
from localrouter import ContentBlock, TextBlock, ChatMessage, MessageRole, openai_format, messages_to_content_blocks
from localrouter import get_response_cached_with_backoff as get_response
import datetime as dt
from abc import ABC, abstractmethod
from uuid import uuid4
import openai
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import time
from rag.is_ignored import is_ignored
import tiktoken
from cache_on_disk import DCache


class DocumentMeta(BaseModel):
    created_at: dt.datetime = Field(default_factory=dt.datetime.now)
    updated_at: dt.datetime = Field(default_factory=dt.datetime.now)
    title: Optional[str] = None
    source: Optional[str] = None
    scope: str = "global"
    summary: Optional[str] = None


class Document(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    content: List[ContentBlock] = Field(default_factory=list)
    meta: DocumentMeta = Field(default_factory=DocumentMeta)

    @staticmethod
    def from_file(path: str):
        """Create a Document from a file path."""
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
        # Use absolute path for consistent source tracking
        abs_path = str(Path(path).resolve())
        return Document(
            content=[TextBlock(text=content)],
            meta=DocumentMeta(
                title=Path(path).name,
                source=abs_path
            )
        )

    @staticmethod
    def from_text(text: str, title: Optional[str] = None, source: Optional[str] = None):
        """Create a Document from text."""
        return Document(
            content=[TextBlock(text=text)],
            meta=DocumentMeta(
                title=title,
                source=source
            )
        )


class Embedding(BaseModel):
    embedder_id: str = Field(..., description="ID of the embedder that generated this embedding")
    embedding: List[float] = Field(..., description="The embedding vector")
    content: List[ContentBlock] = Field(..., description="The content that was embedded")


class DocumentWithEmbeddings(BaseModel):
    document: Document = Field(..., description="The document containing the content and metadata")
    keys: List[Embedding] = Field(default_factory=list)


class Embedder(ABC):
    """Base class for embedding models."""
    def __init__(self, id: str):
        self.id = id
    
    @abstractmethod
    async def get_embedding(self, content: List[ContentBlock]) -> Embedding:
        """Get the embedding for the given content."""
        pass
    
    @abstractmethod
    def to_config(self) -> Dict[str, Any]:
        """Serialize the embedder configuration."""
        pass


class AnnotationResult(BaseModel):
    """Result of document annotation containing keys and summary."""
    keys: List[List[ContentBlock]] = Field(..., description="List of key content blocks for retrieval")
    summary: Optional[str] = Field(None, description="Generated summary of the document")


class Annotator(ABC):
    """Base class for document annotators that generate keys and summaries."""
    def __init__(self, id: str):
        self.id = id
    
    @abstractmethod
    async def __call__(self, document: Document, context: List[ContentBlock] = None) -> AnnotationResult:
        """Annotate a document by generating keys and summary."""
        pass
    
    @abstractmethod
    def to_config(self) -> Dict[str, Any]:
        """Serialize the annotator configuration."""
        pass


# Registry for embedders, annotators, and rerankers
class ComponentRegistry:
    """Registry for embedders, annotators, and rerankers to enable serialization/deserialization."""
    
    def __init__(self):
        self._embedders: Dict[str, Type[Embedder]] = {}
        self._annotators: Dict[str, Type[Annotator]] = {}
        self._rerankers: Dict[str, Type[Reranker]] = {}
    
    def register_embedder(self, embedder_class: Type[Embedder]):
        """Register an embedder class."""
        # Use the class name as the registry key
        self._embedders[embedder_class.__name__] = embedder_class
        return embedder_class
    
    def register_annotator(self, annotator_class: Type[Annotator]):
        """Register an annotator class."""
        self._annotators[annotator_class.__name__] = annotator_class
        return annotator_class
    
    def register_reranker(self, reranker_class):
        """Register a reranker class."""
        self._rerankers[reranker_class.__name__] = reranker_class
        return reranker_class
    
    def create_embedder(self, embedder_id: str, config: Dict[str, Any]) -> Embedder:
        """Create an embedder instance from ID and configuration."""
        class_name = config.get('class_name')
        if class_name not in self._embedders:
            raise ValueError(f"Unknown embedder class: {class_name}")
        
        embedder_class = self._embedders[class_name]
        # Remove class_name from config before passing to constructor
        init_config = {k: v for k, v in config.items() if k != 'class_name'}
        return embedder_class(**init_config)
    
    def create_annotator(self, annotator_id: str, config: Dict[str, Any]) -> Annotator:
        """Create an annotator instance from ID and configuration."""
        class_name = config.get('class_name')
        if class_name not in self._annotators:
            raise ValueError(f"Unknown annotator class: {class_name}")
        
        annotator_class = self._annotators[class_name]
        # Remove class_name from config before passing to constructor
        init_config = {k: v for k, v in config.items() if k != 'class_name'}
        return annotator_class(**init_config)
    
    def create_reranker(self, reranker_id: str, config: Dict[str, Any]):
        """Create a reranker instance from ID and configuration."""
        class_name = config.get('class_name')
        if class_name not in self._rerankers:
            raise ValueError(f"Unknown reranker class: {class_name}")
        
        reranker_class = self._rerankers[class_name]
        # Remove class_name from config before passing to constructor
        init_config = {k: v for k, v in config.items() if k != 'class_name'}
        return reranker_class(**init_config)


# Create a cache instance for embeddings
_embedding_cache = DCache(cache_dir=os.path.expanduser("~/.cache/embeddings"))

@_embedding_cache
def _get_embedding_sync(text: str, model: str, api_key: Optional[str]):
    """Synchronous wrapper for OpenAI embeddings API with caching."""
    import openai
    client = openai.OpenAI(api_key=api_key)
    response = client.embeddings.create(
        model=model,
        input=text
    )
    return response.data[0].embedding


class OpenAIEmbedder(Embedder):
    """OpenAI embeddings implementation with disk caching."""
    
    def __init__(self, model: str = "text-embedding-3-small", api_key: Optional[str] = None):
        super().__init__(id=f"openai-{model}")
        self.model = model
        self.api_key = api_key

    def truncate_to_n_tokens(self, text, max_tokens=8192):
        enc = tiktoken.encoding_for_model(self.model)
        tokens = enc.encode(text)
        truncated_tokens = tokens[:max_tokens]
        return enc.decode(truncated_tokens)
    
    async def get_embedding(self, content: List[ContentBlock]) -> Embedding:
        """Get embedding from OpenAI API with disk caching."""
        # Convert content blocks to text
        text_parts = []
        for block in content:
            if isinstance(block, TextBlock):
                text_parts.append(block.text)
        text = "\n".join(text_parts)

        if not text.strip():
            raise ValueError("No text content found in content blocks")
        text = self.truncate_to_n_tokens(text)
        
        # Use cached embedding function
        embedding_vector = await asyncio.get_event_loop().run_in_executor(
            None, _get_embedding_sync, text, self.model, self.api_key
        )
        
        return Embedding(
            embedder_id=self.id,
            embedding=embedding_vector,
            content=content
        )
    
    def to_config(self) -> Dict[str, Any]:
        """Serialize the embedder configuration."""
        return {
            "class_name": "OpenAIEmbedder",
            "model": self.model,
            "api_key": self.api_key
        }


class NoAdditionalKeys(Annotator):
    """Annotator that returns no additional keys, only the document content, and no summary."""
    
    def __init__(self):
        super().__init__(id="no-additional-keys")
    
    async def __call__(self, document: Document, context: List[ContentBlock] = None) -> AnnotationResult:
        """Returns the document content as the only key and no summary."""
        return AnnotationResult(
            keys=[document.content],
            summary=None
        )
    
    def to_config(self) -> Dict[str, Any]:
        """Serialize the annotator configuration."""
        return {
            "class_name": "NoAdditionalKeys"
        }


class AnnotationOutput(BaseModel):
    """Structured output model for document annotation."""
    queries: List[str] = Field(..., description="List of hypothetical queries that should retrieve this document, ordered from most to least important")
    summary: str = Field(..., description="A concise summary of the document's main content and purpose")


class RerankingOutput(BaseModel):
    """Structured output model for document reranking."""
    sources: List[str] = Field(..., description="List of document sources ordered by relevance to the query, up to top_k items")


class OpenAIAnnotator(Annotator):
    """OpenAI-based annotator using structured outputs to generate hypothetical queries and summaries."""
    
    def __init__(self, model: str = "gpt-5-mini", n_keys: int = 10, api_key: Optional[str] = None):
        super().__init__(id=f"openai-annotator-{model}-{n_keys}")
        self.model = model
        self.n_keys = n_keys
        self.api_key = api_key
    
    async def __call__(self, document: Document, context: List[ContentBlock] = None) -> AnnotationResult:
        """Generate hypothetical queries and summary using localrouter with structured outputs."""
        # Convert document content to text
        text_parts = []
        for block in document.content:
            if isinstance(block, TextBlock):
                text_parts.append(block.text)
        
        document_text = "\n".join(text_parts)
        if not document_text.strip():
            return AnnotationResult(
                keys=[document.content],
                summary=None
            )  # Fallback to document content if no text
        
        # Convert context to text if provided
        context_content = []
        if context:
            context_content.extend(context)
        
        # Build the messages
        system_prompt = f"This document is being stored in a RAG database. Your task is to:\n1. Generate a list of up to {self.n_keys} hypothetical queries that should retrieve this document (ordered from most to least important)\n2. Create a concise summary of the document's main content and purpose"
        
        user_content = []
        if context_content:
            user_content.append(TextBlock(text="Here is some general context:"))
            user_content.extend(context_content)
            user_content.append(TextBlock(text=""))
        
        if document.meta.source:
            user_content.append(TextBlock(text=f"The document source is: {document.meta.source}"))
            user_content.append(TextBlock(text=""))
        
        user_content.append(TextBlock(text=f"Now, please generate keys and a summary for this content:\n{document_text}"))
        
        messages = [
            ChatMessage(role=MessageRole.system, content=[TextBlock(text=system_prompt)]),
            ChatMessage(role=MessageRole.user, content=user_content)
        ]
        
        try:
            # Use localrouter with structured outputs
            response = await get_response(
                model=self.model,
                messages=messages,
                response_format=AnnotationOutput,
                max_tokens=1500,
                cache_seed=0
            )
            
            annotation_output = response.parsed
            queries = annotation_output.queries
            summary = annotation_output.summary
            
            # Convert queries to ContentBlock lists
            key_content_lists = []
            for query in queries:
                if query.strip():  # Only include non-empty queries
                    key_content_lists.append([TextBlock(text=query)])
            
            # Always include the original document content as the first key
            keys = [document.content] + key_content_lists
            
            return AnnotationResult(
                keys=keys,
                summary=summary if summary.strip() else None
            )
            
        except Exception as e:
            print(f"Error generating annotation with localrouter: {e}", file=sys.stderr)
            # Fallback to document content only
            return AnnotationResult(
                keys=[document.content],
                summary=None
            )
    
    def to_config(self) -> Dict[str, Any]:
        """Serialize the annotator configuration."""
        return {
            "class_name": "OpenAIAnnotator",
            "model": self.model,
            "n_keys": self.n_keys,
            "api_key": self.api_key
        }


class Reranker(ABC):
    """Base class for document rerankers."""
    def __init__(self, id: str):
        self.id = id
    
    @abstractmethod
    async def rerank(self, query_content: List[ContentBlock], documents: List[Document], top_k: int, history: List[ChatMessage] | None = None) -> List[Document]:
        """Rerank documents based on query relevance and return top_k results."""
        pass
    
    @abstractmethod
    def to_config(self) -> Dict[str, Any]:
        """Serialize the reranker configuration."""
        pass


class StandardReranker(Reranker):
    """Standard reranker that simply returns the first top_k documents."""
    
    def __init__(self):
        super().__init__(id="standard-reranker")
    
    async def rerank(self, query_content: List[ContentBlock], documents: List[Document], top_k: int, history: List[ChatMessage] | None = None) -> List[Document]:
        """Return the first top_k documents without reranking."""
        return documents[:top_k]
    
    def to_config(self) -> Dict[str, Any]:
        """Serialize the reranker configuration."""
        return {
            "class_name": "StandardReranker"
        }


class OpenAIReranker(Reranker):
    """OpenAI-based reranker using structured outputs to intelligently rerank documents."""
    
    def __init__(self, model: str = "gpt-5", api_key: Optional[str] = None, system_prompt: str = None):
        super().__init__(id=f"openai-reranker-{model}")
        self.model = model
        self.api_key = api_key
        self.system_prompt = system_prompt or "You are a document reranker. Given a query and a list of documents with their sources, summaries, and content previews, rank the documents by relevance to the query. Return up to {top_k} document sources in order of relevance (most relevant first)."
    
    def content_to_text(self, content: List[ContentBlock]):
        text = ''
        for block in content:
            try:
                text += block.text
            except AttributeError:
                pass
        return text

    async def rerank(self, query_content: List[ContentBlock] | None, documents: List[Document], top_k: int, history: List[ChatMessage] | None = None) -> List[Document]:
        """Rerank documents using OpenAI structured outputs."""
        if not documents:
            return []
        
        if len(documents) <= top_k:
            return documents
        
        assert query_content is not None or history is not None

        # Build document information for reranking
        doc_info = []
        source_to_doc = {}
        
        # Find common prefix among sources
        sources = [doc.meta.source or f"document_{i}" for i, doc in enumerate(documents)]
        if sources and all(isinstance(s, str) for s in sources):
            # Find the common prefix path
            common_prefix = os.path.commonpath([s for s in sources if os.path.isabs(s)])
            # If there is a common prefix, we'll use relative paths
            use_relative = bool(common_prefix and common_prefix != '/')
        else:
            use_relative = False
            common_prefix = ''

        for i, doc in enumerate(documents):
            source = doc.meta.source or f"document_{i}"
            if use_relative and os.path.isabs(source):
                # Convert to relative path from common prefix
                display_source = os.path.relpath(source, common_prefix)
            else:
                display_source = source
            
            # Store mapping using display source
            source_to_doc[display_source] = doc
            
            # Get document text preview (first 500 chars)
            doc_text_parts = []
            for block in doc.content:
                if isinstance(block, TextBlock):
                    doc_text_parts.append(block.text)
            doc_text = "\n".join(doc_text_parts)
            doc_preview = doc_text[:500] + "..." if len(doc_text) > 500 else doc_text
            
            doc_info.append({
                "source": display_source,
                "title": doc.meta.title or "Untitled",
                "summary": doc.meta.summary or "No summary available", 
                "preview": doc_preview
            })

        docs_text = "Documents to rank:\n<documents>\n"
        for info in doc_info:
            docs_text += f"\nSource: {info['source']}\n"
            docs_text += f"Title: {info['title']}\n"
            docs_text += f"Summary: {info['summary']}\n"
            docs_text += f"Preview: {info['preview']}\n"
            docs_text += "---"
        docs_text += "\n</documents>\n"

        messages = [
            ChatMessage(
                role=MessageRole.system,
                content=[
                    TextBlock(text=self.system_prompt.format(top_k=top_k))
                ]
            )
        ]
        content = []
        if history is not None:
            content += (
                [TextBlock(text='<rollout>')] +
                messages_to_content_blocks(history) + 
                [TextBlock(text='</rollout>')]
            )
        
        if query_content is not None:
            content += (
                [TextBlock(text=f"Please select up to {top_k} documents relevant to the following query:\n<query>")] +
                query_content + 
                [TextBlock(text=f"</query>")]
            )
        else:
            content += [TextBlock(text=f"Please select up to {top_k} documents that are most likely to be helpful for the current task. If no further information is required for the current task, return an empty list.")]
        
        content += [TextBlock(text=docs_text)]
        messages.append(ChatMessage(role=MessageRole.user, content=content))

        try:
            # Use localrouter with structured outputs to get the reranked sources
            response = await get_response(
                model=self.model,
                messages=messages,
                response_format=RerankingOutput,
                max_tokens=4000,
                cache_seed=0
            )
            reranking_output = response.parsed
            ranked_sources = reranking_output.sources
            
            # Reorder documents based on the ranking
            reranked_docs = []
            used_sources = set()
            
            for source in ranked_sources:
                if source in source_to_doc and source not in used_sources:
                    reranked_docs.append(source_to_doc[source])
                    used_sources.add(source)
            return reranked_docs[:top_k]
            
        except Exception as e:
            print(f"Error reranking documents with localrouter: {e}", file=sys.stderr)
            # Fallback to standard reranking
            return documents[:top_k]
    
    def to_config(self) -> Dict[str, Any]:
        """Serialize the reranker configuration."""
        return {
            "class_name": "OpenAIReranker",
            "model": self.model,
            "api_key": self.api_key
        }


# Global registry instance
registry = ComponentRegistry()

# Register the classes after they are fully defined
registry.register_embedder(OpenAIEmbedder)
registry.register_annotator(NoAdditionalKeys)
registry.register_annotator(OpenAIAnnotator)
registry.register_reranker(StandardReranker)
registry.register_reranker(OpenAIReranker)


class StoreConfig(BaseModel):
    """Configuration for a document store."""
    embedder_id: str
    embedder_config: Dict[str, Any]
    annotator_id: str
    annotator_config: Dict[str, Any]
    reranker_id: str
    reranker_config: Dict[str, Any]
    context: List[ContentBlock] = Field(default_factory=list)
    created_at: dt.datetime = Field(default_factory=dt.datetime.now)
    n_keys_per_doc: int = 5
    p: float = 2.0  # Generalized mean parameter for similarity aggregation
    top_k_retrieval: int = 100  # Number of documents to retrieve before reranking


def _find_files_to_process(
    directory: str,
    existing_sources: set,
    existing_source: str,
    respect_gitignore: bool
) -> tuple[List[tuple[str, str]], int, int]:
    """Find all files that need to be processed."""
    directory_path = Path(directory).resolve()
    text_extensions = {'.txt', '.md', '.py', '.js', '.html', '.css', '.json', '.yaml', '.yml', '.xml', '.csv', '.tsx', '.ts', '.jsx', '.java', '.c', '.cpp', '.h', '.hpp', '.go', '.rb', '.php', '.sh', '.bash'}
    
    files_to_process = []
    ignored_count = 0
    duplicate_count = 0
    
    ignore_files = ('.gitignore', '.ragignore') if respect_gitignore else ('.ragignore',)
    
    # To speed up, walk the directory tree and skip ignored folders in one go.
    def _should_ignore_dir(dir_path, ignore_files):
        # Check if the directory itself is ignored by any ignore file
        # We check the directory path, not a file
        if '.rag/' in str(dir_path):
            return True
        return respect_gitignore and is_ignored(dir_path, ignore_files)

    for root, dirs, files in os.walk(directory_path):
        # Convert root to Path for easier handling
        root_path = Path(root)
        # Remove ignored directories in-place to prevent os.walk from recursing into them
        dirs_to_remove = []
        for d in dirs:
            dir_full_path = root_path / d
            if _should_ignore_dir(dir_full_path, ignore_files):
                dirs_to_remove.append(d)
        for d in dirs_to_remove:
            dirs.remove(d)
            ignored_count += 1  # Count the whole directory as ignored

        for file in files:
            file_path = root_path / file
            if '.rag/' in str(file_path) or '.git/' in str(file_path) or '.venv/' in str(file_path) or 'node_modules/' in str(file_path):
                continue

            if file_path.suffix.lower() not in text_extensions:
                continue

            # Check if file should be ignored
            if respect_gitignore and is_ignored(file_path, ignore_files):
                ignored_count += 1
                continue

            # Use absolute path as source to avoid issues with path resolution later
            source = str(file_path)

            # Check for existing source
            if existing_source == 'ignore' and source in existing_sources:
                duplicate_count += 1
                continue
            files_to_process.append((str(file_path), source))
    return files_to_process, ignored_count, duplicate_count


def _read_file(file_info: tuple[str, str]) -> Optional[Document]:
    """Read a file and create a Document."""
    file_path, source = file_info
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        if content.strip():  # Only process non-empty files
            return Document.from_text(
                text=content,
                title=Path(file_path).name,
                source=source  # source is now already an absolute path
            )
    except Exception as e:
        print(f"Warning: Could not read {file_path}: {e}")
    return None


async def _handle_existing_sources(store, valid_documents: List[Document], existing_source: str):
    """Handle updating existing documents if needed."""
    # Delete documents whose source files no longer exist
    existing_docs = await store.get_all_documents()
    for doc in existing_docs:
        source = doc.document.meta.source
        if not Path(source).exists():
            await store.delete_document(doc.document.id)
    if existing_source == 'update':
        # Delete existing documents with same sources
        existing_docs = await store.get_all_documents()
        sources_to_update = {doc.meta.source for doc in valid_documents}
        for existing_doc in existing_docs:
            if existing_doc.document.meta.source in sources_to_update:
                await store.delete_document(existing_doc.document.id)
                print(f"Updated existing document: {existing_doc.document.meta.source}")
    


async def _ingest_documents_parallel(store, documents: List[Document], max_concurrent: int = 100):
    """Ingest documents in parallel using semaphore for rate limiting."""
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def ingest_with_semaphore(doc: Document):
        async with semaphore:
            await store.ingest(doc)
    
    print("Ingesting documents (generating embeddings)...")
    start_time = time.time()
    
    # Use asyncio.gather with semaphore for better parallelism
    await asyncio.gather(*[ingest_with_semaphore(doc) for doc in documents])
    
    total_time = time.time() - start_time
    print(f"✅ Ingested {len(documents)} documents in {total_time:.1f}s")


class DocumentStore(ABC):
    """Base class for storage systems, e.g. FileSystemStore, MemoryStore, SupabaseStore."""
    def __init__(
        self,
        embedder: Embedder,
        annotator: Annotator,
        reranker: Reranker,
        context: List[ContentBlock] = None,
        n_keys_per_doc: int = 5,
        p: float = 2.0,
        top_k_retrieval: int = 20
    ):
        self.embedder = embedder
        self.annotator = annotator
        self.reranker = reranker
        self.context = context if context else []
        self.n_keys_per_doc = n_keys_per_doc
        self.p = p  # Generalized mean parameter for similarity aggregation
        self.top_k_retrieval = top_k_retrieval
    
    def get_by_source(self, src):
        pass

    async def ingest(self, doc: Document):
        try:
            """Ingest a document by generating keys, embeddings, and summary."""
            annotation_result = await self.annotator(doc, context=self.context)
            
            # Update document metadata with summary if generated
            if annotation_result.summary:
                doc.meta.summary = annotation_result.summary
                doc.meta.updated_at = dt.datetime.now()
            
            # Generate embeddings for the keys
            keys = await asyncio.gather(
                *[self.embedder.get_embedding(content) for content in annotation_result.keys]
            )
            doc_with_embeddings = DocumentWithEmbeddings(
                document=doc,
                keys=keys
            )
            await self.save_document(doc_with_embeddings)
        except Exception as e:
            print(f"Error ingesting document {doc.meta.source}: {e}")
    
    async def ingest_dir(
        self, 
        directory: str, 
        existing_source: str = 'ignore',
        max_workers: int = 10,
        max_concurrent_ingests: int = 5,
        respect_gitignore: bool = False
    ) -> List[Document]:
        """
        Ingest all files in a directory in parallel.
        
        Args:
            directory: Directory path to ingest files from
            existing_source: How to handle documents with existing sources:
                - 'ignore': Skip documents with sources that already exist
                - 'update': Update existing documents with same source
            max_workers: Maximum parallel workers for file processing
            max_concurrent_ingests: Maximum concurrent API calls for ingestion
            respect_gitignore: Whether to respect .gitignore and .ragignore patterns
            
        Returns:
            List of ingested documents
        """
        directory_path = Path(directory).resolve()
        if not directory_path.exists():
            raise ValueError(f"Directory does not exist: {directory}")
        
        # Get existing sources for duplicate checking
        existing_sources = set()
        if existing_source == 'ignore':
            existing_docs = await self.get_all_documents()
            existing_sources = {doc.document.meta.source for doc in existing_docs if doc.document.meta.source}
        # Find files to process
        files_to_process, ignored_count, duplicate_count = _find_files_to_process(
            directory, existing_sources, existing_source, respect_gitignore
        )

        
        if not files_to_process:
            print(f"No new files to ingest (ignored: {ignored_count}, duplicates: {duplicate_count})")
            return []
        
        print(f"Found {len(files_to_process)} files to ingest (ignored: {ignored_count}, duplicates: {duplicate_count})")

        # Read files in parallel using thread pool
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            documents = list(executor.map(_read_file, files_to_process))
        
        # Filter out None documents
        valid_documents = [doc for doc in documents if doc is not None]
        
        if not valid_documents:
            return []
        
        # Handle existing sources for update mode
        await _handle_existing_sources(self, valid_documents, existing_source)
        
        # Ingest documents in parallel with semaphore-based rate limiting
        await _ingest_documents_parallel(self, valid_documents, max_concurrent_ingests)
        
        return valid_documents
    
    @abstractmethod
    async def get_by_source(self, source: str) -> Optional[Document]:
        """Retrieve a document by its source path."""
        raise NotImplementedError("get_by_source() must be implemented by subclass")

    @abstractmethod
    async def query(self, query_content: List[ContentBlock] | None, history: List[ChatMessage] | None = None, top_k: int = 5) -> List[Document]:
        """Query the store and return top_k most relevant documents after reranking."""
        raise NotImplementedError("query() must be implemented by subclass")

    @abstractmethod
    async def save_document(self, document: DocumentWithEmbeddings):
        """Create or update a document in the store."""
        raise NotImplementedError("save_document() must be implemented by subclass")
    
    @abstractmethod
    async def delete_document(self, document_id: str):
        """Delete a document from the store."""
        raise NotImplementedError("delete_document() must be implemented by subclass")
    
    def to_config(self) -> StoreConfig:
        """Serialize the store configuration."""
        return StoreConfig(
            embedder_id=self.embedder.id,
            embedder_config=self.embedder.to_config(),
            annotator_id=self.annotator.id,
            annotator_config=self.annotator.to_config(),
            reranker_id=self.reranker.id,
            reranker_config=self.reranker.to_config(),
            context=self.context,
            n_keys_per_doc=self.n_keys_per_doc,
            p=self.p,
            top_k_retrieval=self.top_k_retrieval
        )


def generalized_mean(scores, p, axis):
    """
    p → -∞: min
    p = -1: harmonic mean
    p = 0: geometric mean
    p = 1: arithmetic mean
    p → +∞: max
    """
    if p == 0:
        return np.exp(np.mean(np.log(np.maximum(scores, 1e-8)), axis=axis))
    else:
        return np.power(np.mean(np.power(np.maximum(scores, 0), p), axis=axis), 1/p)


class MemoryRagStore(DocumentStore):
    """In-memory RAG store for testing purposes."""
    
    def __init__(self, embedder: Embedder, annotator: Annotator, reranker: Reranker, context: List[ContentBlock] = None, n_keys_per_doc: int = 5, p: float = 2.0, top_k_retrieval: int = 20):
        super().__init__(embedder, annotator, reranker, context=context, n_keys_per_doc=n_keys_per_doc, p=p, top_k_retrieval=top_k_retrieval)
        self.documents = {}
    
    def get_by_source(self, src):
        for doc in self.documents.values():
            if doc.document.meta.source == src:
                return doc.document

    async def save_document(self, document: DocumentWithEmbeddings):
        """Save or update a document in the in-memory store."""
        self.documents[document.document.id] = document
    
    async def delete_document(self, document_id: str):
        """Delete a document from the in-memory store."""
        if document_id in self.documents:
            del self.documents[document_id]
    
    async def get_all_documents(self) -> List[DocumentWithEmbeddings]:
        """Retrieve all documents from the in-memory store."""
        return list(self.documents.values())
    
    async def get_query_content(self, history: List[ChatMessage]):
        """Ask an LLM to generate a query that asks for relevant documents for the given history."""
        messages = [
            ChatMessage(
                role='system',
                content=[TextBlock(text='You will be presented with a conversation history between a user and an assistant. You are helping the assistant of this rollout: your task is to generate a query that asks for relevant documents for the given history. The query should be a question or list of questions that can be understood without having access to the conversation history. The query may also be a short paragraph of the kind of information that would be most useful to retrieve. The query will then be used to retrieve documents from a RAG store, and the most relevant documents will be added to the conversation history.')]
            ),
            ChatMessage(
                role='user',
                content=[
                    TextBlock(text="<rollout>")
                ] + messages_to_content_blocks(history) + [
                    TextBlock(text="</rollout>\nNow, please respond with the query for the RAG system that is most likely to retrieve helpful documents.")
                ]
            )
        ]
        response = await get_response(
            messages=messages,
            model='gpt-5',
            max_tokens=250,
            cache_seed=0
        )
        return [TextBlock(text=response.content[0].text)]

    
    async def query(self, query_content: List[ContentBlock] | None = None, history: List[ChatMessage] | None = None, top_k: int = 5) -> List[Document]:
        """Query the store using cosine similarity between query and document keys, then rerank."""
        if not self.documents:
            return []
        top_k_retrieval = max(top_k, self.top_k_retrieval)
        if top_k_retrieval < len(self.documents):
            if query_content is None:
                query_content = await self.get_query_content(history)
            # Get query embedding
            query_embedding = await self.embedder.get_embedding(query_content)
            embedded_docs = await self.get_all_documents()
            
            if not embedded_docs:
                return []
            
            # Compute similarities
            Q = np.array(query_embedding.embedding)
            
            doc_scores = []
            for doc_with_embeddings in embedded_docs:
                if not doc_with_embeddings.keys:
                    doc_scores.append(0.0)
                    continue
                    
                # Get all key embeddings for this document
                key_embeddings = np.array([key.embedding for key in doc_with_embeddings.keys[:self.n_keys_per_doc]])
                
                # Compute cosine similarities between query and each key
                # Normalize vectors for cosine similarity
                Q_norm = Q / (np.linalg.norm(Q) + 1e-8)
                K_norm = key_embeddings / (np.linalg.norm(key_embeddings, axis=1, keepdims=True) + 1e-8)
                
                # Compute dot products (cosine similarities)
                similarities = np.dot(K_norm, Q_norm)
                
                # Aggregate similarities using generalized mean (p=2 for quadratic mean)
                doc_score = generalized_mean(similarities, p=self.p, axis=0)
                doc_scores.append(doc_score)
            
            # Get top_k_retrieval documents for reranking
            doc_scores = np.array(doc_scores)
            top_indices = np.argsort(doc_scores)[-top_k_retrieval:][::-1]  # Get top_k_retrieval indices in descending order
            
            # Filter out documents with zero scores
            candidate_docs = [embedded_docs[i].document for i in top_indices if doc_scores[i] > 0]
        else:
            # If top_k_retrieval is greater than number of documents, use all documents
            candidate_docs = [doc_with_embeddings.document for doc_with_embeddings in self.documents.values()]
        
        if not candidate_docs:
            return []
        
        # Apply reranking
        reranked_docs = await self.reranker.rerank(query_content, candidate_docs, top_k, history)
        
        return reranked_docs


class FileSystemStore(MemoryRagStore):
    """File system-based RAG store with persistence."""
    
    def __init__(self, embedder: Embedder, annotator: Annotator, reranker: Reranker, root_dir: str, context: List[ContentBlock] = None, n_keys_per_doc: int = 5, p: float = 2.0, top_k_retrieval: int = 20):
        super().__init__(embedder, annotator, reranker, context, n_keys_per_doc, p, top_k_retrieval)
        self.root_dir = root_dir
        os.makedirs(self.root_dir, exist_ok=True)
        
        # Save store configuration
        self._save_config()
    
    def _save_config(self):
        """Save the store configuration to disk."""
        config = self.to_config()
        config_path = os.path.join(self.root_dir, "store_config.json")
        with open(config_path, 'w', encoding='utf-8') as f:
            f.write(config.model_dump_json(indent=2))
    
    async def save_document(self, document: DocumentWithEmbeddings):
        """Save a document to the file system."""
        await super().save_document(document)
        # Save to disk
        file_path = os.path.join(self.root_dir, f"{document.document.id}.json")
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(document.model_dump_json(indent=2))
    
    async def delete_document(self, document_id: str):
        """Delete a document from the store."""
        if document_id in self.documents:
            del self.documents[document_id]
            
            # Remove file from disk
            file_path = os.path.join(self.root_dir, f"{document_id}.json")
            if os.path.exists(file_path):
                os.remove(file_path)
    
    @staticmethod
    def create(embedder: Embedder, annotator: Annotator, reranker: Reranker, root_dir: str, context: List[ContentBlock] = None, n_keys_per_doc: int = 5, p: float = 2.0, top_k_retrieval: int = 20) -> 'FileSystemStore':
        """Create a new FileSystemStore and save its configuration."""
        return FileSystemStore(embedder, annotator, reranker, root_dir, context, n_keys_per_doc, p, top_k_retrieval)
    
    @staticmethod
    def load(root_dir: str) -> 'FileSystemStore':
        """Load a FileSystemStore from disk."""
        if not os.path.exists(root_dir):
            raise ValueError(f"Directory does not exist: {root_dir}")
        
        config_path = os.path.join(root_dir, "store_config.json")
        if not os.path.exists(config_path):
            raise ValueError(f"Store configuration not found: {config_path}")
        
        # Load configuration
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = json.load(f)
        
        # Handle backward compatibility for key_generator -> annotator transition
        if 'key_generator_id' in config_data and 'annotator_id' not in config_data:
            config_data['annotator_id'] = config_data.pop('key_generator_id')
            config_data['annotator_config'] = config_data.pop('key_generator_config')
        
        # Handle backward compatibility for missing reranker (use StandardReranker as default)
        if 'reranker_id' not in config_data:
            config_data['reranker_id'] = 'standard-reranker'
            config_data['reranker_config'] = {'class_name': 'StandardReranker'}
        
        # Handle backward compatibility for missing top_k_retrieval
        if 'top_k_retrieval' not in config_data:
            config_data['top_k_retrieval'] = 20
        
        config = StoreConfig(**config_data)
        
        # Recreate embedder, annotator, and reranker
        embedder = registry.create_embedder(config.embedder_id, config.embedder_config)
        annotator = registry.create_annotator(config.annotator_id, config.annotator_config)
        reranker = registry.create_reranker(config.reranker_id, config.reranker_config)
        
        # Create store instance
        store = FileSystemStore(embedder, annotator, reranker, root_dir, config.context, config.n_keys_per_doc, config.p, config.top_k_retrieval)
        
        # Load all documents
        for filename in os.listdir(root_dir):
            if filename.endswith('.json') and filename != 'store_config.json':
                file_path = os.path.join(root_dir, filename)
                with open(file_path, 'r', encoding='utf-8') as f:
                    doc_data = json.load(f)
                
                # Parse the document with embeddings
                try:
                    doc_with_embeddings = DocumentWithEmbeddings(**doc_data)
                    store.documents[doc_with_embeddings.document.id] = doc_with_embeddings
                except Exception as e:
                    print(f"Warning: Could not load document from {file_path}: {e}", file=sys.stderr)
        
        return store
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the store."""
        total_docs = len(self.documents)
        total_keys = sum(len(doc.keys) for doc in self.documents.values())
        
        return {
            "total_documents": total_docs,
            "total_keys": total_keys,
            "embedder_id": self.embedder.id,
            "annotator_id": self.annotator.id,
            "reranker_id": self.reranker.id,
            "top_k_retrieval": self.top_k_retrieval,
            "root_directory": self.root_dir
        }


# Export the registry for external use
__all__ = [
    'Document', 'DocumentMeta', 'Embedding', 'DocumentWithEmbeddings',
    'Embedder', 'OpenAIEmbedder', 'Annotator', 'AnnotationResult', 'NoAdditionalKeys', 'OpenAIAnnotator',
    'Reranker', 'StandardReranker', 'OpenAIReranker',
    'DocumentStore', 'MemoryRagStore', 'FileSystemStore',
    'StoreConfig', 'ComponentRegistry', 'registry'
]