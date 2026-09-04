"""The Vector Store: the persistent local store holding embedded Chunks of every ingested Document.

Backed by a local, persistent Chroma instance through LangChain's Chroma
vector store: Chunks are embedded on write and found by similarity through
the injected langchain Embeddings, and a separate registry collection records
every ingested Document by source name and content hash, so Ingestion can
detect skips and replacements. (Private constants mirror Chroma's own
collection vocabulary.)
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings

from src.logging_setup import get_logger

logger = get_logger("vector_store")

_CHUNK_COLLECTION = "chunks"
_DOCUMENT_COLLECTION = "documents"


class _RegistryEmbeddings(Embeddings):
    """Constant vectors for the Document registry: it is read by id, never by similarity."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return [0.0]


@dataclass(frozen=True)
class Chunk:
    """A split piece of a Document that is embedded and retrieved as one unit."""

    source: str
    index: int
    text: str


def _chunks_from_results(found: Mapping[str, Any]) -> list[Chunk]:
    """Build ordered Chunks from a Chroma result holding documents and metadatas."""
    chunks = [
        Chunk(
            source=str(metadata["source"]),
            index=int(metadata["chunk_index"]),
            text=document,
        )
        for document, metadata in zip(found["documents"], found["metadatas"])
    ]
    return sorted(chunks, key=lambda chunk: chunk.index)


class VectorStore:
    """Stores embedded Chunks and the registry of ingested Documents, rooted at a directory."""

    def __init__(self, path: Path, embeddings: Embeddings) -> None:
        """Open the Chunk and registry collections at ``path``, embedding through ``embeddings``."""
        self._chunks = Chroma(
            collection_name=_CHUNK_COLLECTION,
            embedding_function=embeddings,
            persist_directory=str(path),
            collection_configuration={"hnsw": {"space": "cosine"}},
        )
        self._documents = Chroma(
            collection_name=_DOCUMENT_COLLECTION,
            embedding_function=_RegistryEmbeddings(),
            persist_directory=str(path),
        )

    def register_document(self, source: str, content_hash: str) -> None:
        """Record ``source`` as an ingested Document whose content hashed to ``content_hash``."""
        self._documents.add_texts(texts=[content_hash], ids=[source])

    def registered_hash(self, source: str) -> str | None:
        """Return the recorded content hash for ``source``, or ``None`` if it was never ingested."""
        found = self._documents.get(ids=[source], include=["documents"])
        documents = found.get("documents") or []
        if not documents:
            return None
        return str(documents[0])

    def unregister_document(self, source: str) -> None:
        """Remove ``source`` from the registry of ingested Documents."""
        self._documents.delete(ids=[source])
        logger.debug("Unregistered Document %s", source)

    def document_sources(self) -> list[str]:
        """List the source names of every ingested Document."""
        return sorted(self._documents.get(include=[])["ids"])

    def add_chunks(self, source: str, content_hash: str, texts: Sequence[str]) -> None:
        """Write ``texts`` as the Chunks of ``source``, embedded by the injected Embeddings."""
        if not texts:
            return
        self._chunks.add_texts(
            texts=list(texts),
            ids=[f"{source}:{index}" for index in range(len(texts))],
            metadatas=[
                {"source": source, "chunk_index": index, "content_hash": content_hash}
                for index in range(len(texts))
            ],
        )
        logger.debug("Added %d Chunks for %s", len(texts), source)

    def delete_chunks_for_source(self, source: str) -> None:
        """Delete every Chunk belonging to ``source``."""
        self._chunks.delete(where={"source": source})
        logger.debug("Deleted Chunks for %s", source)

    def count_chunks(self, source: str | None = None) -> int:
        """Count stored Chunks, optionally only those belonging to ``source``."""
        if source is None:
            return len(self._chunks.get(include=[])["ids"])
        return len(self._chunks.get(where={"source": source}, include=[])["ids"])

    def chunks_for_source(self, source: str) -> list[Chunk]:
        """Return every stored Chunk belonging to ``source``, in document order."""
        found = self._chunks.get(
            where={"source": source},
            include=["documents", "metadatas"],
        )
        return _chunks_from_results(found)

    def query_chunks(self, query: str, k: int = 5) -> list[Chunk]:
        """Return the ``k`` Chunks most similar to ``query``, in document order."""
        if not self._chunks.get(limit=1, include=[])["ids"]:
            return []
        found = self._chunks.similarity_search(query=query, k=k)
        chunks = [
            Chunk(
                source=str(document.metadata["source"]),
                index=int(document.metadata["chunk_index"]),
                text=document.page_content,
            )
            for document in found
        ]
        ordered = sorted(chunks, key=lambda chunk: chunk.index)
        logger.debug("Retrieval returned %d Chunks (k=%d)", len(ordered), k)
        return ordered
