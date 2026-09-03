"""The Vector Store: the persistent local store holding embedded Chunks of every ingested Document.

Backed by a local, persistent Chroma instance: the embedded Chunks carry
source-filename metadata, and a separate registry records every ingested
Document by source name and content hash, so Ingestion can detect skips and
replacements. (Private constants mirror Chroma's own collection vocabulary.)
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chromadb

_CHUNK_COLLECTION = "chunks"
_DOCUMENT_COLLECTION = "documents"


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

    def __init__(self, path: Path) -> None:
        client = chromadb.PersistentClient(path=str(path))
        self._chunks = client.get_or_create_collection(
            _CHUNK_COLLECTION, metadata={"hnsw:space": "cosine"}
        )
        self._documents = client.get_or_create_collection(_DOCUMENT_COLLECTION)

    def register_document(self, source: str, content_hash: str) -> None:
        """Record ``source`` as an ingested Document whose content hashed to ``content_hash``."""
        self._documents.upsert(ids=[source], documents=[content_hash])

    def registered_hash(self, source: str) -> str | None:
        """Return the recorded content hash for ``source``, or ``None`` if it was never ingested."""
        found = self._documents.get(ids=[source], include=["documents"])
        documents = found.get("documents") or []
        if not documents:
            return None
        return str(documents[0])

    def document_sources(self) -> list[str]:
        """List the source names of every ingested Document."""
        return sorted(self._documents.get()["ids"])

    def add_chunks(
        self,
        source: str,
        content_hash: str,
        texts: Sequence[str],
        embeddings: Sequence[Sequence[float]],
    ) -> None:
        """Write ``texts`` with ``embeddings`` as the Chunks of ``source``."""
        if not texts:
            return
        self._chunks.add(
            ids=[f"{source}:{index}" for index in range(len(texts))],
            documents=list(texts),
            embeddings=[list(embedding) for embedding in embeddings],
            metadatas=[
                {"source": source, "chunk_index": index, "content_hash": content_hash}
                for index in range(len(texts))
            ],
        )

    def delete_chunks_for_source(self, source: str) -> None:
        """Delete every Chunk belonging to ``source``."""
        self._chunks.delete(where={"source": source})

    def count_chunks(self, source: str | None = None) -> int:
        """Count stored Chunks, optionally only those belonging to ``source``."""
        if source is None:
            return self._chunks.count()
        found = self._chunks.get(where={"source": source}, include=[])
        return len(found["ids"])

    def chunks_for_source(self, source: str) -> list[Chunk]:
        """Return every stored Chunk belonging to ``source``, in document order."""
        found = self._chunks.get(
            where={"source": source},
            include=["documents", "metadatas"],
        )
        return _chunks_from_results(found)

    def query_chunks(self, query_embedding: Sequence[float], k: int = 5) -> list[Chunk]:
        """Return the ``k`` Chunks most similar to ``query_embedding``, most similar first."""
        if self._chunks.count() == 0:
            return []
        found = self._chunks.query(
            query_embeddings=[list(query_embedding)],
            n_results=k,
            include=["documents", "metadatas"],
        )
        return _chunks_from_results(
            {"documents": found["documents"][0], "metadatas": found["metadatas"][0]}
        )
