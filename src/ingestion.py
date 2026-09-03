"""The Ingestion module: turns uploaded Documents into embedded Chunks in the Vector Store.

A Document is hashed, split into Chunks (~1000 characters, ~200 overlap),
embedded through the injected Embedder, and written to the Vector Store with
source-filename metadata. Every ingest reports an outcome: ``ingested`` (new
Document), ``skipped`` (identical content already ingested), or ``replaced``
(changed content under a known filename — that source's old Chunks are
deleted before the new ones land).
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from src.embedders import Embedder
from src.vector_store import VectorStore

Status = Literal["ingested", "skipped", "replaced"]


@dataclass(frozen=True)
class Document:
    """A user-supplied TXT or Markdown file uploaded through the app to be made answerable."""

    filename: str
    content: str


@dataclass(frozen=True)
class IngestResult:
    """The outcome of one ingest: what happened and how many Chunks the Document now has."""

    status: Status
    chunk_count: int


_CHUNK_SIZE = 1000
_CHUNK_OVERLAP = 200


def _chunk_text(text: str) -> list[str]:
    """Split ``text`` into Chunks of about 1000 characters with about 200 characters of overlap.

    Cuts prefer whitespace boundaries so words stay whole; a Chunk never
    exceeds ``_CHUNK_SIZE`` characters, and consecutive Chunks share about
    ``_CHUNK_OVERLAP`` characters so context survives the cut.
    """
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + _CHUNK_SIZE, len(text))
        if end < len(text):
            cut = text.rfind(" ", start + _CHUNK_OVERLAP, end)
            if cut != -1:
                end = cut + 1  # keep the whitespace with the Chunk it closes
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start = max(end - _CHUNK_OVERLAP, start + 1)
    return [chunk for chunk in chunks if chunk.strip()]


class Ingestion:
    """Ingests Documents into the Vector Store, embedding Chunks through the injected Embedder."""

    def __init__(self, store: VectorStore, embedder: Embedder) -> None:
        self._store = store
        self._embedder = embedder

    def ingest(self, document: Document) -> IngestResult:
        """Make ``document`` answerable and report the dedup outcome."""
        content_hash = hashlib.sha256(document.content.encode("utf-8")).hexdigest()
        known_hash = self._store.registered_hash(document.filename)
        if known_hash == content_hash:
            return IngestResult(
                status="skipped",
                chunk_count=self._store.count_chunks(document.filename),
            )
        texts = _chunk_text(document.content)
        embeddings = self._embedder.embed(texts)
        if known_hash is not None:
            # Changed content under a known filename: the old Chunks must go
            # before the new ones land, so Retrieval never sees stale content.
            self._store.delete_chunks_for_source(document.filename)
        self._store.add_chunks(document.filename, content_hash, texts, embeddings)
        self._store.register_document(document.filename, content_hash)
        return IngestResult(
            status="replaced" if known_hash is not None else "ingested",
            chunk_count=len(texts),
        )

    def ingested_documents(self) -> list[str]:
        """List the source names of every ingested Document."""
        return self._store.document_sources()
