"""The Ingestion module: turns uploaded Documents into embedded Chunks in the Vector Store.

A Document is hashed, split into Chunks (~1000 characters, ~200 overlap) by
LangChain's RecursiveCharacterTextSplitter, embedded by the Vector Store's
injected Embeddings on write, and stored with source-filename metadata.
Every ingest reports an outcome: ``ingested`` (new Document), ``skipped``
(identical content already ingested), or ``replaced`` (changed content under
a known filename — that source's old Chunks are deleted before the new ones
land).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Literal

from langchain_text_splitters import RecursiveCharacterTextSplitter

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

_SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=_CHUNK_SIZE,
    chunk_overlap=_CHUNK_OVERLAP,
)


def _chunk_text(text: str) -> list[str]:
    """Split ``text`` into Chunks of about 1000 characters with about 200 characters of overlap.

    The splitter prefers paragraph, line, and word boundaries so words and
    sentences stay whole; a Chunk never exceeds ``_CHUNK_SIZE`` characters,
    and consecutive Chunks share about ``_CHUNK_OVERLAP`` characters so
    context survives the cut.
    """
    return _SPLITTER.split_text(text)


class Ingestion:
    """Ingests Documents into the Vector Store, which embeds Chunks through its injected Embeddings."""

    def __init__(self, store: VectorStore) -> None:
        self._store = store

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
        if known_hash is not None:
            # Changed content under a known filename: the old Chunks must go
            # before the new ones land, so Retrieval never sees stale content.
            self._store.delete_chunks_for_source(document.filename)
        self._store.add_chunks(document.filename, content_hash, texts)
        self._store.register_document(document.filename, content_hash)
        return IngestResult(
            status="replaced" if known_hash is not None else "ingested",
            chunk_count=len(texts),
        )

    def ingested_documents(self) -> list[str]:
        """List the source names of every ingested Document."""
        return self._store.document_sources()
