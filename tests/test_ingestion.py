"""Tests for the Ingestion module: Documents become embedded Chunks in the Vector Store.

Per the ticket, all dedup outcomes (ingested / skipped / replaced) are driven
with a fake Embedder over a real Chroma Vector Store in a temp directory.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest

from src.ingestion import Document, Ingestion
from src.vector_store import VectorStore


class FakeEmbedder:
    """Deterministic Embedder: keyword counts become vector coordinates."""

    def __init__(self) -> None:
        self.calls = 0

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        self.calls += 1

        def vector(text: str) -> list[float]:
            return [1.0 + text.count("alpha"), 1.0 + text.count("beta")]

        return [vector(text) for text in texts]


@pytest.fixture()
def store(tmp_path: Path) -> VectorStore:
    return VectorStore(tmp_path / "chroma")


@pytest.fixture()
def embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture()
def ingestion(store: VectorStore, embedder: FakeEmbedder) -> Ingestion:
    return Ingestion(store=store, embedder=embedder)


def test_ingesting_a_new_document_reports_ingested_and_stores_its_chunks(
    ingestion: Ingestion, store: VectorStore
) -> None:
    result = ingestion.ingest(Document(filename="notes.md", content="alpha " * 40))

    assert result.status == "ingested"
    chunks = store.chunks_for_source("notes.md")
    assert len(chunks) > 0
    assert all(chunk.source == "notes.md" for chunk in chunks)
    assert all("alpha" in chunk.text for chunk in chunks)


def test_reingesting_identical_content_reports_skipped_and_adds_no_duplicate_chunks(
    ingestion: Ingestion, store: VectorStore, embedder: FakeEmbedder
) -> None:
    first = ingestion.ingest(Document(filename="notes.md", content="alpha " * 40))
    chunks_before = store.count_chunks("notes.md")

    second = ingestion.ingest(Document(filename="notes.md", content="alpha " * 40))

    assert second.status == "skipped"
    assert store.count_chunks("notes.md") == chunks_before == first.chunk_count
    assert embedder.calls == 1  # a skip is detected from the hash alone, before any embedding


def test_changed_content_under_the_same_filename_reports_replaced_and_leaves_no_orphaned_chunks(
    ingestion: Ingestion, store: VectorStore
) -> None:
    ingestion.ingest(Document(filename="notes.md", content="alpha " * 40))
    old_texts = {chunk.text for chunk in store.chunks_for_source("notes.md")}

    result = ingestion.ingest(Document(filename="notes.md", content="beta " * 40))

    assert result.status == "replaced"
    chunks = store.chunks_for_source("notes.md")
    assert len(chunks) > 0
    assert all(chunk.text not in old_texts for chunk in chunks)
    assert all("beta" in chunk.text for chunk in chunks)
    # The replace removed exactly the old Chunks: nothing for other sources was touched.
    assert store.count_chunks() == result.chunk_count


def _shared_overlap(earlier: str, later: str) -> int:
    """Length of the common suffix/prefix by which ``later`` overlaps ``earlier``."""
    for size in range(min(len(earlier), len(later)), 0, -1):
        if earlier[-size:] == later[:size]:
            return size
    return 0


def test_documents_split_into_chunks_of_about_1000_characters_with_about_200_overlap(
    ingestion: Ingestion, store: VectorStore
) -> None:
    content = "word " * 300  # 1500 characters: more than one Chunk, fewer than three

    ingestion.ingest(Document(filename="long.txt", content=content))

    chunks = store.chunks_for_source("long.txt")
    assert len(chunks) >= 2
    for chunk in chunks:
        assert len(chunk.text) <= 1000
    for earlier, later in zip(chunks, chunks[1:]):
        assert _shared_overlap(earlier.text, later.text) >= 150


def test_similarity_query_returns_relevant_chunks_for_a_query_embedding(
    ingestion: Ingestion, store: VectorStore, embedder: FakeEmbedder
) -> None:
    ingestion.ingest(Document(filename="cats.md", content="alpha " * 400))
    ingestion.ingest(Document(filename="dogs.md", content="beta " * 400))

    query_embedding = embedder.embed(["alpha alpha"])[0]

    results = store.query_chunks(query_embedding, k=3)

    assert len(results) == 3
    assert {chunk.source for chunk in results} == {"cats.md"}


def test_querying_an_empty_vector_store_returns_no_chunks(
    store: VectorStore, embedder: FakeEmbedder
) -> None:
    query_embedding = embedder.embed(["alpha"])[0]

    assert store.query_chunks(query_embedding, k=5) == []
