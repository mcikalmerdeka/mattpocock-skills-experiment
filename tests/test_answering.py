"""Tests for the Answering module: a Conversation's latest Query is answered with an Answer grounded in Retrieved Chunks.

A fake ChatModel captures every prompt; deterministic fake Embeddings drive
Retrieval over a real Chroma-backed Vector Store in a temp directory.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import pytest
from langchain_core.embeddings import Embeddings

from src.answering import Answering
from src.conversations import Conversation, Message
from src.ingestion import Document, Ingestion
from src.vector_store import VectorStore


class FakeEmbeddings(Embeddings):
    """Deterministic Embeddings: keyword counts become vector coordinates."""

    @staticmethod
    def _vector(text: str) -> list[float]:
        return [1.0 + text.count("alpha"), 1.0 + text.count("beta")]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)


class FakeChatModel:
    """Records every prompt and returns a canned Answer."""

    def __init__(self, answer: str) -> None:
        self._answer = answer
        self.prompts: list[tuple[str, tuple[Message, ...], str]] = []

    def complete(self, system: str, history: Sequence[Message], query: str) -> str:
        self.prompts.append((system, tuple(history), query))
        return self._answer


def _conversation(*messages: Message) -> Conversation:
    """Build a Conversation holding exactly ``messages``."""
    now = datetime.now(UTC)
    return Conversation(
        id="test-conversation",
        name="Test Conversation",
        created_at=now,
        updated_at=now,
        messages=messages,
    )


@pytest.fixture()
def store(tmp_path: Path) -> VectorStore:
    return VectorStore(tmp_path / "chroma", FakeEmbeddings())


@pytest.fixture()
def chat_model() -> FakeChatModel:
    return FakeChatModel(answer="grounded answer")


@pytest.fixture()
def answering(store: VectorStore, chat_model: FakeChatModel) -> Answering:
    return Answering(store=store, chat_model=chat_model)


def test_the_answer_is_the_chat_models_reply(
    answering: Answering, chat_model: FakeChatModel
) -> None:
    conversation = _conversation(Message(role="user", content="What is alpha?"))

    answer = answering.answer(conversation)

    assert answer == "grounded answer"
    assert len(chat_model.prompts) == 1


def test_the_system_prompt_is_grounded_in_the_retrieved_chunks(
    answering: Answering, chat_model: FakeChatModel, store: VectorStore
) -> None:
    Ingestion(store=store).ingest(
        Document(filename="notes.md", content="alpha is the first letter")
    )
    conversation = _conversation(Message(role="user", content="tell me about alpha"))

    answering.answer(conversation)

    system, history, query = chat_model.prompts[0]
    assert "alpha is the first letter" in system
    assert history == ()
    assert query == "tell me about alpha"


def test_prior_turns_are_passed_as_history_and_the_latest_message_is_the_query(
    answering: Answering, chat_model: FakeChatModel
) -> None:
    conversation = _conversation(
        Message(role="user", content="first question"),
        Message(role="assistant", content="first answer"),
        Message(role="user", content="second question"),
    )

    answering.answer(conversation)

    _, history, query = chat_model.prompts[0]
    assert [(message.role, message.content) for message in history] == [
        ("user", "first question"),
        ("assistant", "first answer"),
    ]
    assert query == "second question"


def test_answering_without_ingested_documents_still_answers(
    answering: Answering, chat_model: FakeChatModel
) -> None:
    conversation = _conversation(Message(role="user", content="anything"))

    answer = answering.answer(conversation)

    assert answer == "grounded answer"
    system, _, _ = chat_model.prompts[0]
    # The Retrieved context is empty, and the prompt still says so honestly.
    assert "Excerpts:\n" in system


def test_only_the_twenty_most_recent_messages_are_passed_as_history(
    answering: Answering, chat_model: FakeChatModel
) -> None:
    # 25 messages: the latest (index 24) is the Query, so the history is
    # indices 0-23 — and the cap keeps only its last 20, indices 4-23.
    messages = tuple(
        Message(role="user" if index % 2 == 0 else "assistant", content=f"message {index}")
        for index in range(25)
    )
    conversation = _conversation(*messages)

    answering.answer(conversation)

    _, history, _ = chat_model.prompts[0]
    assert [message.content for message in history] == [
        f"message {index}" for index in range(4, 24)
    ]


def test_retrieval_supplies_the_top_four_chunks(
    answering: Answering, chat_model: FakeChatModel, store: VectorStore
) -> None:
    # Six paragraphs, each its own Chunk: paragraph ``index`` mentions
    # ``alpha`` ``index + 1`` times, so similarity to the Query "alpha"
    # decreases with index and the top-4 Chunks are paragraphs 0-3.
    paragraphs = [
        "alpha " * (index + 1) + "x" * 600 + f" marker-{index}" for index in range(6)
    ]
    Ingestion(store=store).ingest(
        Document(filename="long.md", content="\n\n".join(paragraphs))
    )
    conversation = _conversation(Message(role="user", content="alpha"))

    answering.answer(conversation)

    system, _, _ = chat_model.prompts[0]
    markers_in_context = [
        f"marker-{index}" for index in range(6) if f"marker-{index}" in system
    ]
    assert markers_in_context == ["marker-0", "marker-1", "marker-2", "marker-3"]


def test_answering_a_conversation_without_messages_raises_a_readable_error(
    answering: Answering,
) -> None:
    with pytest.raises(ValueError, match="no messages"):
        answering.answer(_conversation())
