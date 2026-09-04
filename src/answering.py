"""The Answering module: answers a Conversation's latest Query with an Answer grounded in Retrieved Chunks.

Answering Retrieves the top 4 Chunks most similar to the Query, grounds a
system prompt with them — each excerpt labeled with its source Document so the
model can point at passages — and completes the Conversation through the
injected ChatModel, supplied only the 20 most recent messages as history,
however long the Conversation grows. The Answer carries the Retrieved Chunks
that backed it, so callers can show where every Answer came from. The real
ChatModel — OpenAI's Responses API through LangChain — is constructed at the
composition root, from configuration.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from src.conversations import Conversation, Message
from src.logging_setup import get_logger
from src.prompts import grounded_system_prompt
from src.vector_store import Chunk, VectorStore

logger = get_logger("answering")

_RETRIEVAL_K = 4
_MAX_HISTORY = 20


@dataclass(frozen=True)
class Answer:
    """The Answer to a Query: the model's reply text and the Retrieved Chunks backing it."""

    text: str
    sources: tuple[Chunk, ...]


class ChatModel(Protocol):
    """Answers a fully-formed prompt: a grounded system instruction, the prior turns, and the Query."""

    def complete(self, system: str, history: Sequence[Message], query: str) -> str:
        """Return the Answer to ``query``, given ``system`` and the Conversation's prior turns."""
        ...


class OpenAIChatModel:
    """The real ChatModel: OpenAI's Responses API through LangChain."""

    def __init__(self, api_key: str, model: str, reasoning_effort: str) -> None:
        from langchain_openai import ChatOpenAI

        self._client = ChatOpenAI(api_key=api_key, model=model, reasoning_effort=reasoning_effort)

    def complete(self, system: str, history: Sequence[Message], query: str) -> str:
        """Complete the prompt through LangChain and return the Answer text."""
        messages: list[SystemMessage | HumanMessage | AIMessage] = [SystemMessage(content=system)]
        for message in history:
            if message.role == "user":
                messages.append(HumanMessage(content=message.content))
            else:
                messages.append(AIMessage(content=message.content))
        messages.append(HumanMessage(content=query))
        response = self._client.invoke(messages)
        return str(response.content)


class Answering:
    """Answers a Conversation's latest Query, grounded in Retrieved Chunks."""

    def __init__(self, store: VectorStore, chat_model: ChatModel) -> None:
        self._store = store
        self._chat_model = chat_model

    def answer(self, conversation: Conversation) -> Answer:
        """Answer the Conversation's most recent message — its Query — and return the Answer.

        The Answer carries the Retrieved Chunks that grounded it.

        Raises:
            ValueError: If the Conversation has no messages to answer.
        """
        if not conversation.messages:
            raise ValueError(
                "Cannot answer a Conversation with no messages: the latest message is the Query."
            )
        *all_history, query = conversation.messages
        history = all_history[-_MAX_HISTORY:]
        chunks = self._store.query_chunks(query.content, k=_RETRIEVAL_K)
        system = grounded_system_prompt(chunks)
        text = self._chat_model.complete(system, history, query.content)
        logger.info(
            "Answered a Query with %d Retrieved Chunks and %d history messages",
            len(chunks),
            len(history),
        )
        return Answer(text=text, sources=tuple(chunks))
