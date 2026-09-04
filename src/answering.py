"""The Answering module: answers a Conversation's latest Query with an Answer grounded in Retrieved Chunks.

Answering Retrieves the top 4 Chunks most similar to the Query, grounds a
system prompt with them, and completes the Conversation through the injected
ChatModel — supplied only the 20 most recent messages as history, however
long the Conversation grows. The real ChatModel — OpenAI's Responses API
through LangChain — is constructed at the composition root, from
configuration.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from src.conversations import Conversation, Message
from src.vector_store import VectorStore

_RETRIEVAL_K = 4
_MAX_HISTORY = 20

_SYSTEM_TEMPLATE = """You are the assistant of a document question-answering app. Answer the user's Query using only the Excerpts below — passages Retrieved from the Documents the user has ingested. If the Excerpts do not contain the Answer, say so plainly; do not invent content. Point at the relevant passage when that helps.

Excerpts:
{context}"""


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

    def answer(self, conversation: Conversation) -> str:
        """Answer the Conversation's most recent message — its Query — and return the Answer text.

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
        context = "\n\n".join(chunk.text for chunk in chunks)
        system = _SYSTEM_TEMPLATE.format(context=context)
        return self._chat_model.complete(system, history, query.content)
