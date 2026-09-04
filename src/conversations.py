"""The conversations module: creates, lists, loads, and appends to Conversations.

Each Conversation persists as its own JSON file — ``{id}.json`` inside the
store's root directory — holding the conversation's id, name, created/updated
timestamps, and ordered messages. An assistant Message may carry ``sources``:
the Retrieved Chunks that backed its Answer, persisted alongside the message.
A new Conversation is named from its first message; until then it carries a
placeholder name.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.logging_setup import get_logger

logger = get_logger("conversations")

_PLACEHOLDER_NAME = "New Conversation"
_MAX_NAME_LENGTH = 60
_ELLIPSIS = "…"


class ConversationNotFoundError(Exception):
    """Raised when no Conversation exists with the given id."""


@dataclass(frozen=True)
class MessageSource:
    """One Retrieved Chunk that backed an assistant Message's Answer."""

    source: str
    chunk_index: int
    text: str


@dataclass(frozen=True)
class Message:
    """One message in a Conversation.

    An assistant Message may carry ``sources`` — the Retrieved Chunks that
    grounded its Answer — so citations persist with the Conversation.
    """

    role: str
    content: str
    sources: tuple[MessageSource, ...] = ()


@dataclass(frozen=True)
class Conversation:
    """A named Conversation between the user and the model, with its own retained history."""

    id: str
    name: str
    created_at: datetime
    updated_at: datetime
    messages: tuple[Message, ...]


def _derive_name(content: str) -> str:
    """Derive a usable Conversation name from a message's content."""
    collapsed = " ".join(content.split())
    if not collapsed:
        return _PLACEHOLDER_NAME
    if len(collapsed) <= _MAX_NAME_LENGTH:
        return collapsed
    return collapsed[:_MAX_NAME_LENGTH].rstrip() + _ELLIPSIS


def _source_payload(source: MessageSource) -> dict[str, Any]:
    """Serialize a MessageSource to its JSON payload shape."""
    return {"source": source.source, "chunk_index": source.chunk_index, "text": source.text}


def _source_from_payload(payload: dict[str, Any]) -> MessageSource:
    """Deserialize a MessageSource from its JSON payload shape."""
    return MessageSource(
        source=payload["source"],
        chunk_index=int(payload["chunk_index"]),
        text=payload["text"],
    )


def _message_payload(message: Message) -> dict[str, Any]:
    """Serialize a Message to its JSON payload shape."""
    payload: dict[str, Any] = {"role": message.role, "content": message.content}
    if message.sources:
        payload["sources"] = [_source_payload(source) for source in message.sources]
    return payload


def _message_from_payload(payload: dict[str, Any]) -> Message:
    """Deserialize a Message from its JSON payload shape."""
    return Message(
        role=payload["role"],
        content=payload["content"],
        sources=tuple(
            _source_from_payload(source) for source in payload.get("sources", [])
        ),
    )


class ConversationStore:
    """Creates, lists, loads, and appends to Conversations rooted at a directory."""

    def __init__(self, root: Path) -> None:
        self._root = root
        self._root.mkdir(parents=True, exist_ok=True)

    def create(self) -> Conversation:
        """Create a new Conversation, persist it, and return it."""
        now = datetime.now(UTC)
        conversation = Conversation(
            id=uuid.uuid4().hex,
            name=_PLACEHOLDER_NAME,
            created_at=now,
            updated_at=now,
            messages=(),
        )
        self._write(conversation)
        logger.debug("Created Conversation %s", conversation.id)
        return conversation

    def load(self, conversation_id: str) -> Conversation:
        """Load the Conversation with ``conversation_id``.

        Raises:
            ConversationNotFoundError: If no Conversation with that id exists.
        """
        path = self._conversation_path(conversation_id)
        if not path.is_file():
            raise ConversationNotFoundError(
                f"No Conversation with id {conversation_id!r} in {self._root}"
            )
        payload = json.loads(path.read_text(encoding="utf-8"))
        return Conversation(
            id=payload["id"],
            name=payload["name"],
            created_at=datetime.fromisoformat(payload["created_at"]),
            updated_at=datetime.fromisoformat(payload["updated_at"]),
            messages=tuple(
                _message_from_payload(message) for message in payload["messages"]
            ),
        )

    def list_conversations(self) -> list[Conversation]:
        """List every persisted Conversation, most recently updated first."""
        conversations = [self.load(path.stem) for path in self._root.glob("*.json")]
        return sorted(conversations, key=lambda conversation: conversation.updated_at, reverse=True)

    def delete(self, conversation_id: str) -> None:
        """Delete the Conversation with ``conversation_id``.

        Raises:
            ConversationNotFoundError: If no Conversation with that id exists.
        """
        path = self._conversation_path(conversation_id)
        if not path.is_file():
            raise ConversationNotFoundError(
                f"No Conversation with id {conversation_id!r} in {self._root}"
            )
        path.unlink()
        logger.debug("Deleted Conversation %s", conversation_id)

    def append(self, conversation_id: str, message: Message) -> Conversation:
        """Append ``message`` to the Conversation and return the updated Conversation.

        The first usable message names the Conversation; later messages leave
        the name alone.
        """
        conversation = self.load(conversation_id)
        updated = replace(
            conversation,
            name=(
                _derive_name(message.content)
                if conversation.name == _PLACEHOLDER_NAME
                else conversation.name
            ),
            updated_at=datetime.now(UTC),
            messages=(*conversation.messages, message),
        )
        self._write(updated)
        logger.debug(
            "Appended %s message to Conversation %s", message.role, conversation_id
        )
        return updated

    def _write(self, conversation: Conversation) -> None:
        payload = {
            "id": conversation.id,
            "name": conversation.name,
            "created_at": conversation.created_at.isoformat(),
            "updated_at": conversation.updated_at.isoformat(),
            "messages": [_message_payload(message) for message in conversation.messages],
        }
        path = self._conversation_path(conversation.id)
        temp_path = path.with_suffix(".json.tmp")
        temp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        temp_path.replace(path)

    def _conversation_path(self, conversation_id: str) -> Path:
        return self._root / f"{conversation_id}.json"
