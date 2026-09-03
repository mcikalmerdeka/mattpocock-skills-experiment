"""The conversations module: creates, lists, loads, and appends to Conversations.

Each Conversation persists as its own JSON file — ``{id}.json`` inside the
store's root directory — holding the conversation's id, name, created/updated
timestamps, and ordered messages. A new Conversation is named from its first
message; until then it carries a placeholder name.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path

_PLACEHOLDER_NAME = "New Conversation"
_MAX_NAME_LENGTH = 60
_ELLIPSIS = "…"


class ConversationNotFoundError(Exception):
    """Raised when no Conversation exists with the given id."""


@dataclass(frozen=True)
class Message:
    """One message in a Conversation."""

    role: str
    content: str


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


def _message_payload(message: Message) -> dict[str, str]:
    """Serialize a Message to its JSON payload shape."""
    return {"role": message.role, "content": message.content}


def _message_from_payload(payload: dict[str, str]) -> Message:
    """Deserialize a Message from its JSON payload shape."""
    return Message(role=payload["role"], content=payload["content"])


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
