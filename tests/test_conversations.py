"""Tests for the conversations module: create, list, load, and append Conversations against temp directories.

Each Conversation persists as its own JSON file (id, name, created/updated
timestamps, ordered messages) and round-trips losslessly.
"""

import json
from pathlib import Path

import pytest

from src.conversations import (
    Conversation,
    ConversationNotFoundError,
    ConversationStore,
    Message,
)

_NEW_CONVERSATION_KEYS = {"id", "name", "created_at", "updated_at", "messages"}


def test_loaded_conversation_round_trips_losslessly(tmp_path: Path) -> None:
    created = ConversationStore(tmp_path).create()

    reloaded = ConversationStore(tmp_path).load(created.id)

    assert reloaded == created


def test_append_keeps_messages_in_order_and_bumps_updated_at(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path)
    created = store.create()

    updated = store.append(created.id, Message(role="user", content="first message"))
    updated = store.append(updated.id, Message(role="user", content="second message"))

    assert [message.content for message in updated.messages] == ["first message", "second message"]
    assert updated.created_at == created.created_at
    assert updated.updated_at > created.updated_at

    reloaded = ConversationStore(tmp_path).load(created.id)
    assert reloaded == updated


def test_first_message_names_the_conversation(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path)
    created = store.create()

    named = store.append(created.id, Message(role="user", content="What is  retrieval\naugmentation?"))
    again = store.append(named.id, Message(role="user", content="Second message"))

    assert named.name == "What is retrieval augmentation?"
    assert again.name == "What is retrieval augmentation?"


def test_long_first_message_is_truncated_in_the_name(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path)
    created = store.create()
    long_message = "x" * 200

    named = store.append(created.id, Message(role="user", content=long_message))

    assert named.name == "x" * 60 + "…"


def test_whitespace_first_message_defers_naming_to_the_next_usable_message(
    tmp_path: Path,
) -> None:
    store = ConversationStore(tmp_path)
    created = store.create()

    blank = store.append(created.id, Message(role="user", content="   \n\t  "))
    named = store.append(blank.id, Message(role="user", content="Now a real message"))

    assert blank.name == "New Conversation"
    assert named.name == "Now a real message"


def test_list_returns_every_persisted_conversation_most_recently_updated_first(
    tmp_path: Path,
) -> None:
    store = ConversationStore(tmp_path)
    first = store.create()
    second = store.create()
    third = store.create()

    store.append(first.id, Message(role="user", content="hello"))

    listed = store.list_conversations()

    # `first` was updated last, so it leads; the untouched ones may tie behind it.
    assert listed[0].id == first.id
    assert {conversation.id for conversation in listed} == {first.id, second.id, third.id}
    assert len(listed) == 3


def test_loading_an_unknown_conversation_raises_a_readable_error(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path)

    with pytest.raises(ConversationNotFoundError, match="no-such-id"):
        store.load("no-such-id")


def test_create_persists_conversation_as_its_own_json_file(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path)

    conversation = store.create()

    assert conversation.id
    assert conversation.name == "New Conversation"
    assert conversation.messages == ()
    assert conversation.created_at == conversation.updated_at

    conversation_file = tmp_path / f"{conversation.id}.json"
    assert conversation_file.is_file()

    stored = json.loads(conversation_file.read_text(encoding="utf-8"))
    assert set(stored) == _NEW_CONVERSATION_KEYS
    assert stored["id"] == conversation.id
    assert stored["messages"] == []
