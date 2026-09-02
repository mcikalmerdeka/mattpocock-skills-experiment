"""The Doc QA web app (Streamlit composition root).

This module is the only place that wires real configuration into the app;
every module behind it stays injectable and testable. The Conversation shell talks
only to the conversations module — the answering model arrives later.
"""

from __future__ import annotations

import streamlit as st

from doc_qa.config import ConfigError, load_settings
from doc_qa.conversations import ConversationNotFoundError, ConversationStore, Message

st.set_page_config(page_title="Doc QA", page_icon="📄")

try:
    settings = load_settings()
except ConfigError as error:
    st.error(f"**Doc QA can't start: configuration problem**\n\n{error}")
    st.stop()

store = ConversationStore(settings.data_dir / "conversations")

if "selected_conversation_id" not in st.session_state:
    st.session_state.selected_conversation_id = None


def _select(conversation_id: str) -> None:
    st.session_state.selected_conversation_id = conversation_id


with st.sidebar:
    st.subheader("Conversations")
    if st.button("New Conversation", use_container_width=True):
        _select(store.create().id)
        st.rerun()

    for listed in store.list_conversations():
        selected = listed.id == st.session_state.selected_conversation_id
        if st.button(
            listed.name,
            key=f"conversation:{listed.id}",
            use_container_width=True,
            type="primary" if selected else "secondary",
        ):
            _select(listed.id)
            st.rerun()

    st.divider()
    st.subheader("Configuration")
    st.markdown(f"**Chat model:** `{settings.chat_model}`")
    st.markdown(f"**Reasoning effort:** `{settings.reasoning_effort}`")
    st.markdown(f"**Embedding model:** `{settings.embedding_model}`")
    if settings.openai_api_key:
        st.markdown("**OpenAI API key:** ✅ present (masked, never shown)")

st.title("Doc QA 📄")
st.caption("Chat with your TXT & Markdown documents.")

selected_id = st.session_state.selected_conversation_id
if selected_id is None:
    st.info("Start a new Conversation from the sidebar, or reopen a past one.")
    st.stop()

try:
    conversation = store.load(selected_id)
except ConversationNotFoundError:
    st.session_state.selected_conversation_id = None
    st.info("That Conversation no longer exists. Pick another from the sidebar.")
    st.stop()

st.subheader(conversation.name)

if conversation.messages:
    for message in conversation.messages:
        with st.chat_message(message.role):
            st.markdown(message.content)
else:
    st.info("No messages yet — send the first one below.", icon="💬")

if prompt := st.chat_input("Type a message…"):
    store.append(conversation.id, Message(role="user", content=prompt))
    st.rerun()
