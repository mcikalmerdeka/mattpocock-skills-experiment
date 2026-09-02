"""The Doc QA web app (Streamlit composition root).

This module is the only place that wires real configuration into the app;
every module behind it stays injectable and testable.
"""

from __future__ import annotations

import streamlit as st

from doc_qa.config import ConfigError, load_settings

st.set_page_config(page_title="Doc QA", page_icon="📄")

try:
    settings = load_settings()
except ConfigError as error:
    st.error(f"**Doc QA can't start: configuration problem**\n\n{error}")
    st.stop()

with st.sidebar:
    st.subheader("Configuration")
    st.markdown(f"**Chat model:** `{settings.chat_model}`")
    st.markdown(f"**Reasoning effort:** `{settings.reasoning_effort}`")
    st.markdown(f"**Embedding model:** `{settings.embedding_model}`")
    if settings.openai_api_key:
        st.markdown("**OpenAI API key:** ✅ present (masked, never shown)")

st.title("Doc QA 📄")
st.caption("Chat with your TXT & Markdown documents.")

st.info(
    "The walking skeleton is alive: configuration, UI, and packaging are wired. "
    "Document Ingestion and Conversations arrive in the next tickets.",
    icon="🚧",
)
