"""The Doc QA web app (Streamlit composition root).

This module is the only place that wires real configuration into the app;
every module behind it stays injectable and testable. The Conversation shell
talks only to the conversations module, Ingestion writes Chunks through the
Vector Store's injected Embeddings, and Answering completes Conversations
through the injected ChatModel — the real LangChain clients (embeddings and
chat) are constructed here, from configuration.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import streamlit as st

from src.answering import Answering, OpenAIChatModel
from src.config import ConfigError, load_settings
from src.conversations import (
    ConversationNotFoundError,
    ConversationStore,
    Message,
    MessageSource,
)
from src.embedders import OpenAIEmbedder
from src.ingestion import Document, Ingestion
from src.logging_setup import configure_logging, get_logger
from src.vector_store import VectorStore

if TYPE_CHECKING:
    from streamlit.runtime.uploaded_file_manager import UploadedFile

st.set_page_config(page_title="Doc QA", page_icon="📄")

try:
    settings = load_settings()
except ConfigError as error:
    st.error(f"**Doc QA can't start: configuration problem**\n\n{error}")
    st.stop()

configure_logging(settings.data_dir / "logs")
logger = get_logger("app")
logger.info(
    "Doc QA starting — chat_model=%s, reasoning_effort=%s, embedding_model=%s",
    settings.chat_model,
    settings.reasoning_effort,
    settings.embedding_model,
)


@st.cache_resource
def _build_vector_store(path: str, api_key: str, embedding_model: str) -> VectorStore:
    """Build the Vector Store once per process, at an absolute path.

    Chroma keys its shared System cache by the raw path string
    (trychroma/chroma#7253, fix unreleased in 1.5.9), and a failing client
    init can evict the entry out from under a concurrent constructor — which
    surfaces as KeyError on the path. Constructing once per process with a
    resolved absolute path keeps the identifier stable and keeps reruns from
    ever racing a construction.
    """
    return VectorStore(Path(path), OpenAIEmbedder(api_key=api_key, model=embedding_model))


@st.cache_resource
def _build_chat_model(api_key: str, chat_model: str, reasoning_effort: str) -> OpenAIChatModel:
    """Build the Chat Model once per process, from configuration."""
    return OpenAIChatModel(api_key=api_key, model=chat_model, reasoning_effort=reasoning_effort)


vector_store = _build_vector_store(
    str((settings.data_dir / "vector_store").resolve()),
    api_key=settings.openai_api_key,
    embedding_model=settings.embedding_model,
)
store = ConversationStore(settings.data_dir / "conversations")
ingestion = Ingestion(store=vector_store)
answering = Answering(
    store=vector_store,
    chat_model=_build_chat_model(
        api_key=settings.openai_api_key,
        chat_model=settings.chat_model,
        reasoning_effort=settings.reasoning_effort,
    ),
)

if "selected_conversation_id" not in st.session_state:
    st.session_state.selected_conversation_id = None


def _select(conversation_id: str) -> None:
    st.session_state.selected_conversation_id = conversation_id


def _grouped_sources(
    sources: Sequence[MessageSource],
) -> list[tuple[str, list[MessageSource]]]:
    """Group a Message's sources by their source Document, preserving order."""
    grouped: dict[str, list[MessageSource]] = {}
    for source in sources:
        grouped.setdefault(source.source, []).append(source)
    return list(grouped.items())


def _ingest_upload(ingestion: Ingestion, upload: UploadedFile) -> None:
    """Ingest one uploaded Document and report the outcome, every time.

    Streamlit reruns the whole script on every interaction while an upload
    stays selected, but re-ingesting is safe: Ingestion's dedup short-circuits
    identical content (reported as skipped) before any embedding happens, and
    changed content under a known filename reports as replaced. An embedding
    failure reports a readable error and leaves the app running.
    """
    try:
        content = upload.getvalue().decode("utf-8")
    except UnicodeDecodeError:
        st.error(
            f"**{upload.name}** can't be read as UTF-8 text. "
            "Only TXT & Markdown documents are supported."
        )
        return

    if not content.strip():
        st.error(f"**{upload.name}** is empty — there is nothing to ingest.")
        return

    try:
        result = ingestion.ingest(Document(filename=upload.name, content=content))
    except Exception as error:
        st.error(
            f"Ingesting **{upload.name}** failed — the document is not answerable "
            f"right now. Try uploading it again.\n\n{error}"
        )
        return

    if result.status == "ingested":
        st.success(f"Ingested **{upload.name}** — {result.chunk_count} Chunks embedded.")
    elif result.status == "skipped":
        st.info(f"**{upload.name}** skipped — identical content is already ingested.")
    else:
        st.warning(
            f"Replaced **{upload.name}** — old Chunks removed, "
            f"{result.chunk_count} new Chunks embedded."
        )


with st.sidebar:
    st.subheader("Conversations")
    if st.button("New Conversation", use_container_width=True):
        _select(store.create().id)
        st.rerun()

    for listed in store.list_conversations():
        selected = listed.id == st.session_state.selected_conversation_id
        row = st.columns([0.82, 0.18])
        if row[0].button(
            listed.name,
            key=f"conversation:{listed.id}",
            use_container_width=True,
            type="primary" if selected else "secondary",
        ):
            _select(listed.id)
            st.rerun()
        if row[1].button(
            "🗑",
            key=f"delete-conversation:{listed.id}",
            help=f"Delete “{listed.name}”",
        ):
            store.delete(listed.id)
            if selected:
                st.session_state.selected_conversation_id = None
            st.rerun()

    st.divider()
    st.subheader("Documents")
    upload = st.file_uploader("Add a TXT or Markdown document", type=["txt", "md", "markdown"])
    if upload is not None:
        _ingest_upload(ingestion, upload)

    ingested = ingestion.ingested_documents()
    if ingested:
        for name in ingested:
            row = st.columns([0.82, 0.18])
            row[0].markdown(f"- {name}")
            if row[1].button("🗑", key=f"delete-document:{name}", help=f"Delete “{name}”"):
                ingestion.delete_document(name)
                st.rerun()
    else:
        st.caption("No Documents ingested yet.")

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

# A failed model call is reported on the rerun (Streamlit clears inline
# messages on rerun, so the error travels through session state).
if "answer_error" in st.session_state:
    st.error(st.session_state.pop("answer_error"))

if conversation.messages:
    for message in conversation.messages:
        with st.chat_message(message.role):
            st.markdown(message.content)
            if message.sources:
                with st.expander("📚 View Sources"):
                    for source, chunks in _grouped_sources(message.sources):
                        indices = ", ".join(str(chunk.chunk_index) for chunk in chunks)
                        st.markdown(f"**{source}** — Chunks {indices}")
                        for chunk in chunks:
                            st.markdown(chunk.text)
else:
    st.info("No messages yet — send the first one below.", icon="💬")

if query := st.chat_input("Type a message…"):
    updated = store.append(conversation.id, Message(role="user", content=query))
    try:
        with st.spinner("Thinking…"):
            answer = answering.answer(updated)
    except Exception as error:
        # The Query is already persisted, so it stays in the Conversation and
        # can be retried; no assistant Message is appended, so error text
        # never enters the history.
        st.session_state.answer_error = (
            "The model call failed — your message is saved above, try sending "
            f"it again.\n\n{error}"
        )
        st.rerun()
    store.append(
        conversation.id,
        Message(
            role="assistant",
            content=answer.text,
            sources=tuple(
                MessageSource(
                    source=chunk.source, chunk_index=chunk.index, text=chunk.text
                )
                for chunk in answer.sources
            ),
        ),
    )
    st.rerun()
