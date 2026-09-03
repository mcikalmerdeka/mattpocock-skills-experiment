"""The Doc QA web app (Streamlit composition root).

This module is the only place that wires real configuration into the app;
every module behind it stays injectable and testable. The Conversation shell
talks only to the conversations module, and Ingestion embeds through the
injected Embedder — the real OpenAI client is constructed here, from
configuration.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import streamlit as st

from src.config import ConfigError, load_settings
from src.conversations import ConversationNotFoundError, ConversationStore, Message
from src.embedders import OpenAIEmbedder
from src.ingestion import Document, Ingestion
from src.vector_store import VectorStore

if TYPE_CHECKING:
    from streamlit.runtime.uploaded_file_manager import UploadedFile

st.set_page_config(page_title="Doc QA", page_icon="📄")

try:
    settings = load_settings()
except ConfigError as error:
    st.error(f"**Doc QA can't start: configuration problem**\n\n{error}")
    st.stop()


@st.cache_resource
def _build_vector_store(path: str) -> VectorStore:
    """Build the Vector Store once per process, at an absolute path.

    Chroma keys its shared System cache by the raw path string
    (trychroma/chroma#7253, fix unreleased in 1.5.9), and a failing client
    init can evict the entry out from under a concurrent constructor — which
    surfaces as KeyError on the path. Constructing once per process with a
    resolved absolute path keeps the identifier stable and keeps reruns from
    ever racing a construction.
    """
    return VectorStore(Path(path))


store = ConversationStore(settings.data_dir / "conversations")
ingestion = Ingestion(
    store=_build_vector_store(str((settings.data_dir / "vector_store").resolve())),
    embedder=OpenAIEmbedder(api_key=settings.openai_api_key, model=settings.embedding_model),
)

if "selected_conversation_id" not in st.session_state:
    st.session_state.selected_conversation_id = None


def _select(conversation_id: str) -> None:
    st.session_state.selected_conversation_id = conversation_id


def _ingest_upload(ingestion: Ingestion, upload: UploadedFile) -> None:
    """Ingest one uploaded Document and report the outcome, every time.

    Streamlit reruns the whole script on every interaction while an upload
    stays selected, but re-ingesting is safe: Ingestion's dedup short-circuits
    identical content (reported as skipped) before any embedding happens, and
    changed content under a known filename reports as replaced.
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

    result = ingestion.ingest(Document(filename=upload.name, content=content))
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
        if st.button(
            listed.name,
            key=f"conversation:{listed.id}",
            use_container_width=True,
            type="primary" if selected else "secondary",
        ):
            _select(listed.id)
            st.rerun()

    st.divider()
    st.subheader("Documents")
    upload = st.file_uploader("Add a TXT or Markdown document", type=["txt", "md", "markdown"])
    if upload is not None:
        _ingest_upload(ingestion, upload)

    ingested = ingestion.ingested_documents()
    if ingested:
        st.markdown("\n".join(f"- {name}" for name in ingested))
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

if conversation.messages:
    for message in conversation.messages:
        with st.chat_message(message.role):
            st.markdown(message.content)
else:
    st.info("No messages yet — send the first one below.", icon="💬")

if query := st.chat_input("Type a message…"):
    store.append(conversation.id, Message(role="user", content=query))
    st.rerun()
