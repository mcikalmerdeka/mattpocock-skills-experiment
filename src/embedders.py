"""Embedder provider seam: how text turns into embeddings.

The Vector Store embeds Chunks and Queries through an injected langchain
``Embeddings``; the composition root constructs the real OpenAI embeddings
client — through LangChain — from configuration.
"""

from __future__ import annotations

from langchain_core.embeddings import Embeddings


class OpenAIEmbedder(Embeddings):
    """The real Embedder: OpenAI's embeddings API through LangChain, bound to the configured model."""

    def __init__(self, api_key: str, model: str) -> None:
        from langchain_openai import OpenAIEmbeddings

        self._client = OpenAIEmbeddings(api_key=api_key, model=model)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed ``texts`` with the configured model, preserving input order."""
        return self._client.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        """Embed ``text`` for Retrieval."""
        return self._client.embed_query(text)
