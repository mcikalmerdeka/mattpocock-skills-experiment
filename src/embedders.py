"""Embedder provider seam: how text turns into embeddings.

Ingestion talks only to the :class:`Embedder` protocol; the composition root
constructs the real OpenAI embeddings client from configuration.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol


class Embedder(Protocol):
    """Turns texts into embedding vectors, one per input, in input order."""

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed ``texts`` and return one vector per input, in input order."""
        ...


class OpenAIEmbedder:
    """The real Embedder: OpenAI's embeddings API behind the provider seam."""

    def __init__(self, api_key: str, model: str) -> None:
        from openai import OpenAI

        self._client = OpenAI(api_key=api_key)
        self._model = model

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed ``texts`` with the configured model, preserving input order."""
        if not texts:
            return []
        response = self._client.embeddings.create(model=self._model, input=list(texts))
        ordered = sorted(response.data, key=lambda item: item.index)
        return [item.embedding for item in ordered]
