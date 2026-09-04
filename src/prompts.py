"""The prompts module: system prompts for the Answer engine.

The grounded system prompt instructs the model to answer only from the
Retrieved excerpts — each labeled with its source Document, so the model can
point at passages. Keeping prompts here means future modes (e.g. a stricter
grounding toggle) are a one-file change.
"""

from __future__ import annotations

from collections.abc import Sequence

from src.vector_store import Chunk

_SYSTEM_TEMPLATE = """You are the assistant of a document question-answering app. Answer the user's Query using only the Excerpts below — passages Retrieved from the Documents the user has ingested. If the Excerpts do not contain the Answer, say so plainly; do not invent content. Point at the relevant passage when that helps.

Excerpts:
{context}"""


def _labeled_excerpts(chunks: Sequence[Chunk]) -> str:
    """Format Retrieved Chunks as excerpts, each labeled with its source Document."""
    return "\n\n".join(
        f"--- Excerpt {number}: from {chunk.source} (Chunk {chunk.index}) ---\n{chunk.text}"
        for number, chunk in enumerate(chunks, start=1)
    )


def grounded_system_prompt(chunks: Sequence[Chunk]) -> str:
    """Build the grounded system prompt for the Retrieved Chunks."""
    return _SYSTEM_TEMPLATE.format(context=_labeled_excerpts(chunks))
