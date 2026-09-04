# Doc QA — Improvement Tracking

Improvements adopted from the reference RAG project (`reference/`, DocuChat AI),
following the analysis of the current codebase against it and against the spec
(GitHub issue #1). Worked checkpoint by checkpoint, test-first, at the project's
existing seams.

**Verification protocol:** `uv run pytest` after every checkpoint; full suite
plus editor diagnostics at the end. UI changes (`src/app.py`) are outside unit
tests per the spec's testing decisions — verify with the manual smoke checklist.

**Domain language:** see `CONTEXT.md` (Document, Chunk, Ingestion, Vector Store,
Conversation, Query, Retrieval, Answer).

## Checkpoints

| # | Improvement | Scope | Status |
|---|-------------|-------|--------|
| CP1 | Source citations (§3.1) | `src/answering.py`, `src/conversations.py`, `src/app.py`, tests | ✅ done |
| CP2 | Error handling around model & ingest calls (§3.2) | `src/app.py` | ✅ done |
| CP3 | Logging (§3.3) | `src/logging_setup.py` (new), `src/app.py`, core modules | ✅ done |
| CP4 | Delete Conversations & Documents (§3.4) | `src/conversations.py`, `src/ingestion.py`, `src/vector_store.py`, `src/app.py`, tests | ✅ done |
| CP5 | Prompt templates module (§3.5) | `src/prompts.py` (new), `src/answering.py` | ✅ done |

## CP1 — Source citations

Goal: every Answer shows where it came from. The Reference project's standout
UX ("📚 View Sources" with filename + chunk info per answer).

Design decisions:

- `Answering.answer()` returns an `Answer` — the Answer text **plus** the
  Retrieved Chunks backing it (previously the Chunks were discarded).
- The system prompt labels each Retrieved excerpt with its source Document, so
  the model can point at passages (spec stories #12, #13).
- `Message` gains an optional `sources` payload (`source`, `chunk_index`,
  `text` per Retrieved Chunk) so citations persist with the Conversation.
  Old messages without the `sources` key load with empty sources
  (backward compatible).
- The chat UI renders a "📚 View Sources" expander under assistant messages,
  grouped by source Document.

Seams under test:

- `tests/test_answering.py` — the Answer carries the Retrieved Chunks; the
  system prompt labels each excerpt with its source Document (and only the
  Retrieved ones).
- `tests/test_conversations.py` — a Message with sources round-trips
  losslessly; legacy messages (no `sources` key) load with empty sources.

Manual smoke: upload a Document → ask a question → expander lists the source
Document and Chunks → restart → reopening the Conversation still shows them.

## CP2 — Error handling around model & ingest calls

Goal: an OpenAI/Chroma failure (rate limit, timeout, network) shows a readable
`st.error` instead of crashing the Streamlit rerun, and the Conversation stays
usable.

Design decisions:

- The user's Query is already persisted before the model call, so on failure it
  stays in the Conversation history (durable, re-answerable on retry).
- No assistant message is appended on failure — error text never pollutes
  history. The error surfaces via `st.session_state` so it survives the rerun.
- The ingest path is guarded the same way: an embedding failure during upload
  reports the error and leaves the app running.

Seams under test: none — UI boundary, per the spec's testing decisions.
Covered by the manual smoke checklist.

## CP3 — Logging

Goal: the next startup/Chroma/OpenAI misbehavior leaves a trace (the Chroma
cache bug was painful precisely because nothing was logged).

Design decisions:

- New `src/logging_setup.py`: a `"docqa"` logger — INFO on the console, DEBUG
  to a file under `data/logs/` (`delay=True` so the file is only created on
  first write). Configured once at the composition root.
- Core modules log through `logging.getLogger(__name__)` — Ingestion outcomes,
  Retrieval sizes, Answer completions — debug/info only, no behavior change.
- Modules stay pure: they only *use* a logger; configuration happens at the
  composition root.

Seams under test: none — logging is an observability side channel; existing
tests must stay green unchanged.

## CP4 — Delete Conversations & Documents

Goal: ChatGPT-style apps need a way to remove things. Once ingested, a
Document currently pollutes the global knowledge base forever, and abandoned
Conversations accumulate in the sidebar.

Design decisions:

- `ConversationStore.delete(conversation_id)` removes the JSON file; unknown
  ids raise `ConversationNotFoundError` (same contract as `load`).
- `Ingestion.delete_document(source)` removes the Document's Chunks **and** its
  registry entry; returns whether anything was deleted. New
  `VectorStore.unregister_document(source)` supports it.
- Sidebar gains a delete button per Conversation and per Document. Deleting the
  selected Conversation clears the selection.

Seams under test:

- `tests/test_conversations.py` — delete removes the Conversation file; delete
  of an unknown id raises a readable error.
- `tests/test_ingestion.py` — delete removes the Chunks and the registry entry
  (count 0, hash `None`, source gone from listings); deleting an unknown
  Document reports nothing to delete.

## CP5 — Prompt templates module

Goal: prompts live in one place (`src/prompts.py`) so future modes (e.g. a
grounding toggle) are a one-file change — the Reference keeps templates in
`config/prompts/templates.py`.

Design decisions:

- `src/prompts.py` owns the system template and the excerpt formatting
  (including CP1's source labels); `src/answering.py` imports it.
- Pure extraction: behavior unchanged, all existing tests stay green.

## Deliberately out of scope (from the analysis)

- §3.6 Multi-format ingestion (PDF/DOCX) — spec defers formats beyond TXT/MD;
  `Document(filename, content)` is already format-agnostic when needed.
- §3.7 External search fallback — contradicts the spec's grounding contract
  (story #13). Possible future toggle, not an improvement to port now.
- Spec reconciliation — issue #1 still says top-5 Retrieval and uncapped
  history; the code deliberately uses top-4 and a 20-message cap. Edit the
  issue text when convenient (no code change).

## Manual smoke checklist (run once at the end)

1. `uv run doc-qa` starts; sidebar shows settings and the API-key check.
2. Upload a TXT/MD Document → success message with Chunk count.
3. Re-upload the same file → "skipped"; upload changed content under the same
   name → "replaced".
4. Ask a question → Answer appears with a 📚 View Sources expander naming the
   Document; follow-up works; restart → Conversation and citations persist.
5. Stop the model (e.g. invalid key in `.env`) → readable error, Conversation
   still usable.
6. Delete a Conversation → gone from sidebar; delete a Document → gone from
   the ingested list and no longer answerable.
