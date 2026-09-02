# Using the mattpocock-skills for development

A complete, worked guide to the engineering skillset installed in this repo — from installation through implementing tickets. Everything below was actually executed in this repository to build **Doc QA** (a document question-answering app), so real artifacts are referenced throughout.

---

## 1. Installation (the short version)

The skills live in `.agents/skills/` and are pinned by `skills-lock.json`. They are discovered automatically by OpenCode (and compatible agents) as slash commands — no per-session setup needed. Prerequisites for the *tracker integration*:

- **`gh` CLI** installed and authenticated (`gh auth login` — browser flow, one time per machine)
- A GitHub repo (the issue tracker); a git remote pointing at it

---

## 2. One-time setup: `/setup-matt-pocock-skills`

Run this **once, before any engineering flow**. It asks three questions and writes the configuration every other skill assumes:

| Decision | What it configures | Where it's written |
|---|---|---|
| Issue tracker | Where issues live (GitHub Issues via `gh` CLI, in our case) | `docs/agents/issue-tracker.md` |
| Triage labels | Five canonical roles: `needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix` | `docs/agents/triage-labels.md` |
| Domain docs layout | Single-context (root `CONTEXT.md` + `docs/adr/`) | `docs/agents/domain.md` |

It also creates/updates `AGENTS.md` with an `## Agent skills` block pointing at those files. Re-run it only if you switch trackers.

---

## 3. The map: one main flow, idea → ship

```
 idea
  │
  ▼
 /grill-with-docs ──────── sharpen the idea by interview (writes CONTEXT.md + ADRs)
  │
  ├─ design question needs runnable code? → /handoff → /prototype → /handoff → back
  │
  ▼
 multi-session build?
  │
  ├─ NO  → /implement (same window)
  │
  └─ YES → /to-spec ──► /to-tickets ──► /new ──► /implement (one per ticket)
                        (spec issue)  (tickets w/ blocking edges)
```

**Doc QA followed the multi-session branch.** Here's each stage as we ran it.

### Stage A — `/grill-with-docs` (the interview)

The idea ("a simple LLM app with LangChain") entered as fog. The grill works in **rounds**: each round asks every question whose prerequisites are settled — numbered, with a recommended answer — then waits. Decisions are yours; facts are the agent's (it verified `gpt-5.6-luna` against OpenAI's live docs rather than asking or guessing).

Four rounds settled: document QA over uploaded files · this repo, Python + uv · Streamlit UI, Chroma, OpenAI · global vector store, persisted conversations, hash-based dedup, model config in `.env`.

**Domain modeling runs underneath.** As terms crystallized, the agent wrote the glossary — `CONTEXT.md` (Document, Chunk, Ingestion, Vector Store, Conversation, Query, Retrieval, Answer) — and it became the shared vocabulary for everything after. ADRs were considered and deliberately skipped: nothing we decided was hard to reverse.

> Use `/grill-me` instead when there's no repo to write docs into; it's the same interview, stateless.

### Stage B — `/to-spec` (synthesis, not another interview)

Turned the whole conversation into a spec and published it as **issue #1**, labeled `ready-for-agent`. The template: Problem Statement · Solution · 22 User Stories · Implementation Decisions · Testing Decisions · Out of Scope · Further Notes.

The critical step before publishing: the agent proposed the **test seams** and checked them with us. One seam — the OpenAI provider boundary (chat model + embedder injected); Chroma, JSON persistence, and config all run *real* in tests. Fewer seams, fewer fakes, tests that survive refactors.

> The spec issue is the **parent reference** — it stays open while tickets are built, and is closed only when the build is delivered.

### Stage C — `/to-tickets` (tracer bullets with blocking edges)

Split the spec into **vertical slices** — each ticket cuts a complete, demoable path through every layer, sized for one fresh context window:

| # | Ticket | Blocked by | Delivers |
|---|---|---|---|
| [#2](https://github.com/mcikalmerdeka/mattpocock-skills-experiment/issues/2) | T1: Walking skeleton — app boots on real config | — | Streamlit boots, reads `.env`, clear config errors, pytest via uv |
| [#3](https://github.com/mcikalmerdeka/mattpocock-skills-experiment/issues/3) | T2: ChatGPT-style shell with persisted Conversations | #2 | Sidebar, conversations as JSON, survives restart |
| [#4](https://github.com/mcikalmerdeka/mattpocock-skills-experiment/issues/4) | T3: Ingestion — upload Documents into the Vector Store | #2 | TXT/MD upload → chunks → Chroma, dedup outcomes, docs list |
| [#5](https://github.com/mcikalmerdeka/mattpocock-skills-experiment/issues/5) | T4: Grounded Answers — the brain plugs in | #3, #4 | Query → top-4 retrieval → gpt-5.6-luna → grounded answer |

The graph: **#2 → {#3 ∥ #4} → #5**. Edges are GitHub's *native issue dependencies* (so the frontier is queryable), all tickets carry `ready-for-agent`. The breakdown was drafted, quizzed against us (granularity? edges? merge/split?), then published in dependency order.

### Stage D — `/implement`, one ticket per fresh session

This is where the rubber meets the road — and it hasn't run yet for Doc QA. The protocol:

1. **`/new`** (OpenCode's session reset; Claude Code calls it `/clear`) — the grill/spec thinking is published and done; don't drag it forward
2. **`/implement`** against the frontier ticket — the issue whose blockers are all closed (starts with #2)
3. `/implement` drives **`/tdd`** internally: one red-green slice at a time, tests first
4. It closes with **`/code-review`** — a two-axis review (Standards: repo conventions; Spec: does it match the ticket) — before committing
5. Ticket done → close it → the frontier advances (**#3 and #4 unblock in parallel** → #5 last)
6. Repeat `/new` → `/implement` per ticket; each session starts fresh from the ticket alone

**Context hygiene** is the whole trick: grill → spec → tickets happen in *one* unbroken window (they build on the same thinking); each `/implement` gets a *clean* window (the ticket is self-contained by construction).

---

## 4. Command reference

### The main flow

| Command | What it does | When to reach for it |
|---|---|---|
| `/grill-with-docs` | Relentless interview that sharpens an idea, writing `CONTEXT.md` + ADRs as terms resolve | Any new work in a repo |
| `/to-spec` | Synthesizes the conversation into a spec, publishes to the tracker, labels `ready-for-agent` | After grilling, when the build is multi-session |
| `/to-tickets` | Splits spec/plan into tracer-bullet tickets with blocking edges | After the spec |
| `/implement` | Builds one ticket test-first (`/tdd` inside), closes with `/code-review` | One ticket per fresh session |
| `/tdd` | Red-green-refactor discipline on its own | Building a concrete behavior test-first |
| `/code-review` | Two-axis review (Standards + Spec) of changes since a fixed point | Reviewing a branch/PR anytime |

### On-ramps (generate work, merge onto the main flow)

| Command | When |
|---|---|
| `/triage` | Bug reports / incoming requests piling up — moves them through triage roles into agent-ready issues (never for tickets `/to-tickets` made) |
| `/diagnosing-bugs` | The hard bug: refuses to theorize until there's a tight feedback loop (one command red on *this* bug) |
| `/wayfinder` | Huge foggy efforts too big for one session — charts decision tickets until the way is clear, then hands to `/to-spec` |

### Standalone tools

| Command | When |
|---|---|
| `/grill-me` | The interview, stateless — no repo, no docs |
| `/research` | Background agent reads primary sources, leaves a cited markdown in the repo |
| `/prototype` | Throwaway code to answer one design question |
| `/handoff` | Portable markdown to carry context to a new session/directory/collaborator |
| `/wizard` | Human-only steps (API keys, dashboards, migrations) as an interactive script |
| `/teach` | Learn a concept over multiple sessions |
| `/wait-what` | "That didn't land — re-explain it" mid-conversation |
| `/domain-modeling` | The glossary discipline: sharpen terms, record ADRs |
| `/codebase-design` | Deep-module vocabulary for designing module shapes |
| `/improve-codebase-architecture` | Health scan for deepening opportunities |
| `/resolving-merge-conflicts` | Mid-conflict, hunk by hunk, by intent |
| `/to-questionnaire` | When the answer lives in someone else's head |
| `/writing-for-agents` | Reference for writing skills/AGENTS.md/docs agents consume |

---

## 5. Where Doc QA stands right now

- ✅ Setup, grilling, glossary (`CONTEXT.md`), spec (#1), tickets (#2–#5, blocked edges live)
- ⏭️ Next: `/new`, then `/implement` issue **#2**
- 📌 Reminder: put the OpenAI key in `.env` before ticket **#5** (`/wizard` can walk you through it)
- 📌 #1 closes only when #2–#5 are all delivered

---

*This guide lives in `other/` — it documents the process, not the app. The app's own docs belong in the README once T1 lands.*
