# Concepts

This page explains what `reasoning-bank` is doing under the hood. If you just
want to wire it up, start with [`quickstart.md`](quickstart.md).

## The memory item

```python
class MemoryItem:
    id: str
    title: str           # short label
    description: str     # one-line "when does this apply?"
    content: str         # the actual strategy / lesson
    source: "success" | "failure" | "matts_contrast" | "manual"
    task_signature: str  # the task this was learned on
    scope: str           # partition key
    embedding: list[float] | None
    confidence: float
    use_count: int
    related_ids: list[str]
    ...
```

A `MemoryItem` is a **distilled strategy**, not a trace of a past
conversation. The bank deliberately throws away most of the trajectory and
keeps only generalizable lessons. That's what makes them reusable across
tasks.

## The pipeline

`bank.ingest_trajectory(trajectory, task)` runs four stages:

1. **Judge** — an LLM-as-judge call labels the trajectory `"success"` or
   `"failure"` (skipped if you pass `outcome=` explicitly).
2. **Induction** — a second LLM call distills 1–3 memory candidates from the
   trajectory. The prompt is different for success vs. failure: success prompts
   ask "what generalizable strategy worked here?", failure prompts ask "what
   trap should the agent avoid next time?".
3. **Embed** — each candidate's `signature_text()` (title + description +
   content) is embedded via the configured embedder.
4. **Merge / link / replace** — the candidate is compared to existing memories
   by cosine similarity. Below a low threshold → insert as new. Between the
   two thresholds → link as related. Above the high threshold → replace /
   reconcile via a merge prompt.

The two merge thresholds are configurable on the bank
(`merge_thresholds=(low, high)`, default `(0.75, 0.92)`).

## Judge

`judge.py` runs a single LLM call with `prompts/judge.txt`. The model is
expected to return JSON of the form `{"outcome": "success|failure",
"rationale": "..."}`. The rationale is logged but not stored.

You can skip the judge entirely by passing `outcome="success"` or
`outcome="failure"` to `ingest_trajectory`.

## Induction (success vs. failure)

`induction.py` selects between `prompts/induction_success.txt` and
`prompts/induction_failure.txt` based on the resolved outcome. Both prompts
ask the model to return a JSON array of at most 3 candidate memories with
`title`, `description`, `content`. The failure prompt steers toward
**avoidance strategies** ("don't do X when Y") rather than reasons-for-failure
postmortems — those generalize better.

## Retrieval

`bank.retrieve(task, k=5)` embeds the task string, asks the store for the
top-k nearest neighbors by cosine similarity, and filters by
`min_confidence` (default 0.3). Returned memories have their `use_count` and
`last_used_at` updated.

`format_as_system_block(memories)` renders the returned list as a Markdown
block suitable for prepending to a system prompt.

## Merge / link / replace

`merge.py` does the integration step. For each candidate:

- The store returns the most similar existing memory.
- If `sim < low_threshold`: insert as a new memory.
- If `low <= sim < high`: append the existing memory's id to the
  candidate's `related_ids` (and vice versa), then insert.
- If `sim >= high`: run a merge prompt that asks the LLM to consolidate the
  two into one memory; the result replaces the existing one.

This keeps the store from accumulating near-duplicates as the agent runs many
similar tasks.

## MaTTS — Memory-Aware Test-Time Scaling

`matts_run(rollout_fn, task, bank, k=4)` is a thin orchestrator:

1. `asyncio.gather` k copies of `rollout_fn()`.
2. Coerce each result to a `Trajectory`.
3. Render all k trajectories into a single contrast prompt
   (`prompts/matts_contrast.txt`).
4. Ask the contrast LLM for up to 3 cross-rollout insights.
5. Embed + integrate each insight through the normal merge pipeline, with
   `source="matts_contrast"`.

`rollout_fn` is a zero-arg async callable that returns one trajectory. MaTTS
doesn't care how the rollout is produced — single LLM call, agent loop, tool
use — only that it returns turns.

If the contrast LLM returns truncated or invalid JSON, MaTTS logs a warning
and returns an empty memory list rather than raising. This matters in
production: contrast prompts can be long, and a `max_tokens` cap that's too
low silently truncates without throwing.

## Stores

Two stores ship in-tree:

* **`InMemoryStore`** — pure Python list + numpy for similarity. Ephemeral.
  Good for tests and short notebooks.
* **`SQLiteVecStore`** — backed by [`sqlite-vec`](https://github.com/asg017/sqlite-vec).
  A single SQLite file (`./reasoning-bank.db` by default) holds memory rows
  plus a vector virtual table. Cosine similarity is computed in SQL.

Both implement the same `MemoryStore` protocol — `add`, `update`, `delete`,
`get`, `list_all`, `search`. Implement those five methods and you have a
custom store.

## Embedders

* **`MiniLMEmbedder`** — `sentence-transformers/all-MiniLM-L6-v2`. 384 dims.
  Default. Loaded lazily on first use.
* **PolyRT embedder** — optional, via the `polyrt` extra. For hosted setups.

`Embedder` is a one-method protocol — `embed(texts: list[str]) -> list[list[float]]`.
Plug your own in and pass it to `ReasoningBank(embedder=...)`.

## The LLM callable

`ReasoningBank` doesn't import any provider SDK. You pass it:

```python
async def llm(prompt: str, *, system: str | None = None) -> str: ...
```

This is the single integration point. The bank uses it for:

* the judge call
* the induction call
* the merge-resolution call
* (optionally) the MaTTS contrast call, unless you pass `contrast_llm=`

You can wrap any provider — Anthropic, OpenAI, vLLM, a custom HTTP endpoint —
in five lines.

## Scope

Every memory carries a `scope` string. The bank's `scope` is the default; you
can override per-call. Use scope to partition memories across agents, task
families, or environments so retrieval doesn't pull irrelevant lessons across
boundaries.

## Export / import

`bank.export(path)` writes one JSON-encoded `MemoryItem` per line.
`bank.import_(path)` reads them back, re-embedding any items whose embedding
field is missing. The JSONL format is intentionally flat so you can grep,
diff, and version it.

## Async write-back

For latency-sensitive loops, `bank.schedule_writeback(trajectory, task)` runs
ingestion as a background `asyncio.Task` instead of blocking. Requires
`async_writeback=True` on the bank and a running event loop.

## What this library does *not* do

* No agent loop. You write the rollout.
* No tool execution. You handle tools.
* No evaluation harness. Plug into your own benchmark.
* No multi-tenant auth or networking. The bank is an in-process object.

These are deliberate omissions — the original paper bundles all of them, but
this library exists to be the *memory layer* of someone else's stack.
