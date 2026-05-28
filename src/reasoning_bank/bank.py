"""ReasoningBank — public API for experience-augmented agent memory.

A bank owns three collaborators:

* a ``MemoryStore`` — persistence and vector search
* an ``Embedder`` — text → vector
* an async LLM callable — used for judging trajectories, distilling new
  memories, and merging near-duplicates

The two main flows are ``retrieve(task)`` — pull the top-k useful memories
for a new task — and ``ingest_trajectory(trajectory, task)`` — extract
memories from a completed run and integrate them into the store.

The bank is framework-agnostic: it accepts trajectories as ``list[Turn]``,
``list[dict]``, or any list of objects with ``.role``/``.content`` attributes.
"""

from __future__ import annotations

import asyncio
import json
import logging
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Literal

from reasoning_bank.item import MemoryItem
from reasoning_bank.types import _coerce_trajectory

if TYPE_CHECKING:
    from reasoning_bank.embedders import Embedder
    from reasoning_bank.stores import MemoryStore

logger = logging.getLogger(__name__)

LLMCallable = Callable[..., Awaitable[str]]


def _default_store() -> "MemoryStore":
    try:
        from reasoning_bank.stores.sqlite_vec import SQLiteVecStore

        return SQLiteVecStore()
    except Exception as exc:  # pragma: no cover - depends on optional extras
        warnings.warn(
            f"SQLiteVecStore unavailable ({type(exc).__name__}: {exc}); "
            "falling back to InMemoryStore (NOT persistent).",
            stacklevel=3,
        )
        from reasoning_bank.stores.inmemory import InMemoryStore

        return InMemoryStore()


def _default_embedder() -> "Embedder":
    try:
        from reasoning_bank.embedders.minilm import MiniLMEmbedder

        return MiniLMEmbedder()
    except ImportError as exc:  # pragma: no cover - depends on optional extras
        raise RuntimeError(
            "No default embedder available — install sentence-transformers "
            "or pass an explicit embedder to ReasoningBank()."
        ) from exc


class ReasoningBank:
    """High-level API for the agent's experience memory."""

    def __init__(
        self,
        llm: LLMCallable,
        store: "MemoryStore | None" = None,
        embedder: "Embedder | None" = None,
        scope: str = "global",
        merge_thresholds: tuple[float, float] = (0.75, 0.92),
        async_writeback: bool = False,
    ) -> None:
        if not callable(llm):
            raise TypeError(
                "ReasoningBank: llm must be an async callable of the form "
                "`async def llm(prompt, *, system=None) -> str`."
            )
        self.llm = llm
        self.store = store if store is not None else _default_store()
        self.embedder = embedder if embedder is not None else _default_embedder()
        self.scope = scope
        self.merge_thresholds = merge_thresholds
        self.async_writeback = async_writeback

    # ------------------------------------------------------------------
    # retrieval
    # ------------------------------------------------------------------
    async def retrieve(
        self,
        task: str,
        k: int = 5,
        *,
        scope: str | None = None,
        min_confidence: float = 0.3,
    ) -> list[MemoryItem]:
        from reasoning_bank.retrieval import retrieve_relevant

        return await retrieve_relevant(
            self,
            task=task,
            k=k,
            scope=scope if scope is not None else self.scope,
            min_confidence=min_confidence,
        )

    # ------------------------------------------------------------------
    # ingestion
    # ------------------------------------------------------------------
    async def ingest_trajectory(
        self,
        trajectory: Any,
        task: str,
        outcome: Literal["success", "failure"] | None = None,
    ) -> list[MemoryItem]:
        """Judge, distill, embed, and integrate memories from one trajectory.

        Returns the list of MemoryItems that were inserted/merged/linked.
        ``trajectory`` is coerced via ``_coerce_trajectory`` — accepts Turn
        list, dict list, or any list of objects exposing ``.role``/``.content``.
        """
        from reasoning_bank.induction import distill_memories
        from reasoning_bank.judge import judge_trajectory
        from reasoning_bank.merge import integrate_candidate

        turns = _coerce_trajectory(trajectory)
        if not turns:
            return []

        if outcome is None:
            resolved_outcome, _rationale = await judge_trajectory(
                turns, task, self.llm
            )
        else:
            resolved_outcome = outcome

        candidates = await distill_memories(
            trajectory=turns,
            task=task,
            outcome=resolved_outcome,
            llm=self.llm,
            scope=self.scope,
        )

        integrated: list[MemoryItem] = []
        for candidate in candidates:
            if candidate.embedding is None:
                candidate.embedding = self.embedder.embed(
                    [candidate.signature_text()]
                )[0]
            _action, stored = await integrate_candidate(self, candidate)
            integrated.append(stored)
        return integrated

    # ------------------------------------------------------------------
    # direct CRUD-ish helpers
    # ------------------------------------------------------------------
    async def add_manual(
        self,
        title: str,
        description: str,
        content: str,
        *,
        scope: str | None = None,
    ) -> MemoryItem:
        item = MemoryItem(
            title=title,
            description=description,
            content=content,
            source="manual",
            task_signature="",
            scope=scope if scope is not None else self.scope,
        )
        item.embedding = self.embedder.embed([item.signature_text()])[0]
        self.store.add(item)
        return item

    async def list(self, scope: str | None = None) -> list[MemoryItem]:
        return self.store.list_all(scope=scope)

    async def delete(self, id: str) -> None:
        self.store.delete(id)

    # ------------------------------------------------------------------
    # export / import
    # ------------------------------------------------------------------
    async def export(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for item in self.store.list_all():
                f.write(item.model_dump_json() + "\n")

    async def import_(self, path: str | Path) -> int:
        path = Path(path)
        count = 0
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                item = MemoryItem.model_validate(data)
                if item.embedding is None:
                    item.embedding = self.embedder.embed(
                        [item.signature_text()]
                    )[0]
                try:
                    self.store.add(item)
                    count += 1
                except ValueError:
                    self.store.update(item)
                    count += 1
        return count

    # ------------------------------------------------------------------
    # formatting helpers
    # ------------------------------------------------------------------
    def format_as_system_block(self, memories: list[MemoryItem]) -> str:
        """Render a list of memories as a system-prompt-friendly text block.

        Useful for injecting retrieved memories into a chat prompt.
        """
        if not memories:
            return ""
        lines: list[str] = [
            "# Reasoning bank — lessons from prior trajectories",
            "",
        ]
        for i, mem in enumerate(memories, start=1):
            lines.append(f"## {i}. {mem.title}")
            if mem.description:
                lines.append(f"_{mem.description}_")
            lines.append(mem.content)
            lines.append("")
        return "\n".join(lines).rstrip() + "\n"

    # ------------------------------------------------------------------
    # writeback helper
    # ------------------------------------------------------------------
    def schedule_writeback(
        self,
        trajectory: Any,
        task: str,
        outcome: Literal["success", "failure"] | None = None,
    ) -> "asyncio.Task[list[MemoryItem]] | list[MemoryItem]":
        """Kick off ``ingest_trajectory`` according to ``async_writeback``.

        If ``async_writeback`` is True and an event loop is running, schedule
        it as a background task and return the Task handle. Otherwise run
        synchronously via ``asyncio.run`` and return the resulting items.
        """
        coro = self.ingest_trajectory(trajectory, task, outcome=outcome)
        if self.async_writeback:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None
            if loop is not None:
                return loop.create_task(coro)
            return asyncio.run(coro)
        return asyncio.run(coro)
