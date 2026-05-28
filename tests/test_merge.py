"""integrate_candidate: insert / link / merge / replace branches."""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from reasoning_bank.item import MemoryItem
from reasoning_bank.merge import integrate_candidate
from reasoning_bank.stores.inmemory import InMemoryStore


class _FakeEmbedder:
    dim = 4

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.0, 0.0, 0.0, 0.0] for _ in texts]


@dataclass
class _FakeBank:
    embedder: _FakeEmbedder = field(default_factory=_FakeEmbedder)
    store: InMemoryStore = field(default_factory=InMemoryStore)
    scope: str = "global"
    merge_thresholds: tuple[float, float] = (0.75, 0.92)
    llm: object | None = None


def _candidate(title: str, emb: list[float]) -> MemoryItem:
    return MemoryItem(
        title=title,
        description="d",
        content="c",
        source="success",
        task_signature="task",
        embedding=emb,
        scope="global",
    )


async def test_insert_when_store_empty():
    bank = _FakeBank()
    cand = _candidate("a", [1.0, 0.0, 0.0, 0.0])
    action, stored = await integrate_candidate(bank, cand)
    assert action == "insert"
    assert bank.store.get(stored.id) is not None


async def test_link_when_similar_but_below_merge_threshold():
    bank = _FakeBank()
    existing = _candidate("existing", [1.0, 0.0, 0.0, 0.0])
    bank.store.add(existing)

    # cosine ~ 0.78 — between link (0.75) and merge (0.92) thresholds
    cand = _candidate("similar-ish", [0.8, 0.6, 0.0, 0.0])
    action, stored = await integrate_candidate(bank, cand)
    assert action == "link"
    assert existing.id in stored.related_ids
    refreshed_existing = bank.store.get(existing.id)
    assert stored.id in refreshed_existing.related_ids


async def test_insert_when_below_link_threshold():
    bank = _FakeBank()
    existing = _candidate("existing", [1.0, 0.0, 0.0, 0.0])
    bank.store.add(existing)

    cand = _candidate("unrelated", [0.0, 1.0, 0.0, 0.0])
    action, _ = await integrate_candidate(bank, cand)
    assert action == "insert"
    assert len(bank.store.list_all()) == 2


async def test_replace_when_above_merge_threshold_no_llm():
    bank = _FakeBank()
    existing = _candidate("existing", [1.0, 0.0, 0.0, 0.0])
    bank.store.add(existing)

    cand = _candidate("new-better", [1.0, 0.0, 0.0, 0.0])
    action, stored = await integrate_candidate(bank, cand)
    assert action == "replace"
    assert stored.title == "new-better"
    assert len(bank.store.list_all()) == 1


async def test_merge_via_llm_when_above_threshold():
    payload = json.dumps(
        {
            "action": "merge",
            "merged_title": "merged-title",
            "merged_description": "merged-d",
            "merged_content": "merged-c",
        }
    )

    async def llm(prompt: str, *, system: str | None = None) -> str:
        return payload

    bank = _FakeBank(llm=llm)
    existing = _candidate("existing", [1.0, 0.0, 0.0, 0.0])
    bank.store.add(existing)

    cand = _candidate("new", [1.0, 0.0, 0.0, 0.0])
    action, stored = await integrate_candidate(bank, cand)
    assert action == "merge"
    assert stored.title == "merged-title"
    assert len(bank.store.list_all()) == 1


async def test_llm_keep_both_falls_through_to_link():
    payload = json.dumps(
        {
            "action": "keep_both",
            "merged_title": "",
            "merged_description": "",
            "merged_content": "",
        }
    )

    async def llm(prompt: str, *, system: str | None = None) -> str:
        return payload

    bank = _FakeBank(llm=llm)
    existing = _candidate("existing", [1.0, 0.0, 0.0, 0.0])
    bank.store.add(existing)

    cand = _candidate("new", [1.0, 0.0, 0.0, 0.0])
    action, stored = await integrate_candidate(bank, cand)
    assert action == "link"
    assert len(bank.store.list_all()) == 2
    assert existing.id in stored.related_ids


async def test_candidate_without_embedding_is_embedded_on_the_fly():
    bank = _FakeBank()
    cand = MemoryItem(
        title="a",
        description="d",
        content="c",
        source="success",
        task_signature="task",
        scope="global",
    )
    assert cand.embedding is None
    action, stored = await integrate_candidate(bank, cand)
    assert action == "insert"
    assert stored.embedding is not None
