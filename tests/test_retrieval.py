"""retrieve_relevant against an InMemoryStore + fake embedder."""

from __future__ import annotations

from dataclasses import dataclass, field

from reasoning_bank.item import MemoryItem
from reasoning_bank.retrieval import retrieve_relevant
from reasoning_bank.stores.inmemory import InMemoryStore


class _FakeEmbedder:
    """Embed first letter → one-hot vector across [a, b, c, d]."""

    dim = 4
    _CHARS = "abcd"

    def embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for text in texts:
            vec = [0.0] * 4
            ch = (text.strip()[:1] or "a").lower()
            if ch in self._CHARS:
                vec[self._CHARS.index(ch)] = 1.0
            out.append(vec)
        return out


@dataclass
class _FakeBank:
    embedder: _FakeEmbedder = field(default_factory=_FakeEmbedder)
    store: InMemoryStore = field(default_factory=InMemoryStore)
    scope: str = "global"
    merge_thresholds: tuple[float, float] = (0.75, 0.92)
    llm: object | None = None


def _seed(bank: _FakeBank, items: list[tuple[str, list[float], float, str]]):
    for title, emb, conf, scope in items:
        bank.store.add(
            MemoryItem(
                title=title,
                description="d",
                content="c",
                source="success",
                task_signature="task",
                embedding=emb,
                confidence=conf,
                scope=scope,
            )
        )


async def test_retrieve_ranks_by_similarity_times_confidence():
    bank = _FakeBank()
    _seed(
        bank,
        [
            ("alpha", [1.0, 0.0, 0.0, 0.0], 1.0, "global"),
            ("beta", [0.0, 1.0, 0.0, 0.0], 1.0, "global"),
        ],
    )
    out = await retrieve_relevant(bank, task="a question about alpha", k=2)
    assert [it.title for it in out] == ["alpha", "beta"]


async def test_retrieve_drops_below_min_confidence():
    bank = _FakeBank()
    _seed(
        bank,
        [
            ("alpha", [1.0, 0.0, 0.0, 0.0], 0.1, "global"),
            ("beta", [1.0, 0.0, 0.0, 0.0], 1.0, "global"),
        ],
    )
    out = await retrieve_relevant(bank, task="alpha", k=5, min_confidence=0.3)
    assert [it.title for it in out] == ["beta"]


async def test_retrieve_filters_by_scope():
    bank = _FakeBank()
    _seed(
        bank,
        [
            ("g", [1.0, 0.0, 0.0, 0.0], 1.0, "global"),
            ("p", [1.0, 0.0, 0.0, 0.0], 1.0, "private"),
        ],
    )
    out = await retrieve_relevant(bank, task="alpha", k=5, scope="private")
    assert [it.title for it in out] == ["p"]


async def test_retrieve_updates_use_count_and_last_used_at():
    bank = _FakeBank()
    _seed(bank, [("a", [1.0, 0.0, 0.0, 0.0], 1.0, "global")])
    out = await retrieve_relevant(bank, task="alpha", k=1)
    fetched = bank.store.get(out[0].id)
    assert fetched.use_count == 1
    assert fetched.last_used_at is not None


async def test_retrieve_zero_k_returns_empty():
    bank = _FakeBank()
    _seed(bank, [("a", [1.0, 0.0, 0.0, 0.0], 1.0, "global")])
    assert await retrieve_relevant(bank, task="alpha", k=0) == []
