"""InMemoryStore CRUD + cosine search."""

from __future__ import annotations

import pytest

from reasoning_bank.item import MemoryItem
from reasoning_bank.stores.inmemory import InMemoryStore


def _item(title: str, vec: list[float], scope: str = "global") -> MemoryItem:
    return MemoryItem(
        title=title,
        description="d",
        content="c",
        source="success",
        task_signature="task",
        scope=scope,
        embedding=vec,
    )


def test_add_get_update_delete():
    store = InMemoryStore()
    a = _item("a", [1.0, 0.0])
    store.add(a)
    assert store.get(a.id).title == "a"

    a.title = "a2"
    store.update(a)
    assert store.get(a.id).title == "a2"

    store.delete(a.id)
    assert store.get(a.id) is None


def test_add_duplicate_raises():
    store = InMemoryStore()
    a = _item("a", [1.0, 0.0])
    store.add(a)
    with pytest.raises(ValueError):
        store.add(a)


def test_add_without_embedding_raises():
    store = InMemoryStore()
    item = MemoryItem(
        title="t",
        description="d",
        content="c",
        source="success",
        task_signature="task",
    )
    with pytest.raises(ValueError):
        store.add(item)


def test_update_missing_raises():
    store = InMemoryStore()
    a = _item("a", [1.0, 0.0])
    with pytest.raises(KeyError):
        store.update(a)


def test_delete_missing_is_silent():
    store = InMemoryStore()
    store.delete("nope")


def test_search_ranks_by_cosine():
    store = InMemoryStore()
    near = _item("near", [1.0, 0.0])
    far = _item("far", [0.0, 1.0])
    store.add(near)
    store.add(far)

    results = store.search([1.0, 0.0], k=2)
    assert [it.title for it, _ in results] == ["near", "far"]
    assert results[0][1] > results[1][1]


def test_search_filter_by_scope():
    store = InMemoryStore()
    a = _item("a", [1.0, 0.0], scope="global")
    b = _item("b", [1.0, 0.0], scope="private")
    store.add(a)
    store.add(b)
    results = store.search([1.0, 0.0], k=5, filters={"scope": "private"})
    assert [it.title for it, _ in results] == ["b"]


def test_list_all_and_scope_filter():
    store = InMemoryStore()
    a = _item("a", [1.0, 0.0], scope="global")
    b = _item("b", [1.0, 0.0], scope="private")
    store.add(a)
    store.add(b)
    assert {it.title for it in store.list_all()} == {"a", "b"}
    assert {it.title for it in store.list_all(scope="private")} == {"b"}


def test_returned_items_are_copies():
    store = InMemoryStore()
    a = _item("a", [1.0, 0.0])
    store.add(a)
    fetched = store.get(a.id)
    fetched.title = "mutated"
    assert store.get(a.id).title == "a"
