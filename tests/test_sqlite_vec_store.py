"""SQLiteVecStore CRUD + vector search."""

from __future__ import annotations

import pytest

from reasoning_bank.item import MemoryItem
from reasoning_bank.stores.sqlite_vec import SQLiteVecStore


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


@pytest.fixture
def store(tmp_path):
    db_path = tmp_path / "bank.db"
    s = SQLiteVecStore(db_path=db_path, dim=4)
    yield s
    s.close()


def test_add_get(store):
    item = _item("a", [1.0, 0.0, 0.0, 0.0])
    store.add(item)
    fetched = store.get(item.id)
    assert fetched is not None
    assert fetched.title == "a"
    assert fetched.embedding == [1.0, 0.0, 0.0, 0.0]


def test_update_changes_fields_and_embedding(store):
    item = _item("a", [1.0, 0.0, 0.0, 0.0])
    store.add(item)
    item.title = "renamed"
    item.embedding = [0.0, 1.0, 0.0, 0.0]
    store.update(item)
    fetched = store.get(item.id)
    assert fetched.title == "renamed"
    assert fetched.embedding == [0.0, 1.0, 0.0, 0.0]


def test_update_missing_raises(store):
    item = _item("a", [1.0, 0.0, 0.0, 0.0])
    with pytest.raises(KeyError):
        store.update(item)


def test_delete(store):
    item = _item("a", [1.0, 0.0, 0.0, 0.0])
    store.add(item)
    store.delete(item.id)
    assert store.get(item.id) is None


def test_search_returns_nearest_first(store):
    near = _item("near", [1.0, 0.0, 0.0, 0.0])
    far = _item("far", [0.0, 1.0, 0.0, 0.0])
    store.add(near)
    store.add(far)

    results = store.search([1.0, 0.0, 0.0, 0.0], k=2)
    assert len(results) == 2
    assert results[0][0].title == "near"
    assert results[0][1] > results[1][1]


def test_search_dim_mismatch_raises(store):
    with pytest.raises(ValueError):
        store.search([1.0, 0.0], k=1)


def test_add_dim_mismatch_raises(store):
    item = _item("bad", [1.0, 0.0])
    with pytest.raises(ValueError):
        store.add(item)


def test_search_filter_by_scope(store):
    glob = _item("glob", [1.0, 0.0, 0.0, 0.0], scope="global")
    priv = _item("priv", [1.0, 0.0, 0.0, 0.0], scope="private")
    store.add(glob)
    store.add(priv)

    results = store.search([1.0, 0.0, 0.0, 0.0], k=5, filters={"scope": "private"})
    assert {it.title for it, _ in results} == {"priv"}


def test_list_all_scope_filter(store):
    a = _item("a", [1.0, 0.0, 0.0, 0.0], scope="global")
    b = _item("b", [1.0, 0.0, 0.0, 0.0], scope="private")
    store.add(a)
    store.add(b)
    assert {it.title for it in store.list_all()} == {"a", "b"}
    assert {it.title for it in store.list_all(scope="private")} == {"b"}


def test_persistence_round_trip(tmp_path):
    db_path = tmp_path / "persist.db"
    s1 = SQLiteVecStore(db_path=db_path, dim=4)
    item = _item("persistent", [0.1, 0.2, 0.3, 0.4])
    s1.add(item)
    s1.close()

    s2 = SQLiteVecStore(db_path=db_path, dim=4)
    fetched = s2.get(item.id)
    assert fetched is not None
    assert fetched.title == "persistent"
    s2.close()
