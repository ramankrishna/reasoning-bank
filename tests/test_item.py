"""MemoryItem model basics."""

from __future__ import annotations

from datetime import datetime

import pytest

from reasoning_bank.item import MemoryItem


def _make(**kwargs) -> MemoryItem:
    defaults = dict(
        title="t",
        description="d",
        content="c",
        source="success",
        task_signature="task",
    )
    defaults.update(kwargs)
    return MemoryItem(**defaults)  # type: ignore[arg-type]


def test_defaults():
    item = _make()
    assert item.id.startswith("mem_")
    assert len(item.id) == len("mem_") + 10
    assert item.scope == "global"
    assert item.embedding is None
    assert item.use_count == 0
    assert item.confidence == 1.0
    assert item.related_ids == []
    assert isinstance(item.created_at, datetime)
    assert item.last_used_at is None


def test_signature_text_joins_three_fields():
    item = _make(title="alpha", description="beta", content="gamma")
    assert item.signature_text() == "alpha\nbeta\ngamma"


def test_distinct_ids():
    a = _make()
    b = _make()
    assert a.id != b.id


def test_source_must_be_one_of_literals():
    with pytest.raises(Exception):
        _make(source="bogus")


def test_round_trip_json():
    item = _make(embedding=[0.1, 0.2, 0.3])
    blob = item.model_dump_json()
    restored = MemoryItem.model_validate_json(blob)
    assert restored.id == item.id
    assert restored.embedding == [0.1, 0.2, 0.3]
