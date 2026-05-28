"""reasoning_bank — framework-agnostic ReasoningBank (Ouyang et al., ICLR 2026).

Public surface:

* ``ReasoningBank`` — high-level memory bank with retrieval + ingestion
* ``MemoryItem`` — single stored unit
* ``Turn`` / ``Trajectory`` — neutral message types accepted by the bank
* ``Embedder`` protocol + ``get_embedder`` / ``register_embedder`` registry
* ``MemoryStore`` protocol + ``InMemoryStore`` / ``SQLiteVecStore`` backends
* ``matts_run`` — Memory-Aware Test-Time Scaling driver
"""

from __future__ import annotations

from reasoning_bank.bank import ReasoningBank
from reasoning_bank.embedders import (
    Embedder,
    get_embedder,
    register as register_embedder,
)
from reasoning_bank.item import MemoryItem
from reasoning_bank.matts import matts_run
from reasoning_bank.stores import InMemoryStore, MemoryStore
from reasoning_bank.types import Trajectory, Turn

__all__ = [
    "ReasoningBank",
    "MemoryItem",
    "Turn",
    "Trajectory",
    "Embedder",
    "get_embedder",
    "register_embedder",
    "MemoryStore",
    "InMemoryStore",
    "matts_run",
]

try:
    from reasoning_bank.stores import SQLiteVecStore  # noqa: F401

    if SQLiteVecStore is not None:
        __all__.append("SQLiteVecStore")
except ImportError:
    pass
