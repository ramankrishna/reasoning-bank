"""SQLite + sqlite-vec MemoryStore.

Stores MemoryItem rows in a regular ``memory_items`` table and the
corresponding embedding vector in a ``memory_vec`` virtual table created by
the sqlite-vec extension. Search uses cosine distance via
``vec_distance_cosine``.
"""

from __future__ import annotations

import json
import os
import sqlite3
import struct
from datetime import datetime
from pathlib import Path
from typing import Any

from reasoning_bank.item import MemoryItem

DEFAULT_DB_PATH = Path.home() / ".reasoning_bank" / "bank.db"
DEFAULT_DIM = 384


def _pack(vec: list[float]) -> bytes:
    return struct.pack(f"{len(vec)}f", *vec)


def _datetime_to_iso(value: datetime | None) -> str | None:
    return value.isoformat() if value is not None else None


def _iso_to_datetime(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value) if value else None


def _row_to_item(row: sqlite3.Row) -> MemoryItem:
    return MemoryItem(
        id=row["id"],
        title=row["title"],
        description=row["description"],
        content=row["content"],
        source=row["source"],
        task_signature=row["task_signature"],
        scope=row["scope"],
        embedding=json.loads(row["embedding"]) if row["embedding"] else None,
        created_at=_iso_to_datetime(row["created_at"]),  # type: ignore[arg-type]
        last_used_at=_iso_to_datetime(row["last_used_at"]),
        use_count=row["use_count"],
        confidence=row["confidence"],
        related_ids=json.loads(row["related_ids"]) if row["related_ids"] else [],
    )


class SQLiteVecStore:
    """SQLite-backed MemoryStore using the sqlite-vec extension."""

    def __init__(
        self,
        db_path: str | Path | None = None,
        dim: int = DEFAULT_DIM,
    ) -> None:
        import sqlite_vec

        if db_path is None:
            db_path = os.environ.get("REASONING_BANK_DB", str(DEFAULT_DB_PATH))
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.dim = dim

        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.enable_load_extension(True)
        sqlite_vec.load(self._conn)
        self._conn.enable_load_extension(False)
        self._init_schema()

    def _init_schema(self) -> None:
        cur = self._conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS memory_items (
                id              TEXT PRIMARY KEY,
                title           TEXT NOT NULL,
                description     TEXT NOT NULL,
                content         TEXT NOT NULL,
                source          TEXT NOT NULL,
                task_signature  TEXT NOT NULL,
                scope           TEXT NOT NULL,
                embedding       TEXT,
                created_at      TEXT NOT NULL,
                last_used_at    TEXT,
                use_count       INTEGER NOT NULL DEFAULT 0,
                confidence      REAL NOT NULL DEFAULT 1.0,
                related_ids     TEXT
            )
            """
        )
        cur.execute(
            f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS memory_vec USING vec0(
                embedding FLOAT[{self.dim}]
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS memory_vec_map (
                rowid  INTEGER PRIMARY KEY,
                mem_id TEXT NOT NULL UNIQUE
            )
            """
        )
        self._conn.commit()

    def add(self, item: MemoryItem) -> None:
        if item.embedding is None:
            raise ValueError(
                f"MemoryItem {item.id} has no embedding; embed before add()"
            )
        if len(item.embedding) != self.dim:
            raise ValueError(
                f"Embedding dim {len(item.embedding)} != store dim {self.dim}"
            )
        cur = self._conn.cursor()
        cur.execute(
            """
            INSERT INTO memory_items
              (id, title, description, content, source, task_signature, scope,
               embedding, created_at, last_used_at, use_count, confidence, related_ids)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                item.id,
                item.title,
                item.description,
                item.content,
                item.source,
                item.task_signature,
                item.scope,
                json.dumps(item.embedding),
                _datetime_to_iso(item.created_at),
                _datetime_to_iso(item.last_used_at),
                item.use_count,
                item.confidence,
                json.dumps(item.related_ids),
            ),
        )
        cur.execute(
            "INSERT INTO memory_vec(embedding) VALUES (?)",
            (_pack(item.embedding),),
        )
        rowid = cur.lastrowid
        cur.execute(
            "INSERT INTO memory_vec_map(rowid, mem_id) VALUES (?, ?)",
            (rowid, item.id),
        )
        self._conn.commit()

    def get(self, item_id: str) -> MemoryItem | None:
        cur = self._conn.cursor()
        cur.execute("SELECT * FROM memory_items WHERE id = ?", (item_id,))
        row = cur.fetchone()
        return _row_to_item(row) if row else None

    def update(self, item: MemoryItem) -> None:
        cur = self._conn.cursor()
        cur.execute("SELECT 1 FROM memory_items WHERE id = ?", (item.id,))
        if cur.fetchone() is None:
            raise KeyError(item.id)

        cur.execute(
            """
            UPDATE memory_items SET
              title = ?, description = ?, content = ?, source = ?,
              task_signature = ?, scope = ?, embedding = ?, last_used_at = ?,
              use_count = ?, confidence = ?, related_ids = ?
            WHERE id = ?
            """,
            (
                item.title,
                item.description,
                item.content,
                item.source,
                item.task_signature,
                item.scope,
                json.dumps(item.embedding) if item.embedding else None,
                _datetime_to_iso(item.last_used_at),
                item.use_count,
                item.confidence,
                json.dumps(item.related_ids),
                item.id,
            ),
        )
        if item.embedding is not None:
            cur.execute(
                "SELECT rowid FROM memory_vec_map WHERE mem_id = ?", (item.id,)
            )
            map_row = cur.fetchone()
            if map_row is not None:
                cur.execute(
                    "UPDATE memory_vec SET embedding = ? WHERE rowid = ?",
                    (_pack(item.embedding), map_row["rowid"]),
                )
        self._conn.commit()

    def delete(self, item_id: str) -> None:
        cur = self._conn.cursor()
        cur.execute("SELECT rowid FROM memory_vec_map WHERE mem_id = ?", (item_id,))
        row = cur.fetchone()
        if row is not None:
            cur.execute("DELETE FROM memory_vec WHERE rowid = ?", (row["rowid"],))
            cur.execute("DELETE FROM memory_vec_map WHERE rowid = ?", (row["rowid"],))
        cur.execute("DELETE FROM memory_items WHERE id = ?", (item_id,))
        self._conn.commit()

    def search(
        self,
        query_embedding: list[float],
        k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[tuple[MemoryItem, float]]:
        if len(query_embedding) != self.dim:
            raise ValueError(
                f"Query dim {len(query_embedding)} != store dim {self.dim}"
            )
        cur = self._conn.cursor()
        pool = k * 4 if filters else k
        cur.execute(
            """
            SELECT v.rowid AS rowid,
                   vec_distance_cosine(v.embedding, ?) AS distance
            FROM memory_vec v
            ORDER BY distance ASC
            LIMIT ?
            """,
            (_pack(query_embedding), pool),
        )
        candidates = cur.fetchall()
        if not candidates:
            return []

        results: list[tuple[MemoryItem, float]] = []
        for cand in candidates:
            cur.execute(
                "SELECT mem_id FROM memory_vec_map WHERE rowid = ?",
                (cand["rowid"],),
            )
            map_row = cur.fetchone()
            if map_row is None:
                continue
            item = self.get(map_row["mem_id"])
            if item is None:
                continue
            if filters and not _matches(item, filters):
                continue
            similarity = 1.0 - float(cand["distance"])
            results.append((item, similarity))
            if len(results) >= k:
                break
        return results

    def list_all(self, scope: str | None = None) -> list[MemoryItem]:
        cur = self._conn.cursor()
        if scope is None:
            cur.execute("SELECT * FROM memory_items ORDER BY created_at ASC")
        else:
            cur.execute(
                "SELECT * FROM memory_items WHERE scope = ? ORDER BY created_at ASC",
                (scope,),
            )
        return [_row_to_item(row) for row in cur.fetchall()]

    def close(self) -> None:
        self._conn.close()


def _matches(item: MemoryItem, filters: dict[str, Any]) -> bool:
    for key, value in filters.items():
        if getattr(item, key, None) != value:
            return False
    return True
