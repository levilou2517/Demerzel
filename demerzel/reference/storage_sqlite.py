"""SQLite StorageBackend and EvidenceStore reference implementation.

This is the *reference* SQLite layer (V1.md §3.1). Core mechanisms never
import it — they see only :class:`~demerzel.interfaces.storage.StorageBackend`
and :class:`~demerzel.interfaces.storage.EvidenceStore`.

Crash safety (P0-7) is provided by:

* ``PRAGMA synchronous=FULL`` (config-settable) so a committed transaction is
  durable before ``commit`` returns;
* a single connection with ``WAL`` journal and (by default) ``BEGIN IMMEDIATE``
  transactions so readers see only complete writes.

``update_pointer`` writes exactly one column via a parameterised statement, so
content/source_ptr/created_at cannot be altered here (Invariant 2).
"""

from __future__ import annotations

import os
import sqlite3
import threading
from contextlib import contextmanager
from typing import Any, Dict, Iterable, Iterator, List, Optional

from ..interfaces.storage import MemoryRecord


class _SqliteBackendBase:
    """Shared connection handling for the single-file SQLite store."""

    def __init__(self, path: str = "demerzel.db", synchronous: str = "FULL"):
        self._path = path
        if path != ":memory:":
            parent = os.path.dirname(os.path.abspath(path))
            if parent:
                os.makedirs(parent, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute(f"PRAGMA journal_mode=WAL")
        self._conn.execute(f"PRAGMA synchronous={synchronous.upper()}")
        self._init_schema()

    def _init_schema(self) -> None:
        raise NotImplementedError

    @contextmanager
    def _tx(self, immediate: bool = True) -> Iterator[sqlite3.Connection]:
        with self._lock:
            if immediate:
                self._conn.execute("BEGIN IMMEDIATE")
            else:
                self._conn.execute("BEGIN")
            try:
                yield self._conn
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise

    def close(self) -> None:
        self._conn.close()


class SqliteStorage(_SqliteBackendBase):
    """SQLite :class:`~demerzel.interfaces.storage.StorageBackend`."""

    def _init_schema(self) -> None:
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                episode_id TEXT NOT NULL,
                memory_type TEXT NOT NULL,
                content TEXT NOT NULL,
                gist TEXT NOT NULL,
                created_at REAL NOT NULL,
                source_ptr TEXT NOT NULL,
                pointer_strength REAL NOT NULL DEFAULT 1.0,
                retired INTEGER NOT NULL DEFAULT 0,
                metadata TEXT NOT NULL DEFAULT '{}'
            )
            """
        )
        self._conn.commit()

    def get(self, memory_id: str) -> Optional[MemoryRecord]:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM memories WHERE id = ?", (memory_id,)
            ).fetchone()
        return self._row_to_record(row) if row is not None else None

    def put(self, memory: MemoryRecord) -> None:
        with self._tx():
            self._conn.execute(
                """
                INSERT INTO memories
                  (id, session_id, episode_id, memory_type, content, gist,
                   created_at, source_ptr, pointer_strength, retired, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                  content=excluded.content, gist=excluded.gist,
                  source_ptr=excluded.source_ptr
                """,
                (
                    memory.id,
                    memory.session_id,
                    memory.episode_id,
                    memory.memory_type,
                    memory.content,
                    memory.gist,
                    memory.created_at,
                    memory.source_ptr,
                    memory.pointer_strength,
                    memory.retired,
                    _dumps(memory.metadata),
                ),
            )

    def update_pointer(self, memory_id: str, pointer: float) -> None:
        """Update exactly the pointer column. Unknown ids are a no-op."""
        with self._tx():
            cur = self._conn.execute(
                "UPDATE memories SET pointer_strength = ? WHERE id = ?",
                (float(pointer), memory_id),
            )
            # We keep the transaction even when the row is absent; a no-op on
            # an unknown id is the documented behaviour.
            _ = cur

    def list_by_session(self, session_id: str) -> List[MemoryRecord]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM memories WHERE session_id = ? ORDER BY created_at, rowid",
                (session_id,),
            ).fetchall()
        return [self._row_to_record(r) for r in rows]

    def list_all(self) -> List[MemoryRecord]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM memories ORDER BY created_at, rowid"
            ).fetchall()
        return [self._row_to_record(r) for r in rows]

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> MemoryRecord:
        return MemoryRecord(
            id=row["id"],
            session_id=row["session_id"],
            episode_id=row["episode_id"],
            memory_type=row["memory_type"],
            content=row["content"],
            gist=row["gist"],
            created_at=float(row["created_at"]),
            source_ptr=row["source_ptr"],
            pointer_strength=float(row["pointer_strength"]),
            retired=int(row["retired"]),
            metadata=_loads(row["metadata"]),
        )


class SqliteEvidenceStore(_SqliteBackendBase):
    """Append-only SQLite Foundation store (V1.md §5.1).

    There is deliberately no DELETE path in the public API used by the
    forgetting mechanism. The only removal is a private ``_drop_table`` used
    by tests to simulate a clean database — never reachable from ingestion.
    """

    def _init_schema(self) -> None:
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS evidence (
                turn_id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                timestamp REAL NOT NULL,
                speaker TEXT NOT NULL,
                text TEXT NOT NULL,
                metadata TEXT NOT NULL DEFAULT '{}'
            )
            """
        )
        self._conn.commit()

    def append(self, turn: Dict[str, Any]) -> str:
        turn_id = str(turn["turn_id"])
        with self._tx():
            cur = self._conn.execute(
                "SELECT 1 FROM evidence WHERE turn_id = ?", (turn_id,)
            )
            if cur.fetchone() is not None:
                raise ValueError(f"duplicate turn_id: {turn_id}")
            self._conn.execute(
                """
                INSERT INTO evidence (turn_id, session_id, timestamp, speaker, text, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    turn_id,
                    str(turn["session_id"]),
                    float(turn["timestamp"]),
                    str(turn["speaker"]),
                    str(turn["text"]),
                    _dumps(turn.get("metadata", {})),
                ),
            )
        return turn_id

    def get_turn(self, turn_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM evidence WHERE turn_id = ?", (turn_id,)
            ).fetchone()
        if row is None:
            return None
        return {
            "turn_id": row["turn_id"],
            "session_id": row["session_id"],
            "timestamp": float(row["timestamp"]),
            "speaker": row["speaker"],
            "text": row["text"],
            "metadata": _loads(row["metadata"] or "{}"),
        }

    def iter_turns(self, session_id: Optional[str] = None) -> Iterable[Dict[str, Any]]:
        sql = "SELECT * FROM evidence"
        args: tuple = ()
        if session_id is not None:
            sql += " WHERE session_id = ?"
            args = (session_id,)
        sql += " ORDER BY timestamp, rowid"
        with self._lock:
            rows = self._conn.execute(sql, args).fetchall()
        for row in rows:
            yield {
                "turn_id": row["turn_id"],
                "session_id": row["session_id"],
                "timestamp": float(row["timestamp"]),
                "speaker": row["speaker"],
                "text": row["text"],
                "metadata": _loads(row["metadata"] or "{}"),
            }

    def count(self) -> int:
        with self._lock:
            row = self._conn.execute("SELECT COUNT(*) AS n FROM evidence").fetchone()
        return int(row["n"])


def _dumps(obj: Any) -> str:
    import json

    return json.dumps(obj, ensure_ascii=False, default=str)


def _loads(text: str) -> Any:
    import json

    try:
        return json.loads(text)
    except Exception:
        return {}


__all__ = ["SqliteEvidenceStore", "SqliteStorage"]