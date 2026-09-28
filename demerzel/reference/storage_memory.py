"""In-memory StorageBackend and EvidenceStore.

Used for unit tests and for experiments where persistence is not under test.
It implements the same contract as the SQLite reference, including the
Invariant-2 property that ``update_pointer`` touches nothing but the pointer.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

from ..interfaces.storage import MemoryRecord


class InMemoryStorage:
    """Dict-backed :class:`~demerzel.interfaces.storage.StorageBackend`."""

    def __init__(self) -> None:
        self._records: Dict[str, MemoryRecord] = {}
        self._order: List[str] = []
        #: Counts update_pointer calls, so tests can assert on the mechanism
        #: firing without inspecting private state.
        self.pointer_updates = 0

    def get(self, memory_id: str) -> Optional[MemoryRecord]:
        return self._records.get(memory_id)

    def put(self, memory: MemoryRecord) -> None:
        if memory.id not in self._records:
            self._order.append(memory.id)
        self._records[memory.id] = memory

    def update_pointer(self, memory_id: str, pointer: float) -> None:
        record = self._records.get(memory_id)
        if record is None:
            return
        # Replace only the pointer field; content/source_ptr/created_at are
        # carried over untouched (Invariant 2).
        self._records[memory_id] = record.with_pointer(pointer)
        self.pointer_updates += 1

    def list_by_session(self, session_id: str) -> List[MemoryRecord]:
        return [self._records[i] for i in self._order if self._records[i].session_id == session_id]

    def list_all(self) -> List[MemoryRecord]:
        return [self._records[i] for i in self._order]

    def delete(self, memory_id: str) -> bool:
        """Explicit user deletion — distinct from the forgetting mechanism.

        Nothing in the forgetting path calls this (Invariant 6).
        """
        if memory_id not in self._records:
            return False
        del self._records[memory_id]
        self._order.remove(memory_id)
        return True


class InMemoryEvidenceStore:
    """List-backed append-only :class:`~demerzel.interfaces.storage.EvidenceStore`."""

    def __init__(self) -> None:
        self._turns: List[Dict[str, Any]] = []
        self._by_id: Dict[str, Dict[str, Any]] = {}

    def append(self, turn: Dict[str, Any]) -> str:
        turn_id = str(turn["turn_id"])
        # Append-only: a repeated turn_id is an error, never an overwrite.
        if turn_id in self._by_id:
            raise ValueError(f"duplicate turn_id: {turn_id}")
        self._turns.append(dict(turn))
        self._by_id[turn_id] = self._turns[-1]
        return turn_id

    def get_turn(self, turn_id: str) -> Optional[Dict[str, Any]]:
        return self._by_id.get(turn_id)

    def iter_turns(self, session_id: Optional[str] = None) -> Iterable[Dict[str, Any]]:
        for turn in self._turns:
            if session_id is None or turn.get("session_id") == session_id:
                yield turn

    def count(self) -> int:
        return len(self._turns)

    def content_hash(self) -> str:
        """Digest of every stored turn, for the P0-1 persistence assertion."""
        import hashlib
        import json

        blob = json.dumps(self._turns, sort_keys=True, default=str)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()


__all__ = ["InMemoryEvidenceStore", "InMemoryStorage"]
