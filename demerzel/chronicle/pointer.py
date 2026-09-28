"""L0 pointer over Chronicle records (V1.md §4.1, Phase 2).

The pointer layer is the *only* component allowed to change
``pointer_strength`` on a Chronicle record, and it does so through exactly the
two seams the invariants name:

* :class:`~demerzel.interfaces.storage.StorageBackend.update_pointer` — which
  by contract cannot touch content/source_ptr/created_at (Invariant 2); and
* :class:`~demerzel.interfaces.pointer.PointerUpdater` — the MPE path.

The retire operation is separate and explicit: it flips the ``retired`` flag,
which removes a record from active retrieval while its evidence remains
(``retired != deleted``).

There is deliberately **no delete method** on this class (Invariant 6).
"""

from __future__ import annotations

from typing import List, Optional

from ..interfaces.storage import MemoryRecord, StorageBackend


class PointerLayer:
    """Read/write access to pointer state and the retired flag."""

    def __init__(self, storage: StorageBackend):
        self._storage = storage

    def strength(self, memory_id: str) -> Optional[float]:
        """Current ``pointer_strength`` of one record."""
        record = self._storage.get(memory_id)
        return None if record is None else float(record.pointer_strength)

    def set_strength(self, memory_id: str, value: float) -> None:
        """Force a pointer value (used by synthetic ablations, not by MPE)."""
        self._storage.update_pointer(memory_id, float(value))

    def retire(self, memory_id: str) -> bool:
        """Stop actively retrieving a record **without** deleting it.

        The content, source pointer and creation time are preserved; only the
        ``retired`` flag changes, which is what the retrieval layer filters on.
        """
        record = self._storage.get(memory_id)
        if record is None:
            return False
        from dataclasses import replace

        self._storage.put(replace(record, retired=1))
        return True

    def unretire(self, memory_id: str) -> bool:
        """Return a retired record to active retrieval."""
        record = self._storage.get(memory_id)
        if record is None:
            return False
        from dataclasses import replace

        self._storage.put(replace(record, retired=0))
        return True

    def active(self, records: Optional[List[MemoryRecord]] = None) -> List[MemoryRecord]:
        """Non-retired records, i.e. the active retrieval pool."""
        pool = records if records is not None else self._storage.list_all()
        return [r for r in pool if not r.retired]


__all__ = ["PointerLayer"]
