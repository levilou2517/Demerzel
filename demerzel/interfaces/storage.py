"""StorageBackend: persistence of memory records and pointer state.

Contract (V1.md §6.1)
--------------------
- ``get(memory_id)``            -> MemoryRecord or None
- ``put(memory)``               -> write or update a record
- ``update_pointer(id, pointer)``-> update ONLY ``pointer_strength``
- ``list_by_session(session_id)``-> every record of that session

Invariant 2 depends on this interface: ``update_pointer`` must not modify
``content``, ``source_ptr`` or ``created_at``. Reference implementations
enforce that by writing a single column.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Dict, Iterable, List, Optional, Protocol, runtime_checkable

# Memory types permitted in Chronicle (V1.md §5.2).
MEMORY_FACT = "fact"
MEMORY_SCENE = "scene"
MEMORY_TYPES = (MEMORY_FACT, MEMORY_SCENE)


@dataclass(frozen=True)
class MemoryRecord:
    """One Chronicle entry (V1.md §5.2).

    ``retired`` is deliberately separate from deletion: a retired record is no
    longer actively retrieved, but its evidence — and therefore the record —
    still exists (Invariant 1 / Invariant 6).
    """

    id: str
    session_id: str
    episode_id: str
    memory_type: str
    content: str
    gist: str
    created_at: float
    source_ptr: str
    pointer_strength: float = 1.0
    retired: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def with_pointer(self, pointer_strength: float) -> "MemoryRecord":
        """Return a copy carrying a new pointer, leaving every other field alone."""
        return replace(self, pointer_strength=float(pointer_strength))


@runtime_checkable
class StorageBackend(Protocol):
    """Persistence seam. Core mechanisms depend on this, never on SQLite."""

    def get(self, memory_id: str) -> Optional[MemoryRecord]:
        """Return the record, or ``None`` when the id is unknown."""
        ...

    def put(self, memory: MemoryRecord) -> None:
        """Insert or overwrite one record."""
        ...

    def update_pointer(self, memory_id: str, pointer: float) -> None:
        """Write ``pointer_strength`` only.

        Must not touch ``content``, ``source_ptr`` or ``created_at``.
        Must be a no-op for an unknown id rather than creating a record.
        """
        ...

    def list_by_session(self, session_id: str) -> List[MemoryRecord]:
        """Every record belonging to ``session_id``."""
        ...

    def list_all(self) -> List[MemoryRecord]:
        """Every record, in insertion order. Used by retrieval and ablation."""
        ...


@runtime_checkable
class EvidenceStore(Protocol):
    """Append-only Foundation store (V1.md §5.1).

    No method here deletes: ``foundation/raw_store`` is the layer the
    forgetting mechanism may never prune (Invariant 1).
    """

    def append(self, turn: Dict[str, Any]) -> str:
        """Append one turn, returning its ``turn_id``."""
        ...

    def get_turn(self, turn_id: str) -> Optional[Dict[str, Any]]:
        """Return one raw turn by id."""
        ...

    def iter_turns(self, session_id: Optional[str] = None) -> Iterable[Dict[str, Any]]:
        """Iterate raw turns, optionally filtered by session."""
        ...

    def count(self) -> int:
        """Number of stored turns."""
        ...
