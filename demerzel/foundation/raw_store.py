"""Foundation: the append-only evidence layer (V1.md §5.1, Phase 1).

Stores raw conversation turns with timestamp, session_id, turn_id, speaker,
metadata and a content_hash. The layer is append-only and is never pruned by
the forgetting mechanism — that is Invariant 1, and the reason pointer decay
cannot destroy provenance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

from ..interfaces.storage import EvidenceStore


@dataclass(frozen=True)
class Turn:
    """One raw conversation turn."""

    turn_id: str
    session_id: str
    timestamp: float
    speaker: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    content_hash: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "turn_id": self.turn_id,
            "session_id": self.session_id,
            "timestamp": self.timestamp,
            "speaker": self.speaker,
            "text": self.text,
            "metadata": dict(self.metadata),
            "content_hash": self.content_hash,
        }


class RawStore:
    """append-only raw conversation store.

    Thin, deliberate wrapper over an
    :class:`~demerzel.interfaces.storage.EvidenceStore`. It exists so the rest
    of the system talks to ``RawStore`` rather than an SQLite class, and so the
    append-only property has exactly one home.

    There is intentionally **no delete method**: explicit user deletion is a
    separate, audited operation (Invariant 6) and does not belong in the
    forgetting path's vocabulary.
    """

    def __init__(self, store: EvidenceStore):
        self._store = store

    def append(self, turn: Turn) -> str:
        """Append one turn; returns its ``turn_id``. Never overwrites."""
        payload = turn.as_dict()
        return self._store.append(payload)

    def get(self, turn_id: str) -> Optional[Dict[str, Any]]:
        """Fetch one raw turn by id, for provenance resolution."""
        return self._store.get_turn(turn_id)

    def iter_turns(self, session_id: Optional[str] = None) -> Iterable[Dict[str, Any]]:
        """Iterate raw turns, optionally restricted to one session."""
        return self._store.iter_turns(session_id)

    def session_turns(self, session_id: str) -> List[Dict[str, Any]]:
        """All raw turns of one session, in insertion order."""
        return list(self._store.iter_turns(session_id))

    def count(self) -> int:
        return self._store.count()

    def recent(self, session_id: str, limit: int) -> List[Dict[str, Any]]:
        """The last ``limit`` turns of a session (Halo reload helper)."""
        turns = self.session_turns(session_id)
        return turns[-limit:] if limit > 0 else []


__all__ = ["RawStore", "Turn"]
