"""Halo: shallow-memory ring buffer (V1.md §5.4, Phase 4).

The halo keeps the last ``max_turns`` turns in memory as working context. Its
terminology is deliberate — ``append`` / ``peek`` / ``evict`` / ``merge`` /
``retain`` / ``snapshot`` — and there is **no ``delete``**: halo eviction does
not mean memory deletion (Invariant 4). What leaves the ring is only the
*shallow* copy; the same evidence already lives in Foundation/Chronicle.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Sequence


@dataclass
class HaloTurn:
    """One turn held by the ring buffer."""

    index: int
    text: str
    speaker: str = "user"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "text": self.text,
            "speaker": self.speaker,
            "metadata": dict(self.metadata),
        }


class RingBuffer:
    """Fixed-capacity ring buffer of :class:`HaloTurn` objects."""

    def __init__(self, max_turns: int = 8):
        if max_turns <= 0:
            raise ValueError("max_turns must be positive")
        self.max_turns = int(max_turns)
        self._buffer: Deque[HaloTurn] = deque(maxlen=self.max_turns)
        self._evicted: List[HaloTurn] = []
        self._seq = 0

    def append(self, text: str, speaker: str = "user", metadata: Optional[Dict[str, Any]] = None) -> HaloTurn:
        """Push a turn. If it overflows, the oldest is evicted (not deleted)."""
        turn = HaloTurn(
            index=self._seq, text=text, speaker=speaker, metadata=dict(metadata or {})
        )
        self._seq += 1
        if len(self._buffer) == self.max_turns:
            evicted = self._buffer.popleft()
            self._evicted.append(evicted)
        self._buffer.append(turn)
        return turn

    def peek(self, k: int = 1) -> List[HaloTurn]:
        """The most recent ``k`` turns, oldest-first."""
        k = max(1, min(k, len(self._buffer)))
        return list(self._buffer)[-k:]

    def all(self) -> List[HaloTurn]:
        return list(self._buffer)

    @property
    def size(self) -> int:
        return len(self._buffer)

    @property
    def empty(self) -> bool:
        return len(self._buffer) == 0

    @property
    def evicted(self) -> List[HaloTurn]:
        """Turns that left the ring. The Chronicle copies still exist."""
        return list(self._evicted)

    def snapshot(self) -> List[Dict[str, Any]]:
        """A serialisable view of the current ring (term discipline: snapshot)."""
        return [t.as_dict() for t in self._buffer]

    def stats(self) -> Dict[str, Any]:
        return {
            "capacity": self.max_turns,
            "size": self.size,
            "evicted_count": len(self._evicted),
        }


class EvictionPolicy:
    """The three V1.md §5.4 policies. All are metadata/filter decisions.

    - ``evict``  -> drop the shallow turn from the ring (default;
                    the persistent copy is untouched, Invariant 4).
    - ``merge``  -> keep it but tagged as merged (a placeholder retained for
                    the future merge producer; does not itself mutate storage).
    - ``retain`` -> keep it in the ring despite capacity (ring may grow).
    """

    EVICT = "evict"
    MERGE = "merge"
    RETAIN = "retain"

    def __init__(self, policy: str = "evict"):
        if policy not in (self.EVICT, self.MERGE, self.RETAIN):
            raise ValueError(f"unknown policy: {policy}")
        self.policy = policy

    def apply(self, ring: RingBuffer, turn: HaloTurn) -> HaloTurn:
        """Run one policy step. ``evict`` and ``merge`` keep the ring bounded;
        ``retain`` appends without eviction (intended for buffering)."""
        if self.policy in (self.EVICT, self.MERGE):
            return ring.append(
                turn.text, speaker=turn.speaker, metadata={**turn.metadata, "policy": self.policy}
            )
        # retain: push past the logical cap is a caller choice via max_turns
        # support here is minimal; bump the capacity by one for this turn.
        ring.max_turns += 1
        return ring.append(
            turn.text, speaker=turn.speaker, metadata={**turn.metadata, "policy": self.policy}
        )


__all__ = ["EvictionPolicy", "HaloTurn", "RingBuffer"]