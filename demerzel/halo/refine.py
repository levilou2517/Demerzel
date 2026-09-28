"""Halo refinement: evict / merge / retain (V1.md §5.4, Phase 4).

The refine dispatcher runs the selected policy over the ring. It is the layer
that interprets *policy names* into ring behaviour; it holds no "delete"
concept. ``merge`` and ``retain`` are reserved for future producers and are
implemented as metadata-tagging + capacity handling so the interface is
present without dragging an unimplemented merge machinery into the MVP.
"""

from __future__ import annotations

from typing import List

from .ring_buffer import EvictionPolicy, HaloTurn, RingBuffer


class Refine:
    """Applies the halo policy and maps outcomes to trace rows."""

    def __init__(self, policy: EvictionPolicy | None = None):
        self.policy = policy or EvictionPolicy("evict")

    def run(self, ring: RingBuffer, texts: List[str], speaker: str = "user") -> List[dict]:
        """Append ``texts`` under the policy; returns outcomes for the trace.

        Invariant 4 is honoured by construction: no path here deletes a
        Chronicle record — it only touches the shallow ring.
        """
        outcomes = []
        for text in texts:
            turn = HaloTurn(index=0, text=text, speaker=speaker)
            applied = self.policy.apply(ring, turn)
            outcomes.append(
                {
                    "policy": self.policy.policy,
                    "index": applied.index,
                    "ring_size_after": ring.size,
                    "evicted_after": len(ring.evicted),
                }
            )
        return outcomes


__all__ = ["Refine"]