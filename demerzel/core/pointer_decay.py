"""Pointer decay (V1.md §7.2, Phase 5) — a core mechanism.

V1 formula::

    R = exp(-d / S)
    new_pointer = old_pointer * R

Decay is a *multiplicative* drift on the pointer, driven by an accessibility
term ``d``. Two properties are central to the experiment design:

* **Independence.** Context decay and MPE update are separate operations. This
  class computes the decay factor from ``d``; the update is a multiplication.
  Nothing here uses MPE, so decay-on/decay-off is an isolated ablation.
* **Never deletion.** Decay lowers ``pointer_strength`` but never removes a
  record (Invariant 1 / Invariant 5). ``pointer_strength`` may approach but
  never reach the removal threshold through decay alone — removal is an
  explicit user operation, not a forgetting outcome.

Experiment 1 (V1.md §8.1) selects among: none | time | boundary |
context | context_mpe. The first-version default implements the continuous
context-distance mode (D); the other modes are config-driven ablations of the
same factor computation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ..interfaces.pointer import clip


@dataclass(frozen=True)
class DecayResult:
    """The outcome of one decay step for a memory."""

    memory_id: str
    old_pointer: float
    factor: float
    new_pointer: float
    distance: float
    mode: str
    reason: str

    def as_dict(self) -> dict:
        return {
            "memory_id": self.memory_id,
            "old_pointer": round(self.old_pointer, 6),
            "factor": round(self.factor, 6),
            "new_pointer": round(self.new_pointer, 6),
            "distance": round(self.distance, 6),
            "mode": self.mode,
            "reason": self.reason,
        }


class PointerDecay:
    """Applies the context-distance decay factor to a stored pointer.

    Dependencies: a :class:`~demerzel.interfaces.storage.StorageBackend` (to
    read the current pointer and persist the decayed one) and a callable that
    returns the accessibility distance. The factory default resolves to the
    V1 continuous-context mode.
    """

    MODE_NONE = "none"
    MODE_TIME = "time"
    MODE_BOUNDARY = "boundary"
    MODE_CONTEXT = "context"
    MODE_CONTEXT_MPE = "context_mpe"

    def __init__(
        self,
        storage,
        scale: float = 1.0,
        mode: str = MODE_CONTEXT,
        enabled: bool = True,
        boundary_threshold: float = 0.5,
        time_half_life_turns: float = 50.0,
    ):
        if scale <= 0:
            raise ValueError("scale must be positive")
        self._storage = storage
        self.scale = float(scale)
        self.mode = mode
        self.enabled = bool(enabled)
        self.boundary_threshold = float(boundary_threshold)
        self.time_half_life = float(time_half_life_turns)
        self.log: list = []

    def factor(
        self,
        distance: float,
        reason: str = "computed",
        turn_count: int = 0,
    ) -> float:
        """Compute ``R = exp(-d / S)`` for the active decay mode.

        Cold start (``reason == 'cold_start'``) forces factor ``1.0`` so no
        decay penalty is applied (Invariant 7 is the retrieval-side rule, but
        its decay-side half lives here).
        """
        if not self.enabled:
            return 1.0
        if reason == "cold_start":
            return 1.0

        if self.mode == self.MODE_NONE:
            return 1.0
        if self.mode == self.MODE_TIME:
            # Ebbinghaus: factor scales with turn count.
            d_eff = turn_count / self.time_half_life
            return _exp((-d_eff) / self.scale)
        if self.mode == self.MODE_BOUNDARY:
            # HingeMem-style discrete: d in {0,1}.
            d_eff = 1.0 if distance >= self.boundary_threshold else 0.0
            return _exp((-d_eff) / self.scale)
        # MODE_CONTEXT and MODE_CONTEXT_MPE both use the continuous distance.
        return _exp((-distance) / self.scale)

    def apply(
        self,
        memory_id: str,
        distance: float,
        reason: str = "computed",
        turn_count: int = 0,
    ) -> DecayResult:
        """Read the pointer, decay it, persist it, and return the result.

        Never removes the record; never touches content/source_ptr/created_at
        (the storage's ``update_pointer`` enforces that).
        """
        record = self._storage.get(memory_id)
        if record is None:
            return DecayResult(memory_id, 0.0, 1.0, 0.0, distance, self.mode, "absent")

        old = float(record.pointer_strength)
        factor = self.factor(distance, reason, turn_count)
        new = clip(old * factor)
        self._storage.update_pointer(memory_id, new)
        result = DecayResult(
            memory_id, old, factor, new, distance, self.mode, reason
        )
        self.log.append(result)
        return result

    def apply_many(
        self,
        memory_distances: dict,
        reason_map: Optional[dict] = None,
        turn_count: int = 0,
    ) -> list:
        """Decay every ``{memory_id: distance}`` entry; returns results."""
        results = []
        for memory_id, distance in memory_distances.items():
            reason = (reason_map or {}).get(memory_id, "computed")
            results.append(self.apply(memory_id, distance, reason, turn_count))
        return results


def _exp(x: float) -> float:
    import math

    return math.exp(x)


__all__ = ["DecayResult", "PointerDecay"]