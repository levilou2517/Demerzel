"""PointerUpdater: encapsulation of ``pointer_strength`` mutation.

Contract (V1.md §6.3)
---------------------
- ``update(memory_id, mpe)`` -> new ``pointer_strength``

Constraint: touches ONLY ``pointer_strength``. Foundation evidence, Chronicle
content and Nexus topology are untouchable from here — that is Invariant 2
and Invariant 3 expressed as a type.

The update rule (V1.md §7.4)::

    P_next = clip(P_current + eta * MPE, 0, 1)

with ``eta`` configurable (default 0.05). Two deliberate properties:

* ``eta`` defaults to 0.0 when MPE is disabled, so the ablation is a config
  change rather than a code path.
* The clamp keeps ``pointer_strength`` in ``[0, 1]`` so a single strong MPE
  cannot saturate the accessibility state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Protocol, runtime_checkable


def clip(value: float, low: float = 0.0, high: float = 1.0) -> float:
    """Clamp ``value`` into ``[low, high]``."""
    if value < low:
        return low
    if value > high:
        return high
    return value


@dataclass(frozen=True)
class PointerUpdate:
    """One applied pointer change, for the trace and pointer_updates.jsonl."""

    memory_id: str
    old: float
    new: float
    reason: str
    mpe: float = 0.0

    def as_dict(self) -> dict:
        return {
            "memory_id": self.memory_id,
            "old": round(self.old, 6),
            "new": round(self.new, 6),
            "reason": self.reason,
            "mpe": round(self.mpe, 6),
        }


@runtime_checkable
class PointerUpdater(Protocol):
    """Pointer seam. MPEFeedback depends only on this."""

    def update(self, memory_id: str, mpe: float) -> float:
        """Apply one MPE step, persist it, and return the new pointer value."""
        ...

    def current(self, memory_id: str) -> Optional[float]:
        """Read the current pointer without mutating anything."""
        ...


class EtaPointerUpdater:
    """Reference update rule: ``P_next = clip(P + eta * MPE, 0, 1)``.

    Depends only on ``StorageBackend`` and a scalar ``eta``, so it can be
    composed with any storage and switched off by setting ``eta = 0``.
    """

    def __init__(self, storage, eta: float = 0.05, enabled: bool = True):
        self._storage = storage
        self._eta = 0.0 if not enabled else float(eta)
        self._enabled = bool(enabled) and self._eta != 0.0
        self.log: List[PointerUpdate] = []

    @property
    def eta(self) -> float:
        return self._eta

    @property
    def enabled(self) -> bool:
        return self._enabled

    def current(self, memory_id: str) -> Optional[float]:
        record = self._storage.get(memory_id)
        if record is None:
            return None
        return float(record.pointer_strength)

    def update(self, memory_id: str, mpe: float) -> float:
        """Apply ``eta * mpe`` to the stored pointer and return the new value.

        Unknown ids are a no-op returning 0.0 — an updater never conjures a
        record. With MPE disabled the pointer is returned unchanged, which is
        what makes the Exp-2 control group a pure config difference.
        """
        record = self._storage.get(memory_id)
        if record is None:
            return 0.0

        old = float(record.pointer_strength)
        if not self._enabled:
            self.log.append(
                PointerUpdate(memory_id, old, old, "mpe_disabled", float(mpe))
            )
            return old

        new = clip(old + self._eta * float(mpe))
        self._storage.update_pointer(memory_id, new)
        self.log.append(
            PointerUpdate(
                memory_id,
                old,
                new,
                "positive_mpe" if mpe > 0 else ("negative_mpe" if mpe < 0 else "zero_mpe"),
                float(mpe),
            )
        )
        return new
