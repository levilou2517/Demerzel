"""Context decay orchestration (V1.md §4.1, §7.1, Phase 5).

``entropy`` is the layer that *schedules* forgetting. It owns no formula of its
own: it asks the core :class:`~demerzel.core.context_distance.ContextDistance`
for a distance, hands that distance to the core
:class:`~demerzel.core.pointer_decay.PointerDecay`, and records the outcome in
the :class:`~demerzel.entropy.forgetting_log.ForgettingLog`.

Keeping the formula in ``core`` and the scheduling here is what lets Phase 5 be
ablated with a single switch while the experiment harness still sees a stable
call surface.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from ..core.context_distance import ContextDistance
from ..core.pointer_decay import PointerDecay
from ..interfaces.providers import EmbeddingProvider
from ..interfaces.storage import MemoryRecord, StorageBackend
from .forgetting_log import ForgettingLog


@dataclass
class DecayCycleResult:
    """One decay cycle over a record set."""

    distances: Dict[str, float] = None
    reasons: Dict[str, str] = None
    decayed: List[dict] = None
    cold_start: bool = False

    def __post_init__(self):
        self.distances = self.distances or {}
        self.reasons = self.reasons or {}
        self.decayed = self.decayed or []


class ContextDecay:
    """Runs context-distance decay across the active memory pool."""

    def __init__(
        self,
        storage: StorageBackend,
        context_distance: ContextDistance,
        pointer_decay: PointerDecay,
        forgetting_log: Optional[ForgettingLog] = None,
        enabled: bool = True,
    ):
        self._storage = storage
        self._distance = context_distance
        self._decay = pointer_decay
        self._log = forgetting_log
        self.enabled = bool(enabled)

    def run(
        self,
        records: Sequence[MemoryRecord],
        state_vector: Sequence[float],
        turn_count: int,
    ) -> DecayCycleResult:
        """Decay every active record against the current context state.

        The cold-start rule short-circuits the whole cycle: when it applies,
        every distance is 0 with reason ``cold_start`` and the decay factor is
        1.0, so no penalty is applied (Invariant 7).
        """
        result = DecayCycleResult()

        for record in records:
            if record.retired:
                continue
            outcome = self._distance.compute(
                record.content, state_vector, turn_count
            )
            result.distances[record.id] = outcome.distance
            result.reasons[record.id] = outcome.reason
            if outcome.cold_start:
                result.cold_start = True

        if not self.enabled:
            return result

        for record in records:
            if record.retired:
                continue
            applied = self._decay.apply(
                record.id,
                result.distances[record.id],
                result.reasons[record.id],
                turn_count,
            )
            row = applied.as_dict()
            result.decayed.append(row)
            if self._log is not None:
                self._log.record_decay(row)

        return result


__all__ = ["ContextDecay", "DecayCycleResult"]
