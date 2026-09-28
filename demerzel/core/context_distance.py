"""Context distance (V1.md §7.1, Phase 5) — a core mechanism.

Definition::

    d(m, c) = 1 - cosine_similarity(state_vector(m), state_vector(c))

with ``d in [0, 1]``.

``state_vector`` V1::

    state = alpha * conversation_embedding + beta * halo_centroid

Cold start (``docs/cold_start.md``, Invariant 7): when the halo is empty or the
conversation-embedding turn count is below ``K``, the distance is **0** and the
decay step applies no penalty, tagged ``reason = "cold_start"``. ``K`` defaults
to 3 and is a config field.

Dependency discipline: this class depends only on an
:class:`~demerzel.interfaces.providers.EmbeddingProvider`. It does not import
an implementation of one.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from ..interfaces.providers import EmbeddingProvider


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity in ``[-1, 1]``; 0.0 when either side is degenerate."""
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


@dataclass(frozen=True)
class ContextDistanceResult:
    """One distance computation, with the reason that produced it."""

    distance: float
    reason: str
    cold_start: bool = False


class ContextDistance:
    """Compute context distance between a memory and the current context.

    The memory side is embedded from the record's stored text; the context side
    comes from a state vector supplied by the caller (normally
    :class:`~demerzel.halo.context_state.ContextState`). This keeps the
    mechanism testable without a halo.
    """

    def __init__(
        self,
        embedder: EmbeddingProvider,
        cold_start_k: int = 3,
        cold_start_enabled: bool = True,
    ):
        self._embedder = embedder
        self.cold_start_k = int(cold_start_k)
        self.cold_start_enabled = bool(cold_start_enabled)

    def is_cold_start(self, state_vector: Sequence[float], turn_count: int) -> bool:
        """Whether the cold-start rule applies for this context."""
        if not self.cold_start_enabled:
            return False
        if not state_vector:
            return True
        return turn_count < self.cold_start_k

    def compute(
        self,
        memory_text: str,
        state_vector: Sequence[float],
        turn_count: int,
        memory_vector: Optional[Sequence[float]] = None,
    ) -> ContextDistanceResult:
        """Return ``d`` for one memory against the current context.

        Cold start short-circuits to ``0.0`` with ``reason="cold_start"``
        (Invariant 7). Otherwise ``d = 1 - cosine``, clamped into ``[0, 1]``.
        """
        if self.is_cold_start(state_vector, turn_count):
            return ContextDistanceResult(0.0, "cold_start", cold_start=True)

        mem_vec = (
            list(memory_vector)
            if memory_vector is not None
            else self._embedder.embed([memory_text])[0]
        )
        sim = cosine(mem_vec, state_vector)
        distance = 1.0 - sim
        # Numerical guard: cosine can exceed 1.0 by float error.
        if distance < 0.0:
            distance = 0.0
        if distance > 1.0:
            distance = 1.0
        return ContextDistanceResult(distance, "computed", cold_start=False)

    def compute_many(
        self,
        memories: Sequence[tuple],
        state_vector: Sequence[float],
        turn_count: int,
    ) -> Dict[str, ContextDistanceResult]:
        """Batch variant. ``memories`` is a sequence of ``(memory_id, text)``."""
        return {
            memory_id: self.compute(text, state_vector, turn_count)
            for memory_id, text in memories
        }


__all__ = ["ContextDistance", "ContextDistanceResult", "cosine"]