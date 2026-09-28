"""MPE predictor: U_predicted (V1.md §7.3, Phase 6).

Critical timing rule (`docs/mpe_timing.md`, P0-9 / Invariant 8):

    U_predicted MUST be computed BEFORE retrieval happens.

Its inputs may only be:

- ``semantic_similarity(query, memory)`` (from embeddings),
- ``pointer_strength`` (current value),
- ``context_distance`` (current value).

``rank_score`` / retrieval output MUST NOT appear. This module therefore takes
those three scalar channels and no ranking object — the absence of a ranking
input is the guarantee Invariant 8 is testable against.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence


@dataclass(frozen=True)
class PredictedUtility:
    """U_predicted for one memory, before any retrieval."""

    memory_id: str
    value: float
    similarity: float
    pointer: float
    context_distance: float

    def as_dict(self) -> dict:
        return {
            "memory_id": self.memory_id,
            "u_predicted": round(self.value, 6),
            "similarity": round(self.similarity, 6),
            "pointer_strength": round(self.pointer, 6),
            "context_distance": round(self.context_distance, 6),
        }


class MPEPredictor:
    """Estimate predicted utility from the three pre-retrieval channels only."""

    def __init__(
        self,
        w_similarity: float = 0.5,
        w_pointer: float = 0.3,
        w_proximity: float = 0.2,
    ):
        self.w_similarity = float(w_similarity)
        self.w_pointer = float(w_pointer)
        self.w_proximity = float(w_proximity)

    def predict(
        self,
        memory_id: str,
        similarity: float,
        pointer: float,
        context_distance: float,
    ) -> PredictedUtility:
        """``U_pred = w_s*sim + w_p*pointer + w_prox*(1 - distance)``.

        ``(1 - distance)`` turns the accessibility prior into a proximity
        reward; ``distance=0`` (cold start) yields a neutral 1.0 term.
        """
        proximity = 1.0 - context_distance
        value = (
            self.w_similarity * similarity
            + self.w_pointer * pointer
            + self.w_proximity * proximity
        )
        return PredictedUtility(
            memory_id=memory_id,
            value=float(value),
            similarity=float(similarity),
            pointer=float(pointer),
            context_distance=float(context_distance),
        )

    def predict_many(
        self,
        items: Sequence[tuple],
    ) -> List[PredictedUtility]:
        """``items`` = sequence of ``(memory_id, sim, pointer, distance)``."""
        return [
            self.predict(memory_id, sim, pointer, distance)
            for memory_id, sim, pointer, distance in items
        ]


__all__ = ["MPEPredictor", "PredictedUtility"]