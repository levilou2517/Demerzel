"""MPE utility: U_actual (V1.md §7.3, Phase 6).

``U_actual`` is the observed payoff of the retrieval+answer step, computed
AFTER retrieval and answer generation (timeline ``t4`` in ``docs/mpe_timing.md``).

V1 first-stage configuration starts from the simplest signal — evidence only —
and adds answer/retrieval deltas and token cost later::

    U_actual = w1*evidence_score + w2*answer_delta + w3*retrieval_delta - w4*token_cost

with the Phase-0 defaults ``evidence_weight=1.0`` and every other weight 0.0.
The default is therefore exactly the V1.md §7.3 block.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ActualUtility:
    """U_actual for one consumption of a memory."""

    memory_id: Optional[str]
    value: float
    evidence_score: float
    answer_delta: float = 0.0
    retrieval_delta: float = 0.0
    token_cost: float = 0.0

    def as_dict(self) -> dict:
        return {
            "memory_id": self.memory_id,
            "u_actual": round(self.value, 6),
            "evidence_score": round(self.evidence_score, 6),
            "answer_delta": round(self.answer_delta, 6),
            "retrieval_delta": round(self.retrieval_delta, 6),
            "token_cost": round(self.token_cost, 6),
        }


class UtilityEvaluator:
    """Compute U_actual from its four weighted terms."""

    def __init__(
        self,
        evidence_weight: float = 1.0,
        answer_delta_weight: float = 0.0,
        retrieval_delta_weight: float = 0.0,
        token_cost_weight: float = 0.0,
    ):
        self.evidence_weight = float(evidence_weight)
        self.answer_delta_weight = float(answer_delta_weight)
        self.retrieval_delta_weight = float(retrieval_delta_weight)
        self.token_cost_weight = float(token_cost_weight)

    def evaluate(
        self,
        evidence_score: float,
        answer_delta: float = 0.0,
        retrieval_delta: float = 0.0,
        token_cost: float = 0.0,
        memory_id: Optional[str] = None,
    ) -> ActualUtility:
        """``U = w1*evidence + w2*answer + w3*retrieval - w4*token``."""
        value = (
            self.evidence_weight * evidence_score
            + self.answer_delta_weight * answer_delta
            + self.retrieval_delta_weight * retrieval_delta
            - self.token_cost_weight * token_cost
        )
        return ActualUtility(
            memory_id=memory_id,
            value=float(value),
            evidence_score=float(evidence_score),
            answer_delta=float(answer_delta),
            retrieval_delta=float(retrieval_delta),
            token_cost=float(token_cost),
        )


__all__ = ["ActualUtility", "UtilityEvaluator"]