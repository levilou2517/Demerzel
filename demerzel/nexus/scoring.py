"""Hybrid retrieval scoring (V1.md §7.5, Phase 3).

The first-version score is::

    score = w_s * cosine_similarity
          + w_g * PPR
          + w_p * pointer_strength
          - w_d * context_distance

with one semantic channel that blends lexical (BM25) and vector (cosine) when
both are enabled::

    semantic = (w_lex * bm25_norm + w_vec * cosine) / (w_lex + w_vec)

Two deliberate properties:

* **context_distance is a prior, not relevance.** It enters with a negative
  weight and is documented as an accessibility prior so a reader cannot
  mistake it for a semantic term.
* Every weight and both backend switches are config fields, so the Experiment-4
  routing ablation and the ``graph off`` ablation are config changes.

The scorer returns a full :class:`Candidate` per memory, which is what the
trace (§5.5) serialises.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from ..interfaces.storage import MemoryRecord


def _normalize(scores: Dict[str, float]) -> Dict[str, float]:
    """Min-max normalise into ``[0, 1]``; all-equal maps to 0.0."""
    if not scores:
        return {}
    values = list(scores.values())
    lo, hi = min(values), max(values)
    if hi - lo <= 1e-12:
        return {k: 0.0 for k in scores}
    return {k: (v - lo) / (hi - lo) for k, v in scores.items()}


@dataclass
class Candidate:
    """One scored retrieval candidate, trace-ready (V1.md §5.5)."""

    memory_id: str
    semantic_score: float = 0.0
    ppr_score: float = 0.0
    pointer_strength: float = 0.0
    context_distance: float = 0.0
    final_score: float = 0.0
    lexical_score: float = 0.0
    vector_score: float = 0.0
    rank: int = 0
    hops: int = 0
    backend: str = ""

    def as_trace_dict(self) -> Dict[str, float]:
        """The exact candidate shape V1.md §5.5 specifies."""
        return {
            "memory_id": self.memory_id,
            "semantic_score": round(self.semantic_score, 6),
            "ppr_score": round(self.ppr_score, 6),
            "pointer_strength": round(self.pointer_strength, 6),
            "context_distance": round(self.context_distance, 6),
            "final_score": round(self.final_score, 6),
        }


@dataclass
class RetrievalResult:
    """All candidates plus the diagnostics the trace and routers need."""

    query: str
    candidates: List[Candidate] = field(default_factory=list)
    backend: str = "hybrid"
    depth: int = 1
    fallback_reason: str = ""
    seed_node_ids: List[str] = field(default_factory=list)

    @property
    def top(self) -> Optional[Candidate]:
        return self.candidates[0] if self.candidates else None

    def ids(self) -> List[str]:
        return [c.memory_id for c in self.candidates]


class HybridScorer:
    """Combine the semantic, graph and accessibility channels."""

    def __init__(
        self,
        w_semantic: float = 0.5,
        w_ppr: float = 0.3,
        w_pointer: float = 0.2,
        w_context_distance: float = 0.2,
        w_lexical: float = 0.5,
        w_vector: float = 0.5,
    ):
        self.w_semantic = float(w_semantic)
        self.w_ppr = float(w_ppr)
        self.w_pointer = float(w_pointer)
        self.w_context_distance = float(w_context_distance)
        self.w_lexical = float(w_lexical)
        self.w_vector = float(w_vector)

    def score(
        self,
        records: Sequence[MemoryRecord],
        lexical_scores: Dict[str, float],
        vector_scores: Dict[str, float],
        ppr_scores: Dict[str, float],
        context_distances: Dict[str, float],
        backend: str = "hybrid",
    ) -> List[Candidate]:
        """Score every record and return candidates sorted by ``final_score``.

        ``backend`` labels the route (lexical/vector/graph/hybrid) that
        produced this result, for the P1 routing ablation and the trace.
        """
        lex_norm = _normalize(lexical_scores)
        vec_norm = _normalize(vector_scores)
        ppr_norm = _normalize(ppr_scores)

        weight_sum = (self.w_lexical + self.w_vector) or 1.0
        candidates: List[Candidate] = []

        for record in records:
            mid = record.id
            lex = lex_norm.get(mid, 0.0)
            vec = vec_norm.get(mid, 0.0)
            semantic = (self.w_lexical * lex + self.w_vector * vec) / weight_sum
            ppr = ppr_norm.get(mid, 0.0)
            distance = float(context_distances.get(mid, 0.0))
            pointer = float(record.pointer_strength)

            final = (
                self.w_semantic * semantic
                + self.w_ppr * ppr
                + self.w_pointer * pointer
                - self.w_context_distance * distance
            )

            candidates.append(
                Candidate(
                    memory_id=mid,
                    semantic_score=semantic,
                    ppr_score=ppr,
                    pointer_strength=pointer,
                    context_distance=distance,
                    final_score=final,
                    lexical_score=lex,
                    vector_score=vec,
                    backend=backend,
                )
            )

        candidates.sort(key=lambda c: (-c.final_score, c.memory_id))
        for i, candidate in enumerate(candidates):
            candidate.rank = i + 1
        return candidates


__all__ = ["Candidate", "HybridScorer", "RetrievalResult"]
