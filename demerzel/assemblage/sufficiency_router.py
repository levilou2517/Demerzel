"""Sufficiency router (V1.md §7.7, Phase 7).

    Nexus/Halo -> sufficient?
                  ├── yes -> answer
                  └── no  -> Chronicle -> Foundation

Records: ``retrieval_depth``, ``retrieval_backend``, ``candidate_count``,
``token_count``, ``fallback_reason``.

The router never mutates anything. It is a pure decision over the current
candidate set, which keeps the fallback chain inspectable and the
"sufficiency_router" ablation a config change.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

from ..nexus.scoring import Candidate, RetrievalResult


@dataclass
class SufficiencyDecision:
    """Whether the current evidence is enough, and what to do otherwise."""

    sufficient: bool
    next_layer: str
    reason: str
    depth: int = 0
    backend: str = ""
    candidate_count: int = 0
    token_count: int = 0

    def as_dict(self) -> dict:
        return {
            "sufficient": self.sufficient,
            "next_layer": self.next_layer,
            "reason": self.reason,
            "retrieval_depth": self.depth,
            "retrieval_backend": self.backend,
            "candidate_count": self.candidate_count,
            "token_count": self.token_count,
        }


class SufficiencyRouter:
    """Decide whether Nexus/Halo evidence suffices before widening the search."""

    def __init__(self, threshold: float = 0.35, max_fallback_depth: int = 2):
        self.threshold = float(threshold)
        self.max_fallback_depth = int(max_fallback_depth)

    def assess(
        self, result: RetrievalResult, depth: int = 1
    ) -> SufficiencyDecision:
        """Judge one retrieval result.

        ``sufficient`` when the best candidate clears ``threshold``. Otherwise
        the caller is directed deeper: Chronicle first, then Foundation.
        """
        top = result.top
        count = len(result.candidates)
        token_count = sum(
            len((c.memory_id or "")) for c in result.candidates
        )

        if not result.candidates:
            return SufficiencyDecision(
                False,
                "foundation" if depth >= self.max_fallback_depth else "chronicle",
                "no_candidates",
                depth,
                result.backend,
                count,
                token_count,
            )

        assert top is not None
        if top.final_score >= self.threshold:
            return SufficiencyDecision(
                True, "answer", f"top_score {top.final_score:.4f} >= {self.threshold}",
                depth, result.backend, count, token_count,
            )

        next_layer = (
            "foundation" if depth >= self.max_fallback_depth else "chronicle"
        )
        return SufficiencyDecision(
            False,
            next_layer,
            f"top_score {top.final_score:.4f} < {self.threshold}",
            depth,
            result.backend,
            count,
            token_count,
        )


__all__ = ["SufficiencyDecision", "SufficiencyRouter"]