"""Reranker (V1.md §7.5, Phase 7, optional).

The reranker is a cross-encoder rerank over the top-N hybrid candidates. The
V1 default is **identity** (no-op) because §3.3 forbids spending an LLM call
unless it pays for itself; a caller may supply an LLM-backed recompute that
returns a delta map, which is applied without the LLM ever writing storage.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence

from ..nexus.scoring import Candidate


@dataclass
class RerankOutcome:
    candidates: List[Candidate]
    used_llm: bool = False
    deltas: Dict[str, float] = None

    def __post_init__(self):
        self.deltas = self.deltas or {}


class Reranker:
    """Optional cross-encoder rerank. Identity by default.

    ``recompute`` is injected as ``Callable[[Candidate], float]`` — a pure
    function that returns a new score for a candidate. Keeping it a pure
    function preserves the "LLM decides, engine executes" boundary.
    """

    def __init__(
        self,
        recompute: Optional[Callable[[Candidate], float]] = None,
        enabled: bool = True,
    ):
        self._recompute = recompute
        self.enabled = bool(enabled)

    def rerank(self, candidates: Sequence[Candidate]) -> RerankOutcome:
        if not self.enabled or self._recompute is None:
            return RerankOutcome(list(candidates), used_llm=False)

        deltas: Dict[str, float] = {}
        updated = []
        for c in candidates:
            new_score = self._recompute(c)
            deltas[c.memory_id] = new_score - c.final_score
            from dataclasses import replace

            updated.append(replace(c, final_score=float(new_score)))
        updated.sort(key=lambda c: (-c.final_score, c.memory_id))
        for i, c in enumerate(updated, start=1):
            c.rank = i
        return RerankOutcome(updated, used_llm=True, deltas=deltas)


__all__ = ["RerankOutcome", "Reranker"]