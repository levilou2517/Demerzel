"""Context builder (V1.md §7.7, Phase 7).

Assembles the prompt context from retrieval candidates, respecting that the
top-N already selected by the hybrid/moded retrieval is the working set. It is
deterministic and records token usage so the §16 / §§8 token-budget metrics
are computable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

from ..interfaces.storage import MemoryRecord
from ..nexus.scoring import Candidate


@dataclass
class BuiltContext:
    """Assembled context plus token accounting."""

    passage: str
    token_count: int
    used_evidence: List[str] = None
    top_k: int = 0
    truncated: bool = False

    def __post_init__(self):
        self.used_evidence = self.used_evidence or []


class ContextBuilder:
    """Turn ranked candidates into a deterministic prompt passage."""

    def __init__(self, top_k: int = 5):
        self.top_k = int(top_k)

    def build(
        self,
        query: str,
        candidates: Sequence[Candidate],
        records_by_id,
        top_k: Optional[int] = None,
    ) -> BuiltContext:
        """Render top scoring candidates as a passage for the answerer."""
        k = self.top_k if top_k is None else int(top_k)
        top = list(candidates)[:k]

        chunks = [f"Query: {query}"]
        used: List[str] = []
        for candidate in top:
            record = records_by_id.get(candidate.memory_id)
            if record is None:
                continue
            gist = (record.gist or record.content).strip()
            score = candidate.final_score
            chunks.append(f"- [{candidate.memory_id}] (score {score:.4f}): {gist}")
            used.append(candidate.memory_id)
        passage = "\n".join(chunks)

        # Cheap token proxy V1: 1 token ~ 4 chars English, ~1.5 CJK.
        token_count = _approx_tokens(passage)
        return BuiltContext(
            passage=passage,
            token_count=token_count,
            used_evidence=used,
            top_k=len(top),
            truncated=len(top) < len(candidates),
        )


def _approx_tokens(text: str) -> int:
    if not text:
        return 0

    def _char_code_below_512(ch: str) -> bool:
        return ord(ch) < 512

    ascii_len = sum(1 for ch in text if _char_code_below_512(ch))
    cjk_len = sum(1 for ch in text if not _char_code_below_512(ch))
    return ascii_len // 4 + cjk_len // 2 + 1


__all__ = ["BuiltContext", "ContextBuilder"]