"""Delta encoder: rule-based write decisions (V1.md §4.1, Phase 2).

Decides whether an incoming statement ADDs new content, CONFIRMS an existing
record, or SUPERSEDES it. This is the "delta rules" mechanism — and it is the
place where the forbidden shortcut lives: a supersede writes a **new** record
and retires the old one. It never overwrites memory content, and it never
deletes evidence.

Rule order (deterministic, no LLM):

1. No near-duplicate content  -> ADD
2. Near-duplicate, same polarity -> CONFIRM (pointer touch only)
3. Near-duplicate, negated polarity -> SUPERSEDE (new record + retire old)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Sequence, Tuple

from ..interfaces.providers import EmbeddingProvider
from ..interfaces.storage import MemoryRecord

_NEGATIONS = (" not ", " no ", " never ", " isn't ", " aren't ", " wasn't ", "不", "没", "无")


class DeltaOp(str, Enum):
    """The three write outcomes."""

    ADD = "add"
    CONFIRM = "confirm"
    SUPERSEDE = "supersede"


@dataclass(frozen=True)
class DeltaDecision:
    """One delta verdict with the record it concerns."""

    op: DeltaOp
    target_id: Optional[str]
    similarity: float
    reason: str

    def as_dict(self) -> dict:
        return {
            "op": self.op.value,
            "target_id": self.target_id,
            "similarity": round(self.similarity, 6),
            "reason": self.reason,
        }


def _polarity(text: str) -> int:
    lowered = f" {(text or '').lower()} "
    return -1 if any(neg in lowered for neg in _NEGATIONS) else 1


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


class DeltaEncoder:
    """Rule-based delta decisions over an embedding space.

    Depends only on an :class:`~demerzel.interfaces.providers.EmbeddingProvider`
    and the records it is asked about — it never writes. The caller executes
    the decision, which keeps "LLM/algorithm decides, engine executes" intact.
    """

    def __init__(
        self,
        embedder: EmbeddingProvider,
        duplicate_threshold: float = 0.92,
        enabled: bool = True,
    ):
        self._embedder = embedder
        self.duplicate_threshold = float(duplicate_threshold)
        self.enabled = bool(enabled)

    def decide(
        self, content: str, candidates: Sequence[MemoryRecord]
    ) -> DeltaDecision:
        """Classify ``content`` against ``candidates`` as ADD/CONFIRM/SUPERSEDE."""
        if not self.enabled or not candidates:
            return DeltaDecision(DeltaOp.ADD, None, 0.0, "no_candidates")

        texts = [content] + [c.content for c in candidates]
        vectors = self._embedder.embed(texts)
        query_vec, cand_vecs = vectors[0], vectors[1:]

        best: Tuple[float, Optional[MemoryRecord]] = (0.0, None)
        for record, vec in zip(candidates, cand_vecs):
            score = _cosine(query_vec, vec)
            if score > best[0]:
                best = (score, record)

        similarity, record = best
        if record is None or similarity < self.duplicate_threshold:
            return DeltaDecision(
                DeltaOp.ADD, None, similarity, f"similarity {similarity:.3f} below threshold"
            )

        if _polarity(content) == _polarity(record.content):
            return DeltaDecision(
                DeltaOp.CONFIRM,
                record.id,
                similarity,
                f"near-duplicate (sim {similarity:.3f}), same polarity",
            )
        return DeltaDecision(
            DeltaOp.SUPERSEDE,
            record.id,
            similarity,
            f"near-duplicate (sim {similarity:.3f}), polarity flipped",
        )


__all__ = ["DeltaDecision", "DeltaEncoder", "DeltaOp"]
