"""Fact memory track (V1.md §5.2, Phase 2).

Facts are atomic statements. The track's job is to classify an incoming turn as
fact-like and to keep the resulting :class:`~demerzel.interfaces.storage.MemoryRecord`
retired/active distinction honest: retirement removes a record from active
retrieval without touching its evidence.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional

from ..interfaces.storage import MEMORY_FACT, MemoryRecord

#: Cues that mark an atomic assertion. Deliberately lexical — the §3.3 rule
#: forbids spending an LLM call on a decision a deterministic rule can make.
_FACT_CUES = (
    " is ",
    " are ",
    " was ",
    " were ",
    " has ",
    " have ",
    " equals ",
    " means ",
    " refers to ",
    "是",
    "为",
    "等于",
)
_QUESTION_RE = re.compile(r"[?？]\s*$")


@dataclass(frozen=True)
class FactCandidate:
    """One fact-like statement with its extraction provenance."""

    memory_id: str
    statement: str
    confidence: float
    method: str = "lexical"
    degraded: bool = False

    def source_ref(self) -> str:
        return self.memory_id


class FactMemory:
    """Deterministic fact-track classifier.

    No LLM decides *whether* a record is stored — ingestion already stored it.
    This track only labels the record's ``memory_type`` and reports a
    confidence, which keeps it inside the §3.3 "allowed LLM" boundary (it uses
    none at all).
    """

    MEMORY_TYPE = MEMORY_FACT

    def is_fact_like(self, text: str) -> bool:
        """Whether ``text`` asserts something rather than asking it."""
        stripped = (text or "").strip()
        if not stripped:
            return False
        if _QUESTION_RE.search(stripped):
            return False
        lowered = f" {stripped.lower()} "
        return any(cue in lowered for cue in _FACT_CUES)

    def confidence(self, text: str) -> float:
        """Lexical confidence in ``[0, 1]``; higher with more assertion cues."""
        stripped = (text or "").strip()
        if not stripped:
            return 0.0
        lowered = f" {stripped.lower()} "
        hits = sum(1 for cue in _FACT_CUES if cue in lowered)
        if hits == 0:
            return 0.2
        # Saturating: one cue is already fairly confident, more adds little.
        return min(1.0, 0.5 + 0.15 * hits)

    def classify(self, memory_id: str, text: str) -> FactCandidate:
        """Label one record as a fact candidate."""
        return FactCandidate(
            memory_id=memory_id,
            statement=(text or "").strip(),
            confidence=self.confidence(text),
            method="lexical",
        )

    def select(self, records: List[MemoryRecord]) -> List[MemoryRecord]:
        """Those records whose content is fact-like."""
        return [r for r in records if self.is_fact_like(r.content)]


__all__ = ["FactCandidate", "FactMemory"]
