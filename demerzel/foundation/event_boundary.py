"""Event boundary detection (V1.md §7.1 cold-start context, Phase 1/9).

Two implementations behind one callable shape:

* :class:`AlgorithmicEventBoundary` — deterministic, no LLM. It compares
  consecutive turns by token overlap and declares a boundary when overlap
  falls below a threshold. This is the **default** and the fallback.
* :class:`LLMEventBoundary` — asks an :class:`~demerzel.interfaces.providers.LLMProvider`
  to decide. Optional by design.

Section 17 requires that *LLM unavailable must never cause memory loss*: the
LLM implementation therefore degrades to the algorithmic one whenever the
provider is unavailable or the call fails, and marks the result ``degraded``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Sequence

from ..interfaces.providers import LLMProvider

_TOKEN_RE = re.compile(r"[a-z0-9]+|[\u4e00-\u9fff]")


def _tokens(text: str) -> set:
    return set(_TOKEN_RE.findall(text.lower()))


@dataclass(frozen=True)
class BoundaryDecision:
    """Whether a boundary occurs before ``index``, and why."""

    index: int
    is_boundary: bool
    reason: str
    method: str = "algorithmic"
    degraded: bool = False


class AlgorithmicEventBoundary:
    """Deterministic Jaccard-overlap boundary detector. Never calls an LLM."""

    def __init__(self, threshold: float = 0.15):
        self.threshold = float(threshold)

    def detect(self, texts: Sequence[str]) -> List[BoundaryDecision]:
        """Return one decision per turn (index 0 is always a boundary)."""
        decisions: List[BoundaryDecision] = []
        for i, text in enumerate(texts):
            if i == 0:
                decisions.append(
                    BoundaryDecision(i, True, "session_start", "algorithmic")
                )
                continue
            prev, cur = _tokens(texts[i - 1]), _tokens(text)
            if not prev or not cur:
                decisions.append(
                    BoundaryDecision(i, True, "empty_turn", "algorithmic")
                )
                continue
            union = prev | cur
            jaccard = len(prev & cur) / len(union) if union else 0.0
            is_boundary = jaccard < self.threshold
            decisions.append(
                BoundaryDecision(
                    i,
                    is_boundary,
                    f"jaccard={jaccard:.3f}",
                    "algorithmic",
                )
            )
        return decisions


class LLMEventBoundary:
    """Optional LLM-backed boundary detector with an algorithmic fallback."""

    def __init__(
        self,
        provider: Optional[LLMProvider] = None,
        fallback: Optional[AlgorithmicEventBoundary] = None,
    ):
        self._provider = provider
        self._fallback = fallback or AlgorithmicEventBoundary()

    def detect(self, texts: Sequence[str]) -> List[BoundaryDecision]:
        if self._provider is None or not self._provider.available():
            decisions = self._fallback.detect(texts)
            return [
                BoundaryDecision(
                    d.index, d.is_boundary, d.reason, "algorithmic_fallback", True
                )
                for d in decisions
            ]

        decisions = self._fallback.detect(texts)
        out: List[BoundaryDecision] = []
        for i, text in enumerate(texts):
            if i == 0:
                out.append(BoundaryDecision(0, True, "session_start", "llm"))
                continue
            decision = self._provider.classify(
                f"{texts[i-1]}\n---\n{text}",
                ["same_event", "new_event"],
            )
            if decision.degraded:
                base = decisions[i]
                out.append(
                    BoundaryDecision(
                        i, base.is_boundary, base.reason, "algorithmic_fallback", True
                    )
                )
            else:
                is_boundary = decision.label == "new_event"
                out.append(
                    BoundaryDecision(
                        i, is_boundary, decision.rationale or decision.label, "llm"
                    )
                )
        return out


__all__ = [
    "AlgorithmicEventBoundary",
    "BoundaryDecision",
    "LLMEventBoundary",
]
