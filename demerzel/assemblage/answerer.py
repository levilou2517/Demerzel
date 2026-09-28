"""Answerer (V1.md §4.1, Phase 7).

Answer generation is one of the four §3.3-sanctioned LLM uses. The answerer
therefore may call an :class:`~demerzel.interfaces.providers.LLMProvider`, but
it may not touch storage, pointers or the graph — it returns text and the
evidence ids it used, and nothing else.

The deterministic ``extractive`` mode exists so the full loop runs with no LLM
at all, which is what the mock-provider experiments need.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

from ..interfaces.providers import LLMProvider


@dataclass
class Answer:
    """One generated answer plus its evidence trail."""

    text: str
    used_evidence: List[str] = field(default_factory=list)
    method: str = "extractive"
    degraded: bool = False
    error: Optional[str] = None
    token_count: int = 0

    def as_dict(self) -> dict:
        return {
            "answer": self.text,
            "used_evidence": list(self.used_evidence),
            "method": self.method,
            "degraded": self.degraded,
            "token_count": self.token_count,
        }


class Answerer:
    """Generate an answer from an assembled context passage."""

    def __init__(
        self,
        provider: Optional[LLMProvider] = None,
        mode: str = "extractive",
        max_chars: int = 400,
    ):
        self._provider = provider
        self.mode = mode
        self.max_chars = int(max_chars)

    def answer(
        self,
        query: str,
        passage: str,
        used_evidence: Sequence[str],
    ) -> Answer:
        """Produce an answer.

        ``extractive`` returns the passage's evidence lines deterministically.
        ``llm`` asks the provider and degrades to extractive when the provider
        is unavailable or fails — §17's rule that an unavailable LLM must never
        lose memory, applied to the answer step.
        """
        evidence = list(used_evidence)

        if self.mode != "llm" or self._provider is None:
            return self._extractive(query, passage, evidence)

        if not self._provider.available():
            fallback = self._extractive(query, passage, evidence)
            fallback.degraded = True
            fallback.error = "provider unavailable"
            return fallback

        try:
            text = self._provider.generate(
                f"Answer the query using only this context.\n{passage}\n\nQuery: {query}"
            )
        except Exception as exc:
            fallback = self._extractive(query, passage, evidence)
            fallback.degraded = True
            fallback.error = str(exc)
            return fallback

        if not text or not text.strip():
            fallback = self._extractive(query, passage, evidence)
            fallback.degraded = True
            fallback.error = "empty generation"
            return fallback

        out = text.strip()[: self.max_chars]
        return Answer(
            text=out,
            used_evidence=evidence,
            method="llm",
            token_count=_approx_tokens(out),
        )

    def _extractive(
        self, query: str, passage: str, evidence: Sequence[str]
    ) -> Answer:
        lines = [
            line.strip()
            for line in passage.splitlines()
            if line.strip().startswith("- ")
        ]
        body = " ".join(lines) if lines else passage.strip()
        text = body[: self.max_chars]
        return Answer(
            text=text,
            used_evidence=list(evidence),
            method="extractive",
            token_count=_approx_tokens(text),
        )


def _approx_tokens(text: str) -> int:
    if not text:
        return 0
    ascii_len = sum(1 for ch in text if ord(ch) < 512)
    cjk_len = len(text) - ascii_len
    return ascii_len // 4 + cjk_len // 2 + 1


__all__ = ["Answer", "Answerer"]