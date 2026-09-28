"""Deterministic mock providers.

These are the default providers (V1.md §3.2's "Mock providers under
reference/"). They are intentionally deterministic: the same text yields the
same vector, so retrieval and MPE experiments are exactly reproducible without
a network call, and the algorithm-first rule stays testable.

The embedding is a hashed bag-of-words projection: every token is hashed into
one of ``dimension`` buckets with a signed weight, then L2-normalised. That is
enough structure for cosine similarity to behave meaningfully (shared tokens
raise similarity) while remaining dependency-free and stable across runs.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any, Dict, List, Sequence

from ..interfaces.providers import (
    ClassificationDecision,
    LLMCallPolicy,
)

_TOKEN_RE = re.compile(r"[a-z0-9]+|[\u4e00-\u9fff]")


def tokenize(text: str) -> List[str]:
    """Lowercase word/character tokens; CJK is split per character."""
    return _TOKEN_RE.findall(text.lower())


class HashingEmbeddingProvider:
    """Deterministic hashed bag-of-words embeddings.

    ``dimension`` is fixed for the instance's lifetime, as the interface
    requires.
    """

    def __init__(self, dimension: int = 256):
        if dimension <= 0:
            raise ValueError("dimension must be positive")
        self._dimension = int(dimension)

    @property
    def name(self) -> str:
        return "mock-hashing"

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        return [self._embed_one(t) for t in texts]

    def _embed_one(self, text: str) -> List[float]:
        vec = [0.0] * self._dimension
        tokens = tokenize(text)
        if not tokens:
            return vec
        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            bucket = int.from_bytes(digest[:4], "big") % self._dimension
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vec[bucket] += sign
        norm = sum(v * v for v in vec) ** 0.5
        if norm > 0:
            vec = [v / norm for v in vec]
        return vec


class MockLLMProvider:
    """Deterministic stand-in for an LLM.

    ``classify`` scores each candidate label by token overlap and returns the
    best as a :class:`ClassificationDecision` — a structured decision, never a
    mutation. ``generate`` echoes the salient part of the prompt so answer
    assembly stays inspectable.

    ``available()`` is switchable so tests can exercise the §17 degradation
    paths (write gate fail-open, recall gate degraded, injection fail-closed).
    """

    def __init__(
        self,
        policy: LLMCallPolicy | None = None,
        available: bool = True,
        dimension: int = 256,
    ):
        self._policy = policy or LLMCallPolicy()
        self._available = bool(available)
        self._embedder = HashingEmbeddingProvider(dimension)
        self.calls: List[Dict[str, Any]] = []

    @property
    def name(self) -> str:
        return "mock-llm"

    @property
    def policy(self) -> LLMCallPolicy:
        return self._policy

    def set_available(self, value: bool) -> None:
        """Toggle availability to drive the degradation paths in tests."""
        self._available = bool(value)

    def available(self) -> bool:
        return self._available

    def classify(
        self, text: str, labels: Sequence[str], **kwargs: Any
    ) -> ClassificationDecision:
        self.calls.append({"op": "classify", "text": text, "labels": list(labels)})
        if not self._available:
            # Degraded structured decision: the caller decides the fallback.
            return ClassificationDecision(
                label=labels[0] if labels else "",
                confidence=0.0,
                rationale="llm unavailable",
                degraded=True,
                error="provider unavailable",
            )
        if not labels:
            return ClassificationDecision(label="", confidence=0.0, rationale="no labels")

        text_tokens = set(tokenize(text))
        best_label, best_score = labels[0], -1.0
        for label in labels:
            label_tokens = set(tokenize(label))
            if not label_tokens:
                continue
            overlap = len(text_tokens & label_tokens) / len(label_tokens)
            if overlap > best_score:
                best_label, best_score = label, overlap
        confidence = 0.0 if best_score < 0 else round(best_score, 6)
        return ClassificationDecision(
            label=best_label,
            confidence=confidence,
            rationale=f"token overlap {confidence}",
        )

    def generate(self, prompt: str, **kwargs: Any) -> str:
        self.calls.append({"op": "generate", "prompt": prompt})
        if not self._available:
            return ""
        return prompt.strip()


class MockLLMProviderUnavailable(MockLLMProvider):
    """Convenience subclass: an LLM provider that is never available."""

    def __init__(self, policy: LLMCallPolicy | None = None):
        super().__init__(policy=policy, available=False)


__all__ = [
    "HashingEmbeddingProvider",
    "MockLLMProvider",
    "MockLLMProviderUnavailable",
    "tokenize",
]