"""Provider abstractions: LLMProvider and EmbeddingProvider.

Contract (V1.md §6.4, §6.5)
---------------------------
- ``LLMProvider.generate(prompt, ...)``    -> text
- ``LLMProvider.classify(input, labels)``  -> structured decision
- ``EmbeddingProvider.embed(texts)``       -> list of vectors

Constraint: core business logic never calls a concrete API. Every LLM call
goes through this seam, which is what makes the "algorithm-first, LLM on
demand" rule auditable and the mock provider a first-class citizen.

Section 17 additionally requires that every LLM call supports timeout,
retry, fallback and logging, and that LLM unavailability never causes memory
loss. :class:`LLMCallPolicy` carries those knobs so implementations share one
shape.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Protocol, Sequence, runtime_checkable


@dataclass(frozen=True)
class LLMCallPolicy:
    """Retry/timeout/fallback settings for one provider (V1.md §17)."""

    timeout_s: float = 30.0
    max_retries: int = 2
    backoff_s: float = 0.5
    #: When true, an exhausted call degrades to the algorithm fallback
    #: rather than raising. Memory writes must fail *open*, never drop data.
    fail_open: bool = True


@dataclass(frozen=True)
class ClassificationDecision:
    """A structured decision produced by an LLM.

    The LLM produces this object; a deterministic engine acts on it. The LLM
    never mutates storage, pointer state, or graph topology directly.
    """

    label: str
    confidence: float = 0.0
    rationale: str = ""
    degraded: bool = False
    error: Optional[str] = None


@runtime_checkable
class LLMProvider(Protocol):
    """LLM seam. Optional by design: the deterministic path must stand alone."""

    @property
    def name(self) -> str:
        """Provider identity, recorded in run metadata for reproducibility."""
        ...

    def generate(self, prompt: str, **kwargs: Any) -> str:
        """Free-text generation. Used for answer generation only."""
        ...

    def classify(
        self, text: str, labels: Sequence[str], **kwargs: Any
    ) -> ClassificationDecision:
        """Choose one label for ``text``. Returns a decision, never a mutation."""
        ...

    def available(self) -> bool:
        """Whether the provider can currently serve calls."""
        ...


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Embedding seam. ContextDistance depends only on this."""

    @property
    def name(self) -> str:
        """Provider identity, recorded in run metadata."""
        ...

    @property
    def dimension(self) -> int:
        """Vector length. Must be stable for a provider instance."""
        ...

    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        """Embed each text; returns one vector per input, in order."""
        ...


@dataclass(frozen=True)
class ProviderConfig:
    """Names/knobs recorded for reproducibility (V1.md §14)."""

    llm: str = "mock"
    embedding: str = "mock"
    model: str = "mock-deterministic"
    temperature: float = 0.0
    seed: int = 0

    def as_dict(self) -> Mapping[str, Any]:
        return {
            "llm": self.llm,
            "embedding": self.embedding,
            "model": self.model,
            "temperature": self.temperature,
            "seed": self.seed,
        }
