"""Context state vector for the halo (V1.md §7.1, Phase 4).

The first-version state vector is::

    state = alpha * conversation_embedding + beta * halo_centroid

Nexus active-node embedding is deliberately excluded from V1 (V1.md §7.1 says
so explicitly). Both weights are config fields.

The halo centroid is the mean of the halo turns' embeddings, which is what
gives "context distance" its meaning: how far the current conversation has
drifted from the recent shallow context.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

from ..interfaces.providers import EmbeddingProvider


def _mean(vectors: Sequence[Sequence[float]]) -> List[float]:
    if not vectors:
        return []
    dim = len(vectors[0])
    out = [0.0] * dim
    for vec in vectors:
        for i, value in enumerate(vec):
            out[i] += value
    n = float(len(vectors))
    return [v / n for v in out]


def _l2_normalize(vector: Sequence[float]) -> List[float]:
    norm = sum(v * v for v in vector) ** 0.5
    if norm == 0.0:
        return list(vector)
    return [v / norm for v in vector]


class ContextState:
    """Builds the halo's state vector from the shallow ring and an embedder."""

    def __init__(
        self,
        embedder: EmbeddingProvider,
        alpha: float = 0.5,
        beta: float = 0.5,
    ):
        self._embedder = embedder
        self.alpha = float(alpha)
        self.beta = float(beta)

    def conversation_embedding(self, turns: Sequence[str]) -> List[float]:
        """Embedding of the conversation so far (mean of turn embeddings)."""
        if not turns:
            return []
        vectors = self._embedder.embed(list(turns))
        return _mean(vectors)

    def halo_centroid(self, halo_texts: Sequence[str]) -> List[float]:
        """Centroid of the halo turns' embeddings; ``[]`` when the halo is empty."""
        if not halo_texts:
            return []
        vectors = self._embedder.embed(list(halo_texts))
        return _mean(vectors)

    def state_vector(
        self, conversation_turns: Sequence[str], halo_texts: Sequence[str]
    ) -> List[float]:
        """``alpha * conversation_embedding + beta * halo_centroid``.

        Returns ``[]`` when both components are absent, which is the signal
        :class:`~demerzel.core.context_distance.ContextDistance` reads to
        declare the cold-start rule (Invariant 7).
        """
        conv = self.conversation_embedding(conversation_turns)
        halo = self.halo_centroid(halo_texts)

        if not conv and not halo:
            return []
        if not conv:
            return _l2_normalize(halo)
        if not halo:
            return _l2_normalize(conv)

        combined = [
            self.alpha * c + self.beta * h for c, h in zip(conv, halo)
        ]
        return _l2_normalize(combined)

    def turn_count(self, conversation_turns: Sequence[str]) -> int:
        """Number of conversation turns contributing to the state vector."""
        return len(conversation_turns)


__all__ = ["ContextState"]
