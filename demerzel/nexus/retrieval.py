"""Lexical (BM25) and vector (cosine) retrieval backends (V1.md §7.5, Phase 3).

Both backends are deterministic and depend only on the interfaces they need:
the lexical backend on nothing but the corpus, the vector backend on an
:class:`~demerzel.interfaces.providers.EmbeddingProvider`.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Dict, List, Optional, Sequence

from ..interfaces.providers import EmbeddingProvider
from ..interfaces.storage import MemoryRecord

_TOKEN_RE = re.compile(r"[a-z0-9]+|[\u4e00-\u9fff]")


def tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall((text or "").lower())


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity, 0.0 when either vector is degenerate."""
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


class LexicalRetriever:
    """Okapi BM25 over record content. No embedding, no LLM."""

    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        enabled: bool = True,
    ):
        self.k1 = float(k1)
        self.b = float(b)
        self.enabled = bool(enabled)

    def score(
        self, query: str, records: Sequence[MemoryRecord]
    ) -> Dict[str, float]:
        """BM25 score per memory id; ``{}`` when the backend is disabled."""
        if not self.enabled or not records:
            return {}

        docs = {r.id: tokenize(r.content) for r in records}
        n_docs = len(docs)
        avg_len = sum(len(d) for d in docs.values()) / n_docs if n_docs else 0.0

        df: Counter = Counter()
        for toks in docs.values():
            for token in set(toks):
                df[token] += 1

        q_tokens = tokenize(query)
        scores: Dict[str, float] = {}
        for memory_id, toks in docs.items():
            if not toks:
                scores[memory_id] = 0.0
                continue
            tf = Counter(toks)
            doc_len = len(toks)
            total = 0.0
            for token in q_tokens:
                if token not in tf:
                    continue
                freq = tf[token]
                idf = math.log(1.0 + (n_docs - df[token] + 0.5) / (df[token] + 0.5))
                denom = freq + self.k1 * (1.0 - self.b + self.b * doc_len / (avg_len or 1.0))
                total += idf * (freq * (self.k1 + 1.0)) / (denom or 1.0)
            scores[memory_id] = total
        return scores


class VectorRetriever:
    """Cosine similarity over embeddings. Depends only on the embedder."""

    def __init__(self, embedder: EmbeddingProvider, enabled: bool = True):
        self._embedder = embedder
        self.enabled = bool(enabled)
        self._cache: Dict[str, List[float]] = {}

    def clear_cache(self) -> None:
        self._cache.clear()

    def embed_record(self, record: MemoryRecord) -> List[float]:
        """Cached embedding of the record's *content* and *gist*."""
        if record.id not in self._cache:
            vector = self._embedder.embed([record.content])[0]
            self._cache[record.id] = vector
        return self._cache[record.id]

    def score(
        self, query: str, records: Sequence[MemoryRecord]
    ) -> Dict[str, float]:
        """Cosine similarity per memory id; ``{}`` when disabled."""
        if not self.enabled or not records:
            return {}
        query_vec = self._embedder.embed([query])[0]
        return {
            record.id: cosine(query_vec, self.embed_record(record))
            for record in records
        }

    def similarity(self, text: str, record: MemoryRecord) -> float:
        """Similarity between arbitrary text and one record (used by MPE)."""
        query_vec = self._embedder.embed([text])[0]
        return cosine(query_vec, self.embed_record(record))


__all__ = ["LexicalRetriever", "VectorRetriever", "cosine", "tokenize"]
