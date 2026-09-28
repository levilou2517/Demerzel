"""Gist extraction: instant regularities at ingestion time (V1.md §4.1).

Algorithm-first. The default extractor is deterministic: it selects the most
salient sentences by token overlap with the turn's own vocabulary and caps the
result by length. No LLM is used to *decide whether to store* anything — the
gist is a derived convenience field, and the raw evidence is already stored.
"""

from __future__ import annotations

import re
from typing import List, Optional, Sequence

from ..interfaces.providers import LLMProvider

_SENT_RE = re.compile(r"[^.!?。！？\n]+[.!?。！？]?")
_TOKEN_RE = re.compile(r"[a-z0-9]+|[\u4e00-\u9fff]")


def _tokens(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())


class AlgorithmicGist:
    """Deterministic extractive gist. Never calls an LLM."""

    def __init__(self, max_chars: int = 160, max_sentences: int = 2):
        self.max_chars = int(max_chars)
        self.max_sentences = int(max_sentences)

    def extract(self, text: str) -> str:
        """Return a short extractive gist of ``text``."""
        text = (text or "").strip()
        if not text:
            return ""
        sentences = [s.strip() for s in _SENT_RE.findall(text) if s.strip()]
        if not sentences:
            return text[: self.max_chars]

        # Salience = average token frequency of the sentence's tokens, so a
        # sentence built from the turn's dominant vocabulary ranks highest.
        freq: dict = {}
        for token in _tokens(text):
            freq[token] = freq.get(token, 0) + 1
        scored = []
        for idx, sentence in enumerate(sentences):
            toks = _tokens(sentence)
            if not toks:
                scored.append((0.0, idx, sentence))
                continue
            score = sum(freq.get(t, 0) for t in toks) / len(toks)
            scored.append((score, idx, sentence))
        scored.sort(key=lambda row: (-row[0], row[1]))

        picked = sorted(scored[: self.max_sentences], key=lambda row: row[1])
        gist = " ".join(row[2] for row in picked)
        return gist[: self.max_chars].strip()


class LLMGist:
    """Optional LLM-backed gist with deterministic fallback (V1.md §17)."""

    def __init__(
        self,
        provider: Optional[LLMProvider] = None,
        fallback: Optional[AlgorithmicGist] = None,
        max_chars: int = 160,
    ):
        self._provider = provider
        self._fallback = fallback or AlgorithmicGist(max_chars=max_chars)
        self.max_chars = max_chars

    def extract(self, text: str) -> str:
        if self._provider is None or not self._provider.available():
            return self._fallback.extract(text)
        try:
            out = self._provider.generate(
                f"Summarise in one short sentence:\n{text}"
            ).strip()
        except Exception:
            return self._fallback.extract(text)
        if not out:
            return self._fallback.extract(text)
        return out[: self.max_chars]


__all__ = ["AlgorithmicGist", "LLMGist"]
