"""Query router (V1.md §7.6, Phase 7).

First version is rule-based (no trained classifier):

- exact entity / name / phrase  -> ``lexical``
- semantic question            -> ``vector``
- relationship / multi-hop     -> ``graph``

The router returns a backend label; the caller maps it onto the enabled
backends. For Experiment 4 (retrieval modality on-demand routing) this is the
switch that decides whether Nexus PPR is consulted at all.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional, Sequence

_BACKEND_LEXICAL = "lexical"
_BACKEND_VECTOR = "vector"
_BACKEND_GRAPH = "graph"

_MULTIHOP_CUES = ("what caused", "why did", "how did", "between", "related to",
                  "都", "关系", "导致" , "之前", "之后", "和")
_QUESTION_WORDS = ("what", "who", "whom", "when", "where", "which", "how",
                   "为什么", "什么", "谁", "何时", "哪里", "怎样", "怎么")


@dataclass(frozen=True)
class RouteDecision:
    backend: str
    reason: str
    candidate_labels: Optional[Sequence[str]] = None

    def as_dict(self) -> dict:
        return {"backend": self.backend, "reason": self.reason}


class QueryRouter:
    """Deterministic query-to-backend routing. No LLM, no training."""

    def route(self, query: str) -> RouteDecision:
        q = (query or "").strip().lower()

        # 1. Exact entity / phrase -> lexical.
        #    A quote or a bare capitalized proper noun (best-effort heuristic)
        #    signals an exact match.
        if "\"" in q or "'" in q:
            return RouteDecision(_BACKEND_LEXICAL, "quoted exact phrase")
        if re.fullmatch(r"[a-zA-Z][a-z]+", q):
            return RouteDecision(_BACKEND_LEXICAL, "single proper noun")
        tokens = re.findall(r"[a-zA-Z][a-zA-Z]+", q)
        if tokens and all(t[0].isupper() for t in tokens if len(t) > 1):
            return RouteDecision(_BACKEND_LEXICAL, "acronym/uppercase entity")

        # 2. Relationship / multi-hop -> graph.
        if any(cue in q for cue in _MULTIHOP_CUES):
            return RouteDecision(_BACKEND_GRAPH, "relationship/multihop cue")

        # 3. Semantic question -> vector.
        if any(w in q for w in _QUESTION_WORDS):
            return RouteDecision(_BACKEND_VECTOR, "semantic question")

        # Fallback: vector.
        return RouteDecision(_BACKEND_VECTOR, "default")


__all__ = ["QueryRouter", "RouteDecision"]