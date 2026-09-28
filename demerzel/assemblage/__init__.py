"""Assemblage: surface assembly layer (V1.md §4.1, Phase 7).

Query routing, sufficiency routing, optional reranking, context assembly and
answer generation. Answer generation is the only step permitted to call an LLM,
and it never writes memory.
"""

from .answerer import Answer, Answerer
from .context_builder import BuiltContext, ContextBuilder
from .query_router import QueryRouter, RouteDecision
from .reranker import RerankOutcome, Reranker
from .sufficiency_router import SufficiencyDecision, SufficiencyRouter

__all__ = [
    "Answer",
    "Answerer",
    "BuiltContext",
    "ContextBuilder",
    "QueryRouter",
    "RerankOutcome",
    "Reranker",
    "RouteDecision",
    "SufficiencyDecision",
    "SufficiencyRouter",
]