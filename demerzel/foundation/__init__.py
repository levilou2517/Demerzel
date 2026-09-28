"""Foundation: append-only evidence layer (V1.md §5.1, Phase 1)."""

from .event_boundary import (
    AlgorithmicEventBoundary,
    BoundaryDecision,
    LLMEventBoundary,
)
from .gist import AlgorithmicGist, LLMGist
from .ingestion import IngestionPipeline, IngestionResult, content_hash
from .raw_store import RawStore, Turn

__all__ = [
    "AlgorithmicEventBoundary",
    "AlgorithmicGist",
    "BoundaryDecision",
    "IngestionPipeline",
    "IngestionResult",
    "LLMEventBoundary",
    "LLMGist",
    "RawStore",
    "Turn",
    "content_hash",
]
