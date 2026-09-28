"""Core mechanisms (V1.md §3.1, §7).

Each class here is independent and wired by constructor injection, and depends
only on interfaces from ``demerzel.interfaces``:

- :class:`~demerzel.core.pointer_decay.PointerDecay` -> ``StorageBackend``
- :class:`~demerzel.core.context_distance.ContextDistance` -> ``EmbeddingProvider``
- MPEFeedback -> ``PointerUpdater`` (lives in ``demerzel.mpe.updater``)
- EventBoundary -> ``LLMProvider`` (optional; see ``demerzel.foundation``)
"""

from .context_distance import ContextDistance, ContextDistanceResult, cosine
from .pointer_decay import DecayResult, PointerDecay

__all__ = [
    "ContextDistance",
    "ContextDistanceResult",
    "DecayResult",
    "PointerDecay",
    "cosine",
]