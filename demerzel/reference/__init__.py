"""Reference implementations.

These are replaceable: every one of them satisfies an interface in
``demerzel.interfaces``, and no core mechanism imports them directly
(V1.md §3.1). Swap a module here without touching a single mechanism.
"""

from .graph_networkx import DictGraph
from .providers_mock import (
    HashingEmbeddingProvider,
    MockLLMProvider,
    MockLLMProviderUnavailable,
    tokenize,
)
from .storage_memory import InMemoryEvidenceStore, InMemoryStorage
from .storage_sqlite import SqliteEvidenceStore, SqliteStorage

__all__ = [
    "DictGraph",
    "HashingEmbeddingProvider",
    "MockLLMProvider",
    "MockLLMProviderUnavailable",
    "tokenize",
    "InMemoryEvidenceStore",
    "InMemoryStorage",
    "SqliteEvidenceStore",
    "SqliteStorage",
]
