"""Abstract interface contracts for Demerzel.

Every core mechanism depends on an interface from this package and never on a
concrete implementation, which is what keeps the reference layer
(``demerzel.reference``) replaceable and the experiments attributable
(V1.md §3.1).
"""

from .graph import (
    EDGE_CAUSAL,
    EDGE_CONTRADICTS,
    EDGE_FOLLOW_UP,
    EDGE_RELATED,
    EDGE_SAME_GOAL,
    EDGE_TYPES,
    NODE_CONCEPT,
    NODE_EPISODE,
    NODE_FACT,
    NODE_REFLECTION,
    NODE_TYPES,
    Edge,
    GraphBackend,
    Node,
)
from .pointer import EtaPointerUpdater, PointerUpdate, PointerUpdater, clip
from .providers import (
    ClassificationDecision,
    EmbeddingProvider,
    LLMCallPolicy,
    LLMProvider,
    ProviderConfig,
)
from .storage import (
    MEMORY_FACT,
    MEMORY_SCENE,
    MEMORY_TYPES,
    EvidenceStore,
    MemoryRecord,
    StorageBackend,
)

__all__ = [
    "Edge",
    "GraphBackend",
    "Node",
    "EDGE_CAUSAL",
    "EDGE_CONTRADICTS",
    "EDGE_FOLLOW_UP",
    "EDGE_RELATED",
    "EDGE_SAME_GOAL",
    "EDGE_TYPES",
    "NODE_CONCEPT",
    "NODE_EPISODE",
    "NODE_FACT",
    "NODE_REFLECTION",
    "NODE_TYPES",
    "EtaPointerUpdater",
    "PointerUpdate",
    "PointerUpdater",
    "clip",
    "ClassificationDecision",
    "EmbeddingProvider",
    "LLMCallPolicy",
    "LLMProvider",
    "ProviderConfig",
    "MEMORY_FACT",
    "MEMORY_SCENE",
    "MEMORY_TYPES",
    "EvidenceStore",
    "MemoryRecord",
    "StorageBackend",
]
