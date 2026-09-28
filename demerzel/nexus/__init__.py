"""Nexus: association layer (V1.md §5.3, Phase 3).

Four node kinds, five edge kinds, PPR retrieval, lexical/vector channels, and
the hybrid scorer. MPEFeedback never touches these builders — see
:class:`~demerzel.nexus.pointer_update.NexusPointerWriteback`, which is the
only graph-side bookkeeping the MPE path may reach and which changes metadata
only.
"""

from .nodes import NexusBuilder, node_id_for_memory, tokens
from .pointer_update import NexusPointerWriteback
from .ppr import PPRRetriever
from .retrieval import LexicalRetriever, VectorRetriever, cosine, tokenize
from .scoring import Candidate, HybridScorer, RetrievalResult

__all__ = [
    "Candidate",
    "HybridScorer",
    "LexicalRetriever",
    "NexusBuilder",
    "NexusPointerWriteback",
    "PPRRetriever",
    "RetrievalResult",
    "VectorRetriever",
    "cosine",
    "node_id_for_memory",
    "tokenize",
    "tokens",
]