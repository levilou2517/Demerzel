"""GraphBackend: memory-graph storage and query.

Contract (V1.md §6.2)
--------------------
- ``add_node(node)``              -> add a node
- ``add_edge(edge)``              -> add an edge
- ``neighbors(node_id, types)``   -> incident edges
- ``ppr(seeds, damping)``         -> personalised-PageRank scores

Experiment-2 isolation is enforced *here*, at the interface: ``MPEFeedback``
must never call :meth:`GraphBackend.add_edge` or :meth:`GraphBackend.add_node`.
The G-fixed constraint therefore holds by construction rather than by
convention — see Invariant 3.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Sequence, runtime_checkable

# Four node kinds (V1.md §5.3).
NODE_EPISODE = "episode"
NODE_FACT = "fact"
NODE_REFLECTION = "reflection"
NODE_CONCEPT = "concept"
NODE_TYPES = (NODE_EPISODE, NODE_FACT, NODE_REFLECTION, NODE_CONCEPT)

# Five edge kinds (V1.md §5.3).
EDGE_CAUSAL = "causal"
EDGE_RELATED = "related"
EDGE_CONTRADICTS = "contradicts"
EDGE_FOLLOW_UP = "follow_up"
EDGE_SAME_GOAL = "same_goal"
EDGE_TYPES = (
    EDGE_CAUSAL,
    EDGE_RELATED,
    EDGE_CONTRADICTS,
    EDGE_FOLLOW_UP,
    EDGE_SAME_GOAL,
)


@dataclass(frozen=True)
class Node:
    """A Nexus node."""

    node_id: str
    node_type: str
    label: str = ""
    memory_id: Optional[str] = None
    created_at: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Edge:
    """A Nexus edge. Every field of V1.md §5.3 is present."""

    edge_id: str
    source_id: str
    target_id: str
    edge_type: str
    weight: float = 1.0
    evidence_ptr: str = ""
    created_at: float = 0.0
    last_activated: float = 0.0
    co_activation_count: int = 0
    pointer_strength: float = 1.0


@runtime_checkable
class GraphBackend(Protocol):
    """Graph seam. Core mechanisms depend on this, never on NetworkX."""

    def add_node(self, node: Node) -> None:
        """Add or replace one node."""
        ...

    def add_edge(self, edge: Edge) -> None:
        """Add or replace one edge."""
        ...

    def get_node(self, node_id: str) -> Optional[Node]:
        """Return one node, or ``None``."""
        ...

    def neighbors(
        self, node_id: str, edge_types: Optional[Sequence[str]] = None
    ) -> List[Edge]:
        """Edges incident to ``node_id``, optionally filtered by type."""
        ...

    def ppr(
        self, seeds: Sequence[str], damping: float = 0.85
    ) -> Dict[str, float]:
        """Personalised PageRank from ``seeds``, as ``{node_id: score}``."""
        ...

    def nodes(self) -> List[Node]:
        """Every node."""
        ...

    def edges(self) -> List[Edge]:
        """Every edge."""
        ...

    def topology_fingerprint(self) -> str:
        """Stable digest of nodes+edges, used by Invariant 3."""
        ...
