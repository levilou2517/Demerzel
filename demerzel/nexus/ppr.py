"""PPR graph retrieval (V1.md §7.5, Phase 3).

Wraps :meth:`~demerzel.interfaces.graph.GraphBackend.ppr` and maps node scores
back onto Chronicle memory ids, so the retrieval layer works in memory space
while the graph works in node space.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

from ..interfaces.graph import GraphBackend


class PPRRetriever:
    """Personalised PageRank over the Nexus graph.

    Depends only on :class:`~demerzel.interfaces.graph.GraphBackend`. When
    graph retrieval is disabled the ablation returns an empty map, which is
    exactly the ``graph off`` switch P1 requires — not a deleted Nexus.
    """

    def __init__(self, graph: GraphBackend, damping: float = 0.85, enabled: bool = True):
        self._graph = graph
        self.damping = float(damping)
        self.enabled = bool(enabled)

    def score(self, seed_node_ids: Sequence[str]) -> Dict[str, float]:
        """PPR distribution over node ids, or ``{}`` when disabled."""
        if not self.enabled or not seed_node_ids:
            return {}
        return self._graph.ppr(list(seed_node_ids), damping=self.damping)

    def memory_scores(self, seed_node_ids: Sequence[str]) -> Dict[str, float]:
        """PPR scores keyed by memory id, using each node's ``memory_id``."""
        node_scores = self.score(seed_node_ids)
        if not node_scores:
            return {}
        out: Dict[str, float] = {}
        for node_id, score in node_scores.items():
            node = self._graph.get_node(node_id)
            if node is None or node.memory_id is None:
                continue
            # Several nodes may map to one memory; keep the strongest.
            if score > out.get(node.memory_id, 0.0):
                out[node.memory_id] = float(score)
        return out

    def top_nodes(self, seed_node_ids: Sequence[str], k: int = 5) -> List[tuple]:
        """The ``k`` highest-scoring ``(node_id, score)`` pairs."""
        scores = self.score(seed_node_ids)
        return sorted(scores.items(), key=lambda kv: -kv[1])[:k]


__all__ = ["PPRRetriever"]
