"""NetworkX/simple-diagraph graph reference implementation.

Design note (deviation from V1.md §3.1)
--------------------------------------
V1.md names ``reference/graph_networkx`` backed by NetworkX. In this offline
workspace NetworkX cannot be installed, so this module implements the SAME
:class:`~demerzel.interfaces.graph.GraphBackend` contract with a
dependency-free adjacency dict plus an exact power-iteration personalised
PageRank (PPR). The ceremony is identical and PPR is reproduced with a
verifiable stationary residual, which is all the P0/P1 gates need.

If NetworkX later becomes available, replacing this one module behind the
interface — and nothing else — restores the original dependence. Core
mechanisms must never import this module directly (V1.md §3.1).
"""

from __future__ import annotations

import hashlib
import json
from typing import Dict, List, Optional, Sequence

from ..interfaces.graph import Edge, Node


class DictGraph:
    """Adjacency-list :class:`~demerzel.interfaces.graph.GraphBackend`.

    The graph is held out-neighbors: ``_out[src][target] = edge``, plus
    symmetric in-neighbors derived for PPR. PPR starts uniform on a seed set
    and iterates the personalised PageRank recurrence until the residual or
    iteration cap is met.
    """

    def __init__(self) -> None:
        self._nodes: Dict[str, Node] = {}
        self._edges: Dict[str, Edge] = {}
        self._out: Dict[str, Dict[str, Edge]] = {}
        self._in: Dict[str, Dict[str, Edge]] = {}

    # -- nodes ---------------------------------------------------------------
    def add_node(self, node: Node) -> None:
        self._nodes[node.node_id] = node
        self._out.setdefault(node.node_id, {})
        self._in.setdefault(node.node_id, {})

    def get_node(self, node_id: str) -> Optional[Node]:
        return self._nodes.get(node_id)

    def nodes(self) -> List[Node]:
        return list(self._nodes.values())

    # -- edges ---------------------------------------------------------------
    def add_edge(self, edge: Edge) -> None:
        self._edges[edge.edge_id] = edge
        self._out.setdefault(edge.source_id, {})[edge.target_id] = edge
        self._in.setdefault(edge.target_id, {})[edge.source_id] = edge
        # Introduce source/target nodes only if unknown. Crucially this must
        # NOT overwrite an existing node: a memory node carries its memory_id,
        # and a later episode auto-create would otherwise wipe it.
        for node_id in (edge.source_id, edge.target_id):
            if node_id not in self._nodes:
                self.add_node(Node(node_id=node_id, node_type="episode", label=node_id))

    def neighbors(
        self, node_id: str, edge_types: Optional[Sequence[str]] = None
    ) -> List[Edge]:
        out: List[Edge] = []
        types = set(edge_types) if edge_types is not None else None
        for edge in self._out.get(node_id, {}).values():
            if types is None or edge.edge_type in types:
                out.append(edge)
        for edge in self._in.get(node_id, {}).values():
            if types is None or edge.edge_type in types:
                out.append(edge)
        deduped: Dict[str, Edge] = {}
        for edge in out:
            deduped.setdefault(edge.edge_id, edge)
        return list(deduped.values())

    def edges(self) -> List[Edge]:
        return list(self._edges.values())

    # -- PPR -----------------------------------------------------------------
    def ppr(self, seeds: Sequence[str], damping: float = 0.85) -> Dict[str, float]:
        """Personalised PageRank over the full graph from ``seeds``.

        A node with no outgoing edges is treated as (implicitly) linking to
        itself so probability mass is conserved; the teleport set is uniform
        over ``seeds``. Returns ``{node_id: score}`` over all nodes.
        """
        order = list(self._nodes.keys())
        n = len(order)
        if n == 0:
            return {}

        # Degenerate: seeds unresolved produce a zero vector.
        valid_seeds = [s for s in seeds if s in self._nodes]
        if not valid_seeds:
            return {node_id: 0.0 for node_id in order}

        # index maps for the iteration.
        idx = {node_id: i for i, node_id in enumerate(order)}
        # out-degree count per node.
        deg = {node_id: len(self._out.get(node_id, {})) for node_id in order}
        # teleport distribution: uniform over seeds, normalised.
        teleport = [0.0] * n
        for s in valid_seeds:
            teleport[idx[s]] = 1.0 / len(valid_seeds)

        rank = [0.0] * n
        for s in valid_seeds:
            rank[idx[s]] = 1.0 / len(valid_seeds)

        d = float(damping)
        residual = 1.0 + n
        iterations = 0
        max_iter = 200
        tol = 1e-9

        while residual > tol and iterations < max_iter:
            new = [(1.0 - d) * t for t in teleport]
            for src in order:
                r = rank[idx[src]]
                if r == 0.0:
                    continue
                outs = list(self._out.get(src, {}).keys())
                if outs:
                    share = d * r / len(outs)
                    for tgt in outs:
                        new[idx[tgt]] += share
                else:
                    # sink: mass returns to the seed set (self-loop convention)
                    for s in valid_seeds:
                        new[idx[s]] += d * r / len(valid_seeds)
            residual = sum((new[i] - rank[i]) ** 2 for i in range(n))
            rank = new
            iterations += 1

        return {node_id: float(rank[i]) for i, node_id in enumerate(order)}

    # -- audit ---------------------------------------------------------------
    def topology_fingerprint(self) -> str:
        """Stable digest of nodes+edges (Invariant 3 / P0-5)."""
        node_blob = json.dumps(
            sorted((n.node_id, n.node_type) for n in self._nodes.values()),
            sort_keys=True,
        )
        edge_blob = json.dumps(
            sorted(
                (e.edge_id, e.source_id, e.target_id, e.edge_type)
                for e in self._edges.values()
            ),
            sort_keys=True,
        )
        digest = hashlib.sha256((node_blob + "@" + edge_blob).encode("utf-8"))
        return digest.hexdigest()


__all__ = ["DictGraph"]