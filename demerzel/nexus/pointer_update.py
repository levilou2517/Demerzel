"""Nexus pointer-update writeback (V1.md §4.1, Phase 3).

When a memory is re-accessed, the *edges* incident to it are marked "activated":
their ``co_activation_count`` increments and ``last_activated`` refreshes.
Crucially this is **metadata-only**: it never adds or removes a node or edge,
so Invariant 3 (``graph_before == graph_after`` under MPE) holds at the
topology level — the fingerprints P0-5 compares exclude these fields by design.

The core ``pointer_strength`` mutation itself lives in
:class:`~demerzel.interfaces.pointer.EtaPointerUpdater`, not here. This module
owns only the graph-side bookkeeping, and it reads the graph rather than
writing topology.
"""

from __future__ import annotations

import time
from dataclasses import replace
from typing import List, Optional

from ..interfaces.graph import GraphBackend


class NexusPointerWriteback:
    """Mark edges activated as evidence of re-access.

    Depends only on :class:`~demerzel.interfaces.graph.GraphBackend`. Nothing
    here may call ``add_edge`` or ``add_node``.
    """

    def __init__(self, graph: GraphBackend, clock=None):
        self._graph = graph
        self._clock = clock or time.time
        #: rows acknowledged by this writeback, for the trace
        self.touches: List[str] = []

    def touch_node(self, node_id: str) -> None:
        """Refresh activation on the edges incident to ``node_id``."""
        now = self._clock()
        edges = self._graph.neighbors(node_id)
        graph = self._graph
        for edge in edges:
            graph.add_edge(
                replace(
                    edge,
                    last_activated=now,
                    co_activation_count=edge.co_activation_count + 1,
                )
            )
        self.touches.append(node_id)

    def touch_memory(self, memory_node_ids: Optional[List[str]] = None) -> None:
        """Activate the node(s) named, if any. Empty input is a no-op."""
        if not memory_node_ids:
            return
        for node_id in memory_node_ids:
            self.touch_node(node_id)


__all__ = ["NexusPointerWriteback"]
