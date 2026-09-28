"""Nexus node and edge construction (V1.md §5.3, Phase 3).

Four node kinds: episode, fact, reflection, concept.
Five edge kinds: causal, related, contradicts, follow_up, same_goal.

Edges are created from *deterministic* signals — shared episodes, shared
vocabulary, polarity flips, explicit causal cues. No LLM is needed to build the
graph, and nothing in the MPE path may call these builders (Invariant 3).
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Sequence, Tuple

from ..interfaces.graph import (
    EDGE_CAUSAL,
    EDGE_CONTRADICTS,
    EDGE_FOLLOW_UP,
    EDGE_RELATED,
    EDGE_SAME_GOAL,
    NODE_CONCEPT,
    NODE_EPISODE,
    NODE_FACT,
    NODE_REFLECTION,
    Edge,
    GraphBackend,
    Node,
)
from ..interfaces.storage import MemoryRecord

_TOKEN_RE = re.compile(r"[a-z0-9]+|[\u4e00-\u9fff]")
_CAUSAL_CUES = (" because ", " therefore ", " thus ", " caused ", "因为", "所以", "导致")
_FOLLOW_CUES = (" then ", " after that ", " next ", "然后", "接着")
_CONTRADICT_NEG = (" not ", " no ", " never ", "不", "没", "无")


def node_id_for_memory(memory_id: str) -> str:
    """Canonical node id for a Chronicle record."""
    return f"node_{memory_id}"


def tokens(text: str) -> set:
    return set(_TOKEN_RE.findall((text or "").lower()))


class NexusBuilder:
    """Builds Nexus nodes and edges from Chronicle records.

    The builder is the only writer of graph topology. Keeping
    :class:`~demerzel.mpe.updater.MPEFeedback` off these methods is what makes
    the Experiment-2 "G fixed" constraint enforceable at the interface.
    """

    def __init__(
        self,
        graph: GraphBackend,
        related_threshold: float = 0.25,
        enabled: bool = True,
    ):
        self._graph = graph
        self.related_threshold = float(related_threshold)
        self.enabled = bool(enabled)
        self._edge_seq = 0

    def next_edge_id(self, kind: str) -> str:
        self._edge_seq += 1
        return f"edge_{kind}_{self._edge_seq:06d}"

    def add_memory_node(self, record: MemoryRecord) -> Node:
        """Create the node representing one Chronicle record."""
        node = Node(
            node_id=node_id_for_memory(record.id),
            node_type=NODE_FACT if record.memory_type == "fact" else NODE_EPISODE,
            label=record.gist or record.content[:60],
            memory_id=record.id,
            created_at=record.created_at,
        )
        self._graph.add_node(node)
        return node

    def add_episode_node(self, episode_id: str, at: float = 0.0) -> Node:
        """Create an episode hub node."""
        node = Node(
            node_id=f"node_episode_{episode_id}",
            node_type=NODE_EPISODE,
            label=episode_id,
            created_at=at,
        )
        self._graph.add_node(node)
        return node

    def add_concept_node(self, concept: str, at: float = 0.0) -> Node:
        """Create a concept hub node."""
        node = Node(
            node_id=f"node_concept_{concept}",
            node_type=NODE_CONCEPT,
            label=concept,
            created_at=at,
        )
        self._graph.add_node(node)
        return node

    def add_reflection_node(self, reflection_id: str, label: str = "", at: float = 0.0) -> Node:
        """Create a reflection node (higher-order note)."""
        node = Node(
            node_id=f"node_reflection_{reflection_id}",
            node_type=NODE_REFLECTION,
            label=label or reflection_id,
            created_at=at,
        )
        self._graph.add_node(node)
        return node

    def link_episode(self, record: MemoryRecord, at: Optional[float] = None) -> Edge:
        """Attach a memory node to its episode hub (same_goal)."""
        self.add_episode_node(record.episode_id, record.created_at)
        edge = Edge(
            edge_id=self.next_edge_id("same_goal"),
            source_id=node_id_for_memory(record.id),
            target_id=f"node_episode_{record.episode_id}",
            edge_type=EDGE_SAME_GOAL,
            weight=1.0,
            evidence_ptr=record.source_ptr,
            created_at=at if at is not None else record.created_at,
        )
        self._graph.add_edge(edge)
        return edge

    def link_related(
        self, a: MemoryRecord, b: MemoryRecord, similarity: float
    ) -> Optional[Edge]:
        """Link two records as related when vocabulary overlap clears the bar."""
        if similarity < self.related_threshold:
            return None
        edge = Edge(
            edge_id=self.next_edge_id("related"),
            source_id=node_id_for_memory(a.id),
            target_id=node_id_for_memory(b.id),
            edge_type=EDGE_RELATED,
            weight=float(similarity),
            evidence_ptr=a.source_ptr,
            created_at=max(a.created_at, b.created_at),
        )
        self._graph.add_edge(edge)
        return edge

    def link_contradicts(self, a: MemoryRecord, b: MemoryRecord, similarity: float) -> Edge:
        """Link two near-duplicate records whose polarity differs."""
        edge = Edge(
            edge_id=self.next_edge_id("contradicts"),
            source_id=node_id_for_memory(a.id),
            target_id=node_id_for_memory(b.id),
            edge_type=EDGE_CONTRADICTS,
            weight=float(similarity),
            evidence_ptr=a.source_ptr,
            created_at=max(a.created_at, b.created_at),
        )
        self._graph.add_edge(edge)
        return edge

    def link_sequence(self, a: MemoryRecord, b: MemoryRecord, kind: str) -> Edge:
        """Link consecutive turns as follow_up or causal."""
        edge = Edge(
            edge_id=self.next_edge_id(kind),
            source_id=node_id_for_memory(a.id),
            target_id=node_id_for_memory(b.id),
            edge_type=kind,
            weight=1.0,
            evidence_ptr=b.source_ptr,
            created_at=b.created_at,
        )
        self._graph.add_edge(edge)
        return edge

    # -- bulk construction ---------------------------------------------------
    def build_from_records(
        self, records: Sequence[MemoryRecord], embedder=None
    ) -> Dict[str, int]:
        """Create nodes and edges for a batch of records.

        Deterministic rules only:

        * every record gets a node and a same_goal link to its episode hub;
        * consecutive records in the same episode get a follow_up edge, or a
          causal edge when a causal cue is present;
        * record pairs whose token Jaccard overlap clears ``related_threshold``
          get a related edge, or a contradicts edge when polarity differs.
        """
        if not self.enabled:
            return {"nodes": 0, "edges": 0}

        added_nodes, added_edges = 0, 0
        for record in records:
            self.add_memory_node(record)
            added_nodes += 1
            self.link_episode(record)
            added_edges += 1

        by_episode: Dict[str, List[MemoryRecord]] = {}
        for record in records:
            by_episode.setdefault(record.episode_id, []).append(record)

        for members in by_episode.values():
            members = sorted(members, key=lambda r: r.created_at)
            for prev, cur in zip(members, members[1:]):
                lowered = f" {(prev.content or '').lower()} "
                kind = (
                    EDGE_CAUSAL
                    if any(cue in lowered for cue in _CAUSAL_CUES)
                    else EDGE_FOLLOW_UP
                )
                self.link_sequence(prev, cur, kind)
                added_edges += 1

        # Related / contradicts edges by vocabulary overlap.
        for i, a in enumerate(records):
            ta = tokens(a.content)
            if not ta:
                continue
            for b in records[i + 1 :]:
                tb = tokens(b.content)
                if not tb:
                    continue
                union = ta | tb
                jaccard = len(ta & tb) / len(union) if union else 0.0
                if jaccard < self.related_threshold:
                    continue
                neg_a = any(n in f" {(a.content or '').lower()} " for n in _CONTRADICT_NEG)
                neg_b = any(n in f" {(b.content or '').lower()} " for n in _CONTRADICT_NEG)
                if neg_a != neg_b:
                    self.link_contradicts(a, b, jaccard)
                else:
                    self.link_related(a, b, jaccard)
                added_edges += 1

        return {"nodes": added_nodes, "edges": added_edges}


__all__ = [
    "NexusBuilder",
    "node_id_for_memory",
    "tokens",
]
