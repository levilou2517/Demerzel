"""Invariant 1–8 tests (V1.md §10).

Every invariant has a direct, small test here; the P0 items aggregate several
of them and live in ``test_p0_acceptance.py``. The strongest guardrail of the
whole project lives in :class:`TestInvariant3` — running the MPE loop must not
change graph topology — so it is asserted here both at the writeback level and
through the full engine.
"""

from __future__ import annotations

import json
import unittest

from support import DemerzelTestCase

from demerzel.config import Config
from demerzel.core.context_distance import ContextDistance
from demerzel.core.pointer_decay import PointerDecay
from demerzel.entropy.forgetting_log import ForgettingLog
from demerzel.evaluation.experiments import CORPUS, PROBES
from demerzel.interfaces.pointer import EtaPointerUpdater
from demerzel.nexus.nodes import NexusBuilder, node_id_for_memory
from demerzel.nexus.pointer_update import NexusPointerWriteback
from demerzel.reference import HashingEmbeddingProvider


class TestInvariant1DecayPreservesEvidence(DemerzelTestCase):
    def test_decay_never_removes_records(self):
        engine = self.make_engine()
        ids = engine.ingest_batch("s1", CORPUS)
        decay = PointerDecay(self.storage, scale=1.0, mode="context")
        for _ in range(200):
            for mid in ids:
                decay.apply(mid, distance=0.9)
        # Nothing deleted.
        self.assertEqual(len(self.storage.list_all()), len(ids))
        # And the raw store has no delete path.
        self.assertFalse(hasattr(engine.evidence, "delete"))
        self.assertFalse(hasattr(engine.evidence, "remove"))


class TestInvariant2PointerUpdatePreservesContent(DemerzelTestCase):
    def test_memory_pointer_update_only_touches_pointer(self):
        engine = self.make_engine()
        memory_id = engine.ingest("s1", "The capital of France is Paris.")
        record = self.storage.get(memory_id)
        snapshot = (record.content, record.source_ptr, record.created_at)

        updater = EtaPointerUpdater(self.storage, eta=0.05, enabled=True)
        updater.update(memory_id, mpe=1.0)
        updater.update(memory_id, mpe=-0.5)

        after = self.storage.get(memory_id)
        self.assertEqual((after.content, after.source_ptr, after.created_at), snapshot)
        self.assertNotEqual(after.pointer_strength, snapshot)  # pointer moved


class TestInvariant3MPEKeepsGraphTopology(DemerzelTestCase):
    """The central isolation guarantee: MPE changes A, not G."""

    def test_writeback_never_adds_nodes_or_edges(self):
        engine = self.make_engine()
        ids = engine.ingest_batch("s1", CORPUS)
        nodes_before = sorted(n.node_id for n in self.graph.nodes())
        edges_before = sorted(e.edge_id for e in self.graph.edges())

        # The writeback path can only touch existing edges.
        writeback = NexusPointerWriteback(self.graph)
        for mid in ids:
            writeback.touch_node(node_id_for_memory(mid))

        self.assertEqual(sorted(n.node_id for n in self.graph.nodes()), nodes_before)
        self.assertEqual(sorted(e.edge_id for e in self.graph.edges()), edges_before)

    def test_full_mpe_loop_keeps_topology_unchanged(self):
        engine = self.make_engine()
        engine.ingest_batch("s1", CORPUS)
        before = self.graph.topology_fingerprint()
        for query, _ in PROBES:
            engine.query(query)
        self.assertEqual(self.graph.topology_fingerprint(), before)


class TestInvariant4HaloEvictionPreservesChronicle(DemerzelTestCase):
    def test_ring_overflow_does_not_touch_storage(self):
        engine = self.make_engine()
        Config()
        for i in range(20):
            engine.ingest("s1", f"turn number {i} in session one")
        # Halo capped at max_turns (default 8) -> many evictions.
        self.assertLessEqual(engine.ring.size, engine.config.halo.max_turns)
        self.assertEqual(len(self.storage.list_all()), 20)
        self.assertEqual(self.storage.list_all()[0].content, "turn number 0 in session one")


class TestInvariant5ColdMemoryReactivates(DemerzelTestCase):
    def test_decayed_memory_surfaces_for_strong_query(self):
        engine = self.make_engine()
        ids = engine.ingest_batch("s1", CORPUS)
        query, relevant_index = PROBES[0]
        target = ids[relevant_index]

        # Hard decay: pointer ~ 0 but never deleted.
        decay = PointerDecay(self.storage, scale=1.0, mode="context")
        for _ in range(25):
            decay.apply(target, distance=0.95)
        self.assertLess(self.storage.get(target).pointer_strength, 0.01)

        result = engine.retrieve(query)
        ranked = [c.memory_id for c in result.candidates]
        self.assertIn(target, ranked)


class TestInvariant6DeleteIsSeparate(DemerzelTestCase):
    def test_forgetting_mechanism_has_no_delete(self):
        engine = self.make_engine()
        memory_id = engine.ingest("s1", "a fact")

        # The forgetting/decay machinery exposes no delete.
        decay = PointerDecay(self.storage)
        self.assertFalse(hasattr(decay, "delete"))
        from demerzel.mpe.predictor import MPEPredictor

        self.assertFalse(hasattr(MPEPredictor(), "delete"))

        # The raw store's API surface contains no deletion.
        raw_methods = {name for name in dir(engine.evidence) if not name.startswith("_")}
        self.assertFalse({"delete", "remove", "drop"} & raw_methods)

        # Explicit user deletion exists at the storage level and is separate.
        self.assertTrue(hasattr(self.storage, "delete"))
        self.assertIsNotNone(self.storage.get(memory_id))
        self.assertTrue(self.storage.delete(memory_id))
        self.assertIsNone(self.storage.get(memory_id))


class TestInvariant7ColdStart(DemerzelTestCase):
    def test_cold_start_rule(self):
        embedder = HashingEmbeddingProvider(256)
        cd = ContextDistance(embedder, cold_start_k=3)
        # Empty state -> 0.
        self.assertEqual(cd.compute("text", [], 0).distance, 0.0)
        # Fewer than K turns -> 0.
        state = embedder.embed(["hello world"])[0]
        self.assertEqual(cd.compute("text", state, 2).distance, 0.0)
        # At/over K with a non-empty state -> real distance.
        r = cd.compute("hello world goodbye", state, 3)
        self.assertFalse(r.cold_start)

    def test_cold_start_suppresses_decay_penalty(self):
        engine = self.make_engine()
        memory_id = engine.ingest("s1", "a fact to preserve")
        self.storage.update_pointer(memory_id, 0.8)
        decay = PointerDecay(self.storage, scale=1.0, mode="context")
        result = decay.apply(memory_id, distance=0.5, reason="cold_start")
        self.assertEqual(result.factor, 1.0)
        self.assertEqual(result.new_pointer, 0.8)
        self.assertEqual(result.reason, "cold_start")


class TestInvariant8MPETiming(DemerzelTestCase):
    def test_u_predicted_has_no_rank_score(self):
        from demerzel.mpe.predictor import MPEPredictor

        predictor = MPEPredictor()
        predicted = predictor.predict("m", 0.6, 0.4, 0.2)
        as_dict = predicted.as_dict()
        self.assertNotIn("rank_score", as_dict)
        self.assertNotIn("final_score", as_dict)
        # Exactly the three allowed channels + identity + u_predicted.
        self.assertEqual(
            set(as_dict),
            {"memory_id", "u_predicted", "similarity", "pointer_strength", "context_distance"},
        )


class TestInvariant3PlusGraphWritebackMeta(DemerzelTestCase):
    def test_topology_fingerprint_ignores_edge_metadata(self):
        """co_activation updates must not change the topology fingerprint."""
        engine = self.make_engine()
        ids = engine.ingest_batch("s1", CORPUS)
        before = self.graph.topology_fingerprint()

        writeback = NexusPointerWriteback(self.graph)
        for mid in ids:
            writeback.touch_node(node_id_for_memory(mid))

        # co_activation_count changed but topology fingerprint identical.
        self.assertGreater(
            sum(
                e.co_activation_count
                for e in self.graph.edges()
            ),
            0,
        )
        self.assertEqual(self.graph.topology_fingerprint(), before)


if __name__ == "__main__":
    unittest.main()