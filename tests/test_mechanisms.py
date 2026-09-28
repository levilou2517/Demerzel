"""Unit tests for Nexus (Phases 3), Halo (Phase 4) and storage references."""

from __future__ import annotations

import unittest

from support import DemerzelTestCase

from demerzel.config import Config
from demerzel.halo.ring_buffer import EvictionPolicy, RingBuffer
from demerzel.nexus.nodes import NexusBuilder, node_id_for_memory
from demerzel.nexus.ppr import PPRRetriever
from demerzel.nexus.retrieval import LexicalRetriever
from demerzel.reference import (
    InMemoryStorage,
    SqliteEvidenceStore,
    SqliteStorage,
)


class TestStorageReference(DemerzelTestCase):
    def test_sqlite_roundtrip(self):
        from demerzel.engine import Engine
        from demerzel.reference import DictGraph, HashingEmbeddingProvider, MockLLMProvider

        path = self.temp_path("store.db")
        storage = SqliteStorage(path)
        evidence = SqliteEvidenceStore(path)
        engine = Engine(
            storage,
            DictGraph(),
            HashingEmbeddingProvider(256),
            evidence,
            MockLLMProvider(),
            Config(),
        )
        memory_id = engine.ingest("s1", "A durable fact.")
        self.assertEqual(len(storage.list_all()), 1)
        record = storage.get(memory_id)
        self.assertEqual(record.content, "A durable fact.")
        storage.update_pointer(memory_id, 0.42)
        self.assertAlmostEqual(storage.get(memory_id).pointer_strength, 0.42)
        # list_by_session
        session = storage.list_by_session("s1")
        self.assertEqual(len(session), 1)
        storage.close()

    def test_duplicate_turn_rejected_by_evidence_store(self):
        path = self.temp_path("evidence.db")
        store = SqliteEvidenceStore(path)
        store.append(
            {
                "turn_id": "t1",
                "session_id": "s1",
                "timestamp": 1.0,
                "speaker": "user",
                "text": "hello",
            }
        )
        with self.assertRaises(ValueError):
            store.append(
                {
                    "turn_id": "t1",
                    "session_id": "s1",
                    "timestamp": 1.0,
                    "speaker": "user",
                    "text": "different",
                }
            )
        store.close()


class TestNexus(DemerzelTestCase):
    def test_builder_creates_nodes_and_edges(self):
        engine = self.make_engine()
        records = []
        for i, turn in enumerate(
            ["Alpha causes beta.", "Beta follows gamma.", "Delta is separate."]
        ):
            engine.ingest("s1", turn)
        records = self.storage.list_all()

        builder = NexusBuilder(self.graph)
        counts = builder.build_from_records(records)
        self.assertGreaterEqual(counts["nodes"], 3)
        self.assertGreaterEqual(counts["edges"], 3)

        # Every record has a node mapping to its memory.
        for record in records:
            node = self.graph.get_node(node_id_for_memory(record.id))
            self.assertIsNotNone(node)
            self.assertEqual(node.memory_id, record.id)

    def test_ppr_retriever(self):
        engine = self.make_engine()
        records = []
        for turn in (
            "Alpha relates directly to beta.",
            "Beta connects onward to gamma.",
            "Omega is an unrelated filing topic.",
        ):
            engine.ingest("s1", turn)
        records = self.storage.list_all()

        # Build related/sequence edges so PPR has a path to propagate over.
        builder = NexusBuilder(self.graph)
        builder.build_from_records(records)

        ids = [r.id for r in records]
        seed = node_id_for_memory(ids[0])
        retriever = PPRRetriever(self.graph, damping=0.85)
        scores = retriever.memory_scores([seed])
        self.assertIn(ids[0], scores)
        # Seeded memory scores above the far, unrelated one.
        self.assertGreater(scores[ids[0]], scores.get(ids[2], 0.0))
        # A near neighbor with shared vocabulary outranks the unrelated topic.
        self.assertGreater(scores.get(ids[1], 0.0), scores.get(ids[2], 0.0))

    def test_lexical_bm25(self):
        engine = self.make_engine()
        ids = engine.ingest_batch("s1", [
            {"text": "Water boils at a hundred degrees."},
            {"text": "Bicycles have two pedals."},
        ])
        lexical = LexicalRetriever()
        scores = lexical.score("what boils water?", self.storage.list_all())
        self.assertGreater(scores[ids[0]], scores[ids[1]])

    def test_graph_disabled_returns_empty_ppr(self):
        engine = self.make_engine()
        engine.ingest_batch("s1", [{"text": "Alpha beta gamma."}])
        retriever = PPRRetriever(self.graph, enabled=False)
        self.assertEqual(retriever.score([node_id_for_memory("x")]), {})


class TestHalo(DemerzelTestCase):
    def test_ring_buffer_eviction(self):
        ring = RingBuffer(max_turns=3)
        ring.append("one")
        ring.append("two")
        ring.append("three")
        ring.append("four")  # causes 'one' to evict
        self.assertEqual(ring.size, 3)
        self.assertEqual([t.text for t in ring.all()], ["two", "three", "four"])
        self.assertEqual([t.text for t in ring.evicted], ["one"])

    def test_policies(self):
        from demerzel.halo.ring_buffer import HaloTurn

        ring = RingBuffer(max_turns=2)
        policy = EvictionPolicy("evict")
        for text in ("a", "b", "c"):
            policy.apply(ring, HaloTurn(index=0, text=text))
        self.assertEqual(ring.size, 2)
        self.assertEqual([t.text for t in ring.evicted], ["a"])

    def test_snapshot_is_serialisable(self):
        ring = RingBuffer(max_turns=4)
        ring.append("hello", speaker="user", metadata={"k": 1})
        snap = ring.snapshot()
        self.assertTrue(isinstance(snap, list))
        self.assertEqual(snap[0]["text"], "hello")
        self.assertEqual(snap[0]["metadata"]["k"], 1)


if __name__ == "__main__":
    unittest.main()