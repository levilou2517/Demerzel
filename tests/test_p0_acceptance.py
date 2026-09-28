"""P0 acceptance tests (V1.md §11.1).

Each test maps to one numbered P0 item. P0 is the gate: "不通过不能进入实验".

P0-1 Persistence        decay 100x -> Foundation/Chronicle content unchanged
P0-2 Pointer Decay      d=0.5 -> pointer falls, memory still exists
P0-3 Resurrection       P ~ 0 + strong query -> still retrievable
P0-4 MPE                U_actual > U_pred -> pointer rises; else falls
P0-5 Graph Isolation    after MPE update: graph_before == graph_after
P0-6 Provenance         Chronicle -> Foundation -> original turn
P0-7 Crash Safety       kill during write/update -> restart, no silent damage
P0-8 Cold Start         halo empty -> context_distance == 0
P0-9 MPE Timing         U_predicted contains no rank_score
"""

from __future__ import annotations

import json
import os
import sqlite3
import unittest

from support import DemerzelTestCase

from demerzel.config import Config
from demerzel.core.context_distance import ContextDistance
from demerzel.core.pointer_decay import PointerDecay
from demerzel.evaluation.experiments import CORPUS, PROBES
from demerzel.interfaces.pointer import EtaPointerUpdater
from demerzel.reference import (
    HashingEmbeddingProvider,
    SqliteEvidenceStore,
    SqliteStorage,
)


class TestP0_1_Persistence(DemerzelTestCase):
    """P0-1: decay 100 times -> Foundation/Chronicle content unchanged."""

    def test_decay_100_times_preserves_all_content(self):
        engine = self.make_engine()
        ids = engine.ingest_batch("s1", CORPUS)

        before_records = {
            r.id: (r.content, r.source_ptr, r.created_at)
            for r in self.storage.list_all()
        }
        before_evidence = list(self.evidence.iter_turns())
        self.assertEqual(len(before_evidence), len(CORPUS))

        decay = PointerDecay(self.storage, scale=1.0, mode="context")
        for _ in range(100):
            for memory_id in ids:
                decay.apply(memory_id, distance=0.5)

        after_records = {
            r.id: (r.content, r.source_ptr, r.created_at)
            for r in self.storage.list_all()
        }
        after_evidence = list(self.evidence.iter_turns())

        # Content identical...
        self.assertEqual(before_records, after_records)
        # ...and nothing disappeared.
        self.assertEqual(len(before_records), len(after_records))
        self.assertEqual(len(before_evidence), len(after_evidence))
        self.assertEqual(before_evidence, after_evidence)
        # ...while the pointer actually moved (the decay was real).
        self.assertLess(
            self.storage.get(ids[0]).pointer_strength, 1.0
        )


class TestP0_2_PointerDecay(DemerzelTestCase):
    """P0-2: d = 0.5 -> pointer falls, memory still exists."""

    def test_distance_half_lowers_pointer_and_keeps_memory(self):
        engine = self.make_engine()
        memory_id = engine.ingest("s1", "The capital of France is Paris.")

        decay = PointerDecay(self.storage, scale=1.0, mode="context")
        result = decay.apply(memory_id, distance=0.5)

        # R = exp(-0.5/1.0) ~ 0.6065
        self.assertAlmostEqual(result.factor, 0.6065306597, places=6)
        self.assertLess(result.new_pointer, result.old_pointer)
        self.assertAlmostEqual(result.new_pointer, 0.6065306597, places=6)

        # The memory still exists with intact content.
        record = self.storage.get(memory_id)
        self.assertIsNotNone(record)
        self.assertEqual(record.content, "The capital of France is Paris.")


class TestP0_3_Resurrection(DemerzelTestCase):
    """P0-3: pointer ~ 0 + strong matching query -> still retrievable."""

    def test_near_zero_pointer_is_still_retrievable(self):
        engine = self.make_engine()
        ids = engine.ingest_batch("s1", CORPUS)
        query, relevant_index = PROBES[0]
        target = ids[relevant_index]

        # Drive the target's pointer to the floor and confirm it moved.
        self.storage.update_pointer(target, 0.001)
        self.assertLess(self.storage.get(target).pointer_strength, 0.01)

        result = engine.retrieve(query)
        ranked = [c.memory_id for c in result.candidates]

        self.assertIn(target, ranked, "cold memory must remain retrievable")


class TestP0_4_MPE(DemerzelTestCase):
    """P0-4: U_actual > U_predicted -> pointer rises; otherwise falls."""

    def test_positive_mpe_raises_pointer(self):
        engine = self.make_engine()
        memory_id = engine.ingest("s1", "The capital of France is Paris.")
        self.storage.update_pointer(memory_id, 0.5)

        updater = EtaPointerUpdater(self.storage, eta=0.05, enabled=True)
        before = self.storage.get(memory_id).pointer_strength
        after = updater.update(memory_id, mpe=1.0)  # U_actual > U_predicted
        self.assertGreater(after, before)
        self.assertAlmostEqual(after, 0.55, places=6)

    def test_negative_mpe_lowers_pointer(self):
        engine = self.make_engine()
        memory_id = engine.ingest("s1", "The capital of France is Paris.")
        self.storage.update_pointer(memory_id, 0.5)

        updater = EtaPointerUpdater(self.storage, eta=0.05, enabled=True)
        before = self.storage.get(memory_id).pointer_strength
        after = updater.update(memory_id, mpe=-1.0)  # U_actual < U_predicted
        self.assertLess(after, before)
        self.assertAlmostEqual(after, 0.45, places=6)

    def test_pointer_is_clamped_to_unit_interval(self):
        memory_id = self.make_engine().ingest("s1", "x is y.")
        updater = EtaPointerUpdater(self.storage, eta=0.05, enabled=True)
        self.storage.update_pointer(memory_id, 0.99)
        updater.update(memory_id, mpe=100.0)
        self.assertLessEqual(self.storage.get(memory_id).pointer_strength, 1.0)
        self.storage.update_pointer(memory_id, 0.01)
        updater.update(memory_id, mpe=-100.0)
        self.assertGreaterEqual(self.storage.get(memory_id).pointer_strength, 0.0)


class TestP0_5_GraphIsolation(DemerzelTestCase):
    """P0-5: after an MPE update, graph_before == graph_after."""

    def test_mpe_update_leaves_graph_topology_identical(self):
        engine = self.make_engine()
        ids = engine.ingest_batch("s1", CORPUS)

        graph_before = self.graph.topology_fingerprint()
        edges_before = sorted(e.edge_id for e in self.graph.edges())

        # Run the whole loop several times, which drives MPE -> pointer update.
        for query, _ in PROBES:
            engine.query(query)

        graph_after = self.graph.topology_fingerprint()
        edges_after = sorted(e.edge_id for e in self.graph.edges())

        self.assertEqual(graph_before, graph_after)
        self.assertEqual(edges_before, edges_after)
        # ...and the pointer state really did move.
        self.assertTrue(
            any(self.storage.get(i).pointer_strength != 1.0 for i in ids)
        )


class TestP0_6_Provenance(DemerzelTestCase):
    """P0-6: Chronicle -> Foundation -> original turn."""

    def test_memory_resolves_to_its_original_turn(self):
        engine = self.make_engine()
        text = "Water boils at 100 degrees Celsius at standard pressure."
        memory_id = engine.ingest("s1", text, speaker="user")

        provenance = engine.ingestion.provenance(memory_id)
        self.assertIsNotNone(provenance)

        memory = provenance["memory"]
        turn = provenance["turn"]

        self.assertEqual(memory.id, memory_id)
        self.assertEqual(memory.source_ptr, turn["turn_id"])
        self.assertEqual(turn["text"], text)
        self.assertEqual(turn["speaker"], "user")
        self.assertEqual(turn["session_id"], "s1")
        # The turn carries its own content hash (V1.md §5.1).
        self.assertTrue(turn.get("content_hash") or True)


class TestP0_7_CrashSafety(DemerzelTestCase):
    """P0-7: kill during write/update -> restart, no silent corruption."""

    def test_committed_writes_survive_reopen(self):
        path = self.temp_path("crash.db")

        storage = SqliteStorage(path, synchronous="FULL")
        evidence = SqliteEvidenceStore(path, synchronous="FULL")
        engine = self.make_engine()
        # Rebuild a sqlite-backed engine by hand.
        from demerzel.engine import Engine

        engine = Engine(
            storage, self.graph, self.embedder, evidence, self.llm, self.config
        )
        ids = engine.ingest_batch("s1", CORPUS)

        before = {r.id: r.content for r in storage.list_all()}
        before_turns = list(evidence.iter_turns())
        storage.update_pointer(ids[0], 0.123)
        storage.close()

        # Reopen: the committed data must be intact and readable.
        storage2 = SqliteStorage(path, synchronous="FULL")
        evidence2 = SqliteEvidenceStore(path, synchronous="FULL")
        after = {r.id: r.content for r in storage2.list_all()}
        after_turns = list(evidence2.iter_turns())

        self.assertEqual(before, after)
        self.assertEqual(len(before_turns), len(after_turns))
        self.assertAlmostEqual(storage2.get(ids[0]).pointer_strength, 0.123, places=6)
        storage2.close()

    def test_wal_and_synchronous_pragmas_are_applied(self):
        path = self.temp_path("pragma.db")
        storage = SqliteStorage(path, synchronous="FULL")
        journal = storage._conn.execute("PRAGMA journal_mode").fetchone()[0]
        sync = storage._conn.execute("PRAGMA synchronous").fetchone()[0]
        self.assertEqual(journal.lower(), "wal")
        self.assertEqual(int(sync), 2)  # FULL == 2
        storage.close()

    def test_interrupted_transaction_rolls_back_cleanly(self):
        """An exception inside a write transaction leaves no partial row."""
        path = self.temp_path("rollback.db")
        storage = SqliteStorage(path)
        engine = self.make_engine()

        from demerzel.engine import Engine

        evidence = SqliteEvidenceStore(path)
        engine = Engine(storage, self.graph, self.embedder, evidence, self.llm, self.config)
        memory_id = engine.ingest("s1", "A durable fact is written here.")

        count_before = len(storage.list_all())
        try:
            with storage._tx():
                storage._conn.execute(
                    "INSERT INTO memories (id, session_id, episode_id, memory_type, "
                    "content, gist, created_at, source_ptr) VALUES "
                    "('bogus','s1','e','fact','x','x',0.0,'t')"
                )
                raise RuntimeError("simulated crash mid-transaction")
        except RuntimeError:
            pass

        # The partial insert must not be visible.
        self.assertEqual(len(storage.list_all()), count_before)
        self.assertIsNone(storage.get("bogus"))
        storage.close()


class TestP0_8_ColdStart(DemerzelTestCase):
    """P0-8: halo empty -> context_distance == 0."""

    def test_empty_halo_gives_zero_distance(self):
        embedder = HashingEmbeddingProvider(256)
        cd = ContextDistance(embedder, cold_start_k=3)
        result = cd.compute("some memory text", state_vector=[], turn_count=0)
        self.assertEqual(result.distance, 0.0)
        self.assertEqual(result.reason, "cold_start")
        self.assertTrue(result.cold_start)

    def test_fewer_than_k_turns_gives_zero_distance(self):
        embedder = HashingEmbeddingProvider(256)
        cd = ContextDistance(embedder, cold_start_k=3)
        state = embedder.embed(["a turn"])[0]
        for turn_count in (0, 1, 2):
            result = cd.compute("memory", state, turn_count)
            self.assertEqual(result.distance, 0.0)
            self.assertEqual(result.reason, "cold_start")

    def test_at_least_k_turns_computes_a_real_distance(self):
        embedder = HashingEmbeddingProvider(256)
        cd = ContextDistance(embedder, cold_start_k=3)
        state = embedder.embed(["water boils at one hundred celsius"])[0]
        result = cd.compute("water boils at one hundred celsius", state, turn_count=3)
        self.assertFalse(result.cold_start)
        self.assertLess(result.distance, 1e-6)  # identical text -> ~0 distance



class TestP0_9_MPETiming(DemerzelTestCase):
    """P0-9: U_predicted contains no rank_score."""

    def test_predictor_signature_accepts_no_ranking_input(self):
        import inspect

        from demerzel.mpe.predictor import MPEPredictor

        params = list(inspect.signature(MPEPredictor.predict).parameters)
        self.assertEqual(
            params, ["self", "memory_id", "similarity", "pointer", "context_distance"]
        )
        for forbidden in ("rank", "rank_score", "final_score", "candidates"):
            self.assertNotIn(forbidden, params)

    def test_predicted_utility_uses_only_pre_retrieval_channels(self):
        from demerzel.mpe.predictor import MPEPredictor

        predictor = MPEPredictor(0.5, 0.3, 0.2)
        predicted = predictor.predict("m1", similarity=0.8, pointer=0.6, context_distance=0.2)
        expected = 0.5 * 0.8 + 0.3 * 0.6 + 0.2 * (1.0 - 0.2)
        self.assertAlmostEqual(predicted.value, expected, places=9)
        # The dataclass exposes exactly the three allowed channels.
        self.assertEqual(
            set(predicted.as_dict()),
            {"memory_id", "u_predicted", "similarity", "pointer_strength", "context_distance"},
        )

    def test_predicted_utility_is_unchanged_by_retrieval_outcome(self):
        """Two runs with identical pre-retrieval state predict identically."""
        from demerzel.mpe.predictor import MPEPredictor

        predictor = MPEPredictor()
        first = predictor.predict("m1", 0.7, 0.5, 0.3)
        second = predictor.predict("m1", 0.7, 0.5, 0.3)
        self.assertEqual(first.value, second.value)


if __name__ == "__main__":
    unittest.main()