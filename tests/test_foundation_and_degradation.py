"""Foundation + degradation tests (Phase 1, V1.md §17).

The degradation cases are the ones that matter operationally: an unavailable
LLM must never cause memory loss, and the three arbitration gates must degrade
in their specified directions.
"""

from __future__ import annotations

import os
import unittest

from support import DemerzelTestCase, count_lines

from demerzel.config import Config
from demerzel.foundation.event_boundary import (
    AlgorithmicEventBoundary,
    LLMEventBoundary,
)
from demerzel.foundation.gist import AlgorithmicGist, LLMGist
from demerzel.foundation.raw_store import RawStore, Turn
from demerzel.reference import (
    InMemoryEvidenceStore,
    MockLLMProvider,
    MockLLMProviderUnavailable,
)
from demerzel.trace import RunTrace, read_jsonl


class TestFoundation(DemerzelTestCase):
    def setUp(self):
        super().setUp()
        self.evidence = InMemoryEvidenceStore()

    def test_append_only_and_hash(self):
        store = RawStore(self.evidence)
        turn = Turn(
            turn_id="t1",
            session_id="s1",
            timestamp=1.0,
            speaker="user",
            text="hello",
            content_hash="abc",
        )
        store.append(turn)
        # Same turn_id again is rejected, never overwritten.
        with self.assertRaises(ValueError):
            store.append(turn)
        self.assertEqual(store.count(), 1)
        self.assertEqual(store.get("t1")["text"], "hello")

    def test_ingestion_writes_raw_before_derived(self):
        engine = self.make_engine()
        memory_id = engine.ingest("s1", "The capital of France is Paris.")
        # The raw turn exists (through the engine's RawStore wrapper)...
        self.assertEqual(engine.evidence.count(), 1)
        # ...and the derived record points at it.
        record = self.storage.get(memory_id)
        self.assertIsNotNone(engine.evidence.get(record.source_ptr))

    def test_gist_is_deterministic(self):
        gist = AlgorithmicGist()
        text = "Alpha is important. Beta is important. Gamma is rare."
        self.assertEqual(gist.extract(text), gist.extract(text))
        self.assertLessEqual(len(gist.extract(text)), 200)


class TestDegradation(DemerzelTestCase):
    """V1.md §17: unavailable LLM must not cause memory loss or wrong gates."""

    def test_write_gate_fails_open(self):
        """gist failure still writes the turn and marks gate=unavailable."""
        from demerzel.foundation.ingestion import IngestionPipeline

        class ExplodingGist:
            def extract(self, text):
                raise RuntimeError("gist service down")

        engine = self.make_engine()
        pipeline = IngestionPipeline(RawStore(self.evidence), self.storage, ExplodingGist())
        result = pipeline.ingest_turn("s1", "An important fact survives.")

        self.assertEqual(result.gate, "unavailable")
        self.assertTrue(result.degraded)
        # The evidence and the record both exist (fail-open).
        self.assertEqual(engine.evidence.count(), 1)
        self.assertEqual(len(self.storage.list_all()), 1)

    def test_llm_unavailable_still_stores_memory(self):
        engine = self.make_engine(llm=MockLLMProviderUnavailable())
        memory_id = engine.ingest("s1", "Memory must survive an LLM outage.")
        self.assertIsNotNone(self.storage.get(memory_id))
        self.assertEqual(engine.evidence.count(), 1)

    def test_event_boundary_falls_back_when_llm_unavailable(self):
        provider = MockLLMProviderUnavailable()
        boundary = LLMEventBoundary(provider)
        decisions = boundary.detect(["first turn", "second unrelated turn"])
        self.assertEqual(len(decisions), 2)
        self.assertTrue(all(d.method == "algorithmic_fallback" for d in decisions))
        self.assertTrue(all(d.degraded for d in decisions))
        self.assertTrue(decisions[0].is_boundary)

    def test_algorithmic_boundary_detects_topic_shift(self):
        boundary = AlgorithmicEventBoundary(threshold=0.15)
        decisions = boundary.detect(
            [
                "water boils at one hundred celsius",
                "water boils when heated enough",
                "zebra migration crosses the serengeti plains",
            ]
        )
        self.assertTrue(decisions[0].is_boundary)
        self.assertFalse(decisions[1].is_boundary)
        self.assertTrue(decisions[2].is_boundary)

    def test_answerer_degrades_to_extractive(self):
        from demerzel.assemblage.answerer import Answerer

        answerer = Answerer(MockLLMProviderUnavailable(), mode="llm")
        answer = answerer.answer("what?", "Query: what?\n- [m1] (score 0.9): a fact", ["m1"])
        self.assertTrue(answer.degraded)
        self.assertEqual(answer.used_evidence, ["m1"])
        self.assertIn("a fact", answer.text)


class TestTrace(DemerzelTestCase):
    def test_trace_files_are_written(self):
        run_dir = self.temp_path("trace_run")
        trace = RunTrace(run_dir, "run_test", "abc123")
        engine = self.make_engine()
        engine.run_trace = trace

        engine.ingest_batch("s1", [{"text": "Alpha causes beta."}, {"text": "Beta follows."}])
        engine.query("What causes beta?")
        trace.write_metadata({"seed": 0})
        trace.write_metrics({"recall_at_5": 1.0})
        trace.close()

        retrieval_path = os.path.join(run_dir, "retrieval.jsonl")
        self.assertTrue(os.path.exists(retrieval_path))
        self.assertEqual(count_lines(retrieval_path), 1)

        rows = read_jsonl(retrieval_path)
        row = rows[0]
        # The V1.md §5.5 schema keys are present.
        for key in ("run_id", "query", "candidates", "used_evidence", "mpe", "pointer_updates"):
            self.assertIn(key, row)
        if row["candidates"]:
            candidate = row["candidates"][0]
            for key in (
                "memory_id",
                "semantic_score",
                "ppr_score",
                "pointer_strength",
                "context_distance",
                "final_score",
            ):
                self.assertIn(key, candidate)

        for name in ("metadata.json", "metrics.json"):
            self.assertTrue(os.path.exists(os.path.join(run_dir, name)))


if __name__ == "__main__":
    unittest.main()