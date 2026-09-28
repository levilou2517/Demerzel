"""Invariant 7 / P0-8: the context-distance cold-start rule.

This is the Phase-0 acceptance gate named in V1.md §12 ("验收: pytest 通过,
包括 Invariant 7"). The rule under test is specified in ``docs/cold_start.md``.
"""

from __future__ import annotations

import unittest

from support import DemerzelTestCase

from demerzel.config import Config
from demerzel.core.context_distance import ContextDistance
from demerzel.entropy.context_decay import ContextDecay
from demerzel.entropy.forgetting_log import ForgettingLog
from demerzel.reference import HashingEmbeddingProvider


class TestColdStartRule(DemerzelTestCase):
    """docs/cold_start.md: empty halo / few turns -> d = 0, no penalty."""

    def setUp(self):
        super().setUp()
        self.embedder = HashingEmbeddingProvider(256)
        self.cd = ContextDistance(self.embedder, cold_start_k=3)

    def test_empty_state_vector_declares_cold_start(self):
        self.assertTrue(self.cd.is_cold_start([], 0))
        self.assertTrue(self.cd.is_cold_start([], 99))

    def test_turn_count_below_k_declares_cold_start(self):
        state = self.embedder.embed(["a turn"])[0]
        self.assertTrue(self.cd.is_cold_start(state, 0))
        self.assertTrue(self.cd.is_cold_start(state, 1))
        self.assertTrue(self.cd.is_cold_start(state, 2))
        self.assertFalse(self.cd.is_cold_start(state, 3))

    def test_cold_start_distance_is_exactly_zero(self):
        result = self.cd.compute("any memory", [], 0)
        self.assertEqual(result.distance, 0.0)
        self.assertEqual(result.reason, "cold_start")
        self.assertTrue(result.cold_start)

    def test_k_is_configurable(self):
        cd = ContextDistance(self.embedder, cold_start_k=5)
        state = self.embedder.embed(["a turn"])[0]
        self.assertTrue(cd.is_cold_start(state, 4))
        self.assertFalse(cd.is_cold_start(state, 5))

    def test_rule_can_be_disabled_for_control_runs(self):
        cd = ContextDistance(self.embedder, cold_start_k=3, cold_start_enabled=False)
        self.assertFalse(cd.is_cold_start([], 0))
        state = self.embedder.embed(["a turn"])[0]
        result = cd.compute("a turn", state, 0)
        self.assertFalse(result.cold_start)

    def test_decay_cycle_records_cold_start_and_applies_no_penalty(self):
        engine = self.make_engine()
        ids = engine.ingest_batch(
            "s1", [{"text": "The first fact."}, {"text": "The second fact."}]
        )
        log = ForgettingLog()
        decay = ContextDecay(
            self.storage, self.cd, engine.pointer_decay, log, enabled=True
        )
        # Two turns < K=3 -> cold start.
        outcome = decay.run(self.storage.list_all(), state_vector=[], turn_count=2)

        self.assertTrue(outcome.cold_start)
        for memory_id in ids:
            self.assertEqual(outcome.distances[memory_id], 0.0)
            self.assertEqual(outcome.reasons[memory_id], "cold_start")
            # No decay penalty: the pointer is untouched.
            self.assertEqual(self.storage.get(memory_id).pointer_strength, 1.0)

    def test_engine_uses_the_configured_k(self):
        config = Config()
        config.context_distance.cold_start_k = 1
        engine = self.make_engine(config)
        self.assertEqual(engine.context_distance.cold_start_k, 1)


if __name__ == "__main__":
    unittest.main()