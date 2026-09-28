"""Unit tests for the core accessibility mechanisms (Phases 5–6)."""

from __future__ import annotations

import unittest

from support import DemerzelTestCase

from demerzel.config import Config
from demerzel.core.context_distance import ContextDistance
from demerzel.core.pointer_decay import PointerDecay
from demerzel.mpe.evaluator import MPEEvaluator
from demerzel.mpe.predictor import MPEPredictor
from demerzel.mpe.updater import MPEFeedback
from demerzel.mpe.utility import UtilityEvaluator
from demerzel.reference import HashingEmbeddingProvider


class TestContextDistance(DemerzelTestCase):
    def setUp(self):
        super().setUp()
        self.embedder = HashingEmbeddingProvider(256)
        self.cd = ContextDistance(self.embedder, cold_start_k=3)

    def test_identical_text_has_near_zero_distance(self):
        text = "water boils at one hundred celsius"
        state = self.embedder.embed([text])[0]
        result = self.cd.compute(text, state, turn_count=5)
        self.assertFalse(result.cold_start)
        self.assertLess(abs(result.distance), 1e-6)

    def test_orthogonal_text_is_distant(self):
        state = self.embedder.embed(["water boils at one hundred celsius"])[0]
        result = self.cd.compute("zephyr orbits the distant nebula", state, turn_count=5)
        self.assertFalse(result.cold_start)
        self.assertGreater(result.distance, 0.3)

    def test_distance_is_clamped_to_unit_interval(self):
        state = self.embedder.embed(["alpha beta gamma delta"])[0]
        for _ in range(5):
            result = self.cd.compute("omega omega omega", state, turn_count=5)
            self.assertTrue(0.0 <= result.distance <= 1.0)


class TestPointerDecay(DemerzelTestCase):
    def test_modes_match_expectation(self):
        engine = self.make_engine()
        memory_id = engine.ingest("s1", "the northern star is bright at night")

        # none: no decay.
        none = PointerDecay(self.storage, mode="none")
        self.assertEqual(none.factor(0.9), 1.0)

        # boundary: distance >= threshold -> full decay {0,1} style.
        boundary = PointerDecay(self.storage, mode="boundary", boundary_threshold=0.5)
        self.assertAlmostEqual(boundary.factor(0.6), 1.0 / 2.718281828, places=5)
        self.assertEqual(boundary.factor(0.4), 1.0)

        # context: continuous.
        context = PointerDecay(self.storage, mode="context", scale=1.0)
        self.assertAlmostEqual(context.factor(1.0), 1.0 / 2.718281828, places=5)
        self.assertGreater(context.factor(0.1), context.factor(0.9))

        # enabled=False always returns 1.0.
        off = PointerDecay(self.storage, enabled=False)
        self.assertEqual(off.factor(1.0), 1.0)

    def test_cold_start_reason_does_not_decay(self):
        decay = PointerDecay(self.make_engine().storage, mode="context")
        self.assertEqual(decay.factor(0.9, reason="cold_start"), 1.0)


class TestMPEFeedback(DemerzelTestCase):
    def test_positive_evidence_raises_pointer(self):
        engine = self.make_engine()
        memory_id = engine.ingest("s1", "The capital of France is Paris.")
        # Start with a neutral pointer and give strong positive evidence.
        self.storage.update_pointer(memory_id, 0.5)

        mpfe = MPEFeedback(updater=self.new_updater(), enabled=True)
        result = mpfe.step(
            memory_id,
            similarity=0.9,
            pointer=0.5,
            context_distance=0.0,
            evidence_score=0.95,
        )
        self.assertIsNotNone(result)
        self.assertGreater(
            self.storage.get(memory_id).pointer_strength, 0.5, "positive MPE must raise pointer"
        )

    def test_disabled_mpe_leaves_pointer_unchanged(self):
        engine = self.make_engine()
        memory_id = engine.ingest("s1", "The capital of France is Paris.")
        self.storage.update_pointer(memory_id, 0.5)

        from demerzel.interfaces.pointer import EtaPointerUpdater

        mpfe = MPEFeedback(
            updater=EtaPointerUpdater(self.storage, eta=0.05, enabled=False), enabled=False
        )
        before = self.storage.get(memory_id).pointer_strength
        mpfe.step(
            memory_id,
            similarity=0.9,
            pointer=before,
            context_distance=0.0,
            evidence_score=1.0,
        )
        self.assertAlmostEqual(self.storage.get(memory_id).pointer_strength, before, places=9)

    def new_updater(self):
        from demerzel.interfaces.pointer import EtaPointerUpdater

        return EtaPointerUpdater(self.storage, eta=0.05, enabled=True)

    def test_mpe_evaluator_sign_and_bound(self):
        ev = MPEEvaluator(mpe_clip=1.0)
        positive = ev.evaluate("m", u_predicted=0.2, u_actual=0.8)
        self.assertTrue(positive.positive)
        self.assertAlmostEqual(positive.clipped, 0.6)

        negative = ev.evaluate("m", u_predicted=0.8, u_actual=0.2)
        self.assertFalse(negative.positive)
        self.assertAlmostEqual(negative.clipped, -0.6)

        # Bound applies.
        big = ev.evaluate("m", u_predicted=0.0, u_actual=5.0)
        self.assertEqual(big.clipped, 1.0)


class TestUtilityEvaluator(DemerzelTestCase):
    def test_default_weights_are_evidence_only(self):
        ev = UtilityEvaluator()
        actual = ev.evaluate(evidence_score=0.7, answer_delta=1.0, token_cost=10.0)
        # With answer_delta_weight=0 and token_cost_weight=0 this equals evidence only.
        self.assertAlmostEqual(actual.value, 0.7)
        self.assertEqual((ev.answer_delta_weight, ev.token_cost_weight), (0.0, 0.0))


if __name__ == "__main__":
    unittest.main()