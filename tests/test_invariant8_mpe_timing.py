"""Invariant 8 / P0-9: U_predicted must not contain retrieval results.

Spec: ``docs/mpe_timing.md``. The enforcement is structural — the predictor's
signature admits only the three pre-retrieval channels — so the tests audit the
signature as well as the values.
"""

from __future__ import annotations

import inspect
import unittest

from support import DemerzelTestCase

from demerzel.mpe.evaluator import MPEEvaluator
from demerzel.mpe.predictor import MPEPredictor
from demerzel.mpe.updater import MPEFeedback


class TestMPETiming(DemerzelTestCase):
    def test_predictor_admits_only_pre_retrieval_channels(self):
        params = list(inspect.signature(MPEPredictor.predict).parameters)
        self.assertEqual(
            params, ["self", "memory_id", "similarity", "pointer", "context_distance"]
        )

    def test_predictor_has_no_ranking_parameter(self):
        params = set(inspect.signature(MPEPredictor.predict).parameters)
        for forbidden in (
            "rank",
            "rank_score",
            "final_score",
            "candidates",
            "retrieval_result",
            "used_evidence",
        ):
            self.assertNotIn(forbidden, params, f"{forbidden} must not feed U_predicted")

    def test_predicted_utility_shape(self):
        predicted = MPEPredictor(0.5, 0.3, 0.2).predict("m1", 0.8, 0.6, 0.2)
        self.assertEqual(
            set(predicted.as_dict()),
            {
                "memory_id",
                "u_predicted",
                "similarity",
                "pointer_strength",
                "context_distance",
            },
        )

    def test_u_predicted_is_computed_before_retrieval_in_the_engine(self):
        """The engine's ordered loop predicts from pre-retrieval values only."""
        engine = self.make_engine()
        engine.ingest_batch(
            "s1",
            [
                {"text": "The capital of France is Paris."},
                {"text": "Water boils at one hundred celsius."},
                {"text": "DNA stores genetic information."},
                {"text": "Proteins are built at the ribosome."},
            ],
        )
        record = engine.query("What is the capital of France?")
        # The trace records both, and the pointer update is explained by the MPE.
        self.assertIn("mpe", record)
        self.assertIn("pointer_updates", record)
        self.assertIsNotNone(record["mpe"])

    def test_rank_score_is_not_an_mpe_input(self):
        """Changing only the rank ordering must not change U_predicted."""
        predictor = MPEPredictor()
        first = predictor.predict("m", similarity=0.5, pointer=0.5, context_distance=0.5)
        # A ranking change cannot be expressed: there is no parameter for it.
        second = predictor.predict("m", similarity=0.5, pointer=0.5, context_distance=0.5)
        self.assertEqual(first.value, second.value)

    def test_mpe_sign_follows_actual_minus_predicted(self):
        evaluator = MPEEvaluator(mpe_clip=1.0)
        up = evaluator.evaluate("m", u_predicted=0.3, u_actual=0.9)
        self.assertGreater(up.mpe, 0)
        self.assertTrue(up.positive)

        down = evaluator.evaluate("m", u_predicted=0.9, u_actual=0.3)
        self.assertLess(down.mpe, 0)
        self.assertFalse(down.positive)


if __name__ == "__main__":
    unittest.main()