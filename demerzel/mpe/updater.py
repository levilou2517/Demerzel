"""MPE updater and feedback loop (V1.md §7.4, §8.2, Phase 6).

The updater is the ONLY consumer of :class:`~demerzel.interfaces.pointer.PointerUpdater`
inside the MPE path. By contract:

* it depends only on ``PointerUpdater`` — which depends only on ``StorageBackend``;
* it never calls :meth:`~demerzel.interfaces.graph.GraphBackend.add_edge` or
  ``add_node``. **This is the enforcement point of Invariant 3 / P0-5**: under
  MPE (Experiment 2) the graph topology stays fixed while the pointer state A
  evolves, so ``graph_before == graph_after`` holds.
* the pointer mutation touches only ``pointer_strength`` (Invariant 2).

The whole feedback closed loop —
``retrieve -> evaluate -> update -> retrieve again`` (V1.md §7.4) — is exposed
as :meth:`MPEFeedback.step` for the MVP harness, while the low-level primitive
:meth:`MPEFeedback.apply` is the unit under test.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ..interfaces.pointer import EtaPointerUpdater, PointerUpdate, PointerUpdater
from .evaluator import MPEEvaluator, MPEResult
from .predictor import MPEPredictor, PredictedUtility
from .utility import ActualUtility, UtilityEvaluator


@dataclass
class MPEFeedback:
    """The accessibility-learning core of Demerzel.

    ``enabled=False`` is the Experiment-2 control group: applying MPE through a
    disabled updater leaves the pointer unchanged (an update call returns the
    same value and logs ``mpe_disabled``), making the ablation a pure config
    difference, not a code path.
    """

    updater: PointerUpdater
    predictor: MPEPredictor = field(default_factory=MPEPredictor)
    evaluator: UtilityEvaluator = field(default_factory=UtilityEvaluator)
    mpe_evaluator: MPEEvaluator = field(default_factory=MPEEvaluator)
    enabled: bool = True

    @property
    def eta(self) -> float:
        """Expose the learning rate for reporting."""
        return getattr(self.updater, "eta", 0.0)

    def apply(self, memory_id: str, mpe_value: float) -> float:
        """Persist one clipped MPE step into the pointer. Returns new pointer."""
        result = self.mpe_evaluator.evaluate(memory_id, 0.0, mpe_value)
        new = self.updater.update(memory_id, result.clipped)
        return new

    def step(
        self,
        memory_id: str,
        similarity: float,
        pointer: float,
        context_distance: float,
        evidence_score: float,
        answer_delta: float = 0.0,
        retrieval_delta: float = 0.0,
        token_cost: float = 0.0,
    ) -> Optional[MPEResult]:
        """Full update: predict, evaluate, delta, persist.

        Returns the ::class:`MPEResult` (or ``None`` when the memory is absent
        in the updater's storage). This is the `retrieve -> evaluate -> update`
        step, minus the retrieval itself.
        """
        predicted = self.predictor.predict(
            memory_id, similarity, pointer, context_distance
        )
        actual = self.evaluator.evaluate(
            evidence_score,
            answer_delta,
            retrieval_delta,
            token_cost,
            memory_id=memory_id,
        )
        result = self.mpe_evaluator.evaluate(memory_id, predicted.value, actual.value)
        self.updater.update(memory_id, result.clipped)
        return result

    @property
    def pointer_log(self) -> List[PointerUpdate]:
        """The pending pointer updates issued by the wrapped updater."""
        return getattr(self.updater, "log", [])


def build_default_mpfe(
    storage,
    eta: float = 0.05,
    enabled: bool = True,
    evidence_weight: float = 1.0,
    mpe_clip: float = 1.0,
) -> MPEFeedback:
    """Convenience factory wiring the stock predictor/evaluator/updater."""
    return MPEFeedback(
        updater=EtaPointerUpdater(storage, eta=eta, enabled=enabled),
        predictor=MPEPredictor(),
        evaluator=UtilityEvaluator(evidence_weight=evidence_weight),
        mpe_evaluator=MPEEvaluator(mpe_clip=mpe_clip),
        enabled=enabled,
    )


__all__ = [
    "MPEFeedback",
    "build_default_mpfe",
    "ActualUtility",
    "MPEEvaluator",
    "MPEPredictor",
    "MPEResult",
    "PredictedUtility",
    "PointerUpdate",
]