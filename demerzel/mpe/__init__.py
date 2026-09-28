"""MPE: the accessibility-learning core (V1.md §7.3–§7.4, Phase 6).

``MPE = U_actual - U_predicted``; ``P_next = clip(P + eta*MPE, 0, 1)``.

This is Demerzel's central scientific module: it is the mechanism that lets
accessibility state ``A`` be learned while evidence ``E`` and graph topology
``G`` stay fixed (Experiment 2).
"""

from .evaluator import MPEEvaluator, MPEResult
from .predictor import MPEPredictor, PredictedUtility
from .updater import MPEFeedback, build_default_mpfe
from .utility import ActualUtility, UtilityEvaluator

__all__ = [
    "ActualUtility",
    "MPEEvaluator",
    "MPEFeedback",
    "MPEPredictor",
    "MPEResult",
    "PredictedUtility",
    "UtilityEvaluator",
    "build_default_mpfe",
]