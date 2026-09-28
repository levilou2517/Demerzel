"""Evaluation: metrics and experiment adapters (V1.md §4.1, §8, Phase 8).

Demerzel does not embed an evaluation framework (V1.md §5.5); this package
computes the metrics of §15 over Demerzel's own trace format and provides the
adapters that external benchmarks plug into.
"""

from . import metrics
from .experiments import (
    CORPUS,
    EXPERIMENTS,
    PROBES,
    experiment_1,
    experiment_2,
    experiment_3,
    experiment_4,
    run_experiment,
)

__all__ = [
    "CORPUS",
    "EXPERIMENTS",
    "PROBES",
    "experiment_1",
    "experiment_2",
    "experiment_3",
    "experiment_4",
    "metrics",
    "run_experiment",
]