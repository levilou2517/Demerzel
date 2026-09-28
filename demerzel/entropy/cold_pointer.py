"""Cold pointer revival (Experiment 3, P0-3; V1.md §8.3, §11.1).

Soft forgetting keeps a pointer low but never zeroes accessibility entirely
(decay multiplies by ``exp(-d/S) > 0`` for finite distances, and cold start
adds no penalty). A cold pointer is therefore *revivable*: a strong matching
query over a low pointer must still surface the memory. This module measures
that — it is the recovery-rate / recovery-latency / false-recall instrumentation
that Experiment 3 and P0-3 consume.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from ..interfaces.storage import MemoryRecord, StorageBackend


@dataclass
class RecoveryMeasurement:
    """What one revival probe found about a target memory."""

    memory_id: str
    target_present: bool
    recovered: bool
    rank_if_found: Optional[int]
    pointer_at_probe: float
    query: str


@dataclass
class RecoveryReport:
    """Aggregate over many probes."""

    probes: List[RecoveryMeasurement] = field(default_factory=list)

    def add(self, measurement: RecoveryMeasurement) -> None:
        self.probes.append(measurement)

    @property
    def recovery_rate(self) -> float:
        if not self.probes:
            return 0.0
        present = [p for p in self.probes if p.target_present]
        if not present:
            return 0.0
        return sum(1 for p in present if p.recovered) / len(present)

    @property
    def false_recall_rate(self) -> float:
        """Frac of probes where an *absent* target was (wrongly) recalled."""
        if not self.probes:
            return 0.0
        absent = [p for p in self.probes if not p.target_present]
        if not absent:
            return 0.0
        return sum(1 for p in absent if p.recovered) / len(absent)

    def as_dict(self) -> dict:
        return {
            "probe_count": len(self.probes),
            "recovery_rate": round(self.recovery_rate, 6),
            "false_recall_rate": round(self.false_recall_rate, 6),
        }


class ColdPointerProbe:
    """Revival measurement given a callable that ranks memories for a query.

    The ranking callable (normally the composed retrieval step) is injected so
    this module carries no storage or retrieval dependency of its own — it is
    purely measurement.
    """

    def __init__(self, ranker):
        """``ranker(query, candidates) -> List[(memory_id, score)]``."""
        self._ranker = ranker

    def probe(
        self,
        query: str,
        candidates: Sequence[MemoryRecord],
        target_memory_id: str,
        pointer_at_probe: float,
    ) -> RecoveryMeasurement:
        """Run one query and see whether the target surfaces."""
        ranking = self._ranker(query, candidates)
        found_id = next(
            (mid for mid, _ in ranking if mid == target_memory_id), None
        )
        rank = None
        if found_id is not None:
            for i, (mid, _) in enumerate(ranking, start=1):
                if mid == target_memory_id:
                    rank = i
                    break
        target_present = any(c.id == target_memory_id for c in candidates)
        return RecoveryMeasurement(
            memory_id=target_memory_id,
            target_present=target_present,
            recovered=rank is not None,
            rank_if_found=rank,
            pointer_at_probe=float(pointer_at_probe),
            query=query,
        )


__all__ = ["ColdPointerProbe", "RecoveryMeasurement", "RecoveryReport"]