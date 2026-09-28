"""MPE evaluator (V1.md §7.3, Phase 6).

``MPE = U_actual - U_predicted``.

The sign is the entire experiment: a *positive* MPE means the memory was more
useful than predicted, so its pointer should rise; a *negative* MPE means less
useful than predicted, so its pointer should fall. This module computes only
the difference and clips it (V1 `mpe_clip`), keeping the bounded, signed value
that the updater consumes. No storage or graph is touched here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class MPEResult:
    """One MPE computation."""

    memory_id: str
    u_predicted: float
    u_actual: float
    mpe: float
    clipped: float
    positive: bool

    def as_dict(self) -> dict:
        return {
            "memory_id": self.memory_id,
            "u_predicted": round(self.u_predicted, 6),
            "u_actual": round(self.u_actual, 6),
            "mpe": round(self.mpe, 6),
            "clipped_mpe": round(self.clipped, 6),
        }


class MPEEvaluator:
    """Compute and bound the MPE signal."""

    def __init__(self, mpe_clip: float = 1.0):
        self.mpe_clip = float(mpe_clip)

    def evaluate(
        self,
        memory_id: str,
        u_predicted: float,
        u_actual: float,
    ) -> MPEResult:
        mpe = u_actual - u_predicted
        if mpe < -self.mpe_clip:
            clipped = -self.mpe_clip
        elif mpe > self.mpe_clip:
            clipped = self.mpe_clip
        else:
            clipped = mpe
        return MPEResult(
            memory_id=memory_id,
            u_predicted=float(u_predicted),
            u_actual=float(u_actual),
            mpe=float(mpe),
            clipped=float(clipped),
            positive=clipped > 0,
        )


__all__ = ["MPEEvaluator", "MPEResult"]