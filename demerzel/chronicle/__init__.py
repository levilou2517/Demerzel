"""Chronicle: long-term memory layer (V1.md §5.2, Phase 2).

Fact track, scene track, rule-based delta writes, and the L0 pointer layer.
Every record here carries ``source_ptr`` back to its Foundation turn, which is
what makes P0-6 (provenance) checkable end to end.
"""

from .delta_encoder import DeltaDecision, DeltaEncoder, DeltaOp
from .fact_memory import FactCandidate, FactMemory
from .pointer import PointerLayer
from .scene_memory import Scene, SceneMemory

__all__ = [
    "DeltaDecision",
    "DeltaEncoder",
    "DeltaOp",
    "FactCandidate",
    "FactMemory",
    "PointerLayer",
    "Scene",
    "SceneMemory",
]
