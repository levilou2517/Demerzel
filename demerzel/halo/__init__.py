"""Halo: shallow-memory layer (V1.md §5.4, Phase 4).

Term discipline: the halo *evicts / merges / retains*, it never deletes.
Halo eviction is not memory deletion — the persistent Foundation and Chronicle
copies are untouched (Invariant 4).
"""

from .context_state import ContextState
from .refine import Refine
from .ring_buffer import EvictionPolicy, HaloTurn, RingBuffer

__all__ = [
    "ContextState",
    "EvictionPolicy",
    "HaloTurn",
    "Refine",
    "RingBuffer",
]