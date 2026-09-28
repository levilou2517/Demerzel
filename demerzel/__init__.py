"""Demerzel — an LLM external-memory research prototype.

Central hypothesis (V1.md):

    retrieval accessibility can be learned independently from memory
    persistence.

Memory is represented as ``M = (E, G, A)``: persistent evidence ``E``, graph
topology ``G``, and accessibility state ``A`` (``pointer_strength``). The MVP
proves engineering feasibility of::

    write -> retrieve -> evaluate -> MPE -> pointer update -> retrieve again

while preserving evidence and, under the Experiment-2 constraint, keeping
graph topology unchanged.

Package layout mirrors V1.md §4.1. ``demerzel.interfaces`` holds the seams;
``demerzel.reference`` holds replaceable implementations; the core mechanism
packages depend only on the former.
"""

__version__ = "0.1.0"
SYSTEM_NAME = "demerzel"
SYSTEM_VERSION = "v0.1.0"

__all__ = ["__version__", "SYSTEM_NAME", "SYSTEM_VERSION"]
