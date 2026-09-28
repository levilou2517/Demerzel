"""Scene memory track (V1.md §5.2, Phase 2).

Scenes are episodic groupings: a stretch of conversation sharing one
``episode_id``. The track groups Chronicle records into scenes and reports the
grouping, again without an LLM (a deterministic grouping key suffices).
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ..interfaces.storage import MEMORY_SCENE, MemoryRecord


@dataclass(frozen=True)
class Scene:
    """One episode's records, in creation order."""

    episode_id: str
    session_id: str
    memory_ids: List[str] = field(default_factory=list)
    start_at: float = 0.0
    end_at: float = 0.0

    @property
    def size(self) -> int:
        return len(self.memory_ids)


class SceneMemory:
    """Deterministic scene grouping by ``episode_id``."""

    MEMORY_TYPE = MEMORY_SCENE

    def group(self, records: List[MemoryRecord]) -> List[Scene]:
        """Group records into scenes keyed by ``episode_id``."""
        buckets: "OrderedDict[str, List[MemoryRecord]]" = OrderedDict()
        for record in records:
            buckets.setdefault(record.episode_id, []).append(record)

        scenes: List[Scene] = []
        for episode_id, members in buckets.items():
            times = [m.created_at for m in members]
            scenes.append(
                Scene(
                    episode_id=episode_id,
                    session_id=members[0].session_id,
                    memory_ids=[m.id for m in members],
                    start_at=min(times) if times else 0.0,
                    end_at=max(times) if times else 0.0,
                )
            )
        return scenes

    def scene_of(self, record: MemoryRecord) -> str:
        """The scene key of one record."""
        return record.episode_id

    def select(self, records: List[MemoryRecord]) -> List[MemoryRecord]:
        """Records explicitly typed as scenes."""
        return [r for r in records if r.memory_type == MEMORY_SCENE]


__all__ = ["Scene", "SceneMemory"]
