"""Ingestion pipeline (V1.md §4.1, Phase 1-2).

Turns raw input into: an append-only Foundation turn, an episode boundary
decision, and zero or more Chronicle records carrying ``source_ptr`` back to
the Foundation turn.

The write path is deliberately ordered so memory can never be lost by a
downstream failure: **the raw turn is committed first**, then derived records.
If gist/semantic classification is unavailable the turn is still stored and the
derived record is marked degraded — that is the §17 "write gate fail-open"
rule, and it is what makes "LLM unavailable must never cause memory loss" true
rather than aspirational.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..interfaces.storage import MEMORY_FACT, MEMORY_SCENE, MemoryRecord, StorageBackend
from .gist import AlgorithmicGist
from .raw_store import RawStore, Turn


def content_hash(text: str) -> str:
    """Stable hash of raw content, stored beside each turn (V1.md §5.1)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


@dataclass
class IngestionResult:
    """What one ingested turn produced."""

    turn_id: str
    memory_ids: List[str] = field(default_factory=list)
    degraded: bool = False
    gate: str = "ok"
    reason: str = ""


class IngestionPipeline:
    """Raw-first ingestion with derived Chronicle records.

    Dependencies are injected: a ``RawStore`` (Foundation) and a
    ``StorageBackend`` (Chronicle). Neither is imported directly.
    """

    def __init__(
        self,
        raw_store: RawStore,
        storage: StorageBackend,
        gist: Optional[AlgorithmicGist] = None,
        default_memory_type: str = MEMORY_FACT,
        clock=None,
    ):
        self._raw = raw_store
        self._storage = storage
        self._gist = gist or AlgorithmicGist()
        self._default_type = default_memory_type
        self._clock = clock or time.time

    def ingest_turn(
        self,
        session_id: str,
        text: str,
        speaker: str = "user",
        episode_id: Optional[str] = None,
        memory_type: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        timestamp: Optional[float] = None,
        pointer_strength: float = 1.0,
    ) -> IngestionResult:
        """Ingest one turn.

        The Foundation append happens before any derived work, so a failure
        later in the pipeline cannot lose the evidence.
        """
        ts = float(timestamp if timestamp is not None else self._clock())
        turn_id = f"turn_{session_id}_{self._raw.count() + 1:06d}"
        episode = episode_id or f"episode_{session_id}_1"

        turn = Turn(
            turn_id=turn_id,
            session_id=session_id,
            timestamp=ts,
            speaker=speaker,
            text=text,
            metadata=dict(metadata or {}),
            content_hash=content_hash(text),
        )

        # 1. Foundation first: append-only, committed before anything derived.
        self._raw.append(turn)

        # 2. Derived Chronicle record.
        result = IngestionResult(turn_id=turn_id)
        try:
            gist = self._gist.extract(text)
        except Exception as exc:  # fail-open: still write, mark degraded
            gist = ""
            result.degraded = True
            result.gate = "unavailable"
            result.reason = f"gist failed: {exc}"

        memory_id = f"mem_{turn_id}"
        record = MemoryRecord(
            id=memory_id,
            session_id=session_id,
            episode_id=episode,
            memory_type=memory_type or self._default_type,
            content=text,
            gist=gist,
            created_at=ts,
            source_ptr=turn_id,
            pointer_strength=float(pointer_strength),
            retired=0,
            metadata=dict(metadata or {}),
        )
        self._storage.put(record)
        result.memory_ids.append(memory_id)
        return result

    def ingest_batch(
        self,
        session_id: str,
        turns: List[Dict[str, Any]],
    ) -> List[IngestionResult]:
        """Ingest a list of ``{text, speaker, ...}`` dicts in order."""
        results = []
        for turn in turns:
            results.append(
                self.ingest_turn(
                    session_id=session_id,
                    text=turn["text"],
                    speaker=turn.get("speaker", "user"),
                    episode_id=turn.get("episode_id"),
                    memory_type=turn.get("memory_type"),
                    metadata=turn.get("metadata"),
                    timestamp=turn.get("timestamp"),
                    pointer_strength=turn.get("pointer_strength", 1.0),
                )
            )
        return results

    def provenance(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """Resolve ``memory -> source_ptr -> Foundation turn`` (P0-6)."""
        record = self._storage.get(memory_id)
        if record is None:
            return None
        turn = self._raw.get(record.source_ptr)
        if turn is None:
            return None
        return {"memory": record, "turn": turn}


__all__ = [
    "AlgorithmicGist",
    "IngestionPipeline",
    "IngestionResult",
    "MEMORY_FACT",
    "MEMORY_SCENE",
    "content_hash",
]
