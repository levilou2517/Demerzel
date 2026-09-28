"""Forgetting log (V1.md §4.1, Phase 5).

Every decay application and every retrieval outcome that affects accessibility
is recorded here, so the observability questions of §16 are answerable from
artifacts rather than from intent:

    why was this memory recalled? why was the other one not?
    what is its pointer_strength? its context_distance? its MPE?

The log is append-only and JSONL-serialisable. It deliberately records
``reason`` values such as ``cold_start``, ``positive_mpe`` and
``mpe_disabled``, which is what makes an ablation auditable after the fact.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional


class ForgettingLog:
    """In-memory (optionally file-backed) record of accessibility events."""

    def __init__(self, path: Optional[str] = None):
        self._rows: List[Dict[str, Any]] = []
        self._path = path
        self._handle = None
        if path:
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            self._handle = open(path, "a", encoding="utf-8")

    def record(self, event: Dict[str, Any]) -> None:
        """Append one event."""
        row = dict(event)
        self._rows.append(row)
        if self._handle is not None:
            self._handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            self._handle.flush()

    def record_decay(self, decay_row: Dict[str, Any]) -> None:
        self.record({"event": "decay", **decay_row})

    def record_pointer(self, pointer_row: Dict[str, Any]) -> None:
        self.record({"event": "pointer_update", **pointer_row})

    def record_retrieval(self, retrieval_row: Dict[str, Any]) -> None:
        self.record({"event": "retrieval", **retrieval_row})

    @property
    def rows(self) -> List[Dict[str, Any]]:
        return list(self._rows)

    def by_memory(self, memory_id: str) -> List[Dict[str, Any]]:
        """Every event about one memory — the §16 "why didn't it recall" view."""
        return [r for r in self._rows if r.get("memory_id") == memory_id]

    def count(self) -> int:
        return len(self._rows)

    def close(self) -> None:
        if self._handle is not None and not self._handle.closed:
            self._handle.close()
            self._handle = None

    def __enter__(self) -> "ForgettingLog":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


__all__ = ["ForgettingLog"]