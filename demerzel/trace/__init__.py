"""Standard JSONL trace output (V1.md §5.5).

Every query run produces a small set of JSON-lines files under the run's
results directory, following the layout of §14 and the schema of §5.5::

    results/run_xxx/
        config.yaml (design note: produced as config.json)
        metadata.json
        metrics.json
        retrieval.jsonl
        pointer_updates.jsonl
        answers.jsonl

:trace schema (V1.md §5.5):

  run_id, query, candidates[{
      memory_id, semantic_score, ppr_score, pointer_strength,
      context_distance, final_score
  }], used_evidence[], mpe, pointer_updates[{
      memory_id, old, new, reason
  }]

The trace files are append-only audit records; the evaluation layer consumes
this format and does not live inside Demerzel (V1.md §5.5).
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, Optional


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class JsonlWriter:
    """Append-only JSON-lines writer for one result file."""

    def __init__(self, path: str):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self._path = path
        self._handle = open(path, "a", encoding="utf-8")

    def write(self, record: Dict[str, Any]) -> None:
        self._handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._handle.flush()

    def close(self) -> None:
        if self._handle and not self._handle.closed:
            self._handle.close()

    @property
    def path(self) -> str:
        return self._path

    def __enter__(self) -> "JsonlWriter":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


class RunTrace:
    """All trace outputs for one run, as lazily-opened JSON-lines writers."""

    FILES = ("retrieval.jsonl", "answers.jsonl", "pointer_updates.jsonl", "events.jsonl")

    def __init__(self, run_dir: str, run_id: str, config_hash: str):
        self.run_id = run_id
        self.config_hash = config_hash
        self.run_dir = run_dir
        os.makedirs(run_dir, exist_ok=True)
        self._writers: Dict[str, JsonlWriter] = {}

    def _writer(self, name: str) -> JsonlWriter:
        if name not in self._writers:
            self._writers[name] = JsonlWriter(os.path.join(self.run_dir, name))
        return self._writers[name]

    def log_retrieval(self, candidate: Dict[str, Any], query: str) -> None:
        """Append one query's retrieval trace (V1.md §5.5 schema)."""
        record = {
            "run_id": self.run_id,
            "config_hash": self.config_hash,
            "query": query,
            "candidates": candidate.get("candidates", []),
            "used_evidence": candidate.get("used_evidence", []),
            "mpe": candidate.get("mpe"),
            "pointer_updates": candidate.get("pointer_updates", []),
            "metrics": candidate.get("metrics", {}),
            "timestamp": candidate.get("timestamp") or _now(),
        }
        self._writer("retrieval.jsonl").write(record)

    def log_answer(self, answer: Dict[str, Any]) -> None:
        self._writer("answers.jsonl").write(
            {
                "run_id": self.run_id,
                "config_hash": self.config_hash,
                "timestamp": _now(),
                **answer,
            }
        )

    def log_pointer_update(self, pointer_update: Dict[str, Any]) -> None:
        self._writer("pointer_updates.jsonl").write(
            {
                "run_id": self.run_id,
                "config_hash": self.config_hash,
                "timestamp": _now(),
                **pointer_update,
            }
        )

    def log_event(self, event: Dict[str, Any]) -> None:
        self._writer("events.jsonl").write(
            {"run_id": self.run_id, "config_hash": self.config_hash, **event}
        )

    def write_metadata(self, metadata: Dict[str, Any]) -> str:
        """Persist ``metadata.json`` (V1.md §14 reproducibility block)."""
        path = os.path.join(self.run_dir, "metadata.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "run_id": self.run_id,
                    "config_hash": self.config_hash,
                    "system_version": metadata.get("system_version", "v0.1.0"),
                    "llm_model": metadata.get("llm_model"),
                    "embedding_model": metadata.get("embedding_model"),
                    "temperature": metadata.get("temperature"),
                    "seed": metadata.get("seed"),
                    "dataset_version": metadata.get("dataset_version"),
                    "timestamp": _now(),
                    **metadata,
                },
                handle,
                indent=2,
            )
        return path

    def write_metrics(self, metrics: Dict[str, Any]) -> str:
        """Persist ``metrics.json`` for one run."""
        path = os.path.join(self.run_dir, "metrics.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump({"run_id": self.run_id, **metrics}, handle, indent=2)
        return path

    def close(self) -> None:
        for writer in self._writers.values():
            writer.close()

    def __enter__(self) -> "RunTrace":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


def read_jsonl(path: str) -> list:
    """Read every JSON line in a file back as a list of dicts."""
    out = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                out.append(json.loads(line))
    return out


__all__ = [
    "JsonlWriter",
    "RunTrace",
    "read_jsonl",
]