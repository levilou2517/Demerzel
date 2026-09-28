"""Demerzel command-line interface (V1.md §20).

The acceptance commands::

    demerzel init
    demerzel ingest examples/demo.json
    demerzel query "..."
    demerzel inspect-memory <memory_id>
    demerzel inspect-trace <run_id>
    demerzel run-experiment exp1
    demerzel run-experiment exp2

Every command is deterministic and writes its artifacts under the run
directory, so a result can be re-read from disk rather than trusted from
stdout.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional

from .config import Config, load_config, save_config
from .engine import Engine
from .reference import (
    DictGraph,
    HashingEmbeddingProvider,
    InMemoryEvidenceStore,
    InMemoryStorage,
    MockLLMProvider,
    SqliteEvidenceStore,
    SqliteStorage,
)
from .trace import RunTrace, read_jsonl


def build_engine(config: Config, run_dir: Optional[str] = None) -> Engine:
    """Construct an engine from config, choosing the configured backends."""
    use_sqlite = config.storage.backend == "sqlite" and config.storage.path
    if use_sqlite:
        storage = SqliteStorage(config.storage.path, config.storage.synchronous)
        evidence = SqliteEvidenceStore(config.storage.path, config.storage.synchronous)
    else:
        storage = InMemoryStorage()
        evidence = InMemoryEvidenceStore()

    graph = DictGraph()
    embedder = HashingEmbeddingProvider(config.providers.get("embedding_dim", 256))
    llm = MockLLMProvider()

    trace = None
    if config.trace.enabled and run_dir:
        trace = RunTrace(run_dir, config.run_id, config.config_hash())

    engine = Engine(storage, graph, embedder, evidence, llm, config, run_trace=trace)

    # Rebuild the graph index from existing records (restart path).
    for record in storage.list_all():
        engine.builder.add_memory_node(record)
        engine.builder.link_episode(record)
    return engine


def cmd_init(args: argparse.Namespace) -> int:
    """Create the workspace layout and a default config."""
    os.makedirs("data", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs("examples", exist_ok=True)
    config_path = args.config or "demerzel.config.json"
    if not os.path.exists(config_path):
        save_config(Config(), config_path)
        print(f"wrote {config_path}")
    else:
        print(f"kept existing {config_path}")
    print("initialised: data/, results/, examples/")
    return 0


def cmd_ingest(args: argparse.Namespace) -> int:
    """Ingest a JSON file of turns into the configured store."""
    config = load_config(args.config)
    engine = build_engine(config, args.run_dir)

    with open(args.path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    session_id = args.session or payload.get("session_id", "s1")
    turns = payload.get("turns", payload if isinstance(payload, list) else [])
    if not turns:
        print("no turns found", file=sys.stderr)
        return 2

    ids = []
    for turn in turns:
        if isinstance(turn, str):
            turn = {"text": turn}
        ids.append(engine.ingest(session_id, turn["text"], turn.get("speaker", "user")))

    print(f"ingested {len(ids)} turn(s) into session {session_id}")
    for memory_id in ids:
        print(f"  {memory_id}")
    return 0


def cmd_query(args: argparse.Namespace) -> int:
    """Run the full loop for a query and print the trace."""
    config = load_config(args.config)
    engine = build_engine(config, args.run_dir)
    result = engine.query(args.text, args.session)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


def cmd_inspect_memory(args: argparse.Namespace) -> int:
    """Show one memory, its provenance, and its pointer/decay history."""
    config = load_config(args.config)
    engine = build_engine(config, args.run_dir)
    record = engine.storage.get(args.memory_id)
    if record is None:
        print(f"memory not found: {args.memory_id}", file=sys.stderr)
        return 1

    turn = engine._raw.get(record.source_ptr)
    print(json.dumps(
        {
            "memory": {
                "id": record.id,
                "session_id": record.session_id,
                "episode_id": record.episode_id,
                "memory_type": record.memory_type,
                "content": record.content,
                "gist": record.gist,
                "created_at": record.created_at,
                "source_ptr": record.source_ptr,
                "pointer_strength": record.pointer_strength,
                "retired": record.retired,
            },
            "foundation_turn": turn,
            "provenance_ok": turn is not None,
            "forgetting_events": engine.forgetting_log.by_memory(args.memory_id),
        },
        indent=2,
        ensure_ascii=False,
    ))
    return 0


def cmd_inspect_trace(args: argparse.Namespace) -> int:
    """Print the recorded trace rows for a run."""
    run_dir = args.run_dir or os.path.join("results", args.run_id)
    path = os.path.join(run_dir, "retrieval.jsonl")
    if not os.path.exists(path):
        print(f"no trace at {path}", file=sys.stderr)
        return 1
    rows = read_jsonl(path)
    print(json.dumps(rows, indent=2, ensure_ascii=False))
    return 0


def cmd_run_experiment(args: argparse.Namespace) -> int:
    """Run a configured experiment and write results under results/<name>/."""
    from .evaluation.experiments import run_experiment

    config = load_config(args.config)
    out = run_experiment(args.name, config, results_root=args.results_root)
    print(json.dumps(out, indent=2, ensure_ascii=False))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="demerzel", description="Demerzel research prototype")
    parser.add_argument("--config", help="config file (JSON or YAML)")
    parser.add_argument("--run-dir", help="results directory for this run")
    sub = parser.add_subparsers(dest="command", required=True)

    p_init = sub.add_parser("init", help="create workspace layout and config")
    p_init.set_defaults(func=cmd_init)

    p_ing = sub.add_parser("ingest", help="ingest a JSON turn file")
    p_ing.add_argument("path")
    p_ing.add_argument("--session")
    p_ing.set_defaults(func=cmd_ingest)

    p_q = sub.add_parser("query", help="run one query through the core loop")
    p_q.add_argument("text")
    p_q.add_argument("--session", default="s1")
    p_q.set_defaults(func=cmd_query)

    p_im = sub.add_parser("inspect-memory", help="show one memory and its provenance")
    p_im.add_argument("memory_id")
    p_im.set_defaults(func=cmd_inspect_memory)

    p_it = sub.add_parser("inspect-trace", help="print a run's retrieval trace")
    p_it.add_argument("run_id")
    p_it.set_defaults(func=cmd_inspect_trace)

    p_exp = sub.add_parser("run-experiment", help="run exp1..exp4")
    p_exp.add_argument("name")
    p_exp.add_argument("--results-root", default="results")
    p_exp.set_defaults(func=cmd_run_experiment)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())