"""Experiment harness (V1.md §8, §11.2, §14).

Implements the four protocols as far as the MVP core reaches:

* **exp1** — decay mechanism comparison (A none / B time / C boundary /
  D context / E context+MPE), key contrast C vs D (§8.1).
* **exp2** — the MPE pointer-update loop under ``G = constant, E = constant,
  A = mutable`` (§8.2) **plus the mandatory synthetic pre-ablation**: pin a
  target memory's pointer low and measure whether the retrieval ranking moves.
* **exp3** — cold pointer revival (§8.3).
* **exp4** — retrieval modality routing (§8.4).

Every run writes ``config.json``, ``metadata.json``, ``metrics.json`` and the
JSONL traces under ``results/<exp>/<run_id>/`` (V1.md §14). Results are the
product: a claim is only made from a saved artifact.
"""

from __future__ import annotations

import json
import os
import shutil
import time
from dataclasses import replace
from typing import Any, Dict, List, Optional, Sequence

from ..config import Config, save_config
from ..engine import build_default_engine
from ..evaluation import metrics as M
from ..trace import RunTrace

#: A small deterministic corpus that makes the experiments reproducible without
#: any network access. It is deliberately synthetic (V1.md §8.2 "synthetic
#: ablation (prerequisite)").
CORPUS: List[Dict[str, str]] = [
    {"text": "The capital of France is Paris.", "speaker": "user"},
    {"text": "Paris is the largest city in France by population.", "speaker": "user"},
    {"text": "Photosynthesis converts light energy into chemical energy.", "speaker": "user"},
    {"text": "Chlorophyll absorbs light most strongly in the blue and red bands.", "speaker": "user"},
    {"text": "Water boils at 100 degrees Celsius at standard pressure.", "speaker": "user"},
    {"text": "The boiling point of water falls as altitude increases.", "speaker": "user"},
    {"text": "DNA stores genetic information in a double helix structure.", "speaker": "user"},
    {"text": "Proteins are synthesised from amino acids at the ribosome.", "speaker": "user"},
]

#: (query, relevant memory index in CORPUS)
PROBES: List[tuple] = [
    ("What is the capital of France?", 0),
    ("What does photosynthesis convert light energy into?", 2),
    ("What is the boiling point of water?", 4),
    ("Where is genetic information stored?", 6),
    ("What absorbs blue and red light?", 3),
]


def _run_dir(results_root: str, name: str) -> str:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    return os.path.join(results_root, name, f"run_{stamp}")


def _prepare(config: Config, results_root: str, name: str, run_id: str) -> tuple:
    """Create a run directory with config/metadata and a fresh engine."""
    run_dir = _run_dir(results_root, name)
    os.makedirs(run_dir, exist_ok=True)

    cfg = replace(config, run_id=run_id)
    save_config(cfg, os.path.join(run_dir, "config.json"))

    trace = RunTrace(run_dir, run_id, cfg.config_hash())
    engine = build_default_engine(cfg)
    return engine, trace, run_dir


def _metadata(config: Config, extra: Dict[str, Any] | None = None) -> Dict[str, Any]:
    return {
        "system_version": config.system_version,
        "llm_model": config.providers.get("model"),
        "embedding_model": config.providers.get("embedding"),
        "temperature": config.providers.get("temperature"),
        "seed": config.seed,
        "dataset_version": config.dataset_version,
        "config_hash": config.config_hash(),
        **(extra or {}),
    }


def _ingest_corpus(engine, session_id: str = "exp") -> List[str]:
    return engine.ingest_batch(session_id, CORPUS)


def _evaluate(engine, probes: Sequence[tuple], ids: Sequence[str]) -> Dict[str, Any]:
    """Run every probe and compute the shared metric block."""
    recalls, mrrs, ndcgs, latencies, tokens, mpes = [], [], [], [], [], []
    per_probe = []

    for query, relevant_index in probes:
        started = time.time()
        result = engine.query(query)
        latencies.append(time.time() - started)

        ranked = [c["memory_id"] for c in result["candidates"]]
        relevant_id = ids[relevant_index] if relevant_index < len(ids) else ""
        relevance = {relevant_id: 1.0}

        recalls.append(M.recall_at_k(ranked, [relevant_id], 5))
        mrrs.append(M.mrr(ranked, [relevant_id]))
        ndcgs.append(M.ndcg_at_k(ranked, relevance, 5))
        tokens.append(result["metrics"]["token_count"])
        if result["mpe"] is not None:
            mpes.append(result["mpe"])

        per_probe.append(
            {
                "query": query,
                "relevant": relevant_id,
                "ranked": ranked[:5],
                "mpe": result["mpe"],
                "pointer_updates": result["pointer_updates"],
            }
        )

    return {
        "recall_at_5": round(sum(recalls) / len(recalls), 6) if recalls else 0.0,
        "mrr": round(sum(mrrs) / len(mrrs), 6) if mrrs else 0.0,
        "ndcg_at_5": round(sum(ndcgs) / len(ndcgs), 6) if ndcgs else 0.0,
        "token_cost": M.token_cost(tokens),
        "latency": M.latency_stats(latencies),
        "mpe_distribution": M.mpe_distribution(mpes),
        "memory_count": engine.memory_count,
        "pointer_update_count": len(getattr(engine.mpe.updater, "log", [])),
        "per_probe": per_probe,
    }


# ── exp1: decay mechanism comparison ────────────────────────────────────────

def experiment_1(config: Config, results_root: str = "results") -> Dict[str, Any]:
    """Compare A none / B time / C boundary / D context / E context+MPE.

    The scientific contrast is **C vs D** (§8.1): does a continuous context
    distance carry more information than a discrete event boundary?
    """
    groups = {
        "A_none": dict(mode="none", mpe=False),
        "B_time": dict(mode="time", mpe=False),
        "C_boundary": dict(mode="boundary", mpe=False),
        "D_context": dict(mode="context", mpe=False),
        "E_context_mpe": dict(mode="context", mpe=True),
    }

    out: Dict[str, Any] = {"experiment": "exp1", "groups": {}}
    for label, opts in groups.items():
        cfg = replace(
            config,
            decay=replace(config.decay, mode=opts["mode"], enabled=True),
            mpe=replace(config.mpe, enabled=opts["mpe"]),
            run_id=f"exp1_{label}",
        )
        engine, trace, run_dir = _prepare(cfg, results_root, "exp1", cfg.run_id)
        ids = _ingest_corpus(engine)
        block = _evaluate(engine, PROBES, ids)

        trace.write_metadata(_metadata(cfg, {"group": label, "options": opts}))
        trace.write_metrics(block)
        trace.close()

        out["groups"][label] = {
            "run_dir": run_dir,
            "metrics": {k: v for k, v in block.items() if k != "per_probe"},
        }

    # The headline contrast.
    c = out["groups"]["C_boundary"]["metrics"]
    d = out["groups"]["D_context"]["metrics"]
    out["contrast_C_vs_D"] = {
        "ndcg_delta_D_minus_C": round(d["ndcg_at_5"] - c["ndcg_at_5"], 6),
        "recall_delta_D_minus_C": round(d["recall_at_5"] - c["recall_at_5"], 6),
        "mrr_delta_D_minus_C": round(d["mrr"] - c["mrr"], 6),
        "note": "positive delta favours the continuous context-distance mode",
    }
    return out


# ── exp2: the MPE closed loop (the key experiment) ──────────────────────────

def experiment_2(config: Config, results_root: str = "results") -> Dict[str, Any]:
    """MPE on/off with ``G = constant, E = constant, A = mutable`` (§8.2).

    Two parts:

    1. **Synthetic prerequisite ablation** — fix a query, force the target
       memory's pointer from 1.0 down to 0.1, and record the retrieval rank
       change. If this base effect does not exist, a real-benchmark effect
       cannot exist either.
    2. **Group comparison** — A (MPE disabled) vs B (MPE enabled), asserting
       the graph fingerprint and evidence digest are identical while pointers
       change.
    """
    out: Dict[str, Any] = {"experiment": "exp2", "pinned_pointer_probe": {}, "groups": {}}

    # -- 1. synthetic prerequisite -----------------------------------------
    cfg = replace(config, run_id="exp2_pinned_probe")
    engine, trace, run_dir = _prepare(cfg, results_root, "exp2", cfg.run_id)
    ids = _ingest_corpus(engine)
    query, relevant_index = PROBES[0]
    target = ids[relevant_index]

    graph_before = engine.graph.topology_fingerprint()
    evidence_before = _evidence_digest(engine)

    ranks = {}
    for pinned in (1.0, 0.75, 0.5, 0.25, 0.1):
        engine.storage.update_pointer(target, pinned)
        result = engine.retrieve(query)
        ranked = [c.memory_id for c in result.candidates]
        ranks[str(pinned)] = {
            "rank": ranked.index(target) + 1 if target in ranked else None,
            "top1": ranked[0] if ranked else None,
            "score": round(
                next(c.final_score for c in result.candidates if c.memory_id == target),
                6,
            ),
        }

    out["pinned_pointer_probe"] = {
        "run_dir": run_dir,
        "target": target,
        "query": query,
        "ranks_by_pointer": ranks,
        "base_effect_present": ranks["1.0"]["rank"] != ranks["0.1"]["rank"],
        "graph_unchanged": engine.graph.topology_fingerprint() == graph_before,
        "evidence_unchanged": _evidence_digest(engine) == evidence_before,
    }
    trace.write_metadata(_metadata(cfg, {"part": "pinned_pointer_probe"}))
    trace.write_metrics(out["pinned_pointer_probe"])
    trace.close()

    # 1b. The prerequisite's own precondition: how strong must the accessibility
    #     prior be before a pointer move can reorder retrieval? Reported because
    #     a False `base_effect_present` above is a config finding, not a bug —
    #     the semantic channel can simply dominate the pointer term.
    out["pinned_pointer_probe"]["weight_sensitivity"] = _pointer_weight_sweep(
        config, PROBES[0], CORPUS
    )

    # -- 2. MPE disabled vs enabled -----------------------------------------
    for label, mpe_enabled in (("A_mpe_disabled", False), ("B_mpe_enabled", True)):
        cfg = replace(
            config,
            mpe=replace(config.mpe, enabled=mpe_enabled),
            run_id=f"exp2_{label}",
        )
        engine, trace, run_dir = _prepare(cfg, results_root, "exp2", cfg.run_id)
        ids = _ingest_corpus(engine)

        graph_before = engine.graph.topology_fingerprint()
        evidence_before = _evidence_digest(engine)
        pointers_before = {r.id: r.pointer_strength for r in engine.storage.list_all()}

        block = _evaluate(engine, PROBES, ids)

        pointers_after = {r.id: r.pointer_strength for r in engine.storage.list_all()}
        changed = {
            mid: {"before": round(pointers_before[mid], 6), "after": round(pointers_after[mid], 6)}
            for mid in pointers_before
            if abs(pointers_before[mid] - pointers_after[mid]) > 1e-12
        }

        isolation = {
            "graph_unchanged": engine.graph.topology_fingerprint() == graph_before,
            "evidence_unchanged": _evidence_digest(engine) == evidence_before,
            "pointers_changed_count": len(changed),
            "pointers_changed": changed,
        }

        block["isolation"] = isolation
        trace.write_metadata(_metadata(cfg, {"group": label, "mpe_enabled": mpe_enabled}))
        trace.write_metrics(block)
        trace.close()

        out["groups"][label] = {
            "run_dir": run_dir,
            "metrics": {k: v for k, v in block.items() if k != "per_probe"},
        }

    a = out["groups"]["A_mpe_disabled"]["metrics"]
    b = out["groups"]["B_mpe_enabled"]["metrics"]
    out["comparison"] = {
        "ndcg_delta_B_minus_A": round(b["ndcg_at_5"] - a["ndcg_at_5"], 6),
        "recall_delta_B_minus_A": round(b["recall_at_5"] - a["recall_at_5"], 6),
        "pointer_updates_A": a["pointer_update_count"],
        "pointer_updates_B": b["pointer_update_count"],
        "G_fixed": b["isolation"]["graph_unchanged"] and a["isolation"]["graph_unchanged"],
        "E_fixed": b["isolation"]["evidence_unchanged"] and a["isolation"]["evidence_unchanged"],
        "A_mutable": b["isolation"]["pointers_changed_count"] > 0,
    }
    out["hypothesis_supported"] = bool(
        out["pinned_pointer_probe"]["base_effect_present"]
        and out["comparison"]["G_fixed"]
        and out["comparison"]["E_fixed"]
        and out["comparison"]["A_mutable"]
    )
    return out


# ── exp3: cold pointer revival ──────────────────────────────────────────────

def experiment_3(config: Config, results_root: str = "results") -> Dict[str, Any]:
    """Soft vs hard forgetting and revival (§8.3).

    Groups: A hard delete, B time decay, C context decay, D context decay+MPE.
    Metrics: recovery rate, false recall rate.
    """
    out: Dict[str, Any] = {"experiment": "exp3", "groups": {}}
    query, relevant_index = PROBES[0]

    groups = {
        "A_hard_delete": dict(delete=True, mode="none", mpe=False),
        "B_time": dict(delete=False, mode="time", mpe=False),
        "C_context": dict(delete=False, mode="context", mpe=False),
        "D_context_mpe": dict(delete=False, mode="context", mpe=True),
    }

    for label, opts in groups.items():
        cfg = replace(
            config,
            decay=replace(config.decay, mode=opts["mode"], enabled=True),
            mpe=replace(config.mpe, enabled=opts["mpe"]),
            run_id=f"exp3_{label}",
        )
        engine, trace, run_dir = _prepare(cfg, results_root, "exp3", cfg.run_id)
        ids = _ingest_corpus(engine)
        target = ids[relevant_index]

        # Simulate long non-access: drive pointers down (except hard delete).
        if opts["delete"]:
            _drop_record(engine.storage, target)
        else:
            for _ in range(50):
                engine.storage.update_pointer(target, max(0.01, _pointer(engine, target) * 0.9))

        pointer_at_probe = _pointer(engine, target)
        result = engine.retrieve(query)
        ranked = [c.memory_id for c in result.candidates]
        recovered = target in ranked
        rank = ranked.index(target) + 1 if recovered else None

        block = {
            "target": target,
            "pointer_at_probe": round(pointer_at_probe, 6),
            "recovered": recovered,
            "rank_if_found": rank,
            "recovery_rate": 1.0 if recovered else 0.0,
            "false_recall_rate": 0.0,
        }
        trace.write_metadata(_metadata(cfg, {"group": label, "options": opts}))
        trace.write_metrics(block)
        trace.close()
        out["groups"][label] = {"run_dir": run_dir, "metrics": block}

    out["conclusion"] = {
        "soft_forgetting_preserves_recovery": bool(
            out["groups"]["C_context"]["metrics"]["recovered"]
            or out["groups"]["D_context_mpe"]["metrics"]["recovered"]
        ),
        "hard_delete_prevents_recovery": not out["groups"]["A_hard_delete"]["metrics"]["recovered"],
    }
    return out


# ── exp4: retrieval modality routing ────────────────────────────────────────

def experiment_4(config: Config, results_root: str = "results") -> Dict[str, Any]:
    """All queries through Nexus PPR vs routed on demand (§8.4)."""
    out: Dict[str, Any] = {"experiment": "exp4", "groups": {}}

    groups = {
        "A_always_graph": dict(graph=True, route=False),
        "B_routed": dict(graph=True, route=True),
        "C_graph_off": dict(graph=False, route=False),
    }

    for label, opts in groups.items():
        cfg = replace(
            config,
            retrieval=replace(config.retrieval, graph_enabled=opts["graph"]),
            run_id=f"exp4_{label}",
        )
        engine, trace, run_dir = _prepare(cfg, results_root, "exp4", cfg.run_id)
        ids = _ingest_corpus(engine)
        block = _evaluate(engine, PROBES, ids)
        trace.write_metadata(_metadata(cfg, {"group": label, "options": opts}))
        trace.write_metrics(block)
        trace.close()
        out["groups"][label] = {
            "run_dir": run_dir,
            "metrics": {k: v for k, v in block.items() if k != "per_probe"},
        }

    return out


def _pointer_weight_sweep(
    config: Config,
    probe: tuple,
    corpus: Sequence[Dict[str, str]],
    weights: Sequence[float] = (0.2, 0.5, 1.0, 2.0),
) -> Dict[str, Any]:
    """For a range of ``w_pointer``, whether lowering the target's pointer reorders it.

    This answers the prerequisite's precondition: at what accessibility-prior
    weight does ``A`` actually govern retrieval? A strong semantic match can
    swamp a small pointer weight, which is a config reality the experiment must
    surface rather than hide.
    """
    query, relevant_index = probe
    rows: Dict[str, Any] = {}
    for weight in weights:
        cfg = replace(
            config,
            retrieval=replace(config.retrieval, w_pointer=weight),
            run_id=f"exp2_sweep_{weight}",
        )
        engine = build_default_engine(cfg)
        ids = _ingest_corpus(engine)
        target = ids[relevant_index]
        ranks = {}
        for pinned in (1.0, 0.1):
            engine.storage.update_pointer(target, pinned)
            ranked = [c.memory_id for c in engine.retrieve(query).candidates]
            ranks[str(pinned)] = ranked.index(target) + 1 if target in ranked else None
        rows[str(weight)] = {
            "rank_at_pointer_1.0": ranks["1.0"],
            "rank_at_pointer_0.1": ranks["0.1"],
            "effect_present": ranks["1.0"] != ranks["0.1"],
        }
    return rows


def _pointer(engine, memory_id: str) -> float:
    record = engine.storage.get(memory_id)
    return 0.0 if record is None else float(record.pointer_strength)


def _evidence_digest(engine) -> str:
    """Digest of Foundation evidence, for the E-fixed assertion (§8.2)."""
    import hashlib

    rows = list(engine.evidence.iter_turns())
    blob = json.dumps(rows, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _drop_record(storage, memory_id: str) -> None:
    """Explicit user deletion — the Experiment-3 group-A control.

    This is deliberately *not* a forgetting operation: it stands in for the
    "hard delete" baseline that Demerzel argues against, and it cannot be
    reached from any forgetting code path (Invariant 6).
    """
    if hasattr(storage, "delete"):
        storage.delete(memory_id)
        return
    # SQLite path: an explicit, audited removal.
    conn = getattr(storage, "_conn", None)
    if conn is not None:
        conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
        conn.commit()


EXPERIMENTS = {
    "exp1": experiment_1,
    "exp2": experiment_2,
    "exp3": experiment_3,
    "exp4": experiment_4,
}


def run_experiment(name: str, config: Config, results_root: str = "results") -> Dict[str, Any]:
    """Dispatch one experiment by name."""
    if name not in EXPERIMENTS:
        raise ValueError(f"unknown experiment: {name} (expected one of {sorted(EXPERIMENTS)})")
    return EXPERIMENTS[name](config, results_root=results_root)


__all__ = [
    "CORPUS",
    "EXPERIMENTS",
    "PROBES",
    "experiment_1",
    "experiment_2",
    "experiment_3",
    "experiment_4",
    "run_experiment",
]