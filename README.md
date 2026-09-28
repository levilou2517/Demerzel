# Demerzel

**An LLM external-memory research prototype.**

> **Central hypothesis (V1.md §1).**
> *Can retrieval accessibility be learned independently from memory
> persistence?*

Demerzel represents memory as a triple `M = (E, G, A)`:

- **E — Evidence Persistence.** Foundation / Chronicle raw evidence, never
  deleted by the forgetting mechanism.
- **G — Memory Graph.** The Nexus node/edge topology.
- **A — Accessibility State.** The `pointer_strength` set that decides
  retrieval reachability.

The claim under test is narrow and deliberate: **`A` can be learned
independently of `G`.** That retrieval-driven *topology* reorganization is
novel is not claimed (REALM covers it). What Demerzel claims is that
accessibility state evolves on its own.

```
REALM:     G_{t+1} = f(G_t, R_t)          # topology reorganizes on feedback
Demerzel:  A_{t+1} = f(A_t, MPE_t, d_t)   # accessibility evolves independently
           G_{t+1} = G_t                  # Experiment-2 constraint: G fixed
           E_{t+1} ≈ E_t                  # evidence persists
```

## What this is (and is not)

This is a **research prototype**, not a product. Phase 0–6 build the smallest
deterministic system that can prove or falsify the hypothesis, with everything
built for reproducibility, observability, ablation and testing. The MVP is
complete when the P0 invariants (1–8) and the full MPE feedback loop pass — and
then it **stops**.

Deliberately **not** included (V1.md §3.2): neuromodulators, sleep stages,
procedural-memory copies, embedding compression, meta-learning, online
training/RL, autonomous planning, heavyweight agent frameworks.

## The minimal core loop

```
WRITE → INDEX → RETRIEVE → USE → EVALUATE → UPDATE
  │        │        │       │       │          │
Foundation Chronicle Nexus  answer  MPE   pointer update
```

Each stage can be switched off independently for ablation.

## Quick start

```sh
git clone https://github.com/levilou2517/Demerzel.git && cd Demerzel
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip setuptools wheel && pip install -e .

demerzel init
demerzel ingest examples/demo.json
demerzel query "What is the capital of France?"
demerzel inspect-memory mem_turn_s1_000001
python tests/run_tests.py
```

No network, GPU or API key is needed at runtime — the default providers are
deterministic mocks. Full instructions (offline install, Docker,
reproducing a published result): **`docs/deployment.md`**.

Or from Python:

```python
from demerzel.config import Config
from demerzel.engine import build_default_engine

engine = build_default_engine(Config())
engine.ingest("s1", "The capital of France is Paris.")
trace = engine.query("What is the capital of France?")
print(trace["candidates"][0], trace["mpe"])
```

## Layout

```
demerzel/
  interfaces/   StorageBackend, GraphBackend, PointerUpdater,
                LLMProvider, EmbeddingProvider  — the seams
  core/         ContextDistance, PointerDecay  — core mechanisms
  reference/    SQLite, in-memory, DictGraph, mock providers
  foundation/   raw store, gist, event boundary, ingestion
  chronicle/    fact, scene, delta encoder, L0 pointer
  nexus/        nodes, edges, PPR, lexical/vector retrieval, hybrid scoring
  halo/         ring buffer, context state, refine
  mpe/          predictor, utility, evaluator, updater
  entropy/      context decay, forgetting log, cold pointer
  assemblage/   query router, sufficiency router, reranker, context, answerer
  trace/        standard JSONL trace schema
  evaluation/   metrics + the four experiment protocols
configs/        experiment configs
docs/           architecture, api, experiments, invariants,
                cold_start, mpe_timing, test_report, benchmark
tests/          the acceptance + invariant suite
scripts/        experiment reproduction scripts
```

## Guarantees (the invariants)

| # | invariant |
|---|-----------|
| 1 | Pointer decay does not delete evidence |
| 2 | Pointer update does not change evidence content |
| 3 | MPE-only does not change graph topology |
| 4 | Halo eviction does not delete Chronicle |
| 5 | Cold memory can be reactivated |
| 6 | Explicit user deletion is distinct from forgetting |
| 7 | Context distance cold start degrades to 0 |
| 8 | `U_predicted` contains no retrieval result |

Details and test mapping: `docs/invariants.md`.

## Design rules honoured

- **Algorithm-first, LLM on demand.** An LLM may classify, arbitrate, answer or
  rerank; it may **never** decide whether to store evidence, mutate
  `pointer_strength`, mutate the graph, delete memory, or write storage.
- **Interfaces before subsystems.** Every mechanism depends on an interface,
  never on another mechanism or a concrete backend.
- **Never delete memory because of forgetting.** Soft forgetting lowers
  accessibility; `retired != deleted`.
- **Experiments before claims.** No improvement is claimed unless the
  experiment ran and the raw results are saved.

## Status

**Phase 0–6 implemented, P0 gate passing.** See `docs/test_report.md` for the
full pass/fail record and the one substantive negative finding (the Exp-2
prerequisite's dependence on the accessibility-prior weight).

Per the V1.md stop condition, the project **stops after P0** and awaits human
review before P1/P2.

## Documentation

| doc | contents |
|-----|----------|
| `docs/deployment.md` | install, first run, offline install, Docker, reproduction |
| `docs/architecture.md` | modules, interfaces, data flow |
| `docs/api.md` | public API and CLI |
| `docs/experiments.md` | Exp 1–4 protocols, configs, metrics |
| `docs/invariants.md` | invariants 1–8 with enforcement and tests |
| `docs/cold_start.md` | cold-start specification |
| `docs/mpe_timing.md` | MPE timing specification |
| `docs/test_report.md` | the test/acceptance record |
| `docs/example_trace.json` | a real saved trace |
| `docs/benchmark.md` | benchmark instructions |

## Version

`Demerzel V0.1.0` — built and run under the DSH agent preset
`demerzel-v0-1-0`.
