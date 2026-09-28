# Demerzel Architecture

> Research prototype for the claim: *retrieval accessibility can be learned
> independently from memory persistence.* (V1.md §22)

## Memory as `M = (E, G, A)`

| letter | name | holder | mutated by |
|--------|------|--------|-----------|
| **E** | Evidence Persistence | Foundation raw store + Chronicle content | ingestion only |
| **G** | Memory Graph | Nexus topology | graph builder only |
| **A** | Accessibility State | Chronicle `pointer_strength` | decay + MPE updater |

The central experimental discipline: **E and G are constant while A evolves**.
The MPE path (which changes A) is written so it *cannot* change G or E — enforced
at the interface, not by convention.

## Layers and modules (V1.md §4.1)

```
Ingestion (write)
   │  raw turn ──► Foundation  (append-only evidence, content_hash)
   │  derived ──► Chronicle   (fact/scene/delta/pointer; source_ptr)
   │  index ────► Nexus        (nodes, edges, PPR, lexical BM25, vector)
   │  shallow ─> Halo          (ring buffer, context state)
Retrieval (index -> retrieve)
   │  route ───► Assemblage.query_router (lexical/vector/graph)
   │  score ──► Nexus.hybrid    (w_s*cos + w_g*PPR + w_p*pointer - w_d*dist)
   │  enough ─► Assemblage.sufficiency_router (Nexus/Halo -> Chronicle -> Foundation)
Answer (use)
   └─► Assemblage.answerer      (LLM allowed here, and only here for generation)
Learning (evaluate -> update)
   ┌─► MPE.predictor   U_predicted (pre-retrieval channels only)
   ├─► MPE.utility     U_actual    (evidence/answer/retrieval/token terms)
   ├─► MPE.evaluator   MPE = U_actual - U_predicted (clipped)
   └─► MPE.updater     P' = clip(P + eta*MPE, 0, 1)  -> interfaces.StorageBackend
Forgotten-but-not-deleted
   └─► entropy.context_decay / pointer_decay -> lower pointer, keep evidence
```

The closed loop: `WRITE -> INDEX -> RETRIEVE -> USE -> EVALUATE -> UPDATE`
(V1.md §4.2) is executed by `demerzel.engine.Engine.query`.

## Module boundary discipline

Each core mechanism is a class wired by **constructor injection** and depends
only on an interface — never on a reference implementation or on a sibling
mechanism:

| mechanism | depends only on |
|-----------|-----------------|
| `PointerDecay` | `interfaces.StorageBackend` |
| `ContextDistance` | `interfaces.EmbeddingProvider` |
| `MPEFeedback` | `interfaces.PointerUpdater` |
| `EventBoundary` | `interfaces.LLMProvider` (optional) |
| `NexusBuilder` | `interfaces.GraphBackend` |
| `Retriever`/`Scorer` | `interfaces.EmbeddingProvider`, `GraphBackend`, `StorageBackend` |

Consequence: swapping the reference implementations (SQLite -> memory, hashed
embedder -> real model, DictGraph -> NetworkX) changes nothing above the one
`build_default_engine` factory.

## The interface contracts (V1.md §6)

- `StorageBackend`: `get / put / update_pointer / list_by_session`. `update_pointer`
  must touch **only** `pointer_strength` (Invariant 2).
- `GraphBackend`: `add_node / add_edge / neighbors / ppr`. `MPEFeedback` never
  calls `add_node`/`add_edge` (Invariant 3).
- `PointerUpdater`: `update(memory_id, mpe)` touches only the pointer.
- `LLMProvider`: `generate / classify` -> structured `ClassificationDecision`.
- `EmbeddingProvider`: `embed(texts)` -> vectors.

## Data flow for one query

1. `Engine.query(query)` routes the query (lexical/vector/graph).
2. Retrieval returns scored `Candidate`s via the hybrid scorer.
3. Sufficiency decides answer-or-widen.
4. Context builder assembles a passage; answerer generates (extractive or LLM).
5. `ContextDecay` applies the accessibility drift to non-retired pointers.
6. MPE step on the used candidate updates its pointer (positive/negative).
7. The full record is appended to the run's `retrieval.jsonl` / `answers.jsonl`.

## Observability (V1.md §16)

Every query answers "why this memory, why not that one" because the trace
records each candidate's `semantic_score`, `ppr_score`, `pointer_strength`,
`context_distance`, `final_score`, the `used_evidence`, the `mpe`, and the
`pointer_updates`. See `docs/example_trace.json`.

## Ablations (V1.md §11.2)

Config fields, not code paths:

- `ablations.decay` -> `Config.apply_ablations` -> `decay.enabled`
- `ablations.mpe`   -> `mpe.enabled`
- `ablations.graph` -> `retrieval.graph_enabled`

A single switch toggles each mechanism; turning "graph off" disables the Nexus
backend without deleting Nexus (`graph off as an ablation`).

## Reference implementations (replaceable)

| reference | backend | dependency |
|-----------|---------|-----------|
| `storage_sqlite` | SQLite (WAL, FULL sync) | stdlib |
| `storage_memory` | dict | stdlib |
| `graph_networkx` | adjacency dict + power-iteration PPR | stdlib |
| `providers_mock` | hashing embeddings + token-overlap LLM | stdlib + numpy |

> **Deviation note (V1.md §3.1):** NetworkX and PyYAML/pytest are named in V1.md
> but cannot be installed in this offline workspace. The GraphBackend interface
> is preserved; `DictGraph` is a dependency-free PPR implementation behind it.
> `pytest` is unnecessary because the suite runs on `unittest`; swap the
> reference module later without touching any mechanism.

## Failure handling (V1.md §17)

- Write gate `fail-open`: LLM/gist failure still writes, marks `gate=unavailable`.
- Recall gate: degraded to local ordering when the provider is down.
- Injection gate `fail-closed`: an unavailable answerer injects nothing.
- **LLM unavailable never causes memory loss.**