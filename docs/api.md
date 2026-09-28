# Demerzel API

Public surface for embedding Demerzel in an experiment or an application.
Everything here is importable from the package; nothing requires reading the
mechanism internals.

## Quick start

```python
from demerzel.config import Config
from demerzel.engine import build_default_engine

engine = build_default_engine(Config())

engine.ingest("s1", "The capital of France is Paris.")
engine.ingest("s1", "Water boils at 100 degrees Celsius.")

trace = engine.query("What is the capital of France?")
print(trace["candidates"][0]["memory_id"], trace["mpe"])
```

## `demerzel.config`

| symbol | purpose |
|--------|---------|
| `Config` | the whole run configuration; `config_hash()`, `to_dict()`, `from_dict()`, `apply_ablations()` |
| `load_config(path)` / `save_config(cfg, path)` | JSON always; YAML when PyYAML is present |
| sub-configs | `StorageConfig`, `DecayConfig`, `MPEConfig`, `RetrievalConfig`, `ContextDistanceConfig`, `UtilityConfig`, `HaloConfig`, `SufficiencyConfig`, `TraceConfig` |

Ablation switches: `config.ablations = {"decay": bool, "mpe": bool, "graph": bool}`.

## `demerzel.engine`

| symbol | purpose |
|--------|---------|
| `Engine` | composes the layers and runs the core loop |
| `build_default_engine(config=None, ...)` | wires the reference implementations |

`Engine` methods:

- `ingest(session_id, text, speaker="user", metadata=None) -> memory_id`
- `ingest_batch(session_id, turns) -> list[memory_id]`
- `retrieve(query, state_vector=None) -> RetrievalResult`
- `query(query, session_id="default") -> dict` — the full trace record
- properties: `storage`, `graph`, `embedder`, `evidence`, `ring`, `mpe`, `config`, `memory_count`

## `demerzel.interfaces`

The seams. Implement these to replace any backend.

- `StorageBackend` — `get`, `put`, `update_pointer`, `list_by_session`, `list_all`
- `EvidenceStore` — `append`, `get_turn`, `iter_turns`, `count`
- `GraphBackend` — `add_node`, `add_edge`, `get_node`, `neighbors`, `ppr`, `nodes`, `edges`, `topology_fingerprint`
- `PointerUpdater` — `update(memory_id, mpe)`, `current(memory_id)`
- `LLMProvider` — `generate`, `classify`, `available`
- `EmbeddingProvider` — `embed`, `dimension`
- dataclasses: `MemoryRecord`, `Node`, `Edge`, `PointerUpdate`, `ClassificationDecision`, `LLMCallPolicy`, `ProviderConfig`
- constants: node kinds (`NODE_EPISODE`…), edge kinds (`EDGE_CAUSAL`…), memory types

`EtaPointerUpdater` is the stock `PointerUpdater` implementation
(`P' = clip(P + eta*MPE, 0, 1)`).

## `demerzel.core`

- `ContextDistance(embedder, cold_start_k, cold_start_enabled)` — `.compute(text, state_vector, turn_count) -> ContextDistanceResult`, `.is_cold_start(...)`
- `PointerDecay(storage, scale, mode, enabled)` — `.factor(distance, reason)`, `.apply(memory_id, distance, reason) -> DecayResult`

## `demerzel.mpe`

- `MPEPredictor` — `predict(memory_id, similarity, pointer, context_distance) -> PredictedUtility`
- `UtilityEvaluator` — `evaluate(evidence_score, answer_delta, retrieval_delta, token_cost) -> ActualUtility`
- `MPEEvaluator` — `evaluate(memory_id, u_predicted, u_actual) -> MPEResult`
- `MPEFeedback` — `.step(...)`, `.apply(memory_id, mpe)`, plus `build_default_mpfe(storage, ...)`

## `demerzel.nexus`

- `NexusBuilder` — `add_memory_node`, `link_episode`, `build_from_records`
- `PPRRetriever` — `.score(seed_node_ids)`, `.memory_scores(seed_node_ids)`
- `LexicalRetriever` — `.score(query, records)` (BM25)
- `VectorRetriever` — `.score(query, records)`, `.similarity(text, record)`
- `HybridScorer` — `.score(records, lexical, vector, ppr, distances, backend) -> list[Candidate]`
- `NexusPointerWriteback` — topology-neutral edge activation

## `demerzel.halo`

- `RingBuffer(max_turns)` — `append`, `peek`, `all`, `snapshot`, `stats`, `evicted`
- `EvictionPolicy(policy)` — `"evict" | "merge" | "retain"`
- `ContextState(embedder, alpha, beta)` — `state_vector(conversation, halo)`

## `demerzel.entropy`

- `ContextDecay(storage, context_distance, pointer_decay, forgetting_log)` — `.run(records, state_vector, turn_count)`
- `ForgettingLog(path=None)` — `record`, `by_memory`, `rows`
- `ColdPointerProbe(ranker)` — `.probe(...)`, plus `RecoveryReport`

## `demerzel.assemblage`

- `QueryRouter` — `.route(query) -> RouteDecision`
- `SufficiencyRouter` — `.assess(result, depth) -> SufficiencyDecision`
- `ContextBuilder` — `.build(query, candidates, records_by_id) -> BuiltContext`
- `Answerer` — `.answer(query, passage, used_evidence) -> Answer`
- `Reranker` — `.rerank(candidates) -> RerankOutcome` (identity by default)

## `demerzel.trace`

- `RunTrace(run_dir, run_id, config_hash)` — `log_retrieval`, `log_answer`, `log_pointer_update`, `log_event`, `write_metadata`, `write_metrics`
- `JsonlWriter`, `read_jsonl`

## `demerzel.evaluation`

- `metrics` — `recall_at_k`, `mrr`, `ndcg_at_k`, `accuracy`, `token_cost`, `latency_stats`, `mpe_distribution`
- `run_experiment(name, config, results_root)` and `experiment_1..4`
- `CORPUS`, `PROBES`

## `demerzel.reference`

Replaceable implementations: `SqliteStorage`, `SqliteEvidenceStore`,
`InMemoryStorage`, `InMemoryEvidenceStore`, `DictGraph`,
`HashingEmbeddingProvider`, `MockLLMProvider`, `MockLLMProviderUnavailable`.

## CLI

```
demerzel init
demerzel ingest examples/demo.json [--session s1]
demerzel query "..." [--session s1]
demerzel inspect-memory <memory_id>
demerzel inspect-trace <run_id>
demerzel run-experiment exp1|exp2|exp3|exp4
```

Global flags: `--config <file>`, `--run-dir <dir>`.