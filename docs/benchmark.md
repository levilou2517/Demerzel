# Benchmark Instructions

How to run the MVP acceptance surface and attach an external benchmark.
Demerzel does **not** embed an evaluation framework (V1.md §5.5): external
tools consume its trace format.

## 1. Environment

The core requires only the Python standard library plus `numpy`:

```sh
python3 -c "import sys, sqlite3, numpy; print(sys.version)"
```

Optional extras (see `pyproject.toml`):

```sh
pip install -e .              # base: numpy
pip install -e '.[test]'      # adds pytest
pip install -e '.[networkx]'  # original graph reference backend
```

> This workspace is offline: `networkx`, `pytest` and `PyYAML` cannot be
> installed. The suite therefore runs on `unittest`
> (`python tests/run_tests.py`) and `DictGraph` supplies PPR behind the
> unchanged `GraphBackend` interface. See `docs/architecture.md` §"Deviation".

## 2. Install and smoke test

```sh
pip install -e .
demerzel init
demerzel ingest examples/demo.json
demerzel query "What is the capital of France?"
demerzel inspect-memory mem_turn_s1_000001
```

Expected: the query prints a trace with `candidates`, `mpe` and
`pointer_updates`; `inspect-memory` shows `provenance_ok: true`.

## 3. Acceptance commands (V1.md §20)

```sh
demerzel init
demerzel ingest examples/demo.json
demerzel query "..."
demerzel inspect-memory <memory_id>
demerzel inspect-trace <run_id>
demerzel run-experiment exp1
demerzel run-experiment exp2
python tests/run_tests.py        # or: pytest
```

You should observe: memory exists, pointer changed, graph unchanged, retrieval
trace exists, MPE exists, experiment result exists.

## 4. Full test suite

```sh
python tests/run_tests.py -v          # stdlib only
pytest -q                             # when pytest is installed
```

Both collect the same `unittest.TestCase` classes.

## 5. Experiments

```sh
./scripts/run_exp1.sh                 # decay comparison  -> results/exp1/
./scripts/run_exp2.sh                 # MPE loop (key)    -> results/exp2/
python -m demerzel.cli run-experiment exp3   # cold revival
python -m demerzel.cli run-experiment exp4   # modality routing
```

Each writes `config.json`, `metadata.json`, `metrics.json` and the JSONL
traces under `results/<exp>/run_<timestamp>/`.

## 6. Attaching an external benchmark

An external harness needs only two things: a way to **load** items and a way to
**read** Demerzel's answers.

**Load.** Ingest items with `Engine.ingest_batch(session_id, turns)`, where each
turn is `{"text": str, "speaker": str, "metadata": {...}}`. Use one
`session_id` per benchmark conversation.

**Read.** After each query, `retrieval.jsonl` holds one row per query with the
V1.md §5.5 shape:

```json
{
  "run_id": "...", "query": "...",
  "candidates": [{"memory_id": "...", "semantic_score": 0.82, "ppr_score": 0.31,
                  "pointer_strength": 0.62, "context_distance": 0.44, "final_score": 0.71}],
  "used_evidence": ["mem_17"], "mpe": 0.14,
  "pointer_updates": [{"memory_id": "...", "old": 0.62, "new": 0.65, "reason": "positive_mpe"}]
}
```

`answers.jsonl` holds the generated answer plus its `used_evidence`.
`pointer_updates.jsonl` is the accessibility trajectory.

**Verdicts.** Compute Recall@K / MRR / nDCG with `demerzel.evaluation.metrics`
or your own code over the `candidates` ranking.

### LongMemEval / LoCoMo adapters (Phase 8)

The adapters are deliberately thin (V1.md Phase 8): `load -> ingest -> query ->
evaluate -> export`. A skeleton lives at `demerzel/evaluation/` — the shipped
package exposes the metric helpers and the experiment runner; a benchmark
adapter adds dataset loading and answer-equivalence scoring on top without
changing any mechanism.

## 7. Reproducibility checklist (V1.md §14)

For every reported experiment confirm `results/<exp>/run_*/metadata.json` holds:

- [ ] `system_version`, `config_hash`
- [ ] `llm_model`, `embedding_model`, `temperature`
- [ ] `seed`, `dataset_version`
- [ ] `timestamp`

If any is missing, the result is not reproducible and must not be claimed.