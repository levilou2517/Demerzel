# Experiment Protocols

Protocols, configs and metrics for Experiments 1–4 (V1.md §8), plus the
reproducibility contract (§14) and the acceptance surface (§11).

All experiments are implemented in `demerzel/evaluation/experiments.py` and
invoked through `demerzel run-experiment <name>` or `./scripts/run_exp<N>.sh`.

## Shared controls

V1.md §8.1 requires the controls be identical across groups: LLM, embedding,
retrieval budget, answer generator, dataset. In this prototype those are all
deterministic mocks and are recorded in every run's `metadata.json`:

```
system_version, config_hash, llm_model, embedding_model,
temperature, seed, dataset_version, timestamp
```

`config_hash` is a SHA-256 digest over the canonical config, so two runs that
differ in any knob cannot collide.

## Exp 1 — decay mechanism comparison

**Question.** Does a *continuous* context distance carry more information than
a *discrete* event boundary?

| group | decay driver | mode |
|-------|--------------|------|
| A | none | `decay.mode = "none"` |
| B | time (Ebbinghaus) | `"time"` |
| C | event boundary (HingeMem-style, `d ∈ {0,1}`) | `"boundary"` |
| D | continuous context distance (`d ∈ [0,1]`) | `"context"` |
| E | D + MPE | `"context"` + `mpe.enabled` |

**Key contrast.** C vs D. **Metrics.** Recall@5, MRR, nDCG@5, token cost.
**Expected.** If D beats C on multi-session reasoning, continuous drift retains
more information than a discrete boundary.

The reported `contrast_C_vs_D` block gives the D-minus-C deltas; a positive
delta favours the continuous mode.

## Exp 2 — MPE pointer-update loop (the key experiment)

**Question.** Can accessibility `A` be learned while `G` and `E` stay fixed?

**Core constraint.**

```
G = constant
E = constant
A = mutable
```

| group | mechanism |
|-------|-----------|
| A | MPE disabled |
| B | MPE enabled |

**Mandatory synthetic prerequisite (V1.md §8.2).** With a fixed query, manually
pin the target memory's `pointer_strength` from 1.0 down to 0.1 and observe the
ranking change. **If this base effect does not exist, a real-benchmark effect
cannot exist either.**

The runner reports:

- `pinned_pointer_probe.ranks_by_pointer` — rank at 1.0/0.75/0.5/0.25/0.1;
- `pinned_pointer_probe.base_effect_present` — whether the rank moved;
- `pinned_pointer_probe.weight_sensitivity` — the same probe swept across
  `w_pointer ∈ {0.2, 0.5, 1.0, 2.0}`, because a small accessibility-prior weight
  lets a strong semantic match swamp the pointer term. **A False base effect is
  a config finding to report, not a bug to hide**; see `docs/test_report.md`.

**Isolation assertion.** For both groups the runner compares
`graph.topology_fingerprint()` and a Foundation evidence digest before/after,
and counts moved pointers. `G_fixed`, `E_fixed`, `A_mutable` come from those
measurements; `hypothesis_supported` requires all three **and** the
prerequisite.

## Exp 3 — cold pointer revival

**Question.** Does soft forgetting preserve future recoverability?

| group | mechanism |
|-------|-----------|
| A | hard delete (baseline) |
| B | time decay |
| C | context-distance decay |
| D | context-distance decay + MPE |

**Procedure.** Make memory A salient, then simulate long non-access, then ask a
new query that makes A relevant again. **Metrics.** recovery rate, recovery
latency, false recall rate, token cost.

Group A is the "Selective Forgetting" pruning baseline Demerzel argues is
irreversible; it uses explicit storage deletion, which is *not* a forgetting
path (Invariant 6).

## Exp 4 — retrieval modality routing

**Question.** Should Nexus be consulted selectively by query type?

| group | mechanism |
|-------|-----------|
| A | every query through Nexus PPR |
| B | classified, routed on demand |
| C | graph off (ablation) |

**Metrics.** accuracy, latency, tokens, retrieval recall.

Group C is the `graph off` **ablation**, never "delete Nexus" — the mechanism is
switched, not removed.

## Output layout (V1.md §14)

```
results/
  <exp>/run_<timestamp>/
    config.json          # full config, canonical
    metadata.json        # reproducibility block + group label
    metrics.json         # the metric block for this group
    retrieval.jsonl      # one row per query (V1.md §5.5 schema)
    answers.jsonl
    pointer_updates.jsonl
    events.jsonl
```

## Metrics (V1.md §15)

`demerzel/evaluation/metrics.py` implements Recall@K, MRR, nDCG@K, accuracy,
token cost, latency stats, and an MPE distribution summary. Memory storage
size, pointer update count, cold recovery rate and false recall rate are
reported alongside each experiment's block.

## Claim discipline

> **Never claim an experimental improvement unless the experiment has actually
> been executed and the raw results are saved.** (V1.md §22)

Every number in `docs/test_report.md` is read back from a saved artifact under
`results/`.