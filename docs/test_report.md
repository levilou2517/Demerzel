# Test Report — Demerzel V0.1.0

**Scope.** Phase 0–6, the P0 acceptance gate (V1.md §11.1) and invariants 1–8
(V1.md §10). Generated from actually-saved artifacts; see `results/`.

**Runner.** `python tests/run_tests.py` (stdlib `unittest`) — 78 tests, **all
passing**. `pytest -q` collects the same classes whenever pytest is installed.

## Command

```sh
cd /home/levi/AI/Demerzel
python tests/run_tests.py
```

```
Ran 78 tests in 0.702s
OK
```

## Test inventory

| file | tests | covers |
|------|------:|--------|
| `test_p0_acceptance.py` | 17 | P0-1 … P0-9 |
| `test_invariants.py` | 11 | invariants 1–8 + topology-writeback |
| `test_invariant7_cold_start.py` | 7 | cold-start rule (Phase-0 gate) |
| `test_invariant8_mpe_timing.py` | 6 | `U_predicted` excludes retrieval |
| `test_accessibility.py` | 9 | context distance, decay, MPE |
| `test_mechanisms.py` | 9 | Nexus, Halo, storage references |
| `test_foundation_and_degradation.py` | 9 | Foundation, §17 degradation |
| `test_experiments_and_cli.py` | 10 | Exp 1–4, CLI acceptance |
| **total** | **78** | |

## P0 acceptance gate (V1.md §11.1)

| # | requirement | result |
|---|-------------|--------|
| P0-1 | Persistence: decay 100× → Foundation/Chronicle content unchanged | **PASS** |
| P0-2 | Pointer decay: d=0.5 → pointer falls, memory still exists | **PASS** |
| P0-3 | Resurrection: pointer ≈ 0 + strong query → retrievable | **PASS** |
| P0-4 | MPE: U_actual > U_pred → pointer rises; else falls | **PASS** |
| P0-5 | Graph isolation: after MPE update, graph_before == graph_after | **PASS** |
| P0-6 | Provenance: Chronicle → Foundation → original turn | **PASS** |
| P0-7 | Crash safety: kill mid-write/update → restart, no silent damage | **PASS** |
| P0-8 | Cold start: halo empty → context_distance == 0 | **PASS** |
| P0-9 | MPE timing: U_predicted contains no rank_score | **PASS** |

All nine P0 items pass.

## Invariants 1–8 (V1.md §10)

| # | invariant | result |
|---|-----------|--------|
| 1 | Pointer decay does not delete evidence | **PASS** |
| 2 | Pointer update does not change evidence content | **PASS** |
| 3 | MPE-only does not change graph topology | **PASS** |
| 4 | Halo eviction does not delete Chronicle | **PASS** |
| 5 | Cold memory can be reactivated | **PASS** |
| 6 | Explicit user deletion is distinct from forgetting | **PASS** |
| 7 | Context distance cold start degrades to 0 | **PASS** |
| 8 | U_predicted contains no retrieval result | **PASS** |

## Experiments (results in `results/`)

### Exp 1 — decay mechanism comparison
Key contrast C (discrete boundary) vs D (continuous context distance):

```
ndcg_delta_D_minus_C = +0.040
mrr_delta_D_minus_C  = +0.133
recall_delta_D_minus_C = 0.000
```

The continuous context-distance mode edges out the discrete boundary mode on
nDCG and MRR on the synthetic corpus. **Direction consistent with §8.1's
prediction**, though on a single synthetic corpus this is illustrative, not a
benchmark result.

### Exp 2 — the MPE closed loop (the key experiment)

Isolation assertion, read from the saved artifacts:

```
G_fixed = true      (graph.topology_fingerprint unchanged)
E_fixed = true      (Foundation evidence digest unchanged)
A_mutable = true    (pointer_strength moved across the pool)
```

**Synthetic prerequisite.** Fixed query, target pointer pinned 1.0 → 0.1:

```
pointer 1.0  -> rank 1
pointer 0.75 -> rank 1
pointer 0.5  -> rank 1
pointer 0.25 -> rank 2
pointer 0.1  -> rank 5
base_effect_present = true
```

### Substantive negative finding: the prerequisite depends on the weight

When the accessibility prior `w_pointer` is **too small relative to the
semantic weight**, pinning the pointer does **not** reorder retrieval — the
semantic channel dominates:

```
w_pointer 0.2 / 0.5 -> rank stays 1 at both 1.0 and 0.1  (effect absent)
w_pointer 1.0       -> rank 1 -> 5                        (effect present)
w_pointer 2.0       -> rank 1 -> 8                        (effect present)
```

This is exactly the config reality §8.2's prerequisite ablation exists to catch.
The default `RetrievalConfig` was therefore set to `w_pointer = 1.0` (with
`w_semantic = 1.0`, `w_ppr = 0.6`, `w_context_distance = 0.4`) so the central
claim is *testable* out of the box. **The effect being config-fragile is
itself a finding to report, not a bug to paper over** — it means "accessibility
governs retrieval" only once `A`'s term is weighted commensurately with the
semantic term. The runner reports `weight_sensitivity` on every Exp-2 run.

MPE group comparison (A disabled vs B enabled) shows identical recall/MRR on
this small synthetic corpus while pointers move: `A_mutable = true`,
`G_fixed = E_fixed = true`. On real data the improvement is expected to show in
retrieval quality or token efficiency, which requires the Phase-8 adapters.

### Exp 3 — cold pointer revival

```
hard_delete_prevents_recovery      = true
soft_forgetting_preserves_recovery = true   (context / context+MPE groups)
```

Group A (hard delete) cannot recover the target; groups C/D (soft forgetting)
keep it retrievable. Consistent with the "soft forgetting preserves revival"
claim and with §8.3's selective-forgetting contrast.

### Exp 4 — retrieval modality routing

Groups A (always graph), B (routed), C (graph off) all run and write metrics;
the routing-off ablation works as a switch, not a deletion.

## Coverage notes

- Deterministic mock providers make every experiment byte-reproducible
  (`test_experiments_and_cli.test_all_experiments_are_reproducible`).
- SQLite WAL + `synchronous=FULL` + transactional rollback verified under
  P0-7 and `test_mechanisms.TestStorageReference`.
- §17 degradation paths tested directly: write gate fails open, LLM-boundary
  falls back algorithmically, answerer degrades to extractive.

## Stop condition (V1.md §13)

> **完成 P0 后停止.**

P0 passes and the invariants hold, so the build **stops here**. P1/P2 (retrieval
trace completeness on real data, LongMemEval/LoCoMo adapters, benchmark
reproduction) are gated on human review of two things:

1. whether invariants 1–8 hold beyond this synthetic setup, and
2. whether the MPE loop produces an interpretable pointer trajectory on real
   data — with the Exp-2 prerequisite kept visibly at the top of that decision.