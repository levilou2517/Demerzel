# Invariant Specification

The eight invariants of V1.md §10, each with its precise statement, the code
that enforces it, and the test that proves it.

An invariant is a contract. **Never remove or weaken one to make a test pass.**

---

## Invariant 1 — Pointer decay does not delete evidence

**Statement.** Applying pointer decay N times leaves every Foundation turn and
every Chronicle record present and byte-identical in content.

**Enforced by.** `PointerDecay.apply` calls only
`StorageBackend.update_pointer`, whose contract is pointer-only.
`RawStore` exposes no delete method at all.

**Test.** `tests/test_invariants.py::test_invariant_1_decay_preserves_evidence`
(decay 100×, assert record count and content hash unchanged).

---

## Invariant 2 — Pointer update does not change evidence content

**Statement.** After any pointer update, `content`, `source_ptr` and
`created_at` are unchanged; only `pointer_strength` differs.

**Enforced by.** `StorageBackend.update_pointer` implementations write exactly
one column/field (`reference/storage_sqlite.py` uses a single-column UPDATE;
`reference/storage_memory.py` replaces only the pointer field).

**Test.** `test_invariant_2_pointer_update_preserves_content` plus the
`union_roundtrip` assertion for both reference backends.

---

## Invariant 3 — MPE-only does not change graph topology

**Statement.** Under `G = constant, E = constant, A = mutable`, running the MPE
update changes `pointer_strength` values but leaves the graph topology
identical (`graph_before == graph_after`).

**Enforced by.** `MPEFeedback` depends only on `PointerUpdater`; it never calls
`GraphBackend.add_edge` or `add_node`. The graph-side writeback
(`NexusPointerWriteback`) touches only edge *metadata*
(`co_activation_count`, `last_activated`), which is excluded from
`topology_fingerprint()` **by design** — so topology identity is the audited
quantity, not edge timestamps.

**Test.** `tests/test_p0_acceptance.py::test_p0_5_graph_isolation`, and the
engine-level assertion in `test_invariant_3_...`.

---

## Invariant 4 — Halo eviction does not delete Chronicle

**Statement.** Evicting turns from the halo ring leaves every persistent
Chronicle record intact.

**Enforced by.** `RingBuffer` evicts only from its own `deque`; it holds no
reference to storage. `Refine`/`EvictionPolicy` touch only the ring.

**Test.** `test_invariant_4_halo_eviction_preserves_chronicle` (overflow the
ring, assert storage record count and contents unchanged).

---

## Invariant 5 — Cold memory can be reactivated

**Statement.** A memory whose pointer has decayed near zero can still surface
for a strongly matching query.

**Enforced by.** Retrieval never filters on a pointer threshold. Pointer
enters the score as a weighted *additive* term (`w_pointer * pointer`), so a
near-zero pointer reduces but does not eliminate a candidate; the semantic and
PPR channels can carry it back.

**Test.** `test_invariant_5_cold_memory_reactivates` and P0-3.

---

## Invariant 6 — Explicit user deletion is distinct from forgetting

**Statement.** The forgetting mechanism never deletes. Explicit user deletion
exists, is separate, and is not reachable from any forgetting code path.

**Enforced by.** No forgetting module imports or calls a delete. Removal lives
at the storage/CLI level (`InMemoryStorage.delete`, an audited SQL DELETE) and
is used only by the Experiment-3 group-A baseline, which represents the
hard-delete approach Demerzel argues against.

**Test.** `test_invariant_6_delete_is_separate_from_forgetting` (assert
`RawStore` offers no delete, assert a 100-step decay cycle removes nothing).

---

## Invariant 7 — Context distance cold start degrades to 0

**Statement.** When the halo is empty or the conversation turn count is below
`K`, `context_distance == 0` and no decay penalty is applied
(`reason == "cold_start"`).

**Enforced by.** `ContextDistance.is_cold_start` / `.compute`, and
`PointerDecay.factor(..., reason="cold_start") -> 1.0`.

**Test.** `tests/test_invariant7_cold_start.py` (the Phase-0 gate).

---

## Invariant 8 — U_predicted contains no retrieval result

**Statement.** `U_predicted` is computed before retrieval and its inputs are
only `semantic_similarity`, `pointer_strength`, and `context_distance`.
`rank_score` never feeds it.

**Enforced by.** `MPEPredictor.predict`'s signature accepts exactly those three
scalars and no ranking object.

**Test.** `tests/test_invariant8_mpe_timing.py` (signature audit + P0-9).

---

## Invariant → acceptance-test map

| Invariant | P0 item | Test file |
|-----------|---------|-----------|
| 1 | P0-1, P0-2 | `test_invariants.py`, `test_p0_acceptance.py` |
| 2 | P0-1, P0-4 | `test_invariants.py`, `test_p0_acceptance.py` |
| 3 | P0-5 | `test_p0_acceptance.py` |
| 4 | — (P1) | `test_invariants.py` |
| 5 | P0-3 | `test_p0_acceptance.py`, `test_invariants.py` |
| 6 | — | `test_invariants.py` |
| 7 | P0-8 | `test_invariant7_cold_start.py` |
| 8 | P0-9 | `test_invariant8_mpe_timing.py` |