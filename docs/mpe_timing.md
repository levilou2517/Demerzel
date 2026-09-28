# MPE Timing Specification

Precisely defines when `U_predicted` is computed and what may feed it
(V1.md §9.2, Invariant 8 / P0-9).

## Timeline (V1.md §7.3)

```
t0  query arrives
t1  U_predicted computed  (ONLY pre-retrieval inputs)
t2  retrieval executes
t3  answer generated
t4  U_actual computed     (depends on t2/t3 outcomes)
t5  MPE = U_actual - U_predicted
t6  pointer update
```

## U_predicted inputs (Invariant 8 / P0-9)

`U_predicted` MUST be computed **before** retrieval happens. Its inputs may be
**only**:

- `semantic_similarity(query, memory)` — from embeddings,
- `pointer_strength` — the current value,
- `context_distance` — the current value.

`rank_score` / retrieval output MUST NOT appear in `U_predicted`.

## Enforcement in code

`MPEPredictor.predict(memory_id, similarity, pointer, context_distance)` takes
exactly those three scalar channels and **no ranking object**. The signature
is the guarantee: there is no way to pass retrieval output into it. In
`Engine.query`, the predictor is called against the top candidate using
`vector.similarity(query, memory)` (a pure embedding comparison), the stored
pointer, and the context distance — *before* any `used_evidence`/rank is relied
on for the scoring.

## U_actual (t4)

Computed after retrieval + answer from (V1.md §7.3):

```
U_actual = w1*evidence_score + w2*answer_delta + w3*retrieval_delta - w4*token_cost
```

Phase-0 defaults: `evidence_weight = 1.0`, all others `0.0` (the V1.md §7.3
block). The `evidence_score` for the top used candidate is its semantic
similarity to the query — a post-retrieval measurement.

## MPE and update (t5–t6)

- `MPE = U_actual - U_predicted` (`MPEEvaluator`), clipped to `[-mpe_clip, mpe_clip]`.
- `P_next = clip(P_current + eta * MPE, 0, 1)` (`EtaPointerUpdater`).
- The update touches **only** `pointer_strength` (Invariant 2), and the MPE
  path never writes graph topology (Invariant 3).
- When MPE is disabled (`eta=0`), the pointer is returned unchanged and logged
  `reason="mpe_disabled"`, so the Exp-2 control is a pure config change.

## Test

`tests/test_invariant8_mpe_timing.py` asserts the predictor signature has no
ranking input and that `U_predicted` excludes `rank_score` (P0-9).