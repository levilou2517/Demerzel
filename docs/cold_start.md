# Cold Start Specification

Spec: `context_distance` when the Halo is empty or the conversation-embedding
turn count is insufficient (V1.md §9.1, Invariant 7).

## Rule

> When the Halo is empty **or** the conversation turn count is less than `K`:

```
context_distance = 0
pointer_decay   applies no decay penalty
reason = "cold_start"
```

`K` defaults to 3 and is a config field (`context_distance.cold_start_k`).

## Where it is enforced

1. **`ContextDistance.is_cold_start(state_vector, turn_count)`** — returns
   `True` when the state vector is empty (Halo empty / no conversation
   embedding) or `turn_count < K`.

2. **`ContextDistance.compute(...)`** — short-circuits to
   `ContextDistanceResult(0.0, "cold_start", cold_start=True)`.

3. **`PointerDecay.factor(distance, reason="cold_start", ...)`** — returns
   factor `1.0` for `reason == "cold_start"`, so `new_pointer = old_pointer`.

4. **`ContextDecay.run(...)`** — the orchestration layer records
   `reason="cold_start"` for every record when the rule fires, so no decay
   applies and the reason is observable.

## Deterministic contract

| Halo/context | turn_count | context_distance | reason      | decay factor |
|--------------|-----------|------------------|-------------|--------------|
| empty        | any       | 0.0              | cold_start  | 1.0          |
| non-empty    | < K       | 0.0              | cold_start  | 1.0          |
| non-empty    | >= K      | `1 - cosine(...)`| computed    | `exp(-d/S)`  |

The cold-start rule is an *accessibility* statement: with no retrieval history
yet, every memory is "near" by construction and nothing is penalised for not
having been revisited. Disable via `context_distance.cold_start_enabled = false`
to run the other mode for control comparison (though defaults keep it on).

## Test

`tests/test_invariant7_cold_start.py` (see `docs/test_report.md`).