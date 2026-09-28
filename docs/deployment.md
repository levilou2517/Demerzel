# Deployment Guide

How to get Demerzel v0.1.0 running on another machine, and what it takes to
reproduce a result someone else published.

Demerzel is a **research prototype**, not a service: it is a Python library plus
a CLI. "Deploying" means installing the package into a Python environment and
running commands — there is no server, no database service, and no network
dependency at runtime.

---

## 1. Requirements

| requirement | version | notes |
|-------------|---------|-------|
| Python | **3.10+** | 3.12 is what CI/dev used |
| pip | any recent | with `setuptools` + `wheel` |
| OS | Linux / macOS / Windows | no OS-specific code |
| network | **none at runtime** | offline-capable; only `pip install` needs the network |
| disk | ~1 MB source + your data | SQLite file grows with the corpus |

`git` is needed only to clone. No GPU, no model download, no API key: the
default providers are deterministic mocks, so a fresh install runs immediately.

> **Package layout note.** `pyproject.toml` declares `numpy>=1.24`. The current
> codebase does not import numpy anywhere — the package is stdlib-only — but the
> dependency is declared deliberately and pip will fetch it. If you are on a
> restricted network, see §7 "Offline install".

---

## 2. Install

### From GitHub (recommended)

```sh
git clone https://github.com/levilou2517/Demerzel.git
cd Demerzel

python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install --upgrade pip setuptools wheel
pip install -e .
```

The `demerzel` console command is now on your PATH.

> **Why `pip install --upgrade setuptools wheel` first?** Python 3.12 venvs no
> longer bundle `setuptools`, and the editable build needs it. If you skip this
> you may see `ModuleNotFoundError: No module named 'setuptools'`.

### Without installing (zero-install run)

The package is importable straight from the checkout, so you can skip pip
entirely:

```sh
cd Demerzel
export PYTHONPATH="$PWD:$PYTHONPATH"
python3 -m demerzel.cli --help
```

This is the path used when pip cannot reach an index.

### Verify the install

```sh
demerzel --help
python3 tests/run_tests.py            # expect: Ran 78 tests ... OK
```

---

## 3. First run (2 minutes)

```sh
mkdir -p ~/demerzel-work && cd ~/demerzel-work

demerzel init                                   # writes demerzel.config.json
demerzel ingest /path/to/Demerzel/examples/demo.json
demerzel query "What is the capital of France?"
demerzel inspect-memory mem_turn_s1_000001
```

Expected:

- `init` creates `data/`, `results/`, `examples/` and `demerzel.config.json`.
- `ingest` prints one `mem_turn_*` id per turn.
- `query` prints a JSON trace containing `candidates`, `mpe` and
  `pointer_updates` — the top candidate should be `mem_turn_s1_000001`.
- `inspect-memory` prints the record plus `"provenance_ok": true`, proving the
  memory resolves back to its Foundation turn.

---

## 4. Run the experiments

```sh
demerzel run-experiment exp1     # decay comparison        -> results/exp1/
demerzel run-experiment exp2     # the key MPE experiment  -> results/exp2/
demerzel run-experiment exp3     # cold pointer revival
demerzel run-experiment exp4     # retrieval modality routing
```

Or use the reproduction scripts:

```sh
./scripts/run_exp1.sh
./scripts/run_exp2.sh
```

Each run writes a self-describing directory:

```
results/<exp>/run_<timestamp>/
  config.json              # the full config, canonical
  metadata.json            # reproducibility block (see §6)
  metrics.json             # the metric block for that group
  retrieval.jsonl          # one row per query
  answers.jsonl
  pointer_updates.jsonl
  events.jsonl
```

**Exp-2 check.** The output includes:

```
hypothesis_supported   # requires the prerequisite AND G/E fixed AND A mutable
comparison.G_fixed     # graph topology fingerprint unchanged
comparison.E_fixed     # Foundation evidence digest unchanged
comparison.A_mutable   # pointer_strength values moved
```

If `pinned_pointer_probe.base_effect_present` is `false`, read
`pinned_pointer_probe.weight_sensitivity`: the accessibility prior
(`retrieval.w_pointer`) is too small relative to the semantic weight for the
pointer to reorder retrieval. Raise `w_pointer` and re-run — this is a config
finding, documented in `docs/test_report.md`, not an install problem.

---

## 5. Configuration

`demerzel init` writes `demerzel.config.json`. Every mechanism weight and
ablation switch lives there; point at it with `--config`:

```sh
demerzel --config configs/exp2.json run-experiment exp2
```

Key fields:

| field | default | meaning |
|-------|---------|---------|
| `storage.backend` | `sqlite` | or in-memory when `path` is `:memory:` |
| `storage.path` | `demerzel.db` | SQLite file location |
| `retrieval.w_semantic` | `1.0` | semantic channel weight |
| `retrieval.w_pointer` | `1.0` | **accessibility prior** — see §4 Exp-2 note |
| `retrieval.w_context_distance` | `0.4` | distance penalty (a prior, not relevance) |
| `decay.mode` | `context` | `none` \| `time` \| `boundary` \| `context` |
| `mpe.enabled` / `mpe.eta` | `true` / `0.05` | pointer learning rate |
| `ablations` | all `true` | one switch per mechanism: `decay`, `mpe`, `graph` |
| `context_distance.cold_start_k` | `3` | cold-start threshold |

A YAML config is accepted when PyYAML is installed; JSON always works.

---

## 6. Reproducing someone else's result

Every run records what produced it. To reproduce a published number:

1. **Match the commit.** Read `metadata.json` → `system_version`, and check the
   repo tag (`v0.1.0`). `git checkout v0.1.0`.
2. **Match the config.** `config.json` in the run directory is the exact config;
   pass it with `--config`.
3. **Match the environment.** `metadata.json` records `llm_model`,
   `embedding_model`, `temperature`, `seed`, `dataset_version`,
   `config_hash`, `timestamp`.
4. **Compare `config_hash`** — a 16-hex SHA-256 over the canonical config. Two
   runs that differ in any knob cannot share it.
5. Re-run and diff `metrics.json`.

The default stack is fully deterministic (hashing embedder, token-overlap
"LLM"), so identical configs give identical metrics — no seed hunting.

---

## 7. Offline install

If the machine cannot reach PyPI:

```sh
# Option A: no install at all (works because the package is stdlib-only)
export PYTHONPATH="/path/to/Demerzel:$PYTHONPATH"
python3 -m demerzel.cli query "..."

# Option B: install from the checkout without contacting an index
pip install -e . --no-build-isolation --no-deps
```

`--no-deps` skips the declared `numpy`; `--no-build-isolation` uses your local
`setuptools` instead of fetching a build environment.

To build a wheel for another machine:

```sh
pip wheel . --no-build-isolation --no-deps -w dist/
# copy dist/*.whl, then on the target:
pip install dist/demerzel-0.1.0-py3-none-any.whl
```

---

## 8. Docker (optional)

No image is shipped. A minimal one that runs the acceptance surface:

```dockerfile
FROM python:3.12-slim

WORKDIR /app
COPY . /app

RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
 && pip install --no-cache-dir -e .

# Persist the SQLite db and results outside the image.
VOLUME ["/app/results"]

ENTRYPOINT ["demerzel"]
CMD ["--help"]
```

```sh
docker build -t demerzel:0.1.0 .
docker run --rm -v "$PWD/results:/app/results" demerzel:0.1.0 \
    run-experiment exp2
```

---

## 9. Operational notes

- **Crash safety.** SQLite runs with WAL + `synchronous=FULL`, so a committed
  write survives a kill (P0-7). Keep `storage.path` on a local filesystem;
  network filesystems can weaken the fsync guarantee.
- **Concurrency.** The SQLite backend guards writes with a lock and
  `BEGIN IMMEDIATE`. Run one process per database file; use separate files to
  parallelize experiments.
- **Data growth.** `results/` and `*.db` are git-ignored by design. Archive or
  delete them freely — they are reproducible from the config.
- **Deletion is explicit.** Forgetting never deletes evidence; removing data is
  a separate, audited operation (`docs/invariants.md`, Invariant 6).
- **No secrets.** The default providers call nothing external, so there is
  nothing to configure and nothing to leak. Wiring a real LLM/embedding provider
  means implementing the two interfaces in `demerzel/interfaces/providers.py`.

---

## 10. Troubleshooting

| symptom | cause / fix |
|---------|-------------|
| `ModuleNotFoundError: No module named 'setuptools'` | run `pip install --upgrade setuptools wheel` first |
| install hangs downloading | restricted network → §7 offline install |
| `demerzel: command not found` | venv not activated, or use `python3 -m demerzel.cli` |
| `config not found: ...` | `--config` path is wrong; `demerzel init` writes one |
| query returns few candidates | cold start: fewer than `cold_start_k` turns → distance 0 by design (`docs/cold_start.md`) |
| Exp-2 `base_effect_present: false` | raise `retrieval.w_pointer`; see §4 |
| `pytest` not found | not required — `python3 tests/run_tests.py` runs the same suite |

---

## 11. Verified deployment matrix

The install and acceptance surface were exercised end to end from a clean
virtual environment on Python 3.12.3:

| step | result |
|------|--------|
| `pip install -e .` | `Successfully installed demerzel-0.1.0` |
| `demerzel --help` | all six subcommands listed |
| `demerzel init` / `ingest` / `query` | trace returned, top candidate correct |
| `demerzel run-experiment exp2` | `hypothesis_supported: true`, `G_fixed/E_fixed/A_mutable: true` |
| `python tests/run_tests.py` | `Ran 78 tests ... OK` |
