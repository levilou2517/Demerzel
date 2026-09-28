#!/usr/bin/env bash
# Experiment 2 reproduction — the key experiment (V1.md §8.2, §21).
#
#   ./scripts/run_exp2.sh [results_root]
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS_ROOT="${1:-${ROOT}/results}"

cd "${ROOT}"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

python3 -m demerzel.cli --config "${ROOT}/configs/exp2.json" run-experiment exp2 --results-root "${RESULTS_ROOT}"
