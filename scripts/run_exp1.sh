#!/usr/bin/env bash
# Experiment 1 reproduction (V1.md §11.3 / §8.1).
#
#   ./scripts/run_exp1.sh [results_root]
#
# Writes results to <results_root>/exp1/ and prints the JSON summary.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS_ROOT="${1:-${ROOT}/results}"

cd "${ROOT}"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

python3 -m demerzel.cli --config "${ROOT}/configs/exp1.json" run-experiment exp1 --results-root "${RESULTS_ROOT}"
