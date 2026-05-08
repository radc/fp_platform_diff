#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="/home/ruhan/fp_platform_diff/runs/exp1_norm_0_1_fp32/executions"

mapfile -t RUNS < <(find "$BASE_DIR" -mindepth 1 -maxdepth 1 -type d | sort)

for reference in "${RUNS[@]}"; do
  candidates=()

  for candidate in "${RUNS[@]}"; do
    if [[ "$candidate" != "$reference" ]]; then
      candidates+=("$candidate")
    fi
  done

  echo "============================================================"
  echo "Reference: $reference"
  echo "Candidates:"
  printf '  - %s\n' "${candidates[@]}"

  python main.py compare \
    --reference "$reference" \
    --candidate "${candidates[@]}" \
    --format pt \
    --rtol 0.0 \
    --atol 0.0
done