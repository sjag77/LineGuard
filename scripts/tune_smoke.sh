#!/usr/bin/env bash
# Tuning smoke test for the two weak categories (Re-entrancy, TOD): full pipeline with the
# widened candidate budget (--topk_candidates "${TOPK:-100}") and the extended TOD extraction patterns.
# Writes to results_tune/ so the reported results in results_sonnet_v2/ are left untouched.
set -u
cd "$(dirname "$0")/.."
mkdir -p "${ROOT:-results_tune}/log"
for L in ${LABELS:-1 5}; do
  echo "=== $(date '+%H:%M:%S') tuning label $L ==="
  python3 -u main.py --mode real --provider claude_cli --model claude-sonnet-5 \
    --contracts_root ./buggy_contracts \
    --results_root "${ROOT:-results_tune}" --memory_root "${MEM:-memory_tune}" \
    --label_index "$L" --limit_contracts "${LIMIT:-15}" --warmup_contracts 5 --oracle off \
    --ablation full --num_runs 1 \
    --threshold 0.7 --max_attempts 3 --early_stop block \
    --topk_candidates "${TOPK:-100}" --condense_window 5 \
    --block_eval dilated --block_dilation 1 --line_tolerance 0 \
    --smart_feedback llm --resume 2>&1 | tee -a "${ROOT:-results_tune}/log/label_$L.log"
  s=${PIPESTATUS[0]}
  [ "$s" -eq 75 ] && { echo "Usage limit reached (label $L). Rerun after reset."; exit 75; }
  [ "$s" -ne 0 ] && { echo "Failed label $L exit $s"; exit "$s"; }
done
echo "Tuning smoke complete."
