#!/usr/bin/env bash
# Ablation runner. Defaults to the smoke test (contracts 6-15, three configurations);
# set LIMIT=50 and CONFIGS to run the full ablation. Results merge into the same tree,
# so contracts already done are skipped by --resume.
# Env: LABELS (default "1 5"), CONFIGS (default all three), LIMIT (default 15; 50 = full ablation). Contracts 1-5 are warm-up (oracle, excluded); 6..LIMIT are evaluated.
# The `full` configuration is not rerun: its rows for contracts 6-15 come from results_sonnet_v2.
# Cheapest configurations run first, so a usage-limit stop (exit 75) still leaves useful results;
# rerun the script after the reset and finished contracts are skipped (--resume).
set -u
cd "$(dirname "$0")/.."
ROOT="results_ablation_smoke"; MEM="memory_ablation_smoke"; mkdir -p "$ROOT/log"
LIMIT="${LIMIT:-15}"
for L in ${LABELS:-1 5}; do
  for CFG in ${CONFIGS:-pruning_only single_shot feedback_only}; do
    echo "=== $(date '+%H:%M:%S') label $L config $CFG (contracts 6-$LIMIT) ==="
    python3 -u main.py --mode real --provider claude_cli --model claude-sonnet-5 \
      --contracts_root ./buggy_contracts \
      --results_root "$ROOT/$CFG" --memory_root "$MEM/$CFG" \
      --label_index "$L" --limit_contracts "$LIMIT" --warmup_contracts 5 --oracle off \
      --ablation "$CFG" --num_runs 1 \
      --threshold 0.7 --max_attempts 3 --early_stop block \
      --topk_candidates 40 --condense_window 5 \
      --block_eval dilated --block_dilation 1 --line_tolerance 0 \
      --smart_feedback llm --resume 2>&1 | tee -a "$ROOT/log/label_${L}_${CFG}.log"
    s=${PIPESTATUS[0]}
    if [ "$s" -eq 75 ]; then echo "Usage limit reached (label $L, $CFG). Rerun after reset."; exit 75; fi
    if [ "$s" -ne 0 ]; then echo "Failed (label $L, $CFG) exit $s"; exit "$s"; fi
  done
done
echo "Smoke ablation complete."
