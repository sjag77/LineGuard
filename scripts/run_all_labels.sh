#!/usr/bin/env bash
# Runs LineGuard with Claude Sonnet 5 through the Claude CLI, one vulnerability label at a time.
# Protocol per label: first 5 contracts in oracle mode (memory warm-up, excluded from metrics),
# remaining 45 contracts in non-oracle mode (reported).
# Safe to rerun: completed contracts are skipped (--resume). If the subscription usage limit
# is hit, the script stops with exit code 75; rerun it after the limit resets.
#
# Runs exactly ONE label per invocation, so each run fits inside one 5-hour usage window.
#
# Usage:  scripts/run_all_labels.sh <label_index>
#   1 Re-entrancy, 2 Timestamp-Dependency, 3 Unchecked-Send, 4 Unhandled-Exceptions,
#   5 TOD, 6 Overflow-Underflow, 7 tx.origin

set -u
cd "$(dirname "$0")/.."

RESULTS_ROOT="results_sonnet_v2"
MEMORY_ROOT="memory_sonnet_v2"
LOG_DIR="$RESULTS_ROOT/log"
mkdir -p "$LOG_DIR"

if [ $# -ne 1 ] || ! [[ "$1" =~ ^[1-7]$ ]]; then
  echo "Give exactly one label index (1-7). One label per 5-hour usage window."
  echo "  1 Re-entrancy, 2 Timestamp-Dependency, 3 Unchecked-Send, 4 Unhandled-Exceptions,"
  echo "  5 TOD, 6 Overflow-Underflow, 7 tx.origin"
  exit 2
fi
LABELS=("$1")

for L in "${LABELS[@]}"; do
  echo "=== $(date '+%Y-%m-%d %H:%M:%S') label $L ==="
  python3 -u main.py --mode real \
    --provider claude_cli --model claude-sonnet-5 \
    --contracts_root ./buggy_contracts \
    --results_root "$RESULTS_ROOT" --memory_root "$MEMORY_ROOT" \
    --label_index "$L" --limit_contracts 50 \
    --warmup_contracts 5 --oracle off --ablation full --num_runs 1 \
    --threshold 0.7 --max_attempts 3 --early_stop block \
    --topk_candidates 40 --condense_window 5 \
    --block_eval dilated --block_dilation 1 --line_tolerance 0 \
    --smart_feedback llm --resume 2>&1 | tee -a "$LOG_DIR/label_$L.log"
  status=${PIPESTATUS[0]}
  if [ "$status" -eq 75 ]; then
    echo "Usage limit reached during label $L. Rerun this script after the limit resets; finished contracts will be skipped."
    exit 75
  elif [ "$status" -ne 0 ]; then
    echo "Label $L failed with exit code $status. See $LOG_DIR/label_$L.log"
    exit "$status"
  fi
done
echo "All requested labels completed."
