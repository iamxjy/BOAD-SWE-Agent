#!/bin/bash
# Usage: tool_gen/evaluation/run_custom_eval.sh PREDICTIONS_PATH RUN_ID DATASET [NUM_WORKERS]
PREDICTIONS_PATH="$1"
RUN_ID="$2"
DATASET="$3"
NUM_WORKERS="${4:-8}"

python -m swebench.harness.run_evaluation \
  --predictions_path "$PREDICTIONS_PATH" \
  --dataset "$DATASET" \
  --split "test" \
  --run_id "$RUN_ID" \
  --max_workers "$NUM_WORKERS"

