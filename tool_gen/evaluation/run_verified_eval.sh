#!/bin/bash
# Usage: tool_gen/evaluation/run_verified_eval.sh PREDICTIONS_PATH RUN_ID [NUM_WORKERS]
PREDICTIONS_PATH="$1"
RUN_ID="$2"
NUM_WORKERS="${3:-8}"

python -m swebench.harness.run_evaluation \
  --predictions_path "$PREDICTIONS_PATH" \
  --dataset "princeton-nlp/SWE-bench_Verified" \
  --split "test" \
  --run_id "$RUN_ID" \
  --max_workers "$NUM_WORKERS"
