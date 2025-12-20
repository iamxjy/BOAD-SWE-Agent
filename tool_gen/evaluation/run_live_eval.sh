#!/bin/bash
# Usage: tool_gen/evaluation/run_live_eval.sh RUN_ID [NUM_WORKERS]
RUN_ID="$1"
NUM_WORKERS="${2:-8}"

python -m SWE-bench-Live.swebench.harness.run_evaluation \
    --dataset_name SWE-bench-Live/SWE-bench-Live \
    --split lite \
    --namespace starryzhang \
    --predictions_path "tool_gen/evaluation/eval_runs/$RUN_ID/patches/preds_list.json" \
    --max_workers "$NUM_WORKERS" \
    --run_id "$RUN_ID"