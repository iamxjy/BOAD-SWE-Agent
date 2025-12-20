#!/bin/bash
# Usage: tool_gen/evaluation/run_verified_infer_and_eval.sh EXP_ID [NUM_WORKERS]
EXP_ID="$1"
NUM_WORKERS="${2:-8}"
TEST_DIR="tool_gen/evaluation/eval_runs/$EXP_ID"

python tool_gen/evaluation/remove_exit_cost_error_instances.py \
    --test-dir "$TEST_DIR" \
    --eval-dir "logs/run_evaluation/$EXP_ID/patches"

bash tool_gen/evaluation/run_verified_infer.sh "$TEST_DIR" "$NUM_WORKERS" && \
    bash tool_gen/evaluation/run_verified_eval.sh "$TEST_DIR/patches/preds_list.json" "$EXP_ID" "$NUM_WORKERS"