#!/bin/bash
# Usage: tool_gen/evaluation/run_live_infer.sh TEST_DIR [NUM_WORKERS]
TEST_DIR="$1"
NUM_WORKERS="${2:-8}"
INSTANCES_PATH="swebench_data/swebench_live.jsonl"

sweagent run-batch \
    --config "$TEST_DIR/agent.yaml" \
    --config "$TEST_DIR/subagent.yaml" \
    --instances.type file \
    --instances.path "$INSTANCES_PATH" \
    --output_dir "$TEST_DIR/patches" \
    --num_workers "$NUM_WORKERS"