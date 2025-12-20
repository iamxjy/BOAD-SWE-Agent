#!/bin/bash
# Usage: tool_gen/evaluation/run_custom_infer.sh TEST_DIR INSTANCES_PATH [NUM_WORKERS]
TEST_DIR="$1"
INSTANCES_PATH="$2"
NUM_WORKERS="${3:-8}"

sweagent run-batch \
    --config "$TEST_DIR/agent.yaml" \
    --config "$TEST_DIR/subagent.yaml" \
    --instances.type file \
    --instances.path "$INSTANCES_PATH" \
    --output_dir "$TEST_DIR/patches" \
    --num_workers "$NUM_WORKERS"