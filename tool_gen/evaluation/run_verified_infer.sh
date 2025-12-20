#!/bin/bash
# Usage: tool_gen/evaluation/run_verified_infer.sh TEST_DIR [NUM_WORKERS]
TEST_DIR="$1"
NUM_WORKERS="${2:-8}"

sweagent run-batch \
    --config "$TEST_DIR/agent.yaml" \
    --config "$TEST_DIR/subagent.yaml" \
    --instances.type swe_bench \
    --instances.subset verified \
    --instances.split test \
    --output_dir "$TEST_DIR/patches" \
    --num_workers "$NUM_WORKERS"