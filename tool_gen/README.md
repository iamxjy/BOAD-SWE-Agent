# tool_gen: Tool Evolution & Experimentation for SWE-agent

This module automates the generation, configuration, and evaluation of tool bundles (subagents) for the SWE-agent project.

## Features
- Iterative tool (subagent) generation using LLMs
- Automated experiment setup and batch execution
- Evaluation of experiment results with SWE-smith

## Directory Structure
- `tool_evolution_engine.py`: Main engine for running tool evolution cycles
- `subagent_generator.py`: Generates subagent configs and templates using LLM prompts
- `experiment_result.py`: Handles experiment result data
- `generated/`: Output directory for generated configs, logs, and results
- `prompts/`, `templates/`: Prompt and template files for LLM-based generation

## Setup
1. Set up OpenAI or compatible LLM API (Can use existing V3 on localhost:1004 on mitibm-swe01)
2. Use swe-agent environment on mitibm-swe01


## Usage
### 1. Generate Subagents & Experiments
Edit or use the provided prompts/templates, then run `python tool_gen/run_boad.py`
Edit this script to change the configuration and instances.
This will:
- Generate tool bundles and subagent configs
- Run experiments using `sweagent` in batch mode
- Evaluate results using swe smith and log outputs

### 2. Inspect Results

- All outputs are saved in `tool_gen/generated/`.
- Experiment configs are directly under `iteration_***/experiment_***`
- LLM calls for generating the experiments are logged in `llm_calls.log`
- Trajectory and evaluation logs under `logs`
