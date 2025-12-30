# BOAD: Discovering Hierarchical Software Engineering Agents via Bandit Optimization

This repository contains the official implementation for [**BOAD: Discovering Hierarchical Software Engineering Agents via Bandit Optimization**](https://arxiv.org/abs/2512.23631), a framework for automatically discovering hierarchical multi-agent systems for software engineering tasks via bandit optimization.

## 🚀 Key Features

* ✅ **Sub-agent implementation in SWE-agent**: Sub-agents are implemented as tools in SWE-agent's framework, invoked via XML function calling with arguments (no shared execution history)
* ✅ **Automated sub-agent discovery**: Uses LLMs to generate and refine tool configurations
* ✅ **Bandit-based optimization**: UCB algorithm balances exploration and exploitation
* ✅ **Dynamic archive expansion**: Chinese Restaurant Process adds new sub-agents over time
* ✅ **Hindsight credit assignment**: LLM-as-a-judge scores sub-agent helpfulness from trajectories
* ✅ **Warmup refinement**: Automatically refines sub-agent docs for orchestrator compatibility
* ✅ **Comprehensive evaluation**: Supports SWE-bench Live and Verified with analysis

## 📦 Installation

### 1. Install Base Package

```bash
pip install -e .
```

### 2. Install SWE-bench Harness

The SWE-bench harnesses are included as git submodules. If you didn't clone with `--recursive`, initialize them first:

```bash
git submodule update --init --recursive
```

Then choose a harness to install:

**For SWE-bench Live:**
```bash
cd SWE-bench-Live
pip install -e .
cd ..
```

**For SWE-bench Verified:**
```bash
cd SWE-Bench-Verified
pip install -e .
cd ..
```

### 3. Set Up API Keys

```bash
export OPENAI_API_KEY="your-key"          # For OpenAI-compatible models
export ANTHROPIC_API_KEY="your-key"       # For Claude models
```

## 🎯 Quick Start

### 1. Discover Sub-Agents

Run BOAD optimization to discover effective sub-agents:

```bash
python tool_gen/run_boad.py --config claude_seed_oss_36b
```

This runs the bandit optimization loop, generating and evaluating sub-agents. Results are saved in `tool_gen/generated/<config_name>/`.

### 2. Set Up Evaluation Run

Create an evaluation configuration using discovered sub-agents:

```bash
python tool_gen/evaluation/setup_scripts/make_eval_dir_from_generated.py \
    --folder claude_seed_oss_36b \
    --output_folder my_experiment \
    --k 2 \
    --template_dir tool_gen/templates_seed_oss_36b
```

This selects the top-K sub-agents from your optimization run and sets up the evaluation directory.

### 3. Run Inference and Evaluation

Generate patches and evaluate them on SWE-bench:

```bash
# For SWE-bench Live
bash tool_gen/evaluation/run_live_infer_and_eval.sh my_experiment

# For SWE-bench Verified
bash tool_gen/evaluation/run_verified_infer_and_eval.sh my_experiment
```

These scripts automatically:
- Remove exit cost error instances
- Run inference to generate patches
- Evaluate patches against the SWE-bench harness

### 4. Analyze Results

```bash
python tool_gen/evaluation/analyze_eval_results.py --folder my_experiment
```

This prints a summary to console and saves detailed results to:
`logs/run_evaluation/my_experiment/analysis_summary.json`

## 📁 Project Structure

```
tool_gen/
├── run_boad.py                    # Entry point for optimization
├── tool_evolution_engine.py       # Bandit optimization engine
├── subagent_generator.py          # Sub-agent generation
├── main_agent_generator.py        # Orchestrator generation
├── tools.py                       # Tool class with UCB scoring
├── warmup.py                      # Sub-agent warmup/refinement
│
├── config/                        # Configuration files
├── prompts/                       # LLM prompts
├── templates/                     # Agent templates
│
├── evaluation/                    # Evaluation pipeline
│   ├── eval_runs/                 # Evaluation config and patches
│   ├── setup_scripts/             # Setup utilities
│   └── *.sh                       # Inference/eval scripts
│
└── generated/                     # Output directory
    └── <config_name>/
        ├── subagent_archive_snapshot.json
        └── experiments/
```

## 🔧 Main Components

### `tool_gen/`
Contains the core BOAD optimization system: bandit engine, sub-agent generation, warmup, orchestrator instantiation, and archive management. Entry point is `run_boad.py`.

### `tool_gen/evaluation/`
Pipeline for running inference and evaluation on SWE-bench: setup scripts to create eval configs, bash scripts to run inference/evaluation, and analysis tools to aggregate results.

### Sub-Agent Implementation in SWE-agent
Sub-agents are implemented as tools in SWE-agent's tool system. Each sub-agent has:
- **Config** (`config.yaml`): Defines signature, arguments, and description
- **Templates** (`templates.yaml`): System and instance prompts for the sub-agent LLM
- **Tool bundle**: Registered in the orchestrator's tool list and invoked via XML function calling

The orchestrator invokes sub-agents with arguments (e.g., issue description, file paths) and receives outputs, without sharing execution history.

## 📝 Configuration

Configuration files in `tool_gen/config/` specify:
- Model API endpoints
- Evolution parameters (iterations, sub-agent count, CRP theta)
- Design set configuration
- Output directories

See `tool_gen/config/claude_seed_oss_36b.yaml` for an example. For detailed configuration options, see `CONFIG.md`.


## 📚 Citation

If you use BOAD in your research, please cite our paper:

```bibtex
@misc{xu2025boad,
      title={BOAD: Discovering Hierarchical Software Engineering Agents via Bandit Optimization}, 
      author={Iris Xu and Guangtao Zeng and Zexue He and Charles Jin and Aldo Pareja and Dan Gutfreund and Chuang Gan and Zhang-Wei Hong},
      year={2025},
      eprint={2512.23631},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2512.23631}, 
}
```

## 🙏 Acknowledgments

BOAD is built on top of [SWE-agent](https://swe-agent.com/latest/) and evaluated using [SWE-bench](https://www.swebench.com/) and [SWE-Bench-Live](https://swe-bench-live.github.io/). We thank the original SWE-agent, SWE-bench, and SWE-Bench-Live teams for developing these high-quality tools and benchmarks, and for making them publicly available to the research community.
