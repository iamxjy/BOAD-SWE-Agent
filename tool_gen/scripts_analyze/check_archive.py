#!/usr/bin/env python3
"""Load subagent_archive_snapshot.json from each experiment and calculate UCB values."""

import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Add the tool_gen directory to the path to import Tool class
sys.path.append(str(Path(__file__).parent.parent))
from tools import Tool


def load_archive_data(archive_path: Path) -> List[Dict]:
    """Load subagent data from archive.json."""
    if not archive_path.exists():
        return []
    try:
        data = json.loads(archive_path.read_text())
        # Handle new format with step and tools fields
        if isinstance(data, dict) and "tools" in data:
            return data["tools"]
        # Handle old format (list of tools)
        elif isinstance(data, list):
            return data
        return []
    except Exception:
        return []


def load_archive_data_with_step(archive_path: Path) -> Tuple[List[Dict], int]:
    """Load subagent data and step from archive.json."""
    if not archive_path.exists():
        return [], 0
    try:
        data = json.loads(archive_path.read_text())
        # Handle new format with step and tools fields
        if isinstance(data, dict) and "tools" in data:
            return data["tools"], data.get("step", 0)
        # Handle old format (list of tools)
        elif isinstance(data, list):
            return data, 0
        return [], 0
    except Exception:
        return [], 0


def load_experiment_chosen_tools(exp_dir: Path) -> set[str]:
    """Load the chosen tools from experiment.json."""
    exp_json_path = exp_dir / "experiment.json"
    if not exp_json_path.exists():
        return set()
    
    try:
        data = json.loads(exp_json_path.read_text())
        if not isinstance(data, dict):
            return set()
        
        chosen_tools = data.get("chosen_tools", [])
        if not isinstance(chosen_tools, list):
            return set()
        
        chosen_names = set()
        for tool in chosen_tools:
            if isinstance(tool, dict):
                name = tool.get("name")
                if isinstance(name, str):
                    chosen_names.add(name)
        
        return chosen_names
    except Exception:
        return set()


def process_experiment_archive(exp_dir: Path) -> None:
    """Process a single experiment's subagent_archive_snapshot.json."""
    archive_path = exp_dir / "subagent_archive_snapshot.json"
    
    # Load archive data and step
    archive_data, step = load_archive_data_with_step(archive_path)
    if not archive_data:
        print(f"No archive data found in {exp_dir.name}")
        return
    
    # Load chosen tools for this experiment
    chosen_tools = load_experiment_chosen_tools(exp_dir)
    
    # Calculate UCB for each subagent using Tool class
    ucb_results = []
    for item in archive_data:
        if not isinstance(item, dict):
            continue
        
        name = item.get("name")
        n = item.get("n", 0)
        successes = item.get("successes", 0)
        helpful_count = item.get("helpful_count", 0)
        exp_num = item.get("exp_num", 0)
        
        if not isinstance(name, str) or not isinstance(n, int) or not isinstance(successes, int):
            continue
        
        # Create a Tool object to use its ucb_score method
        tool = Tool(
            name=name,
            signature=item.get("signature", ""),
            docstring=item.get("docstring", ""),
            arguments=item.get("arguments", []),
            bundle_dir=Path(item.get("bundle_dir", "")),
            subagent=item.get("subagent", True),
            n=n,
            successes=successes,
            helpful_count=helpful_count
        )
        
        ucb_value = tool.ucb_score(step)
        success_rate = tool.mean_reward()
        was_chosen = name in chosen_tools
        
        ucb_results.append({
            "name": name,
            "n": n,
            "helpful_count": helpful_count,
            "successes": successes,
            "success_rate": success_rate,
            "ucb": ucb_value,
            "chosen": was_chosen,
            "exp_num": exp_num
        })
    
    # Sort by UCB value (descending)
    ucb_results.sort(key=lambda x: x["ucb"], reverse=True)
    
    # Print results
    print(f"\n{'='*80}")
    print(f"EXPERIMENT: {exp_dir.name}")
    print(f"{'='*80}")
    print(f"Total pulls across all subagents: {sum(r['n'] for r in ucb_results)}")
    print()
    print(f"{'Subagent':<40} {'n':>6} {'helpful_count':>10} {'help_rate':>9} {'successes':>10} {'succ_rate':>10} {'UCB':>8} {'Chosen':>7} {'exp_num':>6}")
    print("-" * 67)
    
    for result in ucb_results:
        name = result["name"]
        n = result["n"]
        helpful_count = result["helpful_count"]
        helpful_rate = helpful_count / n if n > 0 else 0.0
        successes = result["successes"]
        rate = result["successes"] / result["n"] if result["n"] > 0 else 0.0
        ucb = result["ucb"]
        chosen = result["chosen"]
        exp_num = result["exp_num"]
        
        ucb_str = f"{ucb:.3f}"
        chosen_str = "  *" if chosen else "   "
        
        print(f"{name:<40} {n:>6} {helpful_count:>10} {helpful_rate:>9.3f} {successes:>10} {rate:>10.3f} {ucb_str:>8} {chosen_str:>7} {exp_num:>6}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Analyze subagent archives in experiment folders.")
    parser.add_argument(
        "--folder",
        type=str,
        default="qwen3_coder_30b_heavy_agent"
    )
    args, unknown = parser.parse_known_args()
    experiments_dir = Path(f"./tool_gen/generated/{args.folder}/experiments")
    if not experiments_dir.exists():
        print(f"Experiments directory not found: {experiments_dir}")
        return
    
    # Find all experiment directories
    exp_dirs = [d for d in experiments_dir.iterdir() if d.is_dir() and d.name.startswith("exp_")]
    exp_dirs.sort(key=lambda d: int(d.name.split("_")[1]) if d.name.split("_")[1].isdigit() else 0)
    
    print(f"Found {len(exp_dirs)} experiment directories")
    print("Processing subagent_archive_snapshot.json files...")
    
    # Process each experiment
    for exp_dir in exp_dirs:
        process_experiment_archive(exp_dir)


if __name__ == "__main__":
    main()