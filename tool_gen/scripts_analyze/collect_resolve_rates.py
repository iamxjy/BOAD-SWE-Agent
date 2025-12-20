from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import List, Optional


def parse_experiment_number(name: str) -> int:
    m = re.search(r"exp_(\d+)", name)
    return int(m.group(1)) if m else -1


def compute_resolve_rate(payload: dict) -> Optional[float]:
    resolved = payload.get("resolved")
    return resolved

def collect_rates(experiments_dir: Path) -> List[float]:
    if not experiments_dir.exists():
        return []

    experiment_dirs = [p for p in experiments_dir.iterdir() if p.is_dir() and p.name.startswith("exp_")]
    experiment_dirs.sort(key=lambda p: parse_experiment_number(p.name))

    rates: List[float] = []
    for exp_dir in experiment_dirs:
        result_path = exp_dir / "experiment_result.json"
        if not result_path.exists():
            continue
        try:
            data = json.loads(result_path.read_text())
        except Exception:
            continue
        rate = compute_resolve_rate(data)
        if rate is not None:
            rates.append(rate)
    return rates


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect resolve rates from experiment results.")
    parser.add_argument(
        "--folder",
        type=str,
        default="v3",
        help="Folder name in tool_gen/generated/ (e.g., 'v3', 'qwen3_coder_30b_heavy_agent')",
    )
    args = parser.parse_args()

    # Derive experiments directory from folder name
    experiments_dir = Path("/root/iris/SWE-agent/tool_gen/generated") / args.folder / "experiments"
    
    rates = collect_rates(experiments_dir)
    print(json.dumps(rates))


if __name__ == "__main__":
    main()


