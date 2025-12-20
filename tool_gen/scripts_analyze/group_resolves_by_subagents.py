from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional


def parse_experiment_number(name: str) -> int:
    m = re.search(r"exp_(\d+)", name)
    return int(m.group(1)) if m else -1


def load_json(path: Path) -> Optional[dict]:
    try:
        if path.exists():
            return json.loads(path.read_text())
    except Exception:
        return None
    return None


def get_subagent_combo_key(experiment_json: dict) -> Optional[str]:
    chosen = experiment_json.get("chosen_tools")
    if not isinstance(chosen, list):
        return None
    names: List[str] = []
    for item in chosen:
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        if not isinstance(name, str) or not name:
            continue
        # If field exists, prefer only subagents; else include all
        is_sub = item.get("subagent")
        if is_sub is False:
            continue
        names.append(name)
    if not names:
        return None
    names.sort()
    return " + ".join(names)


def get_resolved_count(result_json: dict) -> Optional[int]:
    resolved = result_json.get("resolved")
    if isinstance(resolved, int):
        return resolved
    p2p_success = result_json.get("p2p_success")
    f2p_success = result_json.get("f2p_success")
    if isinstance(p2p_success, int) and isinstance(f2p_success, int):
        return p2p_success + f2p_success
    return None


def group_resolves_by_combo(experiments_dir: Path) -> Dict[str, List[int]]:
    if not experiments_dir.exists():
        return {}
    exp_dirs = [p for p in experiments_dir.iterdir() if p.is_dir() and p.name.startswith("exp_")]
    exp_dirs.sort(key=lambda p: parse_experiment_number(p.name))

    grouped: Dict[str, List[int]] = {}
    for exp_dir in exp_dirs:
        exp_json = load_json(exp_dir / "experiment.json")
        res_json = load_json(exp_dir / "experiment_result.json")
        if not exp_json or not res_json:
            continue
        key = get_subagent_combo_key(exp_json)
        resolved = get_resolved_count(res_json)
        if key is None or resolved is None:
            continue
        grouped.setdefault(key, []).append(resolved)
    return grouped


def main() -> None:
    parser = argparse.ArgumentParser(description="Group resolved counts by subagent combinations across experiments.")
    parser.add_argument(
        "--folder",
        type=str,
        default=Path("claude_seed_oss_36b")
    )
    args = parser.parse_args()
    
    experiments_dir = Path("/root/iris/SWE-agent/tool_gen/generated") / args.folder / "experiments"
    grouped = group_resolves_by_combo(experiments_dir)

    # Compute first seen experiment number for each combo
    first_seen: Dict[str, int] = {}
    exp_dirs = [p for p in experiments_dir.iterdir() if p.is_dir() and p.name.startswith("exp_")]
    exp_dirs.sort(key=lambda p: parse_experiment_number(p.name))
    for exp_dir in exp_dirs:
        exp_json = load_json(exp_dir / "experiment.json")
        if not exp_json:
            continue
        key = get_subagent_combo_key(exp_json)
        if key is None:
            continue
        exp_num = parse_experiment_number(exp_dir.name)
        if key not in first_seen:
            first_seen[key] = exp_num

    for key in sorted(grouped.keys(), key=lambda k: (first_seen.get(k, 10**9), k)):
        print(f"{key}: {grouped[key]}")


if __name__ == "__main__":
    main()


