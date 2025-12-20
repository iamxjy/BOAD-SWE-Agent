from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, Optional, Set, Tuple


def load_json(path: Path) -> Optional[dict]:
    try:
        if path.exists():
            return json.loads(path.read_text())
    except Exception:
        return None
    return None


def collect_all_subagent_names(experiments_dir: Path) -> Set[str]:
    names: Set[str] = set()
    if not experiments_dir.exists():
        return names
    for exp_dir in experiments_dir.iterdir():
        if not exp_dir.is_dir() or not exp_dir.name.startswith("exp_"):
            continue
        exp_json = load_json(exp_dir / "experiment.json")
        if not exp_json:
            continue
        chosen = exp_json.get("chosen_tools")
        if not isinstance(chosen, list):
            continue
        for item in chosen:
            if not isinstance(item, dict):
                continue
            name = item.get("name")
            if not isinstance(name, str) or not name:
                continue
            is_sub = item.get("subagent")
            if is_sub is False:
                continue
            names.add(name)
    return names


def collect_subagent_experiment_counts(experiments_dir: Path) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    if not experiments_dir.exists():
        return counts
    for exp_dir in experiments_dir.iterdir():
        if not exp_dir.is_dir() or not exp_dir.name.startswith("exp_"):
            continue
        exp_json = load_json(exp_dir / "experiment.json")
        if not exp_json:
            continue
        chosen = exp_json.get("chosen_tools")
        if not isinstance(chosen, list):
            continue
        names_in_exp: Set[str] = set()
        for item in chosen:
            if not isinstance(item, dict):
                continue
            name = item.get("name")
            if not isinstance(name, str) or not name:
                continue
            is_sub = item.get("subagent")
            if is_sub is False:
                continue
            names_in_exp.add(name)
        for name in names_in_exp:
            counts[name] = counts.get(name, 0) + 1
    return counts


def iter_tool_count_files(logs_dir: Path) -> Iterable[Path]:
    if not logs_dir.exists():
        return []
    return logs_dir.rglob("tool_call_counts.json")


def aggregate_call_counts(logs_dir: Path, subagent_names: Set[str]) -> Dict[str, int]:
    totals: Dict[str, int] = defaultdict(int)
    for path in iter_tool_count_files(logs_dir):
        data = load_json(path)
        if not isinstance(data, dict):
            continue
        for tool_name, count in data.items():
            if tool_name in subagent_names and isinstance(count, int):
                totals[tool_name] += count
    return dict(totals)


def build_inclusion_index(experiments_dir: Path) -> Dict[str, Set[Tuple[int, str]]]:
    """Map subagent name -> set of (exp_num, instance_id) where it is included.

    Uses `experiment.json` and its `instance_ids` list. Assumes each experiment
    contains 10 instances and the subagent is included for all listed instances
    if present in `chosen_tools`.
    """
    inclusion: Dict[str, Set[Tuple[int, str]]] = defaultdict(set)
    if not experiments_dir.exists():
        return {}
    for exp_dir in experiments_dir.iterdir():
        if not exp_dir.is_dir() or not exp_dir.name.startswith("exp_"):
            continue
        exp_num = _parse_exp_num(exp_dir.name)
        if exp_num is None:
            continue
        exp_json = load_json(exp_dir / "experiment.json")
        if not isinstance(exp_json, dict):
            continue
        chosen = exp_json.get("chosen_tools")
        instance_ids = exp_json.get("instance_ids")
        if not isinstance(chosen, list) or not isinstance(instance_ids, list):
            continue
        names_in_exp: Set[str] = set()
        for item in chosen:
            if not isinstance(item, dict):
                continue
            name = item.get("name")
            if not isinstance(name, str) or not name:
                continue
            is_sub = item.get("subagent")
            if is_sub is False:
                continue
            names_in_exp.add(name)
        for name in names_in_exp:
            for instance_id in instance_ids:
                if isinstance(instance_id, str):
                    inclusion[name].add((exp_num, instance_id))
    return inclusion


def _parse_exp_num(name: str) -> Optional[int]:
    parts = name.split("_")
    if len(parts) >= 2 and parts[0] == "exp" and parts[1].isdigit():
        try:
            return int(parts[1])
        except Exception:
            return None
    return None


def build_usage_index(logs_dir: Path, subagent_names: Set[str]) -> Dict[Tuple[int, str], Set[str]]:
    usage: Dict[Tuple[int, str], Set[str]] = {}
    for path in iter_tool_count_files(logs_dir):
        instance_id = path.parent.name
        exp_dir_name = path.parent.parent.name
        exp_num = _parse_exp_num(exp_dir_name)
        if exp_num is None:
            continue
        data = load_json(path)
        if not isinstance(data, dict):
            continue
        used: Set[str] = set()
        for tool_name, count in data.items():
            if tool_name in subagent_names and isinstance(count, int) and count > 0:
                used.add(tool_name)
        if used:
            usage[(exp_num, instance_id)] = used
    return usage


def load_run_eval_resolved(run_eval_root: Path) -> Dict[Tuple[int, str], bool]:
    resolved_map: Dict[Tuple[int, str], bool] = {}
    if not run_eval_root.exists():
        return resolved_map
    for report in run_eval_root.rglob("report.json"):
        instance_id = report.parent.name
        exp_dir_name = report.parent.parent.name
        exp_num = _parse_exp_num(exp_dir_name)
        if exp_num is None:
            continue
        data = load_json(report)
        if not isinstance(data, dict):
            continue
        if instance_id in data and isinstance(data[instance_id], dict):
            val = data[instance_id].get("resolved")
            if isinstance(val, bool):
                resolved_map[(exp_num, instance_id)] = val
                continue
        # Fallback: pick first entry
        for _, obj in data.items():
            if isinstance(obj, dict) and isinstance(obj.get("resolved"), bool):
                resolved_map[(exp_num, instance_id)] = bool(obj["resolved"]) 
                break
    return resolved_map


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate subagent tool call counts from trajectory logs.")
    parser.add_argument(
        "--folder",
        type=str,
        default="qwen3_coder_30b_heavy_agent"
    )
    parser.add_argument(
        "--sort-by",
        choices=["rate", "filt_rate", "calls", "avg", "insts", "name"],
        default="rate",
        help="Column to sort by (default: rate)",
    )
    parser.add_argument(
        "--ascending",
        action="store_true",
        help="Sort in ascending order (default: descending for numeric columns)",
    )
    args = parser.parse_args()

    # Derive paths from folder name
    base_dir = Path("/root/iris/SWE-agent/tool_gen/generated") / args.folder
    logs_dir = base_dir / "logs" / "trajectories"
    experiments_dir = base_dir / "experiments"
    run_eval_dir = base_dir / "logs" / "run_evaluation"

    subagent_names = collect_all_subagent_names(experiments_dir)
    totals = aggregate_call_counts(logs_dir, subagent_names)
    inclusion_index = build_inclusion_index(experiments_dir)
    usage_index = build_usage_index(logs_dir, subagent_names)
    resolved_index = load_run_eval_resolved(run_eval_dir)

    # Prepare rows for aligned output
    rows = []
    # Iterate over all known subagents (union of inclusion and usage and totals)
    all_names: Set[str] = set(subagent_names) | set(inclusion_index.keys()) | set(totals.keys())
    for name in sorted(all_names):
        included_keys = inclusion_index.get(name, set())
        exp_included = len(included_keys)
        calls_total = int(totals.get(name, 0))
        avg_calls = (calls_total / exp_included) if exp_included else 0.0
        avg_str = f"{avg_calls:.2f}"
        # Instances where subagent was actually called (count > 0)
        used_keys = [k for k, used in usage_index.items() if name in used]
        used_inst_count = len(used_keys)
        used_inst_str = str(used_inst_count)
        # Resolved counts
        resolved_included = 0
        for key in included_keys:
            if resolved_index.get(key) is True:
                resolved_included += 1
        resolved_used = 0
        for key in used_keys:
            if resolved_index.get(key) is True:
                resolved_used += 1
        resolved_str = str(resolved_included)
        # Rates
        rate_num = (resolved_included / exp_included) if exp_included else None
        rate_str = "n/a" if rate_num is None else f"{rate_num:.3f}"
        filt_rate_num = (resolved_used / used_inst_count) if used_inst_count else None
        filt_rate_str = "n/a" if filt_rate_num is None else f"{filt_rate_num:.3f}"
        filt_res_str = "n/a" if used_inst_count == 0 else str(resolved_used)
        rows.append(
            (
                name,                        # 0 name (str)
                str(calls_total),            # 1 calls (str)
                str(exp_included),           # 2 insts total (str)
                used_inst_str,               # 3 used insts (str)
                resolved_str,                # 4 resolved (str)
                avg_str,                     # 5 avg (str)
                rate_str,                    # 6 rate (str)
                filt_res_str,                # 7 filtered resolved (str)
                filt_rate_str,               # 8 filtered rate (str)
                calls_total,                 # 9 calls (num)
                int(exp_included),           # 10 insts total (num)
                int(used_inst_count),        # 11 used insts (num)
                int(resolved_included),      # 12 resolved (num)
                float(avg_calls),            # 13 avg (num)
                float(rate_num) if rate_num is not None else -1.0,      # 14 rate (num)
                float(filt_rate_num) if filt_rate_num is not None else -1.0,  # 15 filtered rate (num)
            )
        )

    # Sort by resolve rate descending; place 'n/a' (as -1.0) at the end
    def _sort_value(r):
        if args.sort_by == "rate":
            return r[14]
        if args.sort_by == "filt_rate":
            return r[15]
        if args.sort_by == "calls":
            return r[9]
        if args.sort_by == "avg":
            return r[13]
        if args.sort_by == "insts":
            return r[11]
        if args.sort_by == "name":
            return r[0]
        return r[14]

    if args.ascending:
        reverse = False
    else:
        reverse = False if args.sort_by in ("name", "insts") else True
    rows.sort(key=_sort_value, reverse=reverse)

    # Compute column widths
    name_w = max([len("Tool")] + [len(r[0]) for r in rows])
    calls_w = max([len("Calls")] + [len(r[1]) for r in rows])
    exps_w = max([len("Insts")] + [len(r[2]) for r in rows])
    used_w = max([len("Used")] + [len(r[3]) for r in rows])
    resolved_w = max([len("Resolved")] + [len(r[4]) for r in rows])
    avg_w = max([len("Avg")] + [len(r[5]) for r in rows])
    rate_w = max([len("Rate")] + [len(r[6]) for r in rows])
    filt_res_w = max([len("FiltRes")] + [len(r[7]) for r in rows])
    filt_w = max([len("FiltRate")] + [len(r[8]) for r in rows])

    # Header
    header = f"{('Tool'):<{name_w}}  {('Calls'):>{calls_w}}  {('Insts'):>{exps_w}}  {('Used'):>{used_w}}  {('Resolved'):>{resolved_w}}  {('Avg'):>{avg_w}}  {('Rate'):>{rate_w}}  {('FiltRes'):>{filt_res_w}}  {('FiltRate'):>{filt_w}}"
    sep = "-" * len(header)
    print(header)
    print(sep)
    for name, calls_s, exps_s, used_s, resolved_s, avg_s, rate_s, filt_res_s, filt_s, *_ in rows:
        print(f"{name:<{name_w}}  {calls_s:>{calls_w}}  {exps_s:>{exps_w}}  {used_s:>{used_w}}  {resolved_s:>{resolved_w}}  {avg_s:>{avg_w}}  {rate_s:>{rate_w}}  {filt_res_s:>{filt_res_w}}  {filt_s:>{filt_w}}")



if __name__ == "__main__":
    main()


