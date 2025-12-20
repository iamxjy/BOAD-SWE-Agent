#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict


INPUT_TOKENS_RE = re.compile(r"input_tokens\s*=\s*([0-9,]+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scan patches debug logs and emit JSON of instance -> max context"
    )
    parser.add_argument(
        "--folder",
        type=Path,
    )
    return parser.parse_args()


def extract_max_from_file(log_path: Path) -> int | None:
    text = log_path.read_text(errors="ignore")
    max_val: int | None = None
    for match in INPUT_TOKENS_RE.finditer(text):
        val = int(match.group(1).replace(",", ""))
        if max_val is None or val > max_val:
            max_val = val
    return max_val


def compute_max_by_instance(patches_dir: Path) -> Dict[str, int]:
    results: Dict[str, int] = {}
    for log_path in patches_dir.rglob("*.debug.log"):
        rel = log_path.relative_to(patches_dir)
        if len(rel.parts) < 2:
            continue
        instance = rel.parts[0]
        max_in_file = extract_max_from_file(log_path)
        if max_in_file is None:
            continue
        curr = results.get(instance)
        if curr is None or max_in_file > curr:
            results[instance] = max_in_file
    return results


def main() -> None:
    args = parse_args()
    patches_dir: Path = Path("tool_gen/evaluation/eval_runs") / args.folder / Path("patches")
    if not patches_dir.exists() or not patches_dir.is_dir():
        raise SystemExit(f"Not a directory: {patches_dir}")
    results = compute_max_by_instance(patches_dir)
    per_instance = dict(sorted(results.items()))
    average = (sum(per_instance.values()) / len(per_instance)) if per_instance else 0
    num_instances = len(per_instance)
    payload = {"per_instance_max": per_instance, "average": average, "num_instances": num_instances}
    out_path = patches_dir.parent / "max_context.json"
    print(json.dumps(payload))
    out_path.write_text(json.dumps(payload))


if __name__ == "__main__":
    main()


