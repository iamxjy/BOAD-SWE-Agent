#!/usr/bin/env python3
"""Collect per-instance token counts (main + subagents) from .traj files."""

import argparse
import json
from pathlib import Path
from typing import Dict, Any
import sys


def _counts(data: Dict[str, Any]) -> Dict[str, int | str | None]:
    ms = (data.get("info") or {}).get("model_stats") or {}
    sent = int(ms.get("tokens_sent") or 0)
    recv = int(ms.get("tokens_received") or 0)
    return {
        "tokens_sent": sent,
        "tokens_received": recv,
        "total_tokens": sent + recv,
        "api_calls": int(ms.get("api_calls") or 0),
        "exit_status": (data.get("info") or {}).get("exit_status"),
    }


def collect_token_counts(directory: Path) -> Dict[str, Dict[str, int | str | None]]:
    results: Dict[str, Dict[str, int | str | None]] = {}
    for traj in directory.glob("**/*.traj"):
        try:
            rel = traj.relative_to(directory)
            inst = rel.parts[0]
            res = results.setdefault(inst, {"tokens_sent": 0, "tokens_received": 0, "total_tokens": 0, "api_calls": 0, "num_trajectories": 0, "exit_status": None})
            c = _counts(json.loads(traj.read_text()))
            res["tokens_sent"] = int(res["tokens_sent"]) + int(c["tokens_sent"])  # type: ignore[index]
            res["tokens_received"] = int(res["tokens_received"]) + int(c["tokens_received"])  # type: ignore[index]
            res["total_tokens"] = int(res["total_tokens"]) + int(c["total_tokens"])  # type: ignore[index]
            res["api_calls"] = int(res["api_calls"]) + int(c["api_calls"])  # type: ignore[index]
            res["num_trajectories"] = int(res["num_trajectories"]) + 1  # type: ignore[index]
            if rel.parent == Path(inst) and c["exit_status"] is not None:
                res["exit_status"] = c["exit_status"]
        except Exception as e:
            print(f"Error processing {traj}: {e}")
    return results


def print_summary(results: Dict[str, Dict[str, int | str | None]]) -> None:
    if not results:
        print("No results.")
        return
    total_sent = sum(int(r.get("tokens_sent", 0) or 0) for r in results.values())
    total_recv = sum(int(r.get("tokens_received", 0) or 0) for r in results.values())
    total = sum(int(r.get("total_tokens", 0) or 0) for r in results.values())
    calls = sum(int(r.get("api_calls", 0) or 0) for r in results.values())
    print(f"instances={len(results)} tokens_sent={total_sent:,} tokens_received={total_recv:,} total_tokens={total:,} api_calls={calls:,}")


def save_results(results: Dict[str, Dict[str, int | str | None]], output_file: Path) -> None:
    """Save results to a JSON file."""
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_file}")


def main() -> None:
    """Main function."""
    parser = argparse.ArgumentParser(description="Collect per-instance token counts")
    parser.add_argument(
        "--folder", 
        type=str, 
        required=True,
        help="Folder name in tool_gen/evaluation/eval_runs to analyze")
    
    args = parser.parse_args()
    
    directory = Path("tool_gen/evaluation/eval_runs") / Path(args.folder) / Path("patches")
    if not directory.exists():
        print(f"Error: Directory {directory} does not exist")
        sys.exit(1)
    
    results = collect_token_counts(directory)
    if not results:
        print("No trajectory files found.")
        sys.exit(1)
    print_summary(results)
    # Always save into the same folder
    save_results(results, directory.parent / "token_counts.json")
    


if __name__ == "__main__":
    main()
