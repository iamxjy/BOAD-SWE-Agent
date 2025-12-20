#!/usr/bin/env python3
"""Remove instance directories for instances with (exit_cost) and (exit_error) status, but skip resolved instances."""

import argparse
import json
import shutil
import yaml
from pathlib import Path


def is_instance_resolved(instance_id: str, eval_dir: Path) -> bool:
    """Check if an instance is resolved by reading its report.json file."""
    report_path = eval_dir / instance_id / "report.json"
    
    if not report_path.exists():
        return False
    
    try:
        report_data = json.loads(report_path.read_text())
        # The report structure has instance_id as key, then nested data
        instance_data = report_data.get(instance_id, {})
        return instance_data.get("resolved", False)
    except (json.JSONDecodeError, KeyError):
        return False


def main(args: argparse.Namespace) -> None:
    """Remove directories for instances with exit_cost and exit_error status, but skip resolved instances."""
    yaml_path = Path(args.test_dir) / "patches" / "run_batch_exit_statuses.yaml"
    
    if not yaml_path.exists():
        raise FileNotFoundError(f"YAML file not found: {yaml_path}")
    
    with yaml_path.open() as f:
        data = yaml.safe_load(f)
    
    instance_ids_to_remove = set()
    
    if "instances_by_exit_status" in data:
        for status, instances in data["instances_by_exit_status"].items():
            if "(exit_cost)" in status or "(exit_error)" in status:
                instance_ids_to_remove.update(instances)
    
    if not instance_ids_to_remove:
        print("No instances with (exit_cost) or (exit_error) status found.")
        return
    
    print(f"Found {len(instance_ids_to_remove)} instance IDs to remove:")
    for instance_id in sorted(instance_ids_to_remove):
        print(f"  - {instance_id}")
    
    # Check for resolved instances and filter them out
    eval_dir = Path(args.eval_dir)
    resolved_instances = set()
    instances_to_actually_remove = set()
    
    for instance_id in instance_ids_to_remove:
        if eval_dir.exists() and is_instance_resolved(instance_id, eval_dir):
            resolved_instances.add(instance_id)
        else:
            instances_to_actually_remove.add(instance_id)
    
    if resolved_instances:
        print(f"\nSkipping {len(resolved_instances)} resolved instances:")
        for instance_id in sorted(resolved_instances):
            print(f"  - {instance_id} (resolved)")
    
    if not instances_to_actually_remove:
        print("\nNo instances to remove after filtering out resolved ones.")
        return
    
    print(f"\nProceeding to remove {len(instances_to_actually_remove)} unresolved instances:")
    for instance_id in sorted(instances_to_actually_remove):
        print(f"  - {instance_id}")
    
    patches_dir = Path(args.test_dir) / "patches"
    removed_count = 0
    
    # Remove from patches directory
    for instance_id in instances_to_actually_remove:
        instance_dir = patches_dir / instance_id
        if instance_dir.exists() and instance_dir.is_dir():
            if args.dry_run:
                print(f"[DRY RUN] Would remove: {instance_dir}")
            else:
                shutil.rmtree(instance_dir)
                print(f"Removed: {instance_dir}")
                removed_count += 1
        else:
            print(f"Not found (skipping): {instance_dir}")
    
    # Remove from evaluation directory
    if not eval_dir.exists():
        print(f"Evaluation directory not found: {eval_dir}")
    else:
        print(f"\nProcessing evaluation directory: {eval_dir}")
        for instance_id in instances_to_actually_remove:
            instance_dir = eval_dir / instance_id
            if instance_dir.exists() and instance_dir.is_dir():
                if args.dry_run:
                    print(f"[DRY RUN] Would remove: {instance_dir}")
                else:
                    shutil.rmtree(instance_dir)
                    print(f"Removed: {instance_dir}")
                    removed_count += 1
            else:
                print(f"Not found (skipping): {instance_dir}")
    
    if not args.dry_run:
        print(f"\nTotal directories removed: {removed_count}")
    else:
        print(f"\n[DRY RUN] Would remove {removed_count} directories")
    print(f"Total instances with exit_cost/exit_error: {len(instance_ids_to_remove)}")
    print(f"Resolved instances (skipped): {len(resolved_instances)}")
    print(f"Unresolved instances (processed): {len(instances_to_actually_remove)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Remove instance directories for instances with (exit_cost) and (exit_error) status, but skip resolved instances"
    )
    parser.add_argument(
        "--test-dir",
        required=True,
        help="Path to test directory containing patches/run_batch_exit_statuses.yaml",
    )
    parser.add_argument(
        "--eval-dir",
        required=True,
        type=str,
        help="Path to evaluation directory",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be removed without actually removing",
    )
    args = parser.parse_args()
    main(args)

