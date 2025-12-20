# read in analysis_summary.json
import json
from pathlib import Path

import argparse

parser = argparse.ArgumentParser(description="Filter analysis summary to only include instances that are not part of the training set")
parser.add_argument("--folder", required=True, help="Evaluation subfolder under logs/run_evaluation")
args = parser.parse_args()

analysis_summary_path = Path("logs/run_evaluation") / args.folder / "analysis_summary.json"
with open(analysis_summary_path, "r") as f:
    analysis_summary = json.load(f)

# filter the analysis_summary to only include the instances that are not part of the training set
training_set_path = Path("swebench/swebench_verified_subset_by_repo_1_each.jsonl")
with open(training_set_path, "r") as f:
    training_set = [json.loads(line) for line in f]

# Extract instance IDs from training set
training_instance_ids = {item["instance_id"] for item in training_set}

# Filter resolved instances to exclude training set
filtered_resolved = [
    instance for instance in analysis_summary["resolved_instances"]
    if instance.split(" (")[0] not in training_instance_ids
]

# Filter unresolved instances to exclude training set  
filtered_unresolved = [
    instance for instance in analysis_summary["unresolved_instances"]
    if instance.split(" (")[0] not in training_instance_ids
]

# Create filtered analysis summary
filtered_analysis_summary = {
    "total_instances": len(filtered_resolved) + len(filtered_unresolved),
    "resolved_count": len(filtered_resolved),
    "success_rate": len(filtered_resolved) / (len(filtered_resolved) + len(filtered_unresolved)) if (len(filtered_resolved) + len(filtered_unresolved)) > 0 else 0,
    "by_project": {},
    "patch_stats": analysis_summary["patch_stats"],
    "resolved_instances": filtered_resolved,
    "unresolved_instances": filtered_unresolved
}

# Recalculate by_project statistics for filtered data
project_counts = {}
for instance in filtered_resolved + filtered_unresolved:
    project = instance.split(" (")[1].rstrip(")")
    if project not in project_counts:
        project_counts[project] = {"total": 0, "resolved": 0}
    project_counts[project]["total"] += 1
    if instance in filtered_resolved:
        project_counts[project]["resolved"] += 1

# Calculate success rates for each project
for project, counts in project_counts.items():
    success_rate = counts["resolved"] / counts["total"] if counts["total"] > 0 else 0
    filtered_analysis_summary["by_project"][project] = {
        "total": counts["total"],
        "resolved": counts["resolved"],
        "success_rate": success_rate
    }

# Save the filtered analysis summary
output_path = Path("logs/run_evaluation") / args.folder / "filtered_analysis_summary.json"
with open(output_path, "w") as f:
    json.dump(filtered_analysis_summary, f, indent=2)

print(f"Filtered analysis summary saved to {output_path}")
print(f"Original instances: {analysis_summary['total_instances']}")
print(f"Training set instances: {len(training_instance_ids)}")
print(f"Filtered instances: {filtered_analysis_summary['total_instances']}")
print(f"Filtered success rate: {filtered_analysis_summary['success_rate']:.4f}")