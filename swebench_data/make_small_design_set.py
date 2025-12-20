import json
from pathlib import Path
import random

def main():
    # Load the dataset
    path = Path("swebench/swebench_verified_subset_by_repo_1_each.jsonl")
    with open(path, "r") as f:
        dataset = [json.loads(line) for line in f]

    # Sample 6 instances
    dataset = random.sample(dataset, 6)
    print(f"Sampled {len(dataset)} instances")
    
    output_path = Path("swebench/swebench_verified_subset_by_repo_1_each_6.jsonl")
    with open(output_path, "w") as f:
        for instance in dataset:
            f.write(json.dumps(instance) + "\n")

if __name__ == "__main__":
    main()