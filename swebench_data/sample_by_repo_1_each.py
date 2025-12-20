import json
from pathlib import Path
from datasets import load_dataset


def extract_repo(instance_id: str) -> str:
    """Extract repository identifier from instance_id.
    
    Examples:
        'django__django-10880' -> 'django__django'
        'matplotlib__matplotlib-20826' -> 'matplotlib__matplotlib'
        'psf__requests-2317' -> 'psf__requests'
    """
    # Split on the last occurrence of '-' followed by numbers
    # and take everything before that
    parts = instance_id.rsplit('-', 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0]
    return instance_id


def main():
    # Load swe-bench verified dataset from Hugging Face
    dataset = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")
    
    output_file = Path("swebench/swebench_verified_subset_by_repo_1_each_2.jsonl")
    
    repos = {}
    
    # Sample one instance per repository
    for instance in dataset:
        repo = extract_repo(instance["instance_id"])
        if repo not in repos:
            # Add image_name field required by SimpleBatchInstance
            instance_dict = {key: value for key, value in instance.items()}
            instance_id = instance_dict["instance_id"]
            if "__" in instance_id:
                new_id = instance_id.replace("__", "_1776_").lower()
            else:
                new_id = instance_id.lower()
            instance_dict["image_name"] = f"swebench/sweb.eval.x86_64.{new_id}:latest"
            repos[repo] = instance_dict
    
    # Write to output file
    with open(output_file, "w") as f:
        for data in repos.values():
            f.write(json.dumps(data) + "\n")
    
    print(f"Sampled {len(repos)} unique repositories from swe-bench verified dataset")
    print(f"Output written to {output_file}")
    
    # Print repo list for verification
    print("\nRepositories sampled:")
    for repo in sorted(repos.keys()):
        print(f"  {repo}")


def main_local():
    input_file = Path("swebench/nebius50.jsonl")
    output_file = Path("swebench/swebench_verified_subset_by_repo_1_each_2.jsonl")
    
    repos = {}
    
    with open(input_file) as f:
        for line in f:
            data = json.loads(line)
            repo = extract_repo(data["instance_id"])
            if repo not in repos:
                repos[repo] = data
    
    with open(output_file, "w") as f:
        for data in repos.values():
            f.write(json.dumps(data) + "\n")
    
    print(f"Sampled {len(repos)} unique repositories from {input_file}")
    print(f"Output written to {output_file}")
    
    # Print repo list for verification
    print("\nRepositories sampled:")
    for repo in sorted(repos.keys()):
        print(f"  {repo}")


if __name__ == "__main__":
    main()
