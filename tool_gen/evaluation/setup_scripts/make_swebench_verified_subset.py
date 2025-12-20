from datasets import load_dataset
import json
from pathlib import Path

def make_swebench_verified_subset():
    dataset = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")
    
    # dataset = dataset.filter(lambda x: x["instance_id"] in wanted_ids)
    # dataset.save_to_disk("tool_gen/evaluation/swebench_verified_subset.jsonl")

def make_swebench_verified_subset_by_repo():
    """
    Repositories:
    astropy/astropy               22
    django/django                231
    matplotlib/matplotlib         34
    mwaskom/seaborn                2
    pallets/flask                  1
    psf/requests                   8
    pydata/xarray                 22
    pylint-dev/pylint             10
    pytest-dev/pytest             19
    scikit-learn/scikit-learn     32
    sphinx-doc/sphinx             44
    sympy/sympy                   75
    """

    dataset = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")
    dataset = dataset.to_pandas()
    # Sample one instance at each repository
    dataset = dataset.groupby("repo").sample(1)
    print(f"Sampled {len(dataset)} instances")
    dataset = dataset.reset_index(drop=True)
    dataset = dataset.to_dict(orient="records")

    with open("tool_gen/evaluation/data/swebench_verified_subset_by_repo_1_each.jsonl", "w") as f:
        for instance in dataset:
            f.write(json.dumps(instance) + "\n")

def make_swebench_verified_subset_seedoss_failed():
    dataset = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")
    

    seedoss_failed = set()
    for report_path in Path("logs/run_evaluation/seedoss_36b_ins").glob("**/*.json"):
        with open(report_path, "r") as f:
            report = json.load(f)
            for instance_id, instance in report.items():
                if instance["resolved"] is False:
                    seedoss_failed.add(instance_id)
    print(f"Seedoss failed: {len(seedoss_failed)}")

    dataset = dataset.filter(lambda x: x["instance_id"] in seedoss_failed)
    dataset.save_to_disk("tool_gen/evaluation/data/swebench_verified_subset_seedoss_failed.jsonl")

if __name__ == "__main__":
    # make_swebench_verified_subset_by_repo()
    make_swebench_verified_subset_seedoss_failed()