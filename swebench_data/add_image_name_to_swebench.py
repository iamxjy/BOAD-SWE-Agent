#!/usr/bin/env python3
import json
from pathlib import Path

def main():
    input_file = "swebench_data/swebench_verified_subset_by_repo_1_each.jsonl"
    output_file = "swebench_data/swebench_verified_subset_by_repo_1_each.jsonl"
    
    if not Path(input_file).exists():
        print(f"❌ Input file not found: {input_file}")
        return
    
    instances = []
    with open(input_file, 'r') as f:
        for line in f:
            if line.strip():
                instance = json.loads(line.strip())
                instance_id = instance.get("instance_id", "")
                if instance_id:
                    if "__" in instance_id:
                        new_id = instance_id.replace("__", "_s_").lower()
                    else:
                        new_id = instance_id.lower()
                    instance["image_name"] = f"xingyaoww/sweb.eval.x86_64.{new_id}"
                else:
                    instance["image_name"] = "xingyaoww/sweb.eval.x86_64.default"
                instances.append(instance)
    
    with open(output_file, 'w') as f:
        for instance in instances:
            f.write(json.dumps(instance) + '\n')
    
    print(f"✅ Processed {len(instances)} instances -> {output_file}")

if __name__ == "__main__":
    main()
