# make_instances.py
from datasets import load_dataset
import json, sys
ids=set(sys.argv[1:]); ds=load_dataset("SWE-bench-Live/SWE-bench-Live", split="lite")
with open("swebench/swebench_live.jsonl","w") as f:
  for r in ds:
    if ids and r["instance_id"] not in ids: continue
    img="starryzhang/sweb.eval.x86_64."+r["instance_id"].replace("__","_1776_")+":latest"
    f.write(json.dumps({
      "image_name":img,
      "problem_statement":r["problem_statement"],
      "instance_id":r["instance_id"],
      "repo_name":"testbed",
      "base_commit":r["base_commit"]
    })+"\n")
print("wrote swebench/swebench_live.jsonl")
