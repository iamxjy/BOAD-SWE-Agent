import argparse
import shutil
import sys
from pathlib import Path
import yaml

EVAL_DIR = Path("./tool_gen/evaluation/eval_runs")
PROJECT_ROOT = Path(".")

# Add parent directory to path to import from tools
sys.path.append(str(Path(__file__).parent.parent.parent))
from evaluation.setup_scripts.setup_eval import generate_configs
from evaluation.setup_scripts.generate_main_plan import generate_main_plan

def load_subagents_from_paths(subagent_paths: list[str]) -> list[dict]:
    """Load subagents from given directory paths."""
    subagents = []
    
    for path_str in subagent_paths:
        path = Path(path_str)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        
        if not path.exists():
            print(f"Warning: Subagent directory not found: {path}")
            continue
        
        if not path.is_dir():
            print(f"Warning: Path is not a directory: {path}")
            continue
        
        # Create subagent dict with name and bundle_dir
        subagent_name = path.name
        subagent = {
            "name": subagent_name,
            "bundle_dir": str(path)
        }
        subagents.append(subagent)
        print(f"Loaded subagent: {subagent_name} from {path}")
    
    return subagents

def copy_subagent_bundle(source_bundle_dir: Path, target_tools_dir: Path, subagent_name: str) -> None:
    """Copy a subagent bundle directory to the target tools directory."""
    target_bundle_dir = target_tools_dir / subagent_name
    if target_bundle_dir.exists():
        shutil.rmtree(target_bundle_dir)
    shutil.copytree(source_bundle_dir, target_bundle_dir)

def main(args) -> None:
    if not args.subagent_paths:
        print("Error: --subagent_paths argument is required")
        return

    # Load subagents from provided directory paths
    subagents = load_subagents_from_paths(args.subagent_paths)
    
    if not subagents:
        print("Error: No valid subagents found from provided paths")
        return
    
    print(f"Loaded {len(subagents)} subagents from provided paths")
    
    # Create eval directory structure
    eval_folder = EVAL_DIR / args.output_folder
    eval_folder.mkdir(parents=True, exist_ok=True)
    tools_dir = eval_folder / "tools"
    tools_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy subagents to tools folder
    for subagent in subagents:
        name = subagent.get("name")
        bundle_dir_str = subagent.get("bundle_dir", "")
        
        if not name or not bundle_dir_str:
            continue
        
        # Handle both absolute and relative paths
        if Path(bundle_dir_str).is_absolute():
            source_bundle_dir = Path(bundle_dir_str)
        else:
            # Relative to project root
            source_bundle_dir = PROJECT_ROOT / bundle_dir_str
        
        if not source_bundle_dir.exists():
            print(f"Warning: Bundle directory not found: {source_bundle_dir}")
            continue
        
        copy_subagent_bundle(source_bundle_dir, tools_dir, name)
        print(f"Copied subagent: {name}")
    
    # Generate agent.yaml and subagent.yaml using setup_eval.py
    generate_configs(eval_folder, args.template_dir)
    print(f"Generated agent.yaml and subagent.yaml in {eval_folder}")
    
    # Generate main plan using generate_main_plan.py
    plan = generate_main_plan(eval_folder)
    print(f"Generated main plan in {eval_folder}")

    # Substitute the {{plan}} placeholder in the instance template with the generated plan in setup_eval.py
    with open(eval_folder / "agent.yaml", "r") as f:
        agent_cfg = yaml.safe_load(f)

    agent_cfg["agent"]["templates"]["instance_template"] = agent_cfg["agent"]["templates"]["instance_template"].replace("{{plan}}", plan)
    with open(eval_folder / "agent.yaml", "w") as f:
        yaml.dump(agent_cfg, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subagent_paths", type=str, nargs="+", required=True, help="List of subagent directory paths")
    parser.add_argument("--output_folder", type=str, required=True, help="Output folder name")
    parser.add_argument("--template_dir", type=str, default="tool_gen/templates_seed_oss_36b", help="Template directory for agent configs")
    args = parser.parse_args()
    
    main(args)