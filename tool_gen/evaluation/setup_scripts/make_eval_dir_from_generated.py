import argparse
import shutil
import sys
from pathlib import Path
import yaml

GENERATED_DIR = Path("./tool_gen/generated")
EVAL_DIR = Path("./tool_gen/evaluation/eval_runs")
PROJECT_ROOT = Path("./")

# Add parent directory to path to import from check_archive and tools
sys.path.append(str(Path(__file__).parent.parent.parent))
from scripts_analyze.check_archive import load_archive_data
from evaluation.setup_scripts.setup_eval import generate_configs
from evaluation.setup_scripts.generate_main_plan import generate_main_plan

def get_top_k_subagents(archive_path: Path, k: int) -> list[dict]:
    """Get top-k subagents based on helpful_rate."""
    archive_data = load_archive_data(archive_path)
    if not archive_data:
        return []
    
    # Calculate helpful_rate for each subagent
    subagents_with_rate = []
    for item in archive_data:
        if not isinstance(item, dict):
            continue
        
        name = item.get("name")
        n = item.get("n", 0)
        helpful_count = item.get("helpful_count", 0)
        
        if not isinstance(name, str) or n == 0:
            continue
        
        helpful_rate = helpful_count / n if n > 0 else 0.0
        subagents_with_rate.append({
            "item": item,
            "helpful_rate": helpful_rate
        })
    
    # Sort by helpful_rate (descending) and get top-k
    subagents_with_rate.sort(key=lambda x: x["helpful_rate"], reverse=True)
    return [x["item"] for x in subagents_with_rate[:k]]

def copy_subagent_bundle(source_bundle_dir: Path, target_tools_dir: Path, subagent_name: str) -> None:
    """Copy a subagent bundle directory to the target tools directory."""
    target_bundle_dir = target_tools_dir / subagent_name
    if target_bundle_dir.exists():
        shutil.rmtree(target_bundle_dir)
    shutil.copytree(source_bundle_dir, target_bundle_dir)

def main(args) -> None:
    if not args.folder:
        print("Error: --folder argument is required")
        return

    generated_folder = GENERATED_DIR / args.folder
    archive_path = generated_folder / "subagent_tool_archive" / "archive.json"
    
    if not archive_path.exists():
        print(f"Error: Archive not found at {archive_path}")
        return
    
    # Get top-k subagents based on helpful_rate
    top_subagents = get_top_k_subagents(archive_path, args.k)
    
    if not top_subagents:
        print("Error: No subagents found in archive")
        return
    
    print(f"Selected top {len(top_subagents)} subagents based on helpful_rate")
    
    # Create eval directory structure
    eval_folder = EVAL_DIR / args.output_folder
    eval_folder.mkdir(parents=True, exist_ok=True)
    tools_dir = eval_folder / "tools"
    tools_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy top-k subagents to tools folder
    for subagent in top_subagents:
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
    plan =generate_main_plan(eval_folder)
    print(f"Generated main plan in {eval_folder}")

    # Substitute the {{plan}} placeholder in the instance template with the generated plan in setup_eval.py
    with open(eval_folder / "agent.yaml", "r") as f:
        agent_cfg = yaml.safe_load(f)

    agent_cfg["agent"]["templates"]["instance_template"] = agent_cfg["agent"]["templates"]["instance_template"].replace("{{plan}}", plan)
    with open(eval_folder / "agent.yaml", "w") as f:
        yaml.dump(agent_cfg, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, required=True, help="Folder name in generated directory")
    parser.add_argument("--output_folder", type=str, required=True, help="Output folder name")
    parser.add_argument("--k", type=int, default=2, help="Number of top subagents to select (default: 2)")
    parser.add_argument("--template_dir", type=str, default="tool_gen/templates_seed_oss_36b", help="Template directory for agent configs")
    args = parser.parse_args()
    
    main(args)